import logging
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import xarray as xr
import string
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
import os
from dask.diagnostics import ProgressBar
import gc
import time
import argparse
import warnings
from typing import Tuple, Optional
import torch.nn.functional as F


gc.collect()

# --- Configure logging ---
def setup_logging(rank: int) -> logging.Logger:
    """Configure logging for distributed training."""
    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s [RANK {rank}] %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# --- Constants ---
PATCH_SIZE = 5
TIME_STEPS = 5
LATENT_DIM = 16
NUM_TILES_PER_TIME = 5000
DEFAULT_BATCH_SIZE = 8048  # Reduced from 64384 to fit memory better
DEFAULT_NUM_WORKERS = 8
EARLY_STOPPING_PATIENCE = 10
MIN_LR = 1e-6
DEFAULT_PREFETCH_FACTOR = 25
VALIDATION_SIZE = 1000

class WeatherUNet(nn.Module):
    """U-Net style autoencoder for weather data with temporal context."""
    
    def __init__(self, num_vars: int = 9, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.num_vars = num_vars
        self.input_channels = TIME_STEPS * num_vars
        
        # Encoder
        self.enc1 = self._block(self.input_channels, 64)  # [B, 5*9, 5, 5] -> [B, 64, 5, 5]
        self.enc2 = self._block(64, 128)                  # [B, 64, 2, 2] -> [B, 128, 2, 2]
        self.enc3 = self._block(128, 256)                 # [B, 128, 1, 1] -> [B, 256, 1, 1]
        
        # Bottleneck
        self.bottleneck = self._block(256, latent_dim)     # [B, 256, 1, 1] -> [B, latent_dim, 1, 1]
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(latent_dim, 128, kernel_size=2, stride=2)  # [B, latent_dim, 1, 1] -> [B, 128, 2, 2]
        self.dec1 = self._block(256, 128)  # 128 (up) + 128 (skip) -> [B, 128, 2, 2]
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # [B, 128, 2, 2] -> [B, 64, 4, 4]
        self.dec2 = self._block(128, 64)   # 64 (up) + 64 (skip) -> [B, 64, 4, 4]
        
        # Final output
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, self.input_channels, kernel_size=1),  # [B, 64, 5, 5] -> [B, 5*9, 5, 5]
            nn.Tanh()
        )
        
        # Pooling layers
        self.pool1 = nn.MaxPool2d(2)  # [B, 64, 5, 5] -> [B, 64, 2, 2]
        self.pool2 = nn.MaxPool2d(2)  # [B, 128, 2, 2] -> [B, 128, 1, 1]
        
    def _block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Create a basic convolutional block with batch norm and ReLU."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network."""
        # Reshape input: [B, 5*5*5*9] -> [B, 5*9, 5, 5]
        x = x.view(-1, self.input_channels, PATCH_SIZE, PATCH_SIZE)
        
        # Encoder path
        e1 = self.enc1(x)        # [B, 64, 5, 5]
        e2 = self.pool1(e1)      # [B, 64, 2, 2]
        e2 = self.enc2(e2)       # [B, 128, 2, 2]
        e3 = self.pool2(e2)      # [B, 128, 1, 1]
        e3 = self.enc3(e3)       # [B, 256, 1, 1]
        
        # Bottleneck
        z = self.bottleneck(e3)  # [B, latent_dim, 1, 1]
        
        # Decoder path
        d1 = self.up1(z)         # [B, 128, 2, 2]
        d1 = torch.cat([d1, e2], dim=1)  # [B, 256, 2, 2]
        d1 = self.dec1(d1)       # [B, 128, 2, 2]
        
        d2 = self.up2(d1)        # [B, 64, 4, 4]
        d2 = torch.cat([d2, e1[:, :, :4, :4]], dim=1)  # [B, 128, 4, 4]
        d2 = self.dec2(d2)       # [B, 64, 4, 4]
        
        # Final output
        out = F.interpolate(d2, size=(PATCH_SIZE, PATCH_SIZE), mode='bilinear', align_corners=False)
        out = self.final_conv(out)  # [B, 45, 5, 5]
        
        # Flatten outputs
        out = out.view(-1, PATCH_SIZE * PATCH_SIZE * self.input_channels)
        z = z.view(-1, LATENT_DIM)
        
        return out, z

class WeatherDataset(Dataset):
    """Dataset for weather patches with temporal context."""
    
    def __init__(self, 
                 data: np.ndarray, 
                 patch_size: int = PATCH_SIZE,
                 time_steps: int = TIME_STEPS,
                 num_samples: int = NUM_TILES_PER_TIME,
                 validation: bool = False):
        """
        Args:
            data: Array of shape (time, lat, lon, num_vars)
            patch_size: Size of spatial patches
            time_steps: Number of temporal steps
            num_samples: Samples per time step
            validation: Whether this is validation data
        """
        self.data = data
        self.patch_size = patch_size
        self.time_steps = time_steps
        self.num_vars = data.shape[-1]
        self.pad = patch_size // 2
        self.validation = validation
        
        # Pad spatial dimensions (time padding handled in getitem)
        self.data = np.pad(
            data,
            pad_width=(
                (time_steps-1, 0),  # Time
                (self.pad, self.pad),  # Lat
                (self.pad, self.pad),  # Lon
                (0, 0)  # Vars
            ),
            mode='wrap'
        )
        
        # Generate sample indices
        self.indices = self._generate_indices(
            num_samples if not validation else VALIDATION_SIZE
        )
    
    def _generate_indices(self, num_samples: int) -> list:
        """Generate sample indices for the dataset."""
        indices = []
        valid_times = range(self.time_steps-1, self.data.shape[0])
        
        for t in valid_times:
            max_lat = self.data.shape[1] - self.patch_size
            max_lon = self.data.shape[2] - self.patch_size
            
            if self.validation:
                np.random.seed(42)  # Fixed seed for validation
                
            lats = np.random.randint(0, max_lat, size=num_samples)
            lons = np.random.randint(0, max_lon, size=num_samples)
            
            if self.validation:
                np.random.seed()  # Reset seed
                
            indices.extend([(t, lat, lon) for lat, lon in zip(lats, lons)])
        
        return indices
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        t, lat, lon = self.indices[idx]
        
        # Get temporal patch (t-4 to t)
        patch = self.data[
            t - self.time_steps + 1 : t + 1,  # Time
            lat : lat + self.patch_size,      # Lat
            lon : lon + self.patch_size,      # Lon
            :                                 # Vars
        ]
        
        return torch.tensor(patch, dtype=torch.float32).flatten()

class FullWeatherDataset(Dataset):
    """Dataset for full inference on weather data."""
    
    def __init__(self, 
                 data: np.ndarray, 
                 patch_size: int = PATCH_SIZE,
                 time_steps: int = TIME_STEPS):
        self.data = data
        self.patch_size = patch_size
        self.time_steps = time_steps
        self.num_vars = data.shape[-1]
        self.pad = patch_size // 2
        
        # Pad data
        self.data = np.pad(
            data,
            pad_width=(
                (time_steps-1, 0),  # Time
                (self.pad, self.pad),  # Lat
                (self.pad, self.pad),  # Lon
                (0, 0)  # Vars
            ),
            mode='wrap'
        )
        
        # Generate indices for all valid positions
        self.indices = [
            (t, i, j)
            for t in range(time_steps-1, self.data.shape[0])
            for i in range(self.data.shape[1] - patch_size + 1)
            for j in range(self.data.shape[2] - patch_size + 1)
        ]
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        t, lat, lon = self.indices[idx]
        
        patch = self.data[
            t - self.time_steps + 1 : t + 1,
            lat : lat + self.patch_size,
            lon : lon + self.patch_size,
            :
        ]
        
        return torch.tensor(patch, dtype=torch.float32).flatten()

def train(rank: int, 
          world_size: int, 
          data: np.ndarray, 
          model_path: Optional[str] = None, 
          output_zarr_path: Optional[str] = None, 
          batch_size: int = DEFAULT_BATCH_SIZE,
          num_workers: int = DEFAULT_NUM_WORKERS,
          prefetch_factor: int = DEFAULT_PREFETCH_FACTOR) -> None:
    """Training function for distributed training."""
    logger = setup_logging(rank)
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    
    # Initialize distributed training if using CUDA
    if device.type == 'cuda':
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Create model
    model = WeatherUNet(num_vars=data.shape[-1]).to(device)
    if device.type == 'cuda' and world_size > 1:
        model = DDP(model, device_ids=[rank])
    
    # Create datasets and dataloaders
    train_dataset = WeatherDataset(data, validation=False)
    val_dataset = WeatherDataset(data, validation=True)
    
    if device.type == 'cuda' and world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=True,
        drop_last=False
    )
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=1e-3 * world_size)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    loss_fn = nn.MSELoss()
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(1000):  # Early stopping will break this
        model.train()
        if train_sampler:
            train_sampler.set_epoch(epoch)
        
        total_train_loss = 0
        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)
            recon, _ = model(batch)
            loss = loss_fn(recon, batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device, non_blocking=True)
                recon, _ = model(batch)
                loss = loss_fn(recon, batch)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        # Early stopping and model saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            if rank == 0:
                torch.save(model.module.state_dict() if hasattr(model, 'module') else model.state_dict(), 
                          model_path or "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                break
        
        if rank == 0 and device.type == 'cuda':
            dist.destroy_process_group()

def run_inference(data: np.ndarray,
                  model_path: str = "best_model.pth",
                  output_zarr_path: Optional[str] = None,
                  batch_size: int = DEFAULT_BATCH_SIZE,
                  num_workers: int = DEFAULT_NUM_WORKERS,
                  prefetch_factor: int = DEFAULT_PREFETCH_FACTOR) -> np.ndarray:
    """Run inference on full dataset."""
    logger = setup_logging(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = WeatherUNet(num_vars=data.shape[-1]).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Create dataset and loader
    full_dataset = FullWeatherDataset(data)
    full_loader = DataLoader(
        full_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=True
    )
    
    # Run inference
    all_latents = []
    with torch.no_grad():
        for batch in full_loader:
            batch = batch.to(device)
            _, latents = model(batch)
            all_latents.append(latents.cpu())
    
    all_latents = torch.cat(all_latents, dim=0)
    
    # Reshape to match original data
    valid_time = data.shape[0] - TIME_STEPS + 1
    H, W = data.shape[1], data.shape[2]
    all_latents = all_latents.numpy().reshape(valid_time, H, W, LATENT_DIM)
    
    # Pad time dimension
    full_latents = np.concatenate([
        np.repeat(all_latents[[0]], TIME_STEPS - 1, axis=0),
        all_latents
    ], axis=0)
    
    # Save to Zarr if requested
    if output_zarr_path:
        time_coords = np.arange(data.shape[0])
        lat = np.arange(data.shape[1])
        lon = np.arange(data.shape[2])
        
        latent_ds = xr.Dataset({
            f"latent_{i}": xr.DataArray(
                full_latents[..., i],
                dims=["time", "lat", "lon"],
                coords={"time": time_coords, "lat": lat, "lon": lon}
            )
            for i in range(LATENT_DIM)
        })
        latent_ds.to_zarr(output_zarr_path, mode="w")
    
    return full_latents

def preprocess_data(input_zarr_path, variable_type):
    logger = logging.getLogger(__name__)
    logger.info("Loading and preprocessing data...")
    
    # 1. Open Zarr dataset with Dask chunks
    ds = xr.open_zarr(input_zarr_path, chunks={})
    ds = ds.sel(lat=slice(42, 51), lon=slice(4, 16))
    
    logger.info("Initial dataset loaded. Shape: %s, Variables: %s", 
               dict(ds.dims), list(ds.data_vars.keys()))
    
    # 2. Select variables based on variable_type
    if variable_type == 't2m':
        logger.info("Using only t2m variable")
        ds = ds.drop_vars(['ssrd', 'tp'])
    elif variable_type == 'ssrd':
        logger.info("Using only ssrd variable")
        ds = ds.drop_vars(['t2m', 'tp'])
    elif variable_type == 'tp':
        logger.info("Using only tp variable with log1p transformation")
        ds = ds.drop_vars(['u_850', 'v_850', "z_850", "sin_doy", "cos_doy", "q_850"])
        # Apply log1p transformation to tp
        ds['tp'] = ds['tp'] * 1000
        # Create binary precipitation variable (0 for <0.01mm, 1 for ≥0.01mm)
        ds['precip_binary'] = xr.where(ds['tp'] >= 0.01, 1, 0)
        ds['tp'] = np.cbrt(ds["tp"])
        
    else:
        raise ValueError(f"Invalid variable_type: {variable_type}. Must be one of 't2m', 'ssrd', or 'tp'")
    
    # 3. Preprocess data (still lazy)
    ds = ds.fillna(0)
    variables = sorted(ds.data_vars.keys())
    logger.info("Using variables: %s", variables)

    load_start = time.time()
    
    # Initialize empty list to store normalized chunks
    normalized_chunks = []
    
    # [Previous code remains the same until the normalization part]

    # 4. Process each variable separately to minimize memory usage
    for var in variables:
        logger.info(f"Processing variable: {var}")
        
        # Skip normalization for precip_binary when variable_type is 'tp'
        if variable_type == 'tp' and var == 'precip_binary':
            logger.info("Skipping normalization for precip_binary")
            # Just compute and expand dims without normalization
            with ProgressBar():
                normalized_array = ds[var].compute().expand_dims('variable')
        else:
            # Normalize all other variables
            with ProgressBar():
                var_mean = ds[var].mean(dim=['time', 'lat', 'lon']).compute()
                var_std = ds[var].std(dim=['time', 'lat', 'lon']).compute()
            
            logger.info(f"Computed mean for {var}: {var_mean.values}")
            logger.info(f"Computed std for {var}: {var_std.values}")
            
            # Normalize this variable (still lazy)
            normalized_var = (ds[var] - var_mean) / (var_std + 1e-8)
            
            # Compute this variable
            with ProgressBar():
                normalized_array = normalized_var.compute().expand_dims('variable')
            
            # Clean up
            del normalized_var
        
        # Append to our list (now in memory)
        normalized_chunks.append(normalized_array)
        del normalized_array

    # [Rest of your code continues...]
    
    # 5. Combine all normalized variables along the new dimension
    logger.info("Combining all normalized variables...")
    data = xr.concat(normalized_chunks, dim='variable')
    data = data.transpose('time', 'lat', 'lon', 'variable')
    
    logger.info("Data loaded in %.2f seconds", time.time() - load_start)
    logger.info("Final data shape: %s", data.shape)
    logger.info("Memory usage: %.2f GB", data.nbytes / 1e9)
    
    # Verify no NaN values
    assert not np.isnan(data).any(), "Data contains NaN values after normalization"
    logger.info("Data validation passed - no NaN values detected")
    
    return data.values

def main():
    parser = argparse.ArgumentParser(description="Weather Autoencoder Training and Inference")
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--train', action='store_true', help='Train the model from scratch')
    mode_group.add_argument('--start_from_model', type=str, help='Continue training from an existing model')
    mode_group.add_argument('--infer', type=str, help='Run inference using the specified model')
    
    # Required arguments
    parser.add_argument('--input_zarr', type=str, required=True, help='Path to input Zarr file')
    parser.add_argument('--variable_type', type=str, required=True, choices=['t2m', 'ssrd', 'tp'],
                       help='Which variable to process (t2m, ssrd, or tp)')
    
    # Optional arguments
    parser.add_argument('--model_output', type=str, help='Path to save the trained model')
    parser.add_argument('--output_zarr', type=str, help='Path to save the output Zarr file')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, 
                       help=f'Batch size (default: {DEFAULT_BATCH_SIZE})')
    parser.add_argument('--num_workers', type=int, default=DEFAULT_NUM_WORKERS,
                       help=f'Number of workers (default: {DEFAULT_NUM_WORKERS})')
    parser.add_argument('--prefetch_factor', type=int, default=DEFAULT_PREFETCH_FACTOR,
                       help=f'Prefetch factor (default: {DEFAULT_PREFETCH_FACTOR})')
    
    args = parser.parse_args()
    
    logger = setup_logging(0)
    logger.info("Starting main execution with arguments: %s", args)
    
    try:
        # Preprocess data
        data = preprocess_data(args.input_zarr, args.variable_type)
        
        if args.train:
            logger.info("Starting training from scratch")
            world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
            if world_size > 1:
                logger.info("Starting multi-GPU training with %d GPUs", world_size)
                mp.spawn(train, args=(world_size, data, args.model_output, args.output_zarr, 
                                    args.batch_size, args.num_workers, args.prefetch_factor), 
                        nprocs=world_size, join=True)
            else:
                logger.info("Starting single-GPU/CPU training")
                train(0, 1, data, args.model_output, args.output_zarr, 
                      args.batch_size, args.num_workers, args.prefetch_factor)
            
            # After training, run inference if output_zarr is specified
            if args.output_zarr:
                logger.info("Running inference with the trained model")
                run_inference(data, args.model_output if args.model_output else "best_model.pth", 
                            args.output_zarr, args.batch_size, args.num_workers, args.prefetch_factor)
        
        elif args.start_from_model:
            logger.info("Starting training from existing model: %s", args.start_from_model)
            world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
            if world_size > 1:
                logger.info("Starting multi-GPU training with %d GPUs", world_size)
                mp.spawn(train, args=(world_size, data, args.model_output, args.output_zarr, 
                                    args.batch_size, args.num_workers, args.prefetch_factor), 
                        nprocs=world_size, join=True)
            else:
                logger.info("Starting single-GPU/CPU training")
                train(0, 1, data, args.model_output, args.output_zarr, 
                      args.batch_size, args.num_workers, args.prefetch_factor)
            
            # After training, run inference if output_zarr is specified
            if args.output_zarr:
                logger.info("Running inference with the trained model")
                run_inference(data, args.model_output if args.model_output else "best_model.pth", 
                            args.output_zarr, args.batch_size, args.num_workers, args.prefetch_factor)
        
        elif args.infer:
            logger.info("Running inference with model: %s", args.infer)
            latent_data = run_inference(data, args.infer, args.output_zarr, 
                                      args.batch_size, args.num_workers, args.prefetch_factor)
            
            if not args.output_zarr:
                logger.warning("No output Zarr path specified. Inference results will not be saved.")
        
        logger.info("✅ Successfully completed execution")
    
    except Exception as e:
        logger.error("❌ Execution failed: %s", str(e), exc_info=True)
        raise

if __name__ == "__main__":
    main()