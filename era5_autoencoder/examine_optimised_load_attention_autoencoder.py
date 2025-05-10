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
from tqdm import tqdm  # For progress bars

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
PATCH_SIZE = 4  # Changed from 5 to 4
TIME_STEPS = 6  # Changed from 5 to 6
LATENT_DIM = 16
NUM_TILES_PER_TIME = 8000
DEFAULT_BATCH_SIZE = 64384
DEFAULT_NUM_WORKERS = 2
EARLY_STOPPING_PATIENCE = 20
MIN_LR = 1e-6
DEFAULT_PREFETCH_FACTOR = 2
VALIDATION_SIZE = 800

# --- Model Components ---
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, ch)
        )

    def forward(self, x):
        return F.relu(x + self.block(x))

class SEBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // 8, ch, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x)

class WeatherUNetImproved(nn.Module):
    def __init__(self, num_vars: int):
        super().__init__()
        self.num_vars = num_vars
        self.input_channels = TIME_STEPS * num_vars
        self.output_elements = TIME_STEPS * num_vars * PATCH_SIZE * PATCH_SIZE  # Total elements per sample

        # Encoder path
        self.enc1 = nn.Sequential(
            nn.Conv2d(self.input_channels, 64, kernel_size=3, padding=1),
            ResBlock(64)
        )
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),  # [B,64,4,4] -> [B,64,2,2]
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            ResBlock(128)
        )

        # Bottleneck with SE block
        self.se = SEBlock(128)
        self.bottleneck = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, LATENT_DIM)
        )

        # Decoder path
        self.decode_fc = nn.Linear(LATENT_DIM, 128)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # [B,128,1,1] -> [B,64,2,2]
            ResBlock(64)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),  # [B,64,2,2] -> [B,64,4,4]
            ResBlock(64)
        )
        self.final = nn.Sequential(
            nn.Conv2d(64, self.input_channels, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x):
        B = x.shape[0]
        x = x.view(B, self.input_channels, PATCH_SIZE, PATCH_SIZE)

        # Encoder
        e1 = self.enc1(x)           # [B, 64, 4, 4]
        e2 = self.enc2(e1)          # [B, 128, 2, 2]

        # Bottleneck
        z_pre = self.se(e2)
        z = self.bottleneck(z_pre)  # [B, 16]

        # Decoder
        x = self.decode_fc(z).view(B, 128, 1, 1)
        x = F.interpolate(x, size=e2.shape[-2:], mode='nearest')  # [B,128,2,2]
        x = self.up1(x + e2)        # [B,64,2,2] + [B,128,2,2] -> [B,64,2,2]
        #x = self.up2(x)             # [B,64,4,4]
        
        # Final output
        out = self.final(x)         # [B,54,4,4]
        
        return out.view(B, -1), z   # Flatten output, return latent

class WeatherDataset(Dataset):
    def __init__(self, data: np.ndarray, patch_size: int = PATCH_SIZE,
                 time_steps: int = TIME_STEPS, num_samples: int = NUM_TILES_PER_TIME,
                 validation: bool = False, used_coords: set = None):
        """
        Args:
            data: Input array of shape (time, lat, lon, vars)
            patch_size: Size of spatial patches
            time_steps: Number of time steps in each sample
            num_samples: Number of samples per time point
            validation: Whether this is for validation
            used_coords: Set of coordinates already used (to ensure uniqueness)
        """
        # Convert and pad data in one operation (float32)
        self.data = np.pad(
            data.astype(np.float32),
            pad_width=(
                (time_steps-1, 0),  # Time
                (patch_size//2, patch_size//2),  # Lat
                (patch_size//2, patch_size//2),  # Lon
                (0, 0)  # Vars
            ),
            mode='wrap'
        )
        
        self.patch_size = patch_size
        self.time_steps = time_steps
        self.num_vars = data.shape[-1]
        self.validation = validation
        
        # Generate indices
        self.indices = self._generate_indices(
            num_samples if not validation else VALIDATION_SIZE
        )
        
        # PRELOAD ALL PATCHES INTO RAM
        self.preloaded_patches = self._preload_patches()

    def _generate_indices(self, num_samples: int) -> list:
        """Generate random patch indices for training/validation."""
        indices = []
        valid_times = range(self.time_steps-1, self.data.shape[0])
        
        for t in valid_times:
            max_lat = self.data.shape[1] - self.patch_size
            max_lon = self.data.shape[2] - self.patch_size
            
            if self.validation:
                np.random.seed(42)
                
            lats = np.random.randint(0, max_lat, size=num_samples)
            lons = np.random.randint(0, max_lon, size=num_samples)
            
            if self.validation:
                np.random.seed()
                
            indices.extend([(t, lat, lon) for lat, lon in zip(lats, lons)])
        
        return indices

    def _preload_patches(self) -> torch.Tensor:
        num_patches = len(self.indices)
        patch_shape = (self.time_steps, self.patch_size, self.patch_size, self.num_vars)
        patches = torch.empty((num_patches, np.prod(patch_shape)), dtype=torch.float32)
        
        for idx in tqdm(range(len(self.indices)), desc="Preloading data to RAM"):
            t, lat, lon = self.indices[idx]
            patch = self.data[
                t - self.time_steps + 1 : t + 1,
                lat : lat + self.patch_size,
                lon : lon + self.patch_size,
                :
            ]
            patches[idx] = torch.as_tensor(patch, dtype=torch.float32).flatten()
            
        return patches

    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.preloaded_patches[idx]  # Already float32

class FullWeatherDataset(Dataset):
    def __init__(self, data: np.ndarray, patch_size: int = PATCH_SIZE,
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
                (time_steps-1, 0),
                (self.pad, self.pad),
                (self.pad, self.pad),
                (0, 0)
            ),
            mode='wrap'
        )
        
        # Generate indices
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
        # Flatten to [time_steps * patch_size * patch_size * num_vars]
        return torch.tensor(patch, dtype=torch.float32).flatten()

# --- Training Functions ---
def train(rank: int, world_size: int, data: np.ndarray, 
          model_path: Optional[str] = None, output_zarr_path: Optional[str] = None,
          batch_size: int = DEFAULT_BATCH_SIZE, num_workers: int = DEFAULT_NUM_WORKERS,
          prefetch_factor: int = DEFAULT_PREFETCH_FACTOR) -> None:
    
    logger = setup_logging(rank)
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    
    # Initialize distributed training
    if device.type == 'cuda':
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Create model
    model = WeatherUNetImproved(num_vars=data.shape[-1]).to(device)
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
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(1000):
        model.train()
        if train_sampler:
            train_sampler.set_epoch(epoch)
        
        # Training loop with progress bar
        train_loss = 0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]', disable=rank!=0, leave=True)
        for batch in train_bar:
            batch = batch.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            recon, _ = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation loop with progress bar
        model.eval()
        val_loss = 0
        val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1} [Val]', disable=rank!=0)
        with torch.no_grad():
            for batch in val_bar:
                batch = batch.to(device, non_blocking=True)
                recon, _ = model(batch)
                loss = loss_fn(recon, batch)
                val_loss += loss.item()
                val_bar.set_postfix({'val_loss': f'{loss.item():.4f}'})
        
        avg_val_loss = val_loss / len(val_loader)
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
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        if rank == 0:
            logger.info(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Best Val: {best_val_loss:.4f}")
    
    if device.type == 'cuda':
        dist.destroy_process_group()

def run_inference(data: np.ndarray, model_path: str = "best_model.pth",
                 output_zarr_path: Optional[str] = None, batch_size: int = DEFAULT_BATCH_SIZE,
                 num_workers: int = DEFAULT_NUM_WORKERS, prefetch_factor: int = DEFAULT_PREFETCH_FACTOR) -> np.ndarray:
    
    logger = setup_logging(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = WeatherUNetImproved(num_vars=data.shape[-1]).to(device)
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
    
    # Run inference with progress bar
    all_latents = []
    with torch.no_grad():
        for batch in tqdm(full_loader, desc="Running Inference"):
            batch = batch.to(device)
            _, latents = model(batch)
            all_latents.append(latents.cpu())
    
    all_latents = torch.cat(all_latents, dim=0)
    
    # Reshape and pad time dimension
    valid_time = data.shape[0] - TIME_STEPS + 1
    H, W = data.shape[1], data.shape[2]
    all_latents = all_latents.numpy().reshape(valid_time, H, W, LATENT_DIM)
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
    if variable_type == 't2m' or 'ssrd' or 'tp':
        logger.info("Using only tp variable with log1p transformation")
        ds = ds.drop_vars(["u_850", "v_850"])
        # Apply log1p transformation to tp
        ds['tp'] = ds['tp'] * 1000
        # Create binary precipitation variable (0 for <0.01mm, 1 for ≥0.01mm)
        ds['precip_binary'] = xr.where(ds['tp'] >= 0.01, 1, 0)
        ds['tp'] = np.cbrt(ds["tp"])
        logger.info("Using only t2m variable")  
        ds = ds.sel(time=slice("2005-01-01", "2020-12-31"))

        
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