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

gc.collect()

# --- Configure logging ---
def setup_logging(rank):
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
PATCH_SIZE = 3
TIME_STEPS = 5  # Creates t-4, t-3, t-2, t-1, t patterns
LATENT_DIM = 9
NUM_TILES_PER_TIME = 5000
DEFAULT_BATCH_SIZE = 64384
DEFAULT_NUM_WORKERS = 8
EARLY_STOPPING_PATIENCE = 10
MIN_LR = 1e-6
DEFAULT_PREFETCH_FACTOR = 25

class XarrayWeatherDataset(Dataset):
    def __init__(self, data, patch_size=PATCH_SIZE, time_steps=TIME_STEPS, 
                 num_tiles_per_time=NUM_TILES_PER_TIME, validation=False, validation_size=500):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing dataset with patch_size=%d, time_steps=%d", patch_size, time_steps)
        
        self.data = data
        self.patch_size = patch_size
        self.time_steps = time_steps
        self.pad = patch_size // 2
        self.num_tiles_per_time = num_tiles_per_time
        self.validation = validation
        self.validation_size = validation_size

        # Log dataset stats before padding
        self.logger.info("Original data shape: %s", data.shape)
        
        # Pad data (time, space)
        self.data = np.pad(
            data,
            ((time_steps-1, 0), (self.pad, self.pad), (self.pad, self.pad), (0, 0)),
            mode='wrap'
        )
        self.logger.info("Padded data shape: %s", self.data.shape)

        # Precompute random tile indices
        self.total_time = self.data.shape[0]
        self.H, self.W = self.data.shape[1], self.data.shape[2]
        
        if validation:
            self.logger.info("Generating %d fixed validation tiles", validation_size)
            # Use fixed seed for validation to ensure same tiles every time
            np.random.seed(42)
            self.indices = []
            for t in range(time_steps - 1, self.total_time):
                lat = np.random.randint(0, self.H - patch_size + 1, size=validation_size)
                lon = np.random.randint(0, self.W - patch_size + 1, size=validation_size)
                self.indices.extend([(t, lat[i], lon[i]) for i in range(validation_size)])
            np.random.seed()  # Reset random seed
        else:
            self.logger.info("Generating %d random training tiles per time step", num_tiles_per_time)
            self.indices = []
            for t in range(time_steps - 1, self.total_time):
                lat = np.random.randint(0, self.H - patch_size + 1, size=num_tiles_per_time)
                lon = np.random.randint(0, self.W - patch_size + 1, size=num_tiles_per_time)
                self.indices.extend([(t, lat[i], lon[i]) for i in range(num_tiles_per_time)])

        self.logger.info("Total samples generated: %d", len(self.indices))

    def __len__(self):
        """Returns the total number of samples in the dataset"""
        return len(self.indices)

    def __getitem__(self, idx):
        """Returns a single sample from the dataset"""
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset with length {len(self)}")
            
        t, lat, lon = self.indices[idx]
        # This creates the t-4, t-3, t-2, t-1, t pattern when time_steps=5
        patch = self.data[
            t - self.time_steps + 1 : t + 1,  # Time dimension
            lat : lat + self.patch_size,      # Latitude dimension
            lon : lon + self.patch_size,      # Longitude dimension
            :                                 # Variable dimension
        ]
        return torch.tensor(patch, dtype=torch.float32).flatten()

class FullXarrayWeatherDataset(Dataset):
    def __init__(self, data, patch_size=PATCH_SIZE, time_steps=TIME_STEPS):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing full dataset with patch_size=%d, time_steps=%d", patch_size, time_steps)
        
        self.patch_size = patch_size
        self.time_steps = time_steps
        self.pad = patch_size // 2

        self.data = np.pad(
            data,
            ((time_steps - 1, 0), (self.pad, self.pad), (self.pad, self.pad), (0, 0)),
            mode='wrap'
        )
        self.logger.info("Padded data shape: %s", self.data.shape)

        self.valid_time = data.shape[0]
        self.H, self.W = data.shape[1], data.shape[2]

        self.indices = [
            (t, i, j)
            for t in range(time_steps - 1, self.valid_time)
            for i in range(self.H)
            for j in range(self.W)
        ]
        self.logger.info("Total samples in full dataset: %d", len(self.indices))

    def __len__(self):
        """Returns the total number of samples in the dataset"""
        return len(self.indices)

    def __getitem__(self, idx):
        """Returns a single sample from the dataset"""
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset with length {len(self)}")
            
        t, lat, lon = self.indices[idx]
        patch = self.data[
            t - self.time_steps + 1 : t + 1,
            lat : lat + self.patch_size,
            lon : lon + self.patch_size,
            :
        ]
        return torch.tensor(patch, dtype=torch.float32).flatten()

class WeatherAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=LATENT_DIM):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing autoencoder with input_dim=%d, latent_dim=%d", input_dim, latent_dim)
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
        )
        
        self.logger.info("Model architecture:\n%s", self)

    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon, latent


def train(rank, world_size, data, model_path=None, output_zarr_path=None, batch_size=DEFAULT_BATCH_SIZE, 
          num_workers=DEFAULT_NUM_WORKERS, prefetch_factor=DEFAULT_PREFETCH_FACTOR):
    logger = setup_logging(rank)
    logger.info("Initializing training process on rank %d", rank)
    
    # Check device
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        warnings.warn("Training is happening on CPU instead of CUDA! Performance will be significantly worse.")
    
    # Initialize distributed training if using CUDA
    if device.type == 'cuda':
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Create model
    input_dim = PATCH_SIZE * PATCH_SIZE * TIME_STEPS * data.shape[-1]
    model = WeatherAutoencoder(input_dim).to(device)
    if device.type == 'cuda' and world_size > 1:
        model = DDP(model, device_ids=[rank])
    
    logger.info("Model created on device %s", device)
    logger.info("Number of parameters: %d", sum(p.numel() for p in model.parameters()))
    
    # Create datasets and dataloaders
    train_dataset = XarrayWeatherDataset(data, validation=False)
    val_dataset = XarrayWeatherDataset(data, validation=True)
    
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
    
    logger.info("DataLoaders initialized with batch_size=%d, num_workers=%d", batch_size, num_workers)
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=1e-3 * world_size)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    loss_fn = nn.MSELoss()
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    
    logger.info("Starting training loop at %s", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    for epoch in range(1000):  # Large number for early stopping
        model.train()
        if train_sampler:
            train_sampler.set_epoch(epoch)
        total_train_loss = 0
        
        # Training phase
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device, non_blocking=True)
            recon, _ = model(batch)
            loss = loss_fn(recon, batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
            if batch_idx % 100 == 0 and rank == 0:
                logger.info("Epoch %d, Batch %d, Current Loss: %.4f", epoch+1, batch_idx, loss.item())
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation phase
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
        
        # Early stopping logic based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            if rank == 0:
                save_path = model_path if model_path else "best_model.pth"
                torch.save(model.module.state_dict() if hasattr(model, 'module') else model.state_dict(), save_path)
                logger.info("New best model saved to %s with val loss %.4f", save_path, best_val_loss)
        else:
            patience_counter += 1
            logger.info("No improvement (%d/%d), Current Val Loss: %.4f, Best Val Loss: %.4f", 
                       patience_counter, EARLY_STOPPING_PATIENCE, avg_val_loss, best_val_loss)
            
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                logger.info("Early stopping triggered at epoch %d", epoch+1)
                break
        
        if rank == 0:
            current_lr = optimizer.param_groups[0]['lr']
            logger.info("Epoch %d complete - Train Loss: %.4f, Val Loss: %.4f, Best Val: %.4f, LR: %.2e", 
                       epoch+1, avg_train_loss, avg_val_loss, best_val_loss, current_lr)
            
            if current_lr < MIN_LR:
                logger.info("Learning rate below minimum threshold (%.2e < %.2e), stopping training", 
                           current_lr, MIN_LR)
                break
    
    logger.info("Training completed at %s", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    if device.type == 'cuda':
        dist.destroy_process_group()

def run_inference(data, model_path="best_model.pth", output_zarr_path=None, batch_size=DEFAULT_BATCH_SIZE, 
                 num_workers=DEFAULT_NUM_WORKERS, prefetch_factor=DEFAULT_PREFETCH_FACTOR):
    logger = setup_logging(0)
    logger.info("Starting inference process")
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        warnings.warn("Inference is happening on CPU instead of CUDA! Performance will be significantly worse.")
    
    full_dataset = FullXarrayWeatherDataset(data)
    full_loader = DataLoader(
        full_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=True,
        drop_last=False  # Changed from True to False to process all samples
    )
    
    logger.info("Inference dataset loaded with %d samples", len(full_dataset))
    
    input_dim = PATCH_SIZE * PATCH_SIZE * TIME_STEPS * data.shape[-1]
    model = WeatherAutoencoder(input_dim)
    
    logger.info("Loading model from %s", model_path)
    model.load_state_dict(torch.load(model_path))
    
    if torch.cuda.device_count() > 1:
        logger.info("Using %d GPUs for inference", torch.cuda.device_count())
        model = nn.DataParallel(model)
    
    model.to(device)
    model.eval()
    
    logger.info("Running inference...")
    all_latents = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(full_loader):
            batch = batch.to(device, non_blocking=True)
            _, latents = model(batch)
            all_latents.append(latents.cpu())
            
            if batch_idx % 100 == 0:
                logger.info("Processed %d/%d batches", batch_idx, len(full_loader))
    
    logger.info("Inference completed, processing results...")
    all_latents = torch.cat(all_latents, dim=0)
    
    # Calculate expected shape
    valid_time = data.shape[0] - TIME_STEPS + 1
    H, W = data.shape[1], data.shape[2]
    expected_elements = valid_time * H * W * LATENT_DIM
    
    # Verify shape before reshaping
    if all_latents.numel() != expected_elements:
        logger.error("Shape mismatch! Got %d elements but expected %d", 
                   all_latents.numel(), expected_elements)
        raise ValueError(f"Shape mismatch: Got {all_latents.numel()} elements but expected {expected_elements}")
    
    # Reshape and pad
    all_latents_np = all_latents.numpy().reshape(valid_time, H, W, LATENT_DIM)
    pad_front = np.repeat(all_latents_np[[0]], TIME_STEPS - 1, axis=0)
    full_latents = np.concatenate([pad_front, all_latents_np], axis=0)
    
    logger.info("Final latent array shape: %s", full_latents.shape)
    
    # Save to zarr if output path provided
    if output_zarr_path:
        time_coords = np.arange(data.shape[0])
        lat = np.arange(data.shape[1])
        lon = np.arange(data.shape[2])
        
        variables = {}
        for i in range(full_latents.shape[-1]):
            var_name = f"latent_{string.ascii_uppercase[i]}"
            variables[var_name] = xr.DataArray(
                full_latents[..., i],
                dims=["time", "lat", "lon"],
                coords={"time": time_coords, "lat": lat, "lon": lon}
            )
        
        latent_ds = xr.Dataset(variables)
        logger.info("Saving latent dataset to %s", output_zarr_path)
        latent_ds.to_zarr(output_zarr_path, mode="w")
        logger.info("Zarr file saved successfully")
    
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