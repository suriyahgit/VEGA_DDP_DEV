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
NUM_TILES_PER_TIME = 50000
BATCH_SIZE = 8048
NUM_WORKERS = 16
EARLY_STOPPING_PATIENCE = 25
MIN_LR = 1e-6

class XarrayWeatherDataset(Dataset):
    def __init__(self, data, patch_size=PATCH_SIZE, time_steps=TIME_STEPS, num_tiles_per_time=NUM_TILES_PER_TIME):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing dataset with patch_size=%d, time_steps=%d", patch_size, time_steps)
        
        self.data = data
        self.patch_size = patch_size
        self.time_steps = time_steps
        self.pad = patch_size // 2
        self.num_tiles_per_time = num_tiles_per_time

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
        
        self.logger.info("Generating %d random tiles per time step", num_tiles_per_time)
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


def train(rank, world_size, data):
    logger = setup_logging(rank)
    logger.info("Initializing training process on rank %d", rank)
    
    # Initialize distributed training
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Create model
    input_dim = PATCH_SIZE * PATCH_SIZE * TIME_STEPS * data.shape[-1]
    model = WeatherAutoencoder(input_dim).to(rank)
    model = DDP(model, device_ids=[rank])
    
    logger.info("Model created on device %d", rank)
    logger.info("Number of parameters: %d", sum(p.numel() for p in model.parameters()))
    
    # Create datasets and dataloaders
    train_dataset = XarrayWeatherDataset(data)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    logger.info("DataLoader initialized with batch_size=%d, num_workers=%d", BATCH_SIZE, NUM_WORKERS)
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=1e-3 * world_size)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    loss_fn = nn.MSELoss()
    
    # Training loop with early stopping
    best_loss = float('inf')
    patience_counter = 0
    
    logger.info("Starting training loop at %s", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    for epoch in range(1000):  # Large number for early stopping
        model.train()
        train_sampler.set_epoch(epoch)
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(rank, non_blocking=True)
            recon, _ = model(batch)
            loss = loss_fn(recon, batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx%100==0 & rank == 0:
                logger.info("Epoch %d, Batch %d, Current Loss: %.4f", epoch+1, batch_idx, loss.item())
        
        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)
        
        # Early stopping logic
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            if rank == 0:
                torch.save(model.state_dict(), "best_model.pth")
                logger.info("New best model saved with loss %.4f", best_loss)
        else:
            patience_counter += 1
            logger.info("No improvement (%d/%d), Current Loss: %.4f, Best Loss: %.4f", 
                       patience_counter, EARLY_STOPPING_PATIENCE, avg_loss, best_loss)
            
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                logger.info("Early stopping triggered at epoch %d", epoch+1)
                break
        
        if rank == 0:
            current_lr = optimizer.param_groups[0]['lr']
            logger.info("Epoch %d complete - Loss: %.4f, Best: %.4f, LR: %.2e", 
                       epoch+1, avg_loss, best_loss, current_lr)
            
            if current_lr < MIN_LR:
                logger.info("Learning rate below minimum threshold (%.2e < %.2e), stopping training", 
                           current_lr, MIN_LR)
                break
    
    logger.info("Training completed at %s", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    dist.destroy_process_group()

def run_inference(data, model_path="best_model.pth"):
    logger = setup_logging(0)
    logger.info("Starting inference process")
    
    full_dataset = FullXarrayWeatherDataset(data)
    full_loader = DataLoader(
        full_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    logger.info("Inference dataset loaded with %d samples", len(full_dataset))
    
    input_dim = PATCH_SIZE * PATCH_SIZE * TIME_STEPS * data.shape[-1]
    model = WeatherAutoencoder(input_dim)
    
    logger.info("Loading model from %s", model_path)
    model.load_state_dict(torch.load(model_path))
    
    if torch.cuda.device_count() > 1:
        logger.info("Using %d GPUs for inference", torch.cuda.device_count())
        model = nn.DataParallel(model)
    
    model.to('cuda')
    model.eval()
    
    logger.info("Running inference...")
    all_latents = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(full_loader):
            batch = batch.to('cuda', non_blocking=True)
            _, latents = model(batch)
            all_latents.append(latents.cpu())
            
            if batch_idx % 100 == 0:
                logger.info("Processed %d/%d batches", batch_idx, len(full_loader))
    
    logger.info("Inference completed, processing results...")
    all_latents = torch.cat(all_latents, dim=0)
    
    # Reshape and pad
    valid_time = data.shape[0] - TIME_STEPS + 1
    H, W = data.shape[1], data.shape[2]
    
    all_latents_np = all_latents.numpy().reshape(valid_time, H, W, LATENT_DIM)
    pad_front = np.repeat(all_latents_np[[0]], TIME_STEPS - 1, axis=0)
    full_latents = np.concatenate([pad_front, all_latents_np], axis=0)
    
    logger.info("Final latent array shape: %s", full_latents.shape)
    return full_latents

def main():
    logger = setup_logging(0)
    logger.info("Starting main execution")
    
    try:
        logger.info("Loading and preprocessing data...")

       # 1. Open Zarr dataset with Dask chunks
        ds = xr.open_zarr(
            "/ceph/hpc/home/dhinakarans/data/autoencoder/ERA5_to_latent.zarr",
            chunks={}  # Let Dask determine optimal chunking
        )
        logger.info("Initial dataset loaded. Shape: %s, Variables: %s", 
                   dict(ds.dims), list(ds.data_vars.keys()))
        
        # 2. Preprocess data (still lazy)
        ds = ds.drop_vars(['ssrd', 'tp']).fillna(0)
        variables = list(ds.data_vars.keys())
        logger.info("Using variables: %s", variables)

        load_start = time.time()  # Changed variable name
        
        # Initialize empty list to store normalized chunks
        normalized_chunks = []
        
        # 3. Process each variable separately to minimize memory usage
        for var in variables:
            logger.info(f"Processing variable: {var}")
            
            # Compute normalization parameters for this variable only
            with ProgressBar():
                var_mean = ds[var].mean(dim=['time', 'lat', 'lon']).compute()
                var_std = ds[var].std(dim=['time', 'lat', 'lon']).compute()
            
            logger.info(f"Computed mean for {var}: {var_mean.values}")
            logger.info(f"Computed std for {var}: {var_std.values}")
            
            # Normalize this variable (still lazy)
            normalized_var = (ds[var] - var_mean) / (var_std + 1e-8)
            
            # Compute this variable and add new dimension for stacking
            with ProgressBar():
                normalized_array = normalized_var.compute().expand_dims('variable')
            
            # Append to our list (now in memory)
            normalized_chunks.append(normalized_array)
            
            # Clean up to free memory
            del normalized_var, normalized_array
        
        # 4. Combine all normalized variables along the new dimension
        logger.info("Combining all normalized variables...")
        data = xr.concat(normalized_chunks, dim='variable')
        data = data.transpose('time', 'lat', 'lon', 'variable')
        
        logger.info("Data loaded in %.2f seconds", time.time() - load_start)
        logger.info("Final data shape: %s", data.shape)
        logger.info("Memory usage: %.2f GB", data.nbytes / 1e9)
        
        # Verify no NaN values
        assert not np.isnan(data).any(), "Data contains NaN values after normalization"
        logger.info("Data validation passed - no NaN values detected")
    
        
        # Multi-GPU training
        world_size = torch.cuda.device_count()
        if world_size > 1:
            logger.info("Starting multi-GPU training with %d GPUs", world_size)
            mp.spawn(train, args=(world_size, data), nprocs=world_size, join=True)
        else:
            logger.info("Starting single-GPU training")
            train(0, 1, data)
        
        logger.info("Training completed, starting inference")
        full_latents = run_inference(data)
        
        # Create xarray dataset
        time_coords = ds.coords["time"].values  # Changed variable name
        lat = ds.coords["lat"].values
        lon = ds.coords["lon"].values
        
        variables = {}
        for i in range(full_latents.shape[-1]):
            var_name = f"latent_{string.ascii_uppercase[i]}"
            variables[var_name] = xr.DataArray(
                full_latents[..., i],
                dims=["time", "lat", "lon"],
                coords={"time": time_coords, "lat": lat, "lon": lon}  # Updated variable name
            )
        
        latent_ds = xr.Dataset(variables)
        logger.info("Final dataset created")
        
        return latent_ds
    
    except Exception as e:
        logger.error("Error in main execution: %s", str(e), exc_info=True)
        raise

if __name__ == "__main__":
    logger = setup_logging(0)  # Make sure logger is defined
    try:
        latent_ds = main()
        latent_ds.to_zarr(f"ERA5_dem_by_3_latent_ds_t2m_{LATENT_DIM}.zarr", mode="w")
        logger.info("✅ Successfully completed execution")
        logger.info("Final latent array shape: %s", latent_ds)
    except Exception as e:
        logger.error("❌ Execution failed: %s", str(e), exc_info=True)