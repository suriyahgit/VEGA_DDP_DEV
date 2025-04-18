import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import xarray as xr
import string
from datetime import datetime
import logging
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os

# --- Enhanced logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Constants ---
LATENT_DIM = 9
PATCH_SIZE = 3
TIME_STEPS = 5
NUM_WORKERS = int(os.cpu_count() * 0.8)  # Use 80% of available cores

def setup_distributed():
    """Initialize distributed training if available"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(gpu)
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        return True, rank, world_size
    return False, 0, 1

class XarrayWeatherDataset(Dataset):
    def __init__(self, data, patch_size=3, time_steps=5, num_tiles_per_time=500):
        self.data = data
        self.patch_size = patch_size
        self.time_steps = time_steps
        self.pad = patch_size // 2
        self.num_tiles_per_time = num_tiles_per_time

        # Pad data with memory-efficient method
        self.data = self._pad_data(data)
        
        # Generate indices with chunked processing for large datasets
        self.indices = self._generate_indices()

    def _pad_data(self, data):
        """Memory-efficient padding"""
        logger.info(f"Applying padding to data of shape {data.shape}")
        padded_shape = (
            data.shape[0] + self.time_steps - 1,
            data.shape[1] + 2 * self.pad,
            data.shape[2] + 2 * self.pad,
            data.shape[3]
        )
        padded_data = np.zeros(padded_shape, dtype=data.dtype)
        padded_data[self.time_steps-1:, self.pad:-self.pad, self.pad:-self.pad] = data
        # Handle periodic boundary conditions
        padded_data[self.time_steps-1:, :self.pad] = data[:, -self.pad:]
        padded_data[self.time_steps-1:, -self.pad:] = data[:, :self.pad]
        # Time padding (repeat first frame)
        for t in range(self.time_steps-1):
            padded_data[t] = padded_data[self.time_steps-1]
        return padded_data

    def _generate_indices(self):
        """Generate indices with chunked processing"""
        logger.info("Generating indices...")
        total_time = self.data.shape[0]
        H, W = self.data.shape[1], self.data.shape[2]
        
        # Process in chunks to reduce memory usage
        chunk_size = 100  # Process 100 time steps at a time
        indices = []
        for t_start in range(self.time_steps - 1, total_time, chunk_size):
            t_end = min(t_start + chunk_size, total_time)
            for t in range(t_start, t_end):
                lat = np.random.randint(0, H - self.patch_size + 1, size=self.num_tiles_per_time)
                lon = np.random.randint(0, W - self.patch_size + 1, size=self.num_tiles_per_time)
                indices.extend([(t, lat[i], lon[i]) for i in range(self.num_tiles_per_time)])
        logger.info(f"Generated {len(indices)} samples")
        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t, lat, lon = self.indices[idx]
        patch = self.data[
            t - self.time_steps + 1 : t + 1,
            lat : lat + self.patch_size,
            lon : lon + self.patch_size,
            :
        ]
        return torch.tensor(patch, dtype=torch.float32).flatten()

class FullXarrayWeatherDataset(Dataset):
    def __init__(self, data, patch_size=3, time_steps=5):
        self.data = self._pad_data(data, patch_size, time_steps)
        self.patch_size = patch_size
        self.time_steps = time_steps
        self.valid_time = data.shape[0]
        self.H, self.W = data.shape[1], data.shape[2]
        self.indices = self._generate_indices()

    def _pad_data(self, data, patch_size, time_steps):
        """Optimized padding for inference"""
        pad = patch_size // 2
        padded_shape = (
            data.shape[0] + time_steps - 1,
            data.shape[1] + 2 * pad,
            data.shape[2] + 2 * pad,
            data.shape[3]
        )
        padded_data = np.zeros(padded_shape, dtype=data.dtype)
        padded_data[time_steps-1:, pad:-pad, pad:-pad] = data
        # Handle periodic boundaries
        padded_data[time_steps-1:, :pad] = data[:, -pad:]
        padded_data[time_steps-1:, -pad:] = data[:, :pad]
        # Time padding
        for t in range(time_steps-1):
            padded_data[t] = padded_data[time_steps-1]
        return padded_data

    def _generate_indices(self):
        """Generate indices in memory-efficient way"""
        indices = []
        chunk_size = 100  # Process 100 time steps at a time
        for t_start in range(self.time_steps - 1, self.valid_time + self.time_steps - 1, chunk_size):
            t_end = min(t_start + chunk_size, self.valid_time + self.time_steps - 1)
            for t in range(t_start, t_end):
                for i in range(self.H):
                    for j in range(self.W):
                        indices.append((t, i, j))
        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t, lat, lon = self.indices[idx]
        patch = self.data[
            t - self.time_steps + 1 : t + 1,
            lat : lat + self.patch_size,
            lon : lon + self.patch_size,
            :
        ]
        return torch.tensor(patch, dtype=torch.float32).flatten()

class WeatherAutoencoder(nn.Module):
    def __init__(self, input_dim=360, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),  # Increased capacity
            nn.LeakyReLU(0.1),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, input_dim),
        )

    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon, latent

def main():
    # Initialize distributed training if available
    distributed, rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    
    # --- Data Loading ---
    if rank == 0:
        logger.info("Loading data...")
    
    ds = xr.open_zarr("/ceph/hpc/home/dhinakarans/data/autoencoder/ERA5_to_latent.zarr")
    ds = ds.drop_vars(['ssrd', 'tp'])
    ds = ds.isel(lat=slice(0, 50), lon=slice(0, 50))
    variables = list(ds.data_vars.keys())
    
    # Memory-efficient data loading
    data = np.stack([ds[var].values for var in variables], axis=-1)
    means = np.mean(data, axis=(0,1,2), keepdims=True)
    stds = np.std(data, axis=(0,1,2), keepdims=True)
    data = (data - means) / (stds + 1e-8)
    
    # --- Training Setup ---
    if rank == 0:
        logger.info("Setting up training...")
    
    dataset = XarrayWeatherDataset(
        data,
        patch_size=PATCH_SIZE,
        time_steps=TIME_STEPS,
        num_tiles_per_time=100
    )
    
    # Use DistributedSampler if in distributed mode
    sampler = DistributedSampler(dataset) if distributed else None
    dataloader = DataLoader(
        dataset,
        batch_size=2048 if distributed else 1024,  # Larger batches for distributed
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=2
    )
    
    # --- Model Setup ---
    input_dim = PATCH_SIZE * PATCH_SIZE * TIME_STEPS * len(variables)
    model = WeatherAutoencoder(input_dim=input_dim, latent_dim=LATENT_DIM)
    
    if distributed:
        model = DDP(model.to(device), device_ids=[rank])
    else:
        model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    loss_fn = nn.MSELoss()

    # --- Training Loop ---
    if rank == 0:
        logger.info("Starting training...")
    
    best_loss = float('inf')
    patience = 25
    counter = 0
    
    for epoch in range(250):
        if distributed:
            sampler.set_epoch(epoch)
        
        model.train()
        total_loss = 0
        start_time = datetime.now()
        
        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(device, non_blocking=True)
            recon, _ = model(batch)
            loss = loss_fn(recon, batch)
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if rank == 0 and batch_idx % 100 == 0:
                logger.info(f"Epoch {epoch+1} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")
        
        if distributed:
            dist.barrier()
            total_loss_tensor = torch.tensor(total_loss).to(device)
            dist.all_reduce(total_loss_tensor)
            total_loss = total_loss_tensor.item() / world_size
        
        epoch_loss = total_loss/len(dataloader)
        scheduler.step(epoch_loss)
        
        if rank == 0:
            epoch_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f}s | Avg Loss: {epoch_loss:.4f}")
            
            # Early stopping logic
            if epoch_loss < best_loss:
                logger.info(f"Loss improved from {best_loss:.4f} to {epoch_loss:.4f}")
                best_loss = epoch_loss
                counter = 0
                torch.save(model.module.state_dict() if hasattr(model, 'module') else model.state_dict(), 
                          'best_model.pth')
            else:
                counter += 1
                logger.info(f"No improvement for {counter}/{patience} epochs")
                if counter >= patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs!")
                    break

    # --- Inference ---
    if rank == 0:
        logger.info("Starting inference...")
        model.load_state_dict(torch.load('best_model.pth'))
        model.eval()
        
        full_dataset = FullXarrayWeatherDataset(data, patch_size=PATCH_SIZE, time_steps=TIME_STEPS)
        full_loader = DataLoader(
            full_dataset,
            batch_size=4096,  # Larger batches for inference
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True
        )
        
        all_latents = []
        with torch.no_grad():
            for batch in full_loader:
                batch = batch.to(device, non_blocking=True)
                _, latents = model(batch)
                all_latents.append(latents.cpu())
        
        all_latents = torch.cat(all_latents, dim=0)
        valid_time = data.shape[0] - TIME_STEPS + 1
        H, W = data.shape[1], data.shape[2]
        
        all_latents_np = all_latents.numpy().reshape(valid_time, H, W, LATENT_DIM)
        pad_front = np.repeat(all_latents_np[[0]], TIME_STEPS - 1, axis=0)
        full_latents = np.concatenate([pad_front, all_latents_np], axis=0)
        
        logger.info(f"Final latent array shape: {full_latents.shape}")
        
        # --- Save Results ---
        logger.info("Saving results...")
        time = ds.coords["time"].values
        lat = ds.coords["lat"].values
        lon = ds.coords["lon"].values
        
        variables = {}
        for i in range(full_latents.shape[-1]):
            var_name = f"latent_{string.ascii_uppercase[i]}"
            variables[var_name] = xr.DataArray(
                full_latents[..., i],
                dims=["time", "lat", "lon"],
                coords={"time": time, "lat": lat, "lon": lon}
            )
        
        latent_ds = xr.Dataset(variables)
        latent_ds.to_zarr("weather_latents.zarr", mode="w")
        logger.info("Latent space saved successfully!")

if __name__ == "__main__":
    main()