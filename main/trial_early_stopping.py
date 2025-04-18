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

# --- Constants ---
PATCH_SIZE = 3
TIME_STEPS = 5
LATENT_DIM = 9
NUM_TILES_PER_TIME = 500
BATCH_SIZE = 2048  # Increased batch size for multi-GPU
NUM_WORKERS = 8
NUM_EPOCHS = 10

# --- Dataset Classes (minimal changes) ---
class XarrayWeatherDataset(Dataset):
    def __init__(self, data, patch_size=PATCH_SIZE, time_steps=TIME_STEPS, num_tiles_per_time=NUM_TILES_PER_TIME):
        self.data = data
        self.patch_size = patch_size
        self.time_steps = time_steps
        self.pad = patch_size // 2
        self.num_tiles_per_time = num_tiles_per_time

        # Pad data (time, space)
        self.data = np.pad(
            data,
            ((time_steps-1, 0), (self.pad, self.pad), (self.pad, self.pad), (0, 0)),
            mode='wrap'
        )

        # Precompute random tile indices
        self.total_time = self.data.shape[0]
        self.H, self.W = self.data.shape[1], self.data.shape[2]
        
        # Generate random indices (time, lat, lon)
        self.indices = []
        for t in range(time_steps - 1, self.total_time):
            lat = np.random.randint(0, self.H - patch_size + 1, size=num_tiles_per_time)
            lon = np.random.randint(0, self.W - patch_size + 1, size=num_tiles_per_time)
            self.indices.extend([(t, lat[i], lon[i]) for i in range(num_tiles_per_time)])

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
    def __init__(self, data, patch_size=PATCH_SIZE, time_steps=TIME_STEPS):
        self.patch_size = patch_size
        self.time_steps = time_steps
        self.pad = patch_size // 2

        self.data = np.pad(
            data,
            ((time_steps - 1, 0), (self.pad, self.pad), (self.pad, self.pad), (0, 0)),
            mode='wrap'
        )

        self.valid_time = data.shape[0]
        self.H, self.W = data.shape[1], data.shape[2]

        self.indices = [
            (t, i, j)
            for t in range(time_steps - 1, self.valid_time)
            for i in range(self.H)
            for j in range(self.W)
        ]

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

# --- Model with Multi-GPU support ---
class WeatherAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),  # Increased capacity
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

    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon, latent

# --- Training Function for Multi-GPU ---
def train(rank, world_size, data):
    # Initialize distributed training
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Create model and move to GPU
    input_dim = PATCH_SIZE * PATCH_SIZE * TIME_STEPS * data.shape[-1]
    model = WeatherAutoencoder(input_dim).to(rank)
    model = DDP(model, device_ids=[rank])
    
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
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=1e-3 * world_size)  # Scale learning rate
    loss_fn = nn.MSELoss()
    
    # Training loop with early stopping
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_sampler.set_epoch(epoch)
        total_loss = 0
        
        for batch in train_loader:
            batch = batch.to(rank, non_blocking=True)
            recon, _ = model(batch)
            loss = loss_fn(recon, batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        # Early stopping logic
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # Save best model
            if rank == 0:
                torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1
            if rank == 0:
                print(f"No improvement ({patience_counter}/{EARLY_STOPPING_PATIENCE})")
            
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                if rank == 0:
                    print(f"Early stopping at epoch {epoch+1}")
                break
        
        if rank == 0:
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Best: {best_loss:.4f}")
    
    # Cleanup
    dist.destroy_process_group()

# --- Inference Function ---
def run_inference(data, model_path=None):
    # Load data
    full_dataset = FullXarrayWeatherDataset(data)
    full_loader = DataLoader(
        full_dataset,
        batch_size=BATCH_SIZE * 2,  # Larger batches for inference
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    # Create model
    input_dim = PATCH_SIZE * PATCH_SIZE * TIME_STEPS * data.shape[-1]
    model = WeatherAutoencoder(input_dim)
    
    # Load weights if provided
    if model_path:
        model.load_state_dict(torch.load(model_path))
    
    # Multi-GPU inference
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for inference!")
        model = nn.DataParallel(model)
    
    model.to('cuda')
    model.eval()
    
    # Inference
    all_latents = []
    with torch.no_grad():
        for batch in full_loader:
            batch = batch.to('cuda', non_blocking=True)
            _, latents = model(batch)
            all_latents.append(latents.cpu())
    
    all_latents = torch.cat(all_latents, dim=0)
    
    # Reshape and pad
    valid_time = data.shape[0] - TIME_STEPS + 1
    H, W = data.shape[1], data.shape[2]
    
    all_latents_np = all_latents.numpy().reshape(valid_time, H, W, LATENT_DIM)
    pad_front = np.repeat(all_latents_np[[0]], TIME_STEPS - 1, axis=0)
    full_latents = np.concatenate([pad_front, all_latents_np], axis=0)
    
    return full_latents

# --- Main Function ---
def main():
    # Load and preprocess data
    ds = xr.open_zarr("/mnt/CEPH_PROJECTS/InterTwin/Climate_Downscaling/EMO1_DOWNSCALING/stage_1/ERA5_to_latent.zarr")
    ds = ds.drop_vars(['ssrd', 'tp'])
    ds = ds.isel(lat=slice(0, 30), lon=slice(0, 41))
    ds = ds.fillna(0)
    
    variables = list(ds.data_vars.keys())
    data = np.stack([ds[var].values for var in variables], axis=-1)
    
    # Normalize
    means = data.mean(axis=(0,1,2), keepdims=True)
    stds = data.std(axis=(0,1,2), keepdims=True)
    data = (data - means) / (stds + 1e-8)
    
    # Multi-GPU training
    world_size = torch.cuda.device_count()
    if world_size > 1:
        print(f"Training with {world_size} GPUs!")
        mp.spawn(train, args=(world_size, data), nprocs=world_size, join=True)
    else:
        train(0, 1, data)
    
    # Run inference
    full_latents = run_inference(data)
    
    # Create xarray dataset
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
    return latent_ds

if __name__ == "__main__":
    latent_ds = main()
    print("✅ Final latent array shape:", latent_ds)