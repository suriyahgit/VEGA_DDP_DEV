import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import xarray as xr
import string
from datetime import datetime
import logging

# --- Setup logging ---
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
LATENT_DIM = 9  # Single consistent definition
PATCH_SIZE = 3
TIME_STEPS = 5

class XarrayWeatherDataset(Dataset):
    def __init__(self, data, patch_size=3, time_steps=5, num_tiles_per_time=500):
        logger.info("Initializing dataset...")
        self.data = data
        self.patch_size = patch_size
        self.time_steps = time_steps
        self.pad = patch_size // 2
        self.num_tiles_per_time = num_tiles_per_time

        # Pad data (time, space)
        logger.info("Applying periodic padding...")
        self.data = np.pad(
            data,
            ((time_steps-1, 0), (self.pad, self.pad), (self.pad, self.pad), (0, 0)),
            mode='wrap'
        )

        # Precompute random tile indices
        self.total_time = self.data.shape[0]
        self.H, self.W = self.data.shape[1], self.data.shape[2]
        
        logger.info(f"Generating {num_tiles_per_time} random indices per time step...")
        self.indices = []
        for t in range(time_steps - 1, self.total_time):
            lat = np.random.randint(0, self.H - patch_size + 1, size=num_tiles_per_time)
            lon = np.random.randint(0, self.W - patch_size + 1, size=num_tiles_per_time)
            self.indices.extend([(t, lat[i], lon[i]) for i in range(num_tiles_per_time)])
        
        logger.info(f"Dataset initialized with {len(self.indices)} total samples")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t, lat, lon = self.indices[idx]
        # Temporal order: t-4, t-3, t-2, t-1, t when time_steps=5
        patch = self.data[
            t - self.time_steps + 1 : t + 1,
            lat : lat + self.patch_size,
            lon : lon + self.patch_size,
            :
        ]
        return torch.tensor(patch, dtype=torch.float32).flatten()

class FullXarrayWeatherDataset(Dataset):
    def __init__(self, data, patch_size=3, time_steps=5):
        logger.info("Initializing full dataset for inference...")
        self.patch_size = patch_size
        self.time_steps = time_steps
        self.pad = patch_size // 2

        # Pad data (time, space)
        self.data = np.pad(
            data,
            ((time_steps - 1, 0), (self.pad, self.pad), (self.pad, self.pad), (0, 0)),
            mode='wrap'
        )

        self.valid_time = data.shape[0]  # Original time dimension
        self.H, self.W = data.shape[1], data.shape[2]  # Original spatial dimensions

        # Create indices for all valid points
        self.indices = [
            (t, i, j)
            for t in range(time_steps - 1, self.valid_time + time_steps - 1)
            for i in range(self.H)
            for j in range(self.W)
        ]
        logger.info(f"Full dataset initialized with {len(self.indices)} samples")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t, lat, lon = self.indices[idx]
        # Temporal order: t-4, t-3, t-2, t-1, t when time_steps=5
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
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon, latent



def main():
    # --- Data Loading ---
    logger.info("Loading data...")
    ds = xr.open_zarr("/ceph/hpc/home/dhinakarans/data/autoencoder/ERA5_to_latent.zarr")
    ds = ds.drop_vars(['ssrd', 'tp'])
    variables = list(ds.data_vars.keys())
    
    data = np.stack([ds[var].values for var in variables], axis=-1)
    means = data.mean(axis=(0,1,2), keepdims=True)
    stds = data.std(axis=(0,1,2), keepdims=True)
    data = (data - means) / (stds + 1e-8)
    
    # --- Training Setup ---
    logger.info("Setting up training...")
    dataset = XarrayWeatherDataset(
        data,
        patch_size=PATCH_SIZE,
        time_steps=TIME_STEPS,
        num_tiles_per_time=500
    )
    
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True, num_workers=8)
    
    # --- Multi-GPU Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    input_dim = PATCH_SIZE * PATCH_SIZE * TIME_STEPS * len(variables)
    model = WeatherAutoencoder(input_dim=input_dim, latent_dim=LATENT_DIM)
    
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # --- Training Loop ---
    best_loss = float('inf')
    patience = 25
    counter = 0
    
    logger.info("Starting training...")
    for epoch in range(250):
        model.train()
        total_loss = 0
        start_time = datetime.now()
        
        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(device)
            recon, _ = model(batch)
            loss = loss_fn(recon, batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                logger.info(f"Epoch {epoch+1} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")
        
        epoch_loss = total_loss/len(dataloader)
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
    logger.info("Starting inference...")
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    full_dataset = FullXarrayWeatherDataset(data, patch_size=PATCH_SIZE, time_steps=TIME_STEPS)
    full_loader = DataLoader(full_dataset, batch_size=1024, shuffle=False, num_workers=8)
    
    all_latents = []
    with torch.no_grad():
        for batch in full_loader:
            batch = batch.to(device)
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