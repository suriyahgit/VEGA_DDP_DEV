import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# --- Constants ---
PATCH_SIZE = 4  # Changed from 5 to 4
TIME_STEPS = 6  # Changed from 5 to 6
LATENT_DIM = 16
NUM_TILES_PER_TIME = 5000
DEFAULT_BATCH_SIZE = 8048
DEFAULT_NUM_WORKERS = 8
EARLY_STOPPING_PATIENCE = 10
MIN_LR = 1e-6
DEFAULT_PREFETCH_FACTOR = 25
VALIDATION_SIZE = 500

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
        x = self.up2(x)             # [B,64,4,4]
        
        # Final output
        out = self.final(x)         # [B,54,4,4]
        
        return out.view(B, -1), z   # Flatten output, return latent

# --- Dataset Classes ---
class WeatherDataset(Dataset):
    def __init__(self, data: np.ndarray, patch_size: int = PATCH_SIZE,
                 time_steps: int = TIME_STEPS, num_samples: int = NUM_TILES_PER_TIME,
                 validation: bool = False):
        self.data = data
        self.patch_size = patch_size
        self.time_steps = time_steps
        self.num_vars = data.shape[-1]
        self.pad = patch_size // 2
        self.validation = validation
        
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
        
        # Generate indices
        self.indices = self._generate_indices(
            num_samples if not validation else VALIDATION_SIZE
        )
    
    def _generate_indices(self, num_samples: int) -> list:
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

# --- Synthetic Data Generation ---
def generate_toy_data(num_timesteps=100, height=10, width=10):
    """Creates synthetic weather data with seasonal patterns"""
    time = np.arange(num_timesteps)
    lat = np.linspace(0, 1, height)
    lon = np.linspace(0, 1, width)
    
    # Base patterns
    temperature = (
        15 + 10 * np.sin(2 * np.pi * time/24)  # Daily cycle
        + 5 * np.sin(2 * np.pi * time/365)      # Yearly cycle
        + 2 * lat[:, None]                      # North-south gradient
    )
    
    humidity = (
        0.5 + 0.3 * np.cos(2 * np.pi * time/24)
        - 0.1 * lat[:, None]
    )
    
    pressure = (
        1013 + 10 * np.cos(2 * np.pi * time/365)
        + 5 * (1 - lat[:, None])
    )
    
    # Combine into 4D array [time, lat, lon, vars]
    data = np.stack([
        np.tile(temperature[..., None], (1, 1, width)),
        np.tile(humidity[..., None], (1, 1, width)),
        np.tile(pressure[..., None], (1, 1, width))
    ], axis=-1).transpose(1, 2, 3, 0)  # [time, lat, lon, vars]
    
    return data

# --- Training Setup ---
def train_toy_model():
    # Generate data (small spatial domain for testing)
    toy_data = generate_toy_data(num_timesteps=100, height=10, width=10)
    print(f"Data shape: {toy_data.shape}")  # Should be (100, 10, 10, 3)
    
    # Create datasets
    train_dataset = WeatherDataset(toy_data, num_samples=1000)
    val_dataset = WeatherDataset(toy_data, validation=True)
    
    # Small batch size for testing
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Initialize model (3 variables)
    model = WeatherUNetImproved(num_vars=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    
    # --- Training Loop ---
    losses = []
    for epoch in range(10):  # Short training for testing
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            recon, _ = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                recon, _ = model(batch)
                val_loss += loss_fn(recon, batch).item()
        
        losses.append((epoch_loss/len(train_loader), val_loss/len(val_loader)))
        print(f"Epoch {epoch}: Train Loss = {losses[-1][0]:.4f}, Val Loss = {losses[-1][1]:.4f}")
    
    # --- Visualization ---
    plt.plot([l[0] for l in losses], label='Train')
    plt.plot([l[1] for l in losses], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.title('Toy Problem Training')
    plt.show()
    
    # Test reconstruction
    test_sample = train_dataset[0].unsqueeze(0)
    with torch.no_grad():
        recon, latent = model(test_sample)
    
    print(f"Original shape: {test_sample.shape}")
    print(f"Reconstructed shape: {recon.shape}")
    print(f"Latent shape: {latent.shape}")

if __name__ == "__main__":
    train_toy_model()