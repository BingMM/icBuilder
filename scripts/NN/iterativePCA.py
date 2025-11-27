import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ==========================================
# 1. Synthetic Data Generation
# ==========================================
def generate_data(n_time=50, size=32, missing_rate=0.4):
    """
    Generates a moving Gaussian blob to simulate temporal evolution.
    """
    x = np.linspace(-2, 2, size)
    y = np.linspace(-2, 2, size)
    X, Y = np.meshgrid(x, y)
    
    data = []
    # Create a blob that moves in a circle over time
    for t in np.linspace(0, 2*np.pi, n_time):
        cx = 1.0 * np.cos(t)
        cy = 1.0 * np.sin(t)
        # Gaussian shape
        z = np.exp(-((X-cx)**2 + (Y-cy)**2) / 0.5) 
        # Add some complex "texture" (high frequency)
        z += 0.1 * np.sin(5*X + t) * np.cos(5*Y)
        data.append(z)
        
    data = np.array(data) # (Time, H, W)
    
    # Create Mask
    mask = np.random.rand(*data.shape) > missing_rate
    data_with_nans = data.copy()
    data_with_nans[~mask] = np.nan
    
    return data, data_with_nans, mask

# Generate data
true_data, data_nan, valid_mask = generate_data()
print(f"Data Shape: {true_data.shape}")
print(f"Missing Data: {np.isnan(data_nan).mean()*100:.1f}%")

# ==========================================
# Model 1: Iterative PCA (PPCA Equivalent)
# ==========================================
class IterativePCA:
    """
    Performs PCA on data with NaNs by iteratively imputing 
    the missing values with the current PCA reconstruction.
    """
    def __init__(self, n_components=10, max_iter=50, tol=1e-4):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.pca = None
        self.mean = None
        self.components_ = None

    def fit_transform(self, X_3d):
        # Flatten: (Time, Pixels)
        n_samples, h, w = X_3d.shape
        X = X_3d.reshape(n_samples, -1)
        
        # Initial imputation: fill NaNs with column means
        self.mean = np.nanmean(X, axis=0)
        X_filled = np.where(np.isnan(X), self.mean, X)
        
        prev_recon = np.zeros_like(X_filled)
        
        print("Training Iterative PCA...")
        for i in range(self.max_iter):
            # fit standard PCA on filled data
            self.pca = PCA(n_components=self.n_components)
            W = self.pca.fit_transform(X_filled) # (Time, Components)
            V = self.pca.components_         # (Components, Pixels)
            
            # Reconstruct
            X_recon = np.dot(W, V) + self.mean
            
            # Check convergence
            diff = np.mean((X_recon - prev_recon)**2)
            if diff < self.tol:
                break
            prev_recon = X_recon
            
            # Re-impute: Put reconstructed values ONLY where NaNs were
            X_filled[np.isnan(X)] = X_recon[np.isnan(X)]
            
        self.components_ = self.pca.components_.reshape(self.n_components, h, w)
        self.temporal_coeffs = W
        return X_recon.reshape(n_samples, h, w)

# ==========================================
# Model 2: Masked Autoencoder (The VAE sibling)
# ==========================================
class MaskedAE(nn.Module):
    def __init__(self, input_dim, latent_dim=10):
        super().__init__()
        # Simple MLP Encoder/Decoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
        )
        
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

def train_mae(X_3d, mask, latent_dim=10, epochs=500):
    n_samples, h, w = X_3d.shape
    input_dim = h * w
    
    # Prepare data: Replace NaN with 0 for input, but keep mask for loss
    X_flat = X_3d.reshape(n_samples, -1)
    X_in = np.nan_to_num(X_flat) 
    
    tensor_x = torch.FloatTensor(X_in)
    tensor_mask = torch.FloatTensor(mask.reshape(n_samples, -1))
    
    model = MaskedAE(input_dim, latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    print("Training MAE...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        recon = model(tensor_x)
        
        # CRITICAL: Masked MSE Loss
        # Only calculate error where we actually have data
        loss = torch.sum(tensor_mask * (recon - tensor_x)**2) / torch.sum(tensor_mask)
        
        loss.backward()
        optimizer.step()
        
    return model, tensor_x

# ==========================================
# Model 3: Implicit Neural Representation (INR)
# ==========================================
class SirenLayer(nn.Module):
    """
    SIREN layer: standard linear layer with Sine activation.
    Great for fitting derivatives and signals.
    """
    def __init__(self, in_features, out_features, is_first=False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.is_first = is_first
        self.init_weights()
        
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / 32, 1 / 32)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / 32) / 32, np.sqrt(6 / 32) / 32)
                
    def forward(self, x):
        return torch.sin(30 * self.linear(x))

class INR(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: (t, x, y) -> Output: Value
        self.net = nn.Sequential(
            SirenLayer(3, 64, is_first=True),
            SirenLayer(64, 64),
            SirenLayer(64, 64),
            nn.Linear(64, 1) # Output is linear (intensity)
        )
        
    def forward(self, coords):
        return self.net(coords)

def train_inr(X_3d, mask, epochs=500):
    # Create coordinate grid (T, H, W)
    T, H, W = X_3d.shape
    t_coords = np.linspace(-1, 1, T)
    y_coords = np.linspace(-1, 1, H)
    x_coords = np.linspace(-1, 1, W)
    
    # Meshgrid of coordinates
    grid_t, grid_y, grid_x = np.meshgrid(t_coords, y_coords, x_coords, indexing='ij')
    
    # Flatten everything
    coords = np.stack([grid_t.flatten(), grid_y.flatten(), grid_x.flatten()], axis=-1)
    values = X_3d.flatten()
    valid_mask_flat = mask.flatten()
    
    # FILTER: We only train on VALID pixels. 
    # NaNs are completely excluded from the dataset.
    train_coords = coords[valid_mask_flat]
    train_values = values[valid_mask_flat]
    
    dataset = TensorDataset(torch.FloatTensor(train_coords), torch.FloatTensor(train_values).unsqueeze(1))
    loader = DataLoader(dataset, batch_size=4096, shuffle=True)
    
    model = INR()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    print("Training INR...")
    for epoch in range(epochs):
        for batch_coords, batch_vals in loader:
            optimizer.zero_grad()
            preds = model(batch_coords)
            loss = nn.MSELoss()(preds, batch_vals)
            loss.backward()
            optimizer.step()
            
    # Inference on full grid (to fill gaps)
    with torch.no_grad():
        full_coords = torch.FloatTensor(coords)
        # Process in chunks to avoid OOM
        full_recon = []
        chunk_size = 10000
        for i in range(0, len(full_coords), chunk_size):
            chunk = full_coords[i:i+chunk_size]
            full_recon.append(model(chunk))
        full_recon = torch.cat(full_recon).numpy().reshape(T, H, W)
        
    return full_recon

# ==========================================
# Run Comparison
# ==========================================

# 1. Run PPCA
pca_solver = IterativePCA(n_components=10)
recon_pca = pca_solver.fit_transform(data_nan)

# 2. Run MAE
mae_model, mae_input = train_mae(data_nan, valid_mask, latent_dim=10)
with torch.no_grad():
    recon_mae = mae_model(mae_input).numpy().reshape(true_data.shape)

# 3. Run INR
recon_inr = train_inr(data_nan, valid_mask)

# ==========================================
# Visualization
# ==========================================

# Select a specific time slice to visualize
t_idx = 25 

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
plt.subplots_adjust(hspace=0.3)

# Row 1: Inputs and Truth
vmin, vmax = true_data.min(), true_data.max()

ax = axes[0,0]
ax.imshow(true_data[t_idx], vmin=vmin, vmax=vmax, cmap='viridis')
ax.set_title("Ground Truth")

ax = axes[0,1]
# Create a masked array for correct visualization of NaNs
masked_view = np.ma.array(data_nan[t_idx], mask=~valid_mask[t_idx])
ax.imshow(masked_view, vmin=vmin, vmax=vmax, cmap='viridis')
ax.set_facecolor('black') # Make NaNs black
ax.set_title(f"Input (NaNs shown as black)\nTime: {t_idx}")

ax = axes[0,2]
ax.axis('off') # Empty slot

# Row 2: Reconstructions
ax = axes[1,0]
ax.imshow(recon_pca[t_idx], vmin=vmin, vmax=vmax, cmap='viridis')
mse_pca = np.nanmean((recon_pca - true_data)**2)
ax.set_title(f"PPCA Reconstruction\nMSE: {mse_pca:.5f}")

ax = axes[1,1]
ax.imshow(recon_mae[t_idx], vmin=vmin, vmax=vmax, cmap='viridis')
mse_mae = np.nanmean((recon_mae - true_data)**2)
ax.set_title(f"MAE Reconstruction\nMSE: {mse_mae:.5f}")

ax = axes[1,2]
ax.imshow(recon_inr[t_idx], vmin=vmin, vmax=vmax, cmap='viridis')
mse_inr = np.nanmean((recon_inr - true_data)**2)
ax.set_title(f"INR Reconstruction\nMSE: {mse_inr:.5f}")

plt.show()

# Plot Basis Functions (Eigenimages) for PPCA
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
fig.suptitle("First 5 Basis Functions (PPCA)")
for i in range(5):
    axes[i].imshow(pca_solver.components_[i], cmap='RdBu_r')
    axes[i].axis('off')
plt.show()