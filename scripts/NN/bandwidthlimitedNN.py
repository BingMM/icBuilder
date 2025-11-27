import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. Deterministic Band-Limited Encoder
# ==========================================
class BandLimitedEncoder(nn.Module):
    def __init__(self, domain_size, min_wavelength):
        super().__init__()
        
        # 1. Calculate the Fundamental Frequency (Base wave that fits domain once)
        # f_base = 1 / domain_size
        
        # 2. Calculate the Nyquist Frequency (The limit)
        # f_max = 1 / min_wavelength
        
        # 3. Create the spectrum of frequencies
        # We create a list of integer harmonics: 1, 2, 3 ... N
        # N is determined strictly by how many times min_wavelength fits in domain
        n_bands = int(domain_size / min_wavelength)
        
        # Create frequencies: [1, 2, ..., n_bands]
        # We perform the mapping in "Normalized Phase Space" (0 to 1) for calculation 
        # but scaling happens relative to the physical domain.
        
        print(f"--- Spectral Constraints ({domain_size} / {min_wavelength}) ---")
        print(f"    Max Harmonic Band (N): {n_bands}")
        
        # Generate the bank of frequencies
        # Shape: (n_bands, )
        freqs = torch.linspace(1.0, float(n_bands), n_bands)
        
        # We register this as a buffer (part of state_dict, but NOT a trainable parameter)
        self.register_buffer('freqs', freqs)
        
    def forward(self, x):
        # x is assumed to be normalized to [0, 1] relative to the Domain Size
        # If x goes from 0 to 1, sin(2*pi*1*x) is one cycle.
        
        # Outer product to create all frequency combinations
        # x shape: (Batch, 1)
        # freqs shape: (n_bands, )
        # result: (Batch, n_bands)
        
        # Argument = x * freq * 2pi
        x_expanded = x @ self.freqs.unsqueeze(0) 
        x_phase = 2 * np.pi * x_expanded
        
        # Return both Sin and Cos to capture Phase
        return torch.cat([torch.sin(x_phase), torch.cos(x_phase)], dim=-1)

# ==========================================
# 2. The Model Architecture
# ==========================================
class SpectralINR(nn.Module):
    def __init__(self, 
                 spatial_domain, spatial_min_lambda,
                 temporal_domain, temporal_min_lambda,
                 hidden_dim=256, hidden_layers=3):
        super().__init__()
        
        # --- 1. Deterministic Encoders ---
        # These replace the first "SineLayer" of SIREN.
        # They are fixed basis transforms.
        self.enc_x = BandLimitedEncoder(spatial_domain, spatial_min_lambda)
        self.enc_y = BandLimitedEncoder(spatial_domain, spatial_min_lambda) # Assuming isotropic
        self.enc_t = BandLimitedEncoder(temporal_domain, temporal_min_lambda)
        
        # Calculate input dimension for the MLP
        # Each encoder outputs 2 * n_bands features
        input_dim = (len(self.enc_x.freqs) * 2) + \
                    (len(self.enc_y.freqs) * 2) + \
                    (len(self.enc_t.freqs) * 2)
                    
        print(f"    Total Basis Functions: {input_dim}")

        # --- 2. The Coefficient Solver (MLP) ---
        # This network learns how to MIX the fixed frequencies.
        # It uses standard ReLU or Tanh activations. 
        # Tanh is preferred if you need smooth 2nd derivatives.
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh()) # Smooth activation
        
        for _ in range(hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
            
        self.net = nn.Sequential(*layers)
        
        # --- 3. Positivity Constraint ---
        self.final_output = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softplus() # Strictly Positive
        )
        
    def forward(self, coords):
        # coords: (x, y, t) normalized to [0, 1]
        
        x = coords[:, 0:1]
        y = coords[:, 1:2]
        t = coords[:, 2:3]
        
        # Encode each dimension separately
        # This enforces the "Separable" resolution constraints you asked for
        # (i.e. different resolution for time vs space)
        feat_x = self.enc_x(x)
        feat_y = self.enc_y(y)
        feat_t = self.enc_t(t)
        
        # Concatenate all basis features
        features = torch.cat([feat_x, feat_y, feat_t], dim=1)
        
        # Solve
        return self.final_output(self.net(features))

# ==========================================
# 3. Verification of Resolution Limit
# ==========================================

# Define constraints
S_DOMAIN = 8000
S_RES = 400     # Max 20 cycles
T_DOMAIN = 38
T_RES = 30       # Max 125 cycles

model = SpectralINR(S_DOMAIN, S_RES, T_DOMAIN, T_RES)

# --- TEST ---
# We will feed the model a frequency that is HIGHER than the limit.
# Since the encoder does not contain that frequency, the model 
# should strictly fail to reconstruct it (it will alias or smooth it out).

from os.path import join as pjoin
from icreader import ConductanceImage
from copy import deepcopy as dcopy

base = '/home/bing/Dropbox/work/code/repos/icBuilder/example_data/'
orbit_file = 'or_0085.nc'
conductance_file = pjoin(base, 'conductance', orbit_file)
spline_file = pjoin(base, 'spline', orbit_file)

# Load condutance image

cI = ConductanceImage(conductance_file)

# Get data
Z_true = dcopy(cI.H)
Z_sparse = dcopy(cI.H)

t = np.array([t.total_seconds() for t in (cI.time - cI.time[0])])
t -= t.mean()
t /= abs(t).max()
y = np.linspace(cI.grid.eta.min(), cI.grid.eta.max(), Z_true.shape[1])
y /= np.max(abs(y))
x = np.linspace(cI.grid.xi.min(), cI.grid.xi.max(), Z_true.shape[2])
x /= np.max(abs(x))
T_grid, Y_grid, X_grid = np.meshgrid(t, y, x, indexing='ij')
mask = ~np.isnan(Z_true)

# Prepare Training Data (Flatten and Filter NaNs)
# We effectively turn the 3D grid into a "Point Cloud" of valid data
valid_indices = ~np.isnan(Z_sparse.flatten())

t_flat = T_grid.flatten()[valid_indices]
y_flat = Y_grid.flatten()[valid_indices]
x_flat = X_grid.flatten()[valid_indices]
z_flat = Z_sparse.flatten()[valid_indices]

# Dummy y, t
coords = torch.FloatTensor(np.stack([x_flat, y_flat, t_flat], axis=1))
vals = torch.FloatTensor(z_flat[:, None])

# Train
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("\nTraining on Signal exceeding resolution limit...")
for i in range(5000):
    optimizer.zero_grad()
    preds = model(coords)
    loss = nn.MSELoss()(preds, vals)
    loss.backward()
    optimizer.step()
    if i % 200 == 0: print(f"Iter {i} Loss: {loss.item():.5f}")

#%%

cp = torch.FloatTensor(np.stack([X_grid.flatten(), Y_grid.flatten(), T_grid.flatten()], axis=1))

# Plot
with torch.no_grad():
    pred_y = model(cp).numpy().flatten().reshape(Z_true.shape)

ncol = 10
fig, axes = plt.subplots(2, ncol, figsize=(ncol*16, 2*16))
for i in range(ncol):
    vmax = np.nanmax(abs(Z_true[i]))
    masked_view = np.ma.array(Z_true[i], mask=np.isnan(Z_true[i]))
    axes[0, i].imshow(masked_view, cmap='bwr', vmin=-vmax, vmax=vmax)
        
    masked_view = np.ma.array(pred_y[i], mask=np.isnan(Z_true[i]))
    axes[1, i].imshow(masked_view, cmap='bwr', vmin=-vmax, vmax=vmax)

for ax in axes.flatten():
    ax.set_yticks([])
    ax.set_xticks([])

plt.tight_layout()
plt.show()
