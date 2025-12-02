import torch
import torch.nn as nn
import numpy as np

# ==========================================
# 1. Physical Resolution Layer
# ==========================================
class PhysicalScaleLayer(nn.Module):
    def __init__(self, spatial_domain, spatial_min_wavelength, 
                 temporal_domain, temporal_min_wavelength):
        super().__init__()
        
        # 1. Calculate Spatial Frequency Factor
        # How many 400km waves fit in 8000km? (20 waves)
        # We need the sine to oscillate 20 times across x=[-1, 1]
        self.s_factor = np.pi * (spatial_domain / spatial_min_wavelength)
        
        # 2. Calculate Temporal Frequency Factor
        self.t_factor = np.pi * (temporal_domain / temporal_min_wavelength)
        
        print(f"--- Model Resolution Limits ---")
        print(f"Space: {spatial_domain}km domain / {spatial_min_wavelength}km resolution")
        print(f"       -> {spatial_domain/spatial_min_wavelength:.1f} Cycles")
        print(f"       -> Scaling Factor: {self.s_factor:.4f}")
        
        print(f"Time:  {temporal_domain}min domain / {temporal_min_wavelength}min resolution")
        print(f"       -> {temporal_domain/temporal_min_wavelength:.1f} Cycles")
        print(f"       -> Scaling Factor: {self.t_factor:.4f}")

    def forward(self, coords):
        # coords: (x, y, t) in range [-1, 1]
        
        # Apply the factors directly. 
        # This converts normalized coords into "Phase Angle"
        scaled = coords.clone()
        scaled[:, 0] *= self.s_factor # x
        scaled[:, 1] *= self.s_factor # y (assuming isotropic space)
        scaled[:, 2] *= self.t_factor # t
        
        return scaled

# ==========================================
# 2. The Clean SIREN
# ==========================================
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        
        # Initialize weights
        with torch.no_grad():
            limit = np.sqrt(6 / in_features) 
            self.linear.weight.uniform_(-limit, limit)

    def forward(self, input):
        # omega_0 is GONE. It is implicitly 1.0.
        return torch.sin(self.linear(input))

class ResolutionConstrainedINR(nn.Module):
    def __init__(self, 
                 spatial_domain, spatial_res, 
                 temporal_domain, temporal_res,
                 hidden_features=256, hidden_layers=3):
        super().__init__()
        
        # 1. The Scaling Layer (The Physics)
        self.scaler = PhysicalScaleLayer(spatial_domain, spatial_res, 
                                       temporal_domain, temporal_res)
        
        # 2. The Network
        layers = []
        # First layer sees the physically scaled phase angles
        # Note: We use sin(x) directly on the scaled input
        layers.append(SineLayer(3, hidden_features, is_first=True))
        
        for _ in range(hidden_layers):
            layers.append(SineLayer(hidden_features, hidden_features))
            
        self.net = nn.Sequential(*layers)
        self.final_linear = nn.Linear(hidden_features, 1)
        
    def forward(self, coords):
        # 1. Scale inputs to match resolution constraints
        phase_coords = self.scaler(coords)
        
        # 2. Pass through Sine Network
        return self.final_linear(self.net(phase_coords))


#%%

from os.path import join as pjoin
from icreader import ConductanceImage
from copy import deepcopy as dcopy

base = '/home/bing/Dropbox/work/code/repos/icBuilder/example_data/'
orbit_file = 'or_0099.nc'
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

# Stack inputs: (N_samples, 3)
train_coords = np.stack([x_flat, y_flat, t_flat], axis=1)
train_values = z_flat[:, None]

# Convert to Torch
coords_tensor = torch.FloatTensor(train_coords)
values_tensor = torch.FloatTensor(train_values)

# ==========================================
# 3. Usage
# ==========================================

# Define your constraints
model = ResolutionConstrainedINR(
    spatial_domain=8000,    # km
    spatial_res=2*250,        # km (Wavelength) (2*resolution)
    temporal_domain=308,    # min
    temporal_res=2*3,         # min (Wavelength) (2*resolution)
    hidden_features=256,
    hidden_layers=3
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

print("\nTraining Check...")
losses = []
for i in range(30000):
    optimizer.zero_grad()
    preds = model(coords_tensor)
    loss = nn.MSELoss()(preds, values_tensor)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if i % 200 == 0: print(f"Iter {i}: {loss.item()}")


#%%

import matplotlib.pyplot as plt

res = 36
x_new = np.linspace(x.min(), x.max(), res)
y_new = np.linspace(y.min(), y.max(), res)
Y_new, X_new = np.meshgrid(y_new, x_new)

start = 100
ncol = 10
fig, axes = plt.subplots(2, ncol, figsize=(ncol*16, 2*16))
for i in range(ncol):
    ii = start + i
    input_slice = Z_sparse[ii]
    vmax = np.nanmax(abs(input_slice))
    masked_view = np.ma.array(input_slice, mask=np.isnan(input_slice))
    axes[0, i].imshow(masked_view, cmap='bwr', vmin=-vmax, vmax=vmax)
    
    T_new = np.ones_like(X_new)*t[ii]
    flat_coords = np.stack([X_new.flatten(), Y_new.flatten(), T_new.flatten()], axis=1)
    input_tensor = torch.FloatTensor(flat_coords).requires_grad_(True)
    #pred_values, coords_grad = model(input_tensor)
    pred_values = model(input_tensor)
    pred_img = pred_values.detach().numpy().reshape(res, res)    
    masked_view = np.ma.array(pred_img.T, mask=np.isnan(input_slice))
    axes[1, i].imshow(masked_view, cmap='bwr', vmin=-vmax, vmax=vmax)

for ax in axes.flatten():
    ax.set_yticks([])
    ax.set_xticks([])

plt.tight_layout()
plt.show()