import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. The Architecture (SIREN)
# ==========================================
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.linear.in_features, 
                                             1 / self.linear.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.linear.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.linear.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class INRModel(nn.Module):
    def __init__(self, in_features=3, hidden_features=256, hidden_layers=3, out_features=1):
        super().__init__()
        
        layers = []
        # Input Layer (x, y, t) -> Hidden
        layers.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=30))
        
        # Hidden Layers
        for _ in range(hidden_layers):
            layers.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=30))
            
        # Output Layer (Hidden -> Conductance Value)
        # We use a Linear final layer (no Sine) to allow arbitrary scaling
        self.net = nn.Sequential(*layers)
        self.final_linear = nn.Linear(hidden_features, out_features)
        
    def forward(self, coords):
        # coords: (Batch, 3) -> (x, y, t)
        coords = coords.clone().detach().requires_grad_(True) # Enable gradient tracking w.r.t input
        output = self.final_linear(self.net(coords))
        return output, coords

# ==========================================
# 2. Data Generation (Sparse 3D Data)
# ==========================================

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
x = np.linspace(cI.grid.xi.min(), cI.grid.xi.max(), Z_true.shape[2])
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

print(f"Original Grid Size: {T_grid.size}")
print(f"Training on {coords_tensor.shape[0]} valid pixels (Point Cloud)")

# ==========================================
# 3. Training Loop
# ==========================================
model = INRModel(hidden_features=256, hidden_layers=2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
losses = []

print("Training INR (fitting the manifold)...")
for epoch in range(3000):
    optimizer.zero_grad()
    
    # Forward pass
    preds, _ = model(coords_tensor)
    
    # Simple MSE Loss
    loss = nn.MSELoss()(preds, values_tensor)
    
    loss.backward()
    optimizer.step()
    
    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

#%%

res = 36
x_new = np.linspace(x.min(), x.max(), res)
y_new = np.linspace(y.min(), y.max(), res)
Y_new, X_new = np.meshgrid(y_new, x_new)

ncol = 10
fig, axes = plt.subplots(2, ncol, figsize=(ncol*16, 2*16))
for i in range(ncol):
    input_slice = Z_sparse[i]
    vmax = np.nanmax(abs(input_slice))
    masked_view = np.ma.array(input_slice, mask=np.isnan(input_slice))
    axes[0, i].imshow(masked_view, cmap='bwr', vmin=-vmax, vmax=vmax)
    
    T_new = np.ones_like(X_new)*t[i]
    flat_coords = np.stack([X_new.flatten(), Y_new.flatten(), T_new.flatten()], axis=1)
    input_tensor = torch.FloatTensor(flat_coords).requires_grad_(True)
    pred_values, coords_grad = model(input_tensor)
    pred_img = pred_values.detach().numpy().reshape(res, res)    
    masked_view = np.ma.array(pred_img.T, mask=np.isnan(input_slice))
    axes[1, i].imshow(masked_view, cmap='bwr', vmin=-vmax, vmax=vmax)

for ax in axes.flatten():
    ax.set_yticks([])
    ax.set_xticks([])

plt.tight_layout()
plt.show()



