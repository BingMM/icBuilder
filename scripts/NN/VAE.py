#%% Import

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from os.path import join as pjoin
from icreader import ConductanceImage
import matplotlib.pyplot as plt
from copy import deepcopy as dcopy

#%% Paths

base = '/home/bing/Dropbox/work/code/repos/icBuilder/example_data/'
orbit_file = 'or_0085.nc'
conductance_file = pjoin(base, 'conductance', orbit_file)
spline_file = pjoin(base, 'spline', orbit_file)

#%% Load condutance image

cI = ConductanceImage(conductance_file)

#%% Get data

X = dcopy(cI.H)

n_images, height, width = X.shape

#%%

print(f"Data shape: {X.shape}")
print(f"Percentage of NaNs: {np.isnan(X).mean()*100:.1f}%")

# ============================================
# 1. VAE Approach
# ============================================

class MaskedVAE(nn.Module):
    def __init__(self, image_size=36*36, latent_dim=20):
        super().__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(image_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_var = nn.Linear(128, latent_dim)
        
        # Decoder (this generates your basis functions)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, image_size),
        )
        
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var
    
    def get_basis_functions(self):
        """Extract basis functions from decoder"""
        # Each basis function is the image generated when one latent 
        # dimension is 1 and others are 0
        basis = []
        with torch.no_grad():
            for i in range(self.latent_dim):
                z = torch.zeros(1, self.latent_dim)
                z[0, i] = 1.0
                basis_func = self.decode(z).reshape(36, 36).numpy()
                basis.append(basis_func)
        return np.array(basis)

def train_vae(X, n_epochs=100, latent_dim=20):
    # Prepare data
    X_flat = X.reshape(n_images, -1)
    mask = ~np.isnan(X_flat)
    X_tensor = torch.FloatTensor(np.nan_to_num(X_flat))  # Replace NaN with 0 for computation
    mask_tensor = torch.FloatTensor(mask.astype(float))
    
    model = MaskedVAE(latent_dim=latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        recon, mu, log_var = model(X_tensor)
        
        # Reconstruction loss only on observed pixels
        recon_loss = torch.sum(mask_tensor * (recon - X_tensor)**2) / torch.sum(mask_tensor)
        
        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        
        loss = recon_loss + 0.01 * kl_loss  # Weight KL term down
        
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"VAE Epoch {epoch}: Loss = {loss.item():.4f}")
    
    return model

# Train VAE
vae_model = train_vae(X, n_epochs=10000, latent_dim=10)
vae_basis = vae_model.get_basis_functions()

print(f"\nVAE basis functions shape: {vae_basis.shape}")

# ============================================
# Visualize basis functions
# ============================================

# Plot first 5 VAE basis functions
fig, axes = plt.subplots(4, 10, figsize=(12, 8))
fig.suptitle('VAE Basis Functions')
for i, ax in enumerate(axes.flatten()):
    ax.imshow(vae_basis[i], cmap='RdBu_r')
    ax.set_title(f'Basis {i+1}')
    ax.axis('off')

plt.tight_layout()
plt.show()

# Plot example original images with NaNs
fig, axes = plt.subplots(2, 5, figsize=(12, 8))
fig.suptitle('Images')
for i, ax in enumerate(axes.flatten()):
    ax.imshow(X[i], cmap='RdBu_r')
    ax.set_title(f'Data {i}')
    ax.axis('off')

plt.tight_layout()
plt.show()

# ============================================
# How to use these basis functions
# ============================================

print("\n" + "="*50)
print("How to use these basis functions:")
print("="*50)

print("\n1. VAE approach:")
print("   - Encode new images: latent_codes = vae_model.encode(image)")
print("   - Basis functions are in decoder weights")
print("   - Reconstruction: weighted sum of basis functions")



X_flat = X.reshape(X.shape[0], -1)
mask = ~np.isnan(X_flat)
X_tensor = torch.FloatTensor(np.nan_to_num(X_flat))

# Get reconstructions and latent codes
with torch.no_grad():
    # Encode to get latent representations (these are your coefficients)
    mu, log_var = vae_model.encode(X_tensor)
    latent_codes = mu  # Use mean for deterministic reconstruction
    
    # Decode to reconstruct (this uses the basis functions internally)
    reconstructions = vae_model.decode(latent_codes)

# Convert back to numpy and reshape
reconstructions = reconstructions.numpy().reshape(-1, 36, 36)
latent_codes = latent_codes.numpy()

# Visualize specific examples
fig, axes = plt.subplots(3, 10, figsize=(10, 3))

for i in range(10):
    # Original image
    ax = axes[0, i]
    ax.imshow(X[i], cmap='RdBu_r')
    ax.axis('off')
    
    # Reconstruction
    ax = axes[1, i]
    ax.imshow(reconstructions[i], cmap='RdBu_r')
    ax.axis('off')
    
    # Difference (only where observed)
    diff = np.full_like(X[i], np.nan)
    observed = ~np.isnan(X[i])
    diff[observed] = X[i][observed] - reconstructions[i][observed]
    ax = axes[2, i]
    ax.imshow(diff, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.axis('off')


plt.tight_layout()
plt.show()

G = vae_basis.reshape(vae_basis.shape[0], 36*36).T
d = X[0].flatten()
f = np.isnan(d)
G_ = G[~f]
d_ = d[~f]
m = np.linalg.solve(G_.T@G_, G_.T@d_)
p = G@m
p = p.reshape((36,36))

fig, axs = plt.subplots(1, 3)
axs[0].imshow(X[0], vmin=-75, vmax=75, cmap='bwr')
axs[1].imshow(p, vmin=-75, vmax=75, cmap='bwr')
axs[2].imshow(reconstructions[0], vmin=-75, vmax=75, cmap='bwr')















































