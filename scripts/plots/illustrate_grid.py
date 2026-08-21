#%% Import 

import os
import numpy as np
from secsy import CSgrid, CSprojection
from datetime import datetime
import apexpy
from polplot import pp
import matplotlib.pyplot as plt
import icbuilder

#%% Path out

# Get path to root of icreader repo
base = os.path.dirname(os.path.abspath(icbuilder.__file__))
base = os.path.abspath(os.path.join(base, '..'))  # Move up to repo root

# Paths
fig_out = os.path.join(base, 'example_data', 'figures', 'icGrid.png')

#%% Create grid

position = (0, 90) # lon, lat
orientation = (0, 1) # east, north
L, Lres = 20000e3, 225e3
grid_w = CSgrid(CSprojection(position, orientation), L, L, Lres, Lres, R = 6481.2e3)

#%% Plot CS grid function

def plot_grid(pax, grid, s, col='k'):
    n = grid.xi_mesh.shape[1]
    for i in range(0,n,s):
        pax.plot(grid.lat_mesh[:, i], (grid.lon_mesh[:, i]/15)%24, color=col, linewidth=.5)
    
    n = grid.xi_mesh.shape[0]
    for i in range(0,n,s):
        pax.plot(grid.lat_mesh[i, :], (grid.lon_mesh[i, :]/15)%24, color=col, linewidth=.5)
    
    pax.plot(grid.lat_mesh[0, :], (grid.lon_mesh[0, :]/15)%24, color=col, linewidth=2)
    pax.plot(grid.lat_mesh[-1, :], (grid.lon_mesh[-1, :]/15)%24, color=col, linewidth=2)
    pax.plot(grid.lat_mesh[:, 0], (grid.lon_mesh[:, 0]/15)%24, color=col, linewidth=2)
    pax.plot(grid.lat_mesh[:, -1], (grid.lon_mesh[:, -1]/15)%24, color=col, linewidth=2)

#%% Plot and save grid

plt.ioff()
fig, ax = plt.subplots(1,1,figsize=(10,10))
pax = pp(ax, minlat=40)
date = datetime(2000, 5, 16, 2, 53)
apex = apexpy.Apex(date)
pax.coastlines(color='k', linewidth=1, mag=apex)
pax.coastlines(color='cyan', linewidth=.9, mag=apex)
plot_grid(pax, grid_w, s=1, col='tab:orange')
pax.plotgrid(color='k', linewidth=.1)
pax.plot(np.ones(1000)*40, np.linspace(0, 24, 1000), color='k')
pax.writeLTlabels(fontsize=18)
for lat in np.arange(40, 90, 10):
    pax.text(lat+2.5, 3, f'{lat}' + '$^{\circ}$', fontsize=16)
ax.set_title('CS grid in magnetic coords (2000/05/16)\nResolution 225 km', fontsize=20)
plt.savefig(fig_out, bbox_inches='tight')
plt.close('all')
plt.ion()