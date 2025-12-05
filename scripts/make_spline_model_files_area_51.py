#%% Import

from os.path import join as pjoin
from icreader import ConductanceImage
from icbuilder import SplineImage
import matplotlib.pyplot as plt

#%% Paths

base = '/home/bing/Dropbox/work/code/repos/icBuilder/example_data/'
orbit = 209
orbit_file = f'or_{str(orbit).zfill(4)}.nc'
conductance_file = pjoin(base, 'conductance', orbit_file)
spline_file = pjoin(base, 'spline', orbit_file)

#%% Load condutance image

cI = ConductanceImage(conductance_file)

#%% Make spline image

#sI = SplineImage(cI, ncp = 30, cpt_step = 5, lH = -6, lP=-1, wscaling=True, kt=2)
sI = SplineImage(cI, ncp = 30, cpt_step = 5, lH = -3.5, lP=-1, wscaling=True, kt=2)

#%%

idx = 100
fig, axs = plt.subplots(6, 8, figsize=(16,12))
for i in range(8):
    axs[0, i].imshow(sI.H[idx+i], vmin=0, vmax=75)
    axs[1, i].imshow(sI.pH[idx+i], vmin=0, vmax=75)
    axs[2, i].imshow(sI.dH[idx+i], vmin=0, vmax=75)
    axs[3, i].imshow(sI.pdH[idx+i], vmin=0, vmax=75)
    axs[4, i].imshow(sI.pdH_m[idx+i], vmin=0, vmax=75)
    axs[5, i].imshow(sI.LTL_diag.reshape((sI.ncp, sI.ncp, sI.ncpt))[:, :, int(sI.tknots.size * (idx+i)/sI.nt)].T, vmin=0, vmax=1)    


#%%

import numpy as np
from secsy import CSgrid, CSprojection
from scipy.sparse import kron, vstack, csc_matrix

position = (0, 90) # lon, lat
orientation = (0, 1) # east, north
L, Lres = 20000e3, 50e3
grid = CSgrid(CSprojection(position, orientation), L, L, Lres, Lres, R = 6481.2e3)

t = np.arange(sI.t.min(), sI.t.max()+12, 12)

G = np.zeros((grid.xi.size, sI.ncp**2))
for i, (xi, yi) in enumerate(zip(grid.xi.flatten(), grid.eta.flatten())):
    G[i, :] = sI.splx(xi).dot(np.kron(np.eye(sI.ncp), sI.sply(yi)))
G = csc_matrix(G)
    
pH = np.zeros((t.size, grid.shape[0], grid.shape[0]))
pdH = np.zeros((t.size, grid.shape[0], grid.shape[0]))
for i, ti in enumerate(t):
    Gt = G @ kron(np.eye(G.shape[1]), sI.splt(ti), format='csr')
    pH[i] = (Gt@sI.mH).reshape((grid.shape[0], grid.shape[0]))
    pdH[i] = (Gt@sI.mdH).reshape((grid.shape[0], grid.shape[0]))

pH[pH<0] = 0
pdH[pdH<0] = 0

#%%

for i, ti in enumerate(t):
    fig, axs = plt.subplots(2, 2, figsize=(20,20), sharex=True, sharey=True)
    
    t_idx = np.argmin(abs(sI.t - ti))
    
    f = np.isnan(sI.H[t_idx])
    axs[0, 0].tricontourf(sI.x[~f], sI.y[~f], sI.H[t_idx][~f], levels=np.linspace(0, 75, 40))
    f = np.isnan(sI.dH[t_idx])
    axs[1, 0].tricontourf(sI.x[~f], sI.y[~f], sI.dH[t_idx][~f], levels=np.linspace(0, 75, 40))
    axs[0, 1].tricontourf(grid.xi.flatten(), grid.eta.flatten(), pH[i].flatten(), levels=np.linspace(0, 75, 40))
    axs[1, 1].tricontourf(grid.xi.flatten(), grid.eta.flatten(), pdH[i].flatten(), levels=np.linspace(0, 30, 40), extend='both')
    
    for ax, tit in zip(axs.flatten(), ['Hall', 'Hall model', 'Uncertainty', 'Uncertainty model']):
        ax.set_title(tit, fontsize=20)
        ax.set_axis_off()
    
    plt.suptitle(f't {ti}', fontsize=30)
    plt.savefig(f'/home/bing/Dropbox/work/temp_storage/spline_comp/{str(int(ti)).zfill(5)}.png', bbox_inches='tight')
    plt.close('all')

#%%

import matplotlib
matplotlib.use('Agg') # Must be before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from joblib import Parallel, delayed

# --- 1. Pre-calculate Triangulation (Right Column) ---
tri_grid = mtri.Triangulation(grid.xi.flatten(), grid.eta.flatten())

# --- 2. Extract Data Arrays ---
# We extract these here so we don't pass the 'sI' object (which contains the un-pickleable solver)
sI_t = sI.t
sI_x = sI.x
sI_y = sI.y
sI_H = sI.H
sI_dH = sI.dH

# --- 3. Define Rendering Function ---
# Notice we now accept specific arrays instead of the 'sI' object
def render_frame(i, ti, current_t, x, y, H_all, dH_all, pH_frame, pdH_frame, tri_obj):
    try:
        fig, axs = plt.subplots(2, 2, figsize=(20, 20), sharex=True, sharey=True)
        
        # Calculate index locally
        t_idx = np.argmin(abs(current_t - ti))

        # Get specific time slice data
        H_data = H_all[t_idx]
        dH_data = dH_all[t_idx]

        # --- Top Left ---
        f = np.isnan(H_data)
        if not np.all(f):
            axs[0, 0].tricontourf(x[~f], y[~f], H_data[~f], levels=np.linspace(0, 75, 40))

        # --- Bottom Left ---
        f = np.isnan(dH_data)
        if not np.all(f):
            axs[1, 0].tricontourf(x[~f], y[~f], dH_data[~f], levels=np.linspace(0, 75, 40))

        # --- Right Column (using pre-calc triangulation) ---
        axs[0, 1].tricontourf(tri_obj, pH_frame.flatten(), levels=np.linspace(0, 75, 40))
        axs[1, 1].tricontourf(tri_obj, pdH_frame.flatten(), levels=np.linspace(0, 30, 40), extend='both')

        for ax, tit in zip(axs.flatten(), ['Hall', 'Hall model', 'Uncertainty', 'Uncertainty model']):
            ax.set_title(tit, fontsize=20)
            ax.set_axis_off()

        plt.suptitle(f't {ti}', fontsize=30)
        
        # Save
        filename = f'/home/bing/Dropbox/work/temp_storage/spline_comp/{str(int(ti)).zfill(5)}.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.close(fig) # Crucial to prevent memory leaks in loop
        
    except Exception as e:
        print(f"Frame {i} failed: {e}")

# --- 4. Run Parallel ---
print("Starting parallel rendering...")

# We pass the arrays explicitly. 
# Joblib is smart enough to share the memory of large numpy arrays (sI_H, sI_dH) 
# rather than copying them, provided they are not modified.
Parallel(n_jobs=-1, verbose=5)(
    delayed(render_frame)(
        i, 
        ti, 
        sI_t,   # Pass time array
        sI_x,   # Pass x array
        sI_y,   # Pass y array
        sI_H,   # Pass H array (pointer/memmap)
        sI_dH,  # Pass dH array (pointer/memmap)
        pH[i],  # Pass specific frame for pH
        pdH[i], # Pass specific frame for pdH
        tri_grid
    ) 
    for i, ti in enumerate(t)
)

#%%%

import numpy as np

ncp = [20, 25, 30, 35, 40, 45, 50]
cpt_step = [1, 2, 3, 4, 5]
GCV = np.zeros((len(cpt_step), len(ncp)))
for i, cpt_step_ in enumerate(cpt_step):
    for j, ncp_ in enumerate(ncp):
        print(f'{cpt_step_} : {ncp_}')
        sI = SplineImage(cI, ncp=ncp_, cpt_step=cpt_step_, lH=-10, lP=-1, wscaling=True, kt=2)
        f = np.isnan(sI.H)
        r = (sI.H - sI.pH)[~f]
        RSS = np.sum(r*sI.pdH[~f]*r)
        GCV[i, j] = RSS/(np.sum(f) * (1 - sI.Gt.shape[1]/np.sum(f))**2)

#%%

q = np.log10(GCV)

vmin = np.min(q)
vmax = np.max(q)
fig = plt.figure()
plt.imshow(q, vmin=vmin, vmax=vmax)
plt.colorbar()

#%%

sI.to_nc(spline_file)



#%%

from scipy.sparse.linalg import svds

U, s, Vh = svds(sI.Gt, k=sI.Gt.shape[1]-1)


#%%

idx = 100
fig, axs = plt.subplots(2, 8, figsize=(16,4))
for i in range(8):
    axs[0, i].imshow(cI.dH[idx+i], vmin=0, vmax=75)
    axs[1, i].imshow(cI.w[idx+i], vmin=0, vmax=1)

#%% 

sI = SplineImage(cI, ncp = 40, cpt_step = 2, lH = -3, lP=-1, wscaling=True, kt=3)

#%%

idx = 100
fig, axs = plt.subplots(2, 8, figsize=(16,4))
for i in range(8):
    axs[0, i].imshow(cI.H[idx+i], vmin=0, vmax=75)
    axs[1, i].imshow(sI.pH[idx+i], vmin=0, vmax=75)

#%%

idx = 200 - 50
idx = 100
fig, axs = plt.subplots(6, 8, figsize=(16,12))
for i in range(8):
    axs[0, i].imshow(sI.H[idx+i], vmin=0, vmax=75)
    axs[1, i].imshow(sI.pH[idx+i], vmin=0, vmax=75)
    axs[2, i].imshow(sI.dH[idx+i], vmin=0, vmax=75)
    axs[3, i].imshow(sI.pdH[idx+i], vmin=0, vmax=75)
    #axs[4, i].imshow(sI.pdH_m[idx+i], vmin=0, vmax=10)
    #axs[5, i].imshow(sI.LTL_diag.reshape((sI.ncp, sI.ncp, sI.ncpt))[:, :, int(sI.tknots.size * (idx+i)/sI.nt)].T, vmin=0, vmax=1)    

#%%

idx = 5
fig, axs = plt.subplots(4, 8, figsize=(16,8))
for i in range(8):
    axs[0, i].imshow(sI.dH[idx+i], vmin=0, vmax=75)
    axs[1, i].imshow(sI.pdH[idx+i], vmin=0, vmax=10)
    axs[2, i].imshow(sI.pdH_m[idx+i], vmin=0, vmax=10)
    axs[3, i].imshow(sI.pdH[idx+i] - sI.pdH_m[idx+i], vmin=-1, vmax=1, cmap='bwr')

#%% Create spline image

#sI = SplineImage(cI)

#%% Save spline image to nc

#sI.to_nc(spline_file)


