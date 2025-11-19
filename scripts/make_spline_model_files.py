#%% Import

from os.path import join as pjoin
from icreader import ConductanceImage
from icbuilder import SplineImage
import matplotlib.pyplot as plt

#%% Paths

base = '/home/bing/Dropbox/work/code/repos/icBuilder/example_data/'
orbit_file = 'or_0085.nc'
conductance_file = pjoin(base, 'conductance', orbit_file)
spline_file = pjoin(base, 'spline', orbit_file)

#%% Load condutance image

cI = ConductanceImage(conductance_file)

#%% Make spline image

sI = SplineImage(cI, ncp = 20, cpt_step = 5, lH = -3, lP=-1, wscaling=True, kt=2)


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


idx = 5
fig, axs = plt.subplots(2, 8, figsize=(16,4))
for i in range(8):
    axs[0, i].imshow(cI.dH[idx+i], vmin=0, vmax=75)
    axs[1, i].imshow(cI.w[idx+i], vmin=0, vmax=1)

#%% 

sI = SplineImage(cI, ncp = 30, cpt_step = 3, lH = -3, lP=-1, wscaling=True, kt=3)

#%%

#idx = 100
idx = 5
fig, axs = plt.subplots(2, 8, figsize=(16,4))
for i in range(8):
    axs[0, i].imshow(cI.H[idx+i], vmin=0, vmax=75)
    axs[1, i].imshow(sI.pH[idx+i], vmin=0, vmax=75)

#%%

#idx = 100
idx = 5
fig, axs = plt.subplots(6, 8, figsize=(16,12))
for i in range(8):
    axs[0, i].imshow(sI.H[idx+i], vmin=0, vmax=75)
    axs[1, i].imshow(sI.pH[idx+i], vmin=0, vmax=75)
    axs[2, i].imshow(sI.dH[idx+i], vmin=0, vmax=75)
    axs[3, i].imshow(sI.pdH[idx+i], vmin=0, vmax=10)
    axs[4, i].imshow(sI.pdH_m[idx+i], vmin=0, vmax=10)
    axs[5, i].imshow(sI.LTL_diag.reshape((sI.ncp, sI.ncp, sI.ncpt))[:, :, int(sI.tknots.size * (idx+i)/sI.nt)].T, vmin=0, vmax=1)    

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


