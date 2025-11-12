#%% Import

import os
from icreader import ConductanceImage
from icbuilder import SplineImage
import matplotlib.pyplot as plt

#%% Paths

base = '/home/bing/Dropbox/work/code/repos/icBuilder/example_data/'
orbit_file = 'or_0099.nc'
conductance_file = os.path.join(base, 'conductance', orbit_file)
spline_file = os.path.join(base, 'spline', orbit_file)

#%% Load condutance image

cI = ConductanceImage(conductance_file)

idx = 5
fig, axs = plt.subplots(2, 8, figsize=(16,4))
for i in range(8):
    axs[0, i].imshow(cI.dH[idx+i], vmin=0, vmax=75)
    axs[1, i].imshow(cI.w[idx+i], vmin=0, vmax=1)

#%% 

sI = SplineImage(cI, lH = -1, lP=-1, wscaling=False)

#%%


idx = 100
fig, axs = plt.subplots(2, 8, figsize=(16,4))
for i in range(8):
    axs[0, i].imshow(cI.H[idx+i], vmin=0, vmax=75)
    axs[1, i].imshow(sI.pH[idx+i], vmin=0, vmax=75)

#%%

idx = 100
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


