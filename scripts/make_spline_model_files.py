#%%





#%% Import 

import os
import numpy as np
import pandas as pd
import glob
from secsy import CSgrid, CSprojection
from scipy.io import netcdf
from datetime import datetime, timedelta
from scipy.interpolate import griddata
from copy import deepcopy as dcopy
from tqdm import tqdm
from scipy.io import netcdf_file
from scipy.interpolate import BSpline
import scipy
from scipy.sparse import kron, vstack, csc_matrix

#%%

os.chdir('/home/bing/Dropbox/work/code/repos/icAurora/')
#import imagesat_e0_eflux_estimates as conFun
#from robinson import ped,hall, peduncertainty, halluncertainty
from classes import conductanceImage, splineImage

#%% Fun

def get_orbit_index(filename, orbit):
    
    # Fetch boundary data from orbit
    bf = pd.read_hdf(filename, key='final', where='orbit=="{}"'.format(orbit)).groupby('date')

    # Condition 1: All 'isglobal' values must be True for a given date
    ind0 = bf['isglobal'].all()
    # Condition 2: A_mean > P_mean + P_std + A_std for all MLT in a given date
    ind1 = bf.apply(lambda g: (g['A_mean'] > (g['P_mean'] + g['P_std'] + g['A_std'])).all())
    # Condition 3: A_mean > S_mean + S_std + A_std for all MLT in a given date
    ind2 = bf.apply(lambda g: (g['A_mean'] > (g['S_mean'] + g['S_std'] + g['A_std'])).all())
    # Condition 4: 'count' > 12 for all MLT in a given date
    ind3 = bf['count'].apply(lambda g: (g > 12).all())
    
    # Combined
    ind = ind0 & ind1 & ind2 & ind3
    
    return ind
'''
def longest_true_streak_with_gaps(lst, x):
    max_streak = 0
    current_streak = 0
    gap_count = 0

    for value in lst:
        if value:  # If it's True, extend the current streak
            current_streak += 1 + gap_count
            gap_count = 0  # Reset the gap count
        else:  # If it's False, check if we can tolerate the gap
            if gap_count < x:
                gap_count += 1
            else:  # If the gap is too large, reset the streak
                max_streak = max(max_streak, current_streak)
                current_streak = 0
                gap_count = 0

    return max(max_streak, current_streak)  # Ensure the final streak is considered
'''

def longest_true_streak_with_gaps(lst, x):
    max_streak = 0
    current_streak = 0
    gap_count = 0
    start_index = None
    best_start, best_end = None, None

    for i, value in enumerate(lst):
        if value:  
            if start_index is None:
                start_index = i  # Mark the start of a new sequence
            current_streak += 1 + gap_count
            gap_count = 0  
        else:  
            if gap_count < x:
                gap_count += 1  
            else:  
                if current_streak > max_streak:
                    max_streak = current_streak
                    best_start, best_end = start_index, i - 1  

                current_streak = 0
                gap_count = 0
                start_index = None  

    # Final check in case the longest streak ends at the last element
    if current_streak > max_streak:
        max_streak = current_streak
        best_start, best_end = start_index, len(lst) - 1

    return max_streak, (best_start, best_end)

#%% We only care about the orbits where there are conductance files

# Conductance path
inpath = '/home/bing/dynamit_server/disk/IMAGE_FUV/fuv/conductance/'

# Get orbits numbers
orbits = [int(o[-7:-3]) for o in sorted(glob.glob(inpath + '*.nc'))]

#%% Test load an orbit

orbit = orbits[1]

# Load conductance nc file
filename = inpath + 'or_' + str(orbit).zfill(4) + '.nc'
cI = conductanceImage(filename=filename)

# There has to be estimates of H, P, sH, and sP.
flag = (~np.isnan(cI.H)) & (~np.isnan(cI.P)) & (~np.isnan(cI.dH)) & (~np.isnan(cI.dP))
data_ratio = np.sum(flag, axis=(1,2)) / (flag.shape[1] * flag.shape[2])


#%%

plt.ioff()
fig, axs = plt.subplots(1, 2, figsize=(15, 9))

for orbit in orbits:
    
    print(orbit)
    os.makedirs('/home/bing/Dropbox/work/temp_storage/density_analysis/{}'.format(orbit), exist_ok=True)

    # Load conductance nc file
    filename = inpath + 'or_' + str(orbit).zfill(4) + '.nc'
    cI = conductanceImage(filename=filename)    

    # There has to be estimates of H, P, sH, and sP.
    flag = (~np.isnan(cI.H)) & (~np.isnan(cI.P)) & (~np.isnan(cI.dH)) & (~np.isnan(cI.dP))
    data_ratio = np.sum(flag, axis=(1,2)) / (flag.shape[1] * flag.shape[2])
        
    ylim = (np.min(data_ratio), np.max(data_ratio))
    for i in range(flag.shape[0]):
        axs[0].cla()
        axs[1].cla()
        
        axs[0].plot(data_ratio)
        axs[0].plot([i]*2, ylim)
        
        q = cI.H[i] + 0
        q[~flag[i]] = np.nan
        
        axs[1].imshow(q)
        
        plt.savefig('/home/bing/Dropbox/work/temp_storage/density_analysis/{}/{}.png'.format(orbit, i), bbox_inches='tight')

plt.close('all')        
plt.ion()

#%% parallel

import matplotlib.pyplot as plt
from multiprocessing import Pool

def process_orbit(orbit):
    orbit_dir = f'/home/bing/Dropbox/work/temp_storage/density_analysis/{orbit}'
    os.makedirs(orbit_dir, exist_ok=True)

    # Load conductance nc file
    filename = inpath + 'or_' + str(orbit).zfill(4) + '.nc'
    cI = conductanceImage(filename=filename)    

    # There has to be estimates of H, P, sH, and sP.
    flag = (~np.isnan(cI.H)) & (~np.isnan(cI.P)) & (~np.isnan(cI.dH)) & (~np.isnan(cI.dP))
    data_ratio = np.sum(flag, axis=(1,2)) / (flag.shape[1] * flag.shape[2])

    ylim = (np.min(data_ratio), np.max(data_ratio))

    fig, axs = plt.subplots(1, 2, figsize=(15, 9))
    
    for i in range(flag.shape[0]):
        axs[0].cla()
        axs[1].cla()

        axs[0].plot(data_ratio)
        axs[0].plot([i]*2, ylim)

        q = cI.H[i].copy()
        q[~flag[i]] = np.nan

        axs[1].imshow(q)

        plt.savefig(f'{orbit_dir}/{i}.png', bbox_inches='tight')

    plt.close(fig)

plt.ioff()  # Disable interactive mode to avoid conflicts

pool = Pool()
pool.map(process_orbit, orbits)
pool.close()
pool.join()

plt.ion()  # Re-enable interactive mode

#%% I HATE THIS SHIT SO MUCH... WHY IS NOTHING CONSISTENT!!!!! FUCKING SHIT!!!!

p_b = '/home/bing/Downloads/'

orbit_info = np.zeros((len(orbits), 25))

inds = []

for i, orbit in tqdm(enumerate(orbits), total=len(orbits)):
    
    #################### Boundary file and index
    # Load boundary file orbit information
    bf = pd.read_hdf(p_b + 'final_boundaries.h5', key='final', where='orbit=="{}"'.format(orbit))

    # Is it empty?
    if bf.size == 0:
        continue
    else:
        orbit_info[i, 0] = 1
        
    # Anders boundary index
    bfg = bf.groupby('date')
    # Condition 1: All 'isglobal' values must be True for a given date
    ind0 = bfg['isglobal'].all()
    # Condition 2: A_mean > P_mean + P_std + A_std for all MLT in a given date
    ind1 = bfg.apply(lambda g: (g['A_mean'] > (g['P_mean'] + g['P_std'] + g['A_std'])).all())
    # Condition 3: A_mean > S_mean + S_std + A_std for all MLT in a given date
    ind2 = bfg.apply(lambda g: (g['A_mean'] > (g['S_mean'] + g['S_std'] + g['A_std'])).all())
    # Condition 4: 'count' > 12 for all MLT in a given date
    ind3 = bfg['count'].apply(lambda g: (g > 12).all())    
    # Combined
    ind = ind0 & ind1 & ind2 & ind3
    
    # Are any of the time steps okay?
    if ind.any():
        orbit_info[i, 1] = 1
    else:
        continue
        
    # Size of the boundary file prior to trimming
    orbit_info[i, 2] = ind.size
    
    # How many are okay
    orbit_info[i, 3] = np.sum(ind)
    
    # Get bf time
    btime = ind.index.to_pydatetime()
    
    #################### Conductance file
    # Load conductance nc file
    filename = inpath + 'or_' + str(orbit).zfill(4) + '.nc'
    cI = conductanceImage(filename=filename)
    
    # Get time in conductance file
    ctime = np.copy(cI.time)
    
    #################### Find matches
    bmatches = np.array([np.any(np.abs(ctime - bt) <= np.timedelta64(10, 's'))
                         for bt in btime])
    
    cmatches = np.array([np.any(np.abs(btime - ct) <= np.timedelta64(10, 's'))
                         for ct in ctime])
    
    cind = np.zeros(ctime.size) # Assuming that indices that don't appear in boundary file are bad
    cind[cmatches] = ind[bmatches]
    ind = np.copy(cind)
    inds.append(ind)
    
    #################### Is it still okay?
    # Are any of the time steps okay?
    if ind.any():
        orbit_info[i, 4] = 1
    else:
        continue
        
    # Size of the boundary file prior to trimming
    orbit_info[i, 5] = ind.size
    
    # How many are okay
    orbit_info[i, 6] = np.sum(ind)
    
    #################### Longest streak
    
    ns, idx = longest_true_streak_with_gaps(ind, 0)
    orbit_info[i, 7] = ns
    orbit_info[i, 8] = idx[0]
    orbit_info[i, 9] = idx[1]
    
    ns, idx = longest_true_streak_with_gaps(ind, 1)
    orbit_info[i, 10] = ns
    orbit_info[i, 11] = idx[0]
    orbit_info[i, 12] = idx[1]
    
    ns, idx = longest_true_streak_with_gaps(ind, 2)
    orbit_info[i, 13] = ns
    orbit_info[i, 14] = idx[0]
    orbit_info[i, 15] = idx[1]
    
    ns, idx = longest_true_streak_with_gaps(ind, 3)
    orbit_info[i, 16] = ns
    orbit_info[i, 17] = idx[0]
    orbit_info[i, 18] = idx[1]
    
    ns, idx = longest_true_streak_with_gaps(ind, 4)
    orbit_info[i, 19] = ns
    orbit_info[i, 20] = idx[0]
    orbit_info[i, 21] = idx[1]
    
    ns, idx = longest_true_streak_with_gaps(ind, 5)
    orbit_info[i, 22] = ns
    orbit_info[i, 23] = idx[0]
    orbit_info[i, 24] = idx[1]


#%% Parallel shit

import multiprocessing as mp

p_b = '/home/bing/Downloads/'

def process_orbit(args):
    """Function to process a single orbit."""
    i, orbit = args

    # Load boundary file
    bf = pd.read_hdf(p_b + 'final_boundaries.h5', key='final', where='orbit=="{}"'.format(orbit))
    
    # Initialize output row
    row = np.zeros(25)
    
    # Check if empty
    if bf.size == 0:
        return i, row, []  # Return index and None

    # If boundary data
    row[0] = 1  # Mark as processed

    # Group and apply conditions
    bfg = bf.groupby('date')
    ind0 = bfg['isglobal'].all()
    ind1 = bfg.apply(lambda g: (g['A_mean'] > (g['P_mean'] + g['P_std'] + g['A_std'])).all())
    ind2 = bfg.apply(lambda g: (g['A_mean'] > (g['S_mean'] + g['S_std'] + g['A_std'])).all())
    ind3 = bfg['count'].apply(lambda g: (g > 12).all())

    ind = ind0 & ind1 & ind2 & ind3

    # Check if any good
    if not ind.any():
        return i, row, []
    
    # If any good
    row[1] = 1
    row[2] = ind.size
    row[3] = np.sum(ind)

    # Get boundary file times
    btime = ind.index.to_pydatetime()

    # Load conductance data
    filename = inpath + 'or_' + str(orbit).zfill(4) + '.nc'
    cI = conductanceImage(filename=filename)
    ctime = np.copy(cI.time)

    # Find matches
    bmatches = np.array([np.any(np.abs(ctime - bt) <= np.timedelta64(10, 's')) for bt in btime])
    cmatches = np.array([np.any(np.abs(btime - ct) <= np.timedelta64(10, 's')) for ct in ctime])

    cind = np.zeros(ctime.size)
    cind[cmatches] = ind[bmatches]
    ind = np.copy(cind)

    # Check if any good
    if not ind.any():
        return i, row, ind

    # If any good
    row[4] = 1
    row[5] = ind.size
    row[6] = np.sum(ind)

    # Compute longest streaks
    for j in range(6):
        ns, idx = longest_true_streak_with_gaps(ind, j)
        row[7 + j * 3] = ns
        row[8 + j * 3] = idx[0]
        row[9 + j * 3] = idx[1]

    return i, row, ind  # Return index, processed data, and ind array

# Multiprocessing execution
with mp.Pool(mp.cpu_count()) as pool:
    results = list(tqdm(pool.imap(process_orbit, enumerate(orbits)), total=len(orbits)))

# Initialize output storage
orbit_info = np.zeros((len(orbits), 25))
inds = []

# Store results
for i, row, ind in results:
    orbit_info[i] = row
    inds.append(ind)

#%% Save for speed-up

#np.save('/home/bing/Dropbox/work/temp_storage/orbit_info.npy', orbit_info)
orbit_info = np.load('/home/bing/Dropbox/work/temp_storage/orbit_info.npy')

import pickle
#pickle.dump(inds, open('/home/bing/Dropbox/work/temp_storage/inds.pickle', 'wb'))
inds = pickle.load(open('/home/bing/Dropbox/work/temp_storage/inds.pickle', 'rb'))

#%%

q0 = len(inds)
print('Total amount of orbits:')
print(q0)
print('')

q1 = np.sum(orbit_info[:, 0] == 0)
print('Amount of orbits with no boundary data:')
print(q1)
print('')

q2 = np.sum(orbit_info[:, 1] == 0) - q1
print('Amount of orbits without a single good boundary:')
print(q2)
print('')

print('Remaining orbits:')
print(q0-q1-q2)
print('')

print('Amount of orbits with sequence longer than 14 and gaps smaller than 4:')
print(np.sum(orbit_info[:, 16] > 14))
print('')

print('Mean, median, and std of sequences:')
print(np.mean(orbit_info[orbit_info[:, 16] > 14, 16])*2)
print(np.median(orbit_info[orbit_info[:, 16] > 14, 16])*2)
print(np.std(orbit_info[orbit_info[:, 16] > 14, 16])*2)

#%%

q1 = [np.sum(orbit_info[:, 7] > i) for i in np.arange(0, 150)]
q2 = [np.sum(orbit_info[:, 10] > i) for i in np.arange(0, 150)]
q3 = [np.sum(orbit_info[:, 13] > i) for i in np.arange(0, 150)]
q4 = [np.sum(orbit_info[:, 16] > i) for i in np.arange(0, 150)]
q5 = [np.sum(orbit_info[:, 19] > i) for i in np.arange(0, 150)]
q6 = [np.sum(orbit_info[:, 22] > i) for i in np.arange(0, 150)]


plt.figure(figsize=(6,6))
plt.plot(q1, label='no gap')
plt.plot(q2, label='max gap size 1')
plt.plot(q3, label='max gap size 2')
plt.plot(q4, label='max gap size 3')
plt.plot(q5, label='max gap size 4')
plt.plot(q6, label='max gap size 5')
plt.xlabel('Min sequence length')
plt.ylabel('# of orbits with t above thres')
plt.legend()
plt.grid()

plt.figure()
plt.hist(orbit_info[orbit_info[:, 16] > 14, 16], bins=100, edgecolor='k')
plt.xlabel('Sequence length')
plt.ylabel('# of orbits')
plt.title('Distribution of orbits sequences above 14 t with gaps smaller than 4')

#%% parallel - again

import matplotlib.pyplot as plt
from multiprocessing import Pool

def process_orbit_2(orbit):
    orbit_dir = f'/home/bing/Dropbox/work/temp_storage/criteria_analysis/{orbit}'
    
    j = np.argmin(abs(np.array(orbits) - orbit))
    
    if not good_ones[j]:
        return
    
    os.makedirs(orbit_dir, exist_ok=True)
    
    ind = inds[j]
    
    # Load conductance nc file
    filename = inpath + 'or_' + str(orbit).zfill(4) + '.nc'
    cI = conductanceImage(filename=filename)    
    ctime = np.copy(cI.time)
    # There has to be estimates of H, P, sH, and sP.
    flag = (~np.isnan(cI.H)) & (~np.isnan(cI.P)) & (~np.isnan(cI.dH)) & (~np.isnan(cI.dP))
    data_ratio = np.sum(flag, axis=(1,2)) / (flag.shape[1] * flag.shape[2])

    ylim = (np.min(data_ratio), np.max(data_ratio))

    fig, axs = plt.subplots(1, 2, figsize=(15, 9))
    
    for i in range(flag.shape[0]):
        axs[0].cla()
        axs[1].cla()

        axs[0].plot(data_ratio)
        axs[0].plot([i]*2, ylim)

        q = cI.H[i].copy()
        q[~flag[i]] = np.nan

        axs[1].imshow(q)
                
        tit = 'Bad'
        if ind[i]:
            tit = 'Good'
        
        if (i >= orbit_info[j, 17]) & (i <= orbit_info[j, 18]):
            tit = 'SEQUENCE'
        tit = str(ctime[0]) + ' : ' + tit
        
        #axs[1].set_title(tit)
        axs[1].text(.25, 1.1, tit, ha='left', va='center', transform=axs[1].transAxes, fontsize=16)

        plt.savefig(f'{orbit_dir}/{i}.png', bbox_inches='tight')

    plt.close(fig)

good_ones = orbit_info[:, 16] > 14

plt.ioff()  # Disable interactive mode to avoid conflicts

pool = Pool()
pool.map(process_orbit_2, orbits)
pool.close()
pool.join()

plt.ion()  # Re-enable interactive mode


#%% Check time

orbit = 187
filename = inpath + 'or_' + str(orbit).zfill(4) + '.nc'
cI = conductanceImage(filename=filename)
ctime = np.copy(cI.time)
print(ctime[0])
print(ctime[-1])

#%%

orbit = 187
filename = inpath + 'or_' + str(orbit).zfill(4) + '.nc'
cI = conductanceImage(filename=filename)
ctime = np.copy(cI.time)
#t = np.array([(t-ctime[0]).seconds for t in ctime])
t = np.array([(t-ctime[100]).seconds for t in ctime])

# Initiate Spline image
#sI = splineImage(cI.H, cI.P, cI.grid, cI.dH, cI.dP, t=t, ncpt=20)
#sI = splineImage(cI.H[100:161], cI.P[100:161], cI.grid, cI.dH[100:161], cI.dP[100:161], t=t[100:161], ncp=20, ncpt=20)
sI = splineImage(cI.H[100:161], cI.P[100:161], cI.grid, cI.dH[100:161], cI.dP[100:161], t=t[100:161], ncp=25, ncpt=25)

sI.generate_design_matrix_2d()
sI.generate_design_matrix_3d()

sI.make_models(lH=1e-1, lP=1e-1)
    
pH = sI.eval_Hall()
pP = sI.eval_Pedersen()

#%%

np.save('/home/bing/Dropbox/work/temp_storage/mH.npy', sI.mH)
np.save('/home/bing/Dropbox/work/temp_storage/mP.npy', sI.mP)

#%%

vmax = np.nanmax((sI.H, pH))
vmax_res = np.nanmax(abs(sI.H - pH))
plt.ioff()
for i, (Hi, pHi) in enumerate(zip(sI.H, pH)):
    fig, axs = plt.subplots(1, 3, figsize=(18,8))
    im = axs[0].pcolormesh(sI.grid.xi, sI.grid.eta, Hi, cmap='PuOr', vmin=-vmax, vmax=vmax)    
    fig.colorbar(im, ax=axs[0], orientation='horizontal')
    
    im = axs[1].pcolormesh(sI.grid.xi, sI.grid.eta, pHi, cmap='PuOr', vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=axs[1], orientation='horizontal')
    
    im = axs[2].pcolormesh(sI.grid.xi, sI.grid.eta, Hi-pHi, cmap='PuOr', vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=axs[2], orientation='horizontal')
    
    for ax in axs:
        ax.set_aspect('equal')
    
    plt.savefig(f'/home/bing/Dropbox/work/temp_storage/spline_model_comp_standard_fill/{i}.png', bbox_inches='tight')
    plt.close('all')
plt.ion()

#%% 

class splineModel():
    def __init__(self, sI = None, filename = None):
        
        if (sI is None) and (filename is None):
            print('No input given.')
            return
        
        if not sI is None:
            self.make_model_from_sI(sI)
    
    def make_model_from_sI(self, sI):
        # grid
        self.grid = sI.grid
        
        # Hall and Pedersen model parameters
        self.mH = sI.mH
        self.mH = sI.mH
        
        # self.

sM = splineModel(sI)





















































#%%

def longest_true_streak_with_gaps(lst, x):
    max_streak = 0
    current_streak = 0
    gap_count = 0

    for value in lst:
        if value:  # If it's True, extend the current streak
            current_streak += 1 + gap_count
            gap_count = 0  # Reset the gap count
        else:  # If it's False, check if we can tolerate the gap
            if gap_count < x:
                gap_count += 1
            else:  # If the gap is too large, reset the streak
                max_streak = max(max_streak, current_streak)
                current_streak = 0
                gap_count = 0

    return max(max_streak, current_streak)  # Ensure the final streak is considered


p_b = '/home/bing/Downloads/'

orbit_info = np.zeros((len(orbits), 11))

for i, orbit in tqdm(enumerate(orbits), total=len(orbits)):
    
    #################### Boundary file and index
    # Load boundary file orbit information
    bf = pd.read_hdf(p_b + 'final_boundaries.h5', key='final', where='orbit=="{}"'.format(orbit))

    # Is it empty?
    if bf.size == 0:
        continue
    else:
        orbit_info[i, 0] = 1
        
    # Anders boundary index
    bfg = bf.groupby('date')
    # Condition 1: All 'isglobal' values must be True for a given date
    ind0 = bfg['isglobal'].all()
    # Condition 2: A_mean > P_mean + P_std + A_std for all MLT in a given date
    ind1 = bfg.apply(lambda g: (g['A_mean'] > (g['P_mean'] + g['P_std'] + g['A_std'])).all())
    # Condition 3: A_mean > S_mean + S_std + A_std for all MLT in a given date
    ind2 = bfg.apply(lambda g: (g['A_mean'] > (g['S_mean'] + g['S_std'] + g['A_std'])).all())
    # Condition 4: 'count' > 12 for all MLT in a given date
    ind3 = bfg['count'].apply(lambda g: (g > 12).all())    
    # Combined
    ind = ind0 & ind1 & ind2 & ind3
    
    # Are any of the time steps okay?
    if ind.any():
        orbit_info[i, 1] = 1
        orbit_info[i, 2] = np.sum(ind)
        orbit_info[i, 3] = ind.size
        orbit_info[i, 4] = longest_true_streak_with_gaps(ind.to_list(), 0)
        orbit_info[i, 5] = longest_true_streak_with_gaps(ind.to_list(), 1)
        orbit_info[i, 6] = longest_true_streak_with_gaps(ind.to_list(), 2)
        orbit_info[i, 7] = longest_true_streak_with_gaps(ind.to_list(), 3)
        orbit_info[i, 8] = longest_true_streak_with_gaps(ind.to_list(), 4)
        orbit_info[i, 9] = longest_true_streak_with_gaps(ind.to_list(), 5)
    else:
        continue

#%%
#np.save('/home/bing/Dropbox/work/temp_storage/orbit_info.npy', orbit_info)
orbit_info = np.load('/home/bing/Dropbox/work/temp_storage/orbit_info.npy')

#%%

plt.figure(figsize=(6,6))
plt.hist(orbit_info[orbit_info[:, 2] != 0, 2], bins=100, edgecolor='k')
plt.xlabel('# of time steps')
plt.ylabel('# of orbits')
plt.title('# of time steps that passed per orbit')


plt.figure(figsize=(6,6))
plt.hist(orbit_info[orbit_info[:, 2] != 0, 2] / orbit_info[orbit_info[:, 2] != 0, 3], bins=100, edgecolor='k')
plt.xlabel('relative amount of passed time steps')
plt.ylabel('# of orbits')
plt.title('relative')

#%%

plt.figure(figsize=(6,6))
plt.hist(orbit_info[orbit_info[:, 2] != 0, 4], bins=100, edgecolor='k')
plt.xlabel('# of consecutive passed time steps')
plt.ylabel('# of orbits')
plt.title('maximum # of consecutive time steps that passed per orbit\nno gaps')

plt.figure(figsize=(6,6))
plt.hist(orbit_info[orbit_info[:, 2] != 0, 5], bins=100, edgecolor='k')
plt.xlabel('# of consecutive passed time steps')
plt.ylabel('# of orbits')
plt.title('maximum # of consecutive time steps that passed per orbit\nmax gap size 1')

plt.figure(figsize=(6,6))
plt.hist(orbit_info[orbit_info[:, 2] != 0, 6], bins=100, edgecolor='k')
plt.xlabel('# of consecutive passed time steps')
plt.ylabel('# of orbits')
plt.title('maximum # of consecutive time steps that passed per orbit\nmax gap size 2')

plt.figure(figsize=(6,6))
plt.hist(orbit_info[orbit_info[:, 2] != 0, 7], bins=100, edgecolor='k')
plt.xlabel('# of consecutive passed time steps')
plt.ylabel('# of orbits')
plt.title('maximum # of consecutive time steps that passed per orbit\nmax gap size 3')

plt.figure(figsize=(6,6))
plt.hist(orbit_info[orbit_info[:, 2] != 0, 8], bins=100, edgecolor='k')
plt.xlabel('# of consecutive passed time steps')
plt.ylabel('# of orbits')
plt.title('maximum # of consecutive time steps that passed per orbit\nmax gap size 4')

plt.figure(figsize=(6,6))
plt.hist(orbit_info[orbit_info[:, 2] != 0, 9], bins=100, edgecolor='k')
plt.xlabel('# of consecutive passed time steps')
plt.ylabel('# of orbits')
plt.title('maximum # of consecutive time steps that passed per orbit\nmax gap size 5')

#%%

q1 = [np.sum(orbit_info[:, 4] > i) for i in np.arange(0, 150)]
q2 = [np.sum(orbit_info[:, 5] > i) for i in np.arange(0, 150)]
q3 = [np.sum(orbit_info[:, 6] > i) for i in np.arange(0, 150)]
q4 = [np.sum(orbit_info[:, 7] > i) for i in np.arange(0, 150)]
q5 = [np.sum(orbit_info[:, 8] > i) for i in np.arange(0, 150)]
q6 = [np.sum(orbit_info[:, 9] > i) for i in np.arange(0, 150)]


plt.figure(figsize=(6,6))
plt.plot(q1, label='no gap')
plt.plot(q2, label='max gap size 1')
plt.plot(q3, label='max gap size 2')
plt.plot(q4, label='max gap size 3')
plt.plot(q5, label='max gap size 4')
plt.plot(q6, label='max gap size 5')
plt.xlabel('Threshold')
plt.ylabel('# of orbits with t above thres')
plt.legend()
plt.grid()

#%% parallel - again

import matplotlib.pyplot as plt
from multiprocessing import Pool

def process_orbit_2(orbit):
    orbit_dir = f'/home/bing/Dropbox/work/temp_storage/criteria_analysis/{orbit}'
    os.makedirs(orbit_dir, exist_ok=True)
    
    bf = pd.read_hdf(p_b + 'final_boundaries.h5', key='final', where='orbit=="{}"'.format(orbit))
        
    # Anders boundary index
    bfg = bf.groupby('date')
    # Condition 1: All 'isglobal' values must be True for a given date
    ind0 = bfg['isglobal'].all()
    # Condition 2: A_mean > P_mean + P_std + A_std for all MLT in a given date
    ind1 = bfg.apply(lambda g: (g['A_mean'] > (g['P_mean'] + g['P_std'] + g['A_std'])).all())
    # Condition 3: A_mean > S_mean + S_std + A_std for all MLT in a given date
    ind2 = bfg.apply(lambda g: (g['A_mean'] > (g['S_mean'] + g['S_std'] + g['A_std'])).all())
    # Condition 4: 'count' > 12 for all MLT in a given date
    ind3 = bfg['count'].apply(lambda g: (g > 12).all())    
    # Combined
    ind = ind0 & ind1 & ind2 & ind3
    
    # Load conductance nc file
    filename = inpath + 'or_' + str(orbit).zfill(4) + '.nc'
    cI = conductanceImage(filename=filename)    

    # There has to be estimates of H, P, sH, and sP.
    flag = (~np.isnan(cI.H)) & (~np.isnan(cI.P)) & (~np.isnan(cI.dH)) & (~np.isnan(cI.dP))
    data_ratio = np.sum(flag, axis=(1,2)) / (flag.shape[1] * flag.shape[2])

    ylim = (np.min(data_ratio), np.max(data_ratio))

    fig, axs = plt.subplots(1, 2, figsize=(15, 9))
    
    for i in range(flag.shape[0]):
        axs[0].cla()
        axs[1].cla()

        axs[0].plot(data_ratio)
        axs[0].plot([i]*2, ylim)

        q = cI.H[i].copy()
        q[~flag[i]] = np.nan

        axs[1].imshow(q)
        
        tit = 'Bad'
        if ind[i]:
            tit = 'Good'
        
        axs[1].set_title(tit)

        plt.savefig(f'{orbit_dir}/{i}.png', bbox_inches='tight')

    plt.close(fig)

#orbits_filtered = list(np.array(orbits)[orbit_info[:, 0] == 1])
orbits_filtered = list(np.array(orbits)[orbit_info[:, 7] >= 15])

plt.ioff()  # Disable interactive mode to avoid conflicts

pool = Pool()
pool.map(process_orbit_2, orbits_filtered)
pool.close()
pool.join()

plt.ion()  # Re-enable interactive mode































#%%

def longest_true_streak_with_gaps(lst, x):
    max_streak = 0
    current_streak = 0
    gap_count = 0

    for value in lst:
        if value:  # If it's True, extend the current streak
            current_streak += 1 + gap_count
            gap_count = 0  # Reset the gap count
        else:  # If it's False, check if we can tolerate the gap
            if gap_count < x:
                gap_count += 1
            else:  # If the gap is too large, reset the streak
                max_streak = max(max_streak, current_streak)
                current_streak = 0
                gap_count = 0

    return max(max_streak, current_streak)  # Ensure the final streak is considered

# Example usage:
data = [True, True, False, False, True, True, True, False, False, True, True, False, True]
print(longest_true_streak_with_gaps(data, 0))  # Output: 6 (merging sequences with 1 False gap)
print(longest_true_streak_with_gaps(data, 1))  # Output: 6 (merging sequences with 1 False gap)
print(longest_true_streak_with_gaps(data, 2))  # Output: 10 (allowing larger gaps)


#%%

def longest_true_streak_with_gaps(lst, x):
    max_streak = 0
    current_streak = 0
    gap_count = 0
    in_streak = False  # Tracks whether we are in an active streak

    for value in lst:
        if value:  
            if in_streak:  
                # Continue the existing streak
                current_streak += 1 + gap_count
            else:  
                # Start a new streak
                current_streak = 1
                in_streak = True

            gap_count = 0  # Reset gap count
        else:  
            if in_streak and gap_count < x:
                # Allow the gap if it's within the limit
                gap_count += 1
            else:  
                # End the streak if the gap is too large or not bridging
                max_streak = max(max_streak, current_streak)
                current_streak = 0
                in_streak = False
                gap_count = 0  

    return max(max_streak, current_streak)

# Example usage:
data = [True, True, False, False, True, True, True, False, False, True, True, False, True]
print(longest_true_streak_with_gaps(data, 1))  # Output: 3
print(longest_true_streak_with_gaps(data, 2))  # Output: 6


#%%

from itertools import groupby

def longest_true_streak(lst):
    return max((sum(1 for _ in g) for k, g in groupby(lst) if k), default=0)

# Example usage:
data = [False, True, True, False, True, True, True, False, True]
print(longest_true_streak(data))  # Output: 3


















#%%

orbit_dates = '/home/bing/dynamit_server/disk/IMAGE_FUV/fuv/orbitdates.csv'

#%% Import orbit file

print('Reading orbit date from NIRD file')
orbits = pd.read_csv(orbit_dates)

# Convert str to dt
orbits['dt_start'] = pd.to_datetime(orbits['date_start'],format='%Y-%m-%d %H:%M:%S')
orbits['dt_end'] = pd.to_datetime(orbits['date_end'],format='%Y-%m-%d %H:%M:%S')

#%% Fetch boundaries

#p_b = '/home/bing/dynamit_server/disk/IMAGE_FUV/fuv/boundaries/'
p_b = '/home/bing/Downloads/'


bf = pd.read_hdf(p_b + 'final_boundaries.h5', key='final', where='orbit=="{}"'.format(85))

# Load isglobal flag from boundary files
#isglobal = pd.read_hdf(p_b + 'final_boundaries.h5', columns=['isglobal']).groupby('date').all()
isglobal = pd.read_hdf(p_b + 'final_boundaries.h5', columns=['isglobal'])


#isglobal = pd.read_hdf(p_b + 'final_boundaries.h5', columns=['isglobal', 'A_mean', 'A_std', 'P_mean', ''])


# Quality flags
ind0 = bf.loc[t.values,'isglobal'].all()
ind1 = (bf.loc[t.values,'A_mean'] > bf.loc[t.values,'P_mean']+bf.loc[t.values,'P_std']+bf.loc[t.values,'A_std']).all()
ind2 = (bf.loc[t.values,'A_mean'] > bf.loc[t.values,'S_mean']+bf.loc[t.values,'S_std']+bf.loc[t.values,'A_std']).all()
ind3 = (bf.loc[t.values,'count'] > 12).all()

#alpha = 1 
#linestyle = '-' if (ind0&ind1&ind2&ind3) else ':'

#%% Paths

inpath = '/home/bing/dynamit_server/disk/IMAGE_FUV/fuv/conductance/'

#%% Get orbits numbers

orbits = [int(o[-7:-3]) for o in sorted(glob.glob(inpath + '*.nc'))]

#%% I HATE THIS SHIT SO MUCH... WHY IS NOTHING CONSISTENT!!!!! FUCKING SHIT!!!!

orbit_info = np.zeros((len(orbits), 11))

for i, orbit in tqdm(enumerate(orbits), total=len(orbits)):
    
    #################### Boundary file and index
    # Load boundary file orbit information
    bf = pd.read_hdf(p_b + 'final_boundaries.h5', key='final', where='orbit=="{}"'.format(orbit))

    # Is it empty?
    if bf.size == 0:
        continue
    else:
        orbit_info[i, 0] = 1
        
    # Anders boundary index
    bfg = bf.groupby('date')
    # Condition 1: All 'isglobal' values must be True for a given date
    ind0 = bfg['isglobal'].all()
    # Condition 2: A_mean > P_mean + P_std + A_std for all MLT in a given date
    ind1 = bfg.apply(lambda g: (g['A_mean'] > (g['P_mean'] + g['P_std'] + g['A_std'])).all())
    # Condition 3: A_mean > S_mean + S_std + A_std for all MLT in a given date
    ind2 = bfg.apply(lambda g: (g['A_mean'] > (g['S_mean'] + g['S_std'] + g['A_std'])).all())
    # Condition 4: 'count' > 12 for all MLT in a given date
    ind3 = bfg['count'].apply(lambda g: (g > 12).all())    
    # Combined
    ind = ind0 & ind1 & ind2 & ind3
    
    # Are any of the time steps okay?
    if ind.any():
        orbit_info[i, 1] = 1
    else:
        continue
        
    # Size of the boundary file prior to trimming
    orbit_info[i, 2] = ind.size
    
    # How many are okay
    orbit_info[i, 3] = np.sum(ind)
    
    # Get bf time
    btime = ind.index.to_pydatetime()
    
    #################### Conductance file
    # Load conductance nc file
    filename = inpath + 'or_' + str(orbit).zfill(4) + '.nc'
    cI = conductanceImage(filename=filename)
    
    # Get time in conductance file
    ctime = np.copy(cI.time)
    
    #################### Clip boundary time
    # Clip the boundary file is necessary
    if btime[-1] > ctime[-1]:
        j = np.argmin(np.abs(btime - ctime[-1]))
        btime = btime[:j+1]
        ind = ind.iloc[:j+1]
        orbit_info[i, 5] = j
    else:
        orbit_info[i, 5] = -1
    
    if btime[0] < ctime[0]:
        j = np.argmin(np.abs(btime - ctime[0]))
        btime = btime[j:]
        ind = ind.iloc[j:]
        orbit_info[i, 4] = j
    else:
        orbit_info[i, 4] = -1    
    
    # Check if any time steps are okay, again.
    if ind.any():
        orbit_info[i, 6] = 1
    else:
        continue
        
    # Check truncated size
    orbit_info[i, 7] = ind.size
    
    # How many are okay ?
    orbit_info[i, 8] = np.sum(ind)
    
    #################### Clip conductance time
    # Clip conductance file is necessary
    if ctime[0] < btime[0]:
        j = np.argmin(np.abs(ctime - btime[0]))
        orbit_info[i, 9] = j
    else:
        orbit_info[i, 9] = -1
    
    if ctime[-1] > btime[-1]:
        j = np.argmin(np.abs(ctime - btime[-1]))        
        orbit_info[i, 10] = j
    else:
        orbit_info[i, 10] = -1

#%%
print('A total of ' + str(orbit_info.shape[0]))

f = orbit_info[:, 0] == 1 # There is data
print(np.sum(~f), 'are empty')

f = f & (orbit_info[:, 1] == 1)
print(np.sum(~f), 'are empty or all bad')

plt.figure(figsize=(8, 5))
plt.hist(orbit_info[f, 2], bins=50, edgecolor='k')
plt.title('Time stamps in boundary file orbit')
plt.xlabel('Time stamps')
plt.ylabel('Orbits')

plt.figure(figsize=(8, 5))
plt.hist(orbit_info[f, 3] / orbit_info[f, 2] * 100, bins=50, edgecolor='k')
plt.title('Percentage good frames orbit')
plt.xlabel('Percent')
plt.ylabel('Orbits')


plt.figure(figsize=(8, 5))
plt.hist(orbit_info[f, 7], bins=50, edgecolor='k')
plt.title('Time stamps in boundary file orbit (clipped)')
plt.xlabel('Time stamps')
plt.ylabel('Orbits')

plt.figure(figsize=(8, 5))
plt.hist(orbit_info[f, 8] / orbit_info[f, 7] * 100, bins=50, edgecolor='k')
plt.title('Percentage good frames orbit  (clipped)')
plt.xlabel('Percent')
plt.ylabel('Orbits')



plt.figure()
plt.hist(orbit_info[f, 2], bins=50)


#%% test

def get_info_from_boundary_file(filename, orbits):
    
    bf_info = np.zeros((orbits.size, 10))
    for i, orbit in ernumerate(orbits):
        # Fetch boundary data from orbit
        bf = pd.read_hdf(filename, key='final', where='orbit=="{}"'.format(orbit))
        
        # Is it empty?
        if bf.size == 0:
            continue
        else:
            bf_info[i, 0] = 1
        
        # How many timestamps
        bf_info[i, 1] = 
        
    
    
    # Fetch boundary data from orbit
    bf = pd.read_hdf(filename, key='final', where='orbit=="{}"'.format(orbit))
    bf = pd.read_hdf(filename, key='final', where='orbit=="{}"'.format(orbit)).groupby('date')

    # Condition 1: All 'isglobal' values must be True for a given date
    ind0 = bf['isglobal'].all()
    # Condition 2: A_mean > P_mean + P_std + A_std for all MLT in a given date
    ind1 = bf.apply(lambda g: (g['A_mean'] > (g['P_mean'] + g['P_std'] + g['A_std'])).all())
    # Condition 3: A_mean > S_mean + S_std + A_std for all MLT in a given date
    ind2 = bf.apply(lambda g: (g['A_mean'] > (g['S_mean'] + g['S_std'] + g['A_std'])).all())
    # Condition 4: 'count' > 12 for all MLT in a given date
    ind3 = bf['count'].apply(lambda g: (g > 12).all())
    
    # Combined
    ind = ind0 & ind1 & ind2 & ind3
    

count = 0
for orbit in orbits:
    ind = get_orbit_index(p_b + 'final_boundaries.h5', orbit)
    if ind.size == 0:
        print('No data in orbit')
        continue
    if ind.all():
        count += 1


#%% Loop over all orbits

orbit = 89

for orbit in orbits:
    
    # Get the boundary orbit index
    ind = get_orbit_index(p_b + 'final_boundaries.h5', orbit)
    if ind.size == 0:
        print('No data in orbit')
        continue
    
    # If there is not a single timestamp of good quality, then skip
    if not ind.all():
        continue
        
    # Load conductance file
    filename = inpath + 'or_' + str(orbit).zfill(4) + '.nc'
    cI = conductanceImage(filename=filename)

    t = np.array([(t-cI.time[0]).seconds for t in cI.time])
    
    # Initiate Spline image
    sI = splineImage(cI.H, cI.P, cI.grid, cI.dH, cI.dP, t=t)
    
    sI.generate_design_matrix_2d()
    sI.generate_design_matrix_3d()
    
    sI.make_models(lH=1e0, lP=1e0)
        
    pH = sI.eval_Hall()
    pP = sI.eval_Pedersen()
    
    
    pH, pdH = sI.eval_Hall(uncertainty=True)
    pP, pdP = sI.eval_Pedersen(uncertainty=True)

    sig = np.zeros(sI.G.shape[0])
    for i in range(sig.size):
        Gi = sI.G[i, :].todense()
        sig[i] = (Gi).dot(sI.CpH).dot(Gi.T)    

    # Compute diagonal elements efficiently
    sig = np.zeros(sI.G.shape[0])

    # Compute (G @ Cp) column by column
    for i in tqdm(range(sI.CpH.shape[0]), total=sig.size):
        Gi = sI.G[:, i]  # Extract i-th column of G (sparse)
        Ci = sI.CpH[i, :]  # i-th row of Cp (dense)

        # Compute the diagonal contribution
        sig += np.array(Gi.multiply(sI.G @ Ci).sum(axis=1)).flatten()



    sI.CpH
    sI.G[0, :].todense()
    sI.CpH.dot(sI.G[0, :].todense().T)
    (sI.G[0, :].todense()).dot(sI.CpH).dot(sI.G[0, :].todense().T)

    sI.G[0, :].dot(sI.CpH).dot(sI.G[0, :].T)

    
    



#%%
l = 1e0

f = ~(np.isnan(sI.H) | np.isinf(sI.H) | np.isnan(sI.Hu) | np.isinf(sI.Hu)).flatten()

G = sI.G[f, :]
d = sI.H.flatten()[f]
C = sI.Hu.flatten()[f]


def make_model(G, d, C, l=0):
    Cinv = 1 / C  # Element-wise inverse
    GTCinv = G.T.multiply(Cinv)  # Sparse multiplication

    GTG = (GTCinv @ G).todense()  # Stays sparse
    #reg_term = l * np.median(GTG.diagonal()) * sp.eye(GTG.shape[0], format='csr')
    reg_term = l * np.median(np.diag(GTG)) * np.eye(GTG.shape[0])
    GTG += reg_term  # Sparse regularization

    GTd = GTCinv @ d  # Stays dense

    # Solve for Cp efficiently (Cp = GTG^{-1})
    Cp = scipy.linalg.lstsq(GTG, np.eye(GTG.shape[0]), lapack_driver='gelsy')[0]  # Uses sparse solver

    # Solve for m (m = Cp @ GTd)
    m = Cp.dot(GTd)

    return m, Cp

print('3')
f = ~(np.isnan(self.H) | np.isinf(self.H)).flatten()
self.mH, self.CpH = make_model(self.G[f, :], self.H.flatten()[f], 
                              self.Hu.flatten()[f], self.lH)
print('4')
f = ~(np.isnan(self.P) | np.isinf(self.P)).flatten()
self.mP, self.CpP = make_model(self.G[f, :], self.P.flatten()[f], 
                              self.Pu.flatten()[f], self.lP)































