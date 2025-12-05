#%% Import

from os.path import join as pjoin
from icreader import ConductanceImage
from icbuilder import SplineImage
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

#%%

events = np.array([[179, 24, 157],
                   [209, 23, 200],
                   [218, 20, 270],
                   [221, 25, 145],
                   [234, 0, 274],
                   [239, 27, 268],
                   [318, 30, 265],
                   [320, 30, 220],
                   [338, 44, 146],
                   [346, 24, 190]])

#%% Paths

base = '/home/bing/Dropbox/work/data/conductance/'

#%%

l1 = np.linspace(-10, 3, 50)

rnorms = []
mnorms = []
for orbit, start, end in zip(events[:, 0], events[:, 1], events[:, 2]):
    orbit_file = f'or_{str(orbit).zfill(4)}.nc'    
    conductance_file = pjoin(base, orbit_file)
    cI = ConductanceImage(conductance_file)
    cI.discard(interval=(start, end))
    sI = SplineImage(cI, ncp = 30, cpt_step = 2.5, lH = -3, lP=-1, wscaling=True, kt=2)

    mnorm = []
    rnorm = []
    for l1_ in tqdm(l1, total=l1.size):
        sI.reset_ev()
        sI.reset_model()
        sI.lH = l1_
        
        mnorm.append(sI.mH.T@sI.LTL@sI.mH)
        r = sI.H.flatten()[sI.solverH.f] - sI.pH.flatten()[sI.solverH.f]
        q = 1/sI.solverH.q[sI.solverH.f]**2
        rnorm.append(r.T@(q*r))

    rnorms.append(rnorm)
    mnorms.append(mnorm)

#%%

from kneed import KneeLocator

l_idx_opt = []
for rnorm, mnorm in zip(rnorms, mnorms):
    plt.figure(figsize=(10,10))
    knee = KneeLocator(np.log10(rnorm), np.log10(mnorm), curve='convex', direction='decreasing')
    opt_id = np.argmin(abs(rnorm - 10**knee.knee))
    l_idx_opt.append(opt_id)
    plt.loglog(rnorm, mnorm, '.-')
    plt.loglog(rnorm[opt_id], mnorm[opt_id], '.', markersize=15)

#%%




l_idx_opt = [6, 6, 6, 6, 7, 5, 6, 6, 5]





#%%

orbit_file = f'or_{str(orbit).zfill(4)}.nc'
conductance_file = pjoin(base, 'conductance', orbit_file)
spline_file = pjoin(base, 'spline', orbit_file)

#%% Load condutance image

cI = ConductanceImage(conductance_file)
cI.discard(interval=(50, 210))

#%% Make spline image

sI = SplineImage(cI, ncp = 30, cpt_step = 5, lH = -3, lP=-1, wscaling=True, kt=2)

#%%

l1 = np.linspace(-10, 4, 20)
mnorm = []
rnorm = []
for l1_ in tqdm(l1, total=l1.size):
    sI.reset_ev()
    sI.reset_model()
    sI.lH = l1_
    
    mnorm.append(sI.mH.T@sI.LTL@sI.mH)
    r = sI.H.flatten()[sI.solverH.f] - sI.pH.flatten()[sI.solverH.f]
    q = 1/sI.solverH.q[sI.solverH.f]**2
    rnorm.append(r.T@(q*r))
    
#%%

plt.figure(figsize=(20,20))
plt.loglog(rnorm, mnorm, '.-')

#%%

'''
30  3  -4.5
25  3  -4.1
20  3  -4.1
30  4  -4.1
25  4  -4.1
20  4  -4.1
30  5  
25  5  
20  5  
'''

#%%

plt.figure(figsize=(10,8))

orbit = 99
orbit_file = f'or_{str(orbit).zfill(4)}.nc'
conductance_file = pjoin(base, 'conductance', orbit_file)
cI = ConductanceImage(conductance_file)
cI.discard(interval=(50, 210))
plt.plot( (1 - np.sum(np.isnan(cI.H), axis=(1,2)) / (36*36)) * 100, label=f'orbit : {orbit}')

orbit = 478
orbit_file = f'or_{str(orbit).zfill(4)}.nc'
conductance_file = pjoin(base, 'conductance', orbit_file)
cI = ConductanceImage(conductance_file)
cI.discard(interval=(50, 210))
plt.plot( (1 - np.sum(np.isnan(cI.H), axis=(1,2)) / (36*36)) * 100, label=f'orbit : {orbit}')

plt.legend()
plt.xlabel('Frame')
plt.ylabel('Percent full')
plt.grid()























