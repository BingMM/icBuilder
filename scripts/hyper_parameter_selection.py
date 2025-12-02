#%% Import

from os.path import join as pjoin
from icreader import ConductanceImage
from icbuilder import SplineImage
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

#%% Paths

base = '/home/bing/Dropbox/work/code/repos/icBuilder/example_data/'
orbit = 478
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























