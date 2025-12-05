#%% Import

from os.path import join as pjoin
from icreader import ConductanceImage
from icbuilder import SplineImage
from pathlib import Path
import argparse
import glob
import numpy as np
from tqdm import tqdm

#%% Input

def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ('yes', 'true', 't', '1'):
        return True
    elif v in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description="Generate Spline orbit files.")
parser.add_argument('--parallel', type=str2bool, default=False, help='Run in parallel (default False)')
parser.add_argument('--pool_size', type=int, default=96, help='Size of pool (default 10)')
parser.add_argument('--ncp', type=int, default=30, help='Amount of control points in spatial dimensions (default=30)')
parser.add_argument('--cpt_step', type=float, default=2.5, help='Minutes between control points in temporal dimension (default=2.5)')
parser.add_argument('--lH', type=float, default=-6, help='Log10 of Hall 0th T regularization (default=-6)')
parser.add_argument('--lP', type=float, default=-6, help='Log10 of Pedersen 0th T regularization (default=-6)')
parser.add_argument('--k', type=int, default=3, help='Spatial spline degree (default=3)')
parser.add_argument('--kt', type=int, default=2, help='Spatial spline degree (default=3)')
parser.add_argument('--wscaling', type=str2bool, default=True, help='Sacling regularizationg with BR weights (default True)')
parser.add_argument('--psamp', type=int, default=5000, help='Samples used to estimate posterior covariance (default=5000)')
parser.add_argument('--base', type=str,
                    default=str(pjoin(Path(__file__).resolve().parents[1], 'example_data')),
                    help='Base data directory')

parallel = parser.parse_args().parallel
pool_size = parser.parse_args().pool_size

ncp = parser.parse_args().ncp
cpt_step = parser.parse_args().cpt_step
k = parser.parse_args().k
kt = parser.parse_args().kt

lH = parser.parse_args().lH
lP = parser.parse_args().lP
wscaling = parser.parse_args().wscaling

psamp = parser.parse_args().psamp

base = parser.parse_args().base

p_in = pjoin(base, 'conductance')
p_out = pjoin(base, 'spline')

#%% Overview

print('Running code for Spline model generation:')

print('\n>>Paths<<')
print('Pulling data from: ', p_in)
print('Outputting to: ', p_out)

print('\n>>Spline<<')
print('ncp: ', ncp)
print('cpt step: ', cpt_step)
print('k: ', k)
print('kt: ', kt)

print('\n>>Regularization<<')
print('lH: ', lH)
print('lP: ', lP)
print('wscaling: ', wscaling)

print('\n>>Operation mode<<')
print('parallel: ', parallel)
print('pool size: ', pool_size)
print('posterior samples: ', psamp)

#%% Get orbits

o = np.array([int(o[-7:-3]) for o in sorted(glob.glob(pjoin(p_in, '*.nc')))])

#%% Loop over all orbits

loop = tqdm(o, total=o.size, desc='Looping over all orbits')
for orbit in loop:
    
    # Input and output file
    orbit_file = f'or_{str(orbit).zfill(4)}.nc'    
    conductance_file = pjoin(p_in, orbit_file)
    spline_file = pjoin(base, orbit_file)
    
    # Load conductance image
    cI = ConductanceImage(conductance_file)
    
    # Initiate spline image
    sI = SplineImage(cI, # conductance image
                     ncp=ncp, k=k, # spatial spline
                     cpt_step=cpt_step, kt=kt, # temporal spline
                     lH=lH, lP=lP, wscaling=wscaling, # regularization
                     psamp=psamp # Uncertainty approximator
                     )

    # Save spline image
    sI.to_nc(spline_file)
