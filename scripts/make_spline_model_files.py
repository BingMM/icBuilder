#%% Import

from os.path import join as pjoin
from icreader import ConductanceImage
from icbuilder import SplineImage
from pathlib import Path
import argparse
import glob
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

#%% Worker Function

def process_single_orbit(orbit, p_in, spline_out, factor_out, ncp, k, cpt_step, kt, lH, lP, wscaling, psamp):
    """
    Worker function to process a single orbit. 
    Arguments must be passed explicitly for multiprocessing compatibility.
    """
    # Input and output file
    orbit_file = f'or_{str(orbit).zfill(4)}.nc'    
    conductance_file = pjoin(p_in, orbit_file)
    spline_file = pjoin(spline_out, orbit_file)
    factor_file = pjoin(factor_out, orbit_file)
    
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
    sI.spline_to_nc(spline_file)
    sI.factor_to_nc(factor_file)
    
    return orbit # Return value is useful for tracking progress

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
parser.add_argument('--pool_size', type=int, default=10, help='Size of pool (default 10)')
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

# Parse args once to avoid repeated calls
args = parser.parse_args()

p_in = pjoin(args.base, 'conductance')
spline_out = pjoin(args.base, 'spline')
factor_out = pjoin(args.base, 'factor')

#%% Overview

print('Running code for Spline model generation:')

print('\n>>Paths<<')
print('Pulling data from: ', p_in)
print('Outputting splines to: ', spline_out)
print('Outputting factor to: ', factor_out)

print('\n>>Spline<<')
print('ncp: ', args.ncp)
print('cpt step: ', args.cpt_step)
print('k: ', args.k)
print('kt: ', args.kt)

print('\n>>Regularization<<')
print('lH: ', args.lH)
print('lP: ', args.lP)
print('wscaling: ', args.wscaling)

print('\n>>Operation mode<<')
print('parallel: ', args.parallel)
print('pool size: ', args.pool_size)
print('posterior samples: ', args.psamp)

#%% Get orbits

o = np.array([int(o[-7:-3]) for o in sorted(glob.glob(pjoin(p_in, '*.nc')))])

#%% Processing Logic

# Create a partial function with all constant arguments frozen
# This allows us to map only the 'orbit' list to the function
worker = partial(process_single_orbit, 
                 p_in=p_in, 
                 spline_out=spline_out, 
                 factor_out=factor_out,
                 ncp=args.ncp, k=args.k, 
                 cpt_step=args.cpt_step, kt=args.kt, 
                 lH=args.lH, lP=args.lP, wscaling=args.wscaling, 
                 psamp=args.psamp)

if args.parallel:
    print(f"Starting parallel processing with {args.pool_size} workers.")
    # imap_unordered is usually slightly faster if order doesn't matter
    # imap allows tqdm to update as tasks finish
    with Pool(args.pool_size) as p:
        list(tqdm(p.imap(worker, o), total=len(o), desc='Processing orbits (Parallel)'))
else:
    print("Starting serial processing.")
    # Standard serial loop
    for orbit in tqdm(o, desc='Processing orbits (Serial)'):
        worker(orbit)