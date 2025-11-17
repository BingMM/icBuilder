#%%

import os
from os.path import join as pjoin
import fuvpy as fuv
import pandas as pd
import numpy as np
from multiprocessing import Pool

#%% What data to process

do_wic = True
do_s12 = True
do_s13 = True

parallel = False

print(f'Settings:\n WIC: {do_wic}\n SI12: {do_s12}\n SI13: {do_s13}\n Parallel: {parallel}')

#%% Base

#base = '/Home/siv32/mih008/repos/icBuilder/example_data/'
base = '/home/bing/Dropbox/work/code/repos/icBuilder/example_data/'

print(f'Base set to {base}')

#%% Import orbit files file 

print('Reading h5 files')
wicfiles = pd.read_hdf(pjoin(base, 'wicfiles.h5'), key='data')
s12files = pd.read_hdf(pjoin(base, 's12files.h5'), key='data')
s13files = pd.read_hdf(pjoin(base, 's13files.h5'), key='data')

#%%

def process_single_orbit(orbit, files, inpath, outpath, reflat, file_prefix):
    try:
        print(f'Starting on orbit {orbit}')
        file_list = (inpath + files.loc[files['orbit'] == orbit, 'filename']).tolist()
        s = fuv.read_idl(file_list, dzalim=75, reflat=reflat)
        s = s.sel(date=s.hemisphere.date[s.hemisphere == 'north'])
        s = s.assign({'t_start': np.datetime_as_string(s['date'][0], unit='s')})

        if np.all(np.isnan(s['mlat'].values)):
            print(f'Skipping orbit {orbit}, no data')
            return (orbit, 0)

        s = fuv.backgroundmodel_BS(s, sKnots=[-3.5, -0.25, 0, 0.25, 1.5, 3.5],
                                   stop=0.01, n_tKnots=5, tukeyVal=5, dampingVal=1e-3)
        s = fuv.backgroundmodel_SH(s, 4, 4, n_tKnots=5,
                                   stop=0.01, tukeyVal=5, dampingVal=1e-4)

        outfile = os.path.join(outpath, f"{file_prefix}_or{str(orbit).zfill(4)}.nc")
        encoding = {var: 
                    {"zlib": True, "complevel": 4}
                    for var in s.data_vars
                    }
        s.to_netcdf(outfile, format="NETCDF4", encoding=encoding)

        return (orbit, 1)

    except Exception as e:
        print(f'{file_prefix} : {orbit} : failed with error {e}')
        return (orbit, -1)

def background_removal_parallel(files, inpath, outpath, reflat=False):
    file_prefix = files['filename'].iloc[0][:3]
    orbits = files['orbit'].unique()
    args_list = [(orbit, files, inpath, outpath, reflat, file_prefix) for orbit in orbits]

    with Pool(10) as pool: # 500 GB RAM. Max 33 GB per orbit, I think. Max 15 process
        results = pool.starmap(process_single_orbit, args_list)

    return np.array(results)

def background_removal_serial(files, inpath, outpath, reflat=False):
    file_prefix = files['filename'].iloc[0][:3]
    orbits = files['orbit'].unique()
    results = []

    for orbit in orbits:
        result = process_single_orbit(orbit, files, inpath, outpath, reflat, file_prefix)
        results.append(result)

    return np.array(results)

def background_removal(files, inpath, outpath, reflat=False, parallel=True):
    """
    Background removal per orbit

    Parameters
    ----------
    files : DataFrame
        Must contain columns 'orbit' and 'filename'
    inpath : str
        Path to input files
    outpath : str
        Path to save output netCDFs
    reflat : bool
        Whether to reflatten images
    parallel : bool
        If True, use multiprocessing. If False, run serially.

    Returns
    -------
    avail_orbit : np.ndarray
        2D array with orbit numbers and status (0 = no data, 1 = success, -1 = failure)
    """
    if parallel:
        return background_removal_parallel(files, inpath, outpath, reflat)
    else:
        return background_removal_serial(files, inpath, outpath, reflat)

#%% Run WIC
if do_wic:
    inpath  = base + 'wic_data/'
    outpath = base + 'wic/'

    print('Starting work on WIC')
    print('Pulling data from: ' + inpath)
    print('Offlaoding at: ' + outpath)
    avail_orbit = background_removal(wicfiles, inpath, outpath, reflat=True, parallel=parallel)
    np.save(outpath + '../wic_avail_orbit.npy', avail_orbit)

#%% Run s12
if do_s12:
    inpath  = base + 's12_data/'
    outpath = base + 's12/'

    print('Starting work on SI12')
    print('Pulling data from: ' + inpath)
    print('Offlaoding at: ' + outpath)
    avail_orbit = background_removal(s12files, inpath, outpath, parallel=parallel)
    np.save(outpath + '../s12_avail_orbit.npy', avail_orbit)

#%% Run s13
if do_s13:
    inpath  = base + 's13_data/'
    outpath = base + 's13/'

    print('Starting work on SI13')
    print('Pulling data from: ' + inpath)
    print('Offlaoding at: ' + outpath)

    avail_orbit = background_removal(s13files, inpath, outpath, parallel=parallel)
    np.save(outpath + '../s13_avail_orbit.npy', avail_orbit)


