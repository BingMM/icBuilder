#%% Imports

import xarray as xr
from icreader import load as icload
from pathlib import Path
from datetime import timedelta
from tqdm import tqdm
import pandas as pd
import numpy as np

#%% Data paths

dmsp_path = Path('/home/bing/Dropbox/work/data/dmsp/dmsp_ssj_yearly')
dmsp_files = list(dmsp_path.iterdir())

image_path = Path('/home/bing/Dropbox/work/data/IMAGE_FUV/precipitation_IR_P2')
img_files = sorted(image_path.iterdir())

#%% Load CS grid

grid = icload(img_files[0]).grid

#%% Load all dmsp data

def load_satellite(files, satellite):
    sat_files = sorted(f for f in files if f.name.startswith(f"dmsp_{satellite}_"))
    _dat = xr.concat([xr.open_dataset(f) for f in sat_files], dim="time")
    f = grid.ingrid(_dat['mlt'].values*15, _dat['mlat'])
    return _dat.isel(time=f)

dmsp_sats = ['f12', 'f13', 'f14', 'f15']
dmsp_dat = {}
for dmsp_sat in dmsp_sats:
    dmsp_dat[dmsp_sat] = load_satellite(dmsp_files, dmsp_sat)

#%% Run through all image orbits

samples = []

# Loop over all image orbits
for img_file in tqdm(img_files, total=len(img_files)):
    orbit = int(img_file.stem.split('_')[1])
    img_dat = icload(img_file)
    img_t_start, img_t_stop = img_dat.time[0], img_dat.time[-1] # Is the timestamp centered in the window??
    
    wic_file = img_file.parent.parent / 'binned/wic' / img_file.name
    wic_dat = icload(wic_file)
    
    f = np.any(np.abs(wic_dat.time[:, None] - img_dat.time[None, :]) <= np.timedelta64(5, "s"), axis=1)
    DZA = wic_dat.dza[f]
        
    # Loop over every dmsp sat
    for dmsp_sat, _dmsp_dat in dmsp_dat.items():
        subset = _dmsp_dat.sel(time=slice(img_t_start, img_t_stop))    
    
        if subset.sizes["time"] == 0:
            continue
    
        
        for k, t in enumerate(img_dat.time):
            t_frame_start, t_frame_end = t - timedelta(seconds=60), t + timedelta(seconds=60)
            frame_subset = subset.sel(time=slice(t_frame_start, t_frame_end))
            
            if frame_subset.sizes["time"] == 0:
                continue
            
            dmsp_time = frame_subset['time'].values
            dmsp_mlt = frame_subset['mlt'].values
            dmsp_mlat = frame_subset['mlat'].values
            dmsp_E = frame_subset['electron_mean_energy'].values
            
            i, j = grid.bin_index(dmsp_mlt*15, dmsp_mlat)
            
            img_wic = img_dat.wic_corrected[k, i, j]
            img_s13 = img_dat.si13_corrected[k, i, j]
            img_E = img_dat.E0[k, i, j]
            img_R = img_dat.R[k, i, j]
            wic_DZA = DZA[k, i, j]
            div_R = np.divide(img_wic, img_s13, 
                              out=np.full_like(img_wic, np.nan, dtype=float), 
                              where=img_s13 != 0)
            w = img_dat.w[k, i, j]
            
            samples.append(pd.DataFrame({"orbit": orbit,
                                         "frame_id": k,
                                         "dmsp_sat": dmsp_sat,
                                         "img_time": np.repeat(t, len(dmsp_time)),
                                         "dmsp_time": dmsp_time,
                                         "dmsp_mlt": dmsp_mlt,
                                         "dmsp_mlat": dmsp_mlat,
                                         "dmsp_E": dmsp_E,
                                         "img_wic": img_wic,
                                         "img_s13": img_s13,
                                         "img_R": img_R,
                                         "div_R": div_R,
                                         "img_E": img_E,
                                         "wic_dza": wic_DZA,
                                         "weight": w}))

df = pd.concat(samples, ignore_index=True)

#%%

ds = df.to_xarray()
ds.to_netcdf("/home/bing/Dropbox/work/code/repos/icBuilder/data/matches.nc")
