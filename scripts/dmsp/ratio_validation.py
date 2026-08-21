#%% Imports

import xarray as xr
from icreader import load as icload
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
from tqdm import tqdm

#%% Load data

data = xr.open_dataset("/home/bing/Dropbox/work/code/repos/icBuilder/data/matches.nc", engine="netcdf4")
df_all = data.to_pandas()

#%% load grid

image_path = Path('/home/bing/Dropbox/work/data/IMAGE_FUV/precipitation_IR_P2')
grid = icload(sorted(image_path.iterdir())[0]).grid

#%% Load selected orbit filter

selected_orbits_path = Path('/home/bing/Dropbox/work/code/repos/icAnalyzer/data/good_orbit_segments.csv')
orbit_segments = pd.read_csv(selected_orbits_path, header=None)

orbit_segments.columns = ['orbit', 'shit_start', 'shit_stop', 'good_start', 'good_stop']
orbit_segments = orbit_segments.loc[orbit_segments['orbit'] < 503, :]

keep = []
for orbit in orbit_segments['orbit']:
    df_orbit = df_all.loc[df_all['orbit'] == orbit]
    if df_orbit.shape[0] == 0: continue
    
    start = orbit_segments.loc[orbit_segments['orbit']==orbit].iloc[0]['good_start']
    stop = orbit_segments.loc[orbit_segments['orbit']==orbit].iloc[0]['good_stop']
    if stop == 0: continue    
    if np.isnan(start): start = 0 # New
    if np.isnan(stop): stop = df_orbit['frame_id'].max()
    
    keep.extend(list(df_orbit.loc[(df_orbit['frame_id']>=start) & (df_orbit['frame_id']<=stop)].index))

df = df_all.iloc[keep, :]

#%% Load selected orbit filter

selected_orbits_path = Path('/home/bing/Dropbox/work/code/repos/icBuilder/data/dmsp_frame_annotations.csv')
good_frames = pd.read_csv(selected_orbits_path)
good_frames['satellites'] = good_frames['satellites'].str.lower()

'''
keep = []
for i in tqdm(range(good_frames.shape[0]), total=good_frames.shape[0]):
    if good_frames['accepted'][i] != 1:
        continue
    orbit = good_frames['orbit'][i]
    frame_id = good_frames['frame_id'][i]
    sat = good_frames['satellites'][i]
    
    keep.extend(df_all.loc[(df_all['orbit'] == orbit) & (df_all['frame_id'] == frame_id) & (df_all['dmsp_sat'] == sat)].index)
'''

good = good_frames.loc[good_frames["accepted"] == 1, ["orbit", "frame_id", "satellites"]]

keys_good = pd.MultiIndex.from_frame(good.rename(columns={"satellites": "dmsp_sat"}))

keys_all = pd.MultiIndex.from_frame(df_all[["orbit", "frame_id", "dmsp_sat"]])

keep = df_all.index[keys_all.isin(keys_good)]

df = df_all.iloc[keep, :]

#%%

f = np.isnan(df['img_R']) | (df['img_R'] == 0) | np.isnan(df['dmsp_E']) | np.isnan(df['wic_dza'])
x = df.loc[~f, ['dmsp_E', 'img_R', 'wic_dza', 'dmsp_mlat', 'dmsp_mlt', 'weight', 'img_wic', 'img_s13']]

#%%

xx = x[['dmsp_E', 'img_R', 'wic_dza']]

plt.figure(figsize=(15, 9))
plt.hist(xx['img_R'], bins=1000)
plt.title('Ratio')

plt.figure(figsize=(15, 9))
plt.hist(xx['dmsp_E'], bins=100)
plt.title('E')

plt.figure(figsize=(15, 9))
plt.hist(xx['wic_dza'], bins=100)
plt.title('DZA')

plt.figure(figsize=(15,9))
plt.scatter(xx['img_R'], xx['dmsp_E'])
plt.xlabel('Ratio')
plt.ylabel('E')
            
plt.figure(figsize=(15, 15))
plt.hist2d(xx['img_R'], xx['dmsp_E'], bins=100)
plt.xlabel('Ratio')
plt.ylabel('E')

plt.figure(figsize=(15, 15))
plt.hist2d(xx['img_R'], xx['dmsp_E'], bins=100, norm=LogNorm())
plt.xlabel('Ratio')
plt.ylabel('E')

#%%

f = x['img_R'] <= 300

xx = x.loc[f, ['dmsp_E', 'img_R', 'wic_dza']]

plt.figure(figsize=(15, 9))
plt.hist(xx['img_R'], bins=1000)
plt.title('Ratio')

plt.figure(figsize=(15, 9))
plt.hist(xx['dmsp_E'], bins=100)
plt.title('E')

plt.figure(figsize=(15, 9))
plt.hist(xx['wic_dza'], bins=100)
plt.title('DZA')

plt.figure(figsize=(15,9))
plt.scatter(xx['img_R'], xx['dmsp_E'])
plt.xlabel('Ratio')
plt.ylabel('E')
            
plt.figure(figsize=(15, 15))
plt.hist2d(xx['img_R'], xx['dmsp_E'], bins=100)
plt.xlabel('Ratio')
plt.ylabel('E')

plt.figure(figsize=(15, 15))
plt.hist2d(xx['img_R'], xx['dmsp_E'], bins=100, norm=LogNorm())
plt.xlabel('Ratio')
plt.ylabel('E')

#%%

f = (x['img_R'] <= 300) & (x['wic_dza'] < 40) & (x['dmsp_E'] > .5)

xx = x.loc[f, ['dmsp_E', 'img_R', 'wic_dza']]

plt.figure(figsize=(15, 9))
plt.hist(xx['img_R'], bins=1000)
plt.title('Ratio')

plt.figure(figsize=(15, 9))
plt.hist(xx['dmsp_E'], bins=100)
plt.title('E')

plt.figure(figsize=(15, 9))
plt.hist(xx['wic_dza'], bins=100)
plt.title('DZA')

plt.figure(figsize=(15,9))
plt.scatter(xx['img_R'], xx['dmsp_E'])
plt.xlabel('Ratio')
plt.ylabel('E')
            
plt.figure(figsize=(15, 15))
plt.hist2d(xx['img_R'], xx['dmsp_E'], bins=100)
plt.xlabel('Ratio')
plt.ylabel('E')

plt.figure(figsize=(15, 15))
plt.hist2d(xx['img_R'], xx['dmsp_E'], bins=100, norm=LogNorm())
plt.xlabel('Ratio')
plt.ylabel('E')
            
#%%

#f = (x['img_R'] <= 250) & ((x['dmsp_mlt'] >= 18) | (x['dmsp_mlt'] <= 6))
#f = (x['img_R'] <= 250) & ((x['dmsp_mlt'] >= 18) | (x['dmsp_mlt'] <= 6))
f = ((x['dmsp_mlt'] >= 18) | (x['dmsp_mlt'] <= 6))
#f = (x['img_R'] <= 500) & (x['wic_dza'] < 90) & (x['dmsp_E'] > .0) & ((x['dmsp_mlt'] >= 18) | (x['dmsp_mlt'] <= 6)) & ((x['dmsp_mlat'] <= 80) & (x['dmsp_mlat'] > 55))

xx = x.loc[f, ['dmsp_E', 'img_R', 'wic_dza']]

plt.figure(figsize=(15, 9))
plt.hist(xx['img_R'], bins=1000)
plt.title('Ratio')

plt.figure(figsize=(15, 9))
plt.hist(xx['dmsp_E'], bins=100)
plt.title('E')

plt.figure(figsize=(15, 9))
plt.hist(xx['wic_dza'], bins=100)
plt.title('DZA')

plt.figure(figsize=(15,9))
plt.scatter(xx['img_R'], xx['dmsp_E'])
plt.xlabel('Ratio')
plt.ylabel('E')
            
plt.figure(figsize=(15, 15))
plt.hist2d(xx['img_R'], xx['dmsp_E'], bins=100)
plt.xlabel('Ratio')
plt.ylabel('E')

plt.figure(figsize=(15, 15))
plt.hist2d(xx['img_R'], xx['dmsp_E'], bins=100, norm=LogNorm())
plt.xlabel('Ratio')
plt.ylabel('E')

#%%

#f = (x['img_R'] <= 1000) & (x['wic_dza'] < 40) & (x['dmsp_E'] > 1) & ((x['dmsp_mlt'] >= 18) | (x['dmsp_mlt'] <= 6)) & ((x['dmsp_mlat'] <= 80) & (x['dmsp_mlat'] > 55))
f = (x['weight'] < .5) & (x['img_R'] <= 150) & (x['wic_dza'] < 90) & ((x['dmsp_mlt'] >= 18) | (x['dmsp_mlt'] <= 6)) & (x['dmsp_E'] > 0)  & (x['dmsp_E'] <= 15)

xx = x.loc[f, ['dmsp_E', 'img_R', 'wic_dza', 'img_wic', 'weight', 'img_s13']]

plt.figure(figsize=(15, 9))
plt.hist(xx['img_R'], bins=1000)
plt.title('Ratio')

plt.figure(figsize=(15, 9))
plt.hist(xx['dmsp_E'], bins=100)
plt.title('E')

plt.figure(figsize=(15, 9))
plt.hist(xx['wic_dza'], bins=100)
plt.title('DZA')

plt.figure(figsize=(15,9))
plt.scatter(xx['img_R'], xx['dmsp_E'])
plt.xlabel('Ratio')
plt.ylabel('E')
            
plt.figure(figsize=(15, 15))
plt.hist2d(xx['img_R'], xx['dmsp_E'], bins=100)
plt.xlabel('Ratio')
plt.ylabel('E')

plt.figure(figsize=(15, 15))
plt.hist2d(xx['img_R'], xx['dmsp_E'], bins=100, norm=LogNorm())
#plt.hist2d(xx['img_wic'], xx['dmsp_E'], bins=100, norm=LogNorm())
#plt.hist2d(xx['weight'], xx['dmsp_E'], bins=100, norm=LogNorm())
plt.xlabel('Ratio')
plt.ylabel('E')


#%%

#X = xx["img_R"].values
#X = xx["weight"].values
#X = xx["img_wic"].values
X = xx["img_s13"].values
Y = xx["dmsp_E"].values

bins = 50
quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

plt.figure(figsize=(15, 15))

_, xedges, _, _ = plt.hist2d(
    X, Y,
    bins=bins,
    norm=LogNorm()
)

xcenters = (xedges[:-1] + xedges[1:]) / 2
ibin = np.digitize(X, xedges) - 1

for q in quantiles:
    yq = [
        np.nanquantile(Y[ibin == i], q) if np.any(ibin == i) else np.nan
        for i in range(len(xcenters))
    ]
    plt.plot(xcenters, yq, label=f"{q:.0%}")

plt.xlabel("Ratio")
plt.ylabel("E")
plt.legend()
plt.colorbar(label="Count")

#plt.xlim(0, 200)
#plt.ylim(0, 15)

#%%            


x = xx["img_R"].values
y = xx["dmsp_E"].values
z = xx["wic_dza"].values

# Remove NaNs
f = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
x, y, z = x[f], y[f], z[f]

# 3D histogram
H, edges = np.histogramdd((x, y, z), bins=(30, 30, 30))

xc = (edges[0][:-1] + edges[0][1:]) / 2
yc = (edges[1][:-1] + edges[1][1:]) / 2
zc = (edges[2][:-1] + edges[2][1:]) / 2

X, Y, Z = np.meshgrid(xc, yc, zc, indexing="ij")

# Only plot occupied bins
f = H > 0

fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")

p = ax.scatter(
    X[f], Y[f], Z[f],
    c=H[f],
    norm=LogNorm(),
    s=20
)

ax.set_xlabel("Ratio")
ax.set_ylabel("E")
ax.set_zlabel("New variable")

fig.colorbar(p, ax=ax, label="Count")         
        
#%%

sat = 'f13'
plt.ioff()
for orbit in df['orbit'].unique():
    
    f = (df['orbit'] == orbit) & (df['dmsp_sat'] == sat)
    time = df['dmsp_time'][f]
    E = df['dmsp_E'][f]
    R = df['img_R'][f] + 0
    dza = df['wic_dza'][f]
    
    f = R > 250
    
    fig, axs = plt.subplots(3, 1, figsize=(20, 13), sharex=True)
    axs[0].plot(time, E)
    axs[1].plot(time, R)
    axs[2].plot(time, dza)
    axs[2].set_xlabel('Time')
    axs[0].set_ylabel('dmsp energy')
    axs[1].set_ylabel('image ratio')
    axs[2].set_ylabel('wic dza')
    plt.suptitle(f'{orbit} : {sat}')
    axs[1].set_ylim(-5, 250)
    axs[1].plot(time[f], np.zeros(sum(f)), '.')
    axs[1].plot([time.iloc[0], time.iloc[-1]], [150]*2, color='red')
    
    plt.savefig(f'/home/bing/Dropbox/work/temp_storage/dmsp_test/{sat}/{orbit:04d}.png', bbox_inches='tight')
    plt.close()


#%%


def split_on_time_gaps(time, dt=np.timedelta64(30, "m")):
    time = np.asarray(time)

    if len(time) == 0:
        return []

    breaks = np.where(np.diff(time) > dt)[0] + 1

    return [x.tolist() for x in np.split(np.arange(len(time)), breaks)]

sat = 'f15'
plt.ioff()
for orbit in df.loc[df['dmsp_sat'] == sat, 'orbit'].unique():
    
    f = (df['orbit'] == orbit) & (df['dmsp_sat'] == sat)
    time = df['dmsp_time'][f]
    E = df['dmsp_E'][f]
    img_E = df['img_E'][f]
    R = df['img_R'][f]
    div_R = df['div_R'][f]
    dza = df['wic_dza'][f]
    mlt = df['dmsp_mlt'][f]
    wic = df['img_wic'][f]
    s13 = df['img_s13'][f]
    
    #segments = split_on_nans(dza)
    segments = split_on_time_gaps(time.values)
    
    if len(segments) == 0:
        continue
    
    for i, segment in enumerate(segments):
        time_ = time.iloc[segment]
        E_ = E.iloc[segment]
        img_E_ = img_E.iloc[segment]
        R_ = R.iloc[segment]
        div_R_ = div_R.iloc[segment]
        dza_ = dza.iloc[segment]
        mlt_ = mlt.iloc[segment]
        wic_ = wic.iloc[segment]
        s13_ = s13.iloc[segment]
                
        f = R_ > 250
        
        fig, axs = plt.subplots(4, 1, figsize=(20, 13), sharex=True)
        axs[0].plot(time_, E_, label='dmsp E')
        axs[0].plot(time_, img_E_, label='img_E')
        axs[0].legend()
        axs[1].plot(time_, wic_)
        axs[1].set_ylabel('wic')
        
        axs1 = axs[1].twinx()
        axs1.plot(time_, s13_, color='k')
        axs1.set_ylabel('s13')
        
        axs[2].plot(time_, R_, label='IR')
        axs[2].plot(time_, div_R_, label='wic/si13')
        axs[2].legend()
        axs[3].plot(time_, dza_)
        axs[3].set_xlabel('Time')
        axs[0].set_ylabel('dmsp energy')
        axs[2].set_ylabel('image ratio')
        axs[3].set_ylabel('wic dza')
        axs3 = axs[3].twinx()
        axs3.plot(time_, mlt_, color='k')
        axs3.set_ylabel('mlt')
        axs3.set_ylim(-1, 25)
        axs3.fill_between(time_, [6]*time_.size, [18]*time_.size, color='orange', alpha=.5)
        plt.suptitle(f'{time_.iloc[0]} : {orbit} : {sat}')
        axs[2].set_ylim(-5, 250)
        axs[2].plot(time_[f], np.zeros(sum(f)), '.')
        axs[2].plot([time_.iloc[0], time_.iloc[-1]], [150]*2, color='red')
        
        plt.savefig(f'/home/bing/Dropbox/work/temp_storage/dmsp_test_segment/{sat}/{orbit:04d}_{i+1}.png', bbox_inches='tight')
        plt.close()    
    
#%%

img = icload('/home/bing/Dropbox/work/data/IMAGE_FUV/precipitation_IR_P2/or_0546.nc')
for i, wic in enumerate(img.wic_corrected):
    q = df.loc[df['img_time']==img.time[i]]
    if q.size == 0:
        continue
    
    xi, eta = img.grid.projection.geo2cube(q['dmsp_mlt']*15, q['dmsp_mlat'])
    
    plt.figure()
    plt.contourf(img.grid.eta, img.grid.xi, wic)
    plt.plot(eta, xi, '.')
    plt.title(img.time[i])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
