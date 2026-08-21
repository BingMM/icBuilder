#%%
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import cdflib
from astropy.coordinates import EarthLocation
import astropy.units as u

#%%

TIME_VARIABLES = [
    "SC_GEOCENTRIC_LAT",
    "SC_GEOCENTRIC_LON",
    "SC_GEOCENTRIC_R",
    "SC_AACGM_LAT",
    "SC_AACGM_LON",
    "SC_AACGM_LTIME",
    "ELE_COUNTS_OBS",
    "ELE_COUNTS_BKG",
    "ELE_DIFF_ENERGY_FLUX",
    "ELE_DIFF_ENERGY_FLUX_STD",
    "ELE_TOTAL_ENERGY_FLUX",
    "ELE_TOTAL_ENERGY_FLUX_STD",
    "ELE_AVG_ENERGY",
    "ELE_AVG_ENERGY_STD",
    "ION_COUNTS_OBS",
    "ION_COUNTS_BKG",
    "ION_DIFF_ENERGY_FLUX",
    "ION_DIFF_ENERGY_FLUX_STD",
    "ION_TOTAL_ENERGY_FLUX",
    "ION_TOTAL_ENERGY_FLUX_STD",
    "ION_AVG_ENERGY",
    "ION_AVG_ENERGY_STD",
]

filename = '/home/bing/Dropbox/work/code/repos/icBuilder/example_data/dmsp/raw/dmsp-f15_ssj_precipitating-electrons-ions_20011021_v1.1.2.cdf'

cdf = cdflib.CDF(filename)

time = cdflib.cdfepoch.to_datetime(cdf.varget("Epoch"))
geoc_lat  = cdf.varget("SC_GEOCENTRIC_LAT")
geoc_lon  = cdf.varget("SC_GEOCENTRIC_LON")
radius  = cdf.varget("SC_GEOCENTRIC_R")

latitude = np.deg2rad(geoc_lat)
longitude = np.deg2rad(geoc_lon)

x = radius * np.cos(latitude) * np.cos(longitude)
y = radius * np.cos(latitude) * np.sin(longitude)
z = radius * np.sin(latitude)

spacecraft = EarthLocation.from_geocentric(x * u.km, y * u.km, z * u.km)
geod_lat = spacecraft.lat.deg
geod_lon = spacecraft.lon.deg%360
height = spacecraft.height.to_value(u.km)

#%%

import xarray as xr

fn = '/home/bing/Downloads/dms_20011021_15e.001.nc'

ds = xr.open_dataset(fn)

geodetic_lat = ds.data_vars['gdlat'].values
geodetic_lon = ds.data_vars['glon'].values%360
geodetic_alt = ds.data_vars['gdalt'].values

time2 = [datetime(1970, 1, 1) + timedelta(seconds=t) for t in ds.coords['timestamps'].values]

#%%

fig, axs = plt.subplots(3, 1, figsize=(24,10), sharex=True)

axs[0].plot(time, geod_lat, label='SSJ')
axs[0].plot(time2, geodetic_lat, label='SSJ2')
axs[0].set_ylabel('Geodetic latitude [deg]')
axs[0].legend()

axs[1].plot(time, geod_lon)
axs[1].plot(time2, geodetic_lon)
axs[1].set_ylabel('Longitude [km]')

axs[2].plot(time, height)
axs[2].plot(time2, geodetic_alt)
axs[2].set_ylabel('Altitude [km]')
axs[2].set_xlabel('Time')

plt.suptitle('SSJ data source comparison', fontsize=20)

plt.savefig('/home/bing/Dropbox/work/code/repos/icBuilder/figures/debugging/coumans_2004/SSJ_data_source_comparison.png', bbox_inches='tight')
axs[0].set_xlim(time[0], time[3600*4])
plt.savefig('/home/bing/Dropbox/work/code/repos/icBuilder/figures/debugging/coumans_2004/SSJ_data_source_comparison_closeup.png', bbox_inches='tight')


















