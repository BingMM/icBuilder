#%%

import gzip
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import cdflib
from astropy.coordinates import EarthLocation
import astropy.units as u

#%%

RECORD_SIZE = 2292

def read_dm(filename):
    filename = Path(filename)

    opener = gzip.open if filename.suffix == ".gz" else open

    rows = []

    with opener(filename, "rb") as f:

        while True:
            record = f.read(RECORD_SIZE)

            if not record:
                break

            if len(record) != RECORD_SIZE:
                raise ValueError("Incomplete DM record")

            i = 0

            def get(n):
                nonlocal i
                value = int.from_bytes(record[i:i+n], "big", signed=False)
                i += n
                return value

            spacecraft = record[i:i+5].decode("ascii").strip()
            i += 5

            data_id = record[i:i+6].decode("ascii").strip()
            i += 6

            year   = get(2) + 1950
            doy    = get(2)
            hour   = get(1)
            minute = get(1)

            lat  = get(2) / 10 - 90
            lon  = get(2) / 10

            mlat = get(2) / 10 - 90
            mlt  = get(2) / 10
            mlon = get(2) / 10

            subsolar_lat = get(2) / 10 - 90
            subsolar_lon = get(2) / 10

            lat_110 = get(2) / 10 - 90
            lon_110 = get(2) / 10

            cgm_lat = get(2) / 10 - 90
            cgm_lon = get(2) / 10

            inv_lat = get(2) / 10

            alt_start = get(2)
            alt_end   = get(2)

            time = (
                datetime(year, 1, 1)
                + timedelta(days=doy - 1, hours=hour, minutes=minute)
            )

            rows.append({
                "time": time,
                "spacecraft": spacecraft,
                "lat": lat,
                "lon": lon,
                "mlat": mlat,
                "mlon": mlon,
                "mlt": mlt,
                "lat_110": lat_110,
                "lon_110": lon_110,
                "cgm_lat": cgm_lat,
                "cgm_lon": cgm_lon,
                "inv_lat": inv_lat,
                "alt_start_nm": alt_start,
                "alt_end_nm": alt_end,
            })

    return pd.DataFrame(rows)


#%%

filename = '/home/bing/Downloads/f15dm01oct21.dat.gz'

DMSP_FILE = '/home/bing/Dropbox/work/code/repos/icBuilder/example_data/dmsp/dmsp-f15_ssj_20011021_or_0968.nc'

df = read_dm(filename)


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

fig, axs = plt.subplots(3, 1, figsize=(24,10), sharex=True)

axs[0].plot(time, geod_lat, label='SSJ')
axs[0].plot(df.time, df.lat, label='SSIES')
axs[0].set_ylabel('Geodetic latitude [deg]')
axs[0].legend()

axs[1].plot(time, geod_lon)
axs[1].plot(df.time, df.lon)
axs[1].set_ylabel('Longitude [km]')

axs[2].plot(time, height)
axs[2].plot(df.time, df.alt_start_nm*1.852)
axs[2].set_ylabel('Altitude [km]')
axs[2].set_xlabel('Time')

plt.suptitle('SSJ and SSIES comparison', fontsize=20)

plt.savefig('/home/bing/Dropbox/work/code/repos/icBuilder/figures/debugging/coumans_2004/SSJ_SSIES_comparison.png', bbox_inches='tight')
axs[0].set_xlim(time[0], time[3600*4])
plt.savefig('/home/bing/Dropbox/work/code/repos/icBuilder/figures/debugging/coumans_2004/SSJ_SSIES_comparison_closeup.png', bbox_inches='tight')


















