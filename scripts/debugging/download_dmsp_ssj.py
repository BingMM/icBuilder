"""Download and reduce DMSP SSJ particle data over a time interval.

The first use is the DMSP F15 pass compared with IMAGE orbit 0968 in
Coumans et al. (2004), Figure 4. Daily CDF files are kept in ``raw`` and are
ignored by Git. An interval may cross midnight; every daily file touched by
the interval is downloaded and combined into one reduced NetCDF.
"""

#%% Imports

import argparse
import re
from pathlib import Path
from urllib.request import urlopen, urlretrieve

from apexpy import Apex
from astropy.coordinates import EarthLocation
import astropy.units as u
from cdflib import CDF, cdfepoch
import numpy as np
import xarray as xr


#%% DMSP archive

ARCHIVE = "https://cdaweb.gsfc.nasa.gov/pub/data/dmsp"

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


def archive_url(satellite, year):
    """Return the NASA directory containing one year of SSJ data."""

    return (
        f"{ARCHIVE}/dmsp{satellite}/ssj/"
        f"precipitating-electrons-ions/{year}/"
    )


def find_daily_file(satellite, date):
    """Find the newest archived file for one satellite and date."""

    directory = archive_url(satellite, date[:4])
    page = urlopen(directory).read().decode("utf-8")

    pattern = rf'href="(dmsp-{satellite}_ssj_[^"]*_{date.replace("-", "")}_v[^\"]+\.cdf)"'
    filenames = sorted(set(re.findall(pattern, page)))
    if not filenames:
        raise FileNotFoundError(f"No {satellite.upper()} SSJ file found for {date}")

    return directory + filenames[-1], filenames[-1]


def download_daily_file(satellite, date, raw_dir):
    """Download one daily CDF, or reuse it when it already exists."""

    date_compact = date.replace("-", "")
    local_files = sorted(raw_dir.glob(f"dmsp-{satellite}_ssj_*_{date_compact}_v*.cdf"))
    if local_files:
        output = local_files[-1]
        year = date[:4]
        url = archive_url(satellite, year) + output.name
        print(f"Using existing {output}")
        return output, url

    url, filename = find_daily_file(satellite, date)
    output = raw_dir / filename

    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    urlretrieve(url, output)

    return output, url


#%% Time selection

def dates_in_interval(start, end):
    """Return all UTC dates touched by an inclusive time interval."""

    first_day = np.datetime64(start, "D")
    last_day = np.datetime64(end, "D")
    days = np.arange(first_day, last_day + np.timedelta64(1, "D"))
    return [np.datetime_as_string(day) for day in days]


#%% Coordinate mapping

def map_to_image_height(times, geocentric_lat, geocentric_lon, radius, height=130):
    """Map the DMSP position along the magnetic field to the IMAGE shell."""

    latitude = np.deg2rad(geocentric_lat)
    longitude = np.deg2rad(geocentric_lon)

    x = radius * np.cos(latitude) * np.cos(longitude)
    y = radius * np.cos(latitude) * np.sin(longitude)
    z = radius * np.sin(latitude)

    spacecraft = EarthLocation.from_geocentric(x * u.km, y * u.km, z * u.km)
    spacecraft_lat = spacecraft.lat.deg
    spacecraft_lon = spacecraft.lon.deg
    spacecraft_height = spacecraft.height.to_value(u.km)

    python_times = times.astype("datetime64[us]").astype(object)
    apex = Apex(date=python_times[0])
    footprint_lat, footprint_lon, mapping_error = apex.map_to_height(
        spacecraft_lat,
        spacecraft_lon,
        spacecraft_height,
        height,
        precision=1e-5,
    )

    qd_lat, qd_lon = apex.geo2qd(footprint_lat, footprint_lon, height)
    mlt = np.array([
        apex.mlon2mlt(lon, time)
        for lon, time in zip(qd_lon, python_times)
    ])

    return {
        "spacecraft_geodetic_lat": spacecraft_lat,
        "spacecraft_geodetic_lon": spacecraft_lon,
        "spacecraft_height": spacecraft_height,
        "footprint_geodetic_lat": footprint_lat,
        "footprint_geodetic_lon": footprint_lon,
        "footprint_qd_lat": qd_lat,
        "footprint_qd_lon": qd_lon,
        "footprint_mlt": mlt,
        "footprint_mapping_error": mapping_error,
    }


#%% CDF reduction

def variable_attributes(cdf, name):
    """Keep the useful text metadata from one CDF variable."""

    source = cdf.varattsget(name)
    attributes = {}

    if "CATDESC" in source:
        attributes["long_name"] = str(source["CATDESC"])
    if "UNITS" in source:
        attributes["units"] = str(source["UNITS"])

    return attributes


def reduce_files(cdf_files, source_urls, output, min_mlat, start, end):
    """Combine daily files and retain one high-latitude time interval."""

    start = np.datetime64(start, "ns")
    end = np.datetime64(end, "ns")
    selected_times = []
    selected_values = {name: [] for name in TIME_VARIABLES}

    first_cdf = CDF(cdf_files[0])
    channel_energy = np.asarray(first_cdf.varget("CHANNEL_ENERGIES"), dtype=float)

    for cdf_file in cdf_files:
        cdf = CDF(cdf_file)
        epoch = np.asarray(cdf.varget("Epoch"))
        times = np.asarray(cdfepoch.to_datetime(epoch), dtype="datetime64[ns]")
        magnetic_latitude = np.asarray(cdf.varget("SC_AACGM_LAT"), dtype=float)

        keep = (
            (times >= start)
            & (times <= end)
            & (magnetic_latitude >= min_mlat)
        )
        if not np.any(keep):
            continue

        selected_times.append(times[keep])
        for name in TIME_VARIABLES:
            selected_values[name].append(np.asarray(cdf.varget(name))[keep])

    if not selected_times:
        raise ValueError("No DMSP samples satisfy the time and latitude selection")

    times = np.concatenate(selected_times)
    order = np.argsort(times)
    times = times[order]

    data = xr.Dataset(
        coords={
            "time": times,
            "energy_channel": np.arange(channel_energy.size),
            "channel_energy": ("energy_channel", channel_energy),
        }
    )
    data.channel_energy.attrs = variable_attributes(first_cdf, "CHANNEL_ENERGIES")

    for name in TIME_VARIABLES:
        values = np.concatenate(selected_values[name], axis=0)[order]
        dimensions = ("time", "energy_channel") if values.ndim == 2 else ("time",)
        data[name] = (dimensions, values)
        data[name].attrs = variable_attributes(first_cdf, name)

    mapped = map_to_image_height(
        data.time.values,
        data.SC_GEOCENTRIC_LAT.values,
        data.SC_GEOCENTRIC_LON.values,
        data.SC_GEOCENTRIC_R.values,
    )
    for name, values in mapped.items():
        data[name] = ("time", values)

    data.spacecraft_geodetic_lat.attrs = {"units": "deg", "long_name": "Spacecraft geodetic latitude"}
    data.spacecraft_geodetic_lon.attrs = {"units": "deg", "long_name": "Spacecraft geodetic longitude"}
    data.spacecraft_height.attrs = {"units": "km", "long_name": "Spacecraft geodetic height"}
    data.footprint_geodetic_lat.attrs = {"units": "deg", "long_name": "Field-line footprint geodetic latitude at 130 km"}
    data.footprint_geodetic_lon.attrs = {"units": "deg", "long_name": "Field-line footprint geodetic longitude at 130 km"}
    data.footprint_qd_lat.attrs = {"units": "deg", "long_name": "Field-line footprint quasi-dipole latitude at 130 km"}
    data.footprint_qd_lon.attrs = {"units": "deg", "long_name": "Field-line footprint quasi-dipole longitude at 130 km"}
    data.footprint_mlt.attrs = {"units": "hour", "long_name": "Field-line footprint magnetic local time at 130 km"}
    data.footprint_mapping_error.attrs = {"units": "deg", "long_name": "ApexPy field-line mapping error"}

    data.attrs.update(
        {
            "source_urls": "\n".join(source_urls),
            "source_files": "\n".join(path.name for path in cdf_files),
            "minimum_aacgm_latitude_deg": float(min_mlat),
            "selection_start_utc": np.datetime_as_string(start, unit="s"),
            "selection_end_utc": np.datetime_as_string(end, unit="s"),
            "footprint_height_km": 130.0,
            "footprint_mapping": "ApexPy field-line mapping followed by quasi-dipole coordinates, matching the IMAGE coordinate convention",
        }
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    encoding = {name: {"zlib": True, "complevel": 4} for name in data.data_vars}
    data.to_netcdf(output, encoding=encoding)

    print(f"Kept {times.size} DMSP samples")
    print(f"Saved {output}")


#%% Command line

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--satellite", default="f15")
    #parser.add_argument("--start", default="2001-10-21T23:33:15")
    #parser.add_argument("--end", default="2001-10-22T00:01:58")
    parser.add_argument("--start", default="2001-10-21T00:00:00")
    parser.add_argument("--end", default="2001-10-25T23:59:59")
    parser.add_argument("--output-dir", type=Path, default=Path("example_data/dmsp"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--minimum-mlat", type=float, default=40.0)
    return parser.parse_args()


def main():
    args = parse_args()
    satellite = args.satellite.lower()

    dates = dates_in_interval(args.start, args.end)
    raw_files = []
    source_urls = []
    for date in dates:
        raw_file, source_url = download_daily_file(
            satellite,
            date,
            args.output_dir / "raw",
        )
        raw_files.append(raw_file)
        source_urls.append(source_url)

    first_date = dates[0].replace("-", "")
    last_date = dates[-1].replace("-", "")
    date_label = first_date if first_date == last_date else f"{first_date}-{last_date}"
    output = args.output
    if output is None:
        output = args.output_dir / f"dmsp-{satellite}_ssj_{date_label}_north.nc"

    reduce_files(
        raw_files,
        source_urls,
        output,
        args.minimum_mlat,
        args.start,
        args.end,
    )


if __name__ == "__main__":
    main()
