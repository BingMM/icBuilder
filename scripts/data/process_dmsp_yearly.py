"""Reduce daily DMSP SSJ CDFs to northern satellite-year NetCDF files."""

#%% Imports

import argparse
import re
from pathlib import Path
from functools import partial

from apexpy import Apex
from astropy.coordinates import EarthLocation
import astropy.units as u
from cdflib import CDF, cdfepoch
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import xarray as xr


#%% Input variables retained in the yearly product

VARIABLES = {
    "spacecraft_geocentric_lat": ("SC_GEOCENTRIC_LAT", 1.0),
    "spacecraft_geocentric_lon": ("SC_GEOCENTRIC_LON", 1.0),
    "spacecraft_geocentric_radius": ("SC_GEOCENTRIC_R", 1.0),
    "source_aacgm_lat": ("SC_AACGM_LAT", 1.0),
    "source_aacgm_lon": ("SC_AACGM_LON", 1.0),
    "source_aacgm_mlt": ("SC_AACGM_LTIME", 1.0),
    "electron_mean_energy": ("ELE_AVG_ENERGY", 1e-3),
    "electron_mean_energy_fractional_std": ("ELE_AVG_ENERGY_STD", 1.0),
    "electron_total_energy_flux": ("ELE_TOTAL_ENERGY_FLUX", 1.0),
    "electron_total_energy_flux_fractional_std": ("ELE_TOTAL_ENERGY_FLUX_STD", 1.0),
    "ion_mean_energy": ("ION_AVG_ENERGY", 1e-3),
    "ion_mean_energy_fractional_std": ("ION_AVG_ENERGY_STD", 1.0),
    "ion_total_energy_flux": ("ION_TOTAL_ENERGY_FLUX", 1.0),
    "ion_total_energy_flux_fractional_std": ("ION_TOTAL_ENERGY_FLUX_STD", 1.0)}

UNITS = {
    "spacecraft_geocentric_lat": "degree",
    "spacecraft_geocentric_lon": "degree",
    "spacecraft_geocentric_radius": "km",
    "source_aacgm_lat": "degree",
    "source_aacgm_lon": "degree",
    "source_aacgm_mlt": "hour",
    "mlat": "degree",
    "mlon": "degree",
    "mlt": "hour",
    "electron_mean_energy": "keV",
    "electron_mean_energy_fractional_std": "1",
    "electron_total_energy_flux": "eV cm-2 sr-1 s-1",
    "electron_total_energy_flux_fractional_std": "1",
    "ion_mean_energy": "keV",
    "ion_mean_energy_fractional_std": "1",
    "ion_total_energy_flux": "eV cm-2 sr-1 s-1",
    "ion_total_energy_flux_fractional_std": "1"}

FILE_PATTERN = re.compile(r"dmsp-(f\d+)_ssj_.*_(\d{8})_v([\d.]+)\.cdf$", re.IGNORECASE)


#%% File discovery

def find_years(input_dir):
    """Find the newest CDF version for each satellite and day."""

    newest = {}

    for path in input_dir.rglob("*.cdf"):
        match = FILE_PATTERN.match(path.name)
        if match is None:
            continue

        satellite, date, version_text = match.groups()
        version = tuple(int(value) for value in version_text.strip(".").split("."))
        key = (satellite.lower(), date)

        if key not in newest or version > newest[key][0]:
            newest[key] = (version, path)

    years = {}
    for (satellite, date), (_, path) in newest.items():
        years.setdefault((satellite, int(date[:4])), []).append(path)

    return {key: sorted(paths) for key, paths in years.items()}


#%% Coordinate calculation

def modified_apex(times, latitude, longitude, radius, reference_height):
    """Calculate modified-Apex coordinates with a 130-km reference height."""

    latitude_rad = np.radians(latitude)
    longitude_rad = np.radians(longitude)

    x = radius * np.cos(latitude_rad) * np.cos(longitude_rad)
    y = radius * np.cos(latitude_rad) * np.sin(longitude_rad)
    z = radius * np.sin(latitude_rad)

    spacecraft = EarthLocation.from_geocentric(x * u.km, y * u.km, z * u.km)
    python_times = times.astype("datetime64[us]").astype(object)
    apex = Apex(date=python_times[0], refh=reference_height)

    mlat, mlon = apex.geo2apex(spacecraft.lat.deg, spacecraft.lon.deg, spacecraft.height.to_value(u.km))
    mlt = apex.mlon2mlt(mlon, python_times)

    return np.asarray(mlat), np.asarray(mlon), np.asarray(mlt)


#%% Read and reduce one daily file

def read_day(path, minimum_mlat, reference_height):
    """Read the northern high-latitude records from one daily CDF."""

    cdf = CDF(path)

    try:
        epoch = np.asarray(cdf.varget("Epoch"))
        times = np.asarray(cdfepoch.to_datetime(epoch), dtype="datetime64[ns]")

        latitude = np.asarray(cdf.varget("SC_GEOCENTRIC_LAT"), dtype=float)
        longitude = np.asarray(cdf.varget("SC_GEOCENTRIC_LON"), dtype=float)
        radius = np.asarray(cdf.varget("SC_GEOCENTRIC_R"), dtype=float)

        mlat, mlon, mlt = modified_apex(times, latitude, longitude, radius, reference_height)
        keep = np.isfinite(mlat) & (mlat >= minimum_mlat)

        if not np.any(keep):
            return None

        values = {"time": times[keep]}
        for output_name, (source_name, scale) in VARIABLES.items():
            values[output_name] = np.asarray(cdf.varget(source_name))[keep] * scale

        values["mlat"] = mlat[keep]
        values["mlon"] = mlon[keep]
        values["mlt"] = mlt[keep]

    finally:
        cdf.close()

    return values


#%% Build one satellite-year product

def process_year(task, minimum_mlat, reference_height):
    """Combine the selected daily files and save one yearly NetCDF."""

    files, output, satellite, year = task
    collected = {}

    for path in files:
        day = read_day(path, minimum_mlat, reference_height)
        if day is None:
            continue

        for name, values in day.items():
            collected.setdefault(name, []).append(values)

    if not collected:
        print(f"No northern records found for {satellite.upper()} {year}")
        return

    values = {name: np.concatenate(parts) for name, parts in collected.items()}
    order = np.argsort(values["time"])
    _, unique = np.unique(values["time"][order], return_index=True)
    order = order[unique]

    data = xr.Dataset(coords={"time": values.pop("time")[order]})
    for name, array in values.items():
        data[name] = ("time", np.asarray(array[order], dtype=np.float32))
        data[name].attrs["units"] = UNITS[name]

    data.attrs.update({
        "product_type": "dmsp_ssj_yearly",
        "satellite": satellite.upper(),
        "year": int(year),
        "hemisphere": "north",
        "minimum_mlat_deg": float(minimum_mlat),
        "reference_height_km": float(reference_height),
        "magnetic_coordinates": "Modified Apex calculated at spacecraft altitude with the stated reference height",
        "source_files": "\n".join(path.name for path in files)})

    # Save file
    partial = output.with_name(output.name + ".partial")
    encoding = {name: {"zlib": True, "complevel": 4} for name in data.data_vars}
    data.to_netcdf(partial, engine="netcdf4", encoding=encoding)

    with xr.open_dataset(partial) as check:
        if check.sizes["time"] != data.sizes["time"]:
            raise OSError(f"Incomplete yearly file: {partial}")

    partial.replace(output)

#%% Command line

def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "1"):
        return True
    if value.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("/media/bing/LaCie/dmsp_ssj"))
    parser.add_argument("--output", type=Path, default=Path("/media/bing/LaCie/dmsp_ssj_yearly"))
    parser.add_argument("--satellite", nargs="*", help="For example: f13 f15")
    parser.add_argument("--year", nargs="*", type=int, help="For example: 2000 2001")
    parser.add_argument("--minimum-mlat", type=float, default=40.0)
    parser.add_argument("--reference-height", type=float, default=130.0)
    parser.add_argument("--parallel", type=str2bool, default=False)
    parser.add_argument("--pool_size", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    years = find_years(args.input)
    satellites = None if args.satellite is None else {value.lower() for value in args.satellite}
    selected_years = None if args.year is None else set(args.year)
        
    tasks = []
    for (satellite, year), files in sorted(years.items()):
        if satellites is not None and satellite not in satellites:
            continue
        if selected_years is not None and year not in selected_years:
            continue

        output = args.output / f"dmsp_{satellite}_ssj_{year}_north.nc"
        if output.exists() and not args.overwrite:
            print(f"Skipping existing {output}")
            continue

        tasks.append((files, output, satellite, year))
    
    function = partial(process_year,
                       minimum_mlat = args.minimum_mlat, 
                       reference_height = args.reference_height)

    if args.parallel:
        with ProcessPoolExecutor(max_workers=args.pool_size) as executor:
            futures = [executor.submit(function, task) for task in tasks]
    
            for future in tqdm(as_completed(futures), total=len(futures), desc="Creating DMSP yearly files"):
                future.result()
    else:
        for task in tqdm(tasks, desc="Creating DMSP yearly files"):
            function(task) 

if __name__ == "__main__":
    main()
