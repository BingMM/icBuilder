"""Build sensor-specific binned IMAGE-FUV orbit files."""

import argparse
import os
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path

import apexpy
import numpy as np
from apexpy.helpers import subsol
from netCDF4 import Dataset
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from icbuilder import BinnedImage, PreImage
from icbuilder.grids import make_image_grids

#%%

SENSORS = {
    "WIC": {"folder": "wic", "prefix": "wic", "grid": "wic", "correction": "SH"},
    "SI12": {"folder": "s12", "prefix": "s12", "grid": "si", "correction": "DG"},
    "SI13": {"folder": "s13", "prefix": "s13", "grid": "si", "correction": "DG"}}

BINNED_FIELDS = (
    "counts", "mu", "sigma", "w", "sza", "dza", "los_factor", "coverage"
)

#%%

def safe_apex_convert(apex, time, glat, glon, height=130):
    """Convert the finite geographic pixels in one frame."""
    valid = np.isfinite(glat) & np.isfinite(glon)
    mlat = np.full_like(glat, np.nan)
    mlon = np.full_like(glon, np.nan)
    mlt = np.full_like(glon, np.nan)
    ssalon = np.nan

    if np.any(valid):
        mlat_valid, mlon_valid = apex.convert(glat[valid], glon[valid], "geo", "apex", height=height)
        mlat[valid] = mlat_valid
        mlon[valid] = mlon_valid

        subsolar_lat, subsolar_lon = subsol(time)
        _, ssalon = apex.geo2apex(subsolar_lat, subsolar_lon, 318550)
        mlt[valid] = (180 + np.float64(mlon_valid) - ssalon) / 15 % 24

    return mlat, mlon, mlt, ssalon


def save_binned_file(binned, filename):
    """Write beside the final file, then rename after a successful close."""
    filename = Path(filename)
    partial = Path(str(filename) + ".partial")
    try:
        binned.to_nc(partial)
        status = binned_file_status(
            partial,
            binned.sensor,
            binned.correction or "raw",
            binned.los_correction,
            binned.binning_method,
        )
        if status != "complete":
            raise RuntimeError(f"incomplete binned file: {partial}")
        os.replace(partial, filename)
    finally:
        if partial.exists():
            partial.unlink()


def binned_file_status(filename, sensor, correction, los_correction,
                       binning_method):
    """Return missing, invalid, mismatch, or complete for one Product-1 file."""

    filename = Path(filename)
    if not filename.is_file():
        return "missing"

    try:
        with Dataset(filename) as nc:
            if nc.product_type != "binned_fuv" or int(nc.schema_version) != 1:
                return "invalid"
            if (
                nc.sensor != sensor
                or nc.image_correction != correction
                or bool(nc.los_correction) != bool(los_correction)
                or nc.binning_method != binning_method
            ):
                return "mismatch"

            shape = (
                len(nc.dimensions["time"]),
                len(nc.dimensions["dim1"]),
                len(nc.dimensions["dim2"]),
            )
            if shape[0] == 0:
                return "invalid"
            for name in BINNED_FIELDS:
                if nc.variables[name].shape != shape:
                    return "invalid"
            for name in ("time", "ssalon"):
                if nc.variables[name].shape != (shape[0],):
                    return "invalid"
            grid = nc.groups["grid"]
            for name in ("xi", "eta", "mlat", "mlt"):
                if grid.variables[name].shape != shape[1:]:
                    return "invalid"
    except (OSError, RuntimeError, KeyError, AttributeError, ValueError):
        return "invalid"

    return "complete"


def process_sensor_orbit(task, input_paths, output_dir, grid_wic, grid_si,
                         fullness_threshold=0.1, los_correction=False,
                         binning_method="footprint"):
    """Bin one sensor orbit on that sensor's native grid."""
    sensor, orbit = task
    
    settings = SENSORS[sensor]
    
    grid = grid_wic if settings["grid"] == "wic" else grid_si
    
    input_file = (input_paths[settings["folder"]] / f'{settings["prefix"]}_or{orbit:04d}.nc')

    with Dataset(input_file) as nc:
        
        # Flag NaN frames
        keep = ~np.all(np.isnan(nc.variables["mlat"][:]), axis=(1, 2))
        
        # Get time        
        start = datetime.strptime(nc.variables["t_start"][:], "%Y-%m-%dT%H:%M:%S")        
        time = np.array([start + timedelta(seconds=int(seconds)) for seconds in nc.variables["date"][:]], dtype=object)[keep]

        # Create index array
        indices = np.flatnonzero(keep)
        
        # Generate PreImage
        preimage = PreImage(sensor, nc, indices)

    # Convert to apex coordinates
    for i, frame_time in enumerate(time):        
        apex = apexpy.Apex(frame_time, refh=130)
        converted = safe_apex_convert(apex, frame_time, preimage.glat[i], preimage.glon[i]) 
        preimage.mlat[i], preimage.mlon[i], preimage.mlt[i], preimage.ssalon[i] = converted

    # Additional filter on frame population
    keep = preimage.percent_full(grid) >= fullness_threshold
    time = time[keep]
    preimage.discard(keep)
    if time.size == 0:
        raise ValueError(f"{sensor} orbit {orbit:04d} has no usable frames")

    # Generate BinnedImage
    binned = BinnedImage(preimage, grid, time, 
                         inflate_uncertainty=True, 
                         correction=settings["correction"], 
                         los_correction=los_correction,
                         binning_method=binning_method)
    
    # Save and return
    filename = output_dir / sensor.lower() / f"or_{orbit:04d}.nc"
    save_binned_file(binned, filename)
    return sensor, orbit, time.size


def get_orbits(input_dir):
    """Return orbit numbers discovered from the NetCDF files present."""

    orbits = []
    for filename in sorted(input_dir.glob("*.nc")):
        orbit_text = filename.stem[-4:]
        if orbit_text.isdigit():
            orbits.append(int(orbit_text))
    return np.unique(orbits)


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "1"):
        return True
    if value.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Create native-grid binned IMAGE-FUV orbit files."
    )
    parser.add_argument("--parallel", type=str2bool, default=False)
    parser.add_argument("--pool_size", type=int, default=4)
    parser.add_argument(
        "--base", type=Path,
        default=Path(__file__).resolve().parents[1] / "example_data",
    )
    parser.add_argument("--wic-folder", default="wic")
    parser.add_argument("--s12-folder", default="s12")
    parser.add_argument("--s13-folder", default="s13")
    parser.add_argument("--output-folder", default="binned")
    parser.add_argument(
        "--binning-method",
        choices=("footprint", "centre"),
        default="footprint",
        help="Use footprint overlap or the former point-centre median binning.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)

#%%

def main(argv=None):
    args = parse_args(argv)
    
    base = args.base.expanduser()
    input_paths = {"wic": base / args.wic_folder,
                   "s12": base / args.s12_folder,
                   "s13": base / args.s13_folder}    
    output_dir = base / args.output_folder
    
    grid_wic, grid_si = make_image_grids()

    print(f"Binning method: {args.binning_method}")

    tasks = []
    for sensor, settings in SENSORS.items():
        sensor_dir = output_dir / sensor.lower()
        sensor_dir.mkdir(parents=True, exist_ok=True)
        
        orbits = get_orbits(input_paths[settings["folder"]])
        
        pending = []
        for orbit in orbits:
            if args.overwrite:
                pending.append(orbit)
                continue

            filename = sensor_dir / f"or_{orbit:04d}.nc"
            status = binned_file_status(
                filename,
                sensor,
                settings["correction"],
                False,
                args.binning_method,
            )
            if status == "mismatch":
                raise ValueError(
                    f"{filename} does not match the requested {sensor} "
                    "configuration; choose another output folder or use --overwrite"
                )
            if status != "complete":
                pending.append(orbit)
        
        print(f"{sensor}: {len(orbits) - len(pending)} complete, {len(pending)} pending")
        tasks.extend((sensor, int(orbit)) for orbit in pending)

    if not tasks:
        print("All binned files already exist.")
        return []

    function = partial(
        process_sensor_orbit,
        input_paths=input_paths,
        output_dir=output_dir,
        grid_wic=grid_wic,
        grid_si=grid_si,
        binning_method=args.binning_method,
    )
    if args.parallel:
        return process_map(
            function, tasks, max_workers=args.pool_size, chunksize=1,
            desc="Bin sensor orbits"
        )

    return [function(task) for task in tqdm(tasks, desc="Bin sensor orbits")]


#%%

if __name__ == "__main__":
    main()
