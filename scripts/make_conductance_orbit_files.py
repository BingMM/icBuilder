#%% Import 

from os.path import join as pjoin
from pathlib import Path
import numpy as np
import glob
from netCDF4 import Dataset
from datetime import datetime, timedelta
import apexpy
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map  # tqdm-compatible multiprocessing
from functools import partial
from icbuilder import PreImage, BinnedImage, ConductanceImage
from icbuilder.grids import make_image_grids
from icbuilder.kp import load_gfz_kp, match_gfz_kp
import argparse
from apexpy.helpers import subsol

#%% Fun

def find_common_times_with_indices(list1, list2, list3, tolerance=timedelta(seconds=2)):
    # Sort the lists and keep track of the original indices
    sorted_list1 = sorted((time, i) for i, time in enumerate(list1))
    sorted_list2 = sorted((time, i) for i, time in enumerate(list2))
    sorted_list3 = sorted((time, i) for i, time in enumerate(list3))
    
    # Initialize pointers for each list
    i, j, k = 0, 0, 0
    common_times = []
    indices = []
    
    # Traverse the lists
    while i < len(sorted_list1) and j < len(sorted_list2) and k < len(sorted_list3):
        # Find the maximum and minimum times among the current elements
        max_time = max(sorted_list1[i][0], sorted_list2[j][0], sorted_list3[k][0])
        min_time = min(sorted_list1[i][0], sorted_list2[j][0], sorted_list3[k][0])
        
        # Check if the times are within the tolerance
        if max_time - min_time <= tolerance:
            common_times.append(max_time)
            indices.append((sorted_list1[i][1], sorted_list2[j][1], sorted_list3[k][1]))
            i += 1
            j += 1
            k += 1
        else:
            # Move the pointer of the list with the smallest time
            if sorted_list1[i][0] == min_time:
                i += 1
            elif sorted_list2[j][0] == min_time:
                j += 1
            else:
                k += 1
    
    return np.array(common_times), np.array(indices)

def safe_apex_convert(apex, ti, glat_row, glon_row, height=110):
    valid = ~np.isnan(glat_row) & ~np.isnan(glon_row)
    mlat = np.full_like(glat_row, np.nan)
    mlon = np.full_like(glon_row, np.nan)
    mlt  = np.full_like(glon_row, np.nan)
    if np.any(valid):
        mlat_valid, mlon_valid = apex.convert(glat_row[valid], glon_row[valid], 'geo', 'apex', height=height)
        mlat[valid] = mlat_valid
        mlon[valid] = mlon_valid
        
        ssglat, ssglon = subsol(ti)
        _, ssalon = apex.geo2apex(ssglat, ssglon, 318550)
        mlt[valid] = (180 + np.float64(mlon_valid) - ssalon) / 15 % 24
    return mlat, mlon, mlt, ssalon

def process_orbit(
    orbit,
    p_wic_nc,
    p_s12_nc,
    p_s13_nc,
    p_out,
    grid_w,
    grid_s,
    kp_series,
    f_thres=0.1,
):
    #try:
    # Load nc orbit files
    wic_nc = Dataset(pjoin(p_wic_nc, f'wic_or{orbit:04d}.nc'), 'r')
    s12_nc = Dataset(pjoin(p_s12_nc, f's12_or{orbit:04d}.nc'), 'r')
    s13_nc = Dataset(pjoin(p_s13_nc, f's13_or{orbit:04d}.nc'), 'r')

    # Get start time for each instrument
    t0_wic = datetime.strptime(wic_nc.variables['t_start'][:], '%Y-%m-%dT%H:%M:%S')
    t0_s12 = datetime.strptime(s12_nc.variables['t_start'][:], '%Y-%m-%dT%H:%M:%S')
    t0_s13 = datetime.strptime(s13_nc.variables['t_start'][:], '%Y-%m-%dT%H:%M:%S')

        # Get time for each sensor
    t_wic = [t0_wic + timedelta(seconds=int(sec)) for sec in wic_nc.variables['date'][:]]
    t_s12 = [t0_s12 + timedelta(seconds=int(sec)) for sec in s12_nc.variables['date'][:]]
    t_s13 = [t0_s13 + timedelta(seconds=int(sec)) for sec in s13_nc.variables['date'][:]]
        
        # Get common times and their indices
    t, indices = find_common_times_with_indices(t_wic, t_s12, t_s13)

        # Remove completely empty frames
    empty = np.all(np.isnan(wic_nc.variables['mlat'][indices[:, 0]]), axis=(1,2))
    empty |= np.all(np.isnan(s12_nc.variables['mlat'][indices[:, 1]]), axis=(1,2))
    empty |= np.all(np.isnan(s13_nc.variables['mlat'][indices[:, 2]]), axis=(1,2))
    t, indices = t[~empty], indices[~empty, :]

        # Extract data
    wic_pI = PreImage(wic_nc, indices[:, 0])
    s12_pI = PreImage(s12_nc, indices[:, 1])
    s13_pI = PreImage(s13_nc, indices[:, 2])

        # Close files
    wic_nc.close()
    s12_nc.close()
    s13_nc.close()

        # APEX conversion
    for i, ti in enumerate(t):
        apex = apexpy.Apex(ti)
        wic_pI.mlat[i], wic_pI.mlon[i], wic_pI.mlt[i], wic_pI.ssalon[i] = safe_apex_convert(apex, ti, wic_pI.glat[i], wic_pI.glon[i])
        s12_pI.mlat[i], s12_pI.mlon[i], s12_pI.mlt[i], s12_pI.ssalon[i] = safe_apex_convert(apex, ti, s12_pI.glat[i], s12_pI.glon[i])
        s13_pI.mlat[i], s13_pI.mlon[i], s13_pI.mlt[i], s13_pI.ssalon[i] = safe_apex_convert(apex, ti, s13_pI.glat[i], s13_pI.glon[i])

        # Fullness check
    wic_p = wic_pI.percent_full(grid_w)
    s12_p = s12_pI.percent_full(grid_s)
    s13_p = s13_pI.percent_full(grid_s)
    f = (wic_p >= f_thres) & (s12_p >= f_thres) & (s13_p >= f_thres)

    t = t[f]
    wic_pI.discard(f)
    s12_pI.discard(f)
    s13_pI.discard(f)

    # GFZ timestamps mark the start of each three-hour interval. Match only
    # the final IMAGE frame population that will enter this orbit product.
    frame_kp = match_gfz_kp(t, kp_series)

        # Bin and conductance images
    losc = False
    wic_bI = BinnedImage(wic_pI, grid_w, inflate_uncertainty=True, correction='SH', los_correction=losc)
    s12_bI = BinnedImage(s12_pI, grid_s, inflate_uncertainty=True, target_grid=grid_w, correction='DG', los_correction=losc)
    s13_bI = BinnedImage(s13_pI, grid_s, inflate_uncertainty=True, target_grid=grid_w, correction='DG', los_correction=losc)
    cI = ConductanceImage(
        wic_bI,
        s12_bI,
        s13_bI,
        time=t,
        kp=frame_kp["kp"],
        kp_interval_start=frame_kp["interval_start"],
        kp_provenance=kp_series["provenance"],
        energy_method="zhang_paxton",
    )
        # Save to netCDF
    out_path = pjoin(p_out, f'or_{orbit:04d}.nc')
    cI.to_nc(out_path)
    return orbit  # Success
    #except Exception as e:
    #    print(f"Failed orbit {orbit}: {e}")
    #    return None

def run_all_orbits(
    o,
    p_wic_nc,
    p_s12_nc,
    p_s13_nc,
    p_out,
    grid_w,
    grid_s,
    kp_series,
    parallel=True,
    n_processes=None,
):

    kwargs = {
            'p_wic_nc': p_wic_nc,
            'p_s12_nc': p_s12_nc,
            'p_s13_nc': p_s13_nc,
            'p_out': p_out,
            'grid_w': grid_w,  # must be defined in your namespace
            'grid_s': grid_s,  # must be defined in your namespace
            'kp_series': kp_series,
            'f_thres': 0.1
            }
    
    print(f'Pulling WIC data from {p_wic_nc}')
    print(f'Pulling SI12 data from {p_s12_nc}')
    print(f'Pulling SI13 data from {p_s13_nc}\n')
    
    print(f'Outputting conductance objects in {p_out}\n')
    
    func = partial(process_orbit, **kwargs)
    
    if parallel:
        results = process_map(func, o, max_workers=n_processes, chunksize=1, desc='Loop over orbits')
    else:
        results = []
        for orbit in tqdm(o, desc='Loop over orbits'):
            results.append(func(orbit))

    return results

#%% Paths

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

def get_orbit_list(p_nc, p_npy, sensor):
    # Fetch all orbits
    o = np.array([int(o[-7:-3]) for o in sorted(glob.glob(pjoin(p_nc, '*.nc')))])
    
    # Load file with orbit quality
    avail = np.load(p_npy)
    print(f'{sensor}: {avail.shape[0]} nc files found. {int(np.sum(avail[:, 1]==1))} orbits useable.')
    avail = avail[avail[:, 1]==1, 0]
    
    # Grab orbits
    return o[np.isin(o, avail)]

def data_paths(base, wic_folder='wic', s12_folder='s12',
               s13_folder='s13', output_folder='conductance'):
    """Return the three input directories and output directory under *base*."""
    base = Path(base).expanduser()
    return {
        'base': base,
        'wic': base / wic_folder,
        's12': base / s12_folder,
        's13': base / s13_folder,
        'output': base / output_folder,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Process FUV data.")
    parser.add_argument('--parallel', type=str2bool, default=False,
                        help='Run in parallel (default False)')
    parser.add_argument('--pool_size', type=int, default=96,
                        help='Number of orbit workers (default 96)')
    parser.add_argument(
        '--base', type=Path,
        default=Path(__file__).resolve().parents[1] / 'example_data',
        help='Base directory containing the input and output folders',
    )
    parser.add_argument('--wic-folder', default='wic',
                        help='WIC folder name under --base (default wic)')
    parser.add_argument('--s12-folder', default='s12',
                        help='SI12 folder name under --base (default s12)')
    parser.add_argument('--s13-folder', default='s13',
                        help='SI13 folder name under --base (default s13)')
    parser.add_argument('--output-folder', default='conductance',
                        help='Output folder name under --base (default conductance)')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    paths = data_paths(
        args.base,
        args.wic_folder,
        args.s12_folder,
        args.s13_folder,
        args.output_folder,
    )

    # The availability arrays always live directly under the base directory.
    o_wic = get_orbit_list(
        paths['wic'], paths['base'] / 'wic_avail_orbit.npy', 'WIC')
    o_s12 = get_orbit_list(
        paths['s12'], paths['base'] / 's12_avail_orbit.npy', 'S12')
    o_s13 = get_orbit_list(
        paths['s13'], paths['base'] / 's13_avail_orbit.npy', 'S13')

    # Create list of all overlapping orbits.
    orbits = set(o_wic) & set(o_s12) & set(o_s13)
    print(f'There are {len(orbits)} common orbits between WIC, S12, and SI13\n')

    grid_w, grid_s = make_image_grids()
    print('Fine grid resolution is: ' + str(grid_w.Lres/1e3) + ' km')
    print('Coarse grid target resolution is: 450.0 km')
    print('Coarse grid resolution is : ' + str(grid_s.Lres/1e3) + ' km\n')

    # Load the fixed local Kp source once. Orbit workers never contact GFZ.
    kp_series = load_gfz_kp()

    paths['output'].mkdir(parents=True, exist_ok=True)
    return run_all_orbits(
        orbits,
        paths['wic'],
        paths['s12'],
        paths['s13'],
        paths['output'],
        grid_w,
        grid_s,
        kp_series,
        parallel=args.parallel,
        n_processes=args.pool_size,
    )


if __name__ == '__main__':
    main()
