"""Index the raw IMAGE IDL files by orbit."""

#%% Imports

import argparse
from pathlib import Path

import pandas as pd


#%% Command line

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--base",
    type=Path,
    default=Path(__file__).resolve().parents[1] / "example_data",
    help="Directory containing orbitdates.csv and the *_data folders",
)
args = parser.parse_args()
base = args.base.expanduser()


#%% Orbit intervals

print(f"Reading orbit dates from {base / 'orbitdates.csv'}")
orbits = pd.read_csv(base / "orbitdates.csv")
orbits["dt_start"] = pd.to_datetime(orbits["date_start"], format="%Y-%m-%d %H:%M:%S")
orbits["dt_end"] = pd.to_datetime(orbits["date_end"], format="%Y-%m-%d %H:%M:%S")


#%% Build one sensor index

def generate_files_file(folder):
    filenames = sorted(folder.glob("*.idl"))

    files = pd.DataFrame()
    files["date"] = [
        pd.to_datetime(filename.stem[-11:], format="%Y%j%H%M")
        for filename in filenames
    ]
    files["filename"] = [filename.name for filename in filenames]
    files["orbit"] = -1

    for _, orbit in orbits.iterrows():
        inside_orbit = (
            (files["date"] >= orbit["dt_start"])
            & (files["date"] <= orbit["dt_end"])
        )
        files.loc[inside_orbit, "orbit"] = orbit["orbit_number"]

    files = files.loc[files["orbit"] != -1].copy()
    files["orbit"] = files["orbit"].astype("int64")
    files.set_index("date", inplace=True)
    return files


#%% Write WIC, SI12, and SI13 indices

for prefix, folder in (("wic", "wic_data"), ("s12", "s12_data"), ("s13", "s13_data")):
    print(f"Indexing {prefix.upper()} IDL files")
    files = generate_files_file(base / folder)
    files.to_hdf(base / f"{prefix}files.h5", key="data", mode="w")
    print(f"  {len(files)} files in {files.orbit.nunique()} orbits")
