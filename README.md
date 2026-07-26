# IMAGE Conductance Builder

`icBuilder` is a tool for processing IMAGE WIC, SI12, and SI13 data to estimate ionospheric conductance with propagated uncertainties.

> ⚠️ `icBuilder` is **not intended** for reading the estimated conductances.  
> For that purpose, use the lightweight companion tool [`icReader`](https://github.com/BingMM/icReader).

## Project Description

This code was developed to robustly estimate ionospheric conductances from IMAGE data. All processed IMAGE data (2000-2001) is available on request (~2.5 TB). Estimated ionospheric conductances with associated uncertainties are available [**here**](https://doi.org/10.5281/zenodo.15579301).

The main purpose of this codebase is to document the data processing procedure. While not primarily designed for external use, the code can be run by others if needed.

## Project Memory

For repository-specific operating guidance and technical continuity, start with
[`AGENTS.md`](AGENTS.md) and
[`log/icBuilder/START HERE - AI Onboarding.md`](log/icBuilder/START%20HERE%20-%20AI%20Onboarding.md).

## Dependencies

- [`fuvpy`](https://github.com/aohma/fuvpy) – for FUV image processing **[3]**
- [`secsy`](https://github.com/klaundal/secsy) - for cubed sphere grid generation
- [`tqdm`](https://github.com/tqdm/tqdm) – for progress bars (optional; can be removed with minor edits)
- [`icReader`](https://github.com/BingMM/icReader) - for reading conductance output files (optional; used for creating conductance figures)

## Installation

mamba activate your_environment  
git clone https://github.com/BingMM/icBuilder.git  
cd icBuilder  
pip install -e .

## Step-by-Step Guide

The code is designed to be run in the following sequence:

### `make_orbit_h5_files.py`

- Scans available WIC, SI12, and SI13 data.
- Assigns each file to an orbit based on `orbitsdates.csv`.

### `make_orbit_nc_files.py`

- Reads WIC, SI12, and SI13 data.
- Applies a background removal algorithm.
- Saves the results into a series of NetCDF files **[3]**.

### `make_background_removal_figures.py` *(optional)*

- Plots the data stored in the NetCDF files for quality inspection.

### `grid_resolution_determination.py` *(optional)*

- Analyzes the NetCDF files to determine the optimal grid resolution.
- Uses the trade-off between standard error and the number of bins with ≥30 measurements (lower limit of the central limit theorem).
- Suggested resolution: 225 km for WIC, 450 km for SI12 and SI13.

### `make_conductance_orbit_files.py`

- Ingests the NetCDF files and estimates ionospheric conductance **[1,2,4]**.

### `make_conductance_figures.py`

- *(Not implemented yet)*

### `make_spline_model.py`

- *(Not implemented yet)*

## References

**[1]**. Frey, H. U. et al. (2003). *Summary of Quantitative Interpretation of IMAGE Far Ultraviolet Auroral Data*. In: Burch, J. L. (Ed.), *Magnetospheric Imaging — The IMAGE Prime Mission*. Springer. https://doi.org/10.1007/978-94-010-0027-7_11  
**[2]**. Gasparini, S. et al. (2024). *A quantitative analysis of the uncertainties on reconnection electric field estimates using ionospheric measurements*. *JGR: Space Physics*, 129, e2024JA032599. https://doi.org/10.1029/2024JA032599  
**[3]**. Ohma, A. et al. (2024). *Background removal from global auroral images: Data-driven dayglow modeling*. *Earth Planet. Phys.*, 8(1), 247–257. https://doi.org/10.26464/epp2023051  
**[4]**. Robinson, R. M. et al. (1987). *On calculating ionospheric conductances from the flux and energy of precipitating electrons*. *JGR*, 92(A3), 2565–2569. https://doi.org/10.1029/JA092iA03p02565

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact

For questions or comments, please contact [michael.madelaire@uib.no].
