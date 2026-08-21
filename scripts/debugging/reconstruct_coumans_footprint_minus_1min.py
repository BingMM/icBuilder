"""Repeat the Coumans comparison with DMSP footprints shifted by one minute.

The DMSP energy and plotted time are unchanged. This is an along-track spatial
diagnostic, not a correction to the IMAGE timestamp.
"""

#%% Imports and settings

from pathlib import Path

import reconstruct_coumans_figure4b as comparison


FOOTPRINT_SHIFT_SECONDS = -60
comparison.OUTPUT = (
    Path(comparison.OUTPUT).parent / "dmsp_footprint_minus_1min"
)


#%% Run the separate spatial experiment

def main():
    result = comparison.read_comparison_data(FOOTPRINT_SHIFT_SECONDS)
    comparison.make_figure(result)
    comparison.report(result)


if __name__ == "__main__":
    main()
