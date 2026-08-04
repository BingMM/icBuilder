"""Read the Zhang--Paxton electron-energy lookup for the IMAGE grid."""

from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from .grids import grid_mlt, make_wic_grid


KP_VALUES = np.arange(901) / 100
DEFAULT_LOOKUP_PATH = (
    Path(__file__).parent / "data" / "zhang_paxton_e0_lookup.nc"
)


def kp_to_index(kp):
    """Return the nearest-hundredth Kp index.

    Kp is positive, so adding 0.5 before taking the floor gives conventional
    rounding: 1.519 becomes 1.52 and an exact half rounds upward.
    """

    kp = np.asarray(kp, dtype=float)
    if np.any(~np.isfinite(kp)) or np.any((kp < 0) | (kp > 9)):
        raise ValueError("Kp must be finite and between 0 and 9")
    return np.floor(kp * 100 + 0.5 + 1e-12).astype(int)


def load_zhang_paxton_lookup(kp, path=DEFAULT_LOOKUP_PATH):
    """Load E0, dE0, and median E0 for one or more Kp values.

    Kp is rounded to the nearest hundredth without interpolation.  Scalar Kp
    returns 36 by 36 arrays.  An array of Kp values adds its dimensions before
    the two grid dimensions.
    """

    index = kp_to_index(kp)
    grid = make_wic_grid()

    with Dataset(path) as nc:
        stored_kp = np.asarray(nc["kp"][:])
        stored_mlt = np.asarray(nc["mlt"][:])
        stored_xi = np.asarray(nc["xi"][:])
        stored_eta = np.asarray(nc["eta"][:])
        provenance = {
            "description": nc.description,
            "threshold_energy_flux_mW_m2": float(
                nc.threshold_energy_flux_mW_m2
            ),
            "lower_mlat_degrees": float(nc.lower_mlat_degrees),
            "upper_mlat_degrees": float(nc.upper_mlat_degrees),
            "latitude_step_degrees": float(nc.latitude_step_degrees),
            "zhang_paxton_package_version": nc.zhang_paxton_package_version,
        }

        if stored_kp.shape != (901,) or not np.allclose(stored_kp, KP_VALUES):
            raise ValueError("lookup must contain Kp 0.00 to 9.00 in 0.01 steps")
        expected_shape = grid.shape
        if (
            stored_xi.shape != expected_shape
            or stored_eta.shape != expected_shape
            or stored_mlt.shape != expected_shape
            or not np.allclose(stored_xi, grid.xi)
            or not np.allclose(stored_eta, grid.eta)
            or not np.allclose(stored_mlt, grid_mlt(grid))
        ):
            raise ValueError("lookup coordinates do not match the IMAGE grid")

        # An orbit repeats many Kp values. Read each required layer once
        # instead of loading the complete 901-layer table for every orbit.
        unique_index, inverse = np.unique(index.reshape(-1), return_inverse=True)
        output_shape = index.shape + grid.shape

        E0 = np.asarray(nc["E0"][unique_index, :, :])[inverse]
        E0 = E0.reshape(output_shape)
        dE0 = np.asarray(nc["dE0"][unique_index, :, :])[inverse]
        dE0 = dE0.reshape(output_shape)
        E0_median = np.asarray(nc["E0_median"][unique_index, :, :])[inverse]
        E0_median = E0_median.reshape(output_shape)

    if (
        np.any(~np.isfinite(E0))
        or np.any(~np.isfinite(dE0))
        or np.any(~np.isfinite(E0_median))
    ):
        raise ValueError("lookup contains non-finite values")

    return {
        "kp": index / 100,
        "E0": E0,
        "dE0": dE0,
        "E0_median": E0_median,
        "mlt": stored_mlt,
        "xi": stored_xi,
        "eta": stored_eta,
        "provenance": provenance,
    }
