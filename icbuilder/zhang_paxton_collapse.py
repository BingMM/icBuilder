"""Reduce a Zhang--Paxton latitude profile to one energy per Kp and MLT.

The model gives electron mean energy ``E0`` and energy flux ``Q`` as functions
of Kp, magnetic local time (MLT), and magnetic latitude (MLAT).  For each MLT
slice we:

1. find the largest Q value;
2. keep the contiguous cells around that peak where Q is above a threshold;
3. calculate the spherical-area-weighted E0 mean, median, and spread.

The default ``Q > 0.05 mW m-2`` threshold follows the criterion used for the
global-average energy in Zhang and Paxton (2008), Figure 8.  The 0.01-degree
latitude step resolves movement of the threshold boundary; it does not imply
that the empirical model has 0.01-degree physical resolution.

Functions return ordinary dictionaries so that the calculation and its
outputs remain easy to inspect in a research notebook or script.
"""

import numpy as np

from zhangpaxton2008 import zhang_paxton


DEFAULT_MLAT_STEP_DEGREES = 0.01
FIGURE8_THRESHOLD = 0.05
BOUNDARY_THRESHOLD = 0.25
MODEL_EVALUATION_BATCH_SIZE = 64


def regular_latitude_grid(
    lower_mlat=50.0,
    upper_mlat=90.0,
    step=DEFAULT_MLAT_STEP_DEGREES,
):
    """Return latitude cell centres and edges in degrees."""

    if not np.isfinite([lower_mlat, upper_mlat, step]).all():
        raise ValueError("latitude limits and step must be finite")
    if not 0 <= lower_mlat < upper_mlat <= 90:
        raise ValueError("require 0 <= lower_mlat < upper_mlat <= 90 degrees")
    if step <= 0:
        raise ValueError("latitude step must be positive")

    width = upper_mlat - lower_mlat
    number_of_cells = int(round(width / step))
    if number_of_cells < 1 or not np.isclose(number_of_cells * step, width):
        raise ValueError("latitude step must divide the requested range exactly")

    edges = np.linspace(lower_mlat, upper_mlat, number_of_cells + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    return centres, edges


def latitude_cell_area_weights(latitude_edges):
    """Return exact relative spherical areas for latitude cells.

    For a fixed MLT width, cell area is proportional to
    ``sin(latitude_upper) - sin(latitude_lower)``.  The common Earth-radius
    and MLT-width factors cancel from every weighted statistic.
    """

    edges = np.asarray(latitude_edges, dtype=float)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("latitude_edges must be a one-dimensional edge array")
    if not np.all(np.isfinite(edges)) or not np.all(np.diff(edges) > 0):
        raise ValueError("latitude_edges must be finite and increasing")
    if edges[0] < -90 or edges[-1] > 90:
        raise ValueError("latitude edges must lie between -90 and 90 degrees")

    edges = np.deg2rad(edges)
    return np.sin(edges[1:]) - np.sin(edges[:-1])


def weighted_median(values, weights):
    """Return the value where cumulative positive weight reaches one half."""

    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.ndim != 1 or weights.ndim != 1 or values.shape != weights.shape:
        raise ValueError("values and weights must be one-dimensional and aligned")
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("values must be non-empty and finite")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("weights must be finite and non-negative")
    if weights.sum() <= 0:
        raise ValueError("at least one weight must be positive")

    order = np.argsort(values)
    cumulative_weight = np.cumsum(weights[order])
    median_index = np.searchsorted(cumulative_weight, weights.sum() / 2)
    return float(values[order][median_index])


def _contiguous_cells_around_peak(above_threshold, peak_index):
    """Select the connected above-threshold interval containing the Q peak."""

    selected = np.zeros(above_threshold.size, dtype=bool)
    if not above_threshold[peak_index]:
        return selected

    lower = peak_index
    while lower > 0 and above_threshold[lower - 1]:
        lower -= 1

    upper = peak_index
    while upper + 1 < above_threshold.size and above_threshold[upper + 1]:
        upper += 1

    selected[lower : upper + 1] = True
    return selected


def _empty_slice(area_weights, threshold, peak_mlat=np.nan, peak_flux=np.nan):
    """Return explicit missing values when no oval interval is selected."""

    return {
        "representative_energy": np.nan,
        "area_weighted_median_energy": np.nan,
        "weighted_spread": np.nan,
        "q_weighted_energy": np.nan,
        "selected_lower_mlat": np.nan,
        "selected_upper_mlat": np.nan,
        "selected_area_weight": 0.0,
        "selected_area_fraction": 0.0,
        "threshold_cutoff": float(threshold),
        "peak_mlat": float(peak_mlat),
        "peak_energy_flux": float(peak_flux),
        "empty": True,
        "touches_equatorward_sampling_limit": False,
        "touches_poleward_sampling_limit": False,
        "reaches_physical_pole": False,
        "selected": np.zeros(area_weights.size, dtype=bool),
        "area_weights": area_weights,
    }


def collapse_latitude_slice(
    mean_energy,
    energy_flux,
    latitude_edges,
    threshold=FIGURE8_THRESHOLD,
):
    """Collapse one latitude profile inside its principal auroral interval.

    The representative E0 is the spherical-area-weighted mean.  The median is
    a robust typical-area diagnostic, while ``weighted_spread`` is the MLAT
    variability discarded by the collapse and is used as dE0.  The
    Q-weighted mean is retained only as a sensitivity diagnostic.
    """

    energy = np.asarray(mean_energy, dtype=float)
    flux = np.asarray(energy_flux, dtype=float)
    edges = np.asarray(latitude_edges, dtype=float)
    area_weights = latitude_cell_area_weights(edges)

    if energy.ndim != 1 or flux.ndim != 1:
        raise ValueError("mean_energy and energy_flux must be one-dimensional")
    if energy.shape != flux.shape or energy.size != area_weights.size:
        raise ValueError("energy, flux, and latitude edges must describe the same cells")
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and non-negative")

    latitude_centres = (edges[:-1] + edges[1:]) / 2
    finite_flux = np.isfinite(flux)
    if not finite_flux.any():
        # Match the historical diagnostic: no finite Q means the cutoff was
        # never evaluated, so peak and cutoff diagnostics are all missing.
        return _empty_slice(area_weights, np.nan)

    # Non-finite values are unavailable and can never enter the selected oval.
    peak_index = int(np.argmax(np.where(finite_flux, flux, -np.inf)))
    peak_mlat = latitude_centres[peak_index]
    peak_flux = flux[peak_index]
    above_threshold = finite_flux & np.isfinite(energy) & (flux > threshold)
    selected = _contiguous_cells_around_peak(above_threshold, peak_index)
    if not selected.any():
        return _empty_slice(area_weights, threshold, peak_mlat, peak_flux)

    indices = np.flatnonzero(selected)
    lower_index = indices[0]
    upper_index = indices[-1]
    selected_energy = energy[selected]
    selected_area = area_weights[selected]
    selected_area_sum = selected_area.sum()

    mean = float(np.average(selected_energy, weights=selected_area))
    median = weighted_median(selected_energy, selected_area)
    variance = np.average((selected_energy - mean) ** 2, weights=selected_area)

    # This statistic emphasizes locations carrying most modeled energy flux.
    # It is not the primary spatial mean or a particle-number-weighted energy.
    power_weights = selected_area * np.clip(flux[selected], 0, None)
    if power_weights.sum() > 0:
        q_weighted_mean = float(np.average(selected_energy, weights=power_weights))
    else:
        q_weighted_mean = np.nan

    return {
        "representative_energy": mean,
        "area_weighted_median_energy": median,
        "weighted_spread": float(np.sqrt(max(variance, 0))),
        "q_weighted_energy": q_weighted_mean,
        "selected_lower_mlat": float(edges[lower_index]),
        "selected_upper_mlat": float(edges[upper_index + 1]),
        "selected_area_weight": float(selected_area_sum),
        "selected_area_fraction": float(selected_area_sum / area_weights.sum()),
        "threshold_cutoff": float(threshold),
        "peak_mlat": float(peak_mlat),
        "peak_energy_flux": float(peak_flux),
        "empty": False,
        "touches_equatorward_sampling_limit": lower_index == 0,
        "touches_poleward_sampling_limit": (
            upper_index == energy.size - 1 and not np.isclose(edges[-1], 90)
        ),
        "reaches_physical_pole": (
            upper_index == energy.size - 1 and np.isclose(edges[-1], 90)
        ),
        "selected": selected,
        "area_weights": area_weights,
    }


def collapse_zhang_paxton(
    kp,
    mlt,
    threshold=FIGURE8_THRESHOLD,
    lower_mlat=50.0,
    upper_mlat=90.0,
    latitude_step=DEFAULT_MLAT_STEP_DEGREES,
):
    """Evaluate and collapse all broadcast pairs of Kp and MLT.

    Array fields in the returned dictionary have the broadcast shape of
    ``kp`` and ``mlt``.  Scalar input therefore gives zero-dimensional arrays.
    """

    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and non-negative")

    kp_grid, mlt_grid = np.broadcast_arrays(
        np.asarray(kp, dtype=float),
        np.asarray(mlt, dtype=float),
    )
    latitude_centres, latitude_edges = regular_latitude_grid(
        lower_mlat,
        upper_mlat,
        latitude_step,
    )

    float_fields = [
        "representative_energy",
        "area_weighted_median_energy",
        "weighted_spread",
        "q_weighted_energy",
        "selected_lower_mlat",
        "selected_upper_mlat",
        "selected_area_weight",
        "selected_area_fraction",
        "threshold_cutoff",
        "peak_mlat",
        "peak_energy_flux",
    ]
    flag_fields = [
        "empty",
        "touches_equatorward_sampling_limit",
        "touches_poleward_sampling_limit",
        "reaches_physical_pole",
    ]
    result = {
        name: np.full(kp_grid.shape, np.nan, dtype=float)
        for name in float_fields
    }
    result.update({
        name: np.zeros(kp_grid.shape, dtype=bool)
        for name in flag_fields
    })

    flat_kp = kp_grid.ravel()
    flat_mlt = mlt_grid.ravel()
    for start in range(0, flat_kp.size, MODEL_EVALUATION_BATCH_SIZE):
        stop = min(start + MODEL_EVALUATION_BATCH_SIZE, flat_kp.size)
        energy, flux = zhang_paxton(
            flat_kp[start:stop, None],
            flat_mlt[start:stop, None],
            latitude_centres[None, :],
        )

        # Keep this slice loop explicit: it is the scientific reduction and
        # costs little compared with evaluating the empirical model.
        for row, flat_index in enumerate(range(start, stop)):
            collapsed = collapse_latitude_slice(
                energy[row],
                flux[row],
                latitude_edges,
                threshold,
            )
            output_index = np.unravel_index(flat_index, kp_grid.shape)
            for name in float_fields + flag_fields:
                result[name][output_index] = collapsed[name]

    result["kp"] = kp_grid.copy()
    result["mlt"] = np.mod(mlt_grid, 24)
    result["latitude_edges"] = latitude_edges
    result["threshold"] = float(threshold)
    return result
