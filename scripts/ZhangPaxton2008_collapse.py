"""Collapse Zhang--Paxton (2008) latitude profiles to one energy per MLT.

The Zhang--Paxton model returns electron mean energy, ``E0``, and energy flux,
``Q``, as functions of Kp, magnetic local time (MLT), and magnetic latitude
(MLAT).  Some downstream IMAGE processing needs a single representative
electron energy for each ``(Kp, MLT)`` pair.  A plain mean over latitude is not
appropriate: most of a latitude slice is outside the auroral oval and carries
little or no precipitation.

This script uses a deliberately explicit three-step reduction:

1. Evaluate ``E0`` and ``Q`` at the centres of latitude cells.
2. Locate the Q maximum and retain only the contiguous cells around that
   maximum which exceed a chosen Q threshold.
3. Calculate the area-weighted mean, median, and standard deviation of E0 over
   those cells using their exact spherical-area factors,
   ``sin(latitude_upper) - sin(latitude_lower)``.

The default threshold, Q > 0.05 mW m-2, follows the inclusion criterion used
for the global-average energy in Figure 8 of Zhang and Paxton (2008). The
figures also compare Q > 0.25 mW m-2, the auroral-boundary contour used in
their map figures. Relative thresholds are deliberately not used because they
would redefine the oval according to the peak strength of each individual
slice.

The latitude grid uses 0.01-degree cells by default. The callable model is
continuous in MLT, so the collapse itself accepts arbitrary MLT values rather
than averaging across MLT bins. Diagnostic maps sample that continuous
function every 0.05 MLT hour (3 minutes).

This is an exploratory model reduction, not yet an icBuilder pipeline
integration.  In particular, ``ConductanceImage.Ep`` is proton characteristic
energy, whereas this script calculates electron mean energy.  See the project
handoff for the scientific integration questions that must be settled later.

Run headlessly from the repository root with::

    python scripts/ZhangPaxton2008_collapse.py

PNG and PDF figures are written to ``figures/``.  No production IMAGE data are
read or modified.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from zhangpaxton2008 import zhang_paxton


# Fine latitude sampling prevents the hard Q-threshold boundary from moving in
# visibly coarse 0.25-degree steps. The model functions are continuous, so this
# is numerical resolution rather than additional empirical information.
DEFAULT_MLAT_STEP_DEGREES = 0.01

# Zhang--Paxton's published Fourier representation is continuous in MLT, so
# callers may evaluate it at arbitrary local times. Diagnostic figures use
# 0.05-hour (3-minute) spacing to make within-bin variation negligible for
# display without implying additional empirical resolution.
DIAGNOSTIC_MLT_STEP_HOURS = 0.05

# Evaluate several independent MLT slices in one model call. This keeps the
# 0.01-degree grid practical while limiting temporary arrays to a modest size.
MODEL_EVALUATION_BATCH_SIZE = 64


@dataclass(frozen=True)
class OvalThreshold:
    """Absolute Q cutoff used to distinguish the oval from weak background.

    Parameters
    ----------
    value
        Energy-flux cutoff in mW m-2.
    label
        Short label used in plots and diagnostic output.
    """

    value: float
    label: str | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if not np.isfinite(self.value) or self.value < 0:
            raise ValueError("threshold value must be finite and non-negative")

    @property
    def display_label(self) -> str:
        """Human-readable representation of this threshold."""

        if self.label is not None:
            return self.label
        return f"Q > {self.value:g} mW m$^{{-2}}$"


# The default follows the pixels retained for the mean-energy panel in Figure 8.
FIGURE8_THRESHOLD = OvalThreshold(
    0.05, label=r"$Q > 0.05$ mW m$^{-2}$"
)
BOUNDARY_THRESHOLD = OvalThreshold(
    0.25, label=r"$Q > 0.25$ mW m$^{-2}$"
)
SENSITIVITY_THRESHOLDS = (
    FIGURE8_THRESHOLD,
    BOUNDARY_THRESHOLD,
)


@dataclass(frozen=True)
class LatitudeSliceCollapse:
    """Diagnostics for one collapsed MLT latitude slice.

    All energy quantities are in keV and all energy-flux quantities are in
    mW m-2.  ``selected_area_weight`` is dimensionless: it is the integral of
    cos(latitude) d(latitude) over the selected cells.  The common Earth-radius
    and MLT-bin-width factors cancel from every mean.
    """

    representative_energy: float
    area_weighted_median_energy: float
    weighted_spread: float
    q_weighted_energy: float
    selected_lower_mlat: float
    selected_upper_mlat: float
    selected_area_weight: float
    selected_area_fraction: float
    threshold_cutoff: float
    peak_mlat: float
    peak_energy_flux: float
    empty: bool
    touches_equatorward_sampling_limit: bool
    touches_poleward_sampling_limit: bool
    reaches_physical_pole: bool
    selected: NDArray[np.bool_]
    area_weights: NDArray[np.float64]


@dataclass(frozen=True)
class CollapsedModelResult:
    """Vectorized result from :func:`collapse_zhang_paxton`.

    Fields have the broadcast shape of the input ``kp`` and ``mlt`` arrays.
    A scalar call therefore returns zero-dimensional NumPy arrays, accessible
    as ordinary Python scalars with ``result.representative_energy.item()``.
    """

    kp: NDArray[np.float64]
    mlt: NDArray[np.float64]
    representative_energy: NDArray[np.float64]
    area_weighted_median_energy: NDArray[np.float64]
    weighted_spread: NDArray[np.float64]
    q_weighted_energy: NDArray[np.float64]
    selected_lower_mlat: NDArray[np.float64]
    selected_upper_mlat: NDArray[np.float64]
    selected_area_weight: NDArray[np.float64]
    selected_area_fraction: NDArray[np.float64]
    threshold_cutoff: NDArray[np.float64]
    peak_mlat: NDArray[np.float64]
    peak_energy_flux: NDArray[np.float64]
    empty: NDArray[np.bool_]
    touches_equatorward_sampling_limit: NDArray[np.bool_]
    touches_poleward_sampling_limit: NDArray[np.bool_]
    reaches_physical_pole: NDArray[np.bool_]
    latitude_edges: NDArray[np.float64]
    threshold: OvalThreshold


def regular_latitude_grid(
    lower_mlat: float = 50.0,
    upper_mlat: float = 90.0,
    step: float = DEFAULT_MLAT_STEP_DEGREES,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return cell centres and edges covering a northern-hemisphere MLAT range.

    Cell-centred sampling avoids treating the pole as if it represented a
    full-width latitude cell.  ``step`` must divide the interval exactly.
    """

    width = upper_mlat - lower_mlat
    if not np.isfinite([lower_mlat, upper_mlat, step]).all():
        raise ValueError("latitude limits and step must be finite")
    if not (0 <= lower_mlat < upper_mlat <= 90):
        raise ValueError("require 0 <= lower_mlat < upper_mlat <= 90 degrees")
    if step <= 0:
        raise ValueError("latitude step must be positive")

    cell_count = int(round(width / step))
    if cell_count < 1 or not np.isclose(cell_count * step, width):
        raise ValueError("latitude step must divide the requested range exactly")

    edges = np.linspace(lower_mlat, upper_mlat, cell_count + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres, edges


def regular_mlt_grid(
    step: float = DIAGNOSTIC_MLT_STEP_HOURS,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return cell centres and edges covering one complete MLT day.

    Zhang--Paxton uses a continuous Fourier series in MLT; this grid controls
    only the sampling used in figures and diagnostics. The model was fitted
    from 0.5-hour empirical sectors, so a finer grid gives a smooth numerical
    rendering without adding independent observational information.
    """

    if not np.isfinite(step) or step <= 0:
        raise ValueError("MLT step must be positive and finite")

    bin_count = int(round(24.0 / step))
    if bin_count < 1 or not np.isclose(bin_count * step, 24.0):
        raise ValueError("MLT step must divide 24 hours exactly")

    edges = np.linspace(0.0, 24.0, bin_count + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres, edges


def latitude_cell_area_weights(
    latitude_edges: ArrayLike,
) -> NDArray[np.float64]:
    """Return exact relative spherical areas for latitude cells.

    For a cell spanning latitudes ``lambda_1`` to ``lambda_2`` and a fixed
    longitude width, its spherical area is proportional to
    ``sin(lambda_2) - sin(lambda_1)``.  This correction matters because equal
    latitude widths represent progressively less area towards the pole.
    """

    edges = np.asarray(latitude_edges, dtype=float)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("latitude_edges must be a one-dimensional cell-edge array")
    if not np.all(np.isfinite(edges)) or not np.all(np.diff(edges) > 0):
        raise ValueError("latitude_edges must be finite and strictly increasing")
    if edges[0] < -90 or edges[-1] > 90:
        raise ValueError("latitude edges must lie between -90 and 90 degrees")

    edge_radians = np.deg2rad(edges)
    return np.sin(edge_radians[1:]) - np.sin(edge_radians[:-1])


def weighted_median(values: ArrayLike, weights: ArrayLike) -> float:
    """Return the value where cumulative positive weight reaches one half.

    Sorting by value makes this a true weighted median even when E0 is not
    monotonic with latitude. The result describes a typical unit area inside
    the selected oval; unlike the mean, it does not preserve the first moment.
    """

    value_array = np.asarray(values, dtype=float)
    weight_array = np.asarray(weights, dtype=float)
    if value_array.ndim != 1 or weight_array.ndim != 1:
        raise ValueError("values and weights must be one-dimensional")
    if value_array.shape != weight_array.shape or value_array.size == 0:
        raise ValueError("values and weights must be non-empty and the same shape")
    if not np.all(np.isfinite(value_array)):
        raise ValueError("values must be finite")
    if not np.all(np.isfinite(weight_array)) or np.any(weight_array < 0):
        raise ValueError("weights must be finite and non-negative")

    total_weight = float(np.sum(weight_array))
    if total_weight <= 0:
        raise ValueError("at least one weight must be positive")

    order = np.argsort(value_array)
    sorted_values = value_array[order]
    cumulative_weight = np.cumsum(weight_array[order])
    median_index = int(np.searchsorted(cumulative_weight, 0.5 * total_weight))
    return float(sorted_values[median_index])


def _principal_contiguous_interval(
    above_threshold: NDArray[np.bool_],
    peak_index: int,
) -> NDArray[np.bool_]:
    """Keep only the connected above-threshold component containing Q peak."""

    selected = np.zeros_like(above_threshold, dtype=bool)
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


def collapse_latitude_slice(
    mean_energy: ArrayLike,
    energy_flux: ArrayLike,
    latitude_edges: ArrayLike,
    threshold: OvalThreshold = FIGURE8_THRESHOLD,
) -> LatitudeSliceCollapse:
    """Collapse one latitude profile to an oval-conditional representative E0.

    The selected oval is the contiguous Q-above-threshold interval containing
    the principal Q maximum.  This deliberately excludes any disconnected,
    weak model lobes. The representative energy is an area-weighted
    conditional mean within that interval. The area-weighted median describes
    a typical unit area and is returned alongside the mean. A Q-weighted mean
    is also returned as a sensitivity diagnostic; it emphasizes the cells
    carrying most of the precipitation power and answers a different
    scientific question.

    Empty selections return NaN energy and latitude diagnostics and set
    ``empty=True``.  Reaching the equatorward sampling limit, or a poleward
    limit below 90 degrees, is flagged as possible truncation.  Reaching 90
    degrees is reported separately as contact with the physical pole; it is
    not a domain that can or should be extended.
    """

    energy = np.asarray(mean_energy, dtype=float)
    flux = np.asarray(energy_flux, dtype=float)
    edges = np.asarray(latitude_edges, dtype=float)
    area_weights = latitude_cell_area_weights(edges)

    if energy.ndim != 1 or flux.ndim != 1:
        raise ValueError("mean_energy and energy_flux must be one-dimensional")
    if energy.shape != flux.shape or energy.size != area_weights.size:
        raise ValueError(
            "energy, flux, and latitude-edge arrays describe different cell counts"
        )

    centres = 0.5 * (edges[:-1] + edges[1:])
    finite_flux = np.isfinite(flux)
    if not np.any(finite_flux):
        return _empty_slice_result(area_weights, np.nan, np.nan, np.nan)

    # Treat non-finite values as unavailable, never as an auroral selection.
    safe_flux = np.where(finite_flux, flux, -np.inf)
    peak_index = int(np.argmax(safe_flux))
    peak_flux = float(flux[peak_index])
    cutoff = threshold.value
    above_threshold = finite_flux & np.isfinite(energy) & (flux > cutoff)
    selected = _principal_contiguous_interval(above_threshold, peak_index)

    if not np.any(selected):
        return _empty_slice_result(
            area_weights, cutoff, float(centres[peak_index]), peak_flux
        )

    indices = np.flatnonzero(selected)
    lower_index, upper_index = int(indices[0]), int(indices[-1])
    selected_weights = area_weights[selected]
    selected_energy = energy[selected]
    area_sum = float(np.sum(selected_weights))

    representative = float(np.average(selected_energy, weights=selected_weights))
    median = weighted_median(selected_energy, selected_weights)
    variance = float(
        np.average((selected_energy - representative) ** 2, weights=selected_weights)
    )

    # Sensitivity statistic:
    #
    #     sum(area * Q * E0) / sum(area * Q)
    #
    # It answers "what energy characterizes the locations carrying most
    # modeled energy flux?" It is not the primary spatial mean and is not the
    # particle-number-weighted mean energy.
    power_weights = selected_weights * np.clip(flux[selected], 0.0, None)
    if np.sum(power_weights) > 0:
        q_weighted = float(np.average(selected_energy, weights=power_weights))
    else:
        q_weighted = np.nan

    return LatitudeSliceCollapse(
        representative_energy=representative,
        area_weighted_median_energy=median,
        weighted_spread=float(np.sqrt(max(variance, 0.0))),
        q_weighted_energy=q_weighted,
        selected_lower_mlat=float(edges[lower_index]),
        selected_upper_mlat=float(edges[upper_index + 1]),
        selected_area_weight=area_sum,
        selected_area_fraction=area_sum / float(np.sum(area_weights)),
        threshold_cutoff=float(cutoff),
        peak_mlat=float(centres[peak_index]),
        peak_energy_flux=peak_flux,
        empty=False,
        touches_equatorward_sampling_limit=lower_index == 0,
        touches_poleward_sampling_limit=(
            upper_index == energy.size - 1 and not np.isclose(edges[-1], 90.0)
        ),
        reaches_physical_pole=(
            upper_index == energy.size - 1 and np.isclose(edges[-1], 90.0)
        ),
        selected=selected,
        area_weights=area_weights,
    )


def _empty_slice_result(
    area_weights: NDArray[np.float64],
    cutoff: float,
    peak_mlat: float,
    peak_flux: float,
) -> LatitudeSliceCollapse:
    """Construct an explicit empty result without inventing a fallback value."""

    return LatitudeSliceCollapse(
        representative_energy=np.nan,
        area_weighted_median_energy=np.nan,
        weighted_spread=np.nan,
        q_weighted_energy=np.nan,
        selected_lower_mlat=np.nan,
        selected_upper_mlat=np.nan,
        selected_area_weight=0.0,
        selected_area_fraction=0.0,
        threshold_cutoff=float(cutoff),
        peak_mlat=float(peak_mlat),
        peak_energy_flux=float(peak_flux),
        empty=True,
        touches_equatorward_sampling_limit=False,
        touches_poleward_sampling_limit=False,
        reaches_physical_pole=False,
        selected=np.zeros(area_weights.size, dtype=bool),
        area_weights=area_weights,
    )


def collapse_zhang_paxton(
    kp: ArrayLike,
    mlt: ArrayLike,
    *,
    threshold: OvalThreshold = FIGURE8_THRESHOLD,
    lower_mlat: float = 50.0,
    upper_mlat: float = 90.0,
    latitude_step: float = DEFAULT_MLAT_STEP_DEGREES,
) -> CollapsedModelResult:
    """Evaluate and collapse Zhang--Paxton for broadcastable Kp and MLT inputs.

    Parameters
    ----------
    kp
        Planetary K index, in the range supported by ``zhang_paxton``.
    mlt
        Magnetic local time in hours.  The underlying implementation wraps MLT
        modulo 24.
    threshold
        Absolute Q criterion used to identify the oval.
    lower_mlat, upper_mlat, latitude_step
        Northern-hemisphere cell grid in degrees. The default 50--90 degree
        domain covers the published model and uses 0.01-degree cells. This
        fine numerical grid resolves movement of the hard Q-threshold
        boundaries; it does not add empirical resolution to the model.

    Returns
    -------
    CollapsedModelResult
        Area-weighted representative E0 and diagnostics for every broadcast
        ``(kp, mlt)`` pair.  E0 is in keV and Q is in mW m-2.
    """

    kp_grid, mlt_grid = np.broadcast_arrays(
        np.asarray(kp, dtype=float), np.asarray(mlt, dtype=float)
    )
    latitude_centres, latitude_edges = regular_latitude_grid(
        lower_mlat, upper_mlat, latitude_step
    )

    float_fields = {
        name: np.full(kp_grid.shape, np.nan, dtype=float)
        for name in (
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
        )
    }
    bool_fields = {
        name: np.zeros(kp_grid.shape, dtype=bool)
        for name in (
            "empty",
            "touches_equatorward_sampling_limit",
            "touches_poleward_sampling_limit",
            "reaches_physical_pole",
        )
    }

    flat_kp = kp_grid.ravel()
    flat_mlt = mlt_grid.ravel()

    # Batch only the model evaluation; retain the explicit row-by-row collapse
    # so the scientific selection remains easy to read. A batch has shape
    # (number of slices, number of latitude cells).
    for batch_start in range(0, flat_kp.size, MODEL_EVALUATION_BATCH_SIZE):
        batch_stop = min(
            batch_start + MODEL_EVALUATION_BATCH_SIZE,
            flat_kp.size,
        )
        mean_energy_batch, energy_flux_batch = zhang_paxton(
            flat_kp[batch_start:batch_stop, np.newaxis],
            flat_mlt[batch_start:batch_stop, np.newaxis],
            latitude_centres[np.newaxis, :],
        )

        for row, flat_index in enumerate(range(batch_start, batch_stop)):
            collapsed = collapse_latitude_slice(
                mean_energy_batch[row],
                energy_flux_batch[row],
                latitude_edges,
                threshold,
            )
            output_index = np.unravel_index(flat_index, kp_grid.shape)
            for name, values in float_fields.items():
                values[output_index] = getattr(collapsed, name)
            for name, values in bool_fields.items():
                values[output_index] = getattr(collapsed, name)

    return CollapsedModelResult(
        kp=kp_grid.copy(),
        mlt=np.mod(mlt_grid, 24.0),
        latitude_edges=latitude_edges,
        threshold=threshold,
        **float_fields,
        **bool_fields,
    )


def _model_map(
    kp: float,
    mlt_centres: NDArray[np.float64],
    latitude_centres: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate model fields on a two-dimensional MLT/MLAT centre grid."""

    mlt_grid, latitude_grid = np.meshgrid(
        mlt_centres, latitude_centres, indexing="xy"
    )
    return zhang_paxton(kp, mlt_grid, latitude_grid)


def _polar_axis(ax: object) -> None:
    """Apply the Zhang--Paxton paper's local-time orientation to a polar axis."""

    # Starting MLT at zero avoids a pcolormesh seam at +/-pi.  Putting theta=0
    # at South then gives midnight at bottom, dawn right, noon top, and dusk
    # left, matching the orientation used in the paper.
    ax.set_theta_zero_location("S")
    ax.set_theta_direction(1)
    ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
    ax.set_xticklabels(["00", "06", "12", "18"])
    ax.set_yticks([10, 20, 30, 40])
    ax.set_yticklabels([r"80°", r"70°", r"60°", r"50°"])
    # Radius is defined as 90 - MLAT, so r=0 is the magnetic pole and r=40 is
    # the 50-degree equatorward edge.  Do not reverse these limits: doing so
    # would place low latitude at the centre of the map.
    ax.set_ylim(0, 40)
    ax.grid(color="0.4", linewidth=0.5, alpha=0.55)


def _save_png_and_pdf(figure: object, output_stem: Path, dpi: int) -> None:
    """Save one figure in the two repository-supported formats."""

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    figure.savefig(
        output_stem.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight"
    )


def plot_process_map(output_dir: Path, dpi: int = 180) -> list[Path]:
    """Illustrate the 2-D fields and latitude interval selected at every MLT."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    kp = 5.0
    latitude_centres, latitude_edges = regular_latitude_grid()
    mlt_centres, mlt_edges = regular_mlt_grid()
    mean_energy, energy_flux = _model_map(kp, mlt_centres, latitude_centres)
    collapse = collapse_zhang_paxton(
        np.full_like(mlt_centres, kp), mlt_centres
    )

    selected_map = np.zeros_like(energy_flux, dtype=bool)
    for column, mlt_value in enumerate(mlt_centres):
        selected_map[:, column] = collapse_latitude_slice(
            mean_energy[:, column],
            energy_flux[:, column],
            latitude_edges,
        ).selected

    theta_edges = np.deg2rad(mlt_edges * 15.0)
    radius_edges = 90.0 - latitude_edges
    theta_centres = np.deg2rad(mlt_centres * 15.0)

    figure, axes = plt.subplots(
        1, 3, figsize=(13.6, 4.6), subplot_kw={"projection": "polar"}
    )
    for axis in axes:
        axis.grid(False)
    energy_plot = axes[0].pcolormesh(
        theta_edges,
        radius_edges,
        mean_energy,
        shading="flat",
        cmap="nipy_spectral",
        rasterized=True,
    )
    flux_plot = axes[1].pcolormesh(
        theta_edges,
        radius_edges,
        energy_flux,
        shading="flat",
        cmap="nipy_spectral",
        rasterized=True,
    )
    mask_plot = axes[2].pcolormesh(
        theta_edges,
        radius_edges,
        selected_map.astype(float),
        shading="flat",
        cmap=ListedColormap(["#e8e8e8", "#d95f02"]),
        vmin=0,
        vmax=1,
        rasterized=True,
    )
    del mask_plot

    valid = ~collapse.empty
    axes[2].plot(
        theta_centres[valid],
        90.0 - collapse.selected_lower_mlat[valid],
        color="black",
        linewidth=1.0,
        label="Selected edges",
    )
    axes[2].plot(
        theta_centres[valid],
        90.0 - collapse.selected_upper_mlat[valid],
        color="black",
        linewidth=1.0,
    )

    for axis in axes:
        _polar_axis(axis)
    axes[0].set_title(r"Model $E_0$ (keV)")
    axes[1].set_title(r"Model $Q$ (mW m$^{-2}$)")
    axes[2].set_title("Principal contiguous oval\n" + FIGURE8_THRESHOLD.display_label)
    axes[2].legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), frameon=False)
    figure.colorbar(energy_plot, ax=axes[0], pad=0.10, shrink=0.76)
    figure.colorbar(flux_plot, ax=axes[1], pad=0.10, shrink=0.76)
    figure.suptitle(
        "Zhang–Paxton latitude collapse: select the Q-defined oval before averaging "
        f"(Kp={kp:g})",
        y=1.02,
    )
    figure.text(
        0.5,
        0.01,
        "Noon is at top; dawn is right. Orange cells alone contribute to the "
        "representative energy.",
        ha="center",
        fontsize=9,
    )

    output_stem = output_dir / "zhang_paxton_collapse_process_map"
    _save_png_and_pdf(figure, output_stem, dpi)
    plt.close(figure)
    return [output_stem.with_suffix(".png"), output_stem.with_suffix(".pdf")]


def plot_latitude_slice(output_dir: Path, dpi: int = 180) -> list[Path]:
    """Show the selection, weights, and representative values for one slice."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    kp, mlt = 5.0, 0.0
    latitude_centres, latitude_edges = regular_latitude_grid()
    mean_energy, energy_flux = zhang_paxton(kp, mlt, latitude_centres)

    results = [
        collapse_latitude_slice(
            mean_energy, energy_flux, latitude_edges, threshold
        )
        for threshold in SENSITIVITY_THRESHOLDS
    ]
    default = results[0]

    figure, axes = plt.subplots(
        3, 1, figsize=(9.0, 8.2), sharex=True, constrained_layout=True
    )

    axes[0].plot(latitude_centres, energy_flux, color="#2166ac", linewidth=2)
    for result, threshold, color in zip(
        results, SENSITIVITY_THRESHOLDS, ("#d95f02", "#1b9e77")
    ):
        axes[0].axhline(
            result.threshold_cutoff,
            color=color,
            linestyle="--",
            linewidth=1.2,
            label=threshold.display_label,
        )
        if not result.empty:
            axes[0].axvspan(
                result.selected_lower_mlat,
                result.selected_upper_mlat,
                color=color,
                alpha=0.10,
            )
    axes[0].scatter(
        [default.peak_mlat],
        [default.peak_energy_flux],
        color="black",
        marker="x",
        zorder=5,
        label="Principal Q maximum",
    )
    axes[0].set_ylabel(r"$Q$ (mW m$^{-2}$)")
    axes[0].set_title(
        "Step 1: threshold Q and keep the connected interval around its maximum"
    )
    axes[0].legend(fontsize=8, ncol=2)

    axes[1].plot(latitude_centres, mean_energy, color="0.20", linewidth=2)
    axes[1].fill_between(
        latitude_centres,
        0,
        mean_energy,
        where=default.selected,
        color="#d95f02",
        alpha=0.25,
        label="Cells used by default",
    )
    axes[1].axhline(
        default.representative_energy,
        color="#d95f02",
        linewidth=2,
        label=(
            "Area-weighted mean "
            f"= {default.representative_energy:.2f} keV"
        ),
    )
    axes[1].axhline(
        default.area_weighted_median_energy,
        color="#1b9e77",
        linestyle="--",
        linewidth=2,
        label=(
            "Area-weighted median "
            f"= {default.area_weighted_median_energy:.2f} keV"
        ),
    )
    axes[1].axhline(
        default.q_weighted_energy,
        color="#2166ac",
        linestyle=":",
        linewidth=2,
        label=(
            r"Area $\times$ Q-weighted mean (diagnostic) "
            f"= {default.q_weighted_energy:.2f} keV"
        ),
    )
    axes[1].set_ylabel(r"$E_0$ (keV)")
    axes[1].set_title(
        "Step 2: calculate area-weighted mean and median inside the selected oval"
    )
    axes[1].legend(fontsize=8)

    relative_weights = default.area_weights / np.max(default.area_weights)
    axes[2].fill_between(
        latitude_centres,
        0,
        relative_weights,
        color="0.80",
        linewidth=0,
    )
    axes[2].fill_between(
        latitude_centres,
        0,
        relative_weights,
        where=default.selected,
        color="#d95f02",
        linewidth=0,
    )
    axes[2].set_ylabel("Relative cell area")
    axes[2].set_xlabel("Magnetic latitude (degrees)")
    axes[2].set_title(
        r"Spherical cell-area weights used in Step 2: "
        r"$\sin(\lambda_\mathrm{upper})-\sin(\lambda_\mathrm{lower})$"
    )
    axes[2].set_xlim(latitude_edges[0], latitude_edges[-1])
    axes[2].set_ylim(0, 1.05)
    for axis in axes:
        axis.grid(alpha=0.25)

    figure.suptitle(
        f"One latitude slice: Kp={kp:g}, MLT={mlt:02.0f}", fontsize=14
    )
    output_stem = output_dir / "zhang_paxton_collapse_latitude_slice"
    _save_png_and_pdf(figure, output_stem, dpi)
    plt.close(figure)
    return [output_stem.with_suffix(".png"), output_stem.with_suffix(".pdf")]


def _save_collapsed_metric_figure(
    output_dir: Path,
    dpi: int,
    *,
    kp_values: NDArray[np.float64],
    mlt_centres: NDArray[np.float64],
    mlt_edges: NDArray[np.float64],
    metric_grid: NDArray[np.float64],
    sensitivity_curves: list[
        tuple[str, str, OvalThreshold, NDArray[np.float64]]
    ],
    colorbar_label: str,
    map_title: str,
    sensitivity_title: str,
    y_label: str,
    output_name: str,
) -> list[Path]:
    """Save one Kp/MLT collapse metric and its threshold sensitivity.

    The representative energy and its profile-derived uncertainty use the
    same layout so they can be compared directly. Model evaluation happens
    before this plotting helper, allowing both figures to reuse the expensive
    0.01-degree latitude collapses.
    """

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        2, 1, figsize=(10.2, 8.6), constrained_layout=True
    )
    heatmap = axes[0].pcolormesh(
        mlt_edges,
        np.arange(-0.5, 10.5, 1.0),
        metric_grid,
        shading="flat",
        cmap="nipy_spectral",
        rasterized=True,
    )
    colorbar = figure.colorbar(heatmap, ax=axes[0], pad=0.02)
    colorbar.set_label(colorbar_label)
    axes[0].set(
        title=map_title,
        xlabel="Magnetic local time (hours)",
        ylabel="Kp",
        xlim=(0, 24),
        yticks=kp_values,
    )

    linestyles = ("-", "--", ":")
    for curve_index, (kp_label, color, threshold, values) in enumerate(
        sensitivity_curves
    ):
        # Thresholds repeat for each Kp value, so restart the line-style cycle
        # for every pair of Kp curves.
        linestyle = linestyles[curve_index % len(SENSITIVITY_THRESHOLDS)]
        axes[1].plot(
            mlt_centres,
            values,
            color=color,
            linestyle=linestyle,
            linewidth=2,
            label=f"{kp_label}; {threshold.display_label}",
        )
    axes[1].set(
        title=sensitivity_title,
        xlabel="Magnetic local time (hours)",
        ylabel=y_label,
        xlim=(0, 24),
    )
    axes[1].legend(fontsize=8, ncol=2)
    for axis in axes:
        axis.set_xticks(np.arange(0, 25, 3))
        axis.grid(alpha=0.25)

    output_stem = output_dir / output_name
    _save_png_and_pdf(figure, output_stem, dpi)
    plt.close(figure)
    return [output_stem.with_suffix(".png"), output_stem.with_suffix(".pdf")]


def plot_collapsed_result(output_dir: Path, dpi: int = 180) -> list[Path]:
    """Plot the collapsed E0 mean and profile-derived dE0 companion figure."""

    import matplotlib

    matplotlib.use("Agg", force=True)

    kp_values = np.arange(10.0)
    mlt_centres, mlt_edges = regular_mlt_grid()
    kp_grid, mlt_grid = np.meshgrid(kp_values, mlt_centres, indexing="ij")
    default = collapse_zhang_paxton(kp_grid, mlt_grid)

    colors = {"Kp=2": "#2166ac", "Kp=5": "#b2182b"}
    sensitivity_results: list[
        tuple[str, str, OvalThreshold, CollapsedModelResult]
    ] = []
    for kp_value, (kp_label, color) in zip((2.0, 5.0), colors.items()):
        for threshold in SENSITIVITY_THRESHOLDS:
            result = collapse_zhang_paxton(
                np.full_like(mlt_centres, kp_value),
                mlt_centres,
                threshold=threshold,
            )
            sensitivity_results.append((kp_label, color, threshold, result))

    mean_curves = [
        (kp_label, color, threshold, result.representative_energy)
        for kp_label, color, threshold, result in sensitivity_results
    ]
    spread_curves = [
        (kp_label, color, threshold, result.weighted_spread)
        for kp_label, color, threshold, result in sensitivity_results
    ]

    outputs = _save_collapsed_metric_figure(
        output_dir,
        dpi,
        kp_values=kp_values,
        mlt_centres=mlt_centres,
        mlt_edges=mlt_edges,
        metric_grid=default.representative_energy,
        sensitivity_curves=mean_curves,
        colorbar_label=r"Representative $E_0$ (keV)",
        map_title="End result: oval-conditional, area-weighted electron mean energy",
        sensitivity_title="Sensitivity to the definition of the auroral oval",
        y_label=r"Representative $E_0$ (keV)",
        output_name="zhang_paxton_collapse_result",
    )
    outputs.extend(
        _save_collapsed_metric_figure(
            output_dir,
            dpi,
            kp_values=kp_values,
            mlt_centres=mlt_centres,
            mlt_edges=mlt_edges,
            metric_grid=default.weighted_spread,
            sensitivity_curves=spread_curves,
            colorbar_label=r"Profile-derived $dE_0$ (keV)",
            map_title=(
                "End result: latitude-profile spread discarded by the collapse"
            ),
            sensitivity_title=(
                r"Sensitivity of profile-derived $dE_0$ to the oval definition"
            ),
            y_label=r"Profile-derived $dE_0$ (keV)",
            output_name="zhang_paxton_collapse_dE0_result",
        )
    )
    return outputs


def run_internal_checks() -> None:
    """Run deterministic invariants before creating explanatory figures."""

    centres, edges = regular_latitude_grid(50.0, 90.0, 0.25)
    weights = latitude_cell_area_weights(edges)
    expected_total = np.sin(np.deg2rad(90.0)) - np.sin(np.deg2rad(50.0))
    if not np.isclose(np.sum(weights), expected_total, rtol=0, atol=1e-14):
        raise RuntimeError("latitude-cell area weights do not integrate correctly")

    # A synthetic disconnected lobe verifies that only the component around
    # the principal maximum is retained.
    synthetic_flux = np.zeros(centres.size)
    synthetic_flux[20:25] = 0.10
    synthetic_flux[80:91] = np.linspace(0.10, 1.0, 11)
    synthetic_energy = 1.0 + centres / 90.0
    result = collapse_latitude_slice(
        synthetic_energy,
        synthetic_flux,
        edges,
        OvalThreshold(0.05),
    )
    if not np.array_equal(np.flatnonzero(result.selected), np.arange(80, 91)):
        raise RuntimeError("principal contiguous-oval selection check failed")

    # Exercise scalar and broadcast calls against the installed model package.
    scalar = collapse_zhang_paxton(2.0, 0.0)
    grid = collapse_zhang_paxton(np.array([[2.0], [5.0]]), np.arange(0.0, 24.0, 6.0))
    if scalar.representative_energy.shape != ():
        raise RuntimeError("scalar collapse did not preserve scalar shape")
    if grid.representative_energy.shape != (2, 4):
        raise RuntimeError("Kp/MLT broadcasting check failed")
    if scalar.empty.item() or not np.isfinite(scalar.representative_energy.item()):
        raise RuntimeError("reference model slice unexpectedly produced no result")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for deterministic figure generation."""

    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository_root / "figures",
        help="directory for PNG and PDF outputs (default: repository figures/)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="PNG resolution in dots per inch (default: 180)",
    )
    return parser.parse_args()


def main() -> None:
    """Validate the reduction, generate figures, and summarize diagnostics."""

    args = parse_args()
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")

    run_internal_checks()
    outputs: list[Path] = []
    outputs.extend(plot_process_map(args.output_dir, args.dpi))
    outputs.extend(plot_latitude_slice(args.output_dir, args.dpi))
    outputs.extend(plot_collapsed_result(args.output_dir, args.dpi))

    diagnostic_grid = collapse_zhang_paxton(
        np.arange(10.0)[:, None],
        regular_mlt_grid()[0][None, :],
    )
    mlt_count = diagnostic_grid.mlt.shape[1]
    slice_count = diagnostic_grid.mlt.size
    print("Generated Zhang–Paxton collapse diagnostics:")
    for output in outputs:
        print(f"  {output}")
    print(
        "Default-grid flags for Kp=0,...,9 and "
        f"{mlt_count} MLT samples spaced by {DIAGNOSTIC_MLT_STEP_HOURS:g} hours "
        f"({slice_count} slices; {DEFAULT_MLAT_STEP_DEGREES:g}-degree MLAT "
        "cells cover 50--90 degrees): "
        f"{np.count_nonzero(diagnostic_grid.empty)} empty; "
        f"{np.count_nonzero(diagnostic_grid.touches_equatorward_sampling_limit)} "
        "touch the 50-degree equatorward sampling limit (possible truncation); "
        f"{np.count_nonzero(diagnostic_grid.touches_poleward_sampling_limit)} "
        "touch a non-polar upper sampling limit; "
        f"{np.count_nonzero(diagnostic_grid.reaches_physical_pole)} "
        "reach the 90-degree physical pole (not sampling truncation)."
    )


if __name__ == "__main__":
    main()
