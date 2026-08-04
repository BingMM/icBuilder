"""Make explanatory figures for the Zhang--Paxton latitude collapse.

The reusable scientific calculation is in
``icbuilder/zhang_paxton_collapse.py``.  This script contains only diagnostic
sampling, plotting, and command-line code.

Run from the repository root with::

    python scripts/ZhangPaxton2008_collapse.py
"""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from zhangpaxton2008 import zhang_paxton

from icbuilder.zhang_paxton_collapse import (
    BOUNDARY_THRESHOLD,
    DEFAULT_MLAT_STEP_DEGREES,
    FIGURE8_THRESHOLD,
    collapse_latitude_slice,
    collapse_zhang_paxton,
    latitude_cell_area_weights,
    regular_latitude_grid,
)


# The fitted equations are continuous in MLT.  Sampling every 0.05 hour gives
# smooth diagnostic curves without claiming better empirical resolution.
DIAGNOSTIC_MLT_STEP_HOURS = 0.05
THRESHOLDS = [
    (FIGURE8_THRESHOLD, r"$Q > 0.05$ mW m$^{-2}$"),
    (BOUNDARY_THRESHOLD, r"$Q > 0.25$ mW m$^{-2}$"),
]


def regular_mlt_grid(step=DIAGNOSTIC_MLT_STEP_HOURS):
    """Return MLT cell centres and edges covering one day."""

    if not np.isfinite(step) or step <= 0:
        raise ValueError("MLT step must be positive and finite")
    number_of_cells = int(round(24 / step))
    if number_of_cells < 1 or not np.isclose(number_of_cells * step, 24):
        raise ValueError("MLT step must divide 24 hours exactly")

    edges = np.linspace(0, 24, number_of_cells + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    return centres, edges


def model_map(kp, mlt_centres, latitude_centres):
    """Evaluate E0 and Q on an MLT by MLAT grid."""

    mlt, latitude = np.meshgrid(
        mlt_centres,
        latitude_centres,
        indexing="xy",
    )
    return zhang_paxton(kp, mlt, latitude)


def format_polar_axis(axis):
    """Use the local-time orientation in Zhang and Paxton (2008)."""

    axis.set_theta_zero_location("S")
    axis.set_theta_direction(1)
    axis.set_xticks(np.deg2rad([0, 90, 180, 270]))
    axis.set_xticklabels(["00", "06", "12", "18"])
    axis.set_yticks([10, 20, 30, 40])
    axis.set_yticklabels([r"80°", r"70°", r"60°", r"50°"])
    # Radius is 90 - MLAT: the pole is at the centre, 50 degrees at the edge.
    axis.set_ylim(0, 40)
    axis.grid(color="0.4", linewidth=0.5, alpha=0.55)


def save_figure(figure, output_dir, name, dpi):
    """Save one diagnostic as PNG and publication-quality PDF."""

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"{name}.png", output_dir / f"{name}.pdf"]
    figure.savefig(paths[0], dpi=dpi, bbox_inches="tight")
    figure.savefig(paths[1], dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return paths


def plot_process_map(output_dir, dpi=180):
    """Show model E0, model Q, and the selected oval cells."""

    kp = 5.0
    latitude, latitude_edges = regular_latitude_grid()
    mlt, mlt_edges = regular_mlt_grid()
    energy, flux = model_map(kp, mlt, latitude)
    collapsed = collapse_zhang_paxton(np.full(mlt.shape, kp), mlt)

    selected = np.zeros(flux.shape, dtype=bool)
    for column in range(mlt.size):
        selected[:, column] = collapse_latitude_slice(
            energy[:, column],
            flux[:, column],
            latitude_edges,
        )["selected"]

    theta_edges = np.deg2rad(mlt_edges * 15)
    radius_edges = 90 - latitude_edges
    theta = np.deg2rad(mlt * 15)

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(13.6, 4.6),
        subplot_kw={"projection": "polar"},
    )
    for axis in axes:
        axis.grid(False)

    energy_plot = axes[0].pcolormesh(
        theta_edges,
        radius_edges,
        energy,
        shading="flat",
        cmap="nipy_spectral",
        rasterized=True,
    )
    flux_plot = axes[1].pcolormesh(
        theta_edges,
        radius_edges,
        flux,
        shading="flat",
        cmap="nipy_spectral",
        rasterized=True,
    )
    axes[2].pcolormesh(
        theta_edges,
        radius_edges,
        selected.astype(float),
        shading="flat",
        cmap=ListedColormap(["#e8e8e8", "#d95f02"]),
        vmin=0,
        vmax=1,
        rasterized=True,
    )

    valid = ~collapsed["empty"]
    axes[2].plot(
        theta[valid],
        90 - collapsed["selected_lower_mlat"][valid],
        color="black",
        linewidth=1,
        label="Selected edges",
    )
    axes[2].plot(
        theta[valid],
        90 - collapsed["selected_upper_mlat"][valid],
        color="black",
        linewidth=1,
    )

    for axis in axes:
        format_polar_axis(axis)
    axes[0].set_title(r"Model $E_0$ (keV)")
    axes[1].set_title(r"Model $Q$ (mW m$^{-2}$)")
    axes[2].set_title(
        "Principal contiguous oval\n"
        + r"$Q > 0.05$ mW m$^{-2}$"
    )
    axes[2].legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), frameon=False)
    figure.colorbar(energy_plot, ax=axes[0], pad=0.10, shrink=0.76)
    figure.colorbar(flux_plot, ax=axes[1], pad=0.10, shrink=0.76)
    figure.suptitle(
        "Zhang–Paxton latitude collapse: select the Q-defined oval before "
        f"averaging (Kp={kp:g})",
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
    return save_figure(
        figure,
        output_dir,
        "zhang_paxton_collapse_process_map",
        dpi,
    )


def plot_latitude_slice(output_dir, dpi=180):
    """Explain selection and weighting for one latitude profile."""

    kp = 5.0
    mlt = 0.0
    latitude, latitude_edges = regular_latitude_grid()
    energy, flux = zhang_paxton(kp, mlt, latitude)
    results = [
        collapse_latitude_slice(energy, flux, latitude_edges, threshold)
        for threshold, label in THRESHOLDS
    ]
    default = results[0]

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(9, 8.2),
        sharex=True,
        constrained_layout=True,
    )

    axes[0].plot(latitude, flux, color="#2166ac", linewidth=2)
    for result, (threshold, label), color in zip(
        results,
        THRESHOLDS,
        ("#d95f02", "#1b9e77"),
    ):
        axes[0].axhline(
            threshold,
            color=color,
            linestyle="--",
            linewidth=1.2,
            label=label,
        )
        if not result["empty"]:
            axes[0].axvspan(
                result["selected_lower_mlat"],
                result["selected_upper_mlat"],
                color=color,
                alpha=0.10,
            )
    axes[0].scatter(
        [default["peak_mlat"]],
        [default["peak_energy_flux"]],
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

    axes[1].plot(latitude, energy, color="0.20", linewidth=2)
    axes[1].fill_between(
        latitude,
        0,
        energy,
        where=default["selected"],
        color="#d95f02",
        alpha=0.25,
        label="Cells used by default",
    )
    axes[1].axhline(
        default["representative_energy"],
        color="#d95f02",
        linewidth=2,
        label=(
            "Area-weighted mean "
            f'= {default["representative_energy"]:.2f} keV'
        ),
    )
    axes[1].axhline(
        default["area_weighted_median_energy"],
        color="#1b9e77",
        linestyle="--",
        linewidth=2,
        label=(
            "Area-weighted median "
            f'= {default["area_weighted_median_energy"]:.2f} keV'
        ),
    )
    axes[1].axhline(
        default["q_weighted_energy"],
        color="#2166ac",
        linestyle=":",
        linewidth=2,
        label=(
            r"Area $\times$ Q-weighted mean (diagnostic) "
            f'= {default["q_weighted_energy"]:.2f} keV'
        ),
    )
    axes[1].set_ylabel(r"$E_0$ (keV)")
    axes[1].set_title(
        "Step 2: calculate area-weighted mean and median inside the selected oval"
    )
    axes[1].legend(fontsize=8)

    relative_area = default["area_weights"] / default["area_weights"].max()
    axes[2].fill_between(
        latitude,
        0,
        relative_area,
        color="0.80",
        linewidth=0,
    )
    axes[2].fill_between(
        latitude,
        0,
        relative_area,
        where=default["selected"],
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
        f"One latitude slice: Kp={kp:g}, MLT={mlt:02.0f}",
        fontsize=14,
    )
    return save_figure(
        figure,
        output_dir,
        "zhang_paxton_collapse_latitude_slice",
        dpi,
    )


def save_metric_figure(
    output_dir,
    dpi,
    kp_values,
    mlt,
    mlt_edges,
    metric,
    sensitivity_curves,
    colorbar_label,
    map_title,
    sensitivity_title,
    y_label,
    output_name,
):
    """Plot one collapsed metric over Kp/MLT and for two oval thresholds."""

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(10.2, 8.6),
        constrained_layout=True,
    )
    heatmap = axes[0].pcolormesh(
        mlt_edges,
        np.arange(-0.5, 10.5),
        metric,
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

    for index, (kp_label, color, threshold_label, values) in enumerate(
        sensitivity_curves
    ):
        axes[1].plot(
            mlt,
            values,
            color=color,
            linestyle=("-", "--")[index % 2],
            linewidth=2,
            label=f"{kp_label}; {threshold_label}",
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

    return save_figure(figure, output_dir, output_name, dpi)


def plot_collapsed_results(output_dir, dpi=180):
    """Plot the representative E0 and profile-derived dE0."""

    kp_values = np.arange(10.0)
    mlt, mlt_edges = regular_mlt_grid()
    kp_grid, mlt_grid = np.meshgrid(kp_values, mlt, indexing="ij")
    default = collapse_zhang_paxton(kp_grid, mlt_grid)

    colors = [(2.0, "Kp=2", "#2166ac"), (5.0, "Kp=5", "#b2182b")]
    sensitivity = []
    for kp, kp_label, color in colors:
        for threshold, threshold_label in THRESHOLDS:
            result = collapse_zhang_paxton(
                np.full(mlt.shape, kp),
                mlt,
                threshold=threshold,
            )
            sensitivity.append(
                (kp_label, color, threshold_label, result)
            )

    mean_curves = [
        (kp_label, color, label, result["representative_energy"])
        for kp_label, color, label, result in sensitivity
    ]
    spread_curves = [
        (kp_label, color, label, result["weighted_spread"])
        for kp_label, color, label, result in sensitivity
    ]

    outputs = save_metric_figure(
        output_dir,
        dpi,
        kp_values,
        mlt,
        mlt_edges,
        default["representative_energy"],
        mean_curves,
        r"Representative $E_0$ (keV)",
        "End result: oval-conditional, area-weighted electron mean energy",
        "Sensitivity to the definition of the auroral oval",
        r"Representative $E_0$ (keV)",
        "zhang_paxton_collapse_result",
    )
    outputs += save_metric_figure(
        output_dir,
        dpi,
        kp_values,
        mlt,
        mlt_edges,
        default["weighted_spread"],
        spread_curves,
        r"Profile-derived $dE_0$ (keV)",
        "End result: latitude-profile spread discarded by the collapse",
        r"Sensitivity of profile-derived $dE_0$ to the oval definition",
        r"Profile-derived $dE_0$ (keV)",
        "zhang_paxton_collapse_dE0_result",
    )
    return outputs


def check_calculation():
    """Check the area formula and disconnected-lobe selection."""

    latitude, edges = regular_latitude_grid(50, 90, 0.25)
    weights = latitude_cell_area_weights(edges)
    exact_area = np.sin(np.deg2rad(90)) - np.sin(np.deg2rad(50))
    if not np.isclose(weights.sum(), exact_area, rtol=0, atol=1e-14):
        raise RuntimeError("latitude-cell area weights do not integrate correctly")

    flux = np.zeros(latitude.size)
    flux[20:25] = 0.10
    flux[80:91] = np.linspace(0.10, 1.0, 11)
    energy = 1 + latitude / 90
    result = collapse_latitude_slice(energy, flux, edges)
    if not np.array_equal(np.flatnonzero(result["selected"]), np.arange(80, 91)):
        raise RuntimeError("principal contiguous-oval selection check failed")


def main():
    repository_root = Path(__file__).resolve().parents[1]
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository_root / "figures",
    )
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")

    check_calculation()
    outputs = []
    outputs += plot_process_map(args.output_dir, args.dpi)
    outputs += plot_latitude_slice(args.output_dir, args.dpi)
    outputs += plot_collapsed_results(args.output_dir, args.dpi)

    diagnostic = collapse_zhang_paxton(
        np.arange(10.0)[:, None],
        regular_mlt_grid()[0][None, :],
    )
    print("Generated Zhang–Paxton collapse diagnostics:")
    for output in outputs:
        print(f"  {output}")
    print(
        f"{diagnostic['empty'].size} slices: "
        f"{np.count_nonzero(diagnostic['empty'])} empty; "
        f"{np.count_nonzero(diagnostic['touches_equatorward_sampling_limit'])} "
        "touch the 50-degree equatorward limit; "
        f"{np.count_nonzero(diagnostic['touches_poleward_sampling_limit'])} "
        "touch a non-polar upper limit; "
        f"{np.count_nonzero(diagnostic['reaches_physical_pole'])} "
        "reach the physical pole."
    )


if __name__ == "__main__":
    main()
