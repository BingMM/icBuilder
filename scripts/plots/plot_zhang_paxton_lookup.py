"""Plot one Zhang--Paxton lookup layer in Cubed-Sphere coordinates."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from icbuilder.zhang_paxton_lookup import (
    DEFAULT_LOOKUP_PATH,
    load_zhang_paxton_lookup,
)


def main():
    repository_root = Path(__file__).resolve().parents[1]
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--lookup", type=Path, default=DEFAULT_LOOKUP_PATH)
    parser.add_argument("--kp", type=float, default=1.519)
    parser.add_argument(
        "--output",
        type=Path,
        default=repository_root / "figures" / "zhang_paxton_lookup_kp1_52",
    )
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    lookup = load_zhang_paxton_lookup(args.kp, args.lookup)
    kp_used = float(lookup["kp"])
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(13.2, 4.2),
        constrained_layout=True,
    )
    panels = (
        (lookup["mlt"], "MLT coordinate", "MLT (hours)"),
        (lookup["E0"], rf"$E_0$ at Kp={kp_used:.2f}", r"$E_0$ (keV)"),
        (
            lookup["dE0"],
            rf"Profile spread at Kp={kp_used:.2f}",
            r"$dE_0$ (keV)",
        ),
    )
    for axis, (values, title, colorbar_label) in zip(axes, panels):
        image = axis.pcolormesh(
            lookup["xi"],
            lookup["eta"],
            values,
            shading="nearest",
            cmap="viridis",
            rasterized=True,
        )
        axis.set(
            title=title,
            xlabel=r"$\xi$ (radians)",
            ylabel=r"$\eta$ (radians)",
            aspect="equal",
        )
        figure.colorbar(image, ax=axis, label=colorbar_label, shrink=0.82)

    figure.suptitle(
        "Zhang–Paxton lookup on the 36×36 IMAGE Cubed-Sphere grid"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output.with_suffix(".png"), dpi=args.dpi)
    figure.savefig(args.output.with_suffix(".pdf"))
    plt.close(figure)
    print(args.output.with_suffix(".png"))
    print(args.output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
