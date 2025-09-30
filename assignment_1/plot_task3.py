from __future__ import annotations

import argparse
import os
import re
from typing import Dict, List, Tuple

import matplotlib
# Use non-interactive backend for headless environments (e.g., HPC nodes)
matplotlib.use("Agg")
import matplotlib.pyplot as plt


TTFT_RE = re.compile(r"TTFT:\s*([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)")
TPOT_RE = re.compile(r"TPOT:\s*([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)")


def parse_metrics(path: str) -> Tuple[List[float], List[float]]:
    """Parse TTFT and TPOT arrays from a benchmark results text file.

    Returns:
        (ttft_list, tpot_list)
    """
    ttfts: List[float] = []
    tpots: List[float] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            # Skip empty or comment lines
            if not line.strip() or line.lstrip().startswith("#"):
                continue

            ttft_match = TTFT_RE.search(line)
            tpot_match = TPOT_RE.search(line)
            if ttft_match:
                try:
                    ttfts.append(float(ttft_match.group(1)))
                except ValueError:
                    pass  # ignore malformed number
            if tpot_match:
                try:
                    tpots.append(float(tpot_match.group(1)))
                except ValueError:
                    pass

    return ttfts, tpots


def _style_for(idx: int, mode: str = "colorblind") -> Dict[str, object]:
    """Return a dict of matplotlib style kwargs for the given index and mode.

    Modes:
      - colorblind: Okabe-Ito palette + distinct dash patterns
      - mono: grayscale (black) with distinct dash patterns
    """
    # Okabe-Ito colorblind-friendly palette (prioritized for high contrast on white)
    okabe_ito = [
        "#000000",  # black
        "#0072B2",  # blue
        "#D55E00",  # vermillion
        "#009E73",  # green
        "#CC79A7",  # magenta
        "#E69F00",  # orange
        "#56B4E9",  # sky blue
        "#F0E442",  # yellow
    ]

    # A set of distinct dash patterns to differentiate lines even in grayscale
    dash_patterns = [
        "-",            # solid
        "--",           # dashed
        "-.",           # dash-dot
        ":",            # dotted
        (0, (3, 1, 1, 1)),  # custom short dash pattern
        (0, (5, 2)),        # longer dashes
    ]

    # Optional markers for extra differentiation (set in plot function)
    markers = ["o", "s", "^", "D", "x", "v", "P", "*"]

    color = okabe_ito[idx % len(okabe_ito)] if mode == "colorblind" else "#000000"
    linestyle = dash_patterns[idx % len(dash_patterns)]
    marker = markers[idx % len(markers)]

    return {
        "color": color,
        "linestyle": linestyle,
        "marker": marker,
    }


def plot_sorted(
    series: Dict[str, List[float]],
    ylabel: str,
    title: str,
    outpath: str,
    style_mode: str = "colorblind",
    use_markers: bool = False,
) -> None:
    """Plot sorted series per label and save to outpath."""
    plt.figure(figsize=(8.5, 5.2))

    # Slightly larger font for readability
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
    })

    for idx, (label, values) in enumerate(series.items()):
        if not values:
            continue
        sorted_vals = sorted(values)
        x = list(range(1, len(sorted_vals) + 1))

        style = _style_for(idx, mode=style_mode)
        kwargs = {
            "linewidth": 2.5,
            "label": label,
            "linestyle": style["linestyle"],
            "color": style["color"],
        }

        if use_markers:
            # Place markers sparsely to avoid clutter for long series
            markevery = max(1, len(x) // 25)
            kwargs.update({
                "marker": style["marker"],
                "markersize": 5,
                "markevery": markevery,
                "markerfacecolor": kwargs["color"],
                "markeredgewidth": 0.0,
            })

        plt.plot(x, sorted_vals, **kwargs)

    plt.xlabel("Prompt Index (sorted)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(title="Configuration")
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    print(f"Saved: {outpath}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot sorted TTFT and TPOT across configurations")
    parser.add_argument(
        "--data",
        action="append",
        metavar="LABEL=PATH",
        help=(
            "Add a dataset as 'LABEL=PATH'. Can be repeated. "
            "If omitted, uses defaults: 1GPU, 2GPU, 4GPU in current directory."
        ),
    )
    parser.add_argument("--outdir", default="./figures", help="Directory to save plots")
    parser.add_argument(
        "--style",
        choices=["colorblind", "mono"],
        default="colorblind",
        help="Line styling: colorblind palette or monochrome dashes",
    )
    parser.add_argument(
        "--markers",
        action="store_true",
        help="Add sparse markers for additional differentiation",
    )
    args = parser.parse_args()

    if args.data:
        datasets: Dict[str, str] = {}
        for spec in args.data:
            if "=" not in spec:
                parser.error(f"Invalid --data spec '{spec}', expected LABEL=PATH")
            label, path = spec.split("=", 1)
            label = label.strip()
            path = path.strip()
            if not label or not path:
                parser.error(f"Invalid --data spec '{spec}', expected LABEL=PATH")
            datasets[label] = path
    else:
        # Defaults relative to script directory
        here = os.path.dirname(os.path.abspath(__file__))
        datasets = {
            "1 GPU": os.path.join(here, "ttft_tpot_1gpu.txt"),
            "2 GPUs": os.path.join(here, "ttft_tpot_2gpu.txt"),
            "4 GPUs": os.path.join(here, "ttft_tpot_4gpu.txt"),
        }

    # Parse
    ttft_series: Dict[str, List[float]] = {}
    tpot_series: Dict[str, List[float]] = {}

    for label, path in datasets.items():
        if not os.path.isfile(path):
            print(f"Warning: file not found for '{label}': {path}")
            ttft_series[label] = []
            tpot_series[label] = []
            continue

        ttfts, tpots = parse_metrics(path)
        if not ttfts and not tpots:
            print(f"Warning: no metrics parsed for '{label}' from {path}")
        ttft_series[label] = ttfts
        tpot_series[label] = tpots

    # Ensure output directory exists
    os.makedirs(args.outdir, exist_ok=True)

    # Plot TTFT
    ttft_path = os.path.join(args.outdir, "plot_ttft_sorted.png")
    plot_sorted(
        ttft_series,
        ylabel="TTFT (s)",
        title="Sorted TTFT per Configuration",
        outpath=ttft_path,
        style_mode=args.style,
        use_markers=args.markers,
    )

    # Plot TPOT
    tpot_path = os.path.join(args.outdir, "plot_tpot_sorted.png")
    plot_sorted(
        tpot_series,
        ylabel="TPOT (s/token)",
        title="Sorted TPOT per Configuration",
        outpath=tpot_path,
        style_mode=args.style,
        use_markers=args.markers,
    )


if __name__ == "__main__":
    main()

    """
    Example Command:
    python plot_task3.py \
        --data "1 GPU=./ttft_tpot_1gpu.txt" \
        --data "2 GPUs=./ttft_tpot_2gpu.txt" \
        --data "4 GPUs=./ttft_tpot_4gpu.txt" \
        --outdir ./figures
    """
