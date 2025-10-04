#!/usr/bin/env python3
"""Compare TTFT/TPOT and GPU KV cache usage for default vs. CFS schedulers."""
from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt


@dataclass
class PromptMetrics:
    prompt_index: int
    ttft_s: float
    tpot_s: float


TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S,%f"
KV_USAGE_PATTERN = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*GPU KV cache usage: (?P<usage>[0-9]+\.?[0-9]*)%"
)


def load_prompt_metrics(csv_path: Path) -> Dict[int, PromptMetrics]:
    metrics: Dict[int, PromptMetrics] = {}
    with csv_path.open(newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        required = {"prompt_index", "ttft_ms", "tpot_ms"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            missing_fields = ", ".join(sorted(missing))
            raise ValueError(f"Missing columns in {csv_path}: {missing_fields}")
        for row in reader:
            index = int(row["prompt_index"])
            ttft_ms = row.get("ttft_ms", "").strip()
            tpot_ms = row.get("tpot_ms", "").strip()
            ttft_s = float(ttft_ms) / 1000 if ttft_ms else float("nan")
            tpot_s = float(tpot_ms) / 1000 if tpot_ms else float("nan")
            metrics[index] = PromptMetrics(index, ttft_s, tpot_s)
    if not metrics:
        raise ValueError(f"No prompt metrics found in {csv_path}")
    return metrics


def align_prompt_indices(series: Sequence[Tuple[str, Dict[int, PromptMetrics]]]) -> List[int]:
    indices = set()
    for _, metrics in series:
        indices.update(metrics.keys())
    if not indices:
        raise ValueError("No prompt indices available to plot")
    return sorted(indices)


def build_axis_values(
    all_indices: Sequence[int], metrics: Dict[int, PromptMetrics]
) -> Tuple[List[float], List[float]]:
    ttft = []
    tpot = []
    for idx in all_indices:
        if idx in metrics:
            metric = metrics[idx]
            ttft.append(metric.ttft_s)
            tpot.append(metric.tpot_s)
        else:
            ttft.append(float("nan"))
            tpot.append(float("nan"))
    return ttft, tpot


def plot_ttft_tpot(
    series: Sequence[Tuple[str, Dict[int, PromptMetrics]]],
    output_path: Path,
    title_suffix: str | None = None,
) -> None:
    all_indices = align_prompt_indices(series)

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    for color, (label, metrics) in zip(colors, series):
        ttft_values, tpot_values = build_axis_values(all_indices, metrics)
        axes[0].plot(
            all_indices,
            ttft_values,
            marker="o",
            linewidth=1.5,
            label=label,
            color=color,
        )
        axes[1].plot(
            all_indices,
            tpot_values,
            marker="o",
            linewidth=1.5,
            label=label,
            color=color,
        )

    axes[0].set_ylabel("TTFT (seconds)")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend()

    axes[1].set_xlabel("Prompt index (issue order)")
    axes[1].set_ylabel("TPOT (seconds)")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend()

    title = "Scheduler comparison"
    if title_suffix:
        title = f"{title} - {title_suffix}"
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def load_kv_usage(log_path: Path) -> Tuple[List[float], List[float]]:
    timestamps: List[datetime] = []
    usages: List[float] = []
    with log_path.open(encoding="utf-8") as infile:
        for line in infile:
            match = KV_USAGE_PATTERN.search(line)
            if not match:
                continue
            timestamp = datetime.strptime(match.group("timestamp"), TIMESTAMP_FORMAT)
            usage = float(match.group("usage"))
            timestamps.append(timestamp)
            usages.append(usage)
    if not timestamps:
        raise ValueError(
            "No GPU KV cache usage entries in "
            f"{log_path}. Ensure metric logging is enabled."
        )
    base_time = timestamps[0]
    elapsed = [(ts - base_time).total_seconds() for ts in timestamps]
    return elapsed, usages


def plot_kv_usage(
    series: Sequence[Tuple[str, Tuple[List[float], List[float]]]],
    output_path: Path,
    title_suffix: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    for color, (label, (elapsed, usages)) in zip(colors, series):
        ax.plot(
            elapsed,
            usages,
            marker="o",
            linewidth=1.5,
            label=label,
            color=color,
        )

    ax.set_xlabel("Time since first measurement (s)")
    ax.set_ylabel("GPU KV cache usage (%)")
    title = "GPU KV cache usage"
    if title_suffix:
        title = f"{title} - {title_suffix}"
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot TTFT/TPOT and KV cache usage for default and CFS scheduler runs"
    )
    parser.add_argument(
        "--default-csv",
        type=Path,
        required=True,
        help="CSV export from the vLLM default scheduler run",
    )
    parser.add_argument(
        "--cfs-csv",
        type=Path,
        required=True,
        help="CSV export from the CFS scheduler run",
    )
    parser.add_argument(
        "--default-log",
        type=Path,
        required=True,
        help="vLLM server log captured during the default scheduler run",
    )
    parser.add_argument(
        "--cfs-log",
        type=Path,
        required=True,
        help="vLLM server log captured during the CFS scheduler run",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("plots/ttft_tpot_scheduler_comparison.png"),
        help="Path for the TTFT/TPOT plot image",
    )
    parser.add_argument(
        "--kv-output",
        type=Path,
        default=Path("plots/kv_cache_scheduler_comparison.png"),
        help="Path for the GPU KV cache usage plot image",
    )
    parser.add_argument(
        "--title-suffix",
        type=str,
        default=None,
        help="Optional text appended to figure titles",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)

    default_metrics = load_prompt_metrics(args.default_csv)
    cfs_metrics = load_prompt_metrics(args.cfs_csv)
    plot_ttft_tpot(
        [
            ("vLLM default", default_metrics),
            ("CFS scheduler", cfs_metrics),
        ],
        args.metrics_output,
        args.title_suffix,
    )

    default_usage = load_kv_usage(args.default_log)
    cfs_usage = load_kv_usage(args.cfs_log)
    plot_kv_usage(
        [
            ("vLLM default", default_usage),
            ("CFS scheduler", cfs_usage),
        ],
        args.kv_output,
        args.title_suffix,
    )

    print(f"Saved TTFT/TPOT comparison to {args.metrics_output}")
    print(f"Saved KV cache comparison to {args.kv_output}")


if __name__ == "__main__":
    main()

    """
    Example usage:
    python3 plot_scheduler_comparison.py \
    --default-csv ttft_tpot_vllm.csv \
    --cfs-csv ttft_tpot_cfs.csv \
    --default-log vllm_server_vllm.log \
    --cfs-log vllm_server_cfs.log
    """
