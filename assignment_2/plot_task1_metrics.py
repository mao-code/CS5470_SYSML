#!/usr/bin/env python3
"""Plot TTFT/TPOT metrics and GPU KV cache usage for Task 1."""
from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt


@dataclass
class PromptMetrics:
    prompt_index: int
    ttft_ms: float
    tpot_ms: float


TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S,%f"
KV_USAGE_PATTERN = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*GPU KV cache usage: (?P<gpu_usage>[0-9]+\.?[0-9]*)%"
)


def load_prompt_metrics(csv_path: Path) -> List[PromptMetrics]:
    """Load TTFT and TPOT metrics from the provided CSV file."""
    metrics: List[PromptMetrics] = []
    with csv_path.open(newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            ttft_raw = row.get("ttft_ms", "").strip()
            tpot_raw = row.get("tpot_ms", "").strip()
            ttft_ms = float(ttft_raw) if ttft_raw else float("nan")
            tpot_ms = float(tpot_raw) if tpot_raw else float("nan")

            metrics.append(
                PromptMetrics(
                    prompt_index=int(row["prompt_index"]),
                    ttft_ms=ttft_ms,
                    tpot_ms=tpot_ms,
                )
            )
    if not metrics:
        raise ValueError(f"No rows loaded from {csv_path}")
    # Ensure metrics are sorted by prompt index to maintain issuing order.
    metrics.sort(key=lambda m: m.prompt_index)
    return metrics


def plot_ttft_tpot(metrics: Sequence[PromptMetrics], output_path: Path) -> None:
    """Plot TTFT and TPOT per prompt index and save to output_path."""
    prompt_indices = [m.prompt_index for m in metrics]
    ttft_values = [m.ttft_ms / 1000 for m in metrics]
    tpot_values = [m.tpot_ms / 1000 for m in metrics]

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    axes[0].plot(prompt_indices, ttft_values, marker="o", linewidth=1.5)
    axes[0].set_ylabel("TTFT (seconds)")
    axes[0].set_title("Time to First Token per prompt")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].plot(prompt_indices, tpot_values, marker="o", color="tab:orange", linewidth=1.5)
    axes[1].set_xlabel("Prompt index (issue order)")
    axes[1].set_ylabel("TPOT (seconds)")
    axes[1].set_title("Time per Output Token per prompt")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def load_kv_usage(log_path: Path) -> tuple[list[datetime], list[float]]:
    """Parse GPU KV cache usage samples from the server log."""
    timestamps: List[datetime] = []
    usages: List[float] = []
    with log_path.open(encoding="utf-8") as infile:
        for line in infile:
            match = KV_USAGE_PATTERN.search(line)
            if not match:
                continue
            timestamp = datetime.strptime(match.group("timestamp"), TIMESTAMP_FORMAT)
            usage = float(match.group("gpu_usage"))
            timestamps.append(timestamp)
            usages.append(usage)
    if not timestamps:
        raise ValueError(
            "No GPU KV cache usage records found.\n"
            "Confirm the log was captured with --disable-log-requests and metric logging enabled."
        )
    return timestamps, usages


def plot_kv_usage(timestamps: Sequence[datetime], usages: Sequence[float], output_path: Path) -> None:
    """Plot GPU KV cache usage over time."""
    base_time = timestamps[0]
    elapsed_seconds = [(ts - base_time).total_seconds() for ts in timestamps]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(elapsed_seconds, usages, marker="o", linewidth=1.5)
    ax.set_xlabel("Time since first measurement (s)")
    ax.set_ylabel("GPU KV cache usage (%)")
    ax.set_title("GPU KV cache usage over time")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Task 1 metrics for HW2")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("ttft_tpot_data.csv"),
        help="Path to the CSV file containing TTFT/TPOT metrics.",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("vllm_server.log"),
        help="Path to the vLLM server log file.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("plots"),
        help="Directory where plots will be saved.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    metrics = load_prompt_metrics(args.csv)
    ttft_tpot_path = args.outdir / "ttft_tpot_over_prompt.png"
    plot_ttft_tpot(metrics, ttft_tpot_path)

    timestamps, usages = load_kv_usage(args.log)
    kv_usage_path = args.outdir / "gpu_kv_cache_usage.png"
    plot_kv_usage(timestamps, usages, kv_usage_path)

    print(f"Saved TTFT/TPOT plot to {ttft_tpot_path}")
    print(f"Saved GPU KV cache usage plot to {kv_usage_path}")


if __name__ == "__main__":
    main()
