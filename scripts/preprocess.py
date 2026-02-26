# -*- coding: utf-8 -*-
"""
Preprocess time series data: normalize, segment, and generate visualizations.
"""
import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from scipy import signal
from tqdm import tqdm

MIN_ANOMALY_RATIO = 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess time series CSV files: normalize, segment, and generate visualizations."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Input folder containing CSV files",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output folder for processed samples",
    )
    parser.add_argument(
        "--segment_size",
        type=int,
        default=100,
        help="Segment size (default: 100)",
    )
    parser.add_argument(
        "--sample_ratio",
        type=float,
        default=1.0,
        help="Sample ratio (0-1). 1.0 = process all segments (default: 1.0)",
    )
    return parser.parse_args()


def extract_sequence_id(name: str) -> str:
    """Extract sequence ID from filename."""
    parts = name.split("_")
    if len(parts) > 0:
        try:
            int(parts[0])
            return parts[0]
        except ValueError:
            pass
    return name


def process(
    csv_path: Path,
    output_dir: Path,
    segment_size: int,
    sample_ratio: float,
) -> None:
    df = pd.read_csv(csv_path)
    if not {"Data", "Label"}.issubset(df.columns):
        print(f"Skipping {csv_path.name} (missing required columns)")
        return

    # Normalize + linear detrend
    x = df["Data"].values.astype(float)
    x = (x - x.min()) / (x.max() - x.min() + 1e-12)
    x = signal.detrend(x, type="linear")
    df["Data"] = np.round(x, 6)

    # Global statistics
    diff_x = np.diff(x)
    global_stats = {
        "mean": float(np.round(np.mean(x), 6)),
        "std": float(np.round(np.std(x), 6)),
        "min": float(np.round(np.min(x), 6)),
        "max": float(np.round(np.max(x), 6)),
        "median": float(np.round(np.median(x), 6)),
        "25th_percentile": float(np.round(np.percentile(x, 25), 6)),
        "75th_percentile": float(np.round(np.percentile(x, 75), 6)),
        "90th_percentile": float(np.round(np.percentile(x, 90), 6)),
        "95th_percentile": float(np.round(np.percentile(x, 95), 6)),
        "99th_percentile": float(np.round(np.percentile(x, 99), 6)),
        "data_points": len(x),
        "iqr": float(np.round(np.percentile(x, 75) - np.percentile(x, 25), 6)),
        "diff_mean": float(np.round(np.mean(diff_x), 6)) if len(diff_x) > 0 else 0.0,
        "diff_std": float(np.round(np.std(diff_x), 6)) if len(diff_x) > 0 else 0.0,
        "note": "Global statistics for the entire CSV file.",
    }

    # Extract ground truth anomaly interval indices
    anomalies = df[df["Label"] == 1].index.values
    intervals = []
    if len(anomalies) > 0:
        breaks = np.where(np.diff(anomalies, prepend=-999, append=-999) != 1)[0]
        starts = anomalies[breaks[:-1]]
        ends = anomalies[breaks[1:] - 1]
        intervals = list(zip(starts, ends))

    name = csv_path.stem
    sequence_id = extract_sequence_id(name)
    data = df["Data"].values
    labels = df["Label"].values
    N = len(data)
    num_segments = N // segment_size

    mask_dir = output_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    # Sampling logic (no STRONG_ANOMALY_DRIVEN)
    if sample_ratio < 1.0:
        random.seed(42)

        anomaly_seg_indices = []
        normal_seg_indices = []
        for seg_idx in range(num_segments):
            seg_start = seg_idx * segment_size
            seg_end = min(seg_start + segment_size, N)
            if np.any((anomalies >= seg_start) & (anomalies < seg_end)):
                anomaly_seg_indices.append(seg_idx)
            else:
                normal_seg_indices.append(seg_idx)

        num_anom_found = len(anomaly_seg_indices)
        target_total = max(1, int(num_segments * sample_ratio))

        if num_anom_found > 0:
            min_anomaly_req = int(np.ceil(target_total * MIN_ANOMALY_RATIO))
            num_anomaly_to_sample = min(num_anom_found, min_anomaly_req)
        else:
            num_anomaly_to_sample = 0

        segment_indices = (
            random.sample(anomaly_seg_indices, num_anomaly_to_sample)
            if anomaly_seg_indices
            else []
        )
        remaining_pool = [i for i in range(num_segments) if i not in segment_indices]
        remaining_needed = target_total - len(segment_indices)
        if remaining_needed > 0 and remaining_pool:
            additional = random.sample(
                remaining_pool, min(remaining_needed, len(remaining_pool))
            )
            segment_indices.extend(additional)

        segment_indices = sorted(segment_indices)
    else:
        segment_indices = list(range(num_segments))

    if not segment_indices:
        return

    # Generate full-length 0/1 mask
    full_mask = np.zeros(N, dtype=int)
    for seg_idx in segment_indices:
        start = seg_idx * segment_size
        end = min(start + segment_size, N)
        full_mask[start:end] = 1

    mask_df = pd.DataFrame({"Index": np.arange(N), "Sampled": full_mask})
    mask_path = mask_dir / f"{sequence_id}_sample.csv"
    mask_df.to_csv(mask_path, index=False)

    # Process each sampled segment
    for seg_idx in tqdm(segment_indices, desc=f"Rendering {name}", ncols=100):
        start = seg_idx * segment_size
        end = min(start + segment_size, N)

        sample_folder = output_dir / f"{name}_seg{seg_idx:03d}"
        if (sample_folder / "global_overview.jpg").exists():
            continue

        segment_data = data[start:end]

        segment_intervals = [
            (max(s, start) - start, min(e, end - 1) - start)
            for s, e in intervals
            if not (e < start or s >= end)
        ]

        sample_folder.mkdir(parents=True, exist_ok=True)

        def _plot(
            ax,
            values,
            intervals,
            time_idx,
            show_label,
            show_bbox=False,
            bbox_range=None,
            is_segment=False,
        ):
            ax.set_facecolor("white")
            ax.plot(time_idx, values, color="black", linewidth=1.8, zorder=5)
            grid_color = "#b0b0b0"
            grid_lw = 0.8
            grid_alpha = 0.8
            ax.grid(
                True,
                color="#d0d0d0",
                linewidth=0.7,
                alpha=0.8,
                zorder=0,
                axis="y",
            )
            step = 10 if is_segment else 50
            for x_pos in range(time_idx[0], time_idx[-1] + 1, step):
                if x_pos <= time_idx[-1]:
                    ax.axvline(
                        x_pos,
                        color=grid_color,
                        linewidth=grid_lw,
                        alpha=grid_alpha,
                        zorder=0,
                    )

            for spine in ax.spines.values():
                spine.set_color("black")
                spine.set_linewidth(1.2)

            ax.set_xlabel("Time Step", fontsize=14, fontweight="bold")
            ax.set_ylabel("Value", fontsize=14, fontweight="bold")
            ax.tick_params(axis="both", which="major", labelsize=9, colors="black")

            if show_label:
                for s, e in intervals:
                    ax.axvspan(
                        time_idx[s],
                        time_idx[e] + 1,
                        color="#FF3333",
                        alpha=0.30,
                        zorder=1,
                    )

            if show_bbox and bbox_range:
                bs, be = bbox_range
                yb, yt = ax.get_ylim()
                rect = Rectangle(
                    (time_idx[bs], yb),
                    time_idx[be] - time_idx[bs] + 1,
                    yt - yb,
                    linewidth=1.3,
                    edgecolor="#0066FF",
                    facecolor="none",
                    zorder=10,
                )
                ax.add_patch(rect)

        def save_fig(
            fname,
            values,
            intervals,
            time_idx,
            show_label,
            show_bbox=False,
            bbox_range=None,
            is_segment=False,
        ):
            plt.figure(figsize=(12, 4), dpi=300)
            ax = plt.gca()
            _plot(
                ax,
                values,
                intervals,
                time_idx,
                show_label,
                show_bbox,
                bbox_range,
                is_segment,
            )
            plt.tight_layout()
            plt.savefig(
                sample_folder / fname,
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
            )
            plt.close()

        save_fig(
            "segment_clean.jpg",
            segment_data,
            segment_intervals,
            np.arange(start, end),
            False,
            is_segment=True,
        )

        pd.DataFrame(
            {
                "Index": np.arange(start, end),
                "Value": np.round(segment_data, 3),
                "Label": labels[start:end],
            }
        ).to_csv(sample_folder / "segment_data.csv", index=False)

        seg_anomalies = np.where(labels[start:end] == 1)[0]
        gt_intervals = []
        if len(seg_anomalies) > 0:
            diff = np.diff(seg_anomalies, prepend=-999, append=-999)
            breaks = np.where(diff != 1)[0]
            starts_gt = seg_anomalies[breaks[:-1]]
            ends_gt = seg_anomalies[breaks[1:] - 1]
            gt_intervals = [
                [int(s + start), int(e + start)]
                for s, e in zip(starts_gt, ends_gt)
            ]
        pd.DataFrame(
            gt_intervals, columns=["Start", "End"]
        ).to_csv(sample_folder / "ground_truth.csv", index=False)

        ctx_indices = np.arange(max(0, start - 600), min(N, end + 600))
        pd.DataFrame(
            {
                "Index": ctx_indices,
                "Value": np.round(data[ctx_indices], 3),
                "Label": labels[ctx_indices],
            }
        ).to_csv(sample_folder / "context_data.csv", index=False)

        with open(
            sample_folder / "global_statistics.json", "w", encoding="utf-8"
        ) as f:
            json.dump(global_stats, f, ensure_ascii=False, indent=2)

        prefix_size = min(10000, N)
        pd.DataFrame(
            {
                "Index": np.arange(prefix_size),
                "Value": np.round(data[:prefix_size], 3),
                "Label": labels[:prefix_size],
            }
        ).to_csv(sample_folder / "prefix_10000.csv", index=False)

        plt.figure(figsize=(12, 4), dpi=300)
        ax_ov = plt.gca()
        ax_ov.plot(np.arange(N), data, color="black", linewidth=1.8)
        ax_ov.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.8, axis="y")
        for spine in ax_ov.spines.values():
            spine.set_color("black")
            spine.set_linewidth(1.2)
        ax_ov.set_facecolor("white")
        yb, yt = ax_ov.get_ylim()
        rect = Rectangle(
            (start, yb),
            segment_size,
            yt - yb,
            linewidth=1.3,
            edgecolor="#0066FF",
            facecolor="none",
            zorder=20,
        )
        ax_ov.add_patch(rect)
        plt.tight_layout()
        plt.savefig(
            sample_folder / "global_overview.jpg",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close()

    print(
        f"Done -> {csv_path.name} "
        f"(total points {N}, sampled segments {len(segment_indices)}/{num_segments})"
    )


def main():
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    segment_size = args.segment_size
    sample_ratio = args.sample_ratio

    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.csv"))
    print(f"Found {len(files)} CSV file(s), processing...")
    for f in files:
        process(f, output_dir, segment_size, sample_ratio)
    print("All done!")


if __name__ == "__main__":
    main()
