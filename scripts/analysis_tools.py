from typing import Optional,  List
import json
from pathlib import Path
import numpy as np
import pandas as pd
from langchain_core.tools import StructuredTool

from utils import (
    get_column_names,
    round_value,
)


def get_available_tools_description() -> str:
    tools_desc = """
## Available Tools for Analysis

You have access to the following tools that can help with anomaly detection:

1. **stats()**
   - Purpose: Get complete statistical information for the entire dataset
   - Parameters: No parameters required. Simply call stats()
   - Returns: All statistics including:
     * Basic statistics: mean, std, min, max, median
     * Quantiles: 25th, 75th, 90th, 95th percentiles
     * Overview: data points count, time range, missing values
   - Use case: Understand the overall data distribution and baseline statistics (insufficient to get conclusion)
   - Example: stats()

2. **global_diff_zscore(threshold=3.0)**
   - Purpose: Get global first-order difference Z-scores for all data points
   - Parameters:
     * threshold (OPTIONAL): Threshold for identifying outliers (default: 3.0, you can adjust it by Domain knowledge). Points with |Z-score| > threshold are considered outliers.
   - Returns: JSON containing:
     * "global_diff_mean": Global mean of first-order differences
     * "global_diff_std": Global standard deviation of first-order differences
     * "threshold": The threshold value used
     * "diff_zscore_outliers": List of [index, zscore] pairs for points where |diff_zscore| > threshold
   - Example: global_diff_zscore() or global_diff_zscore(threshold=3)

3. **global_value_zscore(threshold=3.0)**
   - Purpose: Get global value Z-scores for all data points
   - Parameters:
     * threshold (OPTIONAL): Threshold for identifying outliers (default: 3.0, you can adjust it by Domain knowledge). Points with |Z-score| > threshold are considered outliers.
   - Returns: JSON containing:
     * "global_mean": Global mean of the entire dataset
     * "global_std": Global standard deviation of the entire dataset
     * "threshold": The threshold value used
     * "zscore_outliers": List of [index, zscore] pairs for points where |zscore| > threshold
   - Example: global_value_zscore() or global_value_zscore(threshold=3)

4. **local_diff_zscore(threshold=3.0)**
   - Purpose: Get local first-order difference Z-scores using statistics from the middle 100 points
   - Parameters:
     * threshold (OPTIONAL): Threshold for identifying outliers (default: 3.0, you can adjust it by Domain knowledge). Points with |Z-score| > threshold are considered outliers.
   - Returns: JSON containing:
     * "local_diff_mean": Local mean of first-order differences (from middle 100 points)
     * "local_diff_std": Local standard deviation of first-order differences (from middle 100 points)
     * "threshold": The threshold value used
     * "diff_zscore_outliers": List of [index, zscore] pairs for points where |diff_zscore| > threshold
   - **IMPORTANT: Only use this tool for NON-stationary time series** (where mean and variance change over time). For stationary time series, use global_diff_zscore() instead.
   - Note: This tool calculates diff Z-scores using statistics from the middle 100 points, which is more suitable for non-stationary series where the local distribution differs from the global distribution.
   - Example: local_diff_zscore() or local_diff_zscore(threshold=3)
"""
    return tools_desc



def create_all_tools(
    data: pd.DataFrame,
    segment_folder_path: Optional[str] = None,
) -> List[StructuredTool]:

    timestamp_col, value_col = get_column_names(data)
    tools: List[StructuredTool] = []

    def get_data_statistics() -> str:
        value_series = data[value_col]
        interval_data = data
        interval_value_series = value_series

        if timestamp_col in data.columns:
            actual_start_idx = int(data[timestamp_col].iloc[0])
            actual_end_idx = int(data[timestamp_col].iloc[-1])
        else:
            actual_start_idx = 0
            actual_end_idx = len(value_series) - 1

        stats = {
            "interval": [int(actual_start_idx), int(actual_end_idx)],
            "is_full_dataset": True,
            "data_points": len(interval_data),
            "value_column": value_col,
            "timestamp_column": timestamp_col,
            "missing_values": int(interval_value_series.isna().sum()),
            "data_columns": list(interval_data.columns),
        }

        # Time range
        if timestamp_col in interval_data.columns and len(interval_data) > 0:
            try:
                stats["time_range"] = (
                    f"{interval_data[timestamp_col].iloc[0]} to "
                    f"{interval_data[timestamp_col].iloc[-1]}"
                )
            except Exception:
                stats["time_range"] = f"index {actual_start_idx} to {actual_end_idx}"

        # Basic statistics
        if len(interval_value_series) > 0:
            stats["mean"] = round_value(interval_value_series.mean(), 3)
            stats["std"] = round_value(interval_value_series.std(), 3)
            stats["min"] = round_value(interval_value_series.min(), 3)
            stats["max"] = round_value(interval_value_series.max(), 3)
            stats["median"] = round_value(interval_value_series.median(), 3)
        else:
            stats["mean"] = 0.0
            stats["std"] = 0.0
            stats["min"] = 0.0
            stats["max"] = 0.0
            stats["median"] = 0.0

        # Quantile statistics
        if len(interval_value_series) > 0:
            stats["25th_percentile"] = round_value(
                interval_value_series.quantile(0.25), 3
            )
            stats["75th_percentile"] = round_value(
                interval_value_series.quantile(0.75), 3
            )
            stats["90th_percentile"] = round_value(
                interval_value_series.quantile(0.90), 3
            )
            stats["95th_percentile"] = round_value(
                interval_value_series.quantile(0.95), 3
            )
        else:
            stats["25th_percentile"] = 0.0
            stats["75th_percentile"] = 0.0
            stats["90th_percentile"] = 0.0
            stats["95th_percentile"] = 0.0

        return json.dumps(stats, ensure_ascii=False, separators=(",", ":"))

    tools.append(
        StructuredTool.from_function(
            func=get_data_statistics,
            name="stats",
            description=(
                "Get complete statistical information for the entire dataset. "
                "No parameters required. Returns all statistics: mean, std, "
                "min, max, median, percentiles (25th, 75th, 90th, 95th), "
                "data points count, time range, etc. Use this to understand "
                "the overall data distribution."
            ),
        )
    )

    # Tool 2: Global diff Z-score
    def get_global_diff_zscore(threshold: float = 3.0) -> str:
        if segment_folder_path is None:
            return json.dumps(
                {"error": "segment_folder_path not provided"},
                ensure_ascii=False,
                separators=(",", ":"),
            )

        global_stats_file = Path(segment_folder_path) / "global_statistics.json"
        if not global_stats_file.exists():
            return json.dumps(
                {"error": "global_statistics.json not found"},
                ensure_ascii=False,
                separators=(",", ":"),
            )

        try:
            with open(global_stats_file, "r", encoding="utf-8") as f:
                global_stats = json.load(f)
        except Exception as e:
            return json.dumps(
                {"error": f"Failed to load global_statistics.json: {str(e)}"},
                ensure_ascii=False,
                separators=(",", ":"),
            )

        if "diff_mean" not in global_stats or "diff_std" not in global_stats:
            return json.dumps(
                {
                    "error": (
                        "global_statistics.json does not contain diff_mean or "
                        "diff_std. Please regenerate using updated preprocess.py."
                    )
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )

        global_diff_mean = global_stats["diff_mean"]
        global_diff_std = global_stats["diff_std"]

        result = {
            "global_diff_mean": round_value(global_diff_mean, 3),
            "global_diff_std": round_value(global_diff_std, 3),
            "threshold": round_value(threshold, 2),
        }

        value_series = data[value_col]
        if len(value_series) < 2:
            result["diff_zscore_outliers"] = []
            return json.dumps(result, ensure_ascii=False, separators=(",", ":"))

        if global_diff_std > 0:
            diff_values = np.diff(value_series.values)
            diff_zscores = (diff_values - global_diff_mean) / global_diff_std
            if timestamp_col in data.columns:
                diff_indices = data[timestamp_col].values[1:].tolist()
            else:
                diff_indices = list(range(1, len(diff_zscores) + 1))

            diff_zscore_outliers = [
                [int(idx), round_value(z, 2)]
                for idx, z in zip(diff_indices, diff_zscores)
                if abs(z) > threshold
            ]
            result["diff_zscore_outliers"] = diff_zscore_outliers
        else:
            result["diff_zscore_outliers"] = []

        return json.dumps(result, ensure_ascii=False, separators=(",", ":"))

    tools.append(
        StructuredTool.from_function(
            func=get_global_diff_zscore,
            name="global_diff_zscore",
            description=(
                "Get global diff Z-scores for stationary time series. "
                "Parameters: threshold (default: 3.0). Returns outliers "
                "(diff_zscore_outliers) and stats (global_diff_mean, "
                "global_diff_std, threshold). Example: global_diff_zscore(threshold=3). "
                "If two adjacent or nearby points have opposite and large diff "
                "Z-scores, check the points between them for significant "
                "deviation from surrounding values. If present, flag those "
                "intermediate points as anomalies, but exclude the last "
                "recovery point."
            ),
        )
    )

    # Tool 3: Global value Z-score
    def get_global_value_zscore(threshold: float = 3.0) -> str:
        if segment_folder_path is None:
            return json.dumps(
                {"error": "segment_folder_path not provided"},
                ensure_ascii=False,
                separators=(",", ":"),
            )

        global_stats_file = Path(segment_folder_path) / "global_statistics.json"
        if not global_stats_file.exists():
            return json.dumps(
                {"error": "global_statistics.json not found"},
                ensure_ascii=False,
                separators=(",", ":"),
            )

        try:
            with open(global_stats_file, "r", encoding="utf-8") as f:
                global_stats = json.load(f)
        except Exception as e:
            return json.dumps(
                {"error": f"Failed to load global_statistics.json: {str(e)}"},
                ensure_ascii=False,
                separators=(",", ":"),
            )

        if "mean" not in global_stats or "std" not in global_stats:
            return json.dumps(
                {"error": "global_statistics.json does not contain mean or std"},
                ensure_ascii=False,
                separators=(",", ":"),
            )

        global_mean = global_stats["mean"]
        global_std = global_stats["std"]

        result = {
            "global_mean": round_value(global_mean, 3),
            "global_std": round_value(global_std, 3),
            "threshold": round_value(threshold, 2),
        }

        value_series = data[value_col]
        if len(value_series) == 0:
            result["zscore_outliers"] = []
            return json.dumps(result, ensure_ascii=False, separators=(",", ":"))

        if timestamp_col in data.columns:
            indices = data[timestamp_col].values.tolist()
        else:
            indices = list(range(len(value_series)))

        if global_std > 0:
            zscores = ((value_series - global_mean) / global_std).tolist()
            zscore_outliers = [
                [int(idx), round_value(z, 2)]
                for idx, z in zip(indices, zscores)
                if abs(z) > threshold
            ]
            result["zscore_outliers"] = zscore_outliers
        else:
            result["zscore_outliers"] = []

        return json.dumps(result, ensure_ascii=False, separators=(",", ":"))

    tools.append(
        StructuredTool.from_function(
            func=get_global_value_zscore,
            name="global_value_zscore",
            description=(
                "Get global value Z-scores for stationary time series. "
                "Parameters: threshold (default: 3.0, you can adjust it by "
                "Domain knowledge). Returns outliers (zscore_outliers) and "
                "stats (global_mean, global_std, threshold). "
                "Example: global_value_zscore(threshold=3)."
            ),
        )
    )

    # Tool 4: Local diff Z-score (non-stationary)
    def get_local_diff_zscore(threshold: float = 3.0) -> str:
        value_series = data[value_col]

        if len(value_series) == 0:
            return json.dumps(
                {"error": "No data"}, ensure_ascii=False, separators=(",", ":")
            )

        if len(value_series) < 2:
            return json.dumps(
                {"error": "Insufficient data (need at least 2 points)"},
                ensure_ascii=False,
                separators=(",", ":"),
            )

        total_points = len(value_series)
        if total_points <= 100:
            local_start = 0
            local_end = total_points - 1
        else:
            local_start = (total_points - 100) // 2
            local_end = local_start + 100 - 1

        local_values = value_series.iloc[local_start : local_end + 1].values

        if len(local_values) > 1:
            local_diff_values = np.diff(local_values)
            local_diff_mean = np.mean(local_diff_values)
            local_diff_std = np.std(local_diff_values)
        else:
            local_diff_mean = 0.0
            local_diff_std = 0.0

        result = {
            "local_diff_mean": round_value(local_diff_mean, 3),
            "local_diff_std": round_value(local_diff_std, 3),
            "threshold": round_value(threshold, 2),
        }

        if len(local_values) > 1 and local_diff_std > 0:
            diff_zscores = (local_diff_values - local_diff_mean) / local_diff_std

            if timestamp_col in data.columns:
                local_indices = data[timestamp_col].iloc[local_start : local_end + 1].values
                diff_indices = local_indices[1:].tolist()
            else:
                diff_indices = list(range(local_start + 1, local_end + 1))

            diff_zscore_outliers = [
                [int(idx), round_value(z, 2)]
                for idx, z in zip(diff_indices, diff_zscores)
                if abs(z) > threshold
            ]
            result["diff_zscore_outliers"] = diff_zscore_outliers
        else:
            result["diff_zscore_outliers"] = []

        return json.dumps(result, ensure_ascii=False, separators=(",", ":"))

    tools.append(
        StructuredTool.from_function(
            func=get_local_diff_zscore,
            name="local_diff_zscore",
            description=(
                "Get local diff Z-scores for NON-stationary time series. "
                "Parameters: threshold (default: 3.0, you can adjust it by "
                "Domain knowledge). Returns outliers (diff_zscore_outliers) "
                "and stats (local_diff_mean, local_diff_std, threshold). "
                "**IMPORTANT: Only use this tool for NON-stationary time "
                "series. For stationary time series, use global_diff_zscore() "
                "instead.** This tool calculates diff Z-scores using "
                "statistics from the middle 100 points, which is more "
                "suitable for non-stationary series. "
                "Example: local_diff_zscore(threshold=3)."
            ),
        )
    )

    return tools


