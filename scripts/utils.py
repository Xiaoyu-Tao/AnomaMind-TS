import pandas as pd
import numpy as np
import json
import re
import time
import logging
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)
def load_segment_data(segment_folder: Path) -> Dict[str, Any]:
    """Load segment data. Only segment_clean.jpg is required (used by Localization)."""
    folder = Path(segment_folder)

    csv_path = folder / "segment_data.csv"
    segment_image_path = folder / "segment_clean.jpg"
    ground_truth_path = folder / "ground_truth.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"no segment_data.csv in {folder}")
    if not segment_image_path.exists():
        raise FileNotFoundError(f"no segment_clean.jpg in {folder}")

    segment_df = pd.read_csv(csv_path)

    ground_truth = []
    if ground_truth_path.exists():
        try:
            gt_df = pd.read_csv(ground_truth_path)
            if not gt_df.empty and 'Start' in gt_df.columns and 'End' in gt_df.columns:
                for _, row in gt_df.iterrows():
                    start = int(row['Start'])
                    end = int(row['End'])
                    if start > end:
                        start, end = end, start
                    ground_truth.append([start, end])
        except Exception as e:
            logger.warning(f"load ground_truth.csv error ({ground_truth_path}): {e}")
    else:
        logger.debug(f"no ground_truth.csv")
    segment_image_base64 = image_to_base64(str(segment_image_path))

    return {
        "segment_data": segment_df,
        "segment_image_path": str(segment_image_path),
        "segment_image_base64": segment_image_base64,
        "ground_truth": ground_truth,
    }


def save_rollout_records(*args, **kwargs) -> None:
    """Deprecated: rollout record persistence removed."""
    return None

def image_to_base64(image_path: str) -> str:
    """Convert image to base64 encoding."""
    import base64
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')
        
def get_column_names(data: pd.DataFrame) -> Tuple[str, str]:
    """Auto-detect timestamp and value columns from DataFrame.

    Returns:
        (timestamp_col, value_col)
    """
    # Type check: ensure data is DataFrame
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame, got {type(data).__name__}")
    
    # Check DataFrame is not empty
    if data.empty:
        raise ValueError("DataFrame is empty")
    
    # Check columns exist
    if len(data.columns) == 0:
        raise ValueError("DataFrame has no columns")
    
    # Try to find timestamp column
    timestamp_col = None
    for col in data.columns:
        if col.lower() in ['Index', 'time', 'timestamp', 'date']:
            timestamp_col = col
            break
    
    # Try to find value column
    value_col = None
    for col in data.columns:
        if col.lower() in ['Value', 'data', 'val', 'values']:
            value_col = col
            break
    
    # Use defaults if not found
    if timestamp_col is None:
        timestamp_col = 'Index' if 'Index' in data.columns else data.columns[0]
    if value_col is None:
        # Exclude timestamp and Label columns
        remaining_cols = [c for c in data.columns if c != timestamp_col and c.lower() != 'label']
        value_col = remaining_cols[0] if remaining_cols else data.columns[-1]
    
    # Ensure return values are strings
    timestamp_col = str(timestamp_col)
    value_col = str(value_col)
    
    # Validate return types
    if not isinstance(timestamp_col, str) or not isinstance(value_col, str):
        raise ValueError(f"get_column_names returned invalid types: timestamp_col={type(timestamp_col)}, value_col={type(value_col)}")
    
    return timestamp_col, value_col


def round_value(value: float, decimals: int = 4) -> float:
    """Round value to given decimal places (reduces token usage).

    Args:
        value: Value to round
        decimals: Decimal places (default 4)

    Returns:
        Rounded value
    """
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return 0.0
    return round(float(value), decimals)


def format_time_series_value(value: float) -> str:
    """Format time series value with adaptive precision.

    Args:
        value: Time series value

    Returns:
        Formatted string
    """
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return "0.0000"
    
    abs_value = abs(value)
    
    # Choose precision by magnitude
    if abs_value >= 1000:
        return f"{value:.2f}"
    elif abs_value >= 1:
        return f"{value:.3f}"
    elif abs_value >= 0.01:
        return f"{value:.3f}"
    else:
        return f"{value:.4f}"

def clean_json_string(json_str: str) -> str:
    """Remove control characters from JSON string so json.loads can parse it.

    Args:
        json_str: Raw JSON string (may contain control characters)

    Returns:
        Cleaned JSON string
    """
    cleaned_json = ""
    i = 0
    while i < len(json_str):
        char = json_str[i]
        # Check for escape sequence
        if char == '\\' and i + 1 < len(json_str):
            next_char = json_str[i + 1]
            # Keep valid escape sequences
            if next_char in ['"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u']:
                cleaned_json += char + next_char
                i += 2
                continue
        # Skip control chars except newline, carriage return, tab
        if ord(char) < 32 and char not in ['\n', '\r', '\t']:
            i += 1
            continue
        cleaned_json += char
        i += 1
    return cleaned_json


def fix_common_json_errors(json_str: str) -> str:
    """Fix common JSON format errors.

    Args:
        json_str: JSON string that may have format errors

    Returns:
        Fixed JSON string
    """
    # Fix "results[" -> "results": [
    json_str = re.sub(r'"results\[', '"results": [', json_str)
    json_str = re.sub(r'"results\s*\[', '"results": [', json_str)

    # Fix unquoted results[ -> "results": [
    json_str = re.sub(r'(\{|,|\s|:)\s*results\s*\[', r'\1"results": [', json_str)

    # Fix "key[" -> "key": [
    json_str = re.sub(r'"(\w+)\[', r'"\1": [', json_str)

    # Fix missing closing brace (at string end)
    if json_str.strip().endswith(']') and json_str.count('{') > json_str.count('}'):
        json_str = json_str.rstrip().rstrip(']') + ']}'
    
    # Fix empty array: results[] -> "results": []
    json_str = re.sub(r'(\{|,|\s|:)\s*results\s*\[\s*\]', r'\1"results": []', json_str)
    
    # Fix unescaped double quotes in JSON string values (state machine)
    def escape_quotes_in_json_strings(text):
        """Escape unescaped double quotes in JSON string values using state machine."""
        result = []
        i = 0
        in_string = False
        escape_next = False
        
        while i < len(text):
            char = text[i]
            
            if escape_next:
                # Character after escape
                result.append(char)
                escape_next = False
                i += 1
                continue
            
            if char == '\\':
                # Escape character
                result.append(char)
                escape_next = True
                i += 1
                continue
            
            if char == '"':
                if in_string:
                    # Check if this is end of string (followed by comma, }, or whitespace)
                    j = i + 1
                    while j < len(text) and text[j] in ' \t\n\r':
                        j += 1
                    
                    if j >= len(text) or text[j] in ',}':
                        # End of string
                        in_string = False
                        result.append(char)
                    else:
                        # Double quote inside string value, escape it
                        result.append('\\"')
                else:
                    # Check if this is start of string (preceded by : or ,)
                    k = i - 1
                    while k >= 0 and text[k] in ' \t\n\r':
                        k -= 1
                    
                    if k >= 0 and text[k] == ':':
                        # Start of string
                        in_string = True
                        result.append(char)
                    else:
                        # May be part of key name, keep as-is
                        result.append(char)
            else:
                result.append(char)
            
            i += 1
        
        return ''.join(result)
    
    # Apply quote escaping in string values
    json_str = escape_quotes_in_json_strings(json_str)
    
    # Fix missing quotes (key: value -> "key": value) - complex, skipped
    
    return json_str


def safe_json_loads(json_str: str, default=None):
    """Safely parse JSON string, auto-clean control chars and fix common format errors.

    Args:
        json_str: JSON string
        default: Value to return on parse failure

    Returns:
        Parsed JSON object, or default on failure
    """
    try:
        # Try direct parse first
        return json.loads(json_str)
    except json.JSONDecodeError:
        # If failed, fix common format errors
        fixed = fix_common_json_errors(json_str)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            # If still failed, clean control chars and retry
            cleaned = clean_json_string(fixed)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                # If still failed, try aggressive clean (remove all control chars)
                import re as re_module
                aggressive_cleaned = re_module.sub(r'[\x00-\x1f\x7f-\x9f]', '', fixed)
                try:
                    return json.loads(aggressive_cleaned)
                except json.JSONDecodeError as e:
                    print(f"[JSON parse] All fix attempts failed: {e}")
                    print(f"[JSON parse] Raw string (first 500 chars): {json_str[:500]}")
                    if default is not None:
                        return default
                    raise