"""
Helper functions for OpenStatica
Common utility functions used across the application
"""

import hashlib
import json
import uuid
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import io
import base64
import re
from pathlib import Path


def generate_session_id() -> str:
    """Generate unique session ID"""
    return str(uuid.uuid4())


def generate_model_id(prefix: str = "model") -> str:
    """Generate unique model ID"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"{prefix}_{timestamp}_{unique_id}"


def hash_data(data: Any) -> str:
    """Generate hash for data caching"""
    if isinstance(data, pd.DataFrame):
        data_str = pd.util.hash_pandas_object(data).values.tobytes()
    elif isinstance(data, np.ndarray):
        data_str = data.tobytes()
    else:
        data_str = json.dumps(data, sort_keys=True).encode()

    return hashlib.sha256(data_str).hexdigest()


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage"""
    # Remove any path components
    filename = Path(filename).name

    # Replace unsafe characters
    filename = re.sub(r'[^\w\s\-\.]', '', filename)
    filename = re.sub(r'[-\s]+', '-', filename)

    # Limit length
    name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
    if len(name) > 50:
        name = name[:50]

    return f"{name}.{ext}" if ext else name


def format_bytes(bytes_value: int) -> str:
    """Format bytes to human readable string"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def format_duration(seconds: float) -> str:
    """Format duration to human readable string"""
    if seconds < 1:
        return f"{seconds * 1000:.2f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours}h {minutes}m"


def encode_dataframe_to_json(df: pd.DataFrame) -> str:
    """Encode DataFrame to JSON string"""
    return df.to_json(orient='split', date_format='iso')


def decode_json_to_dataframe(json_str: str) -> pd.DataFrame:
    """Decode JSON string to DataFrame"""
    return pd.read_json(json_str, orient='split')


def dataframe_to_base64(df: pd.DataFrame) -> str:
    """Convert DataFrame to base64 encoded string"""
    buffer = io.BytesIO()
    df.to_pickle(buffer)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')


def base64_to_dataframe(b64_str: str) -> pd.DataFrame:
    """Convert base64 string back to DataFrame"""
    buffer = io.BytesIO(base64.b64decode(b64_str))
    return pd.read_pickle(buffer)


def flatten_dict(d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
    """Flatten nested dictionary"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: Dict, sep: str = '.') -> Dict:
    """Unflatten dictionary"""
    result = {}
    for key, value in d.items():
        parts = key.split(sep)
        target = result
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]
        target[parts[-1]] = value
    return result


def chunk_list(lst: List, chunk_size: int) -> List[List]:
    """Split list into chunks"""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with default value for division by zero"""
    if denominator == 0:
        return default
    return numerator / denominator


def get_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """Get column types for a DataFrame"""
    type_mapping = {
        'int64': 'integer',
        'float64': 'numeric',
        'object': 'categorical',
        'bool': 'boolean',
        'datetime64': 'datetime',
        'timedelta64': 'timedelta',
        'category': 'categorical'
    }

    return {
        col: type_mapping.get(str(df[col].dtype), 'unknown')
        for col in df.columns
    }


def infer_delimiter(sample: str) -> str:
    """Infer CSV delimiter from sample"""
    delimiters = [',', '\t', ';', '|']
    delimiter_counts = {}

    for delimiter in delimiters:
        delimiter_counts[delimiter] = sample.count(delimiter)

    # Return delimiter with most occurrences
    return max(delimiter_counts, key=delimiter_counts.get)


def is_numeric_dtype(dtype: Any) -> bool:
    """Check if dtype is numeric"""
    return pd.api.types.is_numeric_dtype(dtype)


def is_categorical_dtype(dtype: Any) -> bool:
    """Check if dtype is categorical"""
    return pd.api.types.is_categorical_dtype(dtype) or pd.api.types.is_object_dtype(dtype)


def is_datetime_dtype(dtype: Any) -> bool:
    """Check if dtype is datetime"""
    return pd.api.types.is_datetime64_any_dtype(dtype)


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names for compatibility"""
    df = df.copy()

    # Replace spaces and special characters
    df.columns = df.columns.str.replace(' ', '_')
    df.columns = df.columns.str.replace('[^a-zA-Z0-9_]', '', regex=True)

    # Ensure column names don't start with numbers
    df.columns = ['col_' + col if col[0].isdigit() else col for col in df.columns]

    # Handle duplicate columns
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols == dup] = [f"{dup}_{i}" if i != 0 else dup for i in range(sum(cols == dup))]
    df.columns = cols

    return df


def calculate_memory_usage(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate memory usage of DataFrame"""
    memory_usage = df.memory_usage(deep=True)
    total_memory = memory_usage.sum()

    return {
        'total': format_bytes(total_memory),
        'total_bytes': int(total_memory),
        'per_column': {
            col: format_bytes(memory_usage[col])
            for col in df.columns
        }
    }


def validate_json(json_str: str) -> bool:
    """Validate JSON string"""
    try:
        json.loads(json_str)
        return True
    except (json.JSONDecodeError, TypeError):
        return False


def merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
    """Deep merge two dictionaries"""
    result = dict1.copy()

    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value

    return result


def format_number(value: float, precision: int = 4) -> str:
    """Format number for display"""
    if pd.isna(value):
        return "N/A"

    if abs(value) < 1e-4 or abs(value) > 1e6:
        return f"{value:.{precision}e}"
    else:
        return f"{value:.{precision}f}"


def create_summary_stats(series: pd.Series) -> Dict[str, Any]:
    """Create summary statistics for a series"""
    if is_numeric_dtype(series.dtype):
        return {
            'count': int(series.count()),
            'mean': float(series.mean()),
            'std': float(series.std()),
            'min': float(series.min()),
            'q1': float(series.quantile(0.25)),
            'median': float(series.median()),
            'q3': float(series.quantile(0.75)),
            'max': float(series.max()),
            'missing': int(series.isna().sum())
        }
    else:
        value_counts = series.value_counts()
        return {
            'count': int(series.count()),
            'unique': int(series.nunique()),
            'top': str(value_counts.index[0]) if len(value_counts) > 0 else None,
            'freq': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
            'missing': int(series.isna().sum())
        }
