import pandas as pd
import numpy as np


def _normalize_id_col(df, col):
    """Coerce an id column to cleaned string form for safe joins."""
    if col not in df.columns:
        return df
    # coerce to str and strip
    df[col] = df[col].astype(str).str.strip()
    # if form contains digits, extract them; else keep cleaned string
    extracted = df[col].str.extract(r"([0-9]+)")
    if extracted is not None and extracted.shape[1] > 0:
        df[col] = extracted[0].fillna(df[col])
    df[col] = df[col].astype(str)
    return df


def _normalize_series_0_1(s):
    """Normalize a pandas Series to [0,1]."""
    if s is None or s.empty:
        return s
    arr = pd.to_numeric(s, errors='coerce')
    mn = float(np.nanmin(arr))
    mx = float(np.nanmax(arr))
    if np.isnan(mn) or np.isnan(mx) or mx == mn:
        return pd.Series(0.0, index=arr.index)
    return (arr - mn) / (mx - mn)
