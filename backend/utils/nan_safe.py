"""NaN-safe reducers that avoid noisy runtime warnings on all-NaN slices."""
from __future__ import annotations

from typing import Any

import numpy as np


def _masked(arr: np.ndarray | list[float] | tuple[float, ...]) -> np.ma.MaskedArray:
    return np.ma.masked_invalid(np.asarray(arr, dtype=np.float32))


def _filled(result: Any, fill_value: float, *, dtype: np.dtype = np.float32):
    if isinstance(result, np.ma.MaskedArray):
        return np.asarray(result.filled(fill_value), dtype=dtype)
    if np.isscalar(result):
        if np.isfinite(result):
            return dtype.type(result)
        return dtype.type(fill_value)
    array = np.asarray(result, dtype=dtype)
    return np.where(np.isfinite(array), array, dtype.type(fill_value)).astype(dtype, copy=False)


def nanmin_safe(arr: np.ndarray, *, axis=None, fill_value: float = 0.0, dtype: np.dtype = np.float32):
    return _filled(np.ma.min(_masked(arr), axis=axis), fill_value, dtype=dtype)


def nanmax_safe(arr: np.ndarray, *, axis=None, fill_value: float = 0.0, dtype: np.dtype = np.float32):
    return _filled(np.ma.max(_masked(arr), axis=axis), fill_value, dtype=dtype)


def nanmean_safe(arr: np.ndarray, *, axis=None, fill_value: float = 0.0, dtype: np.dtype = np.float32):
    return _filled(np.ma.mean(_masked(arr), axis=axis), fill_value, dtype=dtype)


def nanstd_safe(arr: np.ndarray, *, axis=None, fill_value: float = 0.0, dtype: np.dtype = np.float32):
    return _filled(np.ma.std(_masked(arr), axis=axis), fill_value, dtype=dtype)


def nanmedian_safe(arr: np.ndarray, *, axis=None, fill_value: float = 0.0, dtype: np.dtype = np.float32):
    return _filled(np.ma.median(_masked(arr), axis=axis), fill_value, dtype=dtype)
