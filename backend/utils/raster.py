"""Утилиты для бинарных растровых масок."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import label as nd_label


def count_components(mask: np.ndarray) -> int:
    """Количество связных компонент."""
    if not np.any(mask):
        return 0
    _, count = nd_label(mask.astype(bool))
    return int(count)


def remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Удалить компоненты меньше порога."""
    mask = mask.astype(bool, copy=False)
    min_size = max(1, int(min_size))
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    labeled, n_labels = nd_label(mask)
    if n_labels <= 0:
        return np.zeros_like(mask, dtype=bool)
    counts = np.bincount(labeled.ravel())
    keep_labels = np.flatnonzero(counts >= min_size)
    keep_labels = keep_labels[keep_labels != 0]
    if keep_labels.size == 0:
        return np.zeros_like(mask, dtype=bool)
    return np.isin(labeled, keep_labels)


def count_small_components(mask: np.ndarray, min_size: int) -> int:
    """Посчитать компоненты меньше порога."""
    mask = mask.astype(bool, copy=False)
    min_size = max(1, int(min_size))
    if not np.any(mask):
        return 0
    labeled, n_labels = nd_label(mask)
    if n_labels <= 0:
        return 0
    counts = np.bincount(labeled.ravel())[1:]
    return int(np.count_nonzero(counts < min_size))


def select_small_components(mask: np.ndarray, max_size: int) -> np.ndarray:
    """Оставить только компоненты не больше порога."""
    mask = mask.astype(bool, copy=False)
    max_size = max(1, int(max_size))
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    labeled, n_labels = nd_label(mask)
    if n_labels <= 0:
        return np.zeros_like(mask, dtype=bool)
    counts = np.bincount(labeled.ravel())
    keep_labels = np.flatnonzero((counts > 0) & (counts <= max_size))
    keep_labels = keep_labels[keep_labels != 0]
    if keep_labels.size == 0:
        return np.zeros_like(mask, dtype=bool)
    return np.isin(labeled, keep_labels)


def empty_bool_like(mask: np.ndarray) -> np.ndarray:
    """Пустая bool-маска той же формы."""
    return np.zeros_like(mask, dtype=bool)
