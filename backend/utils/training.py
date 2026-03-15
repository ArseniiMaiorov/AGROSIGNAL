"""Общие утилиты для обучения и взвешивания выборки."""
from __future__ import annotations


def compute_tile_quality_weight(
    *,
    used_fallback: bool,
    quality_gate_failed: bool,
    manual_gt: bool = False,
) -> float:
    """Вернуть вес тайла для обучения.

    Верифицированные manual GT получают повышенный вес.
    Fallback-тайлы и тайлы, не прошедшие quality gate, понижаются.
    """
    if manual_gt:
        return 2.0

    weight = 1.0
    if used_fallback:
        weight = min(weight, 0.5)
    if quality_gate_failed:
        weight = min(weight, 0.25)
    return weight
