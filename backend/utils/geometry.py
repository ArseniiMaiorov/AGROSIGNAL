"""Геометрические утилиты без привязки к домену."""
from __future__ import annotations

import math
from typing import Any


def shape_index(area: float, perimeter: float) -> float:
    """Индекс формы для метрических единиц."""
    if area <= 0 or perimeter <= 0:
        return 999.0
    return perimeter / (2.0 * math.sqrt(math.pi * area))


def compactness(area: float, perimeter: float) -> float:
    """Компактность 4πA/P²."""
    if perimeter <= 0:
        return 0.0
    return 4.0 * math.pi * area / (perimeter ** 2)


def elongation(geom: Any) -> float:
    """Отношение длин сторон минимального ориентированного прямоугольника."""
    rect = geom.minimum_rotated_rectangle
    coords = list(rect.exterior.coords)
    if len(coords) < 4:
        return 1.0
    side_a = math.hypot(coords[1][0] - coords[0][0], coords[1][1] - coords[0][1])
    side_b = math.hypot(coords[2][0] - coords[1][0], coords[2][1] - coords[1][1])
    short_side, long_side = sorted([side_a, side_b])
    return long_side / short_side if short_side > 0 else 999.0


def legacy_shape_index(area_px: float, perimeter_px: float) -> float:
    """Старый индекс формы для пиксельных единиц."""
    return perimeter_px / (4.0 * math.sqrt(max(area_px, 1e-6)))
