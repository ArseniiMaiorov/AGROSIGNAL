"""Helpers for user-facing field quality metadata."""
from __future__ import annotations

from typing import Any

from services.trust_service import describe_field_operational_tier


def _clamp_unit(value: float | None) -> float | None:
    if value is None:
        return None
    return float(max(0.0, min(1.0, float(value))))


def extract_runtime_geometry_quality(
    runtime: dict[str, Any] | None,
    *,
    lon: float | None = None,
    lat: float | None = None,
) -> dict[str, Any]:
    payload = dict(runtime or {})
    tiles = [tile for tile in list(payload.get("tiles") or []) if isinstance(tile, dict)]

    def _numeric_from(item: dict[str, Any], key: str) -> float | None:
        value = item.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        return None

    def _aggregate_numeric(key: str) -> float | None:
        direct = _numeric_from(payload, key)
        if direct is not None:
            return direct
        values = [_numeric_from(tile, key) for tile in tiles]
        values = [value for value in values if value is not None]
        if not values:
            return None
        return float(sum(values) / len(values))

    matched_tile: dict[str, Any] | None = None
    if lon is not None and lat is not None:
        candidates: list[tuple[float, dict[str, Any]]] = []
        for tile in tiles:
            bbox = tile.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            try:
                min_x, min_y, max_x, max_y = [float(v) for v in bbox]
            except Exception:
                continue
            if min_x <= float(lon) <= max_x and min_y <= float(lat) <= max_y:
                area = max(0.0, (max_x - min_x) * (max_y - min_y))
                candidates.append((area, tile))
        if candidates:
            candidates.sort(key=lambda item: item[0])
            matched_tile = candidates[0][1]

    geometry_confidence = _numeric_from(matched_tile or {}, "geometry_confidence")
    tta_consensus = _numeric_from(matched_tile or {}, "tta_consensus")
    boundary_uncertainty = _numeric_from(matched_tile or {}, "boundary_uncertainty")
    tta_extent_disagreement = _numeric_from(matched_tile or {}, "tta_extent_disagreement")
    tta_boundary_disagreement = _numeric_from(matched_tile or {}, "tta_boundary_disagreement")

    if geometry_confidence is None:
        geometry_confidence = _aggregate_numeric("geometry_confidence")
    if tta_consensus is None:
        tta_consensus = _aggregate_numeric("tta_consensus")
    if boundary_uncertainty is None:
        boundary_uncertainty = _aggregate_numeric("boundary_uncertainty")
    if tta_extent_disagreement is None:
        tta_extent_disagreement = _aggregate_numeric("tta_extent_disagreement")
    if tta_boundary_disagreement is None:
        tta_boundary_disagreement = _aggregate_numeric("tta_boundary_disagreement")

    uncertainty_source = str(
        (matched_tile or {}).get("uncertainty_source")
        or payload.get("uncertainty_source")
        or ("tta_disagreement" if tta_consensus is not None else "quality_score_proxy")
    ).strip()

    return {
        "geometry_confidence": _clamp_unit(geometry_confidence),
        "tta_consensus": _clamp_unit(tta_consensus),
        "boundary_uncertainty": _clamp_unit(boundary_uncertainty),
        "tta_extent_disagreement": _clamp_unit(tta_extent_disagreement),
        "tta_boundary_disagreement": _clamp_unit(tta_boundary_disagreement),
        "uncertainty_source": uncertainty_source or "quality_score_proxy",
    }


def describe_field_quality(
    score: float | None,
    source: str | None = None,
    *,
    geometry_confidence: float | None = None,
    tta_consensus: float | None = None,
    boundary_uncertainty: float | None = None,
    uncertainty_source: str | None = None,
) -> dict[str, Any]:
    source_value = str(source or "autodetect").strip().lower()
    if source_value == "manual":
        manual_confidence = 1.0 if score is None else float(max(0.0, min(1.0, score)))
        payload = {
            "confidence": manual_confidence,
            "band": "manual",
            "label": "manual",
            "reason": "Контур подтверждён вручную.",
            "reason_code": "manual_confirmation",
            "reason_params": {},
            "color": "#21579c",
            "geometry_confidence": manual_confidence,
            "tta_consensus": 1.0 if score is None else manual_confidence,
            "boundary_uncertainty": 0.0,
            "uncertainty_source": "manual_confirmation",
        }
        payload.update(describe_field_operational_tier(quality_band="manual", source=source))
        return payload

    explicit_geometry = _clamp_unit(geometry_confidence)
    explicit_consensus = _clamp_unit(tta_consensus)
    explicit_boundary_uncertainty = _clamp_unit(boundary_uncertainty)
    base_confidence = (
        explicit_geometry
        if explicit_geometry is not None
        else (float(max(0.0, min(1.0, score))) if score is not None else None)
    )

    if score is None and explicit_geometry is None:
        payload = {
            "confidence": None,
            "band": "unknown",
            "label": "unknown",
            "reason": "Автоматическая оценка уверенности недоступна.",
            "reason_code": "confidence_unavailable",
            "reason_params": {},
            "color": "#6c7b88",
            "geometry_confidence": explicit_geometry,
            "tta_consensus": explicit_consensus,
            "boundary_uncertainty": explicit_boundary_uncertainty,
            "uncertainty_source": str(uncertainty_source or "unavailable"),
        }
        payload.update(describe_field_operational_tier(quality_band="unknown", source=source))
        return payload

    confidence = float(base_confidence or 0.0)
    tta_reason_suffix = ""
    if str(uncertainty_source or "").strip() == "tta_disagreement" and explicit_consensus is not None:
        tta_reason_suffix = " Основано на согласованности TTA."

    if confidence >= 0.82:
        payload = {
            "confidence": round(confidence, 3),
            "band": "high",
            "label": "high",
            "reason": f"Высокая согласованность сегментации, контур стабилен.{tta_reason_suffix}".strip(),
            "reason_code": "high_confidence_stable_contour",
            "reason_params": {"tta_based": bool(tta_reason_suffix)},
            "color": "#1e6a3a",
            "geometry_confidence": round(float(explicit_geometry if explicit_geometry is not None else confidence), 3),
            "tta_consensus": round(float(explicit_consensus), 3) if explicit_consensus is not None else None,
            "boundary_uncertainty": (
                round(float(explicit_boundary_uncertainty), 3)
                if explicit_boundary_uncertainty is not None
                else round(max(0.0, 1.0 - confidence), 3)
            ),
            "uncertainty_source": str(uncertainty_source or "quality_score_proxy"),
        }
        payload.update(describe_field_operational_tier(quality_band="high", source=source))
        return payload
    if confidence >= 0.62:
        payload = {
            "confidence": round(confidence, 3),
            "band": "medium",
            "label": "medium",
            "reason": f"Контур пригоден, но возможны локальные погрешности на границах.{tta_reason_suffix}".strip(),
            "reason_code": "medium_confidence_boundary_review",
            "reason_params": {"tta_based": bool(tta_reason_suffix)},
            "color": "#8a6b18",
            "geometry_confidence": round(float(explicit_geometry if explicit_geometry is not None else confidence), 3),
            "tta_consensus": round(float(explicit_consensus), 3) if explicit_consensus is not None else None,
            "boundary_uncertainty": (
                round(float(explicit_boundary_uncertainty), 3)
                if explicit_boundary_uncertainty is not None
                else round(max(0.0, 1.0 - confidence), 3)
            ),
            "uncertainty_source": str(uncertainty_source or "quality_score_proxy"),
        }
        payload.update(describe_field_operational_tier(quality_band="medium", source=source))
        return payload
    payload = {
        "confidence": round(confidence, 3),
        "band": "low",
        "label": "low",
        "reason": f"Низкая уверенность: контур стоит проверить вручную.{tta_reason_suffix}".strip(),
        "reason_code": "low_confidence_manual_review",
        "reason_params": {"tta_based": bool(tta_reason_suffix)},
        "color": "#a04a24",
        "geometry_confidence": round(float(explicit_geometry if explicit_geometry is not None else confidence), 3),
        "tta_consensus": round(float(explicit_consensus), 3) if explicit_consensus is not None else None,
        "boundary_uncertainty": (
            round(float(explicit_boundary_uncertainty), 3)
            if explicit_boundary_uncertainty is not None
            else round(max(0.0, 1.0 - confidence), 3)
        ),
        "uncertainty_source": str(uncertainty_source or "quality_score_proxy"),
    }
    payload.update(describe_field_operational_tier(quality_band="low", source=source))
    return payload
