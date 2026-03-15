"""Shared production-trust helpers for detect and agronomy outputs."""
from __future__ import annotations

from typing import Any

from services.message_codes import classify_support_reason

_VALIDATED_DETECT_REGION_BANDS = {"south", "central"}
_VALIDATED_PREDICTION_TIERS = {"tenant_calibrated"}


def describe_detect_launch(
    *,
    region_band: str | None,
    preset: str,
    budget_ok: bool,
    hard_block: bool,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    region = str(region_band or "unknown").strip().lower()
    normalized_preset = str(preset or "standard").strip().lower()
    warnings = list(warnings or [])

    if hard_block:
        return {
            "launch_tier": "blocked",
            "review_required": True,
            "review_reason": "Запуск выходит за безопасный envelope текущего хоста.",
            "review_reason_code": "host_safety_envelope_exceeded",
            "review_reason_params": {},
        }

    if region not in _VALIDATED_DETECT_REGION_BANDS:
        return {
            "launch_tier": "experimental_rest",
            "review_required": True,
            "review_reason": "Регион пока не входит в validated core. Результат нужен с обязательной проверкой.",
            "review_reason_code": "region_not_validated_core",
            "review_reason_params": {"region_band": region},
        }

    if normalized_preset == "fast":
        return {
            "launch_tier": "review_needed",
            "review_required": True,
            "review_reason": "Fast — preview-режим. Контуры нужно подтвердить перед операционным использованием.",
            "review_reason_code": "fast_preview_requires_review",
            "review_reason_params": {},
        }

    if not budget_ok:
        return {
            "launch_tier": "review_needed",
            "review_required": True,
            "review_reason": warnings[0] if warnings else "Запуск разрешён, но контур ожидается в пограничном compute-quality режиме.",
            "review_reason_code": "budget_guardrail_warning",
            "review_reason_params": {"warnings_count": len(warnings)},
        }

    return {
        "launch_tier": "validated_core",
        "review_required": False,
        "review_reason": None,
        "review_reason_code": None,
        "review_reason_params": {},
    }


def describe_field_operational_tier(*, quality_band: str | None, source: str | None) -> dict[str, Any]:
    source_value = str(source or "autodetect").strip().lower()
    band = str(quality_band or "unknown").strip().lower()

    if source_value == "autodetect_preview":
        return {
            "operational_tier": "experimental_rest",
            "review_required": True,
            "review_reason": "Preview-контур предназначен только для предварительного просмотра и требует подтверждения.",
            "review_reason_code": "preview_contour_requires_confirmation",
            "review_reason_params": {},
        }
    if source_value == "manual":
        return {
            "operational_tier": "validated_manual",
            "review_required": False,
            "review_reason": None,
            "review_reason_code": None,
            "review_reason_params": {},
        }
    if band == "high":
        return {
            "operational_tier": "validated_core",
            "review_required": False,
            "review_reason": None,
            "review_reason_code": None,
            "review_reason_params": {},
        }
    if band == "medium":
        return {
            "operational_tier": "review_needed",
            "review_required": True,
            "review_reason": "Средняя уверенность контура: перед агрономическим использованием нужен визуальный контроль.",
            "review_reason_code": "field_medium_confidence_review",
            "review_reason_params": {},
        }
    if band == "low":
        return {
            "operational_tier": "review_needed",
            "review_required": True,
            "review_reason": "Низкая уверенность контура: без ручной проверки контур не должен считаться production-grade.",
            "review_reason_code": "field_low_confidence_review",
            "review_reason_params": {},
        }
    return {
        "operational_tier": "experimental_rest",
        "review_required": True,
        "review_reason": "Уверенность контура не подтверждена. Нужна проверка оператором.",
        "review_reason_code": "field_confidence_unconfirmed",
        "review_reason_params": {},
    }


def describe_prediction_operational_tier(
    *,
    supported: bool,
    confidence_tier: str | None,
    crop_suitability: dict[str, Any] | None,
    support_reason: str | None,
) -> dict[str, Any]:
    normalized_confidence_tier = str(confidence_tier or "unsupported").strip().lower()
    suitability = dict(crop_suitability or {})
    suitability_status = str(suitability.get("status") or "unknown").strip().lower()
    suitability_reason = str(suitability.get("support_reason") or "").strip() or None

    if not supported or normalized_confidence_tier == "unsupported" or suitability_status == "unsuitable":
        review_reason = support_reason or suitability_reason or "Модель вне области применимости."
        review_reason_code, review_reason_params = classify_support_reason(review_reason)
        if suitability_status == "unsuitable" and suitability_reason:
            review_reason_code = "crop_unsuitable_for_region"
            review_reason_params = {}
        return {
            "operational_tier": "unsupported",
            "review_required": True,
            "review_reason": review_reason,
            "review_reason_code": review_reason_code,
            "review_reason_params": review_reason_params,
        }

    if normalized_confidence_tier in _VALIDATED_PREDICTION_TIERS and suitability_status in {"high", "moderate"}:
        return {
            "operational_tier": "validated_core",
            "review_required": False,
            "review_reason": None,
            "review_reason_code": None,
            "review_reason_params": {},
        }

    if suitability_status == "low":
        return {
            "operational_tier": "review_needed",
            "review_required": True,
            "review_reason": suitability_reason or "Культура находится на границе пригодности для этого региона.",
            "review_reason_code": "crop_borderline_suitability",
            "review_reason_params": {},
        }

    if normalized_confidence_tier == "global_baseline":
        return {
            "operational_tier": "experimental_rest",
            "review_required": True,
            "review_reason": "Использован global baseline без tenant calibration. Решение требует проверки агрономом.",
            "review_reason_code": "global_baseline_requires_review",
            "review_reason_params": {},
        }

    review_reason = support_reason or suitability_reason or "Результат требует проверки перед операционным использованием."
    review_reason_code, review_reason_params = classify_support_reason(review_reason)
    return {
        "operational_tier": "review_needed",
        "review_required": True,
        "review_reason": review_reason,
        "review_reason_code": review_reason_code,
        "review_reason_params": review_reason_params,
    }
