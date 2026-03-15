"""Structured UI message codes for additive API payloads."""
from __future__ import annotations

from typing import Any


def classify_support_reason(reason: str | None) -> tuple[str | None, dict[str, Any]]:
    text = str(reason or "").strip()
    if not text:
        return None, {}

    lowered = text.lower()
    if "weekly profile" in lowered or "недельный профиль" in lowered:
        return "weekly_profile_insufficient", {}
    if "training envelope" in lowered or "observed training envelope" in lowered:
        return "outside_training_envelope", {}
    if "global baseline" in lowered:
        return "global_baseline_requires_review", {}
    if ("базовый прогноз" in lowered and "не поддерж" in lowered) or ("baseline" in lowered and "not support" in lowered):
        return "baseline_not_supported", {}
    if "границе пригодности" in lowered or "crop is near the suitability boundary" in lowered:
        return "crop_borderline_suitability", {}
    if "области применимости" in lowered or "applicability" in lowered:
        return "outside_model_applicability", {}
    return "support_review_required", {}


def normalize_risk_level(level: str | None) -> str | None:
    token = str(level or "").strip().lower()
    if not token:
        return None
    mapping = {
        "минимальный": "minimal",
        "низкий": "low",
        "умеренный": "moderate",
        "повышенный": "elevated",
        "высокий": "high",
        "критический": "critical",
        "неопределённый": "unknown",
        "неопределенный": "unknown",
        "minimal": "minimal",
        "low": "low",
        "moderate": "moderate",
        "elevated": "elevated",
        "high": "high",
        "critical": "critical",
        "unknown": "unknown",
    }
    return mapping.get(token, token.replace(" ", "_"))
