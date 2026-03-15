from __future__ import annotations

from services.field_quality import describe_field_quality


def test_describe_field_quality_maps_score_bands():
    high = describe_field_quality(0.91, "autodetect")
    medium = describe_field_quality(0.7, "autodetect")
    low = describe_field_quality(0.31, "autodetect")
    manual = describe_field_quality(None, "manual")

    assert high["band"] == "high"
    assert medium["band"] == "medium"
    assert low["band"] == "low"
    assert manual["band"] == "manual"
    assert "вручную" in manual["reason"].lower()
    assert high["operational_tier"] == "validated_core"
    assert high["review_required"] is False
    assert medium["operational_tier"] == "review_needed"
    assert low["review_required"] is True
    assert manual["operational_tier"] == "validated_manual"


def test_describe_field_quality_prefers_real_tta_uncertainty_when_provided():
    payload = describe_field_quality(
        0.91,
        "autodetect",
        geometry_confidence=0.77,
        tta_consensus=0.72,
        boundary_uncertainty=0.18,
        uncertainty_source="tta_disagreement",
    )

    assert payload["confidence"] == 0.77
    assert payload["geometry_confidence"] == 0.77
    assert payload["tta_consensus"] == 0.72
    assert payload["boundary_uncertainty"] == 0.18
    assert payload["uncertainty_source"] == "tta_disagreement"
    assert "tta" in payload["reason"].lower()
