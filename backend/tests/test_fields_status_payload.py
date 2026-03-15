from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

from api.fields import _run_response_payload


def test_run_response_payload_includes_explicit_geometry_summary():
    created_at = datetime.now(timezone.utc)
    run = SimpleNamespace(
        id=uuid4(),
        status="running",
        progress=42,
        error_msg=None,
        created_at=created_at,
        params={
            "runtime": {
                "last_heartbeat_ts": created_at.isoformat().replace("+00:00", "Z"),
                "qc_mode": "boundary_recovery",
                "processing_profile": "boundary_recovery",
                "candidate_branch_counts": {
                    "boundary_first": {"total": 4, "kept": 2},
                    "crop_region": {"total": 3, "kept": 1},
                },
                "candidate_reject_summary": {
                    "suppressed_overlap": 2,
                    "low_object_score": 2,
                },
                "candidates_total": 7,
                "candidates_kept": 3,
                "tiles": [
                    {
                        "components_after_grow": 120,
                        "components_after_gap_close": 96,
                        "components_after_infill": 88,
                        "components_after_merge": 54,
                        "components_after_watershed": 60,
                        "watershed_applied": True,
                        "geometry_confidence": 0.76,
                        "tta_consensus": 0.71,
                        "boundary_uncertainty": 0.19,
                        "tta_extent_disagreement": 0.08,
                        "tta_boundary_disagreement": 0.12,
                        "uncertainty_source": "tta_disagreement",
                        "split_score_p50": 0.44,
                        "split_score_p90": 0.78,
                    },
                    {
                        "components_after_grow": 100,
                        "components_after_gap_close": 90,
                        "components_after_infill": 84,
                        "components_after_merge": 50,
                        "components_after_watershed": 52,
                        "watershed_applied": False,
                        "watershed_rollback_reason": "oversegmentation_low_boundary_conf",
                        "geometry_confidence": 0.68,
                        "tta_consensus": 0.63,
                        "boundary_uncertainty": 0.28,
                        "tta_extent_disagreement": 0.11,
                        "tta_boundary_disagreement": 0.17,
                        "uncertainty_source": "tta_disagreement",
                        "split_score_p50": 0.41,
                        "split_score_p90": 0.72,
                    },
                ],
            }
        },
    )

    payload = _run_response_payload(run)
    summary = payload["geometry_summary"]

    assert payload["qc_mode"] == "boundary_recovery"
    assert payload["processing_profile"] == "boundary_recovery"
    assert payload["candidates_total"] == 7
    assert payload["candidates_kept"] == 3
    assert payload["candidate_branch_counts"]["boundary_first"]["kept"] == 2
    assert payload["candidate_reject_summary"]["suppressed_overlap"] == 2
    assert summary["head_count"] == 3
    assert summary["heads"] == ["extent", "boundary", "distance"]
    assert summary["tta_standard"] == "flip2"
    assert summary["tta_quality"] == "rotate4"
    assert summary["tiles_summarized"] == 2
    assert summary["watershed_applied"] is True
    assert summary["watershed_rollback_reason"] == "oversegmentation_low_boundary_conf"
    assert summary["components_after_merge"] == 52
    assert summary["split_score_p50"] == 0.425
    assert summary["geometry_confidence"] == 0.72
    assert summary["tta_consensus"] == 0.67
    assert summary["boundary_uncertainty"] == 0.235
    assert summary["uncertainty_source"] == "tta_disagreement"
