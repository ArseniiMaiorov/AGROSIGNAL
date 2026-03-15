from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


def _load_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "diagnose_recall_loss.py"
    spec = importlib.util.spec_from_file_location("diagnose_recall_loss", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_parse_modes_defaults_and_single_mode_override():
    module = _load_module()

    assert module._parse_modes(SimpleNamespace(mode=None, modes="standard,quality")) == ["standard", "quality"]
    assert module._parse_modes(SimpleNamespace(mode="quality", modes="standard")) == ["quality"]


def test_classify_run_failure_and_next_fix_target():
    module = _load_module()

    diagnosis = module._classify_run_failure(
        {
            "status": "done",
            "candidates_total": 5,
            "candidates_kept": 0,
            "candidate_reject_summary": {"below_min_score": 3, "suppressed_overlap": 2},
        },
        {"tiles": [{"tile_id": "t1"}]},
        0,
    )
    assert diagnosis == "rank_and_suppress"
    assert (
        module._next_fix_target(
            diagnosis,
            {"below_min_score": 3, "suppressed_overlap": 2},
            {"processing_profile_counts": {}, "qc_mode_counts": {}, "worst_tiles": []},
        )
        == "ranking_thresholds"
    )


def test_write_summary_creates_json_and_markdown(tmp_path):
    module = _load_module()
    summary_path = tmp_path / "recall_diagnosis_summary.json"
    records = [
        {
            "item_id": "krasnodar_01",
            "region": "krasnodar",
            "mode": "standard",
            "run_id": "run-1",
            "status": "done",
            "diagnosed_loss_stage": "candidate_generation",
            "next_fix_target": "generation_recovery_bias",
            "dominant_reject_reasons": ["below_min_score"],
            "qc_mode": "boundary_recovery",
            "processing_profile": "boundary_recovery",
            "field_count": 0,
            "candidates_total": 0,
            "candidates_kept": 0,
        }
    ]

    module._write_summary(summary_path, records)

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["records_total"] == 1
    assert payload["items"][0]["item_id"] == "krasnodar_01"
    md_path = summary_path.with_name(f"{summary_path.stem}_summary.md")
    assert md_path.exists()
