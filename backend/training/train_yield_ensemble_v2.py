#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "backend" / "training" / "config" / "yield_ensemble_v2.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description="Register conformal/ensemble metadata for yield baseline")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--baseline-model", type=Path, default=Path("backend/models/yield_baseline_v2.pkl"))
    parser.add_argument("--output-model", type=Path, default=Path("backend/models/yield_ensemble_v2.pkl"))
    parser.add_argument("--report", type=Path, default=Path("backend/debug/runs/yield_ensemble_v2_report.json"))
    parser.add_argument("--experimental-lstm", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    if args.dry_run:
        payload = {
            "config": str(args.config.resolve()),
            "baseline_model": str(args.baseline_model.resolve()),
            "output_model": str(args.output_model.resolve()),
            "report": str(args.report.resolve()),
            "experimental_lstm": bool(args.experimental_lstm or config.get("experimental_lstm", False)),
            "use_aquacrop_lite": bool(config.get("use_aquacrop_lite", True)),
        }
        print(json.dumps(payload, ensure_ascii=True))
        return 0

    with args.baseline_model.open("rb") as fh:
        baseline = pickle.load(fh)

    payload = {
        "model_version": str(config.get("model_version", "agronomy_ensemble_v2")),
        "dataset_version": str(config.get("dataset_version", baseline.get("dataset_version", "yield_corpus_v2"))),
        "conformal_alpha": float(config.get("conformal_alpha", 0.1)),
        "baseline_backend": baseline.get("backend"),
        "interval_width_p90": baseline.get("interval_width_p90"),
        "use_aquacrop_lite": bool(config.get("use_aquacrop_lite", True)),
        "experimental_lstm": bool(args.experimental_lstm or config.get("experimental_lstm", False)),
    }
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    with args.output_model.open("wb") as fh:
        pickle.dump(payload, fh)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    print(args.output_model.resolve())
    print(args.report.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
