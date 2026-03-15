#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None  # type: ignore[assignment]


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "backend" / "training" / "config" / "yield_xgboost_v2.yaml"


@dataclass(slots=True)
class Dataset:
    rows: list[dict[str, Any]]
    numeric_columns: list[str]
    categorical_columns: list[str]
    target: np.ndarray


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"Dataset is empty: {path}")
    return rows


def _build_dataset(rows: list[dict[str, Any]], *, target_column: str) -> Dataset:
    categorical_columns = ["crop_code", "soil_texture_class"]
    excluded = {"field_id", target_column, "crop_name", "management_event_types", "observed_on_min", "observed_on_max"}
    numeric_columns = [
        key
        for key in rows[0].keys()
        if key not in excluded and key not in categorical_columns and isinstance(rows[0].get(key), (int, float, type(None)))
    ]
    target = np.asarray([float(row[target_column]) for row in rows], dtype=float)
    return Dataset(rows=rows, numeric_columns=numeric_columns, categorical_columns=categorical_columns, target=target)


def _make_model(config: dict[str, Any]):
    if XGBRegressor is not None:
        return XGBRegressor(
            n_estimators=int(config.get("n_estimators", 400)),
            max_depth=int(config.get("max_depth", 6)),
            learning_rate=float(config.get("learning_rate", 0.05)),
            subsample=float(config.get("subsample", 0.9)),
            colsample_bytree=float(config.get("colsample_bytree", 0.9)),
            objective="reg:squarederror",
            n_jobs=4,
            random_state=int(config.get("random_state", 42)),
        ), "xgboost"
    return HistGradientBoostingRegressor(
        learning_rate=float(config.get("learning_rate", 0.05)),
        max_depth=int(config.get("max_depth", 6)),
        max_leaf_nodes=63,
        min_samples_leaf=int(config.get("min_samples_leaf", 16)),
        random_state=int(config.get("random_state", 42)),
    ), "histgradientboosting"


def main() -> int:
    parser = argparse.ArgumentParser(description="Train objective yield baseline on CPU-safe corpus")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data", type=Path, default=Path("backend/debug/runs/yield_corpus_v2.jsonl"))
    parser.add_argument("--output-model", type=Path, default=Path("backend/models/yield_baseline_v2.pkl"))
    parser.add_argument("--report", type=Path, default=Path("backend/debug/runs/yield_baseline_v2_report.json"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    if args.dry_run:
        payload = {
            "config": str(args.config.resolve()),
            "data": str(args.data.resolve()),
            "output_model": str(args.output_model.resolve()),
            "report": str(args.report.resolve()),
            "backend_preference": "xgboost" if XGBRegressor is not None else "histgradientboosting",
        }
        print(json.dumps(payload, ensure_ascii=True))
        return 0

    rows = _load_rows(args.data)
    dataset = _build_dataset(rows, target_column=str(config.get("target_column", "yield_kg_ha")))
    X = dataset.rows
    y = dataset.target
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(config.get("test_size", 0.2)),
        random_state=int(config.get("random_state", 42)),
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), dataset.numeric_columns),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                dataset.categorical_columns,
            ),
        ]
    )
    model, backend_name = _make_model(config)
    pipeline = Pipeline([("prep", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)
    prediction = pipeline.predict(X_test)

    residuals = np.abs(prediction - y_test)
    interval_width = float(np.quantile(residuals, 1.0 - float(config.get("prediction_interval_alpha", 0.1)))) if residuals.size else 0.0
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backend": backend_name,
        "samples_total": int(len(rows)),
        "samples_train": int(len(X_train)),
        "samples_test": int(len(X_test)),
        "numeric_columns": dataset.numeric_columns,
        "categorical_columns": dataset.categorical_columns,
        "metrics": {
            "mae": round(float(mean_absolute_error(y_test, prediction)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, prediction))), 4),
            "r2": round(float(r2_score(y_test, prediction)), 4),
        },
        "prediction_interval_width_p90": round(interval_width, 4),
        "model_version": str(config.get("model_version", "agronomy_xgb_v2")),
        "dataset_version": str(config.get("dataset_version", "yield_corpus_v2")),
    }

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    with args.output_model.open("wb") as fh:
        pickle.dump(
            {
                "pipeline": pipeline,
                "backend": backend_name,
                "interval_width_p90": interval_width,
                "numeric_columns": dataset.numeric_columns,
                "categorical_columns": dataset.categorical_columns,
                "model_version": report["model_version"],
                "dataset_version": report["dataset_version"],
            },
            fh,
        )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print(args.output_model.resolve())
    print(args.report.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
