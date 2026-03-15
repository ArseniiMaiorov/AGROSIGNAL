#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DEFAULT_CASES = (
    {
        "name": "south_smoke",
        "lat": 45.04500016938014,
        "lon": 39.06242376217776,
        "radius_km": 0.35,
    },
    {
        "name": "north_smoke",
        "lat": 58.691208,
        "lon": 29.893892,
        "radius_km": 0.35,
    },
)


def _json_request(method: str, url: str, payload: dict[str, Any] | None = None, timeout: float = 60.0) -> Any:
    data = None
    headers = {"accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["content-type"] = "application/json"
    req = Request(url, data=data, headers=headers, method=method)
    with urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body) if body else {}


def _poll_result(base_url: str, run_id: str, *, poll_s: float, timeout_s: float) -> dict[str, Any]:
    status_url = f"{base_url}/status/{run_id}"
    result_url = f"{base_url}/result/{run_id}"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        status = _json_request("GET", status_url)
        state = str(status.get("status") or "")
        progress = int(status.get("progress") or 0)
        print(f"  run {run_id} status={state} progress={progress}")
        if state == "completed":
            return _json_request("GET", result_url)
        if state == "failed":
            raise RuntimeError(f"Run {run_id} failed: {status.get('error_msg')}")
        time.sleep(poll_s)
    raise TimeoutError(f"Run {run_id} did not finish within {timeout_s:.0f}s")


def _summarize_geojson(geojson: dict[str, Any] | None) -> dict[str, Any]:
    features = list((geojson or {}).get("features") or [])
    areas = [
        float((feat.get("properties") or {}).get("area_ha") or 0.0)
        for feat in features
        if isinstance((feat.get("properties") or {}).get("area_ha"), (int, float))
    ]
    perimeters = [
        float((feat.get("properties") or {}).get("perimeter_m") or 0.0)
        for feat in features
        if isinstance((feat.get("properties") or {}).get("perimeter_m"), (int, float))
    ]
    return {
        "n_fields": len(features),
        "total_area_ha": round(sum(areas), 4),
        "median_area_ha": round(sorted(areas)[len(areas) // 2], 4) if areas else 0.0,
        "median_perimeter_m": round(sorted(perimeters)[len(perimeters) // 2], 2) if perimeters else 0.0,
    }


def _run_case(base_url: str, case: dict[str, Any], *, use_sam: bool, poll_s: float, timeout_s: float) -> dict[str, Any]:
    payload = {
        "aoi": {
            "type": "point_radius",
            "lat": case["lat"],
            "lon": case["lon"],
            "radius_km": case["radius_km"],
        },
        "time_range": {
            "start_date": "2025-04-15",
            "end_date": "2025-09-30",
        },
        "use_sam": bool(use_sam),
        "config": {
            "FEATURE_ML_PRIMARY": True,
        },
    }
    detect = _json_request("POST", f"{base_url}/detect", payload=payload)
    run_id = str(detect["aoi_run_id"])
    print(f"{case['name']} use_sam={use_sam} run_id={run_id}")
    result = _poll_result(base_url, run_id, poll_s=poll_s, timeout_s=timeout_s)
    runtime = dict(result.get("runtime") or {})
    geojson = result.get("geojson")
    return {
        "name": case["name"],
        "use_sam": bool(use_sam),
        "run_id": run_id,
        "status": result.get("status"),
        "progress": int(result.get("progress") or 0),
        "runtime": {
            "progress_stage": runtime.get("progress_stage"),
            "sam_runtime_mode": runtime.get("sam_runtime_mode"),
            "sam_failure_reason": runtime.get("sam_failure_reason"),
            "ml_primary_used": runtime.get("ml_primary_used"),
            "ml_quality_score": runtime.get("ml_quality_score"),
            "region_boundary_profile": runtime.get("region_boundary_profile"),
            "fusion_profile": runtime.get("fusion_profile"),
            "n_valid_scenes": runtime.get("n_valid_scenes"),
            "edge_signal_p90": runtime.get("edge_signal_p90"),
            "area_change_post_smooth": runtime.get("area_change_post_smooth"),
        },
        "summary": _summarize_geojson(geojson),
        "geojson": geojson,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a lightweight live baseline-vs-SAM smoke test")
    parser.add_argument("--base-url", default="http://localhost:8000/api/v1/fields")
    parser.add_argument("--output", type=Path, default=Path("backend/debug/runs/live_sam_ab_smoke.json"))
    parser.add_argument("--poll-s", type=float, default=5.0)
    parser.add_argument("--timeout-s", type=float, default=1200.0)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for case in DEFAULT_CASES:
        for use_sam in (False, True):
            rows.append(
                _run_case(
                    args.base_url,
                    case,
                    use_sam=use_sam,
                    poll_s=float(args.poll_s),
                    timeout_s=float(args.timeout_s),
                )
            )

    summary: dict[str, Any] = {"rows": rows, "comparisons": []}
    for case in DEFAULT_CASES:
        baseline = next(r for r in rows if r["name"] == case["name"] and not r["use_sam"])
        sam = next(r for r in rows if r["name"] == case["name"] and r["use_sam"])
        summary["comparisons"].append(
            {
                "name": case["name"],
                "baseline_run_id": baseline["run_id"],
                "sam_run_id": sam["run_id"],
                "field_count_delta": sam["summary"]["n_fields"] - baseline["summary"]["n_fields"],
                "total_area_ha_delta": round(
                    float(sam["summary"]["total_area_ha"]) - float(baseline["summary"]["total_area_ha"]),
                    4,
                ),
                "baseline_sam_mode": baseline["runtime"].get("sam_runtime_mode"),
                "sam_enabled_mode": sam["runtime"].get("sam_runtime_mode"),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
