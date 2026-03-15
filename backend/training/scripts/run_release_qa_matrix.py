#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


DEFAULT_BASE_URL = os.getenv("AUTODETECT_BASE_URL", "http://localhost:8000")
DEFAULT_EMAIL = os.getenv("AUTH_BOOTSTRAP_ADMIN_EMAIL", "admin@local")
DEFAULT_PASSWORD = os.getenv("AUTH_BOOTSTRAP_ADMIN_PASSWORD", "admin12345")
DEFAULT_ORG = os.getenv("AUTH_BOOTSTRAP_ORG_SLUG", "default-organization")

MODE_PRESETS = {
    "standard": {"use_sam": False, "resolution_m": None, "target_dates": 7},
    "quality": {"use_sam": True, "resolution_m": 10, "target_dates": 9},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the release Russia QA matrix through a live API.")
    parser.add_argument("matrix", nargs="?", default="backend/training/release_russia_qa_matrix.json")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--email", default=DEFAULT_EMAIL)
    parser.add_argument("--password", default=DEFAULT_PASSWORD)
    parser.add_argument("--organization-slug", default=DEFAULT_ORG)
    parser.add_argument("--output", default="backend/training/release_russia_qa_results.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="Run only first N matrix rows (0 = all).")
    parser.add_argument("--detect-timeout", type=int, default=900)
    parser.add_argument("--poll-interval", type=float, default=3.0)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--crop-code", default="wheat")
    return parser.parse_args()


def _require_ok(response: httpx.Response, step: str) -> Any:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(f"{step}: HTTP {response.status_code} -> {response.text}") from exc
    return response.json() if response.content else None


def _sum_area(geojson: dict[str, Any] | None) -> float:
    features = (geojson or {}).get("features") or []
    return float(sum(float((feature.get("properties") or {}).get("area_m2") or 0.0) for feature in features))


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def main() -> int:
    args = parse_args()
    matrix_path = Path(args.matrix)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    items = list(matrix.get("items") or [])
    if args.limit > 0:
        items = items[: args.limit]
    modes = list(matrix.get("modes") or ["standard", "quality"])
    default_radius_km = float(matrix.get("default_radius_km") or 3.0)
    default_resolution_m = int(matrix.get("default_resolution_m") or 30)
    time_range = matrix.get("default_time_range") or {"start_date": "2025-05-01", "end_date": "2025-08-31"}

    with httpx.Client(base_url=args.base_url.rstrip("/"), timeout=args.timeout) as client:
        login = _require_ok(
            client.post(
                "/api/v1/auth/login",
                json={
                    "email": args.email,
                    "password": args.password,
                    "organization_slug": args.organization_slug,
                },
            ),
            "login",
        )
        token = str(login["access_token"])

        with output_path.open("w", encoding="utf-8") as handle:
            for item in items:
                for mode in modes:
                    preset = MODE_PRESETS.get(mode, MODE_PRESETS["standard"])
                    payload = {
                        "aoi": {
                            "type": "point_radius",
                            "lat": float(item["lat"]),
                            "lon": float(item["lon"]),
                            "radius_km": default_radius_km,
                        },
                        "time_range": {
                            "start_date": str(time_range["start_date"]),
                            "end_date": str(time_range["end_date"]),
                        },
                        "resolution_m": int(preset["resolution_m"] or default_resolution_m),
                        "max_cloud_pct": 40,
                        "target_dates": int(preset["target_dates"]),
                        "min_field_area_ha": 0.5,
                        "seed_mode": "edges",
                        "debug": False,
                    }
                    started_at = time.perf_counter()
                    detect = _require_ok(
                        client.post(
                            f"/api/v1/fields/detect?use_sam={'true' if preset['use_sam'] else 'false'}",
                            headers=_headers(token),
                            json=payload,
                        ),
                        f"detect submit {item['id']}:{mode}",
                    )
                    run_id = str(detect["aoi_run_id"])
                    timeline: list[dict[str, Any]] = []
                    final_status: dict[str, Any] | None = None
                    deadline = time.time() + args.detect_timeout
                    while time.time() < deadline:
                        status_payload = _require_ok(
                            client.get(f"/api/v1/fields/status/{run_id}", headers=_headers(token)),
                            f"detect status {item['id']}:{mode}",
                        )
                        timeline.append(
                            {
                                "ts": datetime.now(timezone.utc).isoformat(),
                                "status": status_payload.get("status"),
                                "progress": status_payload.get("progress"),
                                "stage_label": status_payload.get("stage_label"),
                                "stage_detail": status_payload.get("stage_detail"),
                                "stale_running": status_payload.get("stale_running"),
                            }
                        )
                        if status_payload.get("status") in {"done", "failed", "stale", "cancelled"}:
                            final_status = status_payload
                            break
                        time.sleep(args.poll_interval)

                    if final_status is None:
                        raise RuntimeError(f"detect timeout for {item['id']}:{mode} run={run_id}")

                    result_payload = _require_ok(
                        client.get(f"/api/v1/fields/result/{run_id}", headers=_headers(token)),
                        f"detect result {item['id']}:{mode}",
                    )
                    record = {
                        "item_id": item["id"],
                        "region": item["region"],
                        "lat": item["lat"],
                        "lon": item["lon"],
                        "mode": mode,
                        "run_id": run_id,
                        "queue_wall_s": round(time.perf_counter() - started_at, 2),
                        "status": final_status.get("status"),
                        "progress": final_status.get("progress"),
                        "stale_running": bool(final_status.get("stale_running")),
                        "error_msg": result_payload.get("error_msg"),
                        "field_count": len(((result_payload.get("geojson") or {}).get("features") or [])),
                        "total_area_m2": round(_sum_area(result_payload.get("geojson")), 2),
                        "empty_output": not bool(((result_payload.get("geojson") or {}).get("features") or [])),
                        "timeline": timeline,
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    print(json.dumps(record, ensure_ascii=False), flush=True)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover
        print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        raise
