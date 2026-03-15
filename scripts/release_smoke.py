#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx


DEFAULT_BASE_URL = os.getenv("AUTODETECT_BASE_URL", "http://localhost:8000")
DEFAULT_EMAIL = os.getenv("AUTH_BOOTSTRAP_ADMIN_EMAIL", "admin@local")
DEFAULT_PASSWORD = os.getenv("AUTH_BOOTSTRAP_ADMIN_PASSWORD", "admin12345")
DEFAULT_ORG = os.getenv("AUTH_BOOTSTRAP_ORG_SLUG", "default-organization")


@dataclass
class SmokeResult:
    name: str
    ok: bool
    detail: str
    payload: dict[str, Any] | None = None


def _headers(token: str | None) -> dict[str, str]:
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def _polygon(lat: float, lon: float, dx: float, dy: float) -> dict[str, Any]:
    return {
        "type": "Polygon",
        "coordinates": [[
            [lon - dx, lat - dy],
            [lon + dx, lat - dy],
            [lon + dx, lat + dy],
            [lon - dx, lat + dy],
            [lon - dx, lat - dy],
        ]],
    }


def _split_line(lat: float, lon: float, dy: float) -> dict[str, Any]:
    return {
        "type": "LineString",
        "coordinates": [
            [lon, lat - dy],
            [lon, lat + dy],
        ],
    }


def _require_ok(response: httpx.Response, step: str) -> Any:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(f"{step}: HTTP {response.status_code} -> {response.text}") from exc
    return response.json() if response.content else None


def _sum_area(geojson: dict[str, Any] | None) -> float:
    features = (geojson or {}).get("features") or []
    return float(sum(float((feature.get("properties") or {}).get("area_m2") or 0.0) for feature in features))


def _poll_job(
    client: httpx.Client,
    *,
    task_id: str,
    status_url: str,
    result_url: str,
    headers: dict[str, str],
    timeout_s: int,
    poll_interval: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    deadline = time.time() + timeout_s
    final_status: dict[str, Any] | None = None
    while time.time() < deadline:
        status_payload = _require_ok(client.get(status_url, headers=headers), f"job status {task_id}")
        if status_payload.get("status") in {"done", "failed"}:
            final_status = status_payload
            break
        time.sleep(poll_interval)
    if final_status is None:
        raise RuntimeError(f"job poll timeout after {timeout_s}s for task {task_id}")
    result_payload = _require_ok(client.get(result_url, headers=headers), f"job result {task_id}")
    return final_status, result_payload


def run_release_smoke(args: argparse.Namespace) -> int:
    results: list[SmokeResult] = []
    lat = float(args.lat)
    lon = float(args.lon)

    with httpx.Client(base_url=args.base_url.rstrip("/"), timeout=args.timeout) as client:
        bootstrap = _require_ok(client.get("/api/v1/bootstrap"), "bootstrap")
        results.append(SmokeResult("bootstrap", True, bootstrap.get("status", "unknown"), bootstrap))

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
        refresh_token = str(login["refresh_token"])
        results.append(SmokeResult("login", True, login["user"]["email"], {"organization": login["user"]["organization_slug"]}))

        me = _require_ok(client.get("/api/v1/auth/me", headers=_headers(token)), "auth me")
        results.append(SmokeResult("auth.me", True, me["email"]))

        refreshed = _require_ok(
            client.post(
                "/api/v1/auth/refresh",
                json={"refresh_token": refresh_token},
                headers=_headers(token),
            ),
            "auth refresh",
        )
        token = str(refreshed["access_token"])
        results.append(SmokeResult("auth.refresh", True, "rotated"))

        for endpoint, params in (
            ("/api/v1/status", None),
            ("/api/v1/weather/current", {"lat": lat, "lon": lon}),
            ("/api/v1/weather/forecast", {"lat": lat, "lon": lon, "days": 5}),
            ("/api/v1/layers", None),
            ("/api/v1/crops", None),
        ):
            payload = _require_ok(client.get(endpoint, params=params, headers=_headers(token)), endpoint)
            results.append(SmokeResult(endpoint, True, "ok", payload if isinstance(payload, dict) else None))

        manual_1 = _require_ok(
            client.post(
                "/api/v1/manual/fields",
                headers=_headers(token),
                json={"geometry": _polygon(lat, lon - 0.0022, 0.0015, 0.0012), "quality_score": 1.0},
            ),
            "manual field 1",
        )
        manual_2 = _require_ok(
            client.post(
                "/api/v1/manual/fields",
                headers=_headers(token),
                json={"geometry": _polygon(lat, lon + 0.0022, 0.0015, 0.0012), "quality_score": 1.0},
            ),
            "manual field 2",
        )
        field_1 = str(manual_1["field"]["id"])
        field_2 = str(manual_2["field"]["id"])
        results.append(SmokeResult("manual.create", True, f"{field_1}, {field_2}"))

        merged = _require_ok(
            client.post(
                "/api/v1/fields/merge",
                headers=_headers(token),
                json={"field_ids": [field_1, field_2]},
            ),
            "fields merge",
        )
        merged_id = str(merged["id"])
        results.append(SmokeResult("fields.merge", True, merged_id))

        split = _require_ok(
            client.post(
                "/api/v1/fields/split",
                headers=_headers(token),
                json={"field_id": merged_id, "geometry": _split_line(lat, lon, 0.0025)},
            ),
            "fields split",
        )
        split_fields = split.get("fields") or []
        if len(split_fields) < 2:
            raise RuntimeError(f"fields split: expected >=2 fields, got {len(split_fields)}")
        target_field_id = str(split_fields[0]["id"])
        deleted_field_id = str(split_fields[1]["id"])
        results.append(SmokeResult("fields.split", True, f"{len(split_fields)} parts"))

        deleted = _require_ok(
            client.delete(f"/api/v1/fields/{deleted_field_id}", headers=_headers(token)),
            "fields delete",
        )
        results.append(SmokeResult("fields.delete", True, str(deleted.get("field_id"))))

        dashboard = _require_ok(
            client.get(f"/api/v1/fields/{target_field_id}/dashboard", headers=_headers(token)),
            "field dashboard",
        )
        results.append(SmokeResult("fields.dashboard", True, dashboard.get("mode", "unknown")))

        prediction_submit = _require_ok(
            client.post(
                f"/api/v1/predictions/field/{target_field_id}/jobs",
                headers=_headers(token),
                params={"crop_code": args.crop_code, "refresh": True},
            ),
            "prediction submit",
        )
        prediction_status, prediction_result = _poll_job(
            client,
            task_id=str(prediction_submit["task_id"]),
            status_url=f"/api/v1/predictions/jobs/{prediction_submit['task_id']}",
            result_url=f"/api/v1/predictions/jobs/{prediction_submit['task_id']}/result",
            headers=_headers(token),
            timeout_s=min(args.detect_timeout, 90),
            poll_interval=max(args.poll_interval, 0.8),
        )
        prediction = prediction_result.get("result") or {}
        results.append(
            SmokeResult(
                "prediction.refresh",
                True,
                f"{prediction.get('estimated_yield_kg_ha')} kg/ha",
                {
                    "confidence_tier": prediction.get("confidence_tier"),
                    "status": prediction_status.get("status"),
                },
            )
        )

        scenario_submit = _require_ok(
            client.post(
                "/api/v1/modeling/jobs",
                headers=_headers(token),
                json={
                    "field_id": target_field_id,
                    "crop_code": args.crop_code,
                    "scenario_name": "release_smoke",
                    "irrigation_pct": 10.0,
                    "fertilizer_pct": 5.0,
                    "expected_rain_mm": 20.0,
                },
            ),
            "scenario submit",
        )
        scenario_status, scenario_result = _poll_job(
            client,
            task_id=str(scenario_submit["task_id"]),
            status_url=f"/api/v1/modeling/jobs/{scenario_submit['task_id']}",
            result_url=f"/api/v1/modeling/jobs/{scenario_submit['task_id']}/result",
            headers=_headers(token),
            timeout_s=min(args.detect_timeout, 90),
            poll_interval=max(args.poll_interval, 0.8),
        )
        scenario = scenario_result.get("result") or {}
        results.append(
            SmokeResult(
                "scenario.simulate",
                True,
                (
                    f"supported={bool(scenario.get('supported', False))} "
                    f"delta={scenario.get('predicted_yield_change_pct')}"
                ),
                {
                    "confidence_tier": scenario.get("confidence_tier"),
                    "support_reason": scenario.get("support_reason"),
                    "status": scenario_status.get("status"),
                },
            )
        )

        now = datetime.now(timezone.utc)
        archive = _require_ok(
            client.post(
                "/api/v1/archive/create",
                headers=_headers(token),
                json={
                    "field_id": target_field_id,
                    "date_from": now.replace(month=5, day=1, hour=0, minute=0, second=0, microsecond=0).isoformat(),
                    "date_to": now.replace(month=8, day=31, hour=23, minute=59, second=0, microsecond=0).isoformat(),
                    "layers": ["ndvi", "weather"],
                },
            ),
            "archive create",
        )
        archive_id = int(archive["id"])
        _require_ok(
            client.get("/api/v1/archive", headers=_headers(token), params={"field_id": target_field_id}),
            "archive list",
        )
        view = _require_ok(
            client.get(f"/api/v1/archive/{archive_id}/view", headers=_headers(token)),
            "archive view",
        )
        results.append(SmokeResult("archive", True, f"id={archive_id}", {"snapshot_keys": sorted((view.get('snapshot') or {}).keys())}))

        detect_start = time.perf_counter()
        detect = _require_ok(
            client.post(
                f"/api/v1/fields/detect?use_sam={'true' if args.use_sam else 'false'}",
                headers=_headers(token),
                json={
                    "aoi": {"type": "point_radius", "lat": lat, "lon": lon, "radius_km": args.radius_km},
                    "time_range": {"start_date": args.start_date, "end_date": args.end_date},
                    "resolution_m": args.resolution_m,
                    "max_cloud_pct": args.max_cloud_pct,
                    "target_dates": args.target_dates,
                    "min_field_area_ha": args.min_field_area_ha,
                    "seed_mode": "edges",
                    "debug": False,
                },
            ),
            "detect start",
        )
        run_id = str(detect["aoi_run_id"])
        timeline: list[dict[str, Any]] = []
        final_status: dict[str, Any] | None = None
        deadline = time.time() + args.detect_timeout
        while time.time() < deadline:
            status_payload = _require_ok(
                client.get(f"/api/v1/fields/status/{run_id}", headers=_headers(token)),
                "detect status",
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
            raise RuntimeError(f"detect poll timeout after {args.detect_timeout}s for run {run_id}")

        result_payload = _require_ok(
            client.get(f"/api/v1/fields/result/{run_id}", headers=_headers(token)),
            "detect result",
        )
        field_count = len(((result_payload.get("geojson") or {}).get("features") or []))
        total_area = _sum_area(result_payload.get("geojson"))
        results.append(
            SmokeResult(
                "detect",
                final_status.get("status") == "done",
                f"status={final_status.get('status')} progress={final_status.get('progress')} wall_s={time.perf_counter() - detect_start:.1f}",
                {
                    "run_id": run_id,
                    "timeline": timeline,
                    "field_count": field_count,
                    "total_area_m2": round(total_area, 2),
                    "error_msg": result_payload.get("error_msg"),
                },
            )
        )

        _require_ok(
            client.post(
                "/api/v1/auth/logout",
                json={"refresh_token": refreshed["refresh_token"]},
                headers=_headers(token),
            ),
            "auth logout",
        )
        results.append(SmokeResult("auth.logout", True, "ok"))

    failed = [item for item in results if not item.ok]
    summary = {
        "base_url": args.base_url,
        "results": [item.__dict__ for item in results],
        "failed_count": len(failed),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if failed else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run release-candidate API smoke against a live AutoDetect stack.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--email", default=DEFAULT_EMAIL)
    parser.add_argument("--password", default=DEFAULT_PASSWORD)
    parser.add_argument("--organization-slug", default=DEFAULT_ORG)
    parser.add_argument("--lat", type=float, default=45.2307)
    parser.add_argument("--lon", type=float, default=38.7199)
    parser.add_argument("--crop-code", default="wheat")
    parser.add_argument("--radius-km", type=float, default=3.0)
    parser.add_argument("--start-date", default="2025-05-01")
    parser.add_argument("--end-date", default="2025-08-31")
    parser.add_argument("--resolution-m", type=int, default=30)
    parser.add_argument("--max-cloud-pct", type=int, default=40)
    parser.add_argument("--target-dates", type=int, default=7)
    parser.add_argument("--min-field-area-ha", type=float, default=0.5)
    parser.add_argument("--detect-timeout", type=int, default=600)
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--use-sam", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        raise SystemExit(run_release_smoke(parse_args()))
    except Exception as exc:  # pragma: no cover
        print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        raise
