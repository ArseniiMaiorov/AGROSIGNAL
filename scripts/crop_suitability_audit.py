#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

import httpx


DEFAULT_BASE_URL = os.getenv("AUTODETECT_BASE_URL", "http://localhost:8000")
DEFAULT_EMAIL = os.getenv("AUTH_BOOTSTRAP_ADMIN_EMAIL", "admin@local")
DEFAULT_PASSWORD = os.getenv("AUTH_BOOTSTRAP_ADMIN_PASSWORD", "admin12345")
DEFAULT_ORG = os.getenv("AUTH_BOOTSTRAP_ORG_SLUG", "default-organization")

NORTH_LAT = float(os.getenv("CROP_AUDIT_LAT", "59.0353"))
NORTH_LON = float(os.getenv("CROP_AUDIT_LON", "54.6514"))


def _headers(token: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


def _require_ok(response: httpx.Response, step: str) -> Any:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(f"{step}: HTTP {response.status_code} -> {response.text}") from exc
    return response.json() if response.content else None


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


def _poll_job(
    client: httpx.Client,
    *,
    task_id: str,
    status_url: str,
    result_url: str,
    headers: dict[str, str],
    timeout_s: int,
    poll_interval: float,
) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        status_payload = _require_ok(client.get(status_url, headers=headers), f"job status {task_id}")
        if status_payload.get("status") in {"done", "failed"}:
            result_payload = _require_ok(client.get(result_url, headers=headers), f"job result {task_id}")
            return result_payload.get("result") or {}
        time.sleep(poll_interval)
    raise RuntimeError(f"prediction job timeout after {timeout_s}s")


def main() -> int:
    parser = argparse.ArgumentParser(description="Live audit for crop suitability in a northern region.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--email", default=DEFAULT_EMAIL)
    parser.add_argument("--password", default=DEFAULT_PASSWORD)
    parser.add_argument("--organization-slug", default=DEFAULT_ORG)
    parser.add_argument("--lat", type=float, default=NORTH_LAT)
    parser.add_argument("--lon", type=float, default=NORTH_LON)
    parser.add_argument("--timeout", type=float, default=25.0)
    parser.add_argument("--job-timeout", type=int, default=90)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    args = parser.parse_args()

    warm_crop_candidates = {"corn", "maize", "soy", "soybean"}
    field_id = None

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
        headers = _headers(token)

        crops_payload = _require_ok(client.get("/api/v1/crops", headers=headers), "crops")
        crops = crops_payload.get("crops") or []
        target_crop = next((crop for crop in crops if str(crop.get("code") or "").lower() in warm_crop_candidates), None)
        if target_crop is None:
            print("crop suitability audit skipped: no warm-season crop found in catalog")
            return 0

        manual = _require_ok(
            client.post(
                "/api/v1/manual/fields",
                headers=headers,
                json={
                    "geometry": _polygon(args.lat, args.lon, 0.0018, 0.0014),
                    "quality_score": 1.0,
                },
            ),
            "manual field create",
        )
        field_id = str(manual["field"]["id"])

        submit = _require_ok(
            client.post(
                f"/api/v1/predictions/field/{field_id}/jobs",
                headers=headers,
                params={"crop_code": target_crop["code"], "refresh": True},
            ),
            "prediction submit",
        )
        result = _poll_job(
            client,
            task_id=str(submit["task_id"]),
            status_url=f"/api/v1/predictions/jobs/{submit['task_id']}",
            result_url=f"/api/v1/predictions/jobs/{submit['task_id']}/result",
            headers=headers,
            timeout_s=args.job_timeout,
            poll_interval=args.poll_interval,
        )

        suitability = dict(result.get("crop_suitability") or {})
        status = str(suitability.get("status") or "unknown").lower()
        confidence_tier = str(result.get("confidence_tier") or "unsupported").lower()
        operational_tier = str(result.get("operational_tier") or "unsupported").lower()
        review_required = bool(result.get("review_required"))

        print(
            f"crop suitability audit: crop={target_crop['code']} status={status} "
            f"confidence_tier={confidence_tier} operational_tier={operational_tier} review_required={review_required}"
        )

        if status not in {"low", "unsuitable"}:
            raise RuntimeError(f"unexpected crop suitability status for northern warm-season crop: {status}")
        if confidence_tier == "tenant_calibrated":
            raise RuntimeError("warm-season crop in north must not be tenant-calibrated/high-trust by default")
        if operational_tier == "validated_core":
            raise RuntimeError("warm-season crop in north must not be marked as validated_core")
        if not review_required and operational_tier != "unsupported":
            raise RuntimeError("northern warm-season crop must require review or be unsupported")

    if field_id:
        with httpx.Client(base_url=args.base_url.rstrip("/"), timeout=args.timeout) as client:
            try:
                login = _require_ok(
                    client.post(
                        "/api/v1/auth/login",
                        json={
                            "email": args.email,
                            "password": args.password,
                            "organization_slug": args.organization_slug,
                        },
                    ),
                    "cleanup login",
                )
                client.delete(f"/api/v1/fields/{field_id}", headers=_headers(str(login["access_token"])))
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"crop suitability audit failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
