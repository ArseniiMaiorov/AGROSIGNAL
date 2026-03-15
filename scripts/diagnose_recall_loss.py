#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


DEFAULT_BASE_URL = os.getenv("AUTODETECT_BASE_URL", "http://localhost:8000")
DEFAULT_EMAIL = os.getenv("AUTH_BOOTSTRAP_ADMIN_EMAIL", "admin@local")
DEFAULT_PASSWORD = os.getenv("AUTH_BOOTSTRAP_ADMIN_PASSWORD", "admin12345")
DEFAULT_ORG = os.getenv("AUTH_BOOTSTRAP_ORG_SLUG", "default-organization")
DEFAULT_MATRIX = "backend/training/release_russia_qa_matrix.json"
DEFAULT_ITEMS = (
    "krasnodar_01",
    "voronezh_01",
    "bashkortostan_01",
    "novosibirsk_01",
    "primorsky_01",
)


MODE_PRESETS = {
    "standard": {"use_sam": False, "resolution_m": 30, "target_dates": 7},
    "quality": {"use_sam": True, "resolution_m": 10, "target_dates": 9},
}

RANKING_REASONS = {
    "below_min_score",
    "suppressed_overlap",
}

TERMINAL_REASONS = {
    "dropped_by_object_classifier",
    "dropped_after_topology_cleanup",
    "dropped_after_merge",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run targeted autodetect AOI and diagnose recall loss stage.")
    parser.add_argument("--matrix", default=DEFAULT_MATRIX)
    parser.add_argument(
        "--items",
        default=",".join(DEFAULT_ITEMS),
        help="Comma-separated matrix item ids, defaults to the fixed 5-AOI recall set",
    )
    parser.add_argument("--mode", choices=sorted(MODE_PRESETS), help="Single mode shortcut, deprecated by --modes")
    parser.add_argument(
        "--modes",
        default="standard,quality",
        help="Comma-separated modes to run for each AOI, e.g. standard,quality",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--email", default=DEFAULT_EMAIL)
    parser.add_argument("--password", default=DEFAULT_PASSWORD)
    parser.add_argument("--organization-slug", default=DEFAULT_ORG)
    parser.add_argument("--poll-interval", type=float, default=3.0)
    parser.add_argument("--detect-timeout", type=int, default=900)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--output", default="backend/debug/runs/recall_diagnosis.jsonl")
    parser.add_argument("--summary-output", default="")
    parser.add_argument("--crop-code", default="wheat")
    return parser.parse_args()


def _require_ok(response: httpx.Response, step: str) -> Any:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(f"{step}: HTTP {response.status_code} -> {response.text}") from exc
    return response.json() if response.content else None


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _load_items(path: Path, ids: list[str]) -> list[dict[str, Any]]:
    matrix = json.loads(path.read_text(encoding="utf-8"))
    indexed = {str(item["id"]): item for item in list(matrix.get("items") or [])}
    missing = [item_id for item_id in ids if item_id not in indexed]
    if missing:
        raise SystemExit(f"Unknown matrix items: {missing}")
    return [indexed[item_id] for item_id in ids]


def _sum_area(geojson: dict[str, Any] | None) -> float:
    features = (geojson or {}).get("features") or []
    return float(sum(float((feature.get("properties") or {}).get("area_m2") or 0.0) for feature in features))


def _parse_modes(args: argparse.Namespace) -> list[str]:
    if args.mode:
        return [str(args.mode)]
    modes = [token.strip() for token in str(args.modes or "").split(",") if token.strip()]
    invalid = [mode for mode in modes if mode not in MODE_PRESETS]
    if invalid:
        raise SystemExit(f"Unknown modes: {invalid}")
    if not modes:
        raise SystemExit("At least one detection mode must be provided")
    return modes


def _reason_stage(reason: str | None) -> str:
    token = str(reason or "")
    if token in TERMINAL_REASONS:
        return "obia_topology_cleanup"
    if token in RANKING_REASONS or token.startswith("suppressed_by_"):
        return "rank_and_suppress"
    if token == "quality_gate_failed":
        return "quality_gate"
    if token:
        return "candidate_filtering"
    return "unknown"


def _classify_run_failure(result_payload: dict[str, Any], runtime: dict[str, Any], field_count: int) -> str:
    candidates_total = int(result_payload.get("candidates_total") or 0)
    candidates_kept = int(result_payload.get("candidates_kept") or 0)
    reject_summary = dict(result_payload.get("candidate_reject_summary") or {})
    status = str(result_payload.get("status") or "")

    if status in {"failed", "stale", "cancelled"}:
        return "run_failed"
    if candidates_total == 0:
        return "candidate_generation"
    if candidates_kept == 0:
        return "rank_and_suppress"
    if field_count == 0:
        if any(reason in TERMINAL_REASONS for reason in reject_summary):
            return "obia_topology_cleanup"
        return "rank_and_suppress"
    if any(reason in TERMINAL_REASONS for reason in reject_summary):
        return "obia_topology_cleanup"
    if reject_summary:
        return "partial_rank_and_suppress"
    if not runtime.get("tiles"):
        return "unknown"
    return "no_major_recall_loss"


def _tile_loss_summary(runtime: dict[str, Any]) -> dict[str, Any]:
    tiles = list(runtime.get("tiles") or [])
    stage_counts: Counter[str] = Counter()
    profile_counts: Counter[str] = Counter()
    qc_counts: Counter[str] = Counter()
    worst_tiles: list[dict[str, Any]] = []

    for tile in tiles:
        profile = str(tile.get("processing_profile") or "unknown")
        qc_mode = str(tile.get("qc_mode") or "unknown")
        profile_counts[profile] += 1
        qc_counts[qc_mode] += 1

        candidates_total = int(tile.get("candidates_total") or 0)
        candidates_kept = int(tile.get("candidates_kept") or 0)
        reject_summary = dict(tile.get("candidate_reject_summary") or {})

        if candidates_total == 0:
            stage = "candidate_generation"
        elif candidates_kept == 0:
            reasons = list(reject_summary)
            if any(_reason_stage(reason) == "rank_and_suppress" for reason in reasons):
                stage = "rank_and_suppress"
            elif any(_reason_stage(reason) == "obia_topology_cleanup" for reason in reasons):
                stage = "obia_topology_cleanup"
            else:
                stage = "candidate_filtering"
        else:
            stage = "survived"
        stage_counts[stage] += 1

        worst_tiles.append(
            {
                "tile_id": tile.get("tile_id"),
                "qc_mode": qc_mode,
                "processing_profile": profile,
                "candidates_total": candidates_total,
                "candidates_kept": candidates_kept,
                "boundary_first_bias_applied": bool(tile.get("boundary_first_bias_applied")),
                "recovery_second_pass_used": bool(tile.get("recovery_second_pass_used")),
                "recovery_missed_zone_pixels": int(tile.get("recovery_missed_zone_pixels") or 0),
                "candidate_reject_summary": reject_summary,
                "loss_stage": stage,
            }
        )

    worst_tiles.sort(
        key=lambda tile: (
            0 if tile["loss_stage"] != "survived" else 1,
            tile["candidates_kept"],
            tile["candidates_total"],
        )
    )
    return {
        "tile_loss_stage_counts": dict(stage_counts),
        "processing_profile_counts": dict(profile_counts),
        "qc_mode_counts": dict(qc_counts),
        "worst_tiles": worst_tiles[:8],
    }


def _dominant_reject_reasons(reject_summary: dict[str, Any], *, limit: int = 5) -> list[str]:
    pairs = sorted(
        ((str(reason), int(count or 0)) for reason, count in dict(reject_summary or {}).items()),
        key=lambda item: (-item[1], item[0]),
    )
    return [reason for reason, _ in pairs[:limit]]


def _next_fix_target(diagnosis: str, reject_summary: dict[str, Any], tile_summary: dict[str, Any]) -> str:
    normalized = str(diagnosis or "unknown")
    reasons = _dominant_reject_reasons(reject_summary)
    profile_counts = dict(tile_summary.get("processing_profile_counts") or {})
    qc_counts = dict(tile_summary.get("qc_mode_counts") or {})
    worst_tiles = list(tile_summary.get("worst_tiles") or [])
    if normalized == "candidate_generation":
        if profile_counts.get("boundary_recovery") or qc_counts.get("boundary_recovery"):
            return "generation_recovery_bias"
        return "generation_harmonized_inputs"
    if normalized in {"rank_and_suppress", "partial_rank_and_suppress"}:
        if any(reason in {"below_min_score", "suppressed_overlap"} for reason in reasons):
            return "ranking_thresholds"
        return "ranking_object_features"
    if normalized == "obia_topology_cleanup":
        return "obia_topology_cleanup"
    if normalized == "run_failed":
        return "runtime_failure"
    if normalized == "candidate_filtering":
        if any(tile.get("processing_profile") == "degraded_output" for tile in worst_tiles):
            return "degraded_output_thresholds"
        return "candidate_filtering"
    return "observe"


def _summary_output_path(output_path: Path, explicit: str) -> Path:
    if explicit:
        return Path(explicit)
    return output_path.with_name(f"{output_path.stem}_summary.json")


def _markdown_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_summary.md")


def _write_summary(summary_path: Path, records: list[dict[str, Any]]) -> None:
    by_item: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_item.setdefault(str(record["item_id"]), []).append(record)

    item_summaries: list[dict[str, Any]] = []
    stage_counts: Counter[str] = Counter()
    fix_target_counts: Counter[str] = Counter()
    for item_id, item_records in sorted(by_item.items()):
        ordered = sorted(item_records, key=lambda item: (str(item["item_id"]), str(item["mode"])))
        stage_values = [str(item.get("diagnosed_loss_stage") or "unknown") for item in ordered]
        target_values = [str(item.get("next_fix_target") or "observe") for item in ordered]
        for stage in stage_values:
            stage_counts[stage] += 1
        for target in target_values:
            fix_target_counts[target] += 1
        item_summaries.append(
            {
                "item_id": item_id,
                "region": ordered[0].get("region"),
                "modes": [
                    {
                        "mode": item.get("mode"),
                        "run_id": item.get("run_id"),
                        "status": item.get("status"),
                        "diagnosed_loss_stage": item.get("diagnosed_loss_stage"),
                        "next_fix_target": item.get("next_fix_target"),
                        "dominant_reject_reasons": item.get("dominant_reject_reasons"),
                        "qc_mode": item.get("qc_mode"),
                        "processing_profile": item.get("processing_profile"),
                        "field_count": item.get("field_count"),
                        "candidates_total": item.get("candidates_total"),
                        "candidates_kept": item.get("candidates_kept"),
                    }
                    for item in ordered
                ],
            }
        )

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "items": item_summaries,
        "stage_counts": dict(stage_counts),
        "next_fix_target_counts": dict(fix_target_counts),
        "records_total": len(records),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "| item_id | mode | stage | next_fix_target | qc_mode | profile | fields | kept/total | dominant_reject_reasons |",
        "|---|---|---|---|---|---|---:|---:|---|",
    ]
    for item_summary in item_summaries:
        for mode_row in item_summary["modes"]:
            md_lines.append(
                "| {item} | {mode} | {stage} | {target} | {qc} | {profile} | {fields} | {kept}/{total} | {reasons} |".format(
                    item=item_summary["item_id"],
                    mode=mode_row["mode"],
                    stage=mode_row["diagnosed_loss_stage"],
                    target=mode_row["next_fix_target"],
                    qc=mode_row["qc_mode"],
                    profile=mode_row["processing_profile"],
                    fields=mode_row["field_count"],
                    kept=mode_row["candidates_kept"],
                    total=mode_row["candidates_total"],
                    reasons=", ".join(mode_row["dominant_reject_reasons"] or []),
                )
            )
    _markdown_output_path(summary_path).write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def _build_detect_payload(item: dict[str, Any], mode: str) -> dict[str, Any]:
    preset = MODE_PRESETS[mode]
    return {
        "aoi": {
            "type": "point_radius",
            "lat": float(item["lat"]),
            "lon": float(item["lon"]),
            "radius_km": 3.0,
        },
        "time_range": {
            "start_date": "2025-05-01",
            "end_date": "2025-08-31",
        },
        "resolution_m": int(preset["resolution_m"]),
        "max_cloud_pct": 40,
        "target_dates": int(preset["target_dates"]),
        "min_field_area_ha": 0.5,
        "seed_mode": "edges",
        "debug": False,
    }


def _poll_run(
    client: httpx.Client,
    token: str,
    run_id: str,
    *,
    timeout_s: int,
    poll_interval: float,
) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    final_status: dict[str, Any] | None = None
    timeline: list[dict[str, Any]] = []

    while time.time() < deadline:
        status_payload = _require_ok(
            client.get(f"/api/v1/fields/status/{run_id}", headers=_headers(token)),
            f"status {run_id}",
        )
        timeline.append(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "status": status_payload.get("status"),
                "progress": status_payload.get("progress"),
                "stage_label": status_payload.get("stage_label"),
                "stage_detail": status_payload.get("stage_detail"),
                "qc_mode": status_payload.get("qc_mode"),
                "processing_profile": status_payload.get("processing_profile"),
            }
        )
        if status_payload.get("status") in {"done", "failed", "stale", "cancelled"}:
            final_status = status_payload
            break
        time.sleep(poll_interval)

    if final_status is None:
        raise RuntimeError(f"detect timeout for run={run_id}")
    final_status["timeline"] = timeline
    return final_status


def main() -> int:
    args = parse_args()
    item_ids = [item.strip() for item in str(args.items).split(",") if item.strip()]
    modes = _parse_modes(args)
    items = _load_items(Path(args.matrix), item_ids)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

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

        with output_path.open("a", encoding="utf-8") as handle:
            for item in items:
                for mode in modes:
                    payload = _build_detect_payload(item, mode)
                    started_at = time.perf_counter()
                    detect = _require_ok(
                        client.post(
                            f"/api/v1/fields/detect?use_sam={'true' if MODE_PRESETS[mode]['use_sam'] else 'false'}",
                            headers=_headers(token),
                            json=payload,
                        ),
                        f"detect submit {item['id']} {mode}",
                    )
                    run_id = str(detect["aoi_run_id"])
                    final_status = _poll_run(
                        client,
                        token,
                        run_id,
                        timeout_s=args.detect_timeout,
                        poll_interval=args.poll_interval,
                    )
                    result_payload = _require_ok(
                        client.get(f"/api/v1/fields/result/{run_id}", headers=_headers(token)),
                        f"detect result {item['id']} {mode}",
                    )
                    candidate_payload = _require_ok(
                        client.get(
                            f"/api/v1/fields/runs/{run_id}/candidates",
                            headers=_headers(token),
                            params={"limit": 200},
                        ),
                        f"candidate list {item['id']} {mode}",
                    )
                    debug_tiles = _require_ok(
                        client.get(f"/api/v1/fields/runs/{run_id}/debug/tiles", headers=_headers(token)),
                        f"debug tiles {item['id']} {mode}",
                    )

                    runtime = dict(result_payload.get("runtime") or {})
                    tile_summary = _tile_loss_summary(runtime)
                    field_count = len(((result_payload.get("geojson") or {}).get("features") or []))
                    diagnosis = _classify_run_failure(result_payload, runtime, field_count)
                    dominant_reasons = _dominant_reject_reasons(result_payload.get("candidate_reject_summary") or {})
                    next_fix_target = _next_fix_target(
                        diagnosis,
                        result_payload.get("candidate_reject_summary") or {},
                        tile_summary,
                    )

                    record = {
                        "item_id": item["id"],
                        "region": item["region"],
                        "lat": item["lat"],
                        "lon": item["lon"],
                        "mode": mode,
                        "run_id": run_id,
                        "wall_s": round(time.perf_counter() - started_at, 2),
                        "status": result_payload.get("status"),
                        "progress": result_payload.get("progress"),
                        "qc_mode": result_payload.get("qc_mode"),
                        "processing_profile": result_payload.get("processing_profile"),
                        "field_count": field_count,
                        "total_area_m2": round(_sum_area(result_payload.get("geojson")), 2),
                        "candidates_total": int(result_payload.get("candidates_total") or 0),
                        "candidates_kept": int(result_payload.get("candidates_kept") or 0),
                        "candidate_branch_counts": result_payload.get("candidate_branch_counts") or {},
                        "candidate_reject_summary": result_payload.get("candidate_reject_summary") or {},
                        "dominant_reject_reasons": dominant_reasons,
                        "diagnosed_loss_stage": diagnosis,
                        "next_fix_target": next_fix_target,
                        "tile_summary": tile_summary,
                        "sample_candidates": list((candidate_payload.get("candidates") or [])[:12]),
                        "debug_tile_count": len(list(debug_tiles.get("tiles") or [])),
                        "timeline_tail": list((final_status.get("timeline") or [])[-8:]),
                        "error_msg": result_payload.get("error_msg"),
                    }
                    records.append(record)
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    print(json.dumps(record, ensure_ascii=False), flush=True)

    _write_summary(_summary_output_path(output_path, args.summary_output), records)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover
        print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        raise
