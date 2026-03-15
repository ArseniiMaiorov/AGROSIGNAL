#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import geopandas as gpd
import httpx
import numpy as np
from shapely.geometry import shape


def _load_holdout(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("items", [])
    if not isinstance(payload, list):
        raise ValueError("holdout file must contain a list")
    return payload


def _validate_holdout_items(path: Path, items: list[dict[str, Any]]) -> None:
    errors: list[str] = []
    missing_gt: list[str] = []
    for idx, item in enumerate(items):
        item_id = str(item.get("id") or f"index_{idx}")
        request = item.get("request")
        if not isinstance(request, dict):
            aoi = item.get("aoi")
            if isinstance(aoi, dict):
                request = {"aoi": aoi}
        gt_path_raw = item.get("ground_truth_geojson")
        if not isinstance(request, dict):
            errors.append(f"{item_id}: missing or invalid 'request'")
            continue
        if not isinstance(gt_path_raw, str) or not gt_path_raw.strip():
            errors.append(f"{item_id}: missing 'ground_truth_geojson'")
            continue
        if "/absolute/path/" in gt_path_raw or "your_holdout" in gt_path_raw:
            errors.append(
                f"{item_id}: placeholder ground_truth_geojson path detected: {gt_path_raw}"
            )
            continue
        gt_path = Path(gt_path_raw)
        if not gt_path.exists():
            missing_gt.append(f"{item_id}: {gt_path}")

    if missing_gt:
        errors.append("Missing ground truth files:\n" + "\n".join(missing_gt))
    if errors:
        msg = (
            f"Holdout config validation failed ({path}):\n"
            + "\n".join(f"- {line}" for line in errors)
        )
        raise ValueError(msg)


def _features_to_gdf(features: list[dict[str, Any]]) -> gpd.GeoDataFrame:
    geoms = [shape(item["geometry"]) for item in features]
    return gpd.GeoDataFrame(geometry=geoms, crs="EPSG:4326")


def _union_geometry(gdf: gpd.GeoDataFrame):
    if gdf.empty:
        return None
    geom_series = gdf.geometry
    union_all = getattr(geom_series, "union_all", None)
    if callable(union_all):
        return union_all()
    return geom_series.unary_union


def _estimate_utm_epsg(gdf: gpd.GeoDataFrame) -> int:
    """Estimate UTM EPSG code from the centroid of a GeoDataFrame."""
    _union = gdf.geometry.union_all() if hasattr(gdf.geometry, "union_all") else gdf.geometry.unary_union
    centroid = _union.centroid
    zone = int((centroid.x + 180) / 6) + 1
    return 32600 + zone if centroid.y >= 0 else 32700 + zone


def _to_utm(gdf: gpd.GeoDataFrame, ref_gdf: gpd.GeoDataFrame | None = None) -> gpd.GeoDataFrame:
    """Reproject to UTM zone estimated from ref_gdf (or gdf itself)."""
    epsg = _estimate_utm_epsg(ref_gdf if ref_gdf is not None else gdf)
    return gdf.to_crs(epsg=epsg)


def _area_metrics(pred_gdf: gpd.GeoDataFrame, gt_gdf: gpd.GeoDataFrame) -> dict[str, float]:
    pred = _to_utm(pred_gdf, gt_gdf)
    gt = _to_utm(gt_gdf)

    pred_union = _union_geometry(pred)
    gt_union = _union_geometry(gt)
    if pred_union is None:
        pred_area = 0.0
    else:
        pred_area = float(pred_union.area)
    if gt_union is None:
        gt_area = 0.0
    else:
        gt_area = float(gt_union.area)
    if pred_union is None or gt_union is None:
        inter_area = 0.0
    else:
        inter_area = float(pred_union.intersection(gt_union).area)
    union_area = max(pred_area + gt_area - inter_area, 1e-6)

    precision = inter_area / max(pred_area, 1e-6)
    recall = inter_area / max(gt_area, 1e-6)
    iou = inter_area / union_area
    f1 = (2.0 * precision * recall) / max(precision + recall, 1e-6)
    return {
        "iou": float(iou),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
    }


def _sample_boundary_distances(boundary_geom, other_boundary, *, spacing_m: float) -> list[float]:
    if boundary_geom is None or other_boundary is None:
        return []
    if boundary_geom.is_empty or other_boundary.is_empty:
        return []
    length = float(getattr(boundary_geom, "length", 0.0))
    if length <= 1e-6:
        return []
    n_samples = max(8, min(512, int(math.ceil(length / max(spacing_m, 1.0)))))
    distances: list[float] = []
    for i in range(n_samples + 1):
        pos = min(length, (length * i) / max(n_samples, 1))
        pt = boundary_geom.interpolate(pos)
        distances.append(float(pt.distance(other_boundary)))
    return distances


def _boundary_metrics(
    pred_gdf: gpd.GeoDataFrame,
    gt_gdf: gpd.GeoDataFrame,
    *,
    pixel_size_m: float = 10.0,
) -> dict[str, float]:
    pred = _to_utm(pred_gdf, gt_gdf)
    gt = _to_utm(gt_gdf)
    pred_union = _union_geometry(pred)
    gt_union = _union_geometry(gt)

    if pred_union is None or gt_union is None or pred_union.is_empty or gt_union.is_empty:
        return {
            "boundary_iou": 0.0,
            "boundary_iou_geo": 0.0,
            "hausdorff_95_px": 0.0,
            "hd95_m": 0.0,
            "centroid_shift_m": 0.0,
            "area_ratio_pred_gt": 0.0,
            "area_ratio_geo_median": 0.0,
        }

    pred_boundary = pred_union.boundary
    gt_boundary = gt_union.boundary
    pred_band = pred_boundary.buffer(max(pixel_size_m, 1.0))
    gt_band = gt_boundary.buffer(max(pixel_size_m, 1.0))
    inter_area = float(pred_band.intersection(gt_band).area)
    union_area = max(float(pred_band.union(gt_band).area), 1e-6)

    distances = _sample_boundary_distances(
        pred_boundary,
        gt_boundary,
        spacing_m=max(pixel_size_m, 1.0),
    ) + _sample_boundary_distances(
        gt_boundary,
        pred_boundary,
        spacing_m=max(pixel_size_m, 1.0),
    )
    hd95_px = (
        float(np.quantile(np.asarray(distances, dtype=float), 0.95) / max(pixel_size_m, 1.0))
        if distances
        else 0.0
    )

    pred_centroid = pred_union.centroid
    gt_centroid = gt_union.centroid
    centroid_shift_m = float(pred_centroid.distance(gt_centroid))

    pred_area = float(pred_union.area)
    gt_area = float(gt_union.area)
    area_ratio = pred_area / max(gt_area, 1e-6)

    return {
        "boundary_iou": float(inter_area / union_area),
        "boundary_iou_geo": float(inter_area / union_area),
        "hausdorff_95_px": hd95_px,
        "hd95_m": float(hd95_px * max(pixel_size_m, 1.0)),
        "centroid_shift_m": centroid_shift_m,
        "area_ratio_pred_gt": float(area_ratio),
        "area_ratio_geo_median": float(area_ratio),
    }


def _count_detection_mismatches(
    pred_gdf: gpd.GeoDataFrame,
    gt_gdf: gpd.GeoDataFrame,
    *,
    iou_threshold: float = 0.2,
) -> dict[str, int]:
    pred = _to_utm(pred_gdf, gt_gdf).reset_index(drop=True)
    gt = _to_utm(gt_gdf).reset_index(drop=True)
    if pred.empty and gt.empty:
        return {"missed_fields_count": 0, "oversegmented_fields_count": 0}
    if pred.empty:
        return {"missed_fields_count": int(len(gt)), "oversegmented_fields_count": 0}
    if gt.empty:
        return {"missed_fields_count": 0, "oversegmented_fields_count": int(len(pred))}

    matched_pred: set[int] = set()
    matched_gt = 0
    for gt_idx, gt_geom in enumerate(gt.geometry):
        best_idx = None
        best_iou = 0.0
        for pred_idx, pred_geom in enumerate(pred.geometry):
            if pred_idx in matched_pred:
                continue
            if not gt_geom.intersects(pred_geom):
                continue
            inter = float(gt_geom.intersection(pred_geom).area)
            union = max(float(gt_geom.union(pred_geom).area), 1e-6)
            iou = inter / union
            if iou > best_iou:
                best_iou = iou
                best_idx = pred_idx
        if best_idx is not None and best_iou >= iou_threshold:
            matched_gt += 1
            matched_pred.add(best_idx)

    return {
        "missed_fields_count": int(max(len(gt) - matched_gt, 0)),
        "oversegmented_fields_count": int(max(len(pred) - len(matched_pred), 0)),
        "missed_fields_rate": float(max(len(gt) - matched_gt, 0) / max(len(gt), 1)),
        "oversegmented_fields_rate": float(max(len(pred) - len(matched_pred), 0) / max(len(gt), 1)),
    }


def _run_detect(
    client: httpx.Client,
    base_url: str,
    payload: dict[str, Any],
    *,
    use_sam: bool,
    poll_interval_s: float,
    timeout_s: float,
) -> dict[str, Any]:
    resp = client.post(
        f"{base_url}/detect?use_sam={'true' if use_sam else 'false'}",
        json=payload,
        timeout=60.0,
    )
    resp.raise_for_status()
    run_id = resp.json()["aoi_run_id"]

    deadline = time.time() + timeout_s
    last_status = {}
    while time.time() < deadline:
        status_resp = client.get(f"{base_url}/status/{run_id}", timeout=30.0)
        status_resp.raise_for_status()
        last_status = status_resp.json()
        status = last_status.get("status")
        if status == "done":
            result_resp = client.get(f"{base_url}/result/{run_id}", timeout=60.0)
            result_resp.raise_for_status()
            return {"run_id": run_id, "status": last_status, "result": result_resp.json()}
        if status == "failed":
            return {"run_id": run_id, "status": last_status, "result": None}
        time.sleep(poll_interval_s)

    return {"run_id": run_id, "status": last_status, "result": None, "timeout": True}


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0}
    ordered = sorted(values)
    p50 = ordered[int(0.5 * (len(ordered) - 1))]
    p95 = ordered[int(0.95 * (len(ordered) - 1))]
    return {"p50": float(p50), "p95": float(p95)}


def _mode_config(ml_primary: bool) -> dict[str, Any]:
    return {
        "FEATURE_ML_PRIMARY": bool(ml_primary),
    }


def _mode_name(*, ml_primary: bool, use_sam: bool) -> str:
    if ml_primary and use_sam:
        return "ml_primary_sam"
    if ml_primary:
        return "ml_primary"
    if use_sam:
        return "rule_based_sam"
    return "rule_based"


def _evaluate_mode(
    client: httpx.Client,
    base_url: str,
    holdout_items: list[dict[str, Any]],
    *,
    ml_primary: bool,
    use_sam: bool,
    poll_interval_s: float,
    timeout_s: float,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    mode_name = _mode_name(ml_primary=ml_primary, use_sam=use_sam)

    for item in holdout_items:
        request_payload = item.get("request")
        if not isinstance(request_payload, dict):
            request_payload = {"aoi": dict(item.get("aoi") or {})}
        req = dict(request_payload)
        req_config = dict(req.get("config") or {})
        req_config.update(_mode_config(ml_primary))
        req["config"] = req_config

        run_payload = _run_detect(
            client,
            base_url,
            req,
            use_sam=use_sam,
            poll_interval_s=poll_interval_s,
            timeout_s=timeout_s,
        )
        status = run_payload.get("status", {})
        runtime = (status or {}).get("runtime") or {}
        result = run_payload.get("result")

        row = {
            "id": item.get("id"),
            "mode": mode_name,
            "run_id": run_payload.get("run_id"),
            "status": (status or {}).get("status"),
            "requested_use_sam": bool(use_sam),
            "fallback_rate_tile": runtime.get("fallback_rate_tile"),
            "latency_s": runtime.get("total_time_s"),
            "ml_primary_used": runtime.get("ml_primary_used"),
            "model_backend": runtime.get("model_backend"),
            "sam_runtime_mode": runtime.get("sam_runtime_mode"),
            "extent_threshold_used": runtime.get("ml_extent_threshold"),
            "geometry_refine_profile": runtime.get("geometry_refine_profile"),
            "hydro_boundary_profile": runtime.get("hydro_boundary_profile"),
            "hydro_complexity": item.get("hydro_complexity", "unknown"),
            "road_complexity": item.get("road_complexity", "unknown"),
            "water_adjacent_fields_present": bool(item.get("water_adjacent_fields_present", False)),
            "region_band": str(item.get("region_band") or "central"),
            "region_boundary_profile_target": str(
                item.get("region_boundary_profile_target") or "balanced"
            ),
            "error_mode_tag": str(item.get("error_mode_tag") or "none"),
            "parcel_shape_class": str(item.get("parcel_shape_class") or "irregular_large"),
            "adjacency_tag": str(item.get("adjacency_tag") or "none"),
        }

        if result and result.get("geojson"):
            pred_features = result["geojson"].get("features", [])
            pred_gdf = _features_to_gdf(pred_features)
            gt_gdf = gpd.read_file(item["ground_truth_geojson"])
            metrics = _area_metrics(pred_gdf, gt_gdf)
            metrics["iou_geo"] = metrics["iou"]
            metrics["f1_geo"] = metrics["f1"]
            metrics.update(_boundary_metrics(pred_gdf, gt_gdf))
            metrics.update(_count_detection_mismatches(pred_gdf, gt_gdf))
            row.update(metrics)
        else:
            row.update(
                {
                    "iou": 0.0,
                    "iou_geo": 0.0,
                    "f1": 0.0,
                    "f1_geo": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "boundary_iou": 0.0,
                    "boundary_iou_geo": 0.0,
                    "hausdorff_95_px": 0.0,
                    "hd95_m": 0.0,
                    "centroid_shift_m": 0.0,
                    "area_ratio_pred_gt": 0.0,
                    "area_ratio_geo_median": 0.0,
                    "missed_fields_count": 0,
                    "oversegmented_fields_count": 0,
                    "missed_fields_rate": 0.0,
                    "oversegmented_fields_rate": 0.0,
                }
            )
        rows.append(row)

    latencies = [float(r["latency_s"]) for r in rows if isinstance(r.get("latency_s"), (int, float))]
    fallback = [float(r["fallback_rate_tile"]) for r in rows if isinstance(r.get("fallback_rate_tile"), (int, float))]
    near_water_rows = [r for r in rows if bool(r.get("water_adjacent_fields_present"))]
    near_road_rows = [
        r for r in rows if str(r.get("road_complexity", "")).lower() in {"medium", "high"}
    ]
    south_rows = [r for r in rows if str(r.get("region_band", "")).lower() == "south"]
    north_rows = [r for r in rows if str(r.get("region_band", "")).lower() == "north"]
    sam_skip_modes = {"skipped_budget", "skipped_error"}
    sam_fallback_modes = {"fallback_non_sam", "skipped_budget", "skipped_error", "disabled"}
    sam_modes = [str(r.get("sam_runtime_mode") or "") for r in rows]
    return {
        "mode": mode_name,
        "count": len(rows),
        "metrics_mean": {
            "iou": float(statistics.fmean(r["iou"] for r in rows)) if rows else 0.0,
            "iou_geo": float(statistics.fmean(r["iou_geo"] for r in rows)) if rows else 0.0,
            "f1": float(statistics.fmean(r["f1"] for r in rows)) if rows else 0.0,
            "f1_geo": float(statistics.fmean(r["f1_geo"] for r in rows)) if rows else 0.0,
            "precision": float(statistics.fmean(r["precision"] for r in rows)) if rows else 0.0,
            "recall": float(statistics.fmean(r["recall"] for r in rows)) if rows else 0.0,
            "boundary_iou": float(statistics.fmean(r["boundary_iou"] for r in rows)) if rows else 0.0,
            "boundary_iou_geo": float(statistics.fmean(r["boundary_iou_geo"] for r in rows)) if rows else 0.0,
            "hausdorff_95_px": float(statistics.fmean(r["hausdorff_95_px"] for r in rows)) if rows else 0.0,
            "hd95_m": float(statistics.fmean(r["hd95_m"] for r in rows)) if rows else 0.0,
            "hd95_m_p90": (
                float(np.quantile(np.asarray([r["hd95_m"] for r in rows], dtype=float), 0.90))
                if rows
                else 0.0
            ),
            "centroid_shift_m": float(statistics.fmean(r["centroid_shift_m"] for r in rows)) if rows else 0.0,
            "area_ratio_pred_gt": float(statistics.fmean(r["area_ratio_pred_gt"] for r in rows)) if rows else 0.0,
            "area_ratio_geo_median": float(statistics.median(r["area_ratio_geo_median"] for r in rows)) if rows else 0.0,
            "missed_fields_count": float(statistics.fmean(r["missed_fields_count"] for r in rows)) if rows else 0.0,
            "oversegmented_fields_count": float(statistics.fmean(r["oversegmented_fields_count"] for r in rows)) if rows else 0.0,
            "missed_fields_rate": float(statistics.fmean(r["missed_fields_rate"] for r in rows)) if rows else 0.0,
            "oversegmented_fields_rate": (
                float(statistics.fmean(r["oversegmented_fields_rate"] for r in rows))
                if rows
                else 0.0
            ),
            "fallback_rate_tile": float(statistics.fmean(fallback)) if fallback else 0.0,
            "boundary_iou_near_water": (
                float(statistics.fmean(r["boundary_iou"] for r in near_water_rows))
                if near_water_rows
                else 0.0
            ),
            "boundary_iou_near_road": (
                float(statistics.fmean(r["boundary_iou"] for r in near_road_rows))
                if near_road_rows
                else 0.0
            ),
            "missed_fields_near_water": (
                float(statistics.fmean(r["missed_fields_count"] for r in near_water_rows))
                if near_water_rows
                else 0.0
            ),
            "drifted_to_road_fields": (
                float(statistics.fmean(r["oversegmented_fields_count"] for r in near_road_rows))
                if near_road_rows
                else 0.0
            ),
            "sam_skip_rate": (
                float(sum(1 for mode in sam_modes if mode in sam_skip_modes) / len(sam_modes))
                if sam_modes
                else 0.0
            ),
            "sam_fallback_rate": (
                float(sum(1 for mode in sam_modes if mode in sam_fallback_modes) / len(sam_modes))
                if sam_modes
                else 0.0
            ),
            "field_recall_south": (
                float(statistics.fmean(r["recall"] for r in south_rows))
                if south_rows
                else 0.0
            ),
            "field_recall_north": (
                float(statistics.fmean(r["recall"] for r in north_rows))
                if north_rows
                else 0.0
            ),
            "oversegmented_fields_rate_south": (
                float(statistics.fmean(r["oversegmented_fields_rate"] for r in south_rows))
                if south_rows
                else 0.0
            ),
            "missed_fields_rate_south": (
                float(statistics.fmean(r["missed_fields_rate"] for r in south_rows))
                if south_rows
                else 0.0
            ),
            "boundary_iou_north_median": (
                float(statistics.median(r["boundary_iou_geo"] for r in north_rows))
                if north_rows
                else 0.0
            ),
            "boundary_iou_south_median": (
                float(statistics.median(r["boundary_iou_geo"] for r in south_rows))
                if south_rows
                else 0.0
            ),
            "contour_shrink_ratio_north_median": (
                float(statistics.median(float(r.get("area_ratio_geo_median") or 0.0) for r in north_rows))
                if north_rows
                else 0.0
            ),
            "centroid_shift_m_north_p90": (
                float(np.quantile(np.asarray([r["centroid_shift_m"] for r in north_rows], dtype=float), 0.90))
                if north_rows
                else 0.0
            ),
            "centroid_shift_m_north": (
                float(statistics.fmean(r["centroid_shift_m"] for r in north_rows))
                if north_rows
                else 0.0
            ),
            "mean_components_per_gt_field_south": (
                float(statistics.fmean((float(r["oversegmented_fields_count"]) + 1.0) for r in south_rows))
                if south_rows
                else 0.0
            ),
        },
        "latency_s": _quantiles(latencies),
        "rows": rows,
    }


def _go_no_go(summary: dict[str, Any]) -> dict[str, Any]:
    rb = summary["rule_based"]
    ml = summary["ml_primary"]
    delta_iou = ml["metrics_mean"]["iou"] - rb["metrics_mean"]["iou"]
    delta_f1 = ml["metrics_mean"]["f1"] - rb["metrics_mean"]["f1"]
    p95_latency = ml["latency_s"]["p95"]
    fallback_rate = ml["metrics_mean"]["fallback_rate_tile"]

    passed = (
        fallback_rate <= 0.05
        and delta_iou >= 0.04
        and delta_f1 >= 0.05
        and p95_latency <= 120.0
    )
    return {
        "go": bool(passed),
        "delta_iou": float(delta_iou),
        "delta_f1": float(delta_f1),
        "ml_p95_latency_s": float(p95_latency),
        "ml_fallback_rate_tile": float(fallback_rate),
    }


def _compare_modes(summary: dict[str, Any], left_key: str, right_key: str) -> dict[str, Any]:
    left = summary[left_key]
    right = summary[right_key]
    return {
        "left_mode": left_key,
        "right_mode": right_key,
        "delta_iou": float(right["metrics_mean"]["iou"] - left["metrics_mean"]["iou"]),
        "delta_f1": float(right["metrics_mean"]["f1"] - left["metrics_mean"]["f1"]),
        "delta_boundary_iou": float(
            right["metrics_mean"]["boundary_iou"] - left["metrics_mean"]["boundary_iou"]
        ),
        "delta_hausdorff_95_px": float(
            right["metrics_mean"]["hausdorff_95_px"] - left["metrics_mean"]["hausdorff_95_px"]
        ),
        "delta_centroid_shift_m": float(
            right["metrics_mean"]["centroid_shift_m"] - left["metrics_mean"]["centroid_shift_m"]
        ),
        "delta_area_ratio_pred_gt": float(
            right["metrics_mean"]["area_ratio_pred_gt"] - left["metrics_mean"]["area_ratio_pred_gt"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run holdout A/B across runtime modes")
    parser.add_argument("--holdout", type=Path, required=True, help="Path to holdout AOI JSON")
    parser.add_argument("--api-base", type=str, default="http://localhost:8000/api/v1/fields")
    parser.add_argument("--poll-interval-s", type=float, default=5.0)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument(
        "--experiment",
        choices=("ml_primary_vs_rule_based", "sam_ab"),
        default="ml_primary_vs_rule_based",
        help="Compare rule-based vs ML-primary, or compare baseline vs SAM refinement.",
    )
    parser.add_argument("--output", type=Path, default=Path("backend/debug/runs/holdout_ab_summary.json"))
    args = parser.parse_args()

    holdout_items = _load_holdout(args.holdout)
    if not holdout_items:
        raise SystemExit("holdout list is empty")
    _validate_holdout_items(args.holdout, holdout_items)

    with httpx.Client() as client:
        if args.experiment == "sam_ab":
            baseline = _evaluate_mode(
                client,
                args.api_base,
                holdout_items,
                ml_primary=True,
                use_sam=False,
                poll_interval_s=args.poll_interval_s,
                timeout_s=args.timeout_s,
            )
            sam_enabled = _evaluate_mode(
                client,
                args.api_base,
                holdout_items,
                ml_primary=True,
                use_sam=True,
                poll_interval_s=args.poll_interval_s,
                timeout_s=args.timeout_s,
            )
            summary = {
                "experiment": args.experiment,
                "baseline": baseline,
                "sam_enabled": sam_enabled,
            }
            summary["comparison"] = _compare_modes(summary, "baseline", "sam_enabled")
        else:
            rule_based = _evaluate_mode(
                client,
                args.api_base,
                holdout_items,
                ml_primary=False,
                use_sam=False,
                poll_interval_s=args.poll_interval_s,
                timeout_s=args.timeout_s,
            )
            ml_primary = _evaluate_mode(
                client,
                args.api_base,
                holdout_items,
                ml_primary=True,
                use_sam=False,
                poll_interval_s=args.poll_interval_s,
                timeout_s=args.timeout_s,
            )
            summary = {
                "experiment": args.experiment,
                "rule_based": rule_based,
                "ml_primary": ml_primary,
            }
            summary["go_no_go"] = _go_no_go(summary)
            summary["comparison"] = _compare_modes(summary, "rule_based", "ml_primary")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary.get("comparison") or summary.get("go_no_go"), indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
