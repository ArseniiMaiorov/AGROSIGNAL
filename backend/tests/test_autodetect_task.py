from datetime import date, datetime
import time
import uuid
from types import SimpleNamespace

import numpy as np
import pytest
from rasterio.transform import from_bounds
from shapely.geometry import box

pytest.importorskip("celery")

from tasks.autodetect import (
    _aggregate_grid_cell_rows,
    _apply_branch_score_bias,
    _apply_runtime_config,
    _build_recovery_missed_mask,
    _build_temporal_coherence_mask,
    _build_grid_cells_for_tile,
    _build_time_windows,
    _candidate_overlap_score,
    _enrich_runtime_meta,
    _ensure_time_stack,
    _fuse_ml_primary_candidate,
    _format_utc_day_boundary,
    _resolve_regional_extent_threshold,
    _resolve_seed_mode,
    _resolve_auto_detect_version,
    _resolve_detect_pipeline_profile,
    _resolve_processing_profile,
    _safe_valid_fraction,
    _sanitize_json_floats,
    _sam_preflight_budget,
    _summarize_persisted_detection_candidates,
    resolve_region_band,
    resolve_region_boundary_profile,
    run_autodetect,
)
from core.config import Settings


def test_formats_date_start_boundary():
    result = _format_utc_day_boundary(date(2025, 5, 1), end_of_day=False)
    assert result == "2025-05-01T00:00:00Z"


def test_formats_datetime_end_boundary():
    value = datetime(2025, 8, 31, 0, 0, 0)
    result = _format_utc_day_boundary(value, end_of_day=True)
    assert result == "2025-08-31T23:59:59Z"


def test_formats_string_datetime_start_boundary():
    result = _format_utc_day_boundary("2025-05-01 00:00:00", end_of_day=False)
    assert result == "2025-05-01T00:00:00Z"


def test_ensure_time_stack_from_2d():
    arr = np.zeros((16, 32), dtype=np.float32)
    stacked = _ensure_time_stack(arr, name="B2")
    assert stacked.shape == (1, 16, 32)


def test_ensure_time_stack_keeps_3d():
    arr = np.zeros((3, 16, 32), dtype=np.float32)
    stacked = _ensure_time_stack(arr, name="B2")
    assert stacked.shape == (3, 16, 32)


def test_ensure_time_stack_rejects_invalid_dim():
    arr = np.zeros((1, 3, 16, 32), dtype=np.float32)
    with pytest.raises(ValueError, match="must be 2D or 3D"):
        _ensure_time_stack(arr, name="B2")


def test_safe_valid_fraction_handles_missing_scene_total():
    valid_count = np.array([[0, 2], [4, 8]], dtype=np.float32)

    fraction = _safe_valid_fraction(valid_count, None)

    assert fraction.dtype == np.float32
    assert fraction.shape == valid_count.shape
    assert fraction[0, 0] == pytest.approx(0.0)
    assert fraction[0, 1] == pytest.approx(1.0)
    assert fraction[1, 0] == pytest.approx(1.0)


def test_build_time_windows_count_and_bounds():
    windows = _build_time_windows("2025-05-01", "2025-08-31", target_slices=7)
    assert len(windows) == 7
    assert windows[0][0] == "2025-05-01T00:00:00Z"
    assert windows[-1][1] == "2025-08-31T23:59:59Z"


def test_build_time_windows_single_day():
    windows = _build_time_windows("2025-05-01", "2025-05-01", target_slices=7)
    assert windows == [("2025-05-01T00:00:00Z", "2025-05-01T23:59:59Z")]


def test_build_time_windows_respects_season():
    windows = _build_time_windows(
        "2025-03-01",
        "2025-11-30",
        target_slices=4,
        season_start_mmdd="04-15",
        season_end_mmdd="10-15",
    )
    assert windows[0][0] == "2025-04-15T00:00:00Z"
    assert windows[-1][1] == "2025-10-15T23:59:59Z"


def test_apply_runtime_config_updates_supported_settings_only():
    settings = Settings()
    updated = _apply_runtime_config(
        settings,
        {
            "AUTO_DETECT_VERSION": 2,
            "FRAMEWORK_SAM_ENABLED": False,
            "UNKNOWN_FLAG": "ignored",
        },
    )

    assert updated.AUTO_DETECT_VERSION == 2
    assert updated.FRAMEWORK_SAM_ENABLED is False
    assert not hasattr(updated, "UNKNOWN_FLAG")


def test_apply_runtime_config_accepts_compact_wctree_alias():
    settings = Settings()
    updated = _apply_runtime_config(
        settings,
        {
            "WCTREEHARD_EXCLUSION": False,
        },
    )

    assert updated.WC_TREE_HARD_EXCLUSION is False


def test_resolve_detect_pipeline_profile_fast_is_preview_only():
    settings = Settings()

    profile = _resolve_detect_pipeline_profile(settings, "fast")

    assert profile["name"] == "fast_preview"
    assert profile["preview_only"] is True
    assert profile["output_mode"] == "preview_agri_contours"
    assert profile["operational_eligible"] is False
    assert profile["field_source"] == "autodetect_preview"
    assert profile["enable_model_inference"] is False
    assert profile["enable_candidate_ranker"] is False
    assert profile["enable_selective_split"] is False
    assert profile["enable_snake_refine"] is False
    assert profile["enable_object_classifier"] is False
    assert profile["enable_post_merge_smooth"] is False
    assert profile["enable_active_learning"] is False
    assert profile["enable_sam"] is False
    assert profile["max_candidates_per_tile"] == 256
    assert profile["post_merge_max_components"] == 512


def test_resolve_detect_pipeline_profile_standard_and_quality_are_operational():
    settings = Settings()

    standard = _resolve_detect_pipeline_profile(settings, "standard")
    quality = _resolve_detect_pipeline_profile(settings, "quality")

    assert standard["name"] == "standard_balanced"
    assert standard["preview_only"] is False
    assert standard["output_mode"] == "field_boundaries"
    assert standard["operational_eligible"] is True
    assert standard["enable_model_inference"] is True
    assert standard["enable_candidate_ranker"] is True
    assert standard["enable_snake_refine"] is True
    assert standard["enable_object_classifier"] is True
    assert standard["enable_sam"] is False
    assert standard["max_candidates_per_tile"] == 768

    assert quality["name"] == "quality_full"
    assert quality["preview_only"] is False
    assert quality["output_mode"] == "field_boundaries_hifi"
    assert quality["operational_eligible"] is True
    assert quality["enable_model_inference"] is True
    assert quality["enable_candidate_ranker"] is True
    assert quality["enable_snake_refine"] is True
    assert quality["enable_object_classifier"] is True
    assert quality["enable_sam"] is True
    assert quality["max_candidates_per_tile"] == 1200


def test_resolve_auto_detect_version_clamps_to_supported_range():
    settings = Settings(AUTO_DETECT_VERSION=99)
    assert _resolve_auto_detect_version(settings) == 4


def test_resolve_processing_profile_boundary_recovery_enables_second_pass():
    settings = Settings()

    profile = _resolve_processing_profile("boundary_recovery", settings)

    assert profile["name"] == "boundary_recovery"
    assert profile["enable_second_pass"] is True
    assert profile["allow_degraded_output"] is False
    assert profile["prefer_boundary_branch"] is True
    assert profile["force_boundary_union"] is True
    assert profile["config_overrides"]["SKIP_WEAK_EDGE_TILES"] is False
    assert profile["config_overrides"]["WATERSHED_MIN_DISTANCE"] >= settings.WATERSHED_MIN_DISTANCE


def test_resolve_processing_profile_degraded_output_relaxes_thresholds():
    settings = Settings()

    profile = _resolve_processing_profile("degraded_output", settings)

    assert profile["name"] == "degraded_output"
    assert profile["enable_second_pass"] is True
    assert profile["allow_degraded_output"] is True
    assert profile["prefer_boundary_branch"] is True
    assert profile["config_overrides"]["CANDIDATE_MIN_SCORE"] < settings.CANDIDATE_MIN_SCORE


def test_apply_branch_score_bias_prefers_boundary_and_recovery():
    candidates = [
        SimpleNamespace(branch="boundary_first", score=0.40, features={}),
        SimpleNamespace(branch="recovery_second_pass", score=0.35, features={}),
        SimpleNamespace(branch="crop_region", score=0.55, features={}),
    ]
    profile = {
        "boundary_branch_score_boost": 0.08,
        "recovery_second_pass_score_boost": 0.05,
    }

    _apply_branch_score_bias(candidates, profile)

    assert candidates[0].score == pytest.approx(0.48)
    assert candidates[1].score == pytest.approx(0.40)
    assert candidates[2].score == pytest.approx(0.55)
    assert candidates[0].features["branch_score_boost"] == pytest.approx(0.08)
    assert candidates[1].features["branch_score_boost"] == pytest.approx(0.05)


def test_build_recovery_missed_mask_adds_edge_seed_and_boundary_halo():
    candidate_mask = np.zeros((6, 6), dtype=bool)
    candidate_mask[2:4, 2:4] = True
    boundary_field_mask = np.zeros((6, 6), dtype=bool)
    boundary_field_mask[2, 1:5] = True
    crop_soft_mask = np.zeros((6, 6), dtype=bool)
    crop_soft_mask[1:5, 2] = True
    field_candidate = np.zeros((6, 6), dtype=bool)
    field_candidate[1:5, 1:5] = True
    edge_source = np.zeros((6, 6), dtype=np.float32)
    edge_source[1, 2:5] = 0.8
    edge_source[4, 1:4] = 0.75

    missed_mask, diag = _build_recovery_missed_mask(
        candidate_masks_payload={
            "field_candidate": field_candidate,
            "crop_soft_mask": crop_soft_mask,
            "boundary_field_mask": boundary_field_mask,
        },
        candidate_mask=candidate_mask,
        processing_profile={
            "prefer_boundary_branch": True,
            "recovery_dilation_px": 1,
            "boundary_halo_px": 1,
            "recovery_edge_seed_threshold": 0.4,
            "recovery_edge_seed_percentile": 60.0,
        },
        edge_source=edge_source,
    )

    assert np.count_nonzero(missed_mask) > 0
    assert diag["edge_seed_pixels"] > 0
    assert diag["guide_edge_halo_pixels"] > 0
    assert diag["boundary_halo_pixels"] > 0
    assert diag["edge_seed_threshold"] >= 0.4


def test_build_temporal_coherence_mask_uses_default_thresholds_for_normal_profile():
    growth_amplitude = np.array([[0.19, 0.21]], dtype=np.float32)
    has_growth_peak = np.zeros_like(growth_amplitude, dtype=np.float32)
    ndvi_entropy = np.array([[2.4, 2.6]], dtype=np.float32)

    tc_mask, diag = _build_temporal_coherence_mask(
        growth_amplitude=growth_amplitude,
        has_growth_peak=has_growth_peak,
        ndvi_entropy=ndvi_entropy,
        candidate_masks_payload={},
        processing_profile={"name": "normal"},
        cfg=Settings(),
    )

    assert diag["relaxed"] is False
    assert diag["growth_amplitude_min"] == pytest.approx(0.20)
    assert diag["entropy_max"] == pytest.approx(2.5)
    assert tc_mask.tolist() == [[False, False]]


def test_build_temporal_coherence_mask_relaxes_thresholds_and_keeps_boundary_seed():
    growth_amplitude = np.array([[0.15, 0.13]], dtype=np.float32)
    has_growth_peak = np.zeros_like(growth_amplitude, dtype=np.float32)
    ndvi_entropy = np.array([[3.0, 3.2]], dtype=np.float32)
    boundary_field_mask = np.array([[False, True]], dtype=bool)

    tc_mask, diag = _build_temporal_coherence_mask(
        growth_amplitude=growth_amplitude,
        has_growth_peak=has_growth_peak,
        ndvi_entropy=ndvi_entropy,
        candidate_masks_payload={"boundary_field_mask": boundary_field_mask},
        processing_profile={"name": "boundary_recovery"},
        cfg=Settings(
            RECOVERY_TEMPORAL_COHERENCE_RELAXED=True,
            RECOVERY_TEMPORAL_GROWTH_AMPLITUDE_MIN=0.14,
            RECOVERY_TEMPORAL_ENTROPY_MAX=3.1,
        ),
    )

    assert diag["relaxed"] is True
    assert diag["growth_amplitude_min"] == pytest.approx(0.14)
    assert diag["entropy_max"] == pytest.approx(3.1)
    assert diag["boundary_keep_pixels"] == 1
    assert tc_mask.tolist() == [[True, True]]


def test_resolve_seed_mode_maps_legacy_values():
    assert _resolve_seed_mode("auto") == "auto"
    assert _resolve_seed_mode("grid") == "grid"
    assert _resolve_seed_mode("custom") == "custom"
    assert _resolve_seed_mode("edges") == "grid"
    assert _resolve_seed_mode("distance") == "auto"
    assert _resolve_seed_mode("unknown") == "auto"


def test_enrich_runtime_meta_populates_reliability_fields():
    runtime_meta = {
        "tiles": [
            {
                "ml_primary_used": True,
                "fallback_rate_tile": 0.0,
                "model_backend": "onnx",
                "n_valid_scenes": 6,
                "edge_signal_p90": 0.42,
                "ml_quality_score": 0.77,
                "fusion_profile": "balanced:balanced_ml_seed_union",
                "selected_scene_signature": "abc1",
            },
            {
                "ml_primary_used": False,
                "fallback_rate_tile": 1.0,
                "model_backend": "onnx",
                "n_valid_scenes": 4,
                "edge_signal_p90": 0.25,
                "ml_quality_score": 0.11,
                "fusion_profile": "balanced:unet_fallback",
                "selected_scene_signature": "abc2",
            },
        ],
        "weak_label_source": None,
    }
    out = _enrich_runtime_meta(
        runtime_meta,
        latency_breakdown={"tiling_s": 1.2, "tiles_total_s": 10.0},
        t_start=time.time() - 12.0,
    )
    assert out["ml_primary_used"] is True
    assert out["fallback_rate_tile"] == pytest.approx(0.5)
    assert out["quality_gate_failed"] is False
    assert out["weak_label_source"] == "unknown"
    assert out["model_backend"] == "onnx"
    assert out["n_valid_scenes"] == 5
    assert out["edge_signal_p90"] == pytest.approx(0.335, abs=1e-4)
    assert out["ml_quality_score"] == pytest.approx(0.44, abs=1e-4)
    assert out["fusion_profile"] == "mixed"
    assert isinstance(out["selected_scene_signature"], str)
    assert len(out["selected_scene_signature"]) == 16
    assert isinstance(out["latency_breakdown_s"], dict)
    assert out["latency_breakdown_s"]["tiling_s"] == pytest.approx(1.2)


def test_sanitize_json_floats_replaces_non_finite_values():
    data = {
        "ok": 1.0,
        "nan": float("nan"),
        "pos_inf": float("inf"),
        "neg_inf": float("-inf"),
        "nested": [np.float32(2.5), np.float32(np.nan)],
    }

    out = _sanitize_json_floats(data)

    assert out["ok"] == pytest.approx(1.0)
    assert out["nan"] is None
    assert out["pos_inf"] is None
    assert out["neg_inf"] is None
    assert out["nested"] == [pytest.approx(2.5), None]


def test_candidate_overlap_score_prefers_candidate_coverage():
    candidate = box(0, 0, 1, 1)
    final_field = box(0, 0, 2, 2)

    score = _candidate_overlap_score(candidate, final_field)

    assert score == pytest.approx(1.0)


def test_summarize_persisted_detection_candidates_aggregates_final_state():
    rows = [
        SimpleNamespace(branch="boundary_first", kept=True, reject_reason=None),
        SimpleNamespace(branch="boundary_first", kept=False, reject_reason="dropped_after_topology_cleanup"),
        SimpleNamespace(branch="crop_region", kept=False, reject_reason="below_min_score"),
    ]

    summary = _summarize_persisted_detection_candidates(rows)

    assert summary["candidates_total"] == 3
    assert summary["candidates_kept"] == 1
    assert summary["candidate_branch_counts"]["boundary_first"] == {"total": 2, "kept": 1}
    assert summary["candidate_branch_counts"]["crop_region"] == {"total": 1, "kept": 0}
    assert summary["candidate_reject_summary"]["dropped_after_topology_cleanup"] == 1
    assert summary["candidate_reject_summary"]["below_min_score"] == 1


def test_run_autodetect_marks_failed_on_config_validation_error(monkeypatch):
    run_id = str(uuid.uuid4())
    captured: dict[str, object] = {}

    def _raise_invalid_settings():
        raise ValueError("ALLOW_SYNTHETIC_DATA invalid")

    def _capture_mark_failed(*, run_id, error, failure_stage):
        captured["run_id"] = run_id
        captured["error"] = error
        captured["failure_stage"] = failure_stage

    monkeypatch.setattr("tasks.autodetect.get_settings", _raise_invalid_settings)
    monkeypatch.setattr("tasks.autodetect._mark_run_failed_best_effort", _capture_mark_failed)

    result = run_autodetect.run(run_id)

    assert result["status"] == "error"
    assert result["failure_stage"] == "config_validation"
    assert str(captured["run_id"]) == run_id
    assert captured["failure_stage"] == "config_validation"
    assert "ALLOW_SYNTHETIC_DATA invalid" in str(captured["error"])


def test_sam_preflight_budget_rejects_large_candidate_coverage():
    composite = np.zeros((64, 64, 3), dtype=np.uint8)
    candidate_mask = np.ones((64, 64), dtype=bool)
    cfg = Settings(
        SAM_MAX_TILE_PIXELS=10_000,
        SAM_MAX_COMPONENTS=128,
        SAM_MAX_CANDIDATE_COVERAGE_PCT=5.0,
        SAM_MAX_EST_MEMORY_MB=10_000,
    )

    allowed, reason, estimated_mb = _sam_preflight_budget(
        composite_uint8=composite,
        candidate_mask=candidate_mask,
        candidate_coverage_pct=100.0,
        component_count=4,
        cfg=cfg,
    )

    assert allowed is False
    assert reason == "candidate_coverage"
    assert estimated_mb > 0.0


def test_resolve_region_band_uses_latitude_thresholds():
    cfg = Settings(REGION_LAT_SOUTH_MAX=48.0, REGION_LAT_NORTH_MIN=57.0)

    assert resolve_region_band(45.0, cfg) == "south"
    assert resolve_region_band(54.0, cfg) == "central"
    assert resolve_region_band(58.0, cfg) == "north"


def test_resolve_region_boundary_profile_auto_only():
    cfg = Settings()

    assert resolve_region_boundary_profile(45.0, cfg) == "south_recall"
    assert resolve_region_boundary_profile(54.0, cfg) == "balanced"
    assert resolve_region_boundary_profile(58.0, cfg) == "north_boundary"


def test_resolve_regional_extent_threshold_uses_profile_specific_values():
    cfg = Settings(
        ML_EXTENT_BIN_THRESHOLD=0.42,
        SOUTH_ML_EXTENT_BIN_THRESHOLD=0.34,
        NORTH_ML_EXTENT_BIN_THRESHOLD=0.40,
    )

    assert _resolve_regional_extent_threshold(cfg, "south_recall") == pytest.approx(0.34)
    assert _resolve_regional_extent_threshold(cfg, "north_boundary") == pytest.approx(0.40)
    assert _resolve_regional_extent_threshold(cfg, "balanced") == pytest.approx(0.42)


def test_fuse_ml_primary_candidate_keeps_pre_ml_support_for_north():
    ml_seed = np.zeros((10, 10), dtype=bool)
    ml_seed[4:6, 4:6] = True
    pre_ml = np.zeros((10, 10), dtype=bool)
    pre_ml[3:7, 3:7] = True

    fused, actions = _fuse_ml_primary_candidate(ml_seed, pre_ml, "north_boundary")

    assert np.count_nonzero(fused) > np.count_nonzero(ml_seed)
    assert "north_ml_seed_union" in actions
    assert fused[3, 3]


def test_fuse_ml_primary_candidate_unions_for_south():
    ml_seed = np.zeros((10, 10), dtype=bool)
    ml_seed[4:6, 4:6] = True
    pre_ml = np.zeros((10, 10), dtype=bool)
    pre_ml[4:6, 7:9] = True

    fused, actions = _fuse_ml_primary_candidate(ml_seed, pre_ml, "south_recall")

    assert np.count_nonzero(fused) == (
        np.count_nonzero(ml_seed) + np.count_nonzero(pre_ml)
    )
    assert "south_ml_seed_union" in actions


def test_fuse_ml_primary_candidate_unions_for_balanced():
    ml_seed = np.zeros((12, 12), dtype=bool)
    ml_seed[5:7, 5:7] = True
    pre_ml = np.zeros((12, 12), dtype=bool)
    pre_ml[4:8, 4:8] = True

    fused, actions = _fuse_ml_primary_candidate(ml_seed, pre_ml, "balanced")

    assert np.count_nonzero(fused) > np.count_nonzero(ml_seed)
    assert "balanced_ml_seed_union" in actions


def test_build_grid_cells_for_tile_snaps_overlap_tiles_to_global_grid():
    labels = np.ones((128, 128), dtype=np.int32)
    raster = np.ones((128, 128), dtype=np.float32)
    weather = {
        "precipitation_mm": 0.0,
        "wind_speed_m_s": 5.0,
        "u_wind_10m": 2.0,
        "v_wind_10m": 1.0,
        "wind_direction_deg": 45.0,
        "vpd_mean": 0.7,
        "soil_moist": 0.22,
    }
    left_tile = {
        "transform": from_bounds(0.0, 0.0, 1280.0, 1280.0, 128, 128),
        "crs": "EPSG:32640",
        "bbox_utm": (0.0, 0.0, 1280.0, 1280.0),
    }
    right_tile = {
        "transform": from_bounds(780.0, 0.0, 2060.0, 1280.0, 128, 128),
        "crs": "EPSG:32640",
        "bbox_utm": (780.0, 0.0, 2060.0, 1280.0),
    }

    left_rows = _build_grid_cells_for_tile(
        tile=left_tile,
        labels_clean=labels,
        gdf=SimpleNamespace(empty=False),
        ndvi_mean=raster,
        ndwi_mean=raster,
        ndmi_mean=raster,
        bsi_mean=raster,
        weather_snapshot=weather,
        zoom_level=2,
        cell_px=64,
        grid_origin_utm=(0.0, 0.0),
    )
    right_rows = _build_grid_cells_for_tile(
        tile=right_tile,
        labels_clean=labels,
        gdf=SimpleNamespace(empty=False),
        ndvi_mean=raster * 2.0,
        ndwi_mean=raster * 2.0,
        ndmi_mean=raster * 2.0,
        bsi_mean=raster * 2.0,
        weather_snapshot=weather,
        zoom_level=2,
        cell_px=64,
        grid_origin_utm=(0.0, 0.0),
    )

    left_keys = {(row["zoom_level"], row["row"], row["col"]) for row in left_rows}
    right_keys = {(row["zoom_level"], row["row"], row["col"]) for row in right_rows}
    shared_keys = left_keys & right_keys

    assert shared_keys, "overlap tiles should snap to shared global grid cells"

    sample_key = sorted(shared_keys)[0]
    left_cell = next(row for row in left_rows if (row["zoom_level"], row["row"], row["col"]) == sample_key)
    right_cell = next(row for row in right_rows if (row["zoom_level"], row["row"], row["col"]) == sample_key)
    assert left_cell["geometry"].equals_exact(right_cell["geometry"], tolerance=1e-9)


def test_aggregate_grid_cell_rows_deduplicates_overlap_cells():
    geom = box(30.0, 59.0, 30.01, 59.01)
    rows = [
        {
            "geometry": geom,
            "zoom_level": 2,
            "row": 10,
            "col": 15,
            "field_coverage": 0.4,
            "ndvi_mean": 0.2,
            "ndwi_mean": 0.1,
            "ndmi_mean": 0.15,
            "bsi_mean": 0.05,
            "precipitation_mm": 1.0,
            "wind_speed_m_s": 2.0,
            "u_wind_10m": 1.0,
            "v_wind_10m": 0.0,
            "wind_direction_deg": 90.0,
            "gdd_sum": None,
            "vpd_mean": 0.7,
            "soil_moist": 0.2,
        },
        {
            "geometry": geom,
            "zoom_level": 2,
            "row": 10,
            "col": 15,
            "field_coverage": 0.6,
            "ndvi_mean": 0.4,
            "ndwi_mean": 0.3,
            "ndmi_mean": 0.25,
            "bsi_mean": 0.15,
            "precipitation_mm": 1.0,
            "wind_speed_m_s": 4.0,
            "u_wind_10m": 2.0,
            "v_wind_10m": 1.0,
            "wind_direction_deg": 100.0,
            "gdd_sum": None,
            "vpd_mean": 0.9,
            "soil_moist": 0.3,
        },
    ]

    aggregated = _aggregate_grid_cell_rows(rows)

    assert len(aggregated) == 1
    assert aggregated[0]["zoom_level"] == 2
    assert aggregated[0]["row"] == 10
    assert aggregated[0]["col"] == 15
    assert aggregated[0]["ndvi_mean"] == pytest.approx(0.3)
    assert aggregated[0]["field_coverage"] == pytest.approx(0.5)
    assert aggregated[0]["wind_speed_m_s"] == pytest.approx(3.0)
