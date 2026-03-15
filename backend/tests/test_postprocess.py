"""Tests for crop-mask postprocessing."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import label as nd_label

from processing.fields import postprocess as postprocess_module
from processing.fields.phenoclassify import CROP, FOREST, GRASS
from processing.fields.field_grow import seeded_grow_into_field
from processing.fields.postprocess import run_postprocess, seeded_grow_into_grass
from processing.fields.region_merge import merge_crop_regions


class DummyCfg:
    EDGE_CANNY_SIGMA = 1.2

    POST_ROAD_MAX_NDVI = -1.0
    POST_ROAD_NIR_MAX = 0.15
    POST_ROAD_NDBI_MIN = -0.10
    POST_ROAD_HOUGH_THRESHOLD = 5
    POST_ROAD_HOUGH_MIN_LEN = 5
    POST_ROAD_HOUGH_MAX_GAP = 2
    POST_ROAD_BUFFER_PX = 1

    POST_FOREST_NDVI_MIN = 0.65
    POST_FOREST_MIN_AREA_PX = 4

    POST_MORPH_CLOSE_RADIUS = 1

    POST_MERGE_BUFFER_PX = 3
    POST_MERGE_NDVI_DIFF_MAX = 0.12
    POST_MERGE_OVERLAP_MIN = 0.25
    POST_MERGE_BARRIER_RATIO = 0.10
    POST_MERGE_MAX_COMPONENTS = 1500
    BOUNDARY_FILL_MAX_REGIONS = 4000

    PHENO_NDVI_CROP_MIN = 0.25
    PHENO_NDVI_CROP_MAX = 0.62
    POST_GROW_NDVI_RELAX = 0.07
    POST_GROW_BOUNDARY_STOP_THRESHOLD = 0.38
    POST_GROW_MAX_ITERS = 8
    POST_GAP_EDGE_THRESHOLD = 0.15
    ROAD_SNAP_REJECT_ENABLED = True
    ROAD_SNAP_REJECT_BUFFER_PX = 2
    ROAD_SNAP_REJECT_MAX_OVERLAP_RATIO = 0.08
    BOUNDARY_OUTER_EXPAND_WATER_AWARE = True
    BOUNDARY_OUTER_EXPAND_MAX_PX = 4
    BOUNDARY_OUTER_EXPAND_NEAR_WATER_MAX_PX = 2
    HYDRO_FIELD_NEAR_WATER_RESCUE_ENABLED = True
    HYDRO_FIELD_NEAR_WATER_MIN_NDVI_MAX = 0.40
    HYDRO_FIELD_NEAR_WATER_MIN_NDVI_STD = 0.12

    POST_CONVEX_MIN_HA = 0.05
    POST_CONVEX_RATIO_MAX = 1.35

    POST_MIN_FIELD_AREA_HA = 0.05
    POST_PX_AREA_M2 = 100
    WATERSHED_COMPACTNESS = 0.001
    WATERSHED_GRADIENT_EDGE_W = 0.5
    PHENO_FIELD_NDVI_STD_MIN = 0.15
    PHENO_GRASS_NDVI_STD_MAX = 0.12
    PHENO_FOREST_NDVI_STD_MAX = 0.05
    PHENO_FIELD_MAX_NDVI_MIN = 0.45
    PHENO_FIELD_MAX_NDVI_MAX = 0.62
    PHENO_GRASS_MAX_NDVI_MAX = 0.55
    PHENO_FOREST_MAX_NDVI_MIN = 0.65
    USE_WEAK_WORLDCOVER_BARRIER = True
    WC_TREE_HARD_EXCLUSION = True
    SOUTH_POST_GROW_NDVI_RELAX = 0.15
    SOUTH_POST_GROW_MAX_ITERS = 6
    SOUTH_CLEAN_ROLLBACK_MIN_AREA_RATIO = 0.75
    SOUTH_COMPONENT_BRIDGE_ENABLED = True
    SOUTH_COMPONENT_BRIDGE_MAX_GAP_PX = 3
    SOUTH_COMPONENT_BRIDGE_MAX_NDVI_DIFF = 0.08
    SOUTH_COMPONENT_BRIDGE_MAX_BOUNDARY_PROB = 0.45
    SOUTH_MERGE_MIN_OVERLAP_RATIO = 0.03
    SOUTH_POST_MIN_FIELD_AREA_HA = 0.12
    SOUTH_SKIP_WATERSHED_LARGE_COMPONENTS = True
    EDGE_WEAK_THRESHOLD = 0.1
    NORTH_CLEAN_ROLLBACK_MIN_AREA_RATIO = 0.92


def _count_components(mask: np.ndarray) -> int:
    _, count = nd_label(mask.astype(bool))
    return int(count)


def test_region_merge_connects_nearby_fragments():
    candidate_mask = np.zeros((20, 20), dtype=bool)
    candidate_mask[4:8, 4:8] = True
    candidate_mask[4:8, 10:14] = True

    ndvi = np.full((20, 20), 0.5, dtype=np.float32)
    barrier_mask = np.zeros_like(candidate_mask, dtype=bool)
    merged = merge_crop_regions(candidate_mask, ndvi, barrier_mask, DummyCfg())

    assert _count_components(candidate_mask) == 2
    assert _count_components(merged) == 1
    assert merged[5, 9]


def test_region_merge_respects_barrier_mask():
    candidate_mask = np.zeros((24, 24), dtype=bool)
    candidate_mask[8:14, 5:9] = True
    candidate_mask[8:14, 11:15] = True

    ndvi = np.full((24, 24), 0.5, dtype=np.float32)
    barrier_mask = np.zeros_like(candidate_mask, dtype=bool)
    barrier_mask[8:14, 9:11] = True

    merged = merge_crop_regions(candidate_mask, ndvi, barrier_mask, DummyCfg())

    assert _count_components(candidate_mask) == 2
    assert _count_components(merged) == 2
    assert not merged[10, 9]


def test_region_merge_emits_progress_callback():
    candidate_mask = np.zeros((20, 20), dtype=bool)
    candidate_mask[4:8, 4:8] = True
    candidate_mask[4:8, 10:14] = True

    ndvi = np.full((20, 20), 0.5, dtype=np.float32)
    barrier_mask = np.zeros_like(candidate_mask, dtype=bool)
    progress_events: list[tuple[int, int]] = []

    merged = merge_crop_regions(
        candidate_mask,
        ndvi,
        barrier_mask,
        DummyCfg(),
        progress_callback=lambda done, total: progress_events.append((int(done), int(total))),
    )

    assert merged[5, 9]
    assert progress_events
    assert progress_events[0][0] >= 1
    assert progress_events[-1][0] == progress_events[-1][1]


def test_seeded_grow_expands_into_relaxed_grass_neighbors():
    candidate_mask = np.zeros((16, 16), dtype=bool)
    candidate_mask[6:10, 6:10] = True

    classes = np.zeros((16, 16), dtype=np.uint8)
    classes[6:10, 6:10] = CROP
    classes[5:11, 5:11] = GRASS
    classes[6:10, 6:10] = CROP

    ndvi = np.full((16, 16), 0.1, dtype=np.float32)
    ndvi[5:11, 5:11] = 0.2
    ndvi[6:10, 6:10] = 0.5
    barrier_mask = np.zeros_like(candidate_mask, dtype=bool)

    grown, added = seeded_grow_into_grass(candidate_mask, classes, ndvi, barrier_mask, DummyCfg())

    assert added > 0
    assert grown[5, 5]
    assert grown[10, 10]


def test_seeded_grow_into_field_respects_ndvi_std_threshold():
    field_mask = np.zeros((16, 16), dtype=bool)
    field_mask[6:10, 6:10] = True

    ndvi = np.zeros((16, 16), dtype=np.float32)
    ndvi[5:11, 5:11] = 0.46

    ndvi_std = np.zeros((16, 16), dtype=np.float32)
    ndvi_std[5:11, 5:11] = 0.16
    ndvi_std[5, 5] = 0.08  # grass-like, should stay excluded

    barrier_mask = np.zeros_like(field_mask, dtype=bool)

    grown, added = seeded_grow_into_field(
        field_mask,
        ndvi,
        ndvi_std,
        barrier_mask,
        DummyCfg(),
    )

    assert added > 0
    assert grown[5, 6]
    assert not grown[5, 5]


def test_seeded_grow_into_field_relaxes_ndvi_for_south_profile():
    field_mask = np.zeros((16, 16), dtype=bool)
    field_mask[6:10, 6:10] = True

    ndvi = np.zeros((16, 16), dtype=np.float32)
    ndvi[5:11, 5:11] = 0.32

    ndvi_std = np.zeros((16, 16), dtype=np.float32)
    ndvi_std[5:11, 5:11] = 0.16
    barrier_mask = np.zeros_like(field_mask, dtype=bool)

    grown_default, _ = seeded_grow_into_field(field_mask, ndvi, ndvi_std, barrier_mask, DummyCfg())
    grown_south, added_south = seeded_grow_into_field(
        field_mask,
        ndvi,
        ndvi_std,
        barrier_mask,
        DummyCfg(),
        region_profile="south_recall",
    )

    assert not grown_default[5, 6]
    assert added_south > 0
    assert grown_south[5, 6]


def test_seeded_grow_into_field_stops_on_strong_boundary_probability():
    field_mask = np.zeros((16, 16), dtype=bool)
    field_mask[6:10, 6:10] = True

    ndvi = np.zeros((16, 16), dtype=np.float32)
    ndvi[5:11, 5:11] = 0.46
    ndvi_std = np.zeros((16, 16), dtype=np.float32)
    ndvi_std[5:11, 5:11] = 0.16
    barrier_mask = np.zeros_like(field_mask, dtype=bool)
    boundary_prob = np.zeros((16, 16), dtype=np.float32)
    boundary_prob[5, 6] = 0.72

    grown, _ = seeded_grow_into_field(
        field_mask,
        ndvi,
        ndvi_std,
        barrier_mask,
        DummyCfg(),
        boundary_prob=boundary_prob,
    )

    assert not grown[5, 6]


def test_run_postprocess_removes_forest_fills_holes_and_drops_small_objects():
    candidate_mask = np.zeros((24, 24), dtype=bool)
    candidate_mask[8:18, 8:18] = True
    candidate_mask[12, 12] = False
    candidate_mask[1:5, 1:5] = True
    candidate_mask[22, 22] = True

    water_mask = np.zeros_like(candidate_mask, dtype=bool)
    classes = np.zeros_like(candidate_mask, dtype=np.uint8)
    classes[1:5, 1:5] = FOREST

    ndvi = np.full(candidate_mask.shape, 0.5, dtype=np.float32)
    ndvi[1:5, 1:5] = 0.8
    ndwi = np.zeros(candidate_mask.shape, dtype=np.float32)
    edge_composite = np.zeros(candidate_mask.shape, dtype=np.float32)
    nir = np.full(candidate_mask.shape, 0.2, dtype=np.float32)
    swir = np.full(candidate_mask.shape, 0.2, dtype=np.float32)

    processed = run_postprocess(
        candidate_mask=candidate_mask,
        water_mask=water_mask,
        classes=classes,
        ndvi=ndvi,
        ndwi=ndwi,
        cfg=DummyCfg(),
        nir=nir,
        swir=swir,
        edge_composite=edge_composite,
    )

    assert processed[12, 12]
    assert processed[8:18, 8:18].all()
    assert not processed[1:5, 1:5].any()
    assert not processed[22, 22]


def test_run_postprocess_preserves_tree_island_hole_from_worldcover():
    candidate_mask = np.zeros((20, 20), dtype=bool)
    candidate_mask[4:16, 4:16] = True

    water_mask = np.zeros_like(candidate_mask, dtype=bool)
    classes = np.zeros_like(candidate_mask, dtype=np.uint8)
    classes[4:16, 4:16] = CROP
    ndvi = np.full(candidate_mask.shape, 0.5, dtype=np.float32)
    ndwi = np.zeros(candidate_mask.shape, dtype=np.float32)
    edge_composite = np.zeros(candidate_mask.shape, dtype=np.float32)
    nir = np.full(candidate_mask.shape, 0.2, dtype=np.float32)
    swir = np.full(candidate_mask.shape, 0.2, dtype=np.float32)
    worldcover_mask = np.full(candidate_mask.shape, 40, dtype=np.uint8)
    worldcover_mask[8:12, 8:12] = 10

    processed = run_postprocess(
        candidate_mask=candidate_mask,
        water_mask=water_mask,
        classes=classes,
        ndvi=ndvi,
        ndwi=ndwi,
        cfg=DummyCfg(),
        nir=nir,
        swir=swir,
        edge_composite=edge_composite,
        worldcover_mask=worldcover_mask,
    )

    assert processed[6, 6]
    assert not processed[9, 9]


def test_run_postprocess_can_return_step_debug_masks():
    candidate_mask = np.zeros((12, 12), dtype=bool)
    candidate_mask[3:9, 3:9] = True

    water_mask = np.zeros_like(candidate_mask, dtype=bool)
    classes = np.zeros_like(candidate_mask, dtype=np.uint8)
    classes[3:9, 3:9] = CROP
    ndvi = np.full(candidate_mask.shape, 0.5, dtype=np.float32)
    ndwi = np.zeros(candidate_mask.shape, dtype=np.float32)

    processed, debug = run_postprocess(
        candidate_mask=candidate_mask,
        water_mask=water_mask,
        classes=classes,
        ndvi=ndvi,
        ndwi=ndwi,
        cfg=DummyCfg(),
        return_debug_steps=True,
    )

    assert processed.shape == candidate_mask.shape
    assert "masks" in debug
    assert "stats" in debug
    assert "step_00_candidate_initial" in debug["masks"]
    assert "step_09_after_merge" in debug["masks"]
    assert "step_11_after_small_remove" in debug["masks"]
    assert "summary" in debug["stats"]
    assert debug["stats"]["step_00_candidate_initial"]["components"] == 1


def test_run_postprocess_can_return_stats_without_debug_masks():
    candidate_mask = np.zeros((12, 12), dtype=bool)
    candidate_mask[2:10, 2:10] = True

    water_mask = np.zeros_like(candidate_mask, dtype=bool)
    classes = np.zeros_like(candidate_mask, dtype=np.uint8)
    ndvi = np.full(candidate_mask.shape, 0.5, dtype=np.float32)
    ndwi = np.zeros(candidate_mask.shape, dtype=np.float32)

    processed, debug = run_postprocess(
        candidate_mask=candidate_mask,
        water_mask=water_mask,
        classes=classes,
        ndvi=ndvi,
        ndwi=ndwi,
        cfg=DummyCfg(),
        return_debug_stats=True,
    )

    assert processed.shape == candidate_mask.shape
    assert "stats" in debug
    assert "summary" in debug["stats"]
    assert isinstance(debug.get("masks"), dict)
    assert not debug["masks"]
    assert "road_barrier_retry_used" in debug["stats"]["summary"]
    assert "road_snap_reject_used" in debug["stats"]["summary"]
    assert "open_water_pixels" in debug["stats"]["summary"]
    assert "seasonal_wet_pixels" in debug["stats"]["summary"]
    assert "water_edge_overlap_ratio" in debug["stats"]["summary"]


def test_run_postprocess_emits_progress_checkpoints():
    candidate_mask = np.zeros((12, 12), dtype=bool)
    candidate_mask[2:10, 2:10] = True

    water_mask = np.zeros_like(candidate_mask, dtype=bool)
    classes = np.full(candidate_mask.shape, CROP, dtype=np.uint8)
    ndvi = np.full(candidate_mask.shape, 0.5, dtype=np.float32)
    ndwi = np.zeros(candidate_mask.shape, dtype=np.float32)
    ndvi_std = np.full(candidate_mask.shape, 0.18, dtype=np.float32)
    edge = np.zeros(candidate_mask.shape, dtype=np.float32)
    checkpoints: list[str] = []

    run_postprocess(
        candidate_mask=candidate_mask,
        water_mask=water_mask,
        classes=classes,
        ndvi=ndvi,
        ndwi=ndwi,
        ndvi_std=ndvi_std,
        edge_composite=edge,
        cfg=DummyCfg(),
        progress_callback=checkpoints.append,
    )

    assert checkpoints[0] == "road_barrier_start"
    assert checkpoints[-1] == "finalize_done"
    assert "boundary_fill_done" in checkpoints
    assert "merge_done" in checkpoints


def test_run_postprocess_emits_merge_scan_checkpoints(monkeypatch: pytest.MonkeyPatch):
    class MergeProgressCfg(DummyCfg):
        FRAMEWORK_SKIP_WATERSHED_FOR_LARGEST = False

    def _fake_boundary_to_regions(boundary_prob, min_region_px=50, boundary_thresh=None):
        labeled = np.zeros_like(boundary_prob, dtype=np.int32)
        labeled[2:5, 2:5] = 1
        labeled[2:5, 7:10] = 2
        return labeled, 2

    def _fake_hierarchical_merge(
        labeled,
        boundary_prob,
        max_ndvi,
        ndvi_std,
        barrier_mask,
        cfg,
        region_profile=None,
        progress_callback=None,
    ):
        if progress_callback is not None:
            progress_callback(1, 4)
            progress_callback(4, 4)
        return labeled

    monkeypatch.setattr(postprocess_module, "boundary_to_regions", _fake_boundary_to_regions)
    monkeypatch.setattr(postprocess_module, "hierarchical_merge", _fake_hierarchical_merge)

    candidate_mask = np.zeros((14, 14), dtype=bool)
    candidate_mask[2:12, 2:12] = True

    water_mask = np.zeros_like(candidate_mask, dtype=bool)
    classes = np.full(candidate_mask.shape, CROP, dtype=np.uint8)
    ndvi = np.full(candidate_mask.shape, 0.5, dtype=np.float32)
    ndwi = np.zeros(candidate_mask.shape, dtype=np.float32)
    ndvi_std = np.full(candidate_mask.shape, 0.18, dtype=np.float32)
    edge = np.zeros(candidate_mask.shape, dtype=np.float32)
    checkpoints: list[str] = []

    run_postprocess(
        candidate_mask=candidate_mask,
        water_mask=water_mask,
        classes=classes,
        ndvi=ndvi,
        ndwi=ndwi,
        ndvi_std=ndvi_std,
        edge_composite=edge,
        cfg=MergeProgressCfg(),
        progress_callback=checkpoints.append,
    )

    assert any(cp.startswith("boundary_merge_scan:") for cp in checkpoints)
    assert any(cp.startswith("merge_scan:") for cp in checkpoints)


def test_run_postprocess_skips_boundary_fill_when_region_cap_exceeded(monkeypatch: pytest.MonkeyPatch):
    class LowBoundaryCapCfg(DummyCfg):
        BOUNDARY_FILL_MAX_REGIONS = 1

    def _fake_boundary_to_regions(boundary_prob, min_region_px=50, boundary_thresh=None):
        labeled = np.zeros_like(boundary_prob, dtype=np.int32)
        labeled[2:8, 2:8] = 1
        labeled[10:16, 10:16] = 2
        return labeled, 2

    monkeypatch.setattr(postprocess_module, "boundary_to_regions", _fake_boundary_to_regions)

    candidate_mask = np.zeros((18, 18), dtype=bool)
    candidate_mask[2:16, 2:16] = True

    water_mask = np.zeros_like(candidate_mask, dtype=bool)
    classes = np.full(candidate_mask.shape, CROP, dtype=np.uint8)
    ndvi = np.full(candidate_mask.shape, 0.5, dtype=np.float32)
    ndwi = np.zeros(candidate_mask.shape, dtype=np.float32)
    ndvi_std = np.full(candidate_mask.shape, 0.18, dtype=np.float32)
    edge = np.zeros(candidate_mask.shape, dtype=np.float32)

    processed, debug = run_postprocess(
        candidate_mask=candidate_mask,
        water_mask=water_mask,
        classes=classes,
        ndvi=ndvi,
        ndwi=ndwi,
        ndvi_std=ndvi_std,
        edge_composite=edge,
        cfg=LowBoundaryCapCfg(),
        return_debug_stats=True,
    )

    assert processed.shape == candidate_mask.shape
    assert debug["stats"]["boundary_fill_meta"]["skipped"] == 1


def test_run_postprocess_emits_recovery_boundary_anchor_candidate_mask(
    monkeypatch: pytest.MonkeyPatch,
):
    class AnchorCfg(DummyCfg):
        RECOVERY_BOUNDARY_ANCHOR_ENABLED = True
        RECOVERY_BOUNDARY_ANCHOR_DILATION_PX = 1

    def _fake_boundary_to_regions(boundary_prob, min_region_px=50, boundary_thresh=None):
        labeled = np.zeros_like(boundary_prob, dtype=np.int32)
        labeled[5:11, 5:11] = 1
        return labeled, 1

    def _identity_hierarchical_merge(
        labeled_boundary,
        boundary_prob,
        ndvi,
        ndvi_std,
        barrier_mask,
        cfg,
        region_profile=None,
        progress_callback=None,
    ):
        return labeled_boundary

    def _keep_all_regions(labeled_boundary, ndvi, ndvi_std, barrier_mask, cfg):
        return labeled_boundary > 0

    monkeypatch.setattr(postprocess_module, "boundary_to_regions", _fake_boundary_to_regions)
    monkeypatch.setattr(postprocess_module, "hierarchical_merge", _identity_hierarchical_merge)
    monkeypatch.setattr(postprocess_module, "filter_regions_by_phenology", _keep_all_regions)

    candidate_mask = np.zeros((18, 18), dtype=bool)
    candidate_mask[4:13, 4:13] = True
    water_mask = np.zeros_like(candidate_mask, dtype=bool)
    classes = np.full(candidate_mask.shape, CROP, dtype=np.uint8)
    ndvi = np.full(candidate_mask.shape, 0.46, dtype=np.float32)
    ndwi = np.zeros(candidate_mask.shape, dtype=np.float32)
    ndvi_std = np.full(candidate_mask.shape, 0.18, dtype=np.float32)
    edge = np.full(candidate_mask.shape, 0.45, dtype=np.float32)

    processed, debug = run_postprocess(
        candidate_mask=candidate_mask,
        water_mask=water_mask,
        classes=classes,
        ndvi=ndvi,
        ndwi=ndwi,
        ndvi_std=ndvi_std,
        edge_composite=edge,
        cfg=AnchorCfg(),
        return_candidate_masks=True,
        return_debug_stats=True,
    )

    anchor = np.asarray(debug["candidate_masks"]["recovery_boundary_anchor"], dtype=bool)
    assert processed.shape == candidate_mask.shape
    assert np.count_nonzero(anchor) > 0
    assert debug["stats"]["summary"]["recovery_boundary_anchor_pixels"] > 0


def test_run_postprocess_south_profile_bridges_near_components():
    candidate_mask = np.zeros((20, 20), dtype=bool)
    candidate_mask[7:11, 5:8] = True
    candidate_mask[7:11, 11:14] = True

    water_mask = np.zeros_like(candidate_mask, dtype=bool)
    classes = np.full(candidate_mask.shape, GRASS, dtype=np.uint8)
    classes[7:11, 5:8] = CROP
    classes[7:11, 11:14] = CROP
    ndvi = np.full(candidate_mask.shape, 0.05, dtype=np.float32)
    ndvi[7:11, 5:8] = 0.5
    ndvi[7:11, 11:14] = 0.5
    ndwi = np.zeros(candidate_mask.shape, dtype=np.float32)
    edge = np.zeros(candidate_mask.shape, dtype=np.float32)

    processed, debug = run_postprocess(
        candidate_mask=candidate_mask,
        water_mask=water_mask,
        classes=classes,
        ndvi=ndvi,
        ndwi=ndwi,
        cfg=DummyCfg(),
        edge_composite=edge,
        region_profile="south_recall",
        return_debug_stats=True,
    )

    summary = debug["stats"]["summary"]
    assert summary["bridge_added_pixels"] > 0
    assert "south_bridge" in summary["region_profile_actions"]
    assert summary["components_after_merge"] == 1
    assert processed[8, 9]


def test_run_postprocess_does_not_cut_crop_by_grass_barrier():
    candidate_mask = np.zeros((12, 12), dtype=bool)
    candidate_mask[3:9, 3:9] = True

    water_mask = np.zeros_like(candidate_mask, dtype=bool)
    classes = np.full(candidate_mask.shape, CROP, dtype=np.uint8)
    ndvi = np.zeros(candidate_mask.shape, dtype=np.float32)
    ndvi[3:9, 3:9] = 0.52
    ndwi = np.zeros(candidate_mask.shape, dtype=np.float32)
    ndvi_std = np.zeros(candidate_mask.shape, dtype=np.float32)
    ndvi_std[3:9, 3:9] = 0.08  # phenology marks this as grass-like

    processed = run_postprocess(
        candidate_mask=candidate_mask,
        water_mask=water_mask,
        classes=classes,
        ndvi=ndvi,
        ndwi=ndwi,
        ndvi_std=ndvi_std,
        cfg=DummyCfg(),
    )

    assert processed[5:7, 5:7].all()


def test_run_postprocess_uses_worldcover_as_weak_prior():
    candidate_mask = np.zeros((10, 10), dtype=bool)
    candidate_mask[2:8, 2:8] = True

    water_mask = np.zeros_like(candidate_mask, dtype=bool)
    classes = np.full(candidate_mask.shape, CROP, dtype=np.uint8)
    ndvi = np.full(candidate_mask.shape, 0.55, dtype=np.float32)
    ndwi = np.zeros(candidate_mask.shape, dtype=np.float32)
    ndvi_std = np.full(candidate_mask.shape, 0.18, dtype=np.float32)
    worldcover_mask = np.full(candidate_mask.shape, 90, dtype=np.uint8)

    processed = run_postprocess(
        candidate_mask=candidate_mask,
        water_mask=water_mask,
        classes=classes,
        ndvi=ndvi,
        ndwi=ndwi,
        ndvi_std=ndvi_std,
        worldcover_mask=worldcover_mask,
        cfg=DummyCfg(),
    )

    assert processed[3:7, 3:7].all()


def test_run_postprocess_can_seed_from_ndvi_field_candidate_without_crop_class():
    candidate_mask = np.zeros((12, 12), dtype=bool)
    water_mask = np.zeros_like(candidate_mask, dtype=bool)
    classes = np.zeros_like(candidate_mask, dtype=np.uint8)

    ndvi = np.zeros(candidate_mask.shape, dtype=np.float32)
    ndvi[3:9, 3:9] = 0.43
    ndwi = np.zeros(candidate_mask.shape, dtype=np.float32)
    ndvi_std = np.zeros(candidate_mask.shape, dtype=np.float32)
    ndvi_std[3:9, 3:9] = 0.18

    processed = run_postprocess(
        candidate_mask=candidate_mask,
        water_mask=water_mask,
        classes=classes,
        ndvi=ndvi,
        ndwi=ndwi,
        ndvi_std=ndvi_std,
        cfg=DummyCfg(),
    )

    assert processed[4:8, 4:8].all()


def test_run_postprocess_does_not_fill_large_holes_in_large_components():
    candidate_mask = np.zeros((40, 40), dtype=bool)
    candidate_mask[5:35, 5:35] = True
    candidate_mask[17:23, 17:23] = False  # 36 px hole, larger than clean_raster_mask hole threshold

    water_mask = np.zeros_like(candidate_mask, dtype=bool)
    classes = np.full(candidate_mask.shape, CROP, dtype=np.uint8)
    ndvi = np.full(candidate_mask.shape, 0.5, dtype=np.float32)
    ndwi = np.zeros(candidate_mask.shape, dtype=np.float32)
    ndvi_std = np.full(candidate_mask.shape, 0.18, dtype=np.float32)
    ndvi_std[17:23, 17:23] = 0.05

    processed = run_postprocess(
        candidate_mask=candidate_mask,
        water_mask=water_mask,
        classes=classes,
        ndvi=ndvi,
        ndwi=ndwi,
        ndvi_std=ndvi_std,
        cfg=DummyCfg(),
    )

    assert not processed[18:22, 18:22].any()


def test_run_postprocess_keeps_worldcover_tree_island_when_hard_exclusion_enabled():
    candidate_mask = np.zeros((16, 16), dtype=bool)
    candidate_mask[3:13, 3:13] = True
    candidate_mask[7:9, 7:9] = False

    water_mask = np.zeros_like(candidate_mask, dtype=bool)
    classes = np.full(candidate_mask.shape, CROP, dtype=np.uint8)
    ndvi = np.full(candidate_mask.shape, 0.5, dtype=np.float32)
    ndwi = np.zeros(candidate_mask.shape, dtype=np.float32)
    ndvi_std = np.full(candidate_mask.shape, 0.18, dtype=np.float32)
    worldcover_mask = np.zeros(candidate_mask.shape, dtype=np.uint8)
    worldcover_mask[7:9, 7:9] = 10

    processed = run_postprocess(
        candidate_mask=candidate_mask,
        water_mask=water_mask,
        classes=classes,
        ndvi=ndvi,
        ndwi=ndwi,
        ndvi_std=ndvi_std,
        worldcover_mask=worldcover_mask,
        cfg=DummyCfg(),
    )

    assert not processed[7:9, 7:9].any()


def test_run_postprocess_can_disable_worldcover_tree_hard_exclusion():
    class NoTreeHardCfg(DummyCfg):
        WC_TREE_HARD_EXCLUSION = False

    candidate_mask = np.zeros((16, 16), dtype=bool)
    candidate_mask[3:13, 3:13] = True
    candidate_mask[7:9, 7:9] = False

    water_mask = np.zeros_like(candidate_mask, dtype=bool)
    classes = np.full(candidate_mask.shape, CROP, dtype=np.uint8)
    ndvi = np.full(candidate_mask.shape, 0.5, dtype=np.float32)
    ndwi = np.zeros(candidate_mask.shape, dtype=np.float32)
    ndvi_std = np.full(candidate_mask.shape, 0.18, dtype=np.float32)
    worldcover_mask = np.zeros(candidate_mask.shape, dtype=np.uint8)
    worldcover_mask[7:9, 7:9] = 10

    processed = run_postprocess(
        candidate_mask=candidate_mask,
        water_mask=water_mask,
        classes=classes,
        ndvi=ndvi,
        ndwi=ndwi,
        ndvi_std=ndvi_std,
        worldcover_mask=worldcover_mask,
        cfg=NoTreeHardCfg(),
    )

    assert processed[7:9, 7:9].all()


def test_run_postprocess_does_not_grow_into_worldcover_tree_strip():
    candidate_mask = np.zeros((18, 18), dtype=bool)
    candidate_mask[6:12, 4:7] = True

    water_mask = np.zeros_like(candidate_mask, dtype=bool)
    classes = np.full(candidate_mask.shape, GRASS, dtype=np.uint8)
    classes[6:12, 4:7] = CROP
    ndvi = np.full(candidate_mask.shape, 0.48, dtype=np.float32)
    ndwi = np.zeros(candidate_mask.shape, dtype=np.float32)
    ndvi_std = np.full(candidate_mask.shape, 0.18, dtype=np.float32)
    worldcover_mask = np.zeros(candidate_mask.shape, dtype=np.uint8)
    worldcover_mask[4:14, 7:9] = 10

    processed = run_postprocess(
        candidate_mask=candidate_mask,
        water_mask=water_mask,
        classes=classes,
        ndvi=ndvi,
        ndwi=ndwi,
        ndvi_std=ndvi_std,
        worldcover_mask=worldcover_mask,
        cfg=DummyCfg(),
    )

    assert not processed[4:14, 7:9].any()
