import numpy as np

from processing.fields.ml_fusion import boundary_guided_ml_seed, fuse_ml_primary_candidate


class _Cfg:
    ML_EXTENT_CALIBRATION_ENABLED = True
    ML_EXTENT_BIN_THRESHOLD = 0.42
    POST_BOUNDARY_DILATION_PX = 1
    POST_BOUNDARY_DILATION_MAX_PX = 3
    POST_LARGE_FIELD_RESCUE_ENABLED = True
    POST_LARGE_FIELD_RESCUE_MIN_AREA_HA = 2.0
    POST_PX_AREA_M2 = 100.0
    PHENO_FIELD_MAX_NDVI_MIN = 0.35
    PHENO_FIELD_MAX_NDVI_MAX = 0.62
    PHENO_FIELD_NDVI_STD_MIN = 0.15


def test_fuse_balanced_adds_local_pre_ml_support():
    ml_seed = np.zeros((5, 5), dtype=bool)
    ml_seed[2, 2] = True
    pre_ml = np.zeros((5, 5), dtype=bool)
    pre_ml[2, 1] = True
    fused, actions = fuse_ml_primary_candidate(ml_seed, pre_ml, "balanced")
    assert fused[2, 2]
    assert fused[2, 1]
    assert "balanced_ml_seed_union" in actions


def test_boundary_guided_seed_returns_debug_payload():
    extent_prob = np.zeros((8, 8), dtype=np.float32)
    extent_prob[2:6, 2:6] = 0.9
    boundary_prob = np.full((8, 8), 0.2, dtype=np.float32)
    ndvi = np.full((8, 8), 0.5, dtype=np.float32)
    ndvi_std = np.full((8, 8), 0.2, dtype=np.float32)

    seed, debug = boundary_guided_ml_seed(
        extent_prob=extent_prob,
        boundary_prob=boundary_prob,
        ndvi=ndvi,
        ndvi_std=ndvi_std,
        cfg=_Cfg(),
    )
    assert seed.dtype == bool
    assert int(debug["seed_pixels_after_dilation"]) >= int(debug["seed_pixels_before_dilation"])
