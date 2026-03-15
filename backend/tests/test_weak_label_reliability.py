import numpy as np

from training.gen_data import _load_tile_quality_table, compute_tile_quality_weight
from training.generate_weak_labels_real_tiles import targeted_quality_rescue


def test_compute_tile_quality_weight_rules():
    assert compute_tile_quality_weight(used_fallback=False, quality_gate_failed=False) == 1.0
    assert compute_tile_quality_weight(used_fallback=True, quality_gate_failed=False) == 0.5
    assert compute_tile_quality_weight(used_fallback=False, quality_gate_failed=True) == 0.25
    assert compute_tile_quality_weight(used_fallback=True, quality_gate_failed=True) == 0.25


def test_targeted_quality_rescue_accepts_plausible_component():
    candidate = np.zeros((64, 64), dtype=bool)
    candidate[16:40, 20:44] = True
    hard_exclusion = np.zeros_like(candidate, dtype=bool)
    max_ndvi = np.zeros((64, 64), dtype=np.float32)
    max_ndvi[16:40, 20:44] = 0.62

    rescued, meta = targeted_quality_rescue(
        candidate_mask=candidate,
        hard_exclusion=hard_exclusion,
        max_ndvi=max_ndvi,
        lat=55.0,
    )

    assert bool(meta["accepted"]) is True
    assert int(meta["components_kept"]) >= 1
    assert rescued.any()


def test_targeted_quality_rescue_rejects_tiny_noise():
    candidate = np.zeros((64, 64), dtype=bool)
    candidate[5:8, 5:8] = True
    hard_exclusion = np.zeros_like(candidate, dtype=bool)
    max_ndvi = np.full((64, 64), 0.8, dtype=np.float32)

    rescued, meta = targeted_quality_rescue(
        candidate_mask=candidate,
        hard_exclusion=hard_exclusion,
        max_ndvi=max_ndvi,
        lat=55.0,
    )

    assert bool(meta["accepted"]) is False
    assert bool(rescued.any()) is False


def test_load_tile_quality_table_applies_required_weights(tmp_path):
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    summary = labels_dir / "weak_labels_summary.csv"
    summary.write_text(
        "tile_id,used_fallback,quality_gate_failed,temporal_amp,temporal_entropy\n"
        "tile_a,False,False,0.42,1.8\n"
        "tile_b,True,False,0.31,2.2\n"
        "tile_c,True,True,0.15,3.0\n",
        encoding="utf-8",
    )
    table = _load_tile_quality_table(labels_dir)
    assert table["tile_a"]["tile_weight"] == 1.0
    assert table["tile_b"]["tile_weight"] == 0.5
    assert table["tile_c"]["tile_weight"] == 0.25
    assert table["tile_c"]["quality_gate_failed"] is True
