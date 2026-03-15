from __future__ import annotations

import torch

from training.gen_data import _metrics


def test_area_ratio_pred_gt_ignores_empty_ground_truth_samples():
    preds = {
        "extent": torch.tensor(
            [
                [[[8.0, 8.0], [8.0, 8.0]]],
                [[[8.0, 8.0], [8.0, 8.0]]],
            ],
            dtype=torch.float32,
        ),
        "boundary": torch.zeros((2, 1, 2, 2), dtype=torch.float32),
        "distance": torch.zeros((2, 1, 2, 2), dtype=torch.float32),
    }
    target = {
        "extent": torch.tensor(
            [
                [[[1.0, 1.0], [0.0, 0.0]]],
                [[[0.0, 0.0], [0.0, 0.0]]],
            ],
            dtype=torch.float32,
        ),
        "boundary": torch.zeros((2, 1, 2, 2), dtype=torch.float32),
        "distance": torch.zeros((2, 1, 2, 2), dtype=torch.float32),
    }

    metrics = _metrics(preds, target)

    assert metrics["area_ratio_valid_count"] == 1.0
    assert metrics["area_ratio_pred_gt"] == 2.0
