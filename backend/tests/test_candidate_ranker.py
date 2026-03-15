import numpy as np

from processing.fields.candidate_ranker import (
    CandidatePolygon,
    compute_branch_agreement,
    rank_and_suppress,
)


def test_compute_branch_agreement_emits_progress_callback():
    masks = []
    for offset in range(3):
        mask = np.zeros((8, 8), dtype=bool)
        mask[1 + offset : 4 + offset, 1:4] = True
        masks.append(mask)

    candidates = [
        CandidatePolygon(mask=masks[0], branch="boundary_first"),
        CandidatePolygon(mask=masks[1], branch="crop_region"),
        CandidatePolygon(mask=masks[2], branch="recovery_second_pass"),
    ]
    events = []

    compute_branch_agreement(
        candidates,
        progress_callback=lambda completed, total, stage: events.append(
            (int(completed), int(total), str(stage))
        ),
    )

    assert events
    assert events[0] == (0, 3, "branch_agreement")
    assert events[-1] == (3, 3, "branch_agreement")
    assert all(stage == "branch_agreement" for _, _, stage in events)


def test_rank_and_suppress_emits_prepare_and_suppress_progress():
    mask_a = np.zeros((8, 8), dtype=bool)
    mask_a[1:4, 1:4] = True
    mask_b = np.zeros((8, 8), dtype=bool)
    mask_b[1:4, 1:4] = True
    mask_c = np.zeros((8, 8), dtype=bool)
    mask_c[4:7, 4:7] = True

    candidates = [
        CandidatePolygon(mask=mask_a, branch="boundary_first", score=0.92),
        CandidatePolygon(mask=mask_b, branch="crop_region", score=0.81),
        CandidatePolygon(mask=mask_c, branch="recovery_second_pass", score=0.73),
    ]
    events = []

    results = rank_and_suppress(
        candidates,
        min_score=0.2,
        iou_threshold=0.1,
        progress_callback=lambda completed, total, stage: events.append(
            (int(completed), int(total), str(stage))
        ),
    )

    assert results
    assert events
    assert events[0] == (0, 3, "rank_prepare")
    assert any(stage == "suppress" for _, _, stage in events)
    assert events[-1] == (3, 3, "done")
    assert sum(1 for result in results if result.keep) == 2
