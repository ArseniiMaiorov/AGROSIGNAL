# Regional Retrain Plan

This file fixes the exact retrain contour for the `south_recall` and
`north_boundary` regional profiles.

## Scope

- Model topology stays unchanged: `extent + boundary + distance`
- Regional behavior is driven by:
  - region-stratified curated GT
  - region-aware weak-label sampling
  - region-specific calibration thresholds

## Curated GT campaign

- Total: 150 AOI
- South: 60 AOI
- North: 60 AOI
- Control/mixed: 30 AOI

Each manifest item must include:

- `region_band`
- `region_boundary_profile_target`
- `error_mode_tag`
- `parcel_shape_class`
- `adjacency_tag`

Recommended split:

- `train_curated`: 90 AOI
- `calibration`: 30 AOI
- `holdout`: 30 AOI

The split must remain stratified across:

- south / north / control
- road / water / forest adjacency
- fragmentation / shrink error modes

## Sampling policy

- 40% curated GT patches
- 40% strong weak-label patches from the same `region_band`
- 20% hard negatives

Required hard negatives:

- South:
  - dry bare strips
  - heterogeneous open land
  - roadside open areas
- North:
  - wet meadows
  - riparian zones
  - forest edges
  - partially overgrown abandoned parcels

## Training stages

### Stage A: Warm start

- Start from current `boundary_unet_v1`
- 60% weak labels, 40% curated
- 8-12 epochs

### Stage B: Region hard-example fine-tune

- 70% curated
- 30% hard negatives
- Oversample:
  - `south_fragmentation`
  - `north_shrink`
- 6-10 epochs

### Stage C: Calibration

- No learning
- Sweep thresholds per profile

South sweep:

- `ML_EXTENT_BIN_THRESHOLD`: `0.30, 0.32, 0.34, 0.36, 0.38`
- `SOUTH_GAP_CLOSE_MAX_HA`: `0.6, 0.8, 1.0`
- `SOUTH_POST_MIN_FIELD_AREA_HA`: `0.10, 0.12, 0.15`

North sweep:

- `ML_EXTENT_BIN_THRESHOLD`: `0.38, 0.40, 0.42, 0.44`
- `NORTH_VECTORIZE_SIMPLIFY_TOL_M`: `0.0, 0.25, 0.5`
- `NORTH_BOUNDARY_SMOOTH_SIMPLIFY_TOL_M`: `0.0, 0.5, 1.0`

## Acceptance targets

### South

- `field_recall_south >= 0.92`
- `missed_fields_rate_south <= 0.08`
- `oversegmented_fields_rate_south <= 0.12`
- `mean_components_per_gt_field_south <= 1.20`
- `boundary_iou_south_median >= 0.78`

### North

- `boundary_iou_north_median >= 0.82`
- `contour_shrink_ratio_north_median` in `0.96-1.04`
- `centroid_shift_m_north_p90 <= 5`
- visually obvious inward shrink on north holdout fields <= 10%

## Execution

Run the reproducible pipeline with:

```bash
backend/training/run_regional_retrain_pipeline.sh
```

Set `REGIONAL_HOLDOUT_JSON` if you want to use a non-default manifest path.
