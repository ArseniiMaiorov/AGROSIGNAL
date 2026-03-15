"""Land cover classification using knowledge-based phenological rules."""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation

WATER = 1
BUILTUP = 2
FOREST = 3
GRASS = 4
CROP = 5
OTHER = 0


@dataclass
class PhenoThresholds:
    ndwi_water: float = 0.10
    mndwi_water: float = 0.05
    bsi_built: float = 0.15
    std_built: float = 0.02
    ndvi_forest_min: float = 0.45
    delta_forest: float = 0.15
    ndvi_grass_mean: float = 0.35
    delta_grass: float = 0.2
    ndvi_crop_max: float = 0.62
    ndvi_crop_min: float = 0.25
    delta_crop: float = 0.3
    msi_crop: float = 1.1
    n_valid_min: int = 4


def compute_hydro_masks(
    ndwi_max: np.ndarray,
    valid_count: np.ndarray,
    thresholds: PhenoThresholds | None = None,
    *,
    mndwi_max: np.ndarray | None = None,
    scl_water_mask: np.ndarray | None = None,
    ndwi_mean: np.ndarray | None = None,
    hydro_profile: str = "water_aware",
    open_water_ndwi: float | None = None,
    open_water_mndwi: float | None = None,
    seasonal_wet_ndwi: float | None = None,
    seasonal_wet_mndwi: float | None = None,
    riparian_buffer_px: int = 2,
) -> dict[str, np.ndarray]:
    """Derive hydro masks for strict water and soft riparian zones."""
    if thresholds is None:
        thresholds = PhenoThresholds()

    profile = str(hydro_profile or "off").strip().lower()
    enough_observations = valid_count >= thresholds.n_valid_min

    open_water = np.zeros(ndwi_max.shape, dtype=bool)
    if scl_water_mask is not None:
        open_water |= np.asarray(scl_water_mask, dtype=bool)
    open_water |= np.asarray(ndwi_max, dtype=np.float32) > float(
        thresholds.ndwi_water if open_water_ndwi is None else open_water_ndwi
    )
    if mndwi_max is not None:
        open_water |= np.asarray(mndwi_max, dtype=np.float32) > float(
            thresholds.mndwi_water if open_water_mndwi is None else open_water_mndwi
        )
    open_water &= enough_observations

    seasonal_wet = np.zeros_like(open_water, dtype=bool)
    riparian_soft = np.zeros_like(open_water, dtype=bool)
    riparian_hard = np.zeros_like(open_water, dtype=bool)

    if profile != "off":
        if ndwi_mean is not None:
            seasonal_wet |= np.asarray(ndwi_mean, dtype=np.float32) > float(
                (thresholds.ndwi_water * 0.7)
                if seasonal_wet_ndwi is None
                else seasonal_wet_ndwi
            )
        if mndwi_max is not None:
            seasonal_wet |= np.asarray(mndwi_max, dtype=np.float32) > float(
                (thresholds.mndwi_water * 0.5)
                if seasonal_wet_mndwi is None
                else seasonal_wet_mndwi
            )
        seasonal_wet &= enough_observations
        seasonal_wet &= ~open_water

        if profile == "water_aware":
            soft_source = open_water | seasonal_wet
            if np.any(soft_source):
                riparian_soft = binary_dilation(
                    soft_source,
                    iterations=max(0, int(riparian_buffer_px)),
                ) & ~soft_source
            if np.any(open_water):
                riparian_hard = binary_dilation(open_water, iterations=1) & ~open_water

    return {
        "open_water_mask": open_water.astype(bool, copy=False),
        "seasonal_wet_mask": seasonal_wet.astype(bool, copy=False),
        "riparian_soft_mask": riparian_soft.astype(bool, copy=False),
        "riparian_hard_mask": riparian_hard.astype(bool, copy=False),
    }


def classify_land_cover(
    pheno: dict[str, np.ndarray],
    ndwi_max: np.ndarray,
    bsi_med: np.ndarray,
    msi_med: np.ndarray,
    valid_count: np.ndarray,
    thresholds: PhenoThresholds | None = None,
    mndwi_max: np.ndarray | None = None,
    scl_water_mask: np.ndarray | None = None,
    ndwi_mean: np.ndarray | None = None,
) -> np.ndarray:
    """Classify each pixel into land cover classes.

    Args:
        pheno: dict from compute_phenometrics (ndvi_min, ndvi_max, ndvi_mean, ndvi_std, ndvi_delta).
        ndwi_max: (H, W) NDWI statistic used for water masking (median recommended).
        bsi_med: (H, W) median BSI.
        msi_med: (H, W) median MSI.
        valid_count: (H, W) number of valid observations per pixel.
        thresholds: classification thresholds.
        mndwi_max: optional (H, W) max MNDWI across time.
        scl_water_mask: optional (H, W) hard water mask from SCL.
        ndwi_mean: optional (H, W) mean NDWI for seasonal wetland detection.

    Returns:
        (H, W) int array of class labels (0-5).
    """
    if thresholds is None:
        thresholds = PhenoThresholds()

    h, w = pheno["ndvi_min"].shape
    classes = np.zeros((h, w), dtype=np.uint8)

    ndvi_min = pheno["ndvi_min"]
    ndvi_max = pheno["ndvi_max"]
    ndvi_mean = pheno["ndvi_mean"]
    ndvi_std = pheno["ndvi_std"]
    ndvi_delta = pheno["ndvi_delta"]
    enough_observations = valid_count >= thresholds.n_valid_min

    hydro_masks = compute_hydro_masks(
        ndwi_max,
        valid_count,
        thresholds,
        mndwi_max=mndwi_max,
        scl_water_mask=scl_water_mask,
        ndwi_mean=ndwi_mean,
        hydro_profile="balanced",
    )
    water = hydro_masks["open_water_mask"]

    classes[water] = WATER

    builtup = (~water
               & enough_observations
               & (bsi_med > thresholds.bsi_built)
               & (ndvi_std < thresholds.std_built))
    classes[builtup] = BUILTUP

    forest = (~water & ~builtup
              & enough_observations
              & ((ndvi_min > thresholds.ndvi_forest_min)
                 | (ndvi_delta < thresholds.delta_forest)))
    classes[forest] = FOREST

    grass = (~water & ~builtup & ~forest
             & enough_observations
             & (ndvi_mean > thresholds.ndvi_grass_mean)
             & (ndvi_delta < thresholds.delta_grass))
    classes[grass] = GRASS

    crop = (~water & ~builtup & ~forest & ~grass
            & enough_observations
            & (ndvi_max > thresholds.ndvi_crop_max)
            & (ndvi_min < thresholds.ndvi_crop_min)
            & (ndvi_delta > thresholds.delta_crop)
            & (msi_med < thresholds.msi_crop))
    classes[crop] = CROP

    return classes
