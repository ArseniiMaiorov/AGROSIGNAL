"""Compute spectral indices from Sentinel-2 bands."""
import numpy as np


def _safe_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        result = a / b
    return np.where(np.isfinite(result), result, np.nan)


def _normalized_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        result = (a - b) / (a + b)
    return np.where(np.isfinite(result), result, np.nan)


def compute_ndvi(b4: np.ndarray, b8: np.ndarray) -> np.ndarray:
    """NDVI = (B8 - B4) / (B8 + B4). Shape: same as input."""
    return _normalized_diff(b8, b4)


def compute_ndwi(b3: np.ndarray, b8: np.ndarray) -> np.ndarray:
    """NDWI = (B3 - B8) / (B3 + B8)."""
    return _normalized_diff(b3, b8)


def compute_mndwi(b3: np.ndarray, b11: np.ndarray) -> np.ndarray:
    """MNDWI = (B3 - B11) / (B3 + B11)."""
    return _normalized_diff(b3, b11)


def compute_ndmi(b8: np.ndarray, b11: np.ndarray) -> np.ndarray:
    """NDMI = (B8 - B11) / (B8 + B11)."""
    return _normalized_diff(b8, b11)


def compute_bsi(
    b2: np.ndarray, b4: np.ndarray, b8: np.ndarray, b11: np.ndarray
) -> np.ndarray:
    """BSI = ((B11+B4) - (B8+B2)) / ((B11+B4) + (B8+B2))."""
    num = (b11 + b4) - (b8 + b2)
    den = (b11 + b4) + (b8 + b2)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = num / den
    return np.where(np.isfinite(result), result, np.nan)


def compute_msi(b8: np.ndarray, b11: np.ndarray) -> np.ndarray:
    """MSI = B11 / B8."""
    return _safe_ratio(b11, b8)


def compute_ndre(b5: np.ndarray, b8a: np.ndarray) -> np.ndarray:
    """NDRE = (B8A - B05) / (B8A + B05).

    Normalized Difference Red-Edge index.
    More sensitive to chlorophyll content than NDVI, especially
    in dense canopies where NDVI saturates.
    """
    return _normalized_diff(b8a, b5)


def compute_ci_rededge(b5: np.ndarray, b8a: np.ndarray) -> np.ndarray:
    """CIrededge = B8A / B05 - 1.

    Chlorophyll Index Red-Edge (Gitelson et al., 2005).
    Linear relationship with canopy chlorophyll content.
    """
    return _safe_ratio(b8a, b5) - 1.0


def compute_rededge_slope(b5: np.ndarray, b7: np.ndarray) -> np.ndarray:
    """Red-edge slope = (B07 - B05) / (B07 + B05).

    Captures the steepness of the red-edge inflection point.
    Strong discriminator between crop types and crop-vs-natural vegetation.
    """
    return _normalized_diff(b7, b5)


def compute_all_indices(bands: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Compute all indices from band dict.

    Args:
        bands: dict with keys B2, B3, B4, B8, B11 (optional B12, B5, B6, B7, B8A).
               Each value can be (H,W) or (T,H,W).

    Returns:
        dict with keys NDVI, NDWI, MNDWI, NDMI, BSI, MSI,
        and optionally NDRE, CIrededge, RE_SLOPE if red-edge bands present.
    """
    result = {
        "NDVI": compute_ndvi(bands["B4"], bands["B8"]),
        "NDWI": compute_ndwi(bands["B3"], bands["B8"]),
        "MNDWI": compute_mndwi(bands["B3"], bands["B11"]),
        "NDMI": compute_ndmi(bands["B8"], bands["B11"]),
        "BSI": compute_bsi(bands["B2"], bands["B4"], bands["B8"], bands["B11"]),
        "MSI": compute_msi(bands["B8"], bands["B11"]),
    }

    # Red-edge indices (v4 pipeline)
    if "B5" in bands and "B8A" in bands:
        result["NDRE"] = compute_ndre(bands["B5"], bands["B8A"])
        result["CIrededge"] = compute_ci_rededge(bands["B5"], bands["B8A"])
    if "B5" in bands and "B7" in bands:
        result["RE_SLOPE"] = compute_rededge_slope(bands["B5"], bands["B7"])

    return result
