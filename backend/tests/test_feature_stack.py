import numpy as np

from processing.fields.feature_stack import build_feature_stack_v4


def test_build_feature_stack_v4_respects_requested_channel_profile():
    h, w = 4, 4
    edge = np.ones((h, w), dtype=np.float32) * 0.5
    kwargs = {
        "edge_composite": edge,
        "max_ndvi": np.ones((h, w), dtype=np.float32) * 0.8,
        "mean_ndvi": np.ones((h, w), dtype=np.float32) * 0.5,
        "ndvi_std": np.ones((h, w), dtype=np.float32) * 0.1,
        "ndwi_mean": np.zeros((h, w), dtype=np.float32),
        "bsi_mean": np.zeros((h, w), dtype=np.float32),
        "scl_valid_fraction": np.ones((h, w), dtype=np.float32),
        "rgb_r": np.ones((h, w), dtype=np.float32) * 0.3,
        "rgb_g": np.ones((h, w), dtype=np.float32) * 0.2,
        "rgb_b": np.ones((h, w), dtype=np.float32) * 0.1,
        "ndvi_entropy": np.ones((h, w), dtype=np.float32) * 0.4,
        "mndwi_max": np.zeros((h, w), dtype=np.float32),
        "ndmi_mean": np.zeros((h, w), dtype=np.float32),
        "ndwi_median": np.zeros((h, w), dtype=np.float32),
        "green_median": np.ones((h, w), dtype=np.float32) * 0.2,
        "swir_median": np.ones((h, w), dtype=np.float32) * 0.1,
    }

    channels_v2 = (
        "edge_composite",
        "max_ndvi",
        "mean_ndvi",
        "ndvi_std",
        "ndwi_mean",
        "bsi_mean",
        "scl_valid_fraction",
        "rgb_r",
        "rgb_g",
        "rgb_b",
        "ndvi_entropy",
        "mndwi_max",
        "ndmi_mean",
        "ndwi_median",
        "green_median",
        "swir_median",
    )
    stack = build_feature_stack_v4(feature_channels=channels_v2, **kwargs)
    assert stack.shape == (16, h, w)
    assert np.isclose(float(stack[0, 0, 0]), 0.5)
