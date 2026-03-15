"""Primary ML inference for field boundary detection (v4)."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.ndimage import zoom as nd_zoom

try:  # pragma: no cover - optional dependency
    import onnxruntime as ort
except Exception:  # pragma: no cover
    ort = None

try:  # pragma: no cover - optional dependency
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = None  # type: ignore[assignment]

from core.logging import get_logger

logger = get_logger(__name__)

ProgressCallback = Callable[[str, int, int], None]

FEATURE_CHANNELS_V1_18: tuple[str, ...] = (
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
    "s1_vv_mean",
    "s1_vh_mean",
    "ndvi_entropy",
    "mndwi_max",
    "ndmi_mean",
    "ndwi_median",
    "green_median",
    "swir_median",
)

FEATURE_CHANNELS_V2_16: tuple[str, ...] = (
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

FEATURE_CHANNEL_PROFILES: dict[str, tuple[str, ...]] = {
    "v1_18ch": FEATURE_CHANNELS_V1_18,
    "v2_16ch": FEATURE_CHANNELS_V2_16,
}
FEATURE_PROFILE_BY_COUNT: dict[int, str] = {
    len(FEATURE_CHANNELS_V1_18): "v1_18ch",
    len(FEATURE_CHANNELS_V2_16): "v2_16ch",
}


def resolve_feature_profile(profile: str | None) -> str:
    token = str(profile or "v2_16ch").strip().lower()
    return token if token in FEATURE_CHANNEL_PROFILES else "v2_16ch"


def get_feature_channels(profile: str | None) -> tuple[str, ...]:
    return FEATURE_CHANNEL_PROFILES[resolve_feature_profile(profile)]


DEFAULT_FEATURE_PROFILE = resolve_feature_profile(os.getenv("ML_FEATURE_PROFILE", "v2_16ch"))
FEATURE_CHANNELS: tuple[str, ...] = get_feature_channels(DEFAULT_FEATURE_PROFILE)


if torch is not None:

    class ConvBlock(nn.Module):  # pragma: no cover - exercised via integration tests
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.block(x)


    class BoundaryUNet(nn.Module):  # pragma: no cover - exercised via integration tests
        """Lightweight multi-task U-Net with three heads: extent/boundary/distance."""

        def __init__(
            self,
            in_channels: int = len(FEATURE_CHANNELS),
            base_channels: int = 32,
            dropout_bottleneck: float = 0.0,
            dropout_dec4: float = 0.0,
            dropout_dec3: float = 0.0,
        ) -> None:
            super().__init__()
            ch = base_channels
            self.dropout_bottleneck = float(max(0.0, min(0.8, dropout_bottleneck)))
            self.dropout_dec4 = float(max(0.0, min(0.8, dropout_dec4)))
            self.dropout_dec3 = float(max(0.0, min(0.8, dropout_dec3)))

            self.enc1 = ConvBlock(in_channels, ch)
            self.enc2 = ConvBlock(ch, ch * 2)
            self.enc3 = ConvBlock(ch * 2, ch * 4)
            self.enc4 = ConvBlock(ch * 4, ch * 8)

            self.pool = nn.MaxPool2d(2)
            self.bottleneck = ConvBlock(ch * 8, ch * 16)

            self.up4 = nn.ConvTranspose2d(ch * 16, ch * 8, kernel_size=2, stride=2)
            self.dec4 = ConvBlock(ch * 16, ch * 8)

            self.up3 = nn.ConvTranspose2d(ch * 8, ch * 4, kernel_size=2, stride=2)
            self.dec3 = ConvBlock(ch * 8, ch * 4)

            self.up2 = nn.ConvTranspose2d(ch * 4, ch * 2, kernel_size=2, stride=2)
            self.dec2 = ConvBlock(ch * 4, ch * 2)

            self.up1 = nn.ConvTranspose2d(ch * 2, ch, kernel_size=2, stride=2)
            self.dec1 = ConvBlock(ch * 2, ch)

            self.head_extent = nn.Conv2d(ch, 1, kernel_size=1)
            self.head_boundary = nn.Conv2d(ch, 1, kernel_size=1)
            self.head_distance = nn.Conv2d(ch, 1, kernel_size=1)

        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            e4 = self.enc4(self.pool(e3))
            b = self.bottleneck(self.pool(e4))
            if self.dropout_bottleneck > 0:
                b = nn.functional.dropout2d(b, p=self.dropout_bottleneck, training=self.training)

            d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
            if self.dropout_dec4 > 0:
                d4 = nn.functional.dropout2d(d4, p=self.dropout_dec4, training=self.training)
            d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
            if self.dropout_dec3 > 0:
                d3 = nn.functional.dropout2d(d3, p=self.dropout_dec3, training=self.training)
            d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

            return {
                "extent": self.head_extent(d1),
                "boundary": self.head_boundary(d1),
                "distance": self.head_distance(d1),
            }

else:

    class BoundaryUNet:  # pragma: no cover - executed only without torch
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("PyTorch is required for BoundaryUNet")


class FieldBoundaryInferencer:
    """Unified interface for BoundaryUNet inference (PyTorch + ONNX)."""

    def __init__(
        self,
        model_path: str,
        *,
        norm_stats_path: str | None = None,
        device: str = "auto",
        use_onnx: bool = False,
        feature_profile: str | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.norm_stats_path = Path(norm_stats_path) if norm_stats_path else None
        self._use_onnx_flag = use_onnx
        self.device = self._resolve_device(device)
        self.backend = "torch"
        self._torch_model: Any | None = None
        self._onnx_session: Any | None = None
        self._input_name: str | None = None
        self._output_names: list[str] | None = None
        self.metadata: dict[str, Any] = {}
        self.feature_profile = resolve_feature_profile(feature_profile or DEFAULT_FEATURE_PROFILE)
        self.feature_channels = get_feature_channels(self.feature_profile)
        self._norm_mean = np.zeros(len(self.feature_channels), dtype=np.float32)
        self._norm_std = np.ones(len(self.feature_channels), dtype=np.float32)

        self._preload_metadata_profile()
        self._load_norm_stats()
        self._load_model()
        self._load_model_metadata()

    @staticmethod
    def _resolve_device(device: str) -> str:
        mode = str(device or "auto").lower()
        if mode in {"cpu", "cuda"}:
            if mode == "cuda" and torch is not None and not torch.cuda.is_available():
                return "cpu"
            return mode
        if torch is None:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load_norm_stats(self) -> None:
        candidate_paths: list[Path] = []
        if self.norm_stats_path is not None:
            candidate_paths.append(self.norm_stats_path)
        candidate_paths.append(self.model_path.with_suffix(".norm.json"))

        for path in candidate_paths:
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                mean = np.asarray(payload.get("mean", []), dtype=np.float32)
                std = np.asarray(payload.get("std", []), dtype=np.float32)
                if mean.size != std.size:
                    continue
                inferred_profile = FEATURE_PROFILE_BY_COUNT.get(int(mean.size))
                if inferred_profile and inferred_profile != self.feature_profile:
                    self._set_feature_profile(inferred_profile)
                if mean.size == len(self.feature_channels):
                    self._norm_mean = mean
                    self._norm_std = np.clip(std, 1e-6, None)
                    logger.info("ml_norm_stats_loaded", path=str(path), channels=int(mean.size))
                    return
                else:
                    raise ValueError(
                        f"Norm stats channel mismatch in {path}: "
                        f"stats has {mean.size} channels but model expects "
                        f"{len(self.feature_channels)} ({self.feature_profile})"
                    )
            except Exception as exc:
                logger.warning("ml_norm_stats_load_failed", path=str(path), error=str(exc))

    def _set_feature_profile(self, feature_profile: str) -> None:
        resolved = resolve_feature_profile(feature_profile)
        if resolved == self.feature_profile:
            return
        self.feature_profile = resolved
        self.feature_channels = get_feature_channels(resolved)
        self._norm_mean = np.zeros(len(self.feature_channels), dtype=np.float32)
        self._norm_std = np.ones(len(self.feature_channels), dtype=np.float32)

    def _preload_metadata_profile(self) -> None:
        candidates = [
            Path(str(self.model_path) + ".meta.json"),
            self.model_path.with_suffix(".meta.json"),
        ]
        for path in candidates:
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    continue
                profile = payload.get("feature_profile")
                if isinstance(profile, str):
                    self._set_feature_profile(profile)
                    return
                channels = payload.get("channels")
                if isinstance(channels, list):
                    inferred_profile = FEATURE_PROFILE_BY_COUNT.get(len(channels))
                    if inferred_profile:
                        self._set_feature_profile(inferred_profile)
                        return
            except Exception:
                continue

    def _load_model(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"ML model not found: {self.model_path}")

        suffix = self.model_path.suffix.lower()
        # Honour ML_USE_ONNX flag: if True and ONNX sibling exists, prefer it
        if suffix != ".onnx" and self._use_onnx_flag:
            onnx_sibling = self.model_path.with_suffix(".onnx")
            if onnx_sibling.exists():
                logger.info("ml_use_onnx_override", original=str(self.model_path), resolved=str(onnx_sibling))
                self.model_path = onnx_sibling
                suffix = ".onnx"
        if suffix == ".onnx":
            if ort is None:
                raise RuntimeError("onnxruntime is not installed, but ONNX model was provided")
            providers: list[Any] = ["CPUExecutionProvider"]
            if self.device == "cuda":
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            # Use multiple threads for CPU inference (especially beneficial on multi-core CPUs)
            import os as _os
            _n_cpu = _os.cpu_count() or 4
            _intra_threads = max(2, min(_n_cpu, 8))  # Cap at 8 to avoid contention
            sess_opts.intra_op_num_threads = _intra_threads
            sess_opts.inter_op_num_threads = max(1, _intra_threads // 2)
            sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            self._onnx_session = ort.InferenceSession(
                str(self.model_path),
                sess_options=sess_opts,
                providers=providers,
            )
            self._input_name = self._onnx_session.get_inputs()[0].name
            self._output_names = [out.name for out in self._onnx_session.get_outputs()]
            input_shape = self._onnx_session.get_inputs()[0].shape
            if len(input_shape) >= 2 and isinstance(input_shape[1], int):
                inferred_profile = FEATURE_PROFILE_BY_COUNT.get(int(input_shape[1]))
                if inferred_profile:
                    self._set_feature_profile(inferred_profile)
                    self._load_norm_stats()
            self.backend = "onnx"
            logger.info(
                "ml_model_loaded",
                backend="onnx",
                path=str(self.model_path),
                device=self.device,
                feature_profile=self.feature_profile,
            )
            return

        if torch is None:
            raise RuntimeError("PyTorch is not installed, cannot load .pth model")

        payload = torch.load(self.model_path, map_location=self.device)
        state_dict = payload
        if isinstance(payload, dict):
            if "model_state" in payload:
                state_dict = payload["model_state"]
            elif "state_dict" in payload:
                state_dict = payload["state_dict"]
            if isinstance(state_dict, dict):
                first_conv = state_dict.get("enc1.block.0.weight")
                if first_conv is not None and hasattr(first_conv, "shape") and len(first_conv.shape) >= 2:
                    inferred_profile = FEATURE_PROFILE_BY_COUNT.get(int(first_conv.shape[1]))
                    if inferred_profile:
                        self._set_feature_profile(inferred_profile)
                        self._load_norm_stats()
            norm = payload.get("norm_stats")
            if isinstance(norm, dict):
                mean = np.asarray(norm.get("mean", []), dtype=np.float32)
                std = np.asarray(norm.get("std", []), dtype=np.float32)
                if mean.size == len(self.feature_channels) and std.size == len(self.feature_channels):
                    self._norm_mean = mean
                    self._norm_std = np.clip(std, 1e-6, None)

        model = BoundaryUNet(in_channels=len(self.feature_channels))
        # Strict loading prevents silent architecture drift between training and inference.
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        model.to(self.device)

        self._torch_model = model
        self.backend = "torch"
        logger.info(
            "ml_model_loaded",
            backend="torch",
            path=str(self.model_path),
            device=self.device,
            feature_profile=self.feature_profile,
        )

    def _load_model_metadata(self) -> None:
        candidates = [
            Path(str(self.model_path) + ".meta.json"),
            self.model_path.with_suffix(".meta.json"),
        ]
        for path in candidates:
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    self.metadata = payload
                    logger.info("ml_model_metadata_loaded", path=str(path))
                    return
            except Exception as exc:
                logger.info("ml_model_metadata_load_failed", path=str(path), error=str(exc))

    @staticmethod
    def _emit_progress(
        progress_callback: ProgressCallback | None,
        stage: str,
        completed: int,
        total: int,
    ) -> None:
        if progress_callback is None:
            return
        safe_total = max(int(total), 1)
        safe_completed = min(max(int(completed), 0), safe_total)
        try:
            progress_callback(str(stage), safe_completed, safe_total)
        except Exception as exc:
            logger.warning(
                "ml_progress_callback_failed",
                stage=str(stage),
                completed=safe_completed,
                total=safe_total,
                error=str(exc),
            )

    def _default_patch_batch_size(self) -> int:
        if self.backend == "onnx":
            return 8 if self.device == "cpu" else 12
        if self.device == "cuda":
            return 8
        return 4

    def _normalize(self, feature_stack: np.ndarray) -> np.ndarray:
        arr = np.asarray(feature_stack, dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError(f"feature_stack must have shape (C,H,W), got {arr.shape}")
        return self._normalize_batch(arr[np.newaxis, ...])[0]

    def _normalize_batch(self, feature_stacks: np.ndarray) -> np.ndarray:
        arr = np.asarray(feature_stacks, dtype=np.float32)
        if arr.ndim != 4:
            raise ValueError(f"feature_stacks must have shape (N,C,H,W), got {arr.shape}")
        if arr.shape[1] != len(self.feature_channels):
            raise ValueError(
                f"feature_stack channel count must be {len(self.feature_channels)}, got {arr.shape[1]}"
            )
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return (arr - self._norm_mean[None, :, None, None]) / self._norm_std[None, :, None, None]

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _normalize_distance(raw: np.ndarray) -> np.ndarray:
        """Normalize distance prediction to [0, 1] matching training convention."""
        d = np.clip(raw, 0.0, None)
        d_max = float(d.max())
        if d_max < 1e-6:
            return np.zeros_like(d)
        return np.clip(d / d_max, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _quality_score(extent: np.ndarray, boundary: np.ndarray, distance: np.ndarray) -> float:
        area_ratio = float(np.mean(extent > 0.5)) if extent.size else 0.0
        if area_ratio <= 1e-4:
            return 0.0
        boundary_p95 = float(np.nanpercentile(boundary, 95)) if boundary.size else 0.0
        distance_p95 = float(np.nanpercentile(distance, 95)) if distance.size else 0.0
        area_term = min(1.0, area_ratio * 12.0)
        return float(np.clip(0.45 * boundary_p95 + 0.35 * area_term + 0.20 * distance_p95, 0.0, 1.0))

    def _run_model_batch(self, normalized_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.backend == "onnx":
            assert self._onnx_session is not None and self._input_name is not None
            outputs = self._onnx_session.run(
                self._output_names,
                {self._input_name: normalized_batch.astype(np.float32)},
            )
            if len(outputs) == 3:
                extent_raw, boundary_raw, distance_raw = outputs
            else:
                merged = outputs[0]
                if merged.shape[1] < 3:
                    raise RuntimeError(f"Unexpected ONNX output shape: {merged.shape}")
                extent_raw = merged[:, 0:1]
                boundary_raw = merged[:, 1:2]
                distance_raw = merged[:, 2:3]
            extent = self._sigmoid(extent_raw[:, 0]).astype(np.float32)
            boundary = self._sigmoid(boundary_raw[:, 0]).astype(np.float32)
            distance = np.stack(
                [self._normalize_distance(distance_raw[idx, 0]) for idx in range(distance_raw.shape[0])],
                axis=0,
            ).astype(np.float32)
            return extent, boundary, distance

        assert torch is not None and self._torch_model is not None
        with torch.no_grad():
            tensor = torch.from_numpy(normalized_batch).to(self.device).float()
            out = self._torch_model(tensor)
            extent = torch.sigmoid(out["extent"]).detach().cpu().numpy()[:, 0].astype(np.float32)
            boundary = torch.sigmoid(out["boundary"]).detach().cpu().numpy()[:, 0].astype(np.float32)
            distance_raw = out["distance"].detach().cpu().numpy()[:, 0]
            distance = np.stack(
                [self._normalize_distance(distance_raw[idx]) for idx in range(distance_raw.shape[0])],
                axis=0,
            ).astype(np.float32)
            return extent, boundary, distance

    def _predict_patches_batch(
        self,
        patches: np.ndarray,
        *,
        batch_size: int | None = None,
        progress_callback: ProgressCallback | None = None,
        progress_stage: str = "ml_patch_infer",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        arr = np.asarray(patches, dtype=np.float32)
        if arr.ndim != 4:
            raise ValueError(f"patches must have shape (N,C,H,W), got {arr.shape}")
        n = int(arr.shape[0])
        if n == 0:
            raise ValueError("patch batch is empty")
        normalized = self._normalize_batch(arr)

        if batch_size is None:
            batch_size = self._default_patch_batch_size()
        batch_size = max(1, int(batch_size))

        h, w = arr.shape[2], arr.shape[3]
        extent_out = np.empty((n, h, w), dtype=np.float32)
        boundary_out = np.empty((n, h, w), dtype=np.float32)
        distance_out = np.empty((n, h, w), dtype=np.float32)
        self._emit_progress(progress_callback, progress_stage, 0, n)
        for start in range(0, n, batch_size):
            stop = min(n, start + batch_size)
            extent_b, boundary_b, distance_b = self._run_model_batch(normalized[start:stop])
            extent_out[start:stop] = extent_b
            boundary_out[start:stop] = boundary_b
            distance_out[start:stop] = distance_b
            self._emit_progress(progress_callback, progress_stage, stop, n)
            del extent_b, boundary_b, distance_b

        del normalized
        return extent_out, boundary_out, distance_out

    def _predict_patch(self, patch: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        extent, boundary, distance = self._predict_patches_batch(
            np.asarray(patch, dtype=np.float32)[np.newaxis, ...],
            batch_size=1,
        )
        return extent[0], boundary[0], distance[0]

    @staticmethod
    def _tta_transforms(mode: str | None) -> tuple[str, ...]:
        token = str(mode or "none").strip().lower()
        if token in {"", "none", "off"}:
            return ("identity",)
        if token == "flip2":
            return ("identity", "flip_x")
        if token == "flip4":
            return ("identity", "flip_x", "flip_y", "flip_xy")
        if token == "rotate4":
            return ("identity", "rot90", "rot180", "rot270")
        return ("identity",)

    @staticmethod
    def _apply_transform(arr: np.ndarray, transform: str) -> np.ndarray:
        if transform == "identity":
            return arr
        if transform == "flip_x":
            return np.flip(arr, axis=-1)
        if transform == "flip_y":
            return np.flip(arr, axis=-2)
        if transform == "flip_xy":
            return np.flip(np.flip(arr, axis=-1), axis=-2)
        if transform == "rot90":
            return np.rot90(arr, k=1, axes=(-2, -1))
        if transform == "rot180":
            return np.rot90(arr, k=2, axes=(-2, -1))
        if transform == "rot270":
            return np.rot90(arr, k=3, axes=(-2, -1))
        raise ValueError(f"Unsupported TTA transform: {transform}")

    @classmethod
    def _invert_transform(cls, arr: np.ndarray, transform: str) -> np.ndarray:
        inverse_map = {
            "identity": "identity",
            "flip_x": "flip_x",
            "flip_y": "flip_y",
            "flip_xy": "flip_xy",
            "rot90": "rot270",
            "rot180": "rot180",
            "rot270": "rot90",
        }
        return cls._apply_transform(arr, inverse_map[transform])

    @staticmethod
    def _resize_batch(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
        if arr.shape[-2:] == (out_h, out_w):
            return arr.astype(np.float32, copy=False)
        if arr.ndim == 4:
            zoom_factors = (1.0, 1.0, float(out_h) / arr.shape[-2], float(out_w) / arr.shape[-1])
        elif arr.ndim == 3:
            zoom_factors = (1.0, float(out_h) / arr.shape[-2], float(out_w) / arr.shape[-1])
        else:
            raise ValueError(f"Unsupported resize rank: {arr.ndim}")
        resized = nd_zoom(arr, zoom_factors, order=1, prefilter=False)
        resized = np.asarray(resized, dtype=np.float32)
        if resized.shape[-2:] == (out_h, out_w):
            return resized
        y = min(out_h, resized.shape[-2])
        x = min(out_w, resized.shape[-1])
        if arr.ndim == 4:
            fitted = np.zeros((resized.shape[0], resized.shape[1], out_h, out_w), dtype=np.float32)
            fitted[..., :y, :x] = resized[..., :y, :x]
        else:
            fitted = np.zeros((resized.shape[0], out_h, out_w), dtype=np.float32)
            fitted[..., :y, :x] = resized[..., :y, :x]
        return fitted

    @staticmethod
    def _masked_mean(arr: np.ndarray, mask: np.ndarray | None = None) -> float:
        work = np.asarray(arr, dtype=np.float32)
        if mask is not None and np.any(mask):
            values = work[np.asarray(mask, dtype=bool)]
        else:
            values = work.reshape(-1)
        if values.size == 0:
            return 0.0
        return float(np.mean(values))

    def _summarize_tta_batch(
        self,
        extent_stack: np.ndarray,
        boundary_stack: np.ndarray,
        *,
        quality_scores: np.ndarray,
        transform_count: int,
    ) -> dict[str, np.ndarray | str]:
        sample_count = int(extent_stack.shape[1])
        geometry_confidence = np.empty(sample_count, dtype=np.float32)
        tta_consensus = np.empty(sample_count, dtype=np.float32)
        boundary_uncertainty = np.empty(sample_count, dtype=np.float32)
        extent_disagreement = np.empty(sample_count, dtype=np.float32)
        boundary_disagreement = np.empty(sample_count, dtype=np.float32)

        if transform_count <= 1:
            geometry_confidence[:] = np.clip(quality_scores, 0.0, 1.0)
            tta_consensus[:] = np.nan
            boundary_uncertainty[:] = np.nan
            extent_disagreement[:] = 0.0
            boundary_disagreement[:] = 0.0
            return {
                "geometry_confidence": geometry_confidence,
                "tta_consensus": tta_consensus,
                "boundary_uncertainty": boundary_uncertainty,
                "tta_extent_disagreement": extent_disagreement,
                "tta_boundary_disagreement": boundary_disagreement,
                "uncertainty_source": "tta_unavailable",
            }

        extent_mean = np.mean(extent_stack, axis=0, dtype=np.float32)
        boundary_mean = np.mean(boundary_stack, axis=0, dtype=np.float32)

        for idx in range(sample_count):
            extent_mad_map = np.mean(
                np.abs(extent_stack[:, idx] - extent_mean[idx][None, ...]),
                axis=0,
                dtype=np.float32,
            )
            boundary_mad_map = np.mean(
                np.abs(boundary_stack[:, idx] - boundary_mean[idx][None, ...]),
                axis=0,
                dtype=np.float32,
            )
            extent_support = extent_mean[idx] >= 0.5
            boundary_support = boundary_mean[idx] >= 0.35
            extent_mad = self._masked_mean(extent_mad_map, extent_support)
            boundary_mad = self._masked_mean(boundary_mad_map, boundary_support)
            disagreement = min(1.0, max(0.0, (0.4 * extent_mad + 0.6 * boundary_mad) / 0.25))
            consensus = float(np.clip(1.0 - disagreement, 0.0, 1.0))
            boundary_unc = float(np.clip(boundary_mad / 0.25, 0.0, 1.0))
            geometry = float(
                np.clip(0.55 * float(quality_scores[idx]) + 0.45 * consensus, 0.0, 1.0)
            )
            extent_disagreement[idx] = float(np.clip(extent_mad, 0.0, 1.0))
            boundary_disagreement[idx] = float(np.clip(boundary_mad, 0.0, 1.0))
            tta_consensus[idx] = consensus
            boundary_uncertainty[idx] = boundary_unc
            geometry_confidence[idx] = geometry

        return {
            "geometry_confidence": geometry_confidence,
            "tta_consensus": tta_consensus,
            "boundary_uncertainty": boundary_uncertainty,
            "tta_extent_disagreement": extent_disagreement,
            "tta_boundary_disagreement": boundary_disagreement,
            "uncertainty_source": "tta_disagreement",
        }

    def _predict_single_scale_with_tta(
        self,
        arr: np.ndarray,
        *,
        tile_size: int,
        overlap: int,
        tta_mode: str = "none",
        progress_callback: ProgressCallback | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray | str]]:
        transforms = self._tta_transforms(tta_mode)
        extent_acc = np.zeros((arr.shape[0], arr.shape[2], arr.shape[3]), dtype=np.float32)
        boundary_acc = np.zeros_like(extent_acc)
        distance_acc = np.zeros_like(extent_acc)
        extent_outputs: list[np.ndarray] = []
        boundary_outputs: list[np.ndarray] = []

        self._emit_progress(progress_callback, "ml_tta", 0, len(transforms))
        for transform_index, transform in enumerate(transforms, start=1):
            transformed = self._apply_transform(arr, transform)
            extent_t, boundary_t, distance_t = self._predict_spatial_batch(
                transformed,
                tile_size=tile_size,
                overlap=overlap,
                progress_callback=progress_callback,
            )
            extent_inv = self._invert_transform(extent_t, transform).astype(np.float32, copy=False)
            boundary_inv = self._invert_transform(boundary_t, transform).astype(np.float32, copy=False)
            distance_inv = self._invert_transform(distance_t, transform).astype(np.float32, copy=False)
            extent_outputs.append(extent_inv)
            boundary_outputs.append(boundary_inv)
            extent_acc += extent_inv
            boundary_acc += boundary_inv
            distance_acc += distance_inv
            self._emit_progress(progress_callback, "ml_tta", transform_index, len(transforms))

        divisor = float(max(len(transforms), 1))
        extent_mean = (extent_acc / divisor).astype(np.float32)
        boundary_mean = (boundary_acc / divisor).astype(np.float32)
        distance_mean = (distance_acc / divisor).astype(np.float32)
        quality_scores = np.asarray(
            [self._quality_score(extent_mean[idx], boundary_mean[idx], distance_mean[idx]) for idx in range(arr.shape[0])],
            dtype=np.float32,
        )
        tta_stats = self._summarize_tta_batch(
            np.stack(extent_outputs, axis=0),
            np.stack(boundary_outputs, axis=0),
            quality_scores=quality_scores,
            transform_count=len(transforms),
        )
        return extent_mean, boundary_mean, distance_mean, tta_stats

    @staticmethod
    def _tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
        starts = list(range(0, max(1, length - tile_size + 1), stride))
        last = max(0, length - tile_size)
        if starts[-1] != last:
            starts.append(last)
        return starts

    _gauss_cache: dict[tuple[int, int], np.ndarray] = {}

    @classmethod
    def _gaussian_window(cls, height: int, width: int) -> np.ndarray:
        key = (height, width)
        cached = cls._gauss_cache.get(key)
        if cached is not None:
            return cached
        sigma_y = max(1.0, 0.25 * float(height))
        sigma_x = max(1.0, 0.25 * float(width))
        y = np.arange(height, dtype=np.float32) - (height - 1) / 2.0
        x = np.arange(width, dtype=np.float32) - (width - 1) / 2.0
        yy, xx = np.meshgrid(y, x, indexing="ij")
        window = np.exp(-0.5 * ((yy / sigma_y) ** 2 + (xx / sigma_x) ** 2))
        result = np.clip(window.astype(np.float32), 1e-3, None)
        del yy, xx, y, x, window
        cls._gauss_cache[key] = result
        return result

    def _predict_spatial_batch(
        self,
        arr: np.ndarray,
        *,
        tile_size: int,
        overlap: int,
        progress_callback: ProgressCallback | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if arr.ndim != 4:
            raise ValueError(f"feature_stacks must have shape (N,C,H,W), got {arr.shape}")
        n, _, h, w = arr.shape
        if n == 0:
            raise ValueError("feature_stacks must not be empty")

        stride = max(1, tile_size - overlap)
        if h <= tile_size and w <= tile_size:
            return self._predict_patches_batch(
                arr,
                batch_size=self._default_patch_batch_size(),
                progress_callback=progress_callback,
            )

        y_starts = self._tile_starts(h, tile_size, stride)
        x_starts = self._tile_starts(w, tile_size, stride)
        windows = self._gaussian_window(tile_size, tile_size)

        mappings: list[tuple[int, int, int, int, int]] = []
        for sample_idx in range(n):
            for y0 in y_starts:
                y1 = min(h, y0 + tile_size)
                for x0 in x_starts:
                    x1 = min(w, x0 + tile_size)
                    patch_shape = arr[sample_idx, :, y0:y1, x0:x1].shape
                    if patch_shape[1] != tile_size or patch_shape[2] != tile_size:
                        raise ValueError(
                            f"Patch shape mismatch: expected {(tile_size, tile_size)}, got {patch_shape[1:]}"
                        )
                    mappings.append((sample_idx, y0, y1, x0, x1))

        extent_acc = np.zeros((n, h, w), dtype=np.float32)
        boundary_acc = np.zeros((n, h, w), dtype=np.float32)
        distance_acc = np.zeros((n, h, w), dtype=np.float32)
        weight_acc = np.zeros((n, h, w), dtype=np.float32)

        patch_batch_size = self._default_patch_batch_size()
        total_patches = max(len(mappings), 1)
        self._emit_progress(progress_callback, "ml_patches", 0, total_patches)
        self._emit_progress(progress_callback, "ml_blend", 0, total_patches)
        for start in range(0, len(mappings), patch_batch_size):
            stop = min(len(mappings), start + patch_batch_size)
            chunk_mappings = mappings[start:stop]
            patch_arr = np.stack(
                [
                    arr[sample_idx, :, y0:y1, x0:x1]
                    for sample_idx, y0, y1, x0, x1 in chunk_mappings
                ],
                axis=0,
            ).astype(np.float32)
            extent_p, boundary_p, distance_p = self._predict_patches_batch(
                patch_arr,
                batch_size=patch_batch_size,
            )
            self._emit_progress(progress_callback, "ml_patches", stop, total_patches)

            for idx, (sample_idx, y0, y1, x0, x1) in enumerate(chunk_mappings):
                window = windows[: y1 - y0, : x1 - x0]
                extent_acc[sample_idx, y0:y1, x0:x1] += extent_p[idx] * window
                boundary_acc[sample_idx, y0:y1, x0:x1] += boundary_p[idx] * window
                distance_acc[sample_idx, y0:y1, x0:x1] += distance_p[idx] * window
                weight_acc[sample_idx, y0:y1, x0:x1] += window

            self._emit_progress(progress_callback, "ml_blend", stop, total_patches)
            del patch_arr, extent_p, boundary_p, distance_p

        weight_acc = np.maximum(weight_acc, 1e-6)
        return (
            (extent_acc / weight_acc).astype(np.float32),
            (boundary_acc / weight_acc).astype(np.float32),
            (distance_acc / weight_acc).astype(np.float32),
        )

    def predict(
        self,
        feature_stack: np.ndarray,
        tile_size: int = 512,
        overlap: int = 128,
        *,
        tta_mode: str = "none",
        scales: tuple[float, ...] | list[float] | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        """Run overlap-tile inference and return probabilities in [0,1]."""
        arr = np.asarray(feature_stack, dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError(f"feature_stack must have shape (C,H,W), got {arr.shape}")
        n_channels = arr.shape[0]
        expected_channels = len(self.feature_channels)
        if n_channels != expected_channels:
            raise ValueError(
                f"Feature channel mismatch: input has {n_channels} channels "
                f"but model expects {expected_channels} ({self.feature_profile}). "
                f"Expected channels: {self.feature_channels}"
            )
        result = self.predict_batch(
            arr[np.newaxis, ...],
            tile_size=tile_size,
            overlap=overlap,
            tta_mode=tta_mode,
            scales=scales,
            progress_callback=progress_callback,
        )
        return result[0]

    def predict_batch(
        self,
        feature_stacks: np.ndarray,
        tile_size: int = 512,
        overlap: int = 128,
        *,
        tta_mode: str = "none",
        scales: tuple[float, ...] | list[float] | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> list[dict[str, Any]]:
        """Run true batched inference over N stacks with overlap-tile blending."""
        arr = np.asarray(feature_stacks, dtype=np.float32)
        if arr.ndim != 4:
            raise ValueError(f"feature_stacks must have shape (N,C,H,W), got {arr.shape}")
        tile_size = max(32, int(tile_size))
        overlap = max(0, int(overlap))
        scale_values = [float(scale) for scale in (scales or (1.0,)) if float(scale) > 0]
        if not scale_values:
            scale_values = [1.0]
        if 1.0 not in scale_values:
            scale_values.insert(0, 1.0)

        extent_acc = np.zeros((arr.shape[0], arr.shape[2], arr.shape[3]), dtype=np.float32)
        boundary_acc = np.zeros_like(extent_acc)
        distance_acc = np.zeros_like(extent_acc)
        tta_consensus_acc = np.zeros(arr.shape[0], dtype=np.float32)
        boundary_uncertainty_acc = np.zeros(arr.shape[0], dtype=np.float32)
        geometry_confidence_acc = np.zeros(arr.shape[0], dtype=np.float32)
        extent_disagreement_acc = np.zeros(arr.shape[0], dtype=np.float32)
        boundary_disagreement_acc = np.zeros(arr.shape[0], dtype=np.float32)
        uncertainty_sources: list[str] = []
        total_weight = 0.0

        self._emit_progress(progress_callback, "ml_scale", 0, len(scale_values))
        for scale_index, scale in enumerate(scale_values, start=1):
            scaled_h = max(32, int(round(arr.shape[2] * scale)))
            scaled_w = max(32, int(round(arr.shape[3] * scale)))
            scaled_arr = self._resize_batch(arr, scaled_h, scaled_w) if scale != 1.0 else arr
            scaled_tile_size = min(tile_size, scaled_h, scaled_w)
            scaled_overlap = min(overlap, max(0, scaled_tile_size // 2))
            extent_s, boundary_s, distance_s, tta_stats = self._predict_single_scale_with_tta(
                scaled_arr,
                tile_size=scaled_tile_size,
                overlap=scaled_overlap,
                tta_mode=tta_mode,
                progress_callback=progress_callback,
            )
            if scale != 1.0:
                extent_s = self._resize_batch(extent_s, arr.shape[2], arr.shape[3])
                boundary_s = self._resize_batch(boundary_s, arr.shape[2], arr.shape[3])
                distance_s = self._resize_batch(distance_s, arr.shape[2], arr.shape[3])
            extent_acc += extent_s
            boundary_acc += boundary_s
            distance_acc += distance_s
            tta_consensus_acc += np.nan_to_num(
                np.asarray(tta_stats.get("tta_consensus"), dtype=np.float32),
                nan=0.0,
            )
            boundary_uncertainty_acc += np.nan_to_num(
                np.asarray(tta_stats.get("boundary_uncertainty"), dtype=np.float32),
                nan=0.0,
            )
            geometry_confidence_acc += np.nan_to_num(
                np.asarray(tta_stats.get("geometry_confidence"), dtype=np.float32),
                nan=0.0,
            )
            extent_disagreement_acc += np.nan_to_num(
                np.asarray(tta_stats.get("tta_extent_disagreement"), dtype=np.float32),
                nan=0.0,
            )
            boundary_disagreement_acc += np.nan_to_num(
                np.asarray(tta_stats.get("tta_boundary_disagreement"), dtype=np.float32),
                nan=0.0,
            )
            source_value = str(tta_stats.get("uncertainty_source") or "").strip()
            if source_value:
                uncertainty_sources.append(source_value)
            total_weight += 1.0
            self._emit_progress(progress_callback, "ml_scale", scale_index, len(scale_values))

        extent = (extent_acc / max(total_weight, 1.0)).astype(np.float32)
        boundary = (boundary_acc / max(total_weight, 1.0)).astype(np.float32)
        distance = (distance_acc / max(total_weight, 1.0)).astype(np.float32)
        tta_consensus = (tta_consensus_acc / max(total_weight, 1.0)).astype(np.float32)
        boundary_uncertainty = (boundary_uncertainty_acc / max(total_weight, 1.0)).astype(np.float32)
        geometry_confidence = (geometry_confidence_acc / max(total_weight, 1.0)).astype(np.float32)
        extent_disagreement = (extent_disagreement_acc / max(total_weight, 1.0)).astype(np.float32)
        boundary_disagreement = (boundary_disagreement_acc / max(total_weight, 1.0)).astype(np.float32)
        tta_enabled = len(self._tta_transforms(tta_mode)) > 1
        uncertainty_source = "tta_disagreement" if tta_enabled else "tta_unavailable"
        if uncertainty_sources and len(set(uncertainty_sources)) == 1:
            uncertainty_source = uncertainty_sources[0]

        results: list[dict[str, Any]] = []
        for idx in range(arr.shape[0]):
            e = extent[idx]
            b = boundary[idx]
            d = distance[idx]
            score = self._quality_score(e, b, d)
            consensus_value = (
                round(float(tta_consensus[idx]), 4)
                if tta_enabled
                else None
            )
            boundary_uncertainty_value = (
                round(float(boundary_uncertainty[idx]), 4)
                if tta_enabled
                else None
            )
            geometry_confidence_value = round(
                float(geometry_confidence[idx]) if tta_enabled else float(score),
                4,
            )
            results.append(
                {
                    "extent": e.astype(np.float32),
                    "boundary": b.astype(np.float32),
                    "distance": d.astype(np.float32),
                    "score": score,
                    "tta_consensus": consensus_value,
                    "boundary_uncertainty": boundary_uncertainty_value,
                    "geometry_confidence": geometry_confidence_value,
                    "tta_extent_disagreement": round(float(extent_disagreement[idx]), 4),
                    "tta_boundary_disagreement": round(float(boundary_disagreement[idx]), 4),
                    "tta_transform_count": len(self._tta_transforms(tta_mode)),
                    "uncertainty_source": uncertainty_source,
                }
            )
        return results
