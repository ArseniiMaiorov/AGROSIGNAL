"""Проверка совместимости сериализованного sklearn-классификатора."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from utils.pickle_compat import load_pickle_compat


def _extract_feature_names(payload: Any) -> list[str] | None:
    if isinstance(payload, dict):
        names = payload.get("feature_columns") or payload.get("feature_names")
        return list(names) if names is not None else None
    names = getattr(payload, "feature_columns", None) or getattr(payload, "feature_names", None)
    return list(names) if names is not None else None


def _extract_pipeline(payload: Any) -> Any | None:
    if isinstance(payload, dict):
        return payload.get("pipeline")
    return getattr(payload, "pipeline", None)


def _strip_hist_gradient_rng_state(root: Any) -> None:
    """Удалить непереносимое RNG-состояние sklearn HGB перед сериализацией.

    sklearn HistGradientBoosting* сохраняет в fitted estimator приватный
    `_feature_subsample_rng`, который на numpy 2.x сериализуется в формате,
    несовместимом с numpy 1.26.x. Для inference это состояние не требуется.
    """
    seen: set[int] = set()
    stack: list[Any] = [root]
    while stack:
        item = stack.pop()
        item_id = id(item)
        if item_id in seen:
            continue
        seen.add(item_id)

        if hasattr(item, "_feature_subsample_rng"):
            try:
                setattr(item, "_feature_subsample_rng", None)
            except Exception:
                pass

        if isinstance(item, dict):
            stack.extend(item.values())
            continue
        if isinstance(item, (list, tuple, set)):
            stack.extend(item)
            continue

        try:
            attrs = vars(item)
        except Exception:
            continue
        stack.extend(attrs.values())


def make_classifier_payload_portable(payload: Any) -> Any:
    """Подготовить classifier payload к межсредовой сериализации."""
    pipeline = _extract_pipeline(payload)
    if pipeline is not None:
        _strip_hist_gradient_rng_state(pipeline)
    return payload


def validate_classifier_payload(payload: Any, expected_feature_names: list[str] | tuple[str, ...]) -> dict[str, Any]:
    """Проверить, что артефакт согласован с ожидаемым пространством признаков."""
    pipeline = _extract_pipeline(payload)
    if pipeline is None:
        raise ValueError("В payload отсутствует pipeline")

    feature_names = _extract_feature_names(payload)
    if feature_names is None:
        raise ValueError("В payload отсутствует список признаков")

    expected = list(expected_feature_names)
    feature_count = len(feature_names)
    expected_count = len(expected)
    model_feature_count = getattr(pipeline, "n_features_in_", None)

    if feature_count != expected_count:
        raise ValueError(
            f"Размерность признаков в payload не совпадает с ожидаемой: {feature_count} != {expected_count}"
        )
    if model_feature_count is not None and int(model_feature_count) != expected_count:
        raise ValueError(
            f"Размерность pipeline не совпадает с ожидаемой: {model_feature_count} != {expected_count}"
        )
    if feature_names != expected:
        raise ValueError("Порядок признаков в payload отличается от ожидаемого")

    return {
        "feature_names": feature_names,
        "feature_count": feature_count,
        "pipeline_feature_count": int(model_feature_count) if model_feature_count is not None else None,
    }


def validate_classifier_file(path: Path | str, expected_feature_names: list[str] | tuple[str, ...]) -> dict[str, Any]:
    """Загрузить и проверить classifier pickle на диске."""
    resolved_path = Path(path)
    try:
        payload = load_pickle_compat(resolved_path)
    except Exception as exc:
        raise ValueError(f"Не удалось прочитать classifier pickle '{resolved_path.name}': {exc}") from exc
    return validate_classifier_payload(payload, expected_feature_names)
