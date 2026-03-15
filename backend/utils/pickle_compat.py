"""Совместимая загрузка pickle-артефактов sklearn/numpy."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


def _patch_numpy_bit_generators() -> None:
    """Исправить старые pickle с numpy BitGenerator, сохранённые в другой версии numpy."""
    try:
        import numpy.random._pickle as np_pickle  # type: ignore[attr-defined]
        from numpy.random import MT19937, PCG64, PCG64DXSM, Philox, SFC64
    except Exception:
        return

    original_ctor = getattr(np_pickle, "__bit_generator_ctor", None)
    if original_ctor is None or getattr(original_ctor, "__agrovision_patched__", False):
        return

    aliases = {
        "MT19937": MT19937,
        "PCG64": PCG64,
        "PCG64DXSM": PCG64DXSM,
        "Philox": Philox,
        "SFC64": SFC64,
        str(MT19937): MT19937,
        str(PCG64): PCG64,
        str(PCG64DXSM): PCG64DXSM,
        str(Philox): Philox,
        str(SFC64): SFC64,
    }

    def compat_ctor(bit_generator_name: str | type[Any]) -> Any:
        normalized = aliases.get(str(bit_generator_name))
        if normalized is not None:
            return normalized()
        return original_ctor(bit_generator_name)

    compat_ctor.__agrovision_patched__ = True  # type: ignore[attr-defined]
    np_pickle.__bit_generator_ctor = compat_ctor


def load_pickle_compat(path: str | Path) -> Any:
    """Загрузить pickle-артефакт с мягкой совместимостью по numpy RNG."""
    _patch_numpy_bit_generators()
    resolved_path = Path(path)
    with resolved_path.open("rb") as handle:
        return pickle.load(handle)
