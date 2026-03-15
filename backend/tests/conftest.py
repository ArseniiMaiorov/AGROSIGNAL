import os
import sys
from collections.abc import Iterator

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.config import Settings, get_settings


def _iter_settings_env_keys() -> set[str]:
    keys: set[str] = set()
    for field_name, model_field in Settings.model_fields.items():
        keys.add(field_name)
        alias = getattr(model_field, 'validation_alias', None)
        if alias is None:
            continue
        choices = getattr(alias, 'choices', None)
        if choices is not None:
            for choice in choices:
                if isinstance(choice, str):
                    keys.add(choice)
            continue
        if isinstance(alias, str):
            keys.add(alias)
    return keys


_SETTINGS_ENV_KEYS = _iter_settings_env_keys()


@pytest.fixture(autouse=True)
def _isolate_settings_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    get_settings.cache_clear()
    for key in _SETTINGS_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    yield
    get_settings.cache_clear()
