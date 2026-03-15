"""Совместимый вход в конфигурацию приложения.

Этот модуль не меняет текущие импорты, а даёт единое место для новых
точек входа в settings/hygiene-слой.
"""
from core.config import (  # noqa: F401
    Settings,
    get_adaptive_pheno_thresholds,
    get_bool_env_alias_map,
    get_px_area_m2,
    get_settings,
    parse_env_bool,
)
