import pytest

from core.celery_app import _validate_worker_settings, celery


class _BadSettingsLoader:
    def cache_clear(self):
        return None

    def __call__(self):
        raise ValueError("invalid env")


def test_validate_worker_settings_raises_on_invalid_settings(monkeypatch):
    monkeypatch.setattr("core.settings.get_settings", _BadSettingsLoader())
    with pytest.raises(RuntimeError, match="Invalid runtime settings"):
        _validate_worker_settings("worker_init")


def test_celery_rejects_lost_worker_tasks():
    assert celery.conf.task_acks_late is True
    assert celery.conf.task_reject_on_worker_lost is True
