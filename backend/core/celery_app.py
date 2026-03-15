import os

from celery import Celery
from celery.signals import worker_init, worker_process_init
from celery.schedules import crontab

from core.logging import get_logger

celery = Celery(
    "agromap",
    broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1"),
)

celery.conf.update(
    broker_connection_retry_on_startup=True,
    broker_transport_options={"priority_steps": [0, 3, 6, 9]},
    task_default_priority=5,
    task_routes={
        "tasks.autodetect.run_autodetect": {"queue": "gpu", "priority": 9},
        "tasks.archive.cleanup_expired": {"queue": "default", "priority": 3},
        "tasks.analytics.run_prediction": {"queue": "default", "priority": 6},
        "tasks.analytics.run_scenario": {"queue": "default", "priority": 6},
        "tasks.predictions.refresh_field_prediction": {"queue": "default", "priority": 6},
        "tasks.modeling.simulate_scenario_forward": {"queue": "default", "priority": 6},
        "tasks.model.train_global_residual_model": {"queue": "default", "priority": 3},
        "tasks.model.recalibrate_tenant_model": {"queue": "default", "priority": 3},
        "tasks.model.refresh_conformal_calibration": {"queue": "default", "priority": 3},
        "tasks.analytics.run_weekly_backfill": {"queue": "default", "priority": 3},
        "tasks.features.backfill_weekly_features": {"queue": "default", "priority": 3},
        "tasks.*": {"queue": "default"},
    },
    imports=(
        "tasks.autodetect",
        "tasks.archive",
        "tasks.analytics",
        "tasks.features",
        "tasks.model_tasks",
        "tasks.prediction_tasks",
    ),
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    task_time_limit=60 * 90,        # 90 min hard limit
    task_soft_time_limit=60 * 75,  # 75 min soft limit
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    beat_schedule={
        "cleanup-expired-archives-nightly": {
            "task": "tasks.archive.cleanup_expired",
            "schedule": crontab(hour=3, minute=15),
        },
    },
)

logger = get_logger(__name__)


def _inspect_or_none(*, timeout: float = 1.5):
    try:
        return celery.control.inspect(timeout=timeout)
    except Exception:
        return None


def live_workers_for_queue(queue_name: str, *, timeout: float = 1.5) -> list[str]:
    inspect = _inspect_or_none(timeout=timeout)
    if inspect is None:
        return []

    try:
        ping_result = inspect.ping() or {}
    except Exception:
        ping_result = {}
    if not isinstance(ping_result, dict) or not ping_result:
        return []

    live_workers = {str(worker) for worker in ping_result.keys()}

    try:
        active_queues = inspect.active_queues() or {}
    except Exception:
        active_queues = {}

    if not isinstance(active_queues, dict) or not active_queues:
        return sorted(live_workers)

    matched: set[str] = set()
    for worker_name, queues in active_queues.items():
        worker_id = str(worker_name)
        if worker_id not in live_workers:
            continue
        if not isinstance(queues, list):
            continue
        for queue_item in queues:
            if isinstance(queue_item, dict) and str(queue_item.get("name") or "") == queue_name:
                matched.add(worker_id)
                break
    return sorted(matched)


def has_live_workers_for_queue(queue_name: str, *, timeout: float = 1.5) -> bool:
    return bool(live_workers_for_queue(queue_name, timeout=timeout))


def _validate_worker_settings(stage: str) -> None:
    """Fail fast when worker env/config is invalid."""
    from core.settings import get_settings

    try:
        get_settings.cache_clear()
        get_settings()
    except Exception as exc:
        logger.error("celery_settings_validation_failed", stage=stage, error=str(exc), exc_info=True)
        raise RuntimeError(
            f"Invalid runtime settings during {stage}: {exc}"
        ) from exc


@worker_init.connect
def _on_worker_init(**_: object) -> None:
    _validate_worker_settings("worker_init")


@worker_process_init.connect
def _on_worker_process_init(**_: object) -> None:
    _validate_worker_settings("worker_process_init")
