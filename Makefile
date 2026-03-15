PYTHON := ./.venv/bin/python
PYTEST := PYTHONPATH="$(PWD)/backend:$$PYTHONPATH" $(PYTHON) -m pytest

.PHONY: preflight test test-full frontend-build docker-config up up-d local-api local-worker local-beat reset-state fresh-start audit-full release-smoke detect-benchmark-ru benchmark-north-south train-orchestrated train-promote-if-green train-open-data-download train-boundary-v3-cpu train-yield-corpus train-yield-baseline train-yield-ensemble install-training-deps train-import-data train-all-in-one init-training-inputs

preflight:
	$(PYTHON) scripts/preflight_check.py

test:
	$(PYTEST) backend/tests/test_weather_service.py \
		backend/tests/test_yield_service.py \
		backend/tests/test_archive_service.py \
		backend/tests/test_modeling_service.py \
		backend/tests/test_production_routes.py \
		backend/tests/test_smoke_runtime.py \
		backend/tests/test_api.py \
		backend/tests/test_celery_app.py \
		backend/tests/test_config.py -q

test-full:
	$(PYTEST) -q

frontend-build:
	cd frontend && npm run build

docker-config:
	docker compose config

up:
	docker compose up --build

up-d:
	docker compose up --build -d

reset-state:
	./scripts/reset_runtime_state.sh

fresh-start: reset-state up-d

audit-full:
	./scripts/audit_full.sh

release-smoke:
	$(PYTHON) scripts/release_smoke.py

detect-benchmark-ru:
	./scripts/detect_benchmark_ru.sh

benchmark-north-south:
	./scripts/benchmark_north_south.sh

train-orchestrated:
	./scripts/train_orchestrated.sh

train-promote-if-green:
	./scripts/train_promote_if_green.sh

train-open-data-download:
	./scripts/train_open_data_download.sh

train-boundary-v3-cpu:
	./scripts/train_boundary_v3_cpu.sh

train-yield-corpus:
	./scripts/train_yield_corpus.sh

train-yield-baseline:
	./scripts/train_yield_baseline.sh

train-yield-ensemble:
	./scripts/train_yield_ensemble.sh

install-training-deps:
	./scripts/install_training_deps.sh

train-import-data:
	./scripts/train_import_data.sh

init-training-inputs:
	./scripts/init_training_inputs.sh

train-all-in-one:
	DRY_RUN="$${DRY_RUN:-0}" ./scripts/install_training_deps.sh
	TRAINING_INPUTS_DIR="$${TRAINING_INPUTS_DIR:-$(PWD)/training_inputs}" \
	DRY_RUN="$${DRY_RUN:-0}" \
	./scripts/train_import_data.sh
	STAGES="$${STAGES:-download,prepare,train-boundary,train-yield-corpus,train-yield-baseline,train-yield-ensemble,benchmark,export,register}" \
	BATCHED="$${BATCHED:-1}" \
	LOW_MEM="$${LOW_MEM:-1}" \
	EXPERIMENTAL_LSTM="$${EXPERIMENTAL_LSTM:-0}" \
	BACKBONE="$${BACKBONE:-efficientnet_b0}" \
	DRY_RUN="$${DRY_RUN:-0}" \
	./scripts/train_orchestrated.sh

local-api:
	cd backend && PYTHONPATH="$(PWD)/backend:$$PYTHONPATH" ../.venv/bin/uvicorn main:app --reload --host 0.0.0.0 --port 8000

local-worker:
	cd backend && PYTHONPATH="$(PWD)/backend:$$PYTHONPATH" ../.venv/bin/celery -A core.celery_app:celery worker -l INFO -Q gpu,default --concurrency=1

local-beat:
	cd backend && PYTHONPATH="$(PWD)/backend:$$PYTHONPATH" ../.venv/bin/celery -A core.celery_app:celery beat -l INFO
