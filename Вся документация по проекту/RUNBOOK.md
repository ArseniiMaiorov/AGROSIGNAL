# RUNBOOK

## Быстрый предзапуск

```bash
make preflight
make test
make frontend-build
make docker-config
```

## Запуск в Docker

```bash
make up-d
```

Проверки:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/status
```

Интерфейс:

- Frontend: `http://localhost:5173`
- Backend API: `http://localhost:8000`

## Локальный запуск

Инфраструктура:

```bash
docker compose up -d postgres redis
```

API:

```bash
make local-api
```

Worker:

```bash
make local-worker
```

Beat:

```bash
make local-beat
```

Frontend:

```bash
cd frontend && npm run dev
```

## Сценарий демонстрации

1. Открыть интерфейс и показать статус системы.
2. Выбрать пресет региона в панели управления.
3. Проверить погоду и доступные слои.
4. Запустить автодетекцию поля.
5. Открыть найденное поле на карте.
6. Показать прогноз урожайности.
7. Изменить параметры сценария и запустить моделирование.
8. Создать архив по полю и скачать его.
9. При необходимости показать ручную разметку поля.

## Контрольные точки перед защитой

```bash
PYTHONPATH="$PWD/backend:$PYTHONPATH" ./.venv/bin/pytest -q
cd frontend && npm run build
docker compose config
```

Если все три команды проходят, проект находится в консистентном состоянии для демонстрации.
