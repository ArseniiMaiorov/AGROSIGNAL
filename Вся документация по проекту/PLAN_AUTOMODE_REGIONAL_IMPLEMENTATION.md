# План внедрения: AutoDetect в полном авторежиме с региональной специализацией

## 1. Целевое состояние

Система должна:

1. Работать в полном авторежиме без ручного выбора источников, дат и параметров.
2. Иметь приоритетно высокое качество по:
   - Центральному ФО
   - Северо-Западному ФО
   - Приволжскому ФО
3. Использовать Юг России (Ставрополь, Краснодар) как эталонный источник сильных псевдометок и базовый teacher-домен.
4. Для остальных регионов автоматически:
   - оценивать ожидаемую точность детекта в процентах;
   - предупреждать о недостатке данных;
   - запускать автодообучение по кнопке пользователя (полный цикл без ручных действий).
5. Хранить данные и артефакты в иерархии: `округ -> регион`.

---

## 2. Что уже есть в проекте и используется как база

Используем без переписывания ядра:

- Асинхронный pipeline и runtime-диагностика:
  - `backend/tasks/autodetect.py`
- Региональные профили и пороги:
  - `backend/core/config.py` (`SOUTH_*`, `NORTH_*`, `REGION_*`, `ML_*`)
- ML primary + fallback + постпроцесс:
  - `backend/processing/fields/*`
- Тренировочные скрипты полного цикла:
  - `backend/training/fetch_real_tiles.py`
  - `backend/training/generate_weak_labels_real_tiles.py`
  - `backend/training/gen_data.py`
  - `backend/training/check_torch_onnx_parity.py`
  - `backend/training/run_holdout_ab.py`
- UI для запуска и логов:
  - `frontend/src/components/SidePanel.vue`
  - `frontend/src/components/LogPanel.vue`
  - `frontend/src/components/ProgressBar.vue`

Идея внедрения: расширяем текущий pipeline сервисами оркестрации регионов, confidence-оценкой и автотренировкой, не ломая существующий detect-процесс.

---

## 3. Архитектура нового функционала

Добавляются 4 новых слоя.

### 3.1 Region Registry

Назначение:

- Единый справочник `округ -> регион -> AOI policy -> модель`.
- Привязка региона к активной версии модели.
- Правила источников данных для автодообучения.

Новые сущности:

- `federal_districts`
- `regions`
- `region_model_registry`

### 3.2 Confidence Engine

Назначение:

- Возвращать `confidence_pct` для каждого detect-run.
- Отмечать `needs_retrain=true/false`.
- Давать причину низкой уверенности (данные/шум/доменный сдвиг).

Базовые признаки confidence (из уже доступных runtime-полей):

- `ml_quality_score`
- `selected_date_confidence_mean`
- `fallback_rate_tile`
- `edge_signal_p90`
- доля слабых сцен и quality-gate события
- OOD-оценка по feature-space (добавляем)

### 3.3 AutoTrainer Orchestrator

Назначение:

- По кнопке запускать автономный retrain-конвейер для конкретного региона.
- Самостоятельно решать:
  - что догружать,
  - какие тайлы отобрать,
  - когда обновить модель в прод.

Пайплайн retrain (Celery chain):

1. сбор/дозагрузка тайлов региона;
2. генерация weak-label;
3. обучение и экспорт ONNX;
4. parity-check;
5. holdout/A-B;
6. публикация модели в `region_model_registry`.

### 3.4 Regional Storage Layout

Назначение:

- Физически разделить данные по округам/регионам.
- Упростить обслуживание, аудит, откат и переобучение.

---

## 4. Структура папок (округ -> регион)

Рекомендуемый корень:

`backend/data/regions/`

Пример:

```text
backend/data/regions/
  cfo/
    moscow_oblast/
      raw_tiles/
      weak_labels/
      manifests/
      models/
      reports/
      runtime_stats/
  szfo/
    leningrad_oblast/
      raw_tiles/
      weak_labels/
      manifests/
      models/
      reports/
      runtime_stats/
  pfo/
    samara_oblast/
      ...
  south_reference/
    krasnodar_krai/
      ...
    stavropol_krai/
      ...
```

Правило имен: только slug-идентификаторы (ASCII), чтобы исключить ошибки путей в docker/linux/windows.

---

## 5. Изменения API и данных

## 5.1 API detect/status/result

Расширить ответы:

- `region_code`
- `district_code`
- `model_version_used`
- `confidence_pct` (0-100)
- `confidence_level` (`high`/`medium`/`low`)
- `needs_retrain` (bool)
- `retrain_reason` (string)

## 5.2 Новый API автодообучения

- `POST /api/v1/training/retrain-region`
  - вход: `district_code`, `region_code`, `mode=auto`
  - выход: `job_id`
- `GET /api/v1/training/retrain-status/{job_id}`
- `GET /api/v1/training/region-models`

## 5.3 Изменения БД

Добавить:

- таблицу `retrain_jobs`
- таблицу `region_model_registry`
- таблицу `region_quality_snapshots`

Расширить `aoi_runs`:

- `district_code`
- `region_code`
- `confidence_pct`
- `needs_retrain`
- `model_version_used`

---

## 6. Логика confidence в %

Стартовая формула (калибруем по holdout):

```text
confidence = 100 * clamp(
  0.30 * ml_quality_score +
  0.20 * selected_date_confidence_mean +
  0.15 * (1 - fallback_rate_tile) +
  0.10 * edge_quality_score +
  0.15 * temporal_stability_score +
  0.10 * (1 - ood_score),
0, 1)
```

Границы:

- `>= 85`: высокая уверенность
- `70..84`: средняя
- `< 70`: низкая, показываем предупреждение и кнопку дообучения

Важно:

- Для нецелевых регионов confidence обязателен всегда.
- Для целевых ФО допустим автоfallback на strongest regional model, но confidence все равно показываем.

---

## 7. Автодообучение по кнопке (без ручных действий)

Сценарий:

1. Пользователь получает `needs_retrain=true`.
2. Нажимает кнопку `Дообучить регион`.
3. Создается `retrain_job`.
4. Celery запускает автоматический конвейер:
   - определение AOI-пула региона;
   - догрузка сцен Sentinel-2/S1 по policy;
   - weak-label генерация;
   - тренировка `BoundaryUNet v2_16ch` (teacher warm-start с south reference);
   - parity + holdout;
   - публикация новой модели.
5. Новая модель помечается `active` только если прошла критерии acceptance.

Критерии публикации:

- parity pass
- качество не хуже текущей модели на региональном holdout
- нет деградации по критическим метрикам контура

---

## 8. Поэтапный план внедрения

## Этап 1. Регионализация данных и конфигов

Задачи:

- Ввести справочник округов/регионов.
- Реализовать структуру директорий `district/region`.
- Добавить region policy в конфиг.

Изменяем:

- `backend/core/config.py`
- новый модуль `backend/core/regions_registry.py`
- миграции БД

Результат:

- система знает регион и где лежат его данные/модели.

## Этап 2. Confidence Engine

Задачи:

- Реализовать расчет `confidence_pct`.
- Добавить `needs_retrain` и текст причин.
- Встроить в `status/result`.

Изменяем:

- `backend/tasks/autodetect.py`
- `backend/api/schemas.py`
- `backend/api/fields.py`
- новый модуль `backend/core/confidence.py`

Результат:

- каждый run имеет числовую оценку качества и флаг дообучения.

## Этап 3. Автотренировочный оркестратор

Задачи:

- Новый Celery workflow retrain-job.
- Обертка над существующими скриптами обучения.
- Публикация версии модели по региону.

Изменяем:

- `backend/core/celery_app.py`
- новый пакет `backend/tasks/retrain_region.py`
- адаптация `backend/training/*` под параметры региона и выходные директории.

Результат:

- retrain стартует одной командой/API и выполняется до конца автоматически.

## Этап 4. UI и пользовательский сценарий

Задачи:

- Показ `confidence_pct` и причин низкой уверенности.
- Кнопка `Дообучить регион`.
- Отображение прогресса retrain-job.

Изменяем:

- `frontend/src/components/SidePanel.vue`
- `frontend/src/components/LogPanel.vue`
- `frontend/src/store/map.js`

Результат:

- пользователь видит точность в % и запускает автодообучение без ручных шагов.

## Этап 5. Региональные KPI и acceptance gates

Задачи:

- Зафиксировать KPI для ЦФО/СЗФО/ПФО.
- Отдельный quality gate для south reference.
- Автопубликация только при прохождении KPI.

Изменяем:

- `backend/training/run_holdout_ab.py`
- `backend/training/run_regional_retrain_pipeline.sh`
- новый файл `backend/training/REGIONAL_AUTOMODE_GATES.md`

Результат:

- контролируемая эволюция качества без ручной приемки на каждом цикле.

## Этап 6. Автономный режим 24/7

Задачи:

- Планировщик фона (Celery beat): сбор новых данных и nightly retrain-кандидаты.
- Ограничения ресурсов (1 retrain per region, GPU-lock, timeout, rollback).
- Полный аудит и трассировка.

Изменяем:

- `backend/core/celery_app.py`
- новый модуль `backend/tasks/autotrain_scheduler.py`
- таблицы ретраев и логов.

Результат:

- система работает автономно и сама поддерживает актуальность моделей по регионам.

---

## 9. Приоритеты качества по регионам

Приоритет 1 (обязательное высокое качество):

- ЦФО
- СЗФО
- ПФО

Эталонный teacher-домен:

- Краснодарский край
- Ставропольский край

Практика:

- south reference используется как источник warm-start и сильных паттернов границ;
- финальная модель всегда региональная (персональная для целевого региона), а не общая на всю страну.

---

## 10. Definition of Done

Функционал считается внедренным, когда:

1. Для любого detect-run возвращается `confidence_pct` и `needs_retrain`.
2. UI показывает предупреждение и кнопку автодообучения при низкой уверенности.
3. Нажатие кнопки запускает полный retrain-процесс без ручных действий.
4. Папки и артефакты создаются строго в иерархии `округ/регион`.
5. Для ЦФО/СЗФО/ПФО используются региональные модели из `region_model_registry`.
6. При провале качества новая модель не публикуется и выполняется автоматический rollback.

---

## 11. Риски и контроль

Ключевые риски:

- Шум weak-label в сложных регионах.
- Рост времени retrain и стоимости хранения.
- Регрессия качества при автопубликации.

Контроль:

- жесткие quality gates перед публикацией;
- canary rollout по части AOI;
- rollback на предыдущую активную модель;
- обязательный parity-check и holdout-сравнение.

---

## 12. Итоговый принцип внедрения

Не переписывать текущий autodetect, а добавить поверх него:

1. региональную оркестрацию,
2. confidence-оценку в процентах,
3. автоматический retrain-конвейер,
4. модельный registry с публикацией по quality gate.

Это дает требуемый авторежим и управляемое качество на целевых округах при сохранении уже реализованного ядра проекта.
