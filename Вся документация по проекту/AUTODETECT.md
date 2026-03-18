# AutoDetect: как устроен автодетект полей

## Назначение

`AutoDetect` — это основной pipeline проекта для автоматического поиска контуров сельскохозяйственных полей по спутниковым данным в заданной области интереса (`AOI`).

Коротко:

- на вход подаётся область, сезон и параметры детекта;
- система режет AOI на тайлы;
- забирает мультивременные спутниковые сцены;
- строит набор спектральных, текстурных и контурных признаков;
- получает маску-кандидат и уточняет её ML-моделью;
- сегментирует поля в отдельные объекты;
- векторизует и фильтрует их;
- сохраняет результат в PostGIS и отдаёт его на карту и в аналитику.

Главный файл orchestration:

- `backend/tasks/autodetect.py`

Ключевые зависимые модули:

- `backend/api/fields.py`
- `backend/processing/fields/ml_inference.py`
- `backend/processing/fields/postprocess.py`
- `backend/processing/fields/segmentation.py`
- `backend/processing/fields/vectorize.py`
- `backend/processing/fields/object_classifier.py`
- `backend/core/config.py`

---

## Что именно решает autodetect

Автодетект отвечает на вопрос:

> "Где на этой территории находятся отдельные поля и каковы их границы?"

Это не классификация культур и не прогноз урожайности.  
Автодетект решает геометрическую задачу:

- найти полевые контуры;
- отделить одно поле от другого;
- не спутать поля с лесом, водой, дорогами, застройкой и шумом;
- сохранить результат как векторные полигоны и связанные grid-данные.

---

## Входные данные

Основной запрос задаётся через `DetectRequest` и API `POST /api/v1/fields/detect/preflight` / `POST /api/v1/fields/detect`.

Главные входы:

- `aoi`
  - `point_radius`
  - `bbox`
  - `polygon`
- `time_range`
  - `start_date`
  - `end_date`
- `resolution_m`
- `max_cloud_pct`
- `target_dates`
- `min_field_area_ha`
- `seed_mode`
- `use_sam`
- `config.preset`

`AOI` конвертируется в геометрию и потом в набор тайлов.

---

## API-слой autodetect

Основной API находится в `backend/api/fields.py`.

### 1. Preflight

`POST /api/v1/fields/detect/preflight`

Назначение:

- оценить число тайлов;
- оценить условный runtime-class;
- понять, не слишком ли тяжёлый запрос для текущего worker budget;
- заранее предупредить пользователя, а не ронять задачу позже.

Возвращает:

- `budget_ok`
- `estimated_tiles`
- `estimated_runtime_class`
- `recommended_preset`
- `reason`
- `warnings`
- `preset`

### 2. Запуск детекта

`POST /api/v1/fields/detect`

Назначение:

- создать `AoiRun`;
- зафиксировать параметры и preflight metadata;
- отправить задачу в Celery.

### 3. Список запусков

`GET /api/v1/fields/runs`

Нужен для UI, чтобы разделять:

- активный run (`activeRunId`);
- видимый на карте run (`visibleRunId`);
- последний успешный run (`lastCompletedRunId`).

### 4. Статус

`GET /api/v1/fields/status/{aoi_run_id}`

Возвращает runtime-состояние:

- `status`
- `progress`
- `stage_label`
- `stage_detail`
- `started_at`
- `updated_at`
- `last_heartbeat_ts`
- `stale_running`
- `estimated_remaining_s`

### 5. Результат

`GET /api/v1/fields/result/{aoi_run_id}`

Если run ещё не готов или упал, endpoint не должен бросать сырой `500`.  
Он возвращает структурированный `RunResult` с корректным `status`.

---

## Preset-ы детекта

Preset-ы определены в `backend/api/fields.py` и используются также во фронтенде.

### Fast

- `resolution_m = 10`
- `target_dates = 4`
- `use_sam = false`
- `min_field_area_ha = 0.5`

Зачем нужен:

- быстрый preview;
- минимальная вычислительная нагрузка;
- грубее по мелким полям, но должен всё равно искать полеобразные контуры, а не работать на "сломанных" 30 м.

### Standard

- `resolution_m = 10`
- `target_dates = 7`
- `use_sam = false`
- `min_field_area_ha = 0.25`

Зачем нужен:

- основной рабочий режим;
- баланс между скоростью и качеством;
- штатный режим для большинства AOI.

### Quality

- `resolution_m = 10`
- `target_dates = 9`
- `use_sam = true`
- `min_field_area_ha = 0.1`

Зачем нужен:

- максимальная fidelity;
- лучше работает на сложных границах и мелких объектах;
- дольше и тяжелее по runtime, поэтому проходит через preflight budget.

---

## Почему нужен preflight

Без preflight пользователь может задать валидный на вид AOI, который:

- породит слишком много тайлов;
- затянет `south_bridge` и постпроцесс;
- упрётся в Celery soft limit;
- зависнет в UI или завершится только после долгого ожидания с ошибкой.

Поэтому preflight нужен не как "лишняя проверка", а как защита от заранее обречённых запусков.

Сейчас он учитывает:

- число тайлов;
- число временных срезов;
- penalty за `SAM`;
- runtime-class.

---

## Общий pipeline autodetect

Ниже реальный смысл пайплайна в текущей реализации.

### ASCII-схема pipeline

```text
┌─────────────────────────────────────────────────────────────────────┐
│ Client / UI                                                        │
│ POST /fields/detect/preflight -> POST /fields/detect               │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                v
┌─────────────────────────────────────────────────────────────────────┐
│ backend/api/fields.py                                              │
│ - DetectRequest validation                                         │
│ - preset inference                                                 │
│ - tile/runtime preflight                                           │
│ - create AoiRun + enqueue Celery                                   │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                v
┌─────────────────────────────────────────────────────────────────────┐
│ backend/tasks/autodetect.py :: run_autodetect()                    │
│ - runtime_meta init                                                │
│ - progress / heartbeat                                             │
│ - region profile                                                   │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                v
┌─────────────────────────────────────────────────────────────────────┐
│ AOI -> make_tiles()                                                │
│ time_range -> _build_time_windows()                                │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                v
┌─────────────────────────────────────────────────────────────────────┐
│ Per-tile loop                                                      │
│ 1. fetch multitemporal Sentinel-2                                  │
│ 2. select valid dates                                              │
│ 3. build feature stack / indices                                   │
│ 4. build candidate mask + postprocess                              │
│ 5. run BoundaryUNet v2                                             │
│ 6. fuse ML + rule-based mask                                       │
│ 7. watershed segmentation                                          │
│ 8. optional SAM refine                                             │
│ 9. polygonize labels                                               │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                v
┌─────────────────────────────────────────────────────────────────────┐
│ Cross-tile stage                                                   │
│ - merge_tile_polygons()                                            │
│ - object classifier                                                │
│ - topology cleanup                                                 │
│ - DB insert: Field, GridCell, active-learning rows                 │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                v
┌─────────────────────────────────────────────────────────────────────┐
│ Result                                                             │
│ - AoiRun = done / failed / stale                                   │
│ - geojson for fields                                               │
│ - grid layers                                                      │
│ - status/result endpoints for UI                                   │
└─────────────────────────────────────────────────────────────────────┘
```

### Шаг 1. Создание run и runtime metadata

В начале `run_autodetect()`:

- run получает статус `running`;
- инициализируется `runtime_meta`;
- фиксируются preset, region profile, время, warnings, progress stage.

Зачем:

- это даёт UI и backend status-service прозрачное состояние задачи;
- позволяет переживать долгие ранние стадии без ощущения "зависло".

### Шаг 2. Построение временных окон

`_build_time_windows(...)`

Период разбивается на окна внутри сезонных границ.

Зачем:

- модель и эвристики работают не по одной дате, а по временной динамике;
- это повышает устойчивость к облакам, аномальным сценам и случайному шуму;
- помогает брать не "одну красивую сцену", а устойчивый сезонный сигнал.

### Шаг 3. AOI -> тайлы

AOI режется на тайлы через `make_tiles(...)`.

Зачем:

- ограничить память;
- детектировать поля локально, а не пытаться прогнать весь AOI одним монолитом;
- ускорить хранение промежуточного состояния и прогресса.

### Шаг 4. Получение спутниковых данных

Основной источник в прод-режиме:

- Sentinel Hub / Sentinel-2

Параллельно могут использоваться:

- WorldCover prior;
- дорожные маски;
- погодный snapshot для runtime metadata и сопутствующих сервисов.

Зачем:

- получить мультивременной стек сцен;
- собрать спектральную динамику;
- не полагаться на одну дату.

### Шаг 5. Выбор лучших дат

После fetch идёт `date selection`.

Зачем:

- отбросить плохие сцены;
- уменьшить влияние облаков и низкого покрытия;
- сохранить репрезентативные временные срезы.

### Шаг 6. Построение признаков

Из temporal stack формируются признаки:

- `edge_composite`
- `max_ndvi`
- `mean_ndvi`
- `ndvi_std`
- `ndwi_mean`
- `bsi_mean`
- `scl_valid_fraction`
- `rgb_r/g/b`
- `ndvi_entropy`
- `mndwi_max`
- `ndmi_mean`
- `ndwi_median`
- `green_median`
- `swir_median`

Актуальный профиль модели:

- `feature_profile = v2_16ch`

Зачем:

- сочетать форму границ, сезонность, воду, почвенную оголённость, спектральную структуру и quality mask.

### Шаг 7. Candidate mask и postprocess

`backend/processing/fields/postprocess.py`

Это rule-based и geo-aware часть pipeline.

Что там происходит:

- вычисляются phenology masks;
- строится candidate field mask;
- подключаются water / forest / builtup / road barriers;
- делается `gap close`;
- делается `infill`;
- для южных профилей может включаться `south_bridge`;
- затем возможен merge регионов и ориентированный watershed prep.

Зачем:

- ML не должен начинать с полностью сырой сцены;
- сначала нужно получить осмысленную кандидатку;
- затем безопасно дорастить её там, где это правдоподобно, и не пересечь жёсткие барьеры.

### Шаг 8. ML boundary inference

`backend/processing/fields/ml_inference.py`

Основная модель:

- `BoundaryUNet`
- `boundary_unet_v2`
- 3 головы:
  - `extent`
  - `boundary`
  - `distance`

Модель умеет работать через:

- PyTorch
- ONNX Runtime

Зачем:

- rule-based кандидатку недостаточно просто "почистить";
- нужна обученная модель, которая понимает вероятную границу поля по многоканальному стеку признаков.

### Шаг 9. Fusion

ML-результат не всегда используется "как есть".

Система совмещает:

- pre-ML candidate;
- boundary inference;
- regional profile;
- quality gates.

Зачем:

- не доверять слепо ни rules-only, ни ML-only;
- удерживать recall в сложных регионах и precision там, где ML шумит.

### Шаг 10. Сегментация на отдельные поля

`backend/processing/fields/segmentation.py`

Используется watershed pipeline:

- distance transform;
- marker extraction;
- flooding surface;
- oriented watershed при необходимости.

Зачем:

- candidate mask часто покрывает несколько полей сразу;
- watershed делит сплошной массив на отдельные объекты.

### Шаг 11. SAM refinement

Если preset / budget / runtime позволяют, может включаться SAM.

Зачем:

- локально уточнить контур там, где обычный pipeline даёт грубую границу.

Важно:

- SAM не обязателен;
- его включение проходит через preflight и runtime budget;
- если budget не проходит, система должна деградировать контролируемо, а не зависать.

### Шаг 12. Векторизация

`backend/processing/fields/vectorize.py`

Что происходит:

- label raster -> polygons;
- фильтрация по минимальной площади;
- упрощение геометрии;
- расчёт `area_m2` и `perimeter_m`.

Зачем:

- пользовательский результат — это не raster labels, а векторные полигоны полей.

### Шаг 13. Merge между тайлами

После обработки отдельных тайлов полигоны объединяются через `merge_tile_polygons(...)`.

Зачем:

- тайлы перекрываются;
- одно и то же поле может попасть в несколько тайлов;
- нужно убрать дубли и сшить общую геометрию.

### Шаг 14. Object classifier

`backend/processing/fields/object_classifier.py`

Это отдельная модель второго уровня, работающая уже по полигонам.

Использует признаки объектов:

- площадь;
- периметр;
- shape metrics;
- NDVI/NDWI/MSI/BSI aggregates;
- worldcover crop pct;
- growth amplitude;
- ndvi entropy;
- distance to road;
- neighbor context;
- valid fraction и т.д.

Зачем:

- отсечь ложные полигоны после сегментации;
- убрать геометрический и спектральный мусор.

### Шаг 15. Сохранение в БД

После merge и object filtering:

- сохраняются `Field`;
- сохраняются `GridCell`;
- обновляется `AoiRun`;
- могут создаваться active learning candidates.

Зачем:

- результат должен быть доступен карте, слоям, архивам, аналитике и human review.

---

## Модели, используемые в autodetect

### 1. Boundary model

Артефакты:

- `backend/models/boundary_unet_v2.pth`
- `backend/models/boundary_unet_v2.onnx`
- `backend/models/boundary_unet_v2.pth.meta.json`
- `backend/models/boundary_unet_v2.norm.json`

Что важно:

- `model_version = boundary_unet_v2`
- `train_data_version = real_tiles_v5`
- `feature_profile = v2_16ch`
- обучена на weak-label pipeline, а не на полноценно ручном production-grade GT

Почему это важно:

- это сильный инженерный detector, но научный риск всё ещё в данных и holdout validation.

### 2. Object classifier

Артефакты:

- `backend/models/object_classifier.pkl`
- `backend/models/object_classifier_compat.pkl`

Задача:

- бинарно/вероятностно оценить, похож ли итоговый полигон на настоящее поле.

---

## Региональные профили и зачем они нужны

Автодетект не работает одинаково по всем широтам и типам ландшафта.  
Поэтому runtime определяет:

- `region_band`
- `region_boundary_profile`
- `regional_quality_target`

Зачем:

- в разных регионах разная структура полей;
- юг часто даёт крупные массивы и иной recall/merge balance;
- север и смешанные зоны требуют другой степени консервативности.

Отсюда и специальные эвристики вроде:

- `south_bridge`
- разные пороги boundary quality
- разные настройки grow/merge

---

## Почему существует `south_bridge`

`south_bridge` — это не "магия", а эвристика слияния близких компонент в южном профиле.

Зачем нужен:

- на юге поля часто выглядят как разорванные куски одной структуры;
- candidate mask может излишне фрагментироваться;
- bridge помогает восстановить целостность поля.

Почему он опасен:

- это дорогая операция;
- без ограничений она начинает доминировать по runtime;
- на больших AOI может приводить к долгим стадиям и таймаутам.

Поэтому сейчас он ограничен preset-ами:

- `fast`: выключен
- `standard`: ограничен сильнее
- `quality`: разрешён, но с cap по числу компонент

---

## Runtime, progress и heartbeats

Прогресс больше не просто "0-10-90-100".  
Pipeline публикует детальные стадии:

- `fetch`
- `date selection`
- `boundary fill`
- `model inference`
- `segmentation`
- `boundary refine`
- `sam refine`
- `tile finalize`
- `merge`
- `object classifier`
- `db insert`
- `topology`
- `complete`

Также доступны:

- `started_at`
- `last_heartbeat_ts`
- `estimated_remaining_s`
- `stale_running`

Зачем:

- пользователь должен понимать, что задача реально идёт;
- worker hang должен быть отличим от долгой, но живой стадии.

---

## Что пользователь реально видит после autodetect

После успешного run система выдаёт:

- векторные поля;
- grid-слои;
- runtime status;
- quality-related metadata;
- активные/последние run summary для UI.

Во фронтенде важно разделение:

- `activeRunId` — то, что сейчас считается;
- `visibleRunId` — то, что сейчас показывается на карте.

Зачем это было сделано:

- новый run не должен затирать предыдущий успешный результат, пока сам не завершился полностью.

---

## Что делает autodetect хорошо

- сочетает rule-based и ML, а не упирается в один подход;
- устойчив к многодатным данным;
- имеет preflight, heartbeat и runtime meta;
- умеет работать через tile-based pipeline;
- поддерживает SAM как optional refinement;
- имеет вторичную фильтрацию через object classifier;
- сохраняет не только поля, но и сопутствующую grid/diagnostics-информацию.

---

## Ограничения и честные слабые места

### 1. Научный риск всё ещё в данных

Boundary model обучена не на большом независимом manual GT, а в значительной степени на weak labels.

Следствие:

- хороший production-demo;
- но истинная коммерческая надёжность определяется manual holdout и benchmark-ами.

### 2. Сложные AOI всё ещё дорогие

Даже после budget control и preset tuning:

- крупные AOI в `quality` могут идти долго;
- preflight решает часть проблем, но не отменяет реальную стоимость обработки.

### 3. Визуальный слой и геометрическая правда — не одно и то же

На карте пользователь может видеть grid/aggregated overlays, а не исходные пиксели спутника.

Важно понимать:

- source detail может быть `10 м`;
- визуальная подложка или grid могут выглядеть грубее.

### 4. Без источников спутниковых данных прод-детект не работает

Если нет валидных Sentinel credentials:

- продовый autodetect недоступен;
- synthetic mode допустим только для тестов/debug.

---

## Как читать логи autodetect

Примеры стадий:

- `fetch · windows 3/9`
- `date selection · windows 8/9`
- `boundary fill · gap close`
- `boundary fill · bridge 120/320`
- `model inference · tile 4/15`
- `segmentation · tile 4/15`
- `object classifier · tile features`
- `tile finalize · grid zoom 2/3`
- `merge`
- `db insert`

Как интерпретировать:

- `fetch` — идут сцены;
- `date selection` — отбор качественных временных срезов;
- `boundary fill` — rule-based morphology/merge/grow;
- `model inference` — работа U-Net;
- `segmentation` — разбиение на поля;
- `tile finalize` — сборка tile-артефактов;
- `merge/db insert` — глобальная сборка результата и запись в БД.

---

## Что улучшать дальше

Если развивать именно autodetect, а не весь продукт в целом, то приоритеты такие:

1. Увеличить manual GT и locked Russian holdout.
2. Перевести качество модели на независимый geo-benchmark, а не только internal metrics.
3. Доделать полноценный high-detail raster path для high zoom map mode.
4. Довести release QA matrix по регионам РФ.
5. Развивать active learning вокруг слабых регионов и типов ошибок.

---

## Быстрый путь по коду

Если нужно быстро разобраться в autodetect по исходникам, смотреть в таком порядке:

1. `backend/api/fields.py`
2. `backend/tasks/autodetect.py`
3. `backend/processing/fields/postprocess.py`
4. `backend/processing/fields/ml_inference.py`
5. `backend/processing/fields/segmentation.py`
6. `backend/processing/fields/vectorize.py`
7. `backend/processing/fields/object_classifier.py`
8. `backend/core/config.py`
9. `backend/tests/test_autodetect_task.py`
10. `backend/tests/test_postprocess.py`

---

## Версия для разработчика

Ниже тот же autodetect, но уже не как продуктовый pipeline, а как карта кода.

### Карта модулей

```text
backend/api/fields.py
  ├─ _build_detect_preflight()
  ├─ detect_preflight()
  ├─ detect_fields()
  ├─ get_run_status()
  └─ get_run_result()

backend/tasks/autodetect.py
  ├─ run_autodetect()
  ├─ _build_time_windows()
  ├─ _set_progress_stage()
  ├─ _set_tile_progress()
  ├─ _set_post_progress()
  ├─ _sam_preflight_budget()
  └─ runtime_meta enrichment / failure handling

backend/processing/fields/
  ├─ composite.py
  │   └─ date selection / temporal composite / valid mask logic
  ├─ indices.py
  │   └─ spectral indices
  ├─ postprocess.py
  │   └─ candidate mask cleanup, grow, infill, south_bridge
  ├─ ml_inference.py
  │   └─ BoundaryUNet / FieldBoundaryInferencer
  ├─ ml_fusion.py
  │   └─ rule-based + ML fusion
  ├─ segmentation.py
  │   └─ watershed_segment()
  ├─ sam_primary.py / sam_field_boundary.py
  │   └─ optional SAM refinement
  ├─ vectorize.py
  │   ├─ polygonize_labels()
  │   └─ merge_tile_polygons()
  └─ object_classifier.py
      └─ polygon-level false-positive filtering

backend/storage/
  ├─ fields_repo.py
  │   └─ runs / fields / geojson / delete / merge / split
  └─ db.py
      └─ SQLAlchemy models and persistence

frontend/src/store/map.js
  ├─ activeRunId
  ├─ visibleRunId
  └─ polling / map refresh / detect orchestration
```

### Кто за что отвечает

#### `backend/api/fields.py`

Отвечает за boundary между внешним API и internal pipeline:

- валидация запроса;
- вывод preset-а;
- preflight budget;
- создание `AoiRun`;
- выдача status/result;
- трансляция runtime в человекочитаемые `stage_label` и `stage_detail`.

#### `backend/tasks/autodetect.py`

Это главный orchestration-слой.

Он:

- не учит модель;
- не рисует UI;
- не хранит детали бизнес-логики фронтенда;
- а именно собирает и выполняет весь detect-run от старта до финального `done/failed`.

Важные зоны файла:

- bootstrap run и runtime config;
- подготовка time windows;
- tile loop;
- post-tile merge;
- DB insert;
- failure handling;
- progress / heartbeat.

#### `postprocess.py`

Это главная rule-based часть.

Если хочется понять, почему run завис на `boundary fill`, смотреть сюда в первую очередь:

- `gap close`
- `infill`
- `south_bridge`
- merge scan

#### `ml_inference.py`

Точка входа в boundary model:

- загрузка модели;
- выбор backend `torch/onnx`;
- norm stats;
- feature profile;
- запуск inference.

#### `segmentation.py`

Отвечает уже не за "где есть поле", а за "как разделить candidate region на отдельные поля".

#### `vectorize.py`

Тут происходит переход:

- из raster labels
- в GeoDataFrame / polygons

Именно здесь появляются:

- `area_m2`
- `perimeter_m`
- merge cross-tile duplicates

#### `object_classifier.py`

Последний quality gate перед записью в БД.

Если система находит "красивые, но ложные" полигоны, смотреть надо сюда и в upstream feature generation.

### Sequence flow для разработчика

```text
User / Frontend
   |
   | 1. POST /api/v1/fields/detect/preflight
   v
fields.py::_build_detect_preflight
   |
   | 2. estimate tiles + runtime class + preset budget
   v
fields.py::detect_fields
   |
   | 3. create AoiRun(status=queued)
   | 4. celery.delay(run_id)
   v
Celery worker
   |
   | 5. run_autodetect(run_id)
   v
autodetect.py
   |
   | 6. mark running, init runtime_meta, set progress=5..10
   | 7. build AOI polygon
   | 8. make tiles
   | 9. build time windows
   v
Per-tile processing
   |
   | 10. fetch multitemporal scenes
   | 11. date selection
   | 12. feature stack + indices
   | 13. candidate postprocess
   | 14. boundary model inference
   | 15. fusion
   | 16. watershed segmentation
   | 17. optional SAM
   | 18. polygonize_labels
   v
Post-tile processing
   |
   | 19. merge_tile_polygons
   | 20. object classifier
   | 21. topology cleanup
   | 22. save Field + GridCell + run metadata
   v
fields.py::get_run_status / get_run_result
   |
   | 23. UI polls status/result
   | 24. visibleRunId switches only after done + successful load
   v
Map / Analytics
```

### Runtime stages -> code zones

Это полезно, когда пользователь приносит лог вида `boundary fill · bridge 103/319`.

```text
fetch            -> scene download / cache / provider retry logic
date selection   -> temporal slice filtering
boundary fill    -> postprocess.py
model inference  -> ml_inference.py
segmentation     -> segmentation.py
sam refine       -> SAM branch inside autodetect.py
tile finalize    -> tile-level polygon/grid assembly
merge            -> vectorize.py::merge_tile_polygons()
object classifier-> object_classifier.py
db insert        -> fields_repo/db persistence
topology         -> final cleanup and run completion
```

### Что смотреть при типовых проблемах

#### 1. Детект не стартует

Смотреть:

- `backend/api/fields.py`
- preflight reason
- auth / permissions
- celery availability

#### 2. Зависает на `fetch`

Смотреть:

- `providers/sentinelhub/client.py`
- provider cache / retries
- credentials
- scene availability

#### 3. Долго висит на `boundary fill`

Смотреть:

- `postprocess.py`
- `south_bridge`
- число компонент
- preset overrides из `run_autodetect()`

#### 4. Поля есть в grid, но нет контуров

Смотреть:

- `visibleRunId` vs `activeRunId`
- run-scoped loading на фронте
- `get_run_result()`
- `get_fields_geojson(aoi_run_id=...)`

#### 5. Слишком много мусора в результате

Смотреть:

- candidate mask thresholds
- ML fusion
- object classifier
- min area filter

#### 6. Хороший AOI падает по времени

Смотреть:

- preflight budget
- tile count
- `target_dates`
- SAM policy
- bridge caps
- Celery soft/hard time limits

### Минимальный mental model для входа в код

Если упростить autodetect до одного инженерного тезиса, он такой:

> API валидирует и планирует run, Celery orchestrates tile pipeline, postprocess строит разумную кандидатку, ML уточняет границы, watershed делит на поля, vectorize делает геометрию, object classifier дочищает мусор, repo сохраняет результат.

Если держать это в голове, разбираться в конкретном баге становится сильно проще.

---

## Итог

Текущий autodetect в проекте — это не "одна сетка, которая рисует поля".  
Это сложный гибридный pipeline:

- multi-temporal;
- tile-based;
- rule-based + ML;
- с региональными профилями;
- с optional SAM;
- с объектной фильтрацией;
- с прогрессом, heartbeat и preflight budget.

Его сильная сторона — инженерная зрелость pipeline и хорошая composability.  
Его главный риск — качество и независимость ground truth, а не отсутствие алгоритмов как таковых.
