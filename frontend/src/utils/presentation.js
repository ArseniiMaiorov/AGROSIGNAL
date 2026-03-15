import { locale, t } from './i18n'

const UI_LOCALES = {
  ru: 'ru-RU',
  en: 'en-GB',
}

const PRESET_META = {
  ru: {
    fast: {
      label: 'Быстрый',
      description: '10 м, 4 даты. Preview-only режим до 40 км: укрупнённые сельхоз-контуры без тяжёлого ranker, watershed и объектной фильтрации.',
    },
    standard: {
      label: 'Стандартный',
      description: '10 м, 7 дат. Основной рабочий режим точного детекта полей до 20 км.',
    },
    quality: {
      label: 'Точный',
      description: '10 м, 9 дат. Максимальная геометрическая детализация до 8 км: TTA, сложная сегментация и уточнение границ.',
    },
  },
  en: {
    fast: {
      label: 'Fast',
      description: '10 m, 4 dates. Preview-only mode up to 40 km with coarse agricultural contours and without heavy ranking, watershed, or object filtering.',
    },
    standard: {
      label: 'Standard',
      description: '10 m, 7 dates. Main operational field-detection mode up to 20 km.',
    },
    quality: {
      label: 'Quality',
      description: '10 m, 9 dates. Highest-fidelity geometry up to 8 km with TTA, richer segmentation, and boundary refinement.',
    },
  },
}

const TASK_STAGE_LABELS = {
  ru: {
    queued: 'В очереди',
    prepare: 'Подготовка',
    fetch: 'Загрузка сцен',
    tiling: 'Разбиение на тайлы',
    date_selection: 'Отбор дат',
    weekly_profile: 'Недельный профиль',
    cache: 'Проверка кэша',
    features: 'Сбор признаков',
    baseline: 'Базовый прогноз',
    counterfactual: 'Сценарный расчёт',
    candidate_postprocess: 'Очистка контуров',
    model_inference: 'Инференс модели',
    segmentation: 'Сегментация',
    boundary_refine: 'Уточнение границ',
    sam_refine: 'SAM-уточнение',
    tile_finalize: 'Финализация тайла',
    tile_done: 'Тайл готов',
    merge: 'Слияние тайлов',
    object_classifier: 'Фильтрация объектов',
    db_insert: 'Запись в базу',
    persist: 'Сохранение результата',
    topology: 'Топология',
    done: 'Готово',
    failed: 'Ошибка',
    running: 'Выполняется',
    stale: 'Зависло',
  },
  en: {
    queued: 'Queued',
    prepare: 'Preparing',
    fetch: 'Fetching scenes',
    tiling: 'Tiling',
    date_selection: 'Selecting dates',
    weekly_profile: 'Weekly profile',
    cache: 'Checking cache',
    features: 'Building features',
    baseline: 'Loading baseline',
    counterfactual: 'Scenario evaluation',
    candidate_postprocess: 'Contour cleanup',
    model_inference: 'Model inference',
    segmentation: 'Segmentation',
    boundary_refine: 'Boundary refinement',
    sam_refine: 'SAM refinement',
    tile_finalize: 'Finalizing tile',
    tile_done: 'Tile complete',
    merge: 'Merging tiles',
    object_classifier: 'Object filtering',
    db_insert: 'Writing to database',
    persist: 'Persisting result',
    topology: 'Topology cleanup',
    done: 'Done',
    failed: 'Failed',
    running: 'Running',
    stale: 'Stalled',
  },
}

const REASON_TEXT = {
  ru: {
    host_safety_envelope_exceeded: 'Параметры запуска выходят за безопасный диапазон текущего хоста.',
    region_not_validated_core: 'Регион ещё не входит в проверенный производственный контур.',
    fast_preview_requires_review: 'Быстрый режим предназначен для предварительного просмотра и требует ручной проверки.',
    budget_guardrail_warning: 'Запуск разрешён, но находится рядом с вычислительными ограничениями и требует проверки.',
    field_medium_confidence_review: 'Контур пригоден, но перед агрономическим использованием его лучше проверить визуально.',
    field_low_confidence_review: 'Контур с низкой уверенностью нельзя использовать без ручной проверки.',
    field_confidence_unconfirmed: 'Уверенность контура не подтверждена, нужна проверка оператором.',
    manual_confirmation: 'Контур подтверждён вручную.',
    confidence_unavailable: 'Автоматическая оценка уверенности пока недоступна.',
    high_confidence_stable_contour: 'Контур устойчив и согласован между проверками модели.',
    medium_confidence_boundary_review: 'Контур в целом пригоден, но на границах возможны локальные ошибки.',
    low_confidence_manual_review: 'Контур неустойчив, его стоит проверить вручную.',
    crop_unsuitable_for_region: 'Культура плохо подходит для этого региона, автоматический расчёт нельзя считать надёжным.',
    crop_borderline_suitability: 'Культура находится на границе агроклиматической пригодности для этого региона.',
    global_baseline_requires_review: 'Использован общий baseline без локальной калибровки, нужен агрономический review.',
    outside_model_applicability: 'Входные данные выходят за область применимости модели.',
    outside_training_envelope: 'Сценарий выходит за диапазон наблюдений, на которых модель обучалась.',
    baseline_not_supported: 'Базовый прогноз по этому полю не поддержан моделью.',
    weekly_profile_insufficient: 'Для расчёта не хватает качественного недельного профиля поля.',
    support_review_required: 'Результат требует дополнительной проверки перед использованием.',
    preview_contour_requires_confirmation: 'Preview-контур предназначен только для предварительного просмотра и требует подтверждения.',
  },
  en: {
    host_safety_envelope_exceeded: 'Run parameters exceed the safe envelope of the current host.',
    region_not_validated_core: 'This region is not yet part of the validated production core.',
    fast_preview_requires_review: 'Fast mode is a preview mode and requires manual review.',
    budget_guardrail_warning: 'The run is allowed, but it is close to compute guardrails and should be reviewed.',
    field_medium_confidence_review: 'The contour is usable, but it should be visually reviewed before agronomic use.',
    field_low_confidence_review: 'This low-confidence contour should not be used without manual review.',
    field_confidence_unconfirmed: 'Contour confidence is not confirmed yet and requires operator review.',
    manual_confirmation: 'The contour was confirmed manually.',
    confidence_unavailable: 'Automatic confidence estimation is currently unavailable.',
    high_confidence_stable_contour: 'The contour is stable and consistent across model checks.',
    medium_confidence_boundary_review: 'The contour is mostly usable, but local boundary errors remain possible.',
    low_confidence_manual_review: 'The contour is unstable and should be reviewed manually.',
    crop_unsuitable_for_region: 'The crop is poorly suited to this region, so the automatic estimate is not reliable.',
    crop_borderline_suitability: 'The crop is near the agro-climatic suitability boundary for this region.',
    global_baseline_requires_review: 'A global baseline was used without local calibration, so agronomic review is required.',
    outside_model_applicability: 'Inputs are outside the model applicability domain.',
    outside_training_envelope: 'The scenario moves beyond the observed training envelope.',
    baseline_not_supported: 'The baseline prediction for this field is not supported by the model.',
    weekly_profile_insufficient: 'There is not enough reliable weekly profile data for this calculation.',
    support_review_required: 'The result requires additional review before use.',
    preview_contour_requires_confirmation: 'This preview contour is intended for visual review only and requires confirmation.',
  },
}

const LAYER_META = {
  ndvi: { labelKey: 'layers.ndvi' },
  ndmi: { labelKey: 'layers.ndmi' },
  ndwi: { labelKey: 'layers.ndwi' },
  bsi: { labelKey: 'layers.bsi' },
  precipitation: { labelKey: 'layers.precipitation' },
  wind: { labelKey: 'layers.wind' },
  soil_moisture: { labelKey: 'layers.soil_moisture' },
  gdd: { labelKey: 'layers.gdd' },
  vpd: { labelKey: 'layers.vpd' },
}

const FEATURE_LABEL_KEYS = {
  quality_score: 'field.quality',
  field_area_ha: 'field.area',
  compactness: 'field.compactness',
  irrigation_pct: 'field.irrigation',
  fertilizer_pct: 'field.fertilizer',
  expected_rain_mm: 'field.expectedRain',
  temperature_delta_c: 'field.temperatureDelta',
  planting_density_pct: 'field.plantingDensity',
  tillage_type: 'field.tillageType',
  pest_pressure: 'field.pestPressure',
  soil_compaction: 'field.soilCompaction',
  precipitation_mm: 'layers.precipitation',
  soil_moisture: 'layers.soil_moisture',
  cloud_cover_pct: 'weather.cloudCover',
  ndvi_mean: 'layers.ndvi',
  ndmi_mean: 'layers.ndmi',
  ndwi_mean: 'layers.ndwi',
  bsi_mean: 'layers.bsi',
  gdd_mean: 'layers.gdd',
  vpd_mean: 'layers.vpd',
  valid_feature_count: 'field.validFeatures',
  coverage_metrics: 'field.availableMetrics',
  confidence_reason: 'field.confidenceReason',
  crop_suitability_status: 'field.cropSuitability',
  status: 'field.archiveStatus',
  score: 'field.score',
  yield_factor: 'field.suitabilityFactor',
  latitude: 'field.latitude',
  seasonal_precipitation_mm: 'layers.precipitation',
  seasonal_temperature_mean_c: 'field.seasonTemp',
  observed_days: 'field.observedDays',
  warnings: 'field.constraintWarnings',
  baseline_source: 'field.baselineSource',
  counterfactual_mode: 'field.counterfactualMode',
  requires_supported_baseline: 'field.requiresSupportedBaseline',
  response_model: 'field.responseModel',
  interaction_effects: 'field.interactionEffects',
  feature_schema_version: 'field.featureSchemaVersion',
  irrigation_events_total_mm: 'field.irrigationEventsTotal',
  irrigation_events_count: 'field.irrigationEventsCount',
  fertilizer_events_total_n_kg_ha: 'field.fertilizerEventsTotal',
  fertilizer_events_count: 'field.fertilizerEventsCount',
  model_version: 'field.model',
}

const FEATURE_LABELS = {
  ru: {
    hydro_stress: 'Стресс влаги',
    wind_stress: 'Ветровой стресс',
    vegetation_signal: 'Сигнал вегетации',
    area_shape: 'Форма поля',
    climate_suitability: 'Климатическая пригодность',
    soil_profile: 'Почвенный профиль',
    management: 'Управление',
    scenario_fertilizer: 'Сценарное удобрение',
    scenario_rainfall: 'Сценарный дождь',
    scenario_rain: 'Сценарный дождь',
    scenario_irrigation: 'Сценарное орошение',
    scenario_compaction: 'Сценарное уплотнение',
    current_ndvi_mean: 'Текущий NDVI',
    current_ndmi_mean: 'Текущий NDMI',
    current_vpd_mean: 'Текущий VPD',
    current_precipitation_mm: 'Текущие осадки',
    current_wind_speed_m_s: 'Текущая скорость ветра',
    current_soil_moisture: 'Текущая влажность почвы',
    ndvi_auc: 'Накопленный NDVI',
    ndvi_peak: 'Пик NDVI',
    ndvi_mean_season: 'Средний NDVI за сезон',
    seasonal_gdd_sum: 'Сумма GDD за сезон',
    seasonal_vpd_mean: 'Средний VPD за сезон',
    seasonal_observed_days: 'Дней наблюдений за сезон',
    management_total_amount: 'Суммарное воздействие управления',
    management_event_count: 'Число агроопераций',
    crop_baseline: 'Базовый потенциал культуры',
    hist_n_seasons: 'Исторических сезонов',
    hist_latest_year: 'Последний сезон истории',
    soil_ph: 'pH почвы',
    soil_n_ppm: 'Азот в почве',
    soil_p_ppm: 'Фосфор в почве',
    soil_k_ppm: 'Калий в почве',
    soil_organic_matter_pct: 'Органическое вещество почвы',
    soil_texture_code: 'Текстура почвы',
    historical_field_mean_yield: 'Средняя урожайность поля',
    geometry_confidence: 'Уверенность геометрии',
    boundary_uncertainty: 'Неопределённость границы',
    tta_consensus: 'Согласованность TTA',
    longitude: 'Долгота',
  },
  en: {
    hydro_stress: 'Moisture stress',
    wind_stress: 'Wind stress',
    vegetation_signal: 'Vegetation signal',
    area_shape: 'Field shape',
    climate_suitability: 'Climate suitability',
    soil_profile: 'Soil profile',
    management: 'Management',
    scenario_fertilizer: 'Scenario fertilizer',
    scenario_rainfall: 'Scenario rainfall',
    scenario_rain: 'Scenario rainfall',
    scenario_irrigation: 'Scenario irrigation',
    scenario_compaction: 'Scenario compaction',
    current_ndvi_mean: 'Current NDVI',
    current_ndmi_mean: 'Current NDMI',
    current_vpd_mean: 'Current VPD',
    current_precipitation_mm: 'Current precipitation',
    current_wind_speed_m_s: 'Current wind speed',
    current_soil_moisture: 'Current soil moisture',
    ndvi_auc: 'Accumulated NDVI',
    ndvi_peak: 'NDVI peak',
    ndvi_mean_season: 'Season mean NDVI',
    seasonal_gdd_sum: 'Seasonal GDD sum',
    seasonal_vpd_mean: 'Season mean VPD',
    seasonal_observed_days: 'Seasonal observed days',
    management_total_amount: 'Total management intensity',
    management_event_count: 'Management event count',
    crop_baseline: 'Crop baseline potential',
    hist_n_seasons: 'Historical seasons',
    hist_latest_year: 'Latest history year',
    soil_ph: 'Soil pH',
    soil_n_ppm: 'Soil nitrogen',
    soil_p_ppm: 'Soil phosphorus',
    soil_k_ppm: 'Soil potassium',
    soil_organic_matter_pct: 'Soil organic matter',
    soil_texture_code: 'Soil texture',
    historical_field_mean_yield: 'Historical field mean yield',
    geometry_confidence: 'Geometry confidence',
    boundary_uncertainty: 'Boundary uncertainty',
    tta_consensus: 'TTA consensus',
    longitude: 'Longitude',
  },
}

const QUALITY_BAND_KEYS = {
  high: 'field.qualityHigh',
  medium: 'field.qualityMedium',
  low: 'field.qualityLow',
  manual: 'field.qualityManual',
  unknown: 'field.qualityUnknownShort',
}

const SOURCE_LABELS = {
  ru: {
    autodetect: 'Автодетект',
    autodetect_preview: 'Preview-контур',
    manual: 'Ручная разметка',
    import: 'Импорт',
    merged: 'Объединение',
  },
  en: {
    autodetect: 'Autodetect',
    autodetect_preview: 'Preview contour',
    manual: 'Manual markup',
    import: 'Import',
    merged: 'Merged',
  },
}

const CONFIDENCE_TIER_LABELS = {
  ru: {
    tenant_calibrated: 'Локально калиброванный',
    global_baseline: 'Глобальный baseline',
    unsupported: 'Вне применимости',
  },
  en: {
    tenant_calibrated: 'Tenant calibrated',
    global_baseline: 'Global baseline',
    unsupported: 'Unsupported',
  },
}

const RISK_LEVELS = {
  ru: {
    minimal: 'Минимальный',
    low: 'Низкий',
    moderate: 'Умеренный',
    elevated: 'Повышенный',
    high: 'Высокий',
    critical: 'Критический',
    unknown: 'Неопределённый',
  },
  en: {
    minimal: 'Minimal',
    low: 'Low',
    moderate: 'Moderate',
    elevated: 'Elevated',
    high: 'High',
    critical: 'Critical',
    unknown: 'Unknown',
  },
}

const RISK_ITEM_LABELS = {
  ru: {
    drought_risk: 'Риск засухи',
    vegetation_stress: 'Стресс вегетации',
    fhb_risk: 'Риск фузариоза колоса',
  },
  en: {
    drought_risk: 'Drought risk',
    vegetation_stress: 'Vegetation stress',
    fhb_risk: 'FHB risk',
  },
}

const RISK_ITEM_REASONS = {
  ru: {
    drought_risk: 'В корневом слое нарастает дефицит влаги.',
    vegetation_stress: 'Спутниковые индексы показывают ухудшение состояния растительности.',
    fhb_risk: 'Текущая фаза и влажный режим повышают фитосанитарный риск.',
  },
  en: {
    drought_risk: 'The root zone is moving into a meaningful moisture deficit.',
    vegetation_stress: 'Satellite vegetation indices show worsening crop condition.',
    fhb_risk: 'The current stage and humid conditions raise disease risk.',
  },
}

function currentLang() {
  return locale.value === 'en' ? 'en' : 'ru'
}

function dictEntry(dictionary, key) {
  const lang = currentLang()
  return dictionary?.[lang]?.[key]
}

function normalizeFeatureKey(key) {
  return String(key || '')
    .trim()
    .replace(/^_+/, '')
}

function humanizeKey(key) {
  const normalized = normalizeFeatureKey(key)
    .replace(/[_-]+/g, ' ')
    .trim()
  if (!normalized) return '—'
  return normalized.charAt(0).toUpperCase() + normalized.slice(1)
}

export function getUiLocale() {
  return UI_LOCALES[currentLang()]
}

export function formatUiProgress(value) {
  const numeric = Number(value)
  if (!Number.isFinite(numeric)) return '0.00'
  return new Intl.NumberFormat(getUiLocale(), {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(numeric)
}

export function formatUiDateTime(value, options = {}) {
  if (!value) return '—'
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return '—'
  return parsed.toLocaleString(getUiLocale(), options)
}

export function formatUiTime(value, options = {}) {
  if (!value) return '—'
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return '—'
  return parsed.toLocaleTimeString(getUiLocale(), options)
}

export function getDetectionPresetMeta(preset) {
  return dictEntry(PRESET_META, preset) || dictEntry(PRESET_META, 'standard')
}

export function getLayerMeta(layerId, fallback = {}) {
  const meta = LAYER_META[layerId]
  return {
    label: meta?.labelKey ? t(meta.labelKey) : (fallback.name || humanizeKey(layerId)),
    description: fallback.description || '',
  }
}

export function getQualityBandLabel(band) {
  const key = QUALITY_BAND_KEYS[String(band || '').toLowerCase()]
  return key ? t(key) : humanizeKey(band)
}

export function getSourceLabel(source) {
  return dictEntry(SOURCE_LABELS, String(source || '').toLowerCase()) || humanizeKey(source)
}

export function getConfidenceTierLabel(tier) {
  return dictEntry(CONFIDENCE_TIER_LABELS, String(tier || '').toLowerCase()) || humanizeKey(tier)
}

export function getRiskLevelLabel(level) {
  const raw = String(level || '').trim().toLowerCase()
  const normalized = {
    минимальный: 'minimal',
    низкий: 'low',
    умеренный: 'moderate',
    повышенный: 'elevated',
    высокий: 'high',
    критический: 'critical',
    неопределённый: 'unknown',
    неопределенный: 'unknown',
  }[raw] || raw
  return dictEntry(RISK_LEVELS, normalized) || humanizeKey(level)
}

export function getRiskItemLabel(item) {
  return dictEntry(RISK_ITEM_LABELS, String(item?.id || '').toLowerCase()) || item?.label || humanizeKey(item?.id)
}

export function getRiskItemReason(item) {
  return dictEntry(RISK_ITEM_REASONS, String(item?.id || '').toLowerCase()) || item?.reason || '—'
}

export function formatReasonText(code, fallback, params = {}) {
  const template = dictEntry(REASON_TEXT, String(code || '').toLowerCase())
  if (!template) return fallback || ''
  if (code === 'budget_guardrail_warning' && Number.isFinite(Number(params?.warnings_count)) && Number(params.warnings_count) > 0) {
    return currentLang() === 'ru'
      ? `${template} Предупреждений: ${Number(params.warnings_count)}.`
      : `${template} Warnings: ${Number(params.warnings_count)}.`
  }
  return template
}

export function getTaskStageLabel(payloadOrCode, fallbackLabel = '') {
  const code = typeof payloadOrCode === 'string'
    ? payloadOrCode
    : (payloadOrCode?.stage_code || payloadOrCode?.stage_label || payloadOrCode?.status)
  const normalized = String(code || '').trim().toLowerCase().replace(/[^a-z0-9]+/g, '_')
  return dictEntry(TASK_STAGE_LABELS, normalized) || fallbackLabel || humanizeKey(normalized)
}

function detailTemplate(code, params = {}) {
  const lang = currentLang()
  const current = Number(params.current)
  const total = Number(params.total)
  if (code === 'waiting_for_worker') {
    return lang === 'ru' ? 'Ожидает свободный worker.' : 'Waiting for a free worker.'
  }
  if (code === 'initializing_context') {
    return lang === 'ru' ? 'Подготавливаю контекст расчёта.' : 'Initializing calculation context.'
  }
  if (code === 'field_crop_resolution') {
    return lang === 'ru' ? 'Проверяю поле и культуру.' : 'Resolving field and crop.'
  }
  if (code === 'materializing_weekly_profile') {
    return lang === 'ru' ? 'Собираю недельный профиль поля.' : 'Building the weekly field profile.'
  }
  if (code === 'checking_cached_prediction') {
    return lang === 'ru' ? 'Проверяю, есть ли актуальный кэш прогноза.' : 'Checking for a current cached prediction.'
  }
  if (code === 'weather_and_field_analytics') {
    return lang === 'ru' ? 'Собираю погодные и полевые признаки.' : 'Collecting weather and field features.'
  }
  if (code === 'writing_prediction_snapshot') {
    return lang === 'ru' ? 'Сохраняю прогноз и объяснение.' : 'Persisting the prediction snapshot.'
  }
  if (code === 'writing_scenario_snapshot') {
    return lang === 'ru' ? 'Сохраняю сценарий и сравнительные метрики.' : 'Persisting the scenario snapshot.'
  }
  if (code === 'loading_baseline_prediction') {
    return lang === 'ru' ? 'Загружаю базовый прогноз.' : 'Loading the baseline prediction.'
  }
  if (code === 'agronomic_response_evaluation') {
    return lang === 'ru' ? 'Оцениваю агрономический отклик сценария.' : 'Evaluating agronomic scenario response.'
  }
  if (code === 'windows_progress' && Number.isFinite(current) && Number.isFinite(total)) {
    return lang === 'ru'
      ? `Окна наблюдений: ${current}/${total}`
      : `Observation windows: ${current}/${total}`
  }
  if (code === 'tile_progress' && Number.isFinite(current) && Number.isFinite(total)) {
    return lang === 'ru'
      ? `Тайл: ${current}/${total}`
      : `Tile: ${current}/${total}`
  }
  if (code === 'tiles_merged') {
    return lang === 'ru'
      ? `Объединено тайлов: ${Number(params.merged_tiles || 0)}`
      : `Merged tiles: ${Number(params.merged_tiles || 0)}`
  }
  if (code === 'db_insert_counts') {
    return lang === 'ru'
      ? `Сетка: ${Number(params.grid_cells || 0)}, кандидаты: ${Number(params.candidates || 0)}`
      : `Grid cells: ${Number(params.grid_cells || 0)}, candidates: ${Number(params.candidates || 0)}`
  }
  if (code === 'failure_stage' && params.failure_stage) {
    return lang === 'ru'
      ? `Сбой на этапе: ${humanizeKey(params.failure_stage)}`
      : `Failed at stage: ${humanizeKey(params.failure_stage)}`
  }
  if (code === 'postprocess_start') {
    return lang === 'ru' ? 'Запускаю очистку и сшивку кандидатов.' : 'Starting candidate cleanup and stitching.'
  }
  if (code === 'watershed') {
    return lang === 'ru' ? 'Применяю watershed-разделение.' : 'Applying watershed separation.'
  }
  if (code?.endsWith('_progress') && Number.isFinite(current) && Number.isFinite(total)) {
    const action = code.replace(/_progress$/, '')
    return lang === 'ru'
      ? `${humanizeKey(action)}: ${current}/${total}`
      : `${humanizeKey(action)}: ${current}/${total}`
  }
  return ''
}

export function getTaskStageDetail(payload) {
  const detailCode = payload?.stage_detail_code
  const detailParams = payload?.stage_detail_params || {}
  const templated = detailCode ? detailTemplate(detailCode, detailParams) : ''
  return templated || payload?.stage_detail || ''
}

export function getFeatureLabel(key, options = {}) {
  const rawKey = String(key || '')
  const keyName = normalizeFeatureKey(rawKey)
  const lookupKey = keyName.toLowerCase()
  if (QUALITY_BAND_KEYS[lookupKey]) {
    return getQualityBandLabel(lookupKey)
  }
  if (LAYER_META[lookupKey]?.labelKey) {
    return t(LAYER_META[lookupKey].labelKey)
  }
  const directLabel = dictEntry(FEATURE_LABELS, lookupKey)
  if (directLabel) return directLabel
  const tKey = FEATURE_LABEL_KEYS[lookupKey]
  if (tKey) return t(tKey)
  if (options.expertMode) return keyName || rawKey
  return humanizeKey(keyName)
}

function formatEnumValue(key, value) {
  const lang = currentLang()
  if (key === 'baseline_source') {
    const map = {
      latest_prediction: lang === 'ru' ? 'Последний рассчитанный прогноз' : 'Latest stored prediction',
      weekly_profile: lang === 'ru' ? 'Недельный профиль поля' : 'Weekly field profile',
    }
    return map[String(value)] || humanizeKey(value)
  }
  if (key === 'counterfactual_mode') {
    const map = {
      agronomic_response_v3: lang === 'ru' ? 'Табличная модель отклика факторов' : 'Tabular factor-response engine',
      mechanistic_weekly: lang === 'ru' ? 'Недельная механистическая модель' : 'Weekly mechanistic engine',
    }
    return map[String(value)] || humanizeKey(value)
  }
  if (key === 'response_model') {
    const map = {
      'Mitscherlich + Liebig minimum law': lang === 'ru' ? 'Кривая отклика Митчерлиха и правило минимума Либиха' : 'Mitscherlich response curve with Liebig minimum law',
      'weekly mechanistic baseline': lang === 'ru' ? 'Недельный механистический базовый расчёт' : 'Weekly mechanistic baseline',
    }
    return map[String(value)] || String(value)
  }
  if (key === 'tillage_type') {
    const map = {
      0: lang === 'ru' ? 'Без обработки' : 'No tillage',
      1: lang === 'ru' ? 'Минимальная обработка' : 'Minimum tillage',
      2: lang === 'ru' ? 'Обычная обработка' : 'Conventional tillage',
      3: lang === 'ru' ? 'Глубокая обработка' : 'Deep tillage',
    }
    return map[Number(value)] || humanizeKey(value)
  }
  if (key === 'pest_pressure') {
    const map = {
      0: lang === 'ru' ? 'Нет' : 'None',
      1: lang === 'ru' ? 'Низкое' : 'Low',
      2: lang === 'ru' ? 'Среднее' : 'Medium',
      3: lang === 'ru' ? 'Высокое' : 'High',
    }
    return map[Number(value)] || humanizeKey(value)
  }
  if (key === 'confidence_tier') {
    return getConfidenceTierLabel(value)
  }
  return null
}

export function formatDisplayValue(key, value, options = {}) {
  const keyName = normalizeFeatureKey(key).toLowerCase()
  if (value === null || value === undefined || value === '') return '—'
  if (typeof value === 'boolean') {
    return value ? t('field.yes') : t('field.no')
  }
  if (
    keyName === 'prediction_interval' &&
    value &&
    typeof value === 'object' &&
    Number.isFinite(Number(value.lower)) &&
    Number.isFinite(Number(value.upper))
  ) {
    const lower = Number(value.lower).toFixed(0)
    const upper = Number(value.upper).toFixed(0)
    return currentLang() === 'ru' ? `${lower} … ${upper} кг/га` : `${lower} … ${upper} kg/ha`
  }
  const enumValue = formatEnumValue(keyName, value)
  if (enumValue !== null) return enumValue
  if (typeof value === 'number') {
    const lang = currentLang()
    const yieldUnit = lang === 'ru' ? 'кг/га' : 'kg/ha'
    const windUnit = lang === 'ru' ? 'м/с' : 'm/s'
    if (keyName === 'field_area_ha') return lang === 'ru' ? `${Number(value).toFixed(2)} га` : `${Number(value).toFixed(2)} ha`
    if (keyName === 'temperature_delta_c') return `${Number(value).toFixed(1)} °C`
    if (['planting_density_pct', 'irrigation_pct', 'fertilizer_pct', 'soil_organic_matter_pct'].includes(keyName)) return `${Number(value).toFixed(1)}%`
    if (keyName === 'soil_compaction') return Number(value).toFixed(2)
    if (['valid_feature_count', 'management_event_count', 'seasonal_observed_days', 'observed_days', 'hist_n_seasons', 'hist_latest_year'].includes(keyName)) {
      return String(Math.round(Number(value)))
    }
    if (keyName === 'cloud_cover_pct') return `${Number(value).toFixed(0)}%`
    if (['precipitation_mm', 'expected_rain_mm', 'irrigation_events_total_mm', 'current_precipitation_mm', 'seasonal_precipitation_mm'].includes(keyName)) {
      return `${Number(value).toFixed(1)} ${lang === 'ru' ? 'мм' : 'mm'}`
    }
    if (keyName === 'fertilizer_events_total_n_kg_ha') return `${Number(value).toFixed(1)} ${yieldUnit} N`
    if (['temperature_c', 'seasonal_temperature_mean_c'].includes(keyName)) return `${Number(value).toFixed(1)} °C`
    if (['soil_moisture', 'current_soil_moisture', 'geometry_confidence', 'tta_consensus', 'boundary_uncertainty'].includes(keyName)) {
      return `${(Number(value) * 100).toFixed(1)}%`
    }
    if (keyName === 'current_wind_speed_m_s' || keyName === 'wind') return `${Number(value).toFixed(1)} ${windUnit}`
    if (['precipitation', 'gdd', 'seasonal_gdd_sum'].includes(keyName)) return Number(value).toFixed(0)
    if (['soil_ph', 'latitude', 'longitude', 'current_vpd_mean', 'seasonal_vpd_mean', 'vpd'].includes(keyName)) return Number(value).toFixed(3)
    if (['soil_n_ppm', 'soil_p_ppm', 'soil_k_ppm'].includes(keyName)) return `${Number(value).toFixed(0)} ppm`
    if (keyName === 'crop_baseline' || keyName === 'historical_field_mean_yield') return `${Number(value).toFixed(0)} ${yieldUnit}`
    if (['ndvi', 'ndmi', 'ndwi', 'bsi', 'compactness', 'score', 'yield_factor', 'current_ndvi_mean', 'current_ndmi_mean', 'ndvi_auc', 'ndvi_peak', 'ndvi_mean_season'].includes(keyName)) {
      return Number(value).toFixed(3)
    }
    return Number(value).toFixed(2)
  }
  if (Array.isArray(value)) {
    if (!value.length) return '—'
    if (keyName === 'coverage_metrics') {
      return value.map((item) => getLayerMeta(item).label).join(', ')
    }
    return value.map((item) => formatDisplayValue(keyName, item, options)).join(', ')
  }
  if (QUALITY_BAND_KEYS[String(value).toLowerCase()]) {
    return getQualityBandLabel(value)
  }
  return options.expertMode ? String(value) : String(value)
}
