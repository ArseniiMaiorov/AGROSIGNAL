import { defineStore } from 'pinia'
import { computed, ref, watch } from 'vue'
import axios, { API_BASE } from '../services/api'
import {
  formatUiDateTime,
  formatUiProgress,
  getDetectionPresetMeta,
  getTaskStageDetail,
  getTaskStageLabel,
} from '../utils/presentation'
import { t } from '../utils/i18n'

const SETTINGS_STORAGE_KEY = 'agrovision-ui-settings-v3'
const LEGACY_SETTINGS_STORAGE_KEY = 'agrovision-ui-settings-v2'
const LOGS_STORAGE_KEY = 'agrovision-event-log-v1'
const LOG_ENTRY_LIMIT = 240
const TARGET_DATES_MIN = 1
const TARGET_DATES_MAX = 12
const DETECTION_PRESETS = {
  fast: {
    resolutionM: 10,
    targetDates: 4,
    minFieldAreaHa: 0.5,
    useSam: false,
    maxRadiusKm: 40,
    recommendedRadiusKm: 30,
    outputMode: 'preview_agri_contours',
    operationalEligible: false,
  },
  standard: {
    resolutionM: 10,
    targetDates: 7,
    minFieldAreaHa: 0.25,
    useSam: false,
    maxRadiusKm: 20,
    recommendedRadiusKm: 20,
    outputMode: 'field_boundaries',
    operationalEligible: true,
  },
  quality: {
    resolutionM: 10,
    targetDates: 9,
    minFieldAreaHa: 0.1,
    useSam: true,
    maxRadiusKm: 8,
    recommendedRadiusKm: 8,
    outputMode: 'field_boundaries_hifi',
    operationalEligible: true,
  },
}

function isPreviewFieldSource(source) {
  return String(source || '').trim().toLowerCase() === 'autodetect_preview'
}

function inferLogSeverity(message, explicitSeverity = '') {
  const normalized = String(explicitSeverity || '').trim().toLowerCase()
  if (['info', 'warning', 'error'].includes(normalized)) {
    return normalized
  }
  const lower = String(message || '').toLowerCase()
  const errorPatterns = ['ошибка', 'error', 'не удалось', 'failed', 'exception', 'traceback']
  const warningPatterns = [
    'предупреж',
    'warning',
    'preview-only',
    'preview',
    'review_needed',
    'review required',
    'review',
    'fallback',
    'cooldown',
    'degraded',
    'ограничен',
    'недоступ',
    'stale',
    'quota',
    'скорректирован',
    'downgraded',
  ]
  if (errorPatterns.some((pattern) => lower.includes(pattern))) {
    return 'error'
  }
  if (warningPatterns.some((pattern) => lower.includes(pattern))) {
    return 'warning'
  }
  return 'info'
}

function inferLogCategory(message, explicitCategory = '') {
  const normalized = String(explicitCategory || '').trim().toLowerCase()
  if (normalized) {
    return normalized
  }
  const lower = String(message || '').toLowerCase()
  if (lower.includes('debug')) return 'debug'
  if (lower.includes('спутник') || lower.includes('sentinel') || lower.includes('scene')) return 'satellite'
  if (lower.includes('слой') || lower.includes('layer')) return 'layers'
  if (lower.includes('погод') || lower.includes('weather') || lower.includes('gdd')) return 'weather'
  if (lower.includes('автодетек') || lower.includes('preflight') || lower.includes('detect') || lower.includes('тайл')) return 'detect'
  if (lower.includes('аналитик') || lower.includes('temporal')) return 'analytics'
  if (lower.includes('прогноз') || lower.includes('prediction')) return 'prediction'
  if (lower.includes('сценар') || lower.includes('чувствительн') || lower.includes('modeling')) return 'modeling'
  if (lower.includes('архив') || lower.includes('archive')) return 'archive'
  if (lower.includes('событи') || lower.includes('manual') || lower.includes('объедин') || lower.includes('раздел') || lower.includes('удал')) return 'manual'
  return 'system'
}

function normalizeLogEntry(entry) {
  if (typeof entry === 'string') {
    const match = entry.match(/^\[(.+?)\]\s*(.*)$/)
    const message = (match?.[2] || entry).trim()
    return {
      ts: new Date().toISOString(),
      message,
      severity: inferLogSeverity(message),
      category: inferLogCategory(message),
      code: null,
      runId: null,
      fieldId: null,
      taskId: null,
    }
  }
  const payload = entry && typeof entry === 'object' ? entry : { message: String(entry || '') }
  const message = String(payload.message || '').trim()
  return {
    ts: payload.ts || new Date().toISOString(),
    message,
    severity: inferLogSeverity(message, payload.severity),
    category: inferLogCategory(message, payload.category),
    code: payload.code || null,
    runId: payload.runId || null,
    fieldId: payload.fieldId || null,
    taskId: payload.taskId || null,
  }
}

function hasForecastCurve(payload) {
  return Boolean(payload?.forecast_curve?.points?.length)
}

function mergePredictionPayload(preferred, fallback) {
  if (!preferred) return fallback || null
  if (!fallback) return preferred
  return {
    ...fallback,
    ...preferred,
    crop: preferred.crop || fallback.crop || null,
    details: { ...(fallback.details || {}), ...(preferred.details || {}) },
    explanation: { ...(fallback.explanation || {}), ...(preferred.explanation || {}) },
    data_quality: { ...(fallback.data_quality || {}), ...(preferred.data_quality || {}) },
    input_features: { ...(fallback.input_features || {}), ...(preferred.input_features || {}) },
    forecast_curve: hasForecastCurve(preferred)
      ? preferred.forecast_curve
      : (hasForecastCurve(fallback) ? fallback.forecast_curve : (preferred.forecast_curve || fallback.forecast_curve || {})),
    history_trend: preferred?.history_trend?.points?.length
      ? preferred.history_trend
      : (fallback.history_trend || preferred.history_trend || {}),
  }
}

export const useMapStore = defineStore('map', () => {
  const centerLat = ref(58.689077)
  const centerLon = ref(29.892103)
  const radiusKm = ref(15)
  const startDate = ref('2025-05-01')
  const endDate = ref('2025-08-31')
  const resolutionM = ref(10)
  const maxCloudPct = ref(40)
  const targetDates = ref(7)
  const minFieldAreaHa = ref(0.5)
  const useSam = ref(false)
  const showFieldsOnly = ref(true)
  const showFieldBoundaries = ref(true)
  const drawMode = ref(false)
  const mergeMode = ref(false)
  const splitMode = ref(false)
  const mergeSelectionIds = ref([])
  const isPickingSearchCenter = ref(false)

  const activeRunId = ref(null)
  const visibleRunId = ref(null)
  const lastCompletedRunId = ref(null)
  const currentRunId = computed({
    get: () => visibleRunId.value,
    set: (value) => {
      visibleRunId.value = value
      if (value) {
        lastCompletedRunId.value = value
      }
    },
  })
  const runSummaries = ref([])
  const runStatus = ref(null)
  const runProgress = ref(0)
  const runRuntime = ref(null)
  const lastPreflight = ref(null)
  const runStageCode = ref(null)
  const runStageLabel = ref(null)
  const runStageDetailCode = ref(null)
  const runStageDetailParams = ref({})
  const runStageDetail = ref(null)
  const runStartedAt = ref(null)
  const runUpdatedAt = ref(null)
  const runLastHeartbeatTs = ref(null)
  const runStaleRunning = ref(false)
  const runEstimatedRemainingS = ref(null)
  const debugTilesCatalog = ref([])
  const selectedDebugRunId = ref(null)
  const selectedDebugTileId = ref('')
  const selectedDebugLayerId = ref('')
  const selectedDebugTileDetail = ref(null)
  const debugOverlayEnabled = ref(false)
  const debugOverlayOpacity = ref(0.5)
  const debugLayerPayload = ref(null)
  const isLoadingDebugTiles = ref(false)
  const isLoadingDebugLayer = ref(false)
  const fieldsGeoJson = ref(null)
  const manualFieldsGeoJson = ref({ type: 'FeatureCollection', features: [] })
  const availableLayers = ref([])
  const activeLayers = ref({
    ndvi: true,
    ndmi: false,
    ndwi: false,
    bsi: false,
    gdd: false,
    vpd: false,
    precipitation: false,
    wind: false,
    soil_moisture: false,
  })
  const showSatelliteBrowse = ref(false)
  const satelliteBrowseDate = ref('')
  const satelliteScene = ref(null)
  const satelliteLoadStatus = ref('idle') // idle | loading | ready | no_data | error

  const isDetecting = ref(false)
  const isLoadingWeather = ref(false)
  const isCreatingManualField = ref(false)
  const error = ref(null)
  const logs = ref([])
  const weatherCurrent = ref(null)
  const weatherForecast = ref([])
  const systemStatus = ref(null)
  const lastWeatherUpdatedAt = ref(null)
  const lastStatusUpdatedAt = ref(null)
  const lastWeatherSyncState = ref('idle')
  const lastWeatherSyncDetail = ref('')
  const lastStatusSyncState = ref('idle')
  const lastStatusSyncDetail = ref('')
  const selectedField = ref(null)
  const selectedFieldIds = ref([])
  const fieldDashboard = ref(null)
  const groupDashboard = ref(null)
  const fieldTemporalAnalytics = ref(null)
  const fieldForecastAnalytics = ref(null)
  const fieldTemporalAnalyticsKey = ref('')
  const fieldForecastAnalyticsKey = ref('')
  const fieldManagementZones = ref(null)
  const selectedFieldPrediction = ref(null)
  const crops = ref([])
  const selectedCropCode = ref('wheat')
  const isRefreshingPrediction = ref(false)
  const isSimulatingScenario = ref(false)
  const predictionTaskProgress = ref(0)
  const scenarioTaskProgress = ref(0)
  const temporalAnalyticsTaskProgress = ref(0)
  const predictionTaskState = ref(null)
  const scenarioTaskState = ref(null)
  const temporalAnalyticsTaskState = ref(null)
  const isCreatingArchive = ref(false)
  const isLoadingFieldDashboard = ref(false)
  const isLoadingGroupDashboard = ref(false)
  const isLoadingArchiveView = ref(false)
  const isLoadingMetricsAnalytics = ref(false)
  const isLoadingForecastAnalytics = ref(false)
  const isLoadingTemporalAnalytics = computed(() => isLoadingMetricsAnalytics.value || isLoadingForecastAnalytics.value)
  const isLoadingManagementZones = ref(false)
  const gridLayerStatus = ref('idle') // idle | loading | ready | no_data | error
  const modelingForm = ref({
    irrigation_pct: 10,
    fertilizer_pct: 5,
    expected_rain_mm: 20,
    temperature_delta_c: 0,
    planting_density_pct: 0,
    tillage_type: null,
    pest_pressure: null,
    soil_compaction: null,
    cloud_cover_factor: 1.0,
  })
  const modelingAutoSources = ref({
    expected_rain_mm: '',
    soil_compaction: '',
  })
  const scenarioName = ref('')
  const modelingResult = ref(null)
  const sensitivityData = ref(null)
  const useManualModeling = ref(false)
  const isLoadingSensitivity = ref(false)
  const fieldScenarios = ref([])
  const fieldArchives = ref([])
  const selectedArchiveView = ref(null)
  const activeFieldTab = ref('overview')
  const metricsDisplayMode = ref('cards')
  const metricsSelectedSeries = ref('ndvi')
  const forecastGraphMode = ref('xy')
  const scenarioGraphMode = ref('xy')
  const showForecastGraphs = ref(true)
  const showScenarioGraphs = ref(true)
  const showScenarioFactors = ref(true)
  const showScenarioRisks = ref(true)
  const showManagementZonesOverlay = ref(false)
  const fieldEvents = ref([])
  const fieldEventsTotal = ref(0)
  const isLoadingEvents = ref(false)
  const isSubmittingEvent = ref(false)
  const selectedEventSeasonYear = ref(null)
  const seriesDateFrom = ref('')
  const seriesDateTo = ref('')
  const detectionPreset = ref('standard')
  const autoRefreshIntervalS = ref(60)
  const progressVerbosity = ref('detailed')
  const animationDensity = ref('normal')
  const showFreshnessBadges = ref(true)
  const mapLabelDensity = ref('compact')
  const expertMode = ref(false)
  const beginnerMode = ref(false)
  const uiWindows = ref({
    control: true,
    status: true,
    weather: true,
    fieldActions: false,
    legend: false,
    logs: false,
    help: false,
  })

  let statusTimer = null
  let systemTimer = null
  let predictionTimer = null
  let scenarioTimer = null
  let temporalTimer = null
  let lastLoggedProgress = null
  let lastLoggedStatus = null
  let queuedDispatchWarningRunId = null
  let systemPollFailCount = 0
  let statusPollFailCount = 0
  let lastSystemErrorMsg = ''
  const requestControllers = new Map()
  const stateSignatures = new Map()

  // Spectral layers are mutually exclusive; weather/climate layers can coexist
  const SPECTRAL_LAYERS = new Set(['ndvi', 'ndmi', 'ndwi', 'bsi'])
  const WEATHER_LAYERS = new Set(['precipitation', 'wind', 'soil_moisture', 'gdd', 'vpd'])

  const activeLayerIds = computed(() => {
    return Object.entries(activeLayers.value)
      .filter(([, v]) => v)
      .map(([k]) => k)
  })

  const primaryLayerId = computed(() => {
    const spectral = activeLayerIds.value.find((id) => SPECTRAL_LAYERS.has(id))
    if (spectral) return spectral
    return activeLayerIds.value[0] || 'ndvi'
  })
  const selectedFieldCount = computed(() => selectedFieldIds.value.length)
  const hasGroupSelection = computed(() => selectedFieldIds.value.length > 1)
  const activeDashboard = computed(() => (hasGroupSelection.value ? groupDashboard.value : fieldDashboard.value))
  const activeDetectionPreset = computed(() => {
    const presetId = DETECTION_PRESETS[detectionPreset.value] ? detectionPreset.value : 'standard'
    return {
      ...(DETECTION_PRESETS[presetId] || DETECTION_PRESETS.standard),
      ...getDetectionPresetMeta(presetId),
    }
  })
  const selectedFieldSource = computed(() => {
    return fieldDashboard.value?.field?.source || selectedField.value?.source || ''
  })
  const selectedFieldIsPreviewOnly = computed(() => {
    return !hasGroupSelection.value && isPreviewFieldSource(selectedFieldSource.value)
  })

  let _persistSettingsTimer = null
  function debouncedPersistUiSettings() {
    if (_persistSettingsTimer) clearTimeout(_persistSettingsTimer)
    _persistSettingsTimer = setTimeout(persistUiSettings, 500)
  }

  watch(
    [
      centerLat,
      centerLon,
      radiusKm,
      startDate,
      endDate,
      resolutionM,
      maxCloudPct,
      targetDates,
      minFieldAreaHa,
      selectedCropCode,
      metricsDisplayMode,
      metricsSelectedSeries,
      forecastGraphMode,
      scenarioGraphMode,
      showForecastGraphs,
      showScenarioGraphs,
      showScenarioFactors,
      showScenarioRisks,
      showManagementZonesOverlay,
      fieldEvents,
      fieldEventsTotal,
      isLoadingEvents,
      isSubmittingEvent,
      selectedEventSeasonYear,
      useSam,
      showFieldsOnly,
      detectionPreset,
      autoRefreshIntervalS,
      progressVerbosity,
      animationDensity,
      showFreshnessBadges,
      mapLabelDensity,
      expertMode,
      activeLayers,
      showSatelliteBrowse,
      satelliteBrowseDate,
      uiWindows,
    ],
    debouncedPersistUiSettings,
    { deep: true }
  )

  let _persistLogsTimer = null
  function debouncedPersistLogs() {
    if (_persistLogsTimer) clearTimeout(_persistLogsTimer)
    _persistLogsTimer = setTimeout(persistLogs, 1000)
  }

  function addLog(message) {
    logs.value.push(normalizeLogEntry(message))
    if (logs.value.length > LOG_ENTRY_LIMIT) {
      logs.value = logs.value.slice(-LOG_ENTRY_LIMIT)
    }
    debouncedPersistLogs()
  }

  function persistLogs() {
    if (typeof window === 'undefined') {
      return
    }
    window.localStorage.setItem(LOGS_STORAGE_KEY, JSON.stringify(logs.value.slice(-LOG_ENTRY_LIMIT)))
  }

  function clearLogs() {
    logs.value = []
    if (typeof window !== 'undefined') {
      window.localStorage.removeItem(LOGS_STORAGE_KEY)
    }
  }

  function restoreLogs() {
    if (typeof window === 'undefined') {
      return
    }
    const raw = window.localStorage.getItem(LOGS_STORAGE_KEY)
    if (!raw) {
      return
    }
    try {
      const parsed = JSON.parse(raw)
      if (Array.isArray(parsed)) {
        logs.value = parsed.map((item) => normalizeLogEntry(item)).slice(-LOG_ENTRY_LIMIT)
      }
    } catch {
      window.localStorage.removeItem(LOGS_STORAGE_KEY)
    }
  }

  function listKnownFieldProperties() {
    const autoFeatures = fieldsGeoJson.value?.features || []
    const manualFeatures = manualFieldsGeoJson.value?.features || []
    return [...autoFeatures, ...manualFeatures]
      .map((feature) => feature?.properties || null)
      .filter(Boolean)
  }

  function findFieldProperties(fieldId) {
    if (!fieldId) {
      return null
    }
    return listKnownFieldProperties().find((item) => item.field_id === fieldId) || null
  }

  function syncSelectedFieldFromCollections(preferredFieldId = null) {
    if (!selectedFieldIds.value.length) {
      selectedField.value = null
      return
    }

    const validIds = selectedFieldIds.value.filter((fieldId) => Boolean(findFieldProperties(fieldId)))
    if (validIds.length !== selectedFieldIds.value.length) {
      selectedFieldIds.value = validIds
    }
    if (!selectedFieldIds.value.length) {
      selectedField.value = null
      return
    }

    const anchorId =
      (preferredFieldId && selectedFieldIds.value.includes(preferredFieldId) && preferredFieldId) ||
      (selectedField.value?.field_id && selectedFieldIds.value.includes(selectedField.value.field_id) && selectedField.value.field_id) ||
      selectedFieldIds.value[selectedFieldIds.value.length - 1]
    selectedField.value = findFieldProperties(anchorId)
  }

  function clampNumber(value, fallback, min, max) {
    const parsed = Number(value)
    if (!Number.isFinite(parsed)) {
      return fallback
    }
    return Math.min(max, Math.max(min, parsed))
  }

  function nextRequestConfig(key) {
    cancelRequest(key)
    const controller = new AbortController()
    requestControllers.set(key, controller)
    return { signal: controller.signal }
  }

  function cancelRequest(key) {
    const controller = requestControllers.get(key)
    if (controller) {
      controller.abort()
      requestControllers.delete(key)
    }
  }

  function cancelAllRequests() {
    for (const controller of requestControllers.values()) {
      controller.abort()
    }
    requestControllers.clear()
  }

  function isAbortError(requestError) {
    return requestError?.code === 'ERR_CANCELED' || requestError?.name === 'CanceledError'
  }

  function logProgressUpdate(payload) {
    const stage = getTaskStageLabel(payload, payload.stage_label || payload.status)
    const detailLabel = getTaskStageDetail(payload)
    const stagePct = Number(payload.stage_progress_pct)
    const tilePct = Number(payload.tile_progress_pct)
    const localProgressParts = []
    if (Number.isFinite(tilePct)) {
      localProgressParts.push(`тайл ${formatUiProgress(tilePct)}%`)
    }
    if (Number.isFinite(stagePct)) {
      localProgressParts.push(`этап ${formatUiProgress(stagePct)}%`)
    }
    const detailParts = []
    if (detailLabel) {
      detailParts.push(detailLabel)
    }
    if (localProgressParts.length) {
      detailParts.push(localProgressParts.join(' · '))
    }
    const detail = detailParts.length ? ` · ${detailParts.join(' · ')}` : ''
    const message =
      progressVerbosity.value === 'detailed'
        ? `Прогресс: ${formatUiProgress(payload.progress_pct ?? payload.progress)}% (${payload.status}) · ${stage}${detail}`
        : `Прогресс: ${formatUiProgress(payload.progress_pct ?? payload.progress)}% (${stage})`
    if (lastLoggedProgress !== (payload.progress_pct ?? payload.progress) || lastLoggedStatus !== message) {
      addLog(message)
      lastLoggedProgress = payload.progress_pct ?? payload.progress
      lastLoggedStatus = message
    }
  }

  function startTaskProgress(progressRef, startMessage) {
    progressRef.value = 8
    if (startMessage) {
      addLog(startMessage)
    }
    const timerId = window.setInterval(() => {
      progressRef.value = Math.min(88, progressRef.value + Math.max(1, Math.round((90 - progressRef.value) * 0.18)))
    }, 320)
    return (doneMessage = '', failed = false) => {
      clearInterval(timerId)
      progressRef.value = failed ? Math.max(progressRef.value, 0) : 100
      if (doneMessage) {
        addLog(doneMessage)
      }
      window.setTimeout(() => {
        progressRef.value = 0
      }, failed ? 0 : 900)
    }
  }

  function buildCurrentSeasonDateRange() {
    const now = new Date()
    const year = now.getUTCMonth() < 2 ? now.getUTCFullYear() - 1 : now.getUTCFullYear()
    const start = `${year}-03-01`
    const seasonEnd = `${year}-10-31`
    const today = now.toISOString().slice(0, 10)
    return {
      dateFrom: start,
      dateTo: today > seasonEnd ? seasonEnd : today,
    }
  }

  function buildTemporalAnalyticsKey(fieldId, dateFrom, dateTo, cropCode) {
    return [
      fieldId || '',
      dateFrom || '',
      dateTo || '',
      cropCode || '',
    ].join('::')
  }

  function resolveTemporalTarget(target = 'metrics') {
    if (target === 'forecast') {
      return {
        dataRef: fieldForecastAnalytics,
        keyRef: fieldForecastAnalyticsKey,
        requestKey: 'fieldForecastAnalytics',
        loadingRef: isLoadingForecastAnalytics,
      }
    }
    return {
      dataRef: fieldTemporalAnalytics,
      keyRef: fieldTemporalAnalyticsKey,
      requestKey: 'fieldTemporalAnalytics',
      loadingRef: isLoadingMetricsAnalytics,
    }
  }

  function temporalPayloadHasWeatherSeries(payload) {
    const metrics = payload?.seasonal_series?.metrics || []
    if (!Array.isArray(metrics) || !metrics.length) {
      return false
    }
    const metricIds = new Set(metrics.map((item) => item?.metric).filter(Boolean))
    return ['precipitation', 'soil_moisture', 'vpd', 'wind'].some((metric) => metricIds.has(metric))
  }

  function applyAsyncTaskStatus(progressRef, stateRef, payload) {
    if (!payload || typeof payload !== 'object') {
      return
    }
    stateRef.value = payload
    const nextProgress = payload.progress_pct ?? payload.progress
    progressRef.value = Number.isFinite(Number(nextProgress)) ? Number(nextProgress) : progressRef.value
  }

  function logAsyncTaskProgress(payload, prefix) {
    if (!payload) return
    const stage = getTaskStageLabel(payload, payload?.stage_label || payload?.status || 'running')
    const detailText = getTaskStageDetail(payload)
    const detail = detailText ? ` · ${detailText}` : ''
    const eta = Number.isFinite(Number(payload?.estimated_remaining_s))
      ? ` · ETA ${Math.max(0, Math.round(Number(payload.estimated_remaining_s)))}s`
      : ''
    const signature = `${prefix}:${payload?.progress_pct ?? payload?.progress ?? 0}:${stage}:${detailText || ''}`
    if (stateSignatures.get(prefix) === signature) {
      return
    }
    stateSignatures.set(prefix, signature)
    addLog(`${prefix}: ${formatUiProgress(payload?.progress_pct ?? payload?.progress ?? 0)}% (${stage})${detail}${eta}`)
  }

  async function pollAsyncJob({
    kind,
    taskId,
    statusUrl,
    resultUrl,
    progressRef,
    stateRef,
    onResult,
    onDoneMessage,
    onFailureMessage,
  }) {
    const timerKey = kind === 'prediction'
      ? 'predictionJob'
      : kind === 'scenario'
        ? 'scenarioJob'
        : 'temporalAnalyticsJob'
    const progressPrefix = kind === 'prediction' ? 'Прогноз' : kind === 'scenario' ? 'Сценарий' : 'Сезонная аналитика'
    const setTimer = (callback, delayMs) => {
      if (kind === 'prediction') {
        if (predictionTimer) clearTimeout(predictionTimer)
        predictionTimer = window.setTimeout(callback, delayMs)
      } else if (kind === 'scenario') {
        if (scenarioTimer) clearTimeout(scenarioTimer)
        scenarioTimer = window.setTimeout(callback, delayMs)
      } else {
        if (temporalTimer) clearTimeout(temporalTimer)
        temporalTimer = window.setTimeout(callback, delayMs)
      }
    }

    const requestConfig = nextRequestConfig(timerKey)
    try {
      const response = await axios.get(statusUrl, requestConfig)
      applyAsyncTaskStatus(progressRef, stateRef, response.data)
      logAsyncTaskProgress(response.data, progressPrefix)

      if (response.data.status === 'done') {
        const resultResponse = await axios.get(resultUrl, nextRequestConfig(`${timerKey}Result`))
        applyAsyncTaskStatus(progressRef, stateRef, resultResponse.data)
        if (resultResponse.data.result) {
          await onResult(resultResponse.data.result)
        }
        addLog(onDoneMessage)
        progressRef.value = 100
        window.setTimeout(() => {
          progressRef.value = 0
        }, 900)
        return resultResponse.data.result || null
      }

      if (response.data.status === 'failed') {
        const message = response.data.error_msg || onFailureMessage
        addLog(message)
        progressRef.value = 0
        return null
      }

      setTimer(() => {
        pollAsyncJob({
          kind,
          taskId,
          statusUrl,
          resultUrl,
          progressRef,
          stateRef,
          onResult,
          onDoneMessage,
          onFailureMessage,
        })
      }, 1200)
      return null
    } catch (requestError) {
      if (isAbortError(requestError)) {
        return null
      }
      addLog(onFailureMessage ? `${onFailureMessage}: ${resolveError(requestError)}` : resolveError(requestError))
      progressRef.value = 0
      return null
    }
  }

  function formatRefreshTime(value) {
    return formatUiDateTime(value, {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      day: '2-digit',
      month: '2-digit',
    })
  }

  function clearTimers() {
    if (statusTimer) {
      clearTimeout(statusTimer)
      statusTimer = null
    }
    if (systemTimer) {
      clearTimeout(systemTimer)
      systemTimer = null
    }
    if (predictionTimer) {
      clearTimeout(predictionTimer)
      predictionTimer = null
    }
    if (scenarioTimer) {
      clearTimeout(scenarioTimer)
      scenarioTimer = null
    }
    if (temporalTimer) {
      clearTimeout(temporalTimer)
      temporalTimer = null
    }
    cancelAllRequests()
  }

  function clearFieldSelection() {
    selectedField.value = null
    selectedFieldIds.value = []
    mergeMode.value = false
    splitMode.value = false
    mergeSelectionIds.value = []
    fieldDashboard.value = null
    groupDashboard.value = null
    fieldTemporalAnalytics.value = null
    fieldForecastAnalytics.value = null
    fieldTemporalAnalyticsKey.value = ''
    fieldForecastAnalyticsKey.value = ''
    fieldManagementZones.value = null
    fieldEvents.value = []
    fieldEventsTotal.value = 0
    selectedEventSeasonYear.value = null
    selectedFieldPrediction.value = null
    fieldArchives.value = []
    fieldScenarios.value = []
    selectedArchiveView.value = null
    modelingResult.value = null
    sensitivityData.value = null
    temporalAnalyticsTaskState.value = null
    temporalAnalyticsTaskProgress.value = 0
    activeFieldTab.value = 'overview'
  }

  function promoteVisibleRun(runId, options = {}) {
    if (!runId) {
      return
    }
    visibleRunId.value = runId
    lastCompletedRunId.value = runId
    if (options.setCurrent !== false) {
      currentRunId.value = runId
    }
  }

  function applyRunStatusPayload(payload) {
    runStatus.value = payload.status
    runProgress.value = Number.isFinite(Number(payload.progress_pct ?? payload.progress))
      ? Number(payload.progress_pct ?? payload.progress)
      : 0
    runRuntime.value = payload.runtime || null
    runStageCode.value = payload.stage_code || payload.stage_label || payload.status || null
    runStageLabel.value = payload.stage_label || payload.status
    runStageDetailCode.value = payload.stage_detail_code || null
    runStageDetailParams.value = payload.stage_detail_params || {}
    runStageDetail.value = payload.stage_detail || null
    runStartedAt.value = payload.started_at || null
    runUpdatedAt.value = payload.updated_at || null
    runLastHeartbeatTs.value = payload.last_heartbeat_ts || null
    runStaleRunning.value = Boolean(payload.stale_running)
    runEstimatedRemainingS.value = payload.estimated_remaining_s ?? null
  }

  function clearDebugOverlay() {
    debugLayerPayload.value = null
    debugOverlayEnabled.value = false
  }

  async function loadRunDebugTiles(runId = selectedDebugRunId.value || visibleRunId.value) {
    if (!runId) {
      debugTilesCatalog.value = []
      selectedDebugRunId.value = null
      selectedDebugTileId.value = ''
      selectedDebugTileDetail.value = null
      selectedDebugLayerId.value = ''
      debugLayerPayload.value = null
      return []
    }
    isLoadingDebugTiles.value = true
    const requestConfig = nextRequestConfig('runDebugTiles')
    try {
      const response = await axios.get(`${API_BASE}/fields/runs/${runId}/debug/tiles`, requestConfig)
      const tiles = response.data?.tiles || []
      selectedDebugRunId.value = runId
      debugTilesCatalog.value = tiles
      if (!tiles.length) {
        selectedDebugTileId.value = ''
        selectedDebugTileDetail.value = null
        selectedDebugLayerId.value = ''
        debugLayerPayload.value = null
        return []
      }
      const selectedExists = tiles.some((tile) => tile.tile_id === selectedDebugTileId.value)
      const nextTileId = selectedExists ? selectedDebugTileId.value : String(tiles[0].tile_id || '')
      selectedDebugTileId.value = nextTileId
      await loadDebugTile(runId, nextTileId, { silent: true })
      return tiles
    } catch (requestError) {
      if (isAbortError(requestError)) {
        return debugTilesCatalog.value
      }
      debugTilesCatalog.value = []
      selectedDebugTileId.value = ''
      selectedDebugTileDetail.value = null
      selectedDebugLayerId.value = ''
      debugLayerPayload.value = null
      addLog(`Не удалось загрузить debug-тайлы: ${resolveError(requestError)}`)
      return []
    } finally {
      isLoadingDebugTiles.value = false
    }
  }

  async function loadDebugTile(runId = selectedDebugRunId.value, tileId = selectedDebugTileId.value, options = {}) {
    if (!runId || !tileId) {
      selectedDebugTileDetail.value = null
      return null
    }
    const requestConfig = nextRequestConfig('runDebugTile')
    try {
      const response = await axios.get(`${API_BASE}/fields/runs/${runId}/debug/tiles/${tileId}`, requestConfig)
      const payload = response.data || null
      selectedDebugTileDetail.value = payload
      selectedDebugTileId.value = String(tileId)
      const layers = payload?.available_layers || []
      if (layers.length) {
        const existingLayer = layers.some((item) => item.id === selectedDebugLayerId.value)
        if (!existingLayer) {
          selectedDebugLayerId.value = String(layers[0].id || '')
        }
      } else {
        selectedDebugLayerId.value = ''
        debugLayerPayload.value = null
      }
      if (!options.silent) {
        addLog(`Debug-диагностика загружена для тайла ${tileId}`)
      }
      return payload
    } catch (requestError) {
      if (isAbortError(requestError)) {
        return selectedDebugTileDetail.value
      }
      selectedDebugTileDetail.value = null
      selectedDebugLayerId.value = ''
      debugLayerPayload.value = null
      if (!options.silent) {
        addLog(`Не удалось загрузить debug-тайл: ${resolveError(requestError)}`)
      }
      return null
    }
  }

  async function loadDebugLayer(
    runId = selectedDebugRunId.value,
    tileId = selectedDebugTileId.value,
    layerName = selectedDebugLayerId.value,
    options = {},
  ) {
    if (!runId || !tileId || !layerName) {
      debugLayerPayload.value = null
      return null
    }
    isLoadingDebugLayer.value = true
    const requestConfig = nextRequestConfig('runDebugLayer')
    try {
      const response = await axios.get(
        `${API_BASE}/fields/runs/${runId}/debug/tiles/${tileId}/layers/${layerName}`,
        requestConfig,
      )
      debugLayerPayload.value = response.data || null
      selectedDebugLayerId.value = String(layerName)
      if (!options.silent) {
        addLog(`Debug-слой ${layerName} загружен для тайла ${tileId}`)
      }
      return debugLayerPayload.value
    } catch (requestError) {
      if (isAbortError(requestError)) {
        return debugLayerPayload.value
      }
      debugLayerPayload.value = null
      if (!options.silent) {
        addLog(`Не удалось загрузить debug-слой: ${resolveError(requestError)}`)
      }
      return null
    } finally {
      isLoadingDebugLayer.value = false
    }
  }

  function detectionPayload() {
    const presetProfile = activeDetectionPreset.value
    const effectiveResolutionM = expertMode.value ? resolutionM.value : presetProfile.resolutionM
    const effectiveTargetDates = expertMode.value ? targetDates.value : presetProfile.targetDates
    const effectiveMinFieldAreaHa = expertMode.value ? minFieldAreaHa.value : presetProfile.minFieldAreaHa
    return {
      aoi: {
        type: 'point_radius',
        lat: centerLat.value,
        lon: centerLon.value,
        radius_km: radiusKm.value,
      },
      time_range: {
        start_date: startDate.value,
        end_date: endDate.value,
      },
      resolution_m: effectiveResolutionM,
      max_cloud_pct: maxCloudPct.value,
      target_dates: effectiveTargetDates,
      min_field_area_ha: effectiveMinFieldAreaHa,
      seed_mode: 'edges',
      debug: false,
      config: {
        preset: detectionPreset.value,
      },
    }
  }

  function resolveDetectFailureMessage(payload) {
    if (payload?.error_msg) {
      return payload.error_msg
    }
      if (payload?.status === 'stale') {
        return 'Задача зависла: heartbeat от worker-а потерян. Карта оставлена на предыдущем успешном результате.'
      }
    if (payload?.status === 'cancelled') {
      return 'Автодетекция была отменена.'
    }
    if (payload?.status === 'failed') {
      return 'Автодетекция завершилась с ошибкой.'
    }
    return 'Автодетекция завершилась ошибкой.'
  }

  function toggleWindow(windowId) {
    if (!(windowId in uiWindows.value)) {
      return
    }
    uiWindows.value[windowId] = !uiWindows.value[windowId]
  }

  function showWindow(windowId) {
    if (!(windowId in uiWindows.value)) {
      return
    }
    uiWindows.value[windowId] = true
  }

  function hideWindow(windowId) {
    if (!(windowId in uiWindows.value)) {
      return
    }
    uiWindows.value[windowId] = false
  }

  function toggleSearchCenterPicking() {
    isPickingSearchCenter.value = !isPickingSearchCenter.value
    addLog(
      isPickingSearchCenter.value
        ? 'Выбор центра поиска активирован: щёлкните по карте'
        : 'Выбор центра поиска отменён'
    )
  }

  function applySearchCenter(lat, lon) {
    centerLat.value = Number(lat.toFixed(6))
    centerLon.value = Number(lon.toFixed(6))
    isPickingSearchCenter.value = false
    persistUiSettings()
    addLog(`Новый центр поиска: ${centerLat.value}, ${centerLon.value}`)
  }

  async function initialize() {
    restoreUiSettings()
    restoreLogs()
    await Promise.allSettled([
      loadRunSummaries(),
      loadLayers(),
      loadSystemStatus(),
      loadWeather(),
      loadCrops(),
      loadPersistedFields(),
      loadManualFields(),
    ])
    startSystemPolling()
    // Pause polling when tab is hidden, resume when visible
    if (typeof document !== 'undefined') {
      document.addEventListener('visibilitychange', () => {
        if (document.hidden) {
          if (systemTimer) { clearTimeout(systemTimer); systemTimer = null }
        } else {
          startSystemPolling()
        }
      })
    }
  }

  function startSystemPolling() {
    if (systemTimer) {
      clearTimeout(systemTimer)
    }
    if (!autoRefreshIntervalS.value) {
      return
    }
    const poll = async () => {
      if (typeof document !== 'undefined' && document.hidden) return
      const pollTasks = [loadSystemStatus(), loadWeather()]
      // Lite field refresh: reload current_metrics, archives, scenarios — skip temporal analytics
      if (selectedFieldIds.value.length === 1 && !isLoadingFieldDashboard.value && !isDetecting.value) {
        pollTasks.push(loadFieldDashboard(selectedFieldIds.value[0], { lite: true }))
      }
      const results = await Promise.allSettled(pollTasks)
      const anyFailed = results.some(r => r.status === 'rejected')
      if (anyFailed) {
        systemPollFailCount = Math.min(systemPollFailCount + 1, 6)
      } else {
        systemPollFailCount = 0
      }
      if (autoRefreshIntervalS.value > 0) {
        const backoffMs = autoRefreshIntervalS.value * 1000 * Math.pow(2, systemPollFailCount)
        systemTimer = window.setTimeout(poll, Math.min(backoffMs, 300_000))
      }
    }
    systemTimer = window.setTimeout(poll, autoRefreshIntervalS.value * 1000)
  }

  async function loadSystemStatus(options = {}) {
    const manual = Boolean(options?.manual)
    const requestConfig = nextRequestConfig('systemStatus')
    try {
      const previous = systemStatus.value
      const response = await axios.get(`${API_BASE}/status`, requestConfig)
      systemStatus.value = response.data
      lastStatusUpdatedAt.value = response.data?.timestamp || new Date().toISOString()
      if (lastSystemErrorMsg) {
        addLog('Соединение с сервером восстановлено')
        lastSystemErrorMsg = ''
      }
      if (manual) {
        const previousStatus = previous?.status
        const currentStatus = response.data?.status || 'unknown'
        const diffLabel = previousStatus && previousStatus !== currentStatus ? ` (${previousStatus} → ${currentStatus})` : ''
        lastStatusSyncState.value = 'ok'
        lastStatusSyncDetail.value = `${currentStatus}${diffLabel}`
        addLog(`Статус синхронизирован${diffLabel} · ${formatRefreshTime(lastStatusUpdatedAt.value)}`)
      }
    } catch (requestError) {
      if (isAbortError(requestError)) {
        return
      }
      const msg = resolveError(requestError)
      if (manual) {
        lastStatusSyncState.value = 'error'
        lastStatusSyncDetail.value = msg
      }
      if (msg !== lastSystemErrorMsg) {
        addLog(`Не удалось обновить статус системы: ${msg}`)
        lastSystemErrorMsg = msg
      }
    }
  }

  async function loadWeather(options = {}) {
    const manual = Boolean(options?.manual)
    isLoadingWeather.value = true
    const requestConfig = nextRequestConfig('weather')
    try {
      const [currentResponse, forecastResponse] = await Promise.all([
        axios.get(`${API_BASE}/weather/current`, { ...requestConfig, params: { lat: centerLat.value, lon: centerLon.value } }),
        axios.get(`${API_BASE}/weather/forecast`, { ...requestConfig, params: { lat: centerLat.value, lon: centerLon.value, days: 5 } }),
      ])
      weatherCurrent.value = currentResponse.data
      weatherForecast.value = forecastResponse.data.forecast || []
      lastWeatherUpdatedAt.value =
        currentResponse.data?.freshness?.cache_written_at ||
        currentResponse.data?.freshness?.fetched_at ||
        new Date().toISOString()
      if (manual) {
        const provider = currentResponse.data?.provider || 'weather'
        const cloud = currentResponse.data?.cloud_cover_pct
        const cloudLabel = cloud === null || cloud === undefined ? 'облачность —' : `облачность ${Number(cloud).toFixed(0)}%`
        lastWeatherSyncState.value = 'ok'
        lastWeatherSyncDetail.value = `${provider} · ${cloudLabel}`
        addLog(`Погода обновлена · ${provider} · ${cloudLabel} · ${formatRefreshTime(lastWeatherUpdatedAt.value)}`)
      }
    } catch (requestError) {
      if (isAbortError(requestError)) {
        return
      }
      const msg = resolveError(requestError)
      if (manual) {
        lastWeatherSyncState.value = 'error'
        lastWeatherSyncDetail.value = msg
      }
      if (manual || !lastSystemErrorMsg) {
        addLog(`Не удалось обновить погоду: ${msg}`)
      }
    } finally {
      isLoadingWeather.value = false
    }
  }

  async function refreshAll() {
    const tasks = [
      loadRunSummaries(),
      loadSystemStatus({ manual: true }),
      loadWeather({ manual: true }),
      loadLayers(),
      loadPersistedFields(),
      loadManualFields(),
    ]
    if (selectedFieldIds.value.length) {
      tasks.push(loadSelectionAnalytics())
    }
    await Promise.allSettled(tasks)
    addLog('Справочные данные обновлены')
  }

  async function loadCrops() {
    const requestConfig = nextRequestConfig('crops')
    try {
      const response = await axios.get(`${API_BASE}/crops`, requestConfig)
      crops.value = response.data.crops || []
      if (!selectedCropCode.value && crops.value.length) {
        selectedCropCode.value = crops.value[0].code
      }
    } catch (requestError) {
      if (isAbortError(requestError)) {
        return
      }
      addLog(`Не удалось загрузить культуры: ${resolveError(requestError)}`)
    }
  }

  async function loadRunSummaries() {
    const requestConfig = nextRequestConfig('runSummaries')
    try {
      const response = await axios.get(`${API_BASE}/fields/runs`, {
        ...requestConfig,
        params: { limit: 20 },
      })
      runSummaries.value = response.data.runs || []
      const latestCompleted = runSummaries.value.find((run) => run.status === 'done')
      if (latestCompleted?.id) {
        lastCompletedRunId.value = latestCompleted.id
        if (!visibleRunId.value && !isDetecting.value) {
          visibleRunId.value = latestCompleted.id
        }
      }
      return runSummaries.value
    } catch (requestError) {
      if (isAbortError(requestError)) {
        return runSummaries.value
      }
      addLog(`Не удалось загрузить список запусков: ${resolveError(requestError)}`)
      return runSummaries.value
    }
  }

  async function loadFieldDashboard(fieldId = selectedField.value?.field_id, options = {}) {
    if (!fieldId) {
      fieldDashboard.value = null
      fieldTemporalAnalytics.value = null
      fieldForecastAnalytics.value = null
      fieldTemporalAnalyticsKey.value = ''
      fieldForecastAnalyticsKey.value = ''
      fieldManagementZones.value = null
      selectedFieldPrediction.value = null
      fieldArchives.value = []
      fieldScenarios.value = []
      return null
    }
    isLoadingFieldDashboard.value = true
    const requestConfig = nextRequestConfig('fieldDashboard')
    try {
      const response = await axios.get(`${API_BASE}/fields/${fieldId}/dashboard`, requestConfig)
      const mergedPrediction = mergePredictionPayload(selectedFieldPrediction.value, response.data?.prediction || null)
      fieldDashboard.value = {
        ...(response.data || {}),
        prediction: mergedPrediction,
      }
      groupDashboard.value = null
      selectedFieldPrediction.value = mergedPrediction
      fieldArchives.value = response.data?.archives || []
      fieldScenarios.value = response.data?.scenarios || []
      // Only clear temporal analytics when loading a different field
      const isSameField = fieldTemporalAnalyticsKey.value.startsWith(fieldId + '::')
      if (!isSameField) {
        fieldTemporalAnalytics.value = null
        fieldForecastAnalytics.value = null
        fieldTemporalAnalyticsKey.value = ''
        fieldForecastAnalyticsKey.value = ''
      }
      if (response.data?.field) {
        selectedField.value = {
          ...selectedField.value,
          ...response.data.field,
        }
        if (response.data.field.aoi_run_id && !isDetecting.value) {
          promoteVisibleRun(response.data.field.aoi_run_id)
        }
      }
      const previewOnlyField = isPreviewFieldSource(response.data?.field?.source)
      if (previewOnlyField) {
        selectedFieldPrediction.value = null
        fieldArchives.value = []
        fieldScenarios.value = []
        fieldForecastAnalytics.value = null
        fieldForecastAnalyticsKey.value = ''
        fieldManagementZones.value = null
        if (['forecast', 'scenarios', 'archive'].includes(activeFieldTab.value)) {
          activeFieldTab.value = 'overview'
        }
      }
      if (!options.lite) {
        const currentSeasonRange = buildCurrentSeasonDateRange()
        const preferExisting = isSameField
        const dashboardLoads = [
          loadFieldTemporalAnalytics(fieldId, { silent: true, preferExisting, target: 'metrics' }),
        ]
        if (!previewOnlyField) {
          dashboardLoads.push(
            loadFieldTemporalAnalytics(fieldId, {
              silent: true,
              preferExisting,
              target: 'forecast',
              dateFrom: currentSeasonRange.dateFrom,
              dateTo: currentSeasonRange.dateTo,
              autoBackfill: true,
            }),
            loadFieldManagementZones(fieldId, { silent: true }),
          )
        }
        // Release the dossier UI as soon as the primary dashboard payload is ready.
        // Secondary analytics continue in the background and update their own sections.
        isLoadingFieldDashboard.value = false
        void Promise.allSettled(dashboardLoads)
      }
      return response.data
    } catch (requestError) {
      if (isAbortError(requestError)) {
        return null
      }
      addLog(`Не удалось загрузить досье поля: ${resolveError(requestError)}`)
      return null
    } finally {
      isLoadingFieldDashboard.value = false
    }
  }

  async function loadGroupDashboard(fieldIds = selectedFieldIds.value) {
    if (!fieldIds.length) {
      groupDashboard.value = null
      return null
    }
    isLoadingGroupDashboard.value = true
    const requestConfig = nextRequestConfig('groupDashboard')
    try {
      const response = await axios.post(`${API_BASE}/fields/dashboard/group`, {
        field_ids: fieldIds,
      }, requestConfig)
      groupDashboard.value = response.data
      fieldDashboard.value = null
      fieldTemporalAnalytics.value = null
      fieldForecastAnalytics.value = null
      fieldTemporalAnalyticsKey.value = ''
      fieldForecastAnalyticsKey.value = ''
      fieldManagementZones.value = null
      selectedFieldPrediction.value = null
      fieldArchives.value = []
      fieldScenarios.value = []
      return response.data
    } catch (requestError) {
      if (isAbortError(requestError)) {
        return null
      }
      addLog(`Не удалось загрузить групповую аналитику: ${resolveError(requestError)}`)
      return null
    } finally {
      isLoadingGroupDashboard.value = false
    }
  }

  async function loadSelectionAnalytics() {
    selectedArchiveView.value = null
    if (!selectedFieldIds.value.length) {
      fieldDashboard.value = null
      groupDashboard.value = null
      fieldTemporalAnalytics.value = null
      fieldForecastAnalytics.value = null
      fieldTemporalAnalyticsKey.value = ''
      fieldForecastAnalyticsKey.value = ''
      fieldManagementZones.value = null
      return null
    }
    if (selectedFieldIds.value.length > 1) {
      return loadGroupDashboard([...selectedFieldIds.value])
    }
    return loadFieldDashboard(selectedFieldIds.value[0])
  }

  async function loadArchiveView(archiveId) {
    if (!archiveId) {
      selectedArchiveView.value = null
      return null
    }
    if (selectedFieldIsPreviewOnly.value) {
      addLog(t('field.previewOnlyActionHint'))
      return null
    }
    isLoadingArchiveView.value = true
    const requestConfig = nextRequestConfig('archiveView')
    try {
      const response = await axios.get(`${API_BASE}/archive/${archiveId}/view`, requestConfig)
      selectedArchiveView.value = response.data
      activeFieldTab.value = 'archive'
      return response.data
    } catch (requestError) {
      if (isAbortError(requestError)) {
        return null
      }
      addLog(`Не удалось открыть архив: ${resolveError(requestError)}`)
      return null
    } finally {
      isLoadingArchiveView.value = false
    }
  }

  async function loadFieldTemporalAnalytics(fieldId = selectedField.value?.field_id, options = {}) {
    const target = resolveTemporalTarget(options.target)
    if (!fieldId || hasGroupSelection.value) {
      target.dataRef.value = null
      target.keyRef.value = ''
      return null
    }
    const dateFrom = options.dateFrom ?? seriesDateFrom.value ?? ''
    const dateTo = options.dateTo ?? seriesDateTo.value ?? ''
    const cropCode = options.cropCode ?? selectedCropCode.value ?? ''
    const cacheKey = buildTemporalAnalyticsKey(fieldId, dateFrom, dateTo, cropCode)
    const canReuseExisting =
      options.preferExisting &&
      target.keyRef.value === cacheKey &&
      target.dataRef.value?.seasonal_series &&
      temporalPayloadHasWeatherSeries(target.dataRef.value)
    if (canReuseExisting) {
      return target.dataRef.value
    }
    const isSameKey = target.keyRef.value === cacheKey
    const shouldResetPayload = !options.preferExisting || !isSameKey
    // Only clear displayed data when switching to a different field/range — not on same-field refresh
    if (shouldResetPayload && !isSameKey) {
      target.dataRef.value = null
    }
    target.keyRef.value = cacheKey
    target.loadingRef.value = true
    const requestConfig = nextRequestConfig(target.requestKey)
    try {
      const params = {}
      if (dateFrom) params.date_from = dateFrom
      if (dateTo) params.date_to = dateTo
      if (cropCode) params.crop_code = cropCode
      const response = await axios.get(`${API_BASE}/fields/${fieldId}/temporal-analytics`, { ...requestConfig, params })
      const payload = response.data || null
      if (payload) {
        target.dataRef.value = payload
        target.keyRef.value = cacheKey
      }
      const shouldAutoBackfill = options.autoBackfill !== false && payload?.data_status?.backfill_required
      if (shouldAutoBackfill && dateFrom && dateTo) {
        return await runTemporalAnalyticsBackfill(fieldId, {
          target: options.target || 'metrics',
          dateFrom,
          dateTo,
          cropCode,
          silent: options.silent,
        })
      }
      if (!options.silent) {
        addLog(`Сезонная аналитика загружена для поля ${fieldId}`)
      }
      return target.dataRef.value
    } catch (requestError) {
      if (isAbortError(requestError)) {
        return shouldResetPayload ? null : target.dataRef.value
      }
      if (shouldResetPayload) {
        target.dataRef.value = null
        target.keyRef.value = ''
      }
      if (!options.silent) {
        addLog(`Не удалось загрузить сезонную аналитику: ${resolveError(requestError)}`)
      }
      return shouldResetPayload ? null : target.dataRef.value
    } finally {
      if (!target.dataRef.value) {
        target.keyRef.value = ''
      }
      target.loadingRef.value = false
    }
  }

  async function runTemporalAnalyticsBackfill(fieldId, options = {}) {
    if (!fieldId || hasGroupSelection.value) {
      return null
    }
    const dateFrom = options.dateFrom ?? seriesDateFrom.value ?? ''
    const dateTo = options.dateTo ?? seriesDateTo.value ?? ''
    if (!dateFrom || !dateTo) {
      return null
    }
    temporalAnalyticsTaskState.value = null
    temporalAnalyticsTaskProgress.value = 5
    if (!options.silent) {
      addLog(`Запускаю materialization сезонной аналитики для поля ${fieldId}...`)
    }
    const requestConfig = nextRequestConfig('temporalAnalyticsJob')
    try {
      const response = await axios.post(
        `${API_BASE}/fields/${fieldId}/temporal-analytics/jobs`,
        null,
        {
          ...requestConfig,
          params: {
            date_from: dateFrom,
            date_to: dateTo,
            crop_code: options.cropCode ?? selectedCropCode.value ?? null,
          },
        }
      )
      applyAsyncTaskStatus(temporalAnalyticsTaskProgress, temporalAnalyticsTaskState, response.data)
      if (!options.silent) {
        addLog(`Задача сезонной аналитики поставлена в очередь: ${response.data.task_id}`)
      }
      return await pollAsyncJob({
        kind: 'temporal',
        taskId: response.data.task_id,
        statusUrl: `${API_BASE}/fields/temporal-analytics/jobs/${response.data.task_id}`,
        resultUrl: `${API_BASE}/fields/temporal-analytics/jobs/${response.data.task_id}/result`,
        progressRef: temporalAnalyticsTaskProgress,
        stateRef: temporalAnalyticsTaskState,
        onResult: async () => {
          await loadFieldTemporalAnalytics(fieldId, {
            target: options.target || 'metrics',
            dateFrom,
            dateTo,
            cropCode: options.cropCode,
            preferExisting: false,
            autoBackfill: false,
            silent: true,
          })
        },
        onDoneMessage: `Сезонная аналитика обновлена для поля ${fieldId}`,
        onFailureMessage: 'Не удалось обновить сезонную аналитику',
      })
    } catch (requestError) {
      if (isAbortError(requestError)) {
        return null
      }
      temporalAnalyticsTaskProgress.value = 0
      if (!options.silent) {
        addLog(`Не удалось запустить materialization сезонной аналитики: ${resolveError(requestError)}`)
      }
      return null
    }
  }

  async function loadFieldManagementZones(fieldId = selectedField.value?.field_id, options = {}) {
    if (!fieldId || hasGroupSelection.value) {
      fieldManagementZones.value = null
      return null
    }
    isLoadingManagementZones.value = true
    const requestConfig = nextRequestConfig('fieldManagementZones')
    try {
      const response = await axios.get(`${API_BASE}/fields/${fieldId}/management-zones`, requestConfig)
      fieldManagementZones.value = response.data
      if (!options.silent) {
        addLog(`Зоны управления ${response.data?.summary?.supported ? 'обновлены' : 'недоступны'} для поля ${fieldId}`)
      }
      return fieldManagementZones.value
    } catch (requestError) {
      if (isAbortError(requestError)) {
        return fieldManagementZones.value
      }
      if (!options.silent) {
        addLog(`Не удалось загрузить зоны управления: ${resolveError(requestError)}`)
      }
      return fieldManagementZones.value
    } finally {
      isLoadingManagementZones.value = false
    }
  }

  async function loadFieldEvents(fieldId = selectedField.value?.field_id, options = {}) {
    if (!fieldId || hasGroupSelection.value) {
      fieldEvents.value = []
      fieldEventsTotal.value = 0
      return
    }
    isLoadingEvents.value = true
    const requestConfig = nextRequestConfig('fieldEvents')
    try {
      const params = {}
      if (selectedEventSeasonYear.value) params.season_year = selectedEventSeasonYear.value
      const response = await axios.get(`${API_BASE}/fields/${fieldId}/events`, { ...requestConfig, params })
      fieldEvents.value = response.data.events || []
      fieldEventsTotal.value = response.data.total || 0
      if (!options.silent) addLog(`Загружено событий: ${fieldEventsTotal.value}`)
      return fieldEvents.value
    } catch (requestError) {
      if (isAbortError(requestError)) return fieldEvents.value
      if (!options.silent) addLog(`Не удалось загрузить события: ${resolveError(requestError)}`)
      return fieldEvents.value
    } finally {
      isLoadingEvents.value = false
    }
  }

  async function createFieldEvent(fieldId, eventData) {
    isSubmittingEvent.value = true
    try {
      const response = await axios.post(`${API_BASE}/fields/${fieldId}/events`, eventData)
      addLog(`Событие «${eventData.event_type}» добавлено`)
      await loadFieldEvents(fieldId, { silent: true })
      return response.data
    } catch (requestError) {
      addLog(`Ошибка создания события: ${resolveError(requestError)}`)
      throw requestError
    } finally {
      isSubmittingEvent.value = false
    }
  }

  async function updateFieldEvent(fieldId, eventId, eventData) {
    isSubmittingEvent.value = true
    try {
      const response = await axios.patch(`${API_BASE}/fields/${fieldId}/events/${eventId}`, eventData)
      addLog(`Событие обновлено`)
      await loadFieldEvents(fieldId, { silent: true })
      return response.data
    } catch (requestError) {
      addLog(`Ошибка обновления события: ${resolveError(requestError)}`)
      throw requestError
    } finally {
      isSubmittingEvent.value = false
    }
  }

  async function deleteFieldEvent(fieldId, eventId) {
    try {
      await axios.delete(`${API_BASE}/fields/${fieldId}/events/${eventId}`)
      addLog(`Событие удалено`)
      await loadFieldEvents(fieldId, { silent: true })
    } catch (requestError) {
      addLog(`Ошибка удаления события: ${resolveError(requestError)}`)
      throw requestError
    }
  }

  async function startDetection() {
    persistUiSettings()
    if (isDetecting.value) {
      return
    }
    isDetecting.value = true
    error.value = null
    runProgress.value = 0
    lastLoggedProgress = null
    lastLoggedStatus = null
    runRuntime.value = null
    lastPreflight.value = null
    runStageCode.value = null
    runStageLabel.value = null
    runStageDetailCode.value = null
    runStageDetailParams.value = {}
    runStageDetail.value = null
    runStartedAt.value = null
    runUpdatedAt.value = null
    runLastHeartbeatTs.value = null
    runStaleRunning.value = false
    runEstimatedRemainingS.value = null
    queuedDispatchWarningRunId = null
    clearFieldSelection()
    showWindow('logs')

    const sanitizedTargetDates = clampNumber(targetDates.value, TARGET_DATES_MIN, TARGET_DATES_MIN, TARGET_DATES_MAX)
    if (sanitizedTargetDates !== targetDates.value) {
      targetDates.value = sanitizedTargetDates
      addLog(`Параметр "Даты" скорректирован до допустимого диапазона: ${TARGET_DATES_MIN}-${TARGET_DATES_MAX}`)
    }

    addLog('Запускаю автодетекцию полей...')

    try {
      const payload = detectionPayload()
      if (expertMode.value) {
        payload.target_dates = sanitizedTargetDates
      }
      const preflight = await axios.post(`${API_BASE}/fields/detect/preflight?use_sam=${useSam.value}`, payload)
      const preflightData = preflight.data || {}
      lastPreflight.value = preflightData
      if (preflightData.hard_block) {
        throw {
          response: {
            data: {
              detail: preflightData.reason || 'Параметры запуска превышают бюджет выбранного профиля.',
            },
          },
        }
      }
      for (const warning of preflightData.warnings || []) {
        addLog(`Preflight: ${warning}`)
      }
      addLog(
        `Preflight: preset ${getDetectionPresetMeta(preflightData.preset || detectionPreset.value).label.toLowerCase()} · ` +
        `runtime class ${preflightData.estimated_runtime_class || 'unknown'} · ${preflightData.estimated_tiles || '—'} тайлов · ` +
        `~${preflightData.estimated_ram_mb || '—'} MB · TTA ${preflightData.tta_mode || 'none'} · ` +
        `S1 ${preflightData.s1_planned ? 'on' : 'off'} · профиль ${preflightData.regional_profile || 'default'}`
      )
      if (preflightData.launch_tier) {
        addLog(`Preflight trust: ${preflightData.launch_tier}${preflightData.review_reason ? ` · ${preflightData.review_reason}` : ''}`)
      }
      if (!preflightData.budget_ok && preflightData.reason) {
        addLog(`Preflight: ${preflightData.reason}`)
      }
      if (visibleRunId.value) {
        addLog('Новый запуск начат. Карта продолжает показывать предыдущий завершённый результат до полной готовности нового.')
      }

      const response = await axios.post(`${API_BASE}/fields/detect?use_sam=${useSam.value}`, payload)

      activeRunId.value = response.data.aoi_run_id
      runStatus.value = 'queued'
      runStageLabel.value = 'queued'
      addLog({
        message: `Задача передана worker-у: ${activeRunId.value}`,
        severity: 'info',
        category: 'detect',
        code: 'detect_submitted',
        runId: activeRunId.value,
      })
      await loadRunSummaries()
      pollStatus()
    } catch (requestError) {
      error.value = resolveError(requestError)
      isDetecting.value = false
      addLog(`Ошибка запуска: ${error.value}`)
    }
  }

  async function pollStatus() {
    if (!activeRunId.value) {
      return
    }
    const requestConfig = nextRequestConfig('detectStatus')

    try {
      const response = await axios.get(`${API_BASE}/fields/status/${activeRunId.value}`, requestConfig)
      applyRunStatusPayload(response.data)

      if (response.data.status === 'done') {
        addLog('Автодетекция завершена')
        await Promise.allSettled([fetchResult(activeRunId.value), loadSystemStatus(), loadWeather(), loadRunSummaries()])
        isDetecting.value = false
        return
      }

      if (['failed', 'cancelled', 'stale'].includes(response.data.status)) {
        error.value = resolveDetectFailureMessage(response.data)
        addLog(`Ошибка детекции: ${error.value}`)
        isDetecting.value = false
        await loadRunSummaries()
        return
      }

      logProgressUpdate(response.data)
      if (response.data.status === 'queued') {
        const elapsedS = Number(response.data.elapsed_s || 0)
        if (elapsedS >= 8 && queuedDispatchWarningRunId !== activeRunId.value) {
          queuedDispatchWarningRunId = activeRunId.value
          addLog({
            message: 'Worker ещё не забрал задачу детекта. Проверь backend-api/celery-worker.',
            severity: 'warning',
            category: 'detect',
            code: 'detect_queue_delay',
            runId: activeRunId.value,
          })
        }
      }
      statusPollFailCount = 0
      statusTimer = window.setTimeout(pollStatus, response.data.status === 'queued' ? 750 : 2000)
    } catch (requestError) {
      if (isAbortError(requestError)) {
        return
      }
      statusPollFailCount = Math.min(statusPollFailCount + 1, 5)
      if (statusPollFailCount <= 1) {
        addLog(`Ошибка опроса статуса: ${resolveError(requestError)}`)
      }
      const backoffMs = 5000 * Math.pow(2, statusPollFailCount - 1)
      statusTimer = window.setTimeout(pollStatus, Math.min(backoffMs, 60_000))
    }
  }

  async function fetchResult(runId = activeRunId.value) {
    if (!runId) {
      return
    }
    const requestConfig = nextRequestConfig('detectResult')
    try {
      const response = await axios.get(`${API_BASE}/fields/result/${runId}`, requestConfig)
      applyRunStatusPayload({
        ...response.data,
        progress: response.data.progress ?? runProgress.value,
        progress_pct: response.data.progress_pct ?? response.data.progress ?? runProgress.value,
      })
      if (response.data.geojson) {
        fieldsGeoJson.value = response.data.geojson
        promoteVisibleRun(runId)
        const loadedCount = response.data.geojson.features?.length || 0
        if (response.data.preview_only || response.data.output_mode === 'preview_agri_contours') {
          addLog(`${t('field.previewOnlyTitle')}: ${loadedCount}`)
        } else {
          addLog(`Загружено полей: ${loadedCount}`)
        }
      }
      await Promise.allSettled([loadPersistedFields({ runId, promoteVisible: true }), loadManualFields(), loadRunSummaries()])
    } catch (requestError) {
      if (isAbortError(requestError)) {
        return
      }
      addLog(`Не удалось загрузить результат: ${resolveError(requestError)}`)
    }
  }

  async function loadPersistedFields(options = {}) {
    const requestConfig = nextRequestConfig('persistedFields')
    const runId = options.runId || visibleRunId.value || lastCompletedRunId.value || null
    try {
      const response = await axios.get(`${API_BASE}/fields/geojson`, {
        ...requestConfig,
        params: runId ? { aoi_run_id: runId } : {},
      })
      fieldsGeoJson.value = response.data || { type: 'FeatureCollection', features: [] }
      const loadedRunId = fieldsGeoJson.value?.features?.[0]?.properties?.aoi_run_id || runId
      if (options.promoteVisible && loadedRunId) {
        promoteVisibleRun(loadedRunId)
      } else if (!visibleRunId.value && !isDetecting.value && loadedRunId) {
        promoteVisibleRun(loadedRunId)
      } else if (!lastCompletedRunId.value && loadedRunId) {
        lastCompletedRunId.value = loadedRunId
      }
      syncSelectedFieldFromCollections()
      return fieldsGeoJson.value
    } catch (requestError) {
      if (isAbortError(requestError)) {
        return fieldsGeoJson.value
      }
      if (requestError?.response?.status !== 404) {
        addLog(`Не удалось загрузить сохранённые поля: ${resolveError(requestError)}`)
      }
      return fieldsGeoJson.value
    }
  }

  async function loadLayers() {
    const requestConfig = nextRequestConfig('layers')
    try {
      const response = await axios.get(`${API_BASE}/layers`, requestConfig)
      availableLayers.value = response.data.layers || []
    } catch (requestError) {
      if (isAbortError(requestError)) {
        return
      }
      addLog(`Не удалось загрузить слои: ${resolveError(requestError)}`)
    }
  }

  async function loadSatelliteScene({ bbox, width, height, manual = false } = {}) {
    cancelRequest('satellite')
    if (!showSatelliteBrowse.value || !bbox) {
      satelliteScene.value = null
      satelliteLoadStatus.value = 'idle'
      return null
    }
    satelliteLoadStatus.value = 'loading'
    const requestConfig = nextRequestConfig('satellite')
    try {
      const params = {
        minx: bbox[0],
        miny: bbox[1],
        maxx: bbox[2],
        maxy: bbox[3],
        width,
        height,
        max_cloud_pct: maxCloudPct.value,
        start_date: startDate.value,
        end_date: endDate.value,
      }
      if (satelliteBrowseDate.value) {
        params.scene_date = satelliteBrowseDate.value
      }
      const response = await axios.get(`${API_BASE}/satellite/true-color`, {
        ...requestConfig,
        params,
      })
      satelliteScene.value = response.data
      satelliteLoadStatus.value = 'ready'
      if (manual) {
        const cloud = response.data?.cloud_cover_pct
        const cloudLabel = cloud === null || cloud === undefined ? 'облачность —' : `облачность ${Number(cloud).toFixed(0)}%`
        addLog(`Спутниковая сцена обновлена · ${cloudLabel} · аккаунт ${response.data?.provider_account || 'primary'}`)
      }
      return response.data
    } catch (requestError) {
      if (isAbortError(requestError)) {
        return satelliteScene.value
      }
      satelliteLoadStatus.value = 'error'
      satelliteScene.value = null
      addLog(`Не удалось загрузить спутниковую сцену: ${resolveError(requestError)}`)
      return null
    }
  }

  function toggleLayer(layerId) {
    const isSpectral = SPECTRAL_LAYERS.has(layerId)
    const currentlyActive = activeLayers.value[layerId]

    if (currentlyActive) {
      // Turning off
      activeLayers.value[layerId] = false
    } else {
      // Turning on — disable conflicting layers in same group
      if (isSpectral) {
        // Spectral layers are mutually exclusive
        for (const key of SPECTRAL_LAYERS) {
          activeLayers.value[key] = false
        }
      }
      activeLayers.value[layerId] = true
    }
  }

  async function selectField(fieldProperties, options = {}) {
    const additive = Boolean(options.additive)
    if (!fieldProperties?.field_id) {
      clearFieldSelection()
      return
    }

    if (additive) {
      const next = new Set(selectedFieldIds.value)
      if (next.has(fieldProperties.field_id)) {
        next.delete(fieldProperties.field_id)
      } else {
        next.add(fieldProperties.field_id)
      }
      selectedFieldIds.value = [...next]
      if (!selectedFieldIds.value.length) {
        clearFieldSelection()
        return
      }
      syncSelectedFieldFromCollections(fieldProperties.field_id)
    } else {
      selectedFieldIds.value = [fieldProperties.field_id]
      selectedField.value = fieldProperties
    }

    if (!isDetecting.value && fieldProperties?.aoi_run_id) {
      promoteVisibleRun(fieldProperties.aoi_run_id)
    }
    selectedArchiveView.value = null
    modelingResult.value = null
    activeFieldTab.value = 'overview'
    // Initialize date range to current season (Jan 1 → today) if not yet set
    if (!additive) {
      const today = new Date()
      const todayStr = today.toISOString().slice(0, 10)
      const yearStart = `${today.getFullYear()}-01-01`
      if (!seriesDateFrom.value) seriesDateFrom.value = yearStart
      if (!seriesDateTo.value) seriesDateTo.value = todayStr
    }
    showWindow('fieldActions')
    await loadSelectionAnalytics()
  }

  async function refreshPrediction(forceRefresh = true) {
    if (!selectedField.value?.field_id || hasGroupSelection.value) {
      return null
    }
    if (selectedFieldIsPreviewOnly.value) {
      addLog(t('field.previewOnlyActionHint'))
      return null
    }
    if (isRefreshingPrediction.value) {
      return null
    }
    isRefreshingPrediction.value = true
    predictionTaskState.value = null
    predictionTaskProgress.value = 5
    addLog(`Запускаю ${forceRefresh ? 'пересчёт' : 'загрузку'} прогноза для поля ${selectedField.value.field_id}...`)
    const requestConfig = nextRequestConfig('prediction')
    try {
      const response = await axios.post(
        `${API_BASE}/predictions/field/${selectedField.value.field_id}/jobs`,
        null,
        {
          ...requestConfig,
          params: {
            crop_code: selectedCropCode.value || null,
            refresh: forceRefresh,
          },
        }
      )
      applyAsyncTaskStatus(predictionTaskProgress, predictionTaskState, response.data)
      addLog(`Задача прогноза поставлена в очередь: ${response.data.task_id}`)
      return await pollAsyncJob({
        kind: 'prediction',
        taskId: response.data.task_id,
        statusUrl: `${API_BASE}/predictions/jobs/${response.data.task_id}`,
        resultUrl: `${API_BASE}/predictions/jobs/${response.data.task_id}/result`,
        progressRef: predictionTaskProgress,
        stateRef: predictionTaskState,
        onResult: async (payload) => {
          selectedFieldPrediction.value = payload
          const fieldId = payload?.field_id || selectedField.value?.field_id
          if (fieldId) {
            await loadFieldDashboard(fieldId)
          }
        },
        onDoneMessage: `Прогноз обновлён для поля ${selectedField.value.field_id}`,
        onFailureMessage: 'Не удалось обновить прогноз',
      })
    } catch (requestError) {
      if (isAbortError(requestError)) {
        return null
      }
      addLog(`Не удалось обновить прогноз: ${resolveError(requestError)}`)
      predictionTaskProgress.value = 0
      return null
    } finally {
      isRefreshingPrediction.value = false
    }
  }

  function autoFillModelingFactors() {
    // Derive factor defaults from real field data.
    // expected_rain_mm prefers a satellite wetness proxy and only falls back
    // to weather-based history when satellite series are unavailable.
    // soil_compaction uses satellite moisture and bare-soil signal proxies.
    const pred = selectedFieldPrediction.value || fieldDashboard.value?.prediction
    const waterBal = pred?.water_balance || {}
    const inputFeats = pred?.input_features || {}
    const predSeries = pred?.seasonal_series?.metrics || []
    // Also check temporal analytics series (contains soil_moisture, precipitation from FieldFeatureWeekly)
    const temporalSeries = fieldForecastAnalytics.value?.seasonal_series?.metrics
      || fieldTemporalAnalytics.value?.seasonal_series?.metrics
      || []
    const series = temporalSeries.length ? temporalSeries : predSeries
    const forecastCurve = pred?.forecast_curve?.points || []

    // Always reset to safe defaults before filling — prevents stale manual values leaking into satellite mode
    modelingForm.value.expected_rain_mm = 20
    modelingForm.value.soil_compaction = null
    modelingAutoSources.value = {
      expected_rain_mm: '',
      soil_compaction: '',
    }

    const _normalizeFraction = (value) => {
      const numeric = Number(value)
      if (!Number.isFinite(numeric)) {
        return null
      }
      if (numeric > 1) {
        return Math.max(0, Math.min(1, numeric / 100))
      }
      return Math.max(0, Math.min(1, numeric))
    }
    const _metricPoints = (metricId) => series.find((m) => m.metric === metricId)?.points || []
    const _latestMetricValue = (metricId) => {
      const points = _metricPoints(metricId)
      for (let idx = points.length - 1; idx >= 0; idx -= 1) {
        const point = points[idx]
        const raw = point?.value ?? point?.mean ?? point?.median ?? null
        const numeric = Number(raw)
        if (Number.isFinite(numeric)) {
          return numeric
        }
      }
      return null
    }
    const _meanMetricValue = (metricId) => {
      const values = _metricPoints(metricId)
        .map((point) => Number(point?.value ?? point?.mean ?? point?.median ?? NaN))
        .filter((value) => Number.isFinite(value))
      if (!values.length) {
        return null
      }
      return values.reduce((acc, value) => acc + value, 0) / values.length
    }

    // expected_rain_mm: first from satellite wetness proxy, then from weather-derived history.
    const soilMoistureValue = _normalizeFraction(_meanMetricValue('soil_moisture') ?? inputFeats.current_soil_moisture)
    const ndwiValue = _latestMetricValue('ndwi') ?? inputFeats.current_ndwi_mean ?? inputFeats.current_ndwi ?? null
    const ndmiValue = _latestMetricValue('ndmi') ?? inputFeats.current_ndmi_mean ?? inputFeats.current_ndmi ?? null
    const bsiValue = _latestMetricValue('bsi') ?? inputFeats.current_bsi_mean ?? inputFeats.current_bsi ?? null

    const weightedWetness = []
    if (soilMoistureValue !== null) {
      weightedWetness.push({
        weight: 0.5,
        value: Math.max(0, Math.min(1, soilMoistureValue / 0.45)),
      })
    }
    if (Number.isFinite(Number(ndwiValue))) {
      weightedWetness.push({
        weight: 0.25,
        value: Math.max(0, Math.min(1, (Number(ndwiValue) + 0.35) / 0.7)),
      })
    }
    if (Number.isFinite(Number(ndmiValue))) {
      weightedWetness.push({
        weight: 0.25,
        value: Math.max(0, Math.min(1, (Number(ndmiValue) + 0.35) / 0.7)),
      })
    }
    if (weightedWetness.length) {
      const weightSum = weightedWetness.reduce((acc, item) => acc + item.weight, 0)
      let wetnessScore = weightedWetness.reduce((acc, item) => acc + item.value * item.weight, 0) / Math.max(weightSum, 1e-6)
      if (Number.isFinite(Number(bsiValue))) {
        const drynessPenalty = Math.max(0, Math.min(1, (Number(bsiValue) + 0.2) / 0.8))
        wetnessScore *= 1 - drynessPenalty * 0.35
      }
      const satelliteRainProxy = Math.round(Math.max(0, Math.min(120, wetnessScore * 80)))
      modelingForm.value.expected_rain_mm = satelliteRainProxy
      modelingAutoSources.value.expected_rain_mm = Number.isFinite(Number(bsiValue))
        ? 'satellite_wetness_bsi'
        : 'satellite_wetness'
    } else {
      const precipRaw = (
        waterBal.period_precipitation_mm
        ?? inputFeats.seasonal_precipitation_mm
        ?? inputFeats.current_precipitation_mm
        ?? inputFeats.precipitation_sum
        ?? null
      )
      if (precipRaw !== null && Number.isFinite(Number(precipRaw))) {
        modelingForm.value.expected_rain_mm = Math.round(Math.min(500, Math.max(0, Number(precipRaw))))
        modelingAutoSources.value.expected_rain_mm = 'observed_weather'
      } else {
        // Fallback: sum precipitation points from temporal analytics series
        const precipMetric = series.find((m) => m.metric === 'precipitation')
        if (precipMetric?.points?.length) {
          const precipSum = precipMetric.points.reduce((acc, p) => acc + (p.value ?? p.mean ?? 0), 0)
          if (Number.isFinite(precipSum) && precipSum > 0) {
            modelingForm.value.expected_rain_mm = Math.round(Math.min(500, Math.max(0, precipSum)))
            modelingAutoSources.value.expected_rain_mm = 'observed_weather'
          }
        }
      }
    }

    // soil_compaction: from soil moisture plus optional BSI dryness proxy.
    const soilMoistMean = soilMoistureValue
    const latestBsi = Number.isFinite(Number(bsiValue)) ? Number(bsiValue) : null
    if (soilMoistMean !== null && Number.isFinite(Number(soilMoistMean))) {
      const moistureComponent = Math.max(0, Math.min(1, 1.0 - Number(soilMoistMean) * 1.5))
      const bareSoilComponent = Number.isFinite(latestBsi)
        ? Math.max(0, Math.min(1, (Number(latestBsi) + 0.2) / 0.8))
        : null
      const rawCompaction = bareSoilComponent === null
        ? moistureComponent
        : moistureComponent * 0.75 + bareSoilComponent * 0.25
      const comp = Math.round(Math.max(0, Math.min(1, rawCompaction)) * 100) / 100
      modelingForm.value.soil_compaction = comp
      modelingAutoSources.value.soil_compaction = bareSoilComponent === null
        ? 'satellite_soil_moisture'
        : 'satellite_soil_moisture_bsi'
    }

    // temperature_delta_c: keep at 0 — user intent, not satellite-derived
    modelingForm.value.temperature_delta_c = 0

    // management-based: if no management history reset defaults to neutral
    const mgmtTotal = inputFeats.management_total_amount ?? null
    if (mgmtTotal !== null && Number(mgmtTotal) < 1.0) {
      // Field with no management history — default to small positive intervention
      modelingForm.value.irrigation_pct = 10
      modelingForm.value.fertilizer_pct = 5
    }
  }

  function enableAutoModeling() {
    useManualModeling.value = false
    autoFillModelingFactors()
  }

  function enableManualModeling() {
    useManualModeling.value = true
  }

  watch(useManualModeling, (manualMode) => {
    if (!manualMode) {
      autoFillModelingFactors()
    }
  })

  async function simulateScenario() {
    if (!selectedField.value?.field_id || hasGroupSelection.value) {
      return null
    }
    if (selectedFieldIsPreviewOnly.value) {
      addLog(t('field.previewOnlyActionHint'))
      return null
    }
    if (isSimulatingScenario.value) {
      return null
    }
    isSimulatingScenario.value = true
    scenarioTaskState.value = null
    scenarioTaskProgress.value = 5
    addLog(`Запускаю моделирование сценария для поля ${selectedField.value.field_id}...`)
    const requestConfig = nextRequestConfig('scenario')
    try {
      if (!useManualModeling.value) {
        autoFillModelingFactors()
      }
      const normalizedFactors = {
        irrigation_pct: clampNumber(modelingForm.value.irrigation_pct, 0, -100, 100),
        fertilizer_pct: clampNumber(modelingForm.value.fertilizer_pct, 0, -100, 100),
        expected_rain_mm: clampNumber(modelingForm.value.expected_rain_mm, 0, 0, 500),
        temperature_delta_c: clampNumber(modelingForm.value.temperature_delta_c, 0, -10, 10),
        planting_density_pct: clampNumber(modelingForm.value.planting_density_pct, 0, -80, 100),
        tillage_type: modelingForm.value.tillage_type === '' || modelingForm.value.tillage_type === null || modelingForm.value.tillage_type === undefined
          ? null
          : clampNumber(Number(modelingForm.value.tillage_type), 0, 0, 3),
        pest_pressure: modelingForm.value.pest_pressure === '' || modelingForm.value.pest_pressure === null || modelingForm.value.pest_pressure === undefined
          ? null
          : clampNumber(Number(modelingForm.value.pest_pressure), 0, 0, 3),
        soil_compaction: modelingForm.value.soil_compaction === '' || modelingForm.value.soil_compaction === null || modelingForm.value.soil_compaction === undefined
          ? null
          : clampNumber(Number(modelingForm.value.soil_compaction), 0, 0, 1),
        cloud_cover_factor: clampNumber(Number(modelingForm.value.cloud_cover_factor ?? 1.0), 1.0, 0.1, 3.0),
      }
      const wasNormalized =
        normalizedFactors.irrigation_pct !== Number(modelingForm.value.irrigation_pct || 0) ||
        normalizedFactors.fertilizer_pct !== Number(modelingForm.value.fertilizer_pct || 0) ||
        normalizedFactors.expected_rain_mm !== Number(modelingForm.value.expected_rain_mm || 0) ||
        normalizedFactors.temperature_delta_c !== Number(modelingForm.value.temperature_delta_c || 0) ||
        normalizedFactors.planting_density_pct !== Number(modelingForm.value.planting_density_pct || 0) ||
        normalizedFactors.tillage_type !== (
          modelingForm.value.tillage_type === '' || modelingForm.value.tillage_type === null || modelingForm.value.tillage_type === undefined
            ? null
            : Number(modelingForm.value.tillage_type)
        ) ||
        normalizedFactors.pest_pressure !== (
          modelingForm.value.pest_pressure === '' || modelingForm.value.pest_pressure === null || modelingForm.value.pest_pressure === undefined
            ? null
            : Number(modelingForm.value.pest_pressure)
        ) ||
        normalizedFactors.soil_compaction !== (
          modelingForm.value.soil_compaction === '' || modelingForm.value.soil_compaction === null || modelingForm.value.soil_compaction === undefined
            ? null
            : Number(modelingForm.value.soil_compaction)
        )
      if (wasNormalized) {
        modelingForm.value = {
          ...modelingForm.value,
          ...normalizedFactors,
        }
        addLog('Параметры сценария были ограничены допустимым диапазоном.')
      }
      const response = await axios.post(`${API_BASE}/modeling/jobs`, {
        field_id: selectedField.value.field_id,
        crop_code: selectedCropCode.value || null,
        scenario_name: scenarioName.value || null,
        irrigation_pct: normalizedFactors.irrigation_pct,
        fertilizer_pct: normalizedFactors.fertilizer_pct,
        expected_rain_mm: normalizedFactors.expected_rain_mm,
        temperature_delta_c: normalizedFactors.temperature_delta_c,
        planting_density_pct: normalizedFactors.planting_density_pct,
        tillage_type: normalizedFactors.tillage_type,
        pest_pressure: normalizedFactors.pest_pressure,
        soil_compaction: normalizedFactors.soil_compaction,
        cloud_cover_factor: normalizedFactors.cloud_cover_factor,
      }, requestConfig)
      applyAsyncTaskStatus(scenarioTaskProgress, scenarioTaskState, response.data)
      addLog(`Задача сценария поставлена в очередь: ${response.data.task_id}`)
      return await pollAsyncJob({
        kind: 'scenario',
        taskId: response.data.task_id,
        statusUrl: `${API_BASE}/modeling/jobs/${response.data.task_id}`,
        resultUrl: `${API_BASE}/modeling/jobs/${response.data.task_id}/result`,
        progressRef: scenarioTaskProgress,
        stateRef: scenarioTaskState,
        onResult: async (payload) => {
          modelingResult.value = payload
          const fieldId = payload?.field_id || selectedField.value?.field_id
          if (fieldId) {
            await loadFieldDashboard(fieldId)
          }
        },
        onDoneMessage: `Сценарий смоделирован для поля ${selectedField.value.field_id}`,
        onFailureMessage: 'Не удалось смоделировать сценарий',
      })
    } catch (requestError) {
      if (isAbortError(requestError)) {
        return null
      }
      addLog(`Не удалось смоделировать сценарий: ${resolveError(requestError)}`)
      scenarioTaskProgress.value = 0
      return null
    } finally {
      isSimulatingScenario.value = false
    }
  }

  async function fetchSensitivitySweep(sweepParam) {
    if (!selectedField.value?.field_id || !modelingResult.value) {
      return null
    }
    isLoadingSensitivity.value = true
    sensitivityData.value = null
    try {
      const factors = modelingResult.value.factors || {}
      const response = await axios.post(`${API_BASE}/modeling/sensitivity`, {
        field_id: selectedField.value.field_id,
        crop_code: selectedCropCode.value || null,
        sweep_param: sweepParam,
        sweep_min: sweepParam === 'expected_rain_mm' ? 0 : -80,
        sweep_max: sweepParam === 'expected_rain_mm' ? 200 : 80,
        sweep_steps: 11,
        base_adjustments: factors,
      })
      sensitivityData.value = response.data
      addLog(`Анализ чувствительности загружен: ${sweepParam}`)
      return response.data
    } catch (err) {
      addLog(`Ошибка анализа чувствительности: ${resolveError(err)}`)
      return null
    } finally {
      isLoadingSensitivity.value = false
    }
  }

  async function createArchiveForSelectedField() {
    if (!selectedField.value?.field_id || hasGroupSelection.value) {
      return null
    }
    if (selectedFieldIsPreviewOnly.value) {
      addLog(t('field.previewOnlyActionHint'))
      return null
    }
    if (isCreatingArchive.value) {
      return null
    }
    isCreatingArchive.value = true
    const requestConfig = nextRequestConfig('archiveCreate')
    try {
      const response = await axios.post(`${API_BASE}/archive/create`, {
        field_id: selectedField.value.field_id,
        date_from: new Date(`${startDate.value}T00:00:00`).toISOString(),
        date_to: new Date(`${endDate.value}T23:59:59`).toISOString(),
        layers: [primaryLayerId.value, 'weather'],
      }, requestConfig)
      fieldArchives.value = [response.data, ...fieldArchives.value]
      addLog(`Архив создан: ${response.data.id}`)
      await Promise.allSettled([
        loadFieldDashboard(selectedField.value.field_id),
        loadArchiveView(response.data.id),
      ])
      return response.data
    } catch (requestError) {
      if (isAbortError(requestError)) {
        return null
      }
      addLog(`Не удалось создать архив: ${resolveError(requestError)}`)
      return null
    } finally {
      isCreatingArchive.value = false
    }
  }

  async function loadFieldArchives() {
    if (!selectedField.value?.field_id || hasGroupSelection.value) {
      fieldArchives.value = []
      return []
    }
    const requestConfig = nextRequestConfig('archiveList')
    try {
      const response = await axios.get(`${API_BASE}/archive`, {
        ...requestConfig,
        params: { field_id: selectedField.value.field_id },
      })
      fieldArchives.value = response.data.archives || []
      return fieldArchives.value
    } catch (requestError) {
      if (isAbortError(requestError)) {
        return []
      }
      addLog(`Не удалось загрузить архивы: ${resolveError(requestError)}`)
      return []
    }
  }

  function persistUiSettings() {
    if (typeof window === 'undefined') {
      return
    }
    window.localStorage.setItem(
      SETTINGS_STORAGE_KEY,
      JSON.stringify({
        centerLat: centerLat.value,
        centerLon: centerLon.value,
        radiusKm: radiusKm.value,
        startDate: startDate.value,
        endDate: endDate.value,
        resolutionM: resolutionM.value,
        maxCloudPct: maxCloudPct.value,
        targetDates: targetDates.value,
        minFieldAreaHa: minFieldAreaHa.value,
        selectedCropCode: selectedCropCode.value,
        metricsDisplayMode: metricsDisplayMode.value,
        metricsSelectedSeries: metricsSelectedSeries.value,
        forecastGraphMode: forecastGraphMode.value,
        scenarioGraphMode: scenarioGraphMode.value,
        showForecastGraphs: showForecastGraphs.value,
        showScenarioGraphs: showScenarioGraphs.value,
        showScenarioFactors: showScenarioFactors.value,
        showScenarioRisks: showScenarioRisks.value,
        showManagementZonesOverlay: showManagementZonesOverlay.value,
        useSam: useSam.value,
        showFieldsOnly: showFieldsOnly.value,
        showFieldBoundaries: showFieldBoundaries.value,
        detectionPreset: detectionPreset.value,
        autoRefreshIntervalS: autoRefreshIntervalS.value,
        progressVerbosity: progressVerbosity.value,
        animationDensity: animationDensity.value,
        showFreshnessBadges: showFreshnessBadges.value,
        mapLabelDensity: mapLabelDensity.value,
        expertMode: expertMode.value,
        beginnerMode: beginnerMode.value,
        showSatelliteBrowse: showSatelliteBrowse.value,
        satelliteBrowseDate: satelliteBrowseDate.value,
        activeLayers: activeLayers.value,
        uiWindows: uiWindows.value,
      })
    )
  }

  function restoreUiSettings() {
    if (typeof window === 'undefined') {
      return
    }
    let raw = window.localStorage.getItem(SETTINGS_STORAGE_KEY)
    if (!raw) {
      raw = window.localStorage.getItem(LEGACY_SETTINGS_STORAGE_KEY)
    }
    if (!raw) {
      return
    }
    try {
      const value = JSON.parse(raw)
      centerLat.value = Number(value.centerLat || centerLat.value)
      centerLon.value = Number(value.centerLon || centerLon.value)
      radiusKm.value = Number(value.radiusKm || radiusKm.value)
      startDate.value = value.startDate || startDate.value
      endDate.value = value.endDate || endDate.value
      resolutionM.value = Number(value.resolutionM || resolutionM.value)
      maxCloudPct.value = Number(value.maxCloudPct || maxCloudPct.value)
      targetDates.value = clampNumber(value.targetDates, targetDates.value, TARGET_DATES_MIN, TARGET_DATES_MAX)
      minFieldAreaHa.value = Number(value.minFieldAreaHa || minFieldAreaHa.value)
      selectedCropCode.value = value.selectedCropCode || selectedCropCode.value
      metricsDisplayMode.value = value.metricsDisplayMode || metricsDisplayMode.value
      metricsSelectedSeries.value = value.metricsSelectedSeries || metricsSelectedSeries.value
      forecastGraphMode.value = value.forecastGraphMode || forecastGraphMode.value
      scenarioGraphMode.value = value.scenarioGraphMode || scenarioGraphMode.value
      showForecastGraphs.value = Boolean(value.showForecastGraphs ?? showForecastGraphs.value)
      showScenarioGraphs.value = Boolean(value.showScenarioGraphs ?? showScenarioGraphs.value)
      showScenarioFactors.value = Boolean(value.showScenarioFactors ?? showScenarioFactors.value)
      showScenarioRisks.value = Boolean(value.showScenarioRisks ?? showScenarioRisks.value)
      showManagementZonesOverlay.value = Boolean(value.showManagementZonesOverlay ?? showManagementZonesOverlay.value)
      useSam.value = Boolean(value.useSam ?? useSam.value)
      showFieldsOnly.value = Boolean(value.showFieldsOnly ?? showFieldsOnly.value)
      showFieldBoundaries.value = Boolean(value.showFieldBoundaries ?? showFieldBoundaries.value)
      detectionPreset.value = value.detectionPreset || detectionPreset.value
      autoRefreshIntervalS.value = Number(value.autoRefreshIntervalS ?? autoRefreshIntervalS.value)
      progressVerbosity.value = value.progressVerbosity || progressVerbosity.value
      animationDensity.value = value.animationDensity || animationDensity.value
      showFreshnessBadges.value = Boolean(value.showFreshnessBadges ?? showFreshnessBadges.value)
      mapLabelDensity.value = value.mapLabelDensity || mapLabelDensity.value
      expertMode.value = Boolean(value.expertMode ?? expertMode.value)
      beginnerMode.value = Boolean(value.beginnerMode ?? beginnerMode.value)
      showSatelliteBrowse.value = Boolean(value.showSatelliteBrowse ?? showSatelliteBrowse.value)
      satelliteBrowseDate.value = value.satelliteBrowseDate || satelliteBrowseDate.value
      if (value.activeLayers && typeof value.activeLayers === 'object') {
        activeLayers.value = {
          ...activeLayers.value,
          ...value.activeLayers,
        }
      }
      if (value.uiWindows && typeof value.uiWindows === 'object') {
        uiWindows.value = {
          ...uiWindows.value,
          ...value.uiWindows,
          control: true,
        }
      }
      applyDetectionPreset(detectionPreset.value)
    } catch {
      window.localStorage.removeItem(SETTINGS_STORAGE_KEY)
    }
  }

  function applyDetectionPreset(preset) {
    const profile = DETECTION_PRESETS[preset] || DETECTION_PRESETS.standard
    detectionPreset.value = DETECTION_PRESETS[preset] ? preset : 'standard'
    useSam.value = profile.useSam
    targetDates.value = profile.targetDates
    resolutionM.value = profile.resolutionM
    minFieldAreaHa.value = profile.minFieldAreaHa
    if (detectionPreset.value === 'fast') {
      radiusKm.value = clampNumber(radiusKm.value, profile.recommendedRadiusKm, profile.recommendedRadiusKm, profile.maxRadiusKm)
      return
    }
    radiusKm.value = clampNumber(radiusKm.value, profile.recommendedRadiusKm, 1, profile.maxRadiusKm)
  }

  function restartSystemPolling() {
    startSystemPolling()
  }

  async function createManualField(geometry) {
    isCreatingManualField.value = true
    try {
      const response = await axios.post(`${API_BASE}/manual/fields`, {
        geometry,
        quality_score: 1.0,
      })
      await loadManualFields()
      selectedFieldIds.value = [response.data.field.id]
      syncSelectedFieldFromCollections(response.data.field.id)
      await loadSelectionAnalytics()
      addLog(`Ручное поле создано: ${response.data.field.id}`)
      return response.data.field
    } catch (requestError) {
      addLog(`Не удалось сохранить ручное поле: ${resolveError(requestError)}`)
      return null
    } finally {
      isCreatingManualField.value = false
      drawMode.value = false
    }
  }

  async function loadManualFields() {
    const requestConfig = nextRequestConfig('manualFields')
    try {
      const response = await axios.get(`${API_BASE}/manual/fields/geojson`, requestConfig)
      manualFieldsGeoJson.value = response.data || { type: 'FeatureCollection', features: [] }
      syncSelectedFieldFromCollections()
      return manualFieldsGeoJson.value
    } catch (requestError) {
      if (isAbortError(requestError)) {
        return manualFieldsGeoJson.value
      }
      if (requestError?.response?.status !== 404) {
        addLog(`Не удалось загрузить ручные поля: ${resolveError(requestError)}`)
      }
      return manualFieldsGeoJson.value
    }
  }

  function startMergeMode() {
    splitMode.value = false
    drawMode.value = false
    mergeMode.value = true
    mergeSelectionIds.value = selectedField.value?.field_id ? [selectedField.value.field_id] : []
    addLog('Режим объединения активирован: выберите поля на карте и нажмите объединить')
  }

  function cancelMergeMode() {
    mergeMode.value = false
    mergeSelectionIds.value = []
    addLog('Режим объединения отменён')
  }

  function toggleMergeFieldSelection(fieldId) {
    if (!mergeMode.value || !fieldId) {
      return
    }
    const next = new Set(mergeSelectionIds.value)
    if (next.has(fieldId)) {
      next.delete(fieldId)
    } else {
      next.add(fieldId)
    }
    mergeSelectionIds.value = [...next]
  }

  async function mergeSelectedFields() {
    if (mergeSelectionIds.value.length < 2) {
      addLog('Для объединения нужно выбрать минимум два поля')
      return null
    }
    try {
      const response = await axios.post(`${API_BASE}/fields/merge`, {
        field_ids: mergeSelectionIds.value,
      })
      await Promise.allSettled([loadPersistedFields(), loadManualFields()])
      selectedFieldIds.value = [response.data.id]
      syncSelectedFieldFromCollections(response.data.id)
      addLog(`Поля объединены: ${mergeSelectionIds.value.length} → 1 (${response.data.id})`)
      mergeMode.value = false
      mergeSelectionIds.value = []
      await loadSelectionAnalytics()
      return response.data
    } catch (requestError) {
      addLog(`Не удалось объединить поля: ${resolveError(requestError)}`)
      return null
    }
  }

  function startSplitMode() {
    if (!selectedField.value?.field_id) {
      addLog('Сначала выберите поле для разделения')
      return
    }
    mergeMode.value = false
    mergeSelectionIds.value = []
    drawMode.value = false
    splitMode.value = true
    addLog('Режим разделения активирован: проведите линию через выбранное поле')
  }

  function cancelSplitMode() {
    splitMode.value = false
    addLog('Режим разделения отменён')
  }

  async function splitSelectedField(geometry) {
    if (!selectedField.value?.field_id) {
      return null
    }
    try {
      const response = await axios.post(`${API_BASE}/fields/split`, {
        field_id: selectedField.value.field_id,
        geometry,
      })
      await Promise.allSettled([loadPersistedFields(), loadManualFields()])
      addLog(`Поле разделено на ${response.data.fields?.length || 0} части`)
      splitMode.value = false
      if (response.data.fields?.length) {
        selectedFieldIds.value = [response.data.fields[0].id]
        syncSelectedFieldFromCollections(response.data.fields[0].id)
        await loadSelectionAnalytics()
      }
      return response.data.fields || []
    } catch (requestError) {
      addLog(`Не удалось разделить поле: ${resolveError(requestError)}`)
      return null
    }
  }

  async function deleteSelectedField() {
    if (!selectedField.value?.field_id) {
      addLog('Для удаления выберите поле')
      return false
    }
    const fieldId = selectedField.value.field_id
    if (typeof window !== 'undefined' && typeof window.confirm === 'function') {
      const confirmed = window.confirm(
        `Удалить поле ${fieldId}? Это действие нельзя отменить.`
      )
      if (!confirmed) {
        addLog(`Удаление поля отменено: ${fieldId}`)
        return false
      }
    }
    try {
      await axios.delete(`${API_BASE}/fields/${fieldId}`)
      addLog(`Поле удалено: ${fieldId}`)

      // Immediately remove the field from local GeoJSON collections
      // to avoid waiting for a full reload
      if (fieldsGeoJson.value?.features) {
        fieldsGeoJson.value = {
          ...fieldsGeoJson.value,
          features: fieldsGeoJson.value.features.filter(
            (f) => f.properties?.field_id !== fieldId,
          ),
        }
      }
      if (manualFieldsGeoJson.value?.features) {
        manualFieldsGeoJson.value = {
          ...manualFieldsGeoJson.value,
          features: manualFieldsGeoJson.value.features.filter(
            (f) => f.properties?.field_id !== fieldId,
          ),
        }
      }

      clearFieldSelection()

      // Background reload to sync with server state
      Promise.allSettled([loadPersistedFields(), loadManualFields()]).catch(() => {})
      return true
    } catch (requestError) {
      addLog(`Не удалось удалить поле: ${resolveError(requestError)}`)
      return false
    }
  }

  async function loadFieldsList() {
    try {
      const response = await axios.get(`${API_BASE}/fields`, {
        params: visibleRunId.value ? { aoi_run_id: visibleRunId.value } : {},
      })
      const features = (response.data.fields || []).map((field) => ({
        type: 'Feature',
        geometry: null,
        properties: {
          field_id: field.id,
          aoi_run_id: field.aoi_run_id,
          area_m2: field.area_m2,
          perimeter_m: field.perimeter_m,
          quality_score: field.quality_score,
          source: field.source,
        },
      }))
      if (!fieldsGeoJson.value || !fieldsGeoJson.value.features?.length) {
        return
      }
      const byId = new Map(features.map((feature) => [feature.properties.field_id, feature.properties]))
      fieldsGeoJson.value = {
        ...fieldsGeoJson.value,
        features: fieldsGeoJson.value.features.map((feature) => ({
          ...feature,
          properties: {
            ...feature.properties,
            ...(byId.get(feature.properties.field_id) || {}),
          },
        })),
      }
    } catch (requestError) {
      addLog(`Не удалось обновить список полей: ${resolveError(requestError)}`)
    }
  }

  function resolveError(requestError) {
    const detail = requestError?.response?.data?.detail
    if (Array.isArray(detail)) {
      return detail
        .map((item) => {
          if (typeof item === 'object' && item !== null) {
            const loc = Array.isArray(item.loc) ? item.loc.join(' → ') : ''
            return loc ? `${loc}: ${item.msg}` : item.msg || JSON.stringify(item)
          }
          return String(item)
        })
        .join('; ')
    }
    if (detail && typeof detail === 'object') {
      return detail.msg || JSON.stringify(detail)
    }
    return detail || requestError?.message || 'Неизвестная ошибка'
  }

  function setActiveFieldTab(tabId) {
    activeFieldTab.value = tabId
  }

  return {
    centerLat,
    centerLon,
    radiusKm,
    startDate,
    endDate,
    resolutionM,
    maxCloudPct,
    targetDates,
    minFieldAreaHa,
    useSam,
    showFieldsOnly,
    showFieldBoundaries,
    drawMode,
    mergeMode,
    splitMode,
    mergeSelectionIds,
    isPickingSearchCenter,
    activeRunId,
    visibleRunId,
    lastCompletedRunId,
    currentRunId,
    runSummaries,
    runStatus,
    runProgress,
    runRuntime,
    lastPreflight,
    runStageCode,
    runStageLabel,
    runStageDetailCode,
    runStageDetailParams,
    runStageDetail,
    runStartedAt,
    runUpdatedAt,
    runLastHeartbeatTs,
    runStaleRunning,
    runEstimatedRemainingS,
    debugTilesCatalog,
    selectedDebugRunId,
    selectedDebugTileId,
    selectedDebugLayerId,
    selectedDebugTileDetail,
    debugOverlayEnabled,
    debugOverlayOpacity,
    debugLayerPayload,
    isLoadingDebugTiles,
    isLoadingDebugLayer,
    fieldsGeoJson,
    manualFieldsGeoJson,
    availableLayers,
    activeLayers,
    showSatelliteBrowse,
    satelliteBrowseDate,
    satelliteScene,
    satelliteLoadStatus,
    activeLayerIds,
    primaryLayerId,
    isDetecting,
    isLoadingWeather,
    isCreatingManualField,
    error,
    logs,
    weatherCurrent,
    weatherForecast,
    systemStatus,
    lastWeatherUpdatedAt,
    lastStatusUpdatedAt,
    lastWeatherSyncState,
    lastWeatherSyncDetail,
    lastStatusSyncState,
    lastStatusSyncDetail,
    selectedField,
    selectedFieldIds,
    selectedFieldCount,
    hasGroupSelection,
    selectedFieldSource,
    selectedFieldIsPreviewOnly,
    fieldDashboard,
    groupDashboard,
    fieldTemporalAnalytics,
    fieldForecastAnalytics,
    fieldManagementZones,
    activeDashboard,
    activeDetectionPreset,
    selectedFieldPrediction,
    crops,
    selectedCropCode,
    isRefreshingPrediction,
    isSimulatingScenario,
    predictionTaskProgress,
    scenarioTaskProgress,
    temporalAnalyticsTaskProgress,
    predictionTaskState,
    scenarioTaskState,
    temporalAnalyticsTaskState,
    isCreatingArchive,
    isLoadingFieldDashboard,
    isLoadingGroupDashboard,
    isLoadingArchiveView,
    isLoadingTemporalAnalytics,
    isLoadingManagementZones,
    gridLayerStatus,
    modelingForm,
    scenarioName,
    modelingResult,
    sensitivityData,
    isLoadingSensitivity,
    useManualModeling,
    modelingAutoSources,
    enableAutoModeling,
    enableManualModeling,
    autoFillModelingFactors,
    fieldScenarios,
    fieldArchives,
    selectedArchiveView,
    activeFieldTab,
    metricsDisplayMode,
    metricsSelectedSeries,
    forecastGraphMode,
    scenarioGraphMode,
    showForecastGraphs,
    showScenarioGraphs,
    showScenarioFactors,
    showScenarioRisks,
    showManagementZonesOverlay,
    fieldEvents,
    fieldEventsTotal,
    isLoadingEvents,
    isSubmittingEvent,
    selectedEventSeasonYear,
    loadFieldEvents,
    createFieldEvent,
    updateFieldEvent,
    deleteFieldEvent,
    seriesDateFrom,
    seriesDateTo,
    detectionPreset,
    autoRefreshIntervalS,
    progressVerbosity,
    animationDensity,
    showFreshnessBadges,
    mapLabelDensity,
    expertMode,
    beginnerMode,
    uiWindows,
    addLog,
    persistLogs,
    clearLogs,
    initialize,
    loadSystemStatus,
    loadWeather,
    loadSatelliteScene,
    refreshAll,
    loadCrops,
    loadRunSummaries,
    loadRunDebugTiles,
    loadDebugTile,
    loadDebugLayer,
    loadFieldDashboard,
    loadFieldTemporalAnalytics,
    runTemporalAnalyticsBackfill,
    loadFieldManagementZones,
    loadGroupDashboard,
    loadSelectionAnalytics,
    loadArchiveView,
    startDetection,
    pollStatus,
    fetchResult,
    loadPersistedFields,
    loadManualFields,
    loadLayers,
    toggleLayer,
    selectField,
    refreshPrediction,
    simulateScenario,
    fetchSensitivitySweep,
    createArchiveForSelectedField,
    loadFieldArchives,
    applyDetectionPreset,
    restartSystemPolling,
    clearDebugOverlay,
    toggleWindow,
    showWindow,
    hideWindow,
    clearFieldSelection,
    toggleSearchCenterPicking,
    applySearchCenter,
    createManualField,
    startMergeMode,
    cancelMergeMode,
    toggleMergeFieldSelection,
    mergeSelectedFields,
    startSplitMode,
    cancelSplitMode,
    splitSelectedField,
    deleteSelectedField,
    setActiveFieldTab,
    clearTimers,
    cancelRequest,
    nextRequestConfig,
    isAbortError,
  }
})
