/**
 * useFieldPanelData — composable that centralises all reactive data and
 * business logic for FieldActionsPanel.vue (and its future sub-components).
 *
 * Extracted from the monolithic FieldActionsPanel.vue to improve
 * readability and testability.  The component itself becomes a thin
 * presenter that delegates every computed property here.
 */
import { computed, ref, watch, watchEffect } from 'vue'
import { useMapStore } from '../store/map'
import { locale, t } from '../utils/i18n'
import {
  formatDisplayValue,
  formatReasonText,
  formatUiDateTime,
  formatUiProgress,
  getConfidenceTierLabel,
  getFeatureLabel,
  getLayerMeta,
  getQualityBandLabel,
  getRiskItemLabel,
  getRiskItemReason,
  getRiskLevelLabel,
  getSourceLabel,
  getTaskStageDetail,
  getTaskStageLabel,
} from '../utils/presentation'

// ── Colour palette shared across charts ──────────────────────────────────────
const SERIES_PALETTE = {
  ndvi: '#2f8a63',
  ndmi: '#1f6aa0',
  ndwi: '#1b87b7',
  bsi: '#b37632',
  gdd: '#c98b24',
  gdd_daily: '#c98b24',
  gdd_cumulative: '#d28a1f',
  vpd: '#8e4fc6',
  soil_moisture: '#3d8f7f',
  precipitation: '#3f7ee8',
  precipitation_mm: '#3f7ee8',
  temperature_mean_c: '#c45b2d',
  wind: '#5d6fb3',
}

const TEMPORAL_WEATHER_METRICS = new Set([
  'precipitation', 'soil_moisture', 'vpd', 'wind', 'gdd', 'temperature_mean_c',
])

// ── Pure format helpers (no store dependency) ─────────────────────────────────
export function formatYield(value) {
  return value === null || value === undefined ? '—' : `${Number(value).toFixed(0)} кг/га`
}
export function formatPercent(value) {
  return value === null || value === undefined ? '—' : `${(Number(value) * 100).toFixed(0)}%`
}
export function formatSigned(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—'
  const n = Number(value)
  return `${n >= 0 ? '+' : ''}${n.toFixed(3)}`
}
export function formatDelta(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—'
  const n = Number(value)
  return `${n >= 0 ? '+' : ''}${n.toFixed(2)}%`
}
export function formatInteger(value) {
  return Number.isFinite(Number(value)) ? String(Math.round(Number(value))) : '—'
}
export function formatDecimal(value) {
  return Number.isFinite(Number(value)) ? Number(value).toFixed(3) : '—'
}
export function formatFeatureValue(key, value, expertMode = false) {
  return formatDisplayValue(key, value, { expertMode })
}
export function formatMetricValue(metricId, value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—'
  const n = Number(value)
  if (metricId === 'precipitation') return `${n.toFixed(1)} мм`
  if (metricId === 'wind') return `${n.toFixed(1)} м/с`
  if (metricId === 'gdd') return n.toFixed(0)
  if (metricId === 'soil_moisture') return `${(n * 100).toFixed(0)}%`
  if (metricId === 'vpd') return n.toFixed(2)
  return n.toFixed(3)
}
export function resolveMetricColor(metricId) {
  return SERIES_PALETTE[metricId] || '#21579c'
}
export function formatSeconds(value, currentLocale) {
  const total = Math.max(0, Math.round(Number(value) || 0))
  const minutes = Math.floor(total / 60)
  const seconds = total % 60
  const isRu = currentLocale !== 'en'
  if (minutes >= 60) {
    const hours = Math.floor(minutes / 60)
    const rm = minutes % 60
    return isRu ? `${hours}ч ${rm}м` : `${hours}h ${rm}m`
  }
  if (minutes > 0) return isRu ? `${minutes}м ${seconds}с` : `${minutes}m ${seconds}s`
  return isRu ? `${seconds}с` : `${seconds}s`
}
export function formatTaskTiming(payload, currentLocale) {
  if (!payload) return '—'
  const elapsed = Number(payload.elapsed_s)
  const eta = Number(payload.estimated_remaining_s)
  const isRu = currentLocale !== 'en'
  const elapsedLabel = Number.isFinite(elapsed)
    ? (isRu ? `Прошло ${formatSeconds(elapsed, currentLocale)}` : `Elapsed ${formatSeconds(elapsed, currentLocale)}`)
    : null
  const etaLabel = Number.isFinite(eta) ? `ETA ${formatSeconds(Math.max(0, eta), currentLocale)}` : null
  return [elapsedLabel, etaLabel].filter(Boolean).join(' · ') || '—'
}
export function formatDateTime(value) {
  return formatUiDateTime(value, { day: '2-digit', month: '2-digit', year: 'numeric', hour: '2-digit', minute: '2-digit' })
}
export function formatDateShort(value) {
  if (!value) return ''
  return formatUiDateTime(value, { day: '2-digit', month: '2-digit', year: 'numeric' })
}

// ── Series utilities ──────────────────────────────────────────────────────────
export function seriesAreIdentical(a, b) {
  if (!a?.length || !b?.length || a.length !== b.length) return false
  return a.every((ap, i) => {
    const bv = Number(b[i]?.value ?? b[i]?.y)
    const av = Number(ap?.value ?? ap?.y)
    if (!Number.isFinite(av) || !Number.isFinite(bv)) return true
    return Math.abs(av - bv) / Math.max(Math.abs(av), Math.abs(bv), 1e-9) <= 0.02
  })
}
export function mapForecastCurvePoints(points, metricId) {
  return (points || [])
    .map(p => ({ date: p?.date, value: p?.[metricId] }))
    .filter(p => p.date && p.value !== null && p.value !== undefined && !Number.isNaN(Number(p.value)))
}

// ── Main composable ───────────────────────────────────────────────────────────
export function useFieldPanelData() {
  const store = useMapStore()

  // Core derived state
  const isGroup = computed(() => store.hasGroupSelection)
  const dashboard = computed(() => store.activeDashboard)
  const isLoadingDashboard = computed(() => store.isLoadingFieldDashboard || store.isLoadingGroupDashboard)
  const isPreviewField = computed(() => !isGroup.value && store.selectedFieldIsPreviewOnly)
  const prediction = computed(() => store.selectedFieldPrediction || dashboard.value?.prediction)
  const temporalAnalytics = computed(() => store.fieldTemporalAnalytics || null)
  const forecastTemporalAnalytics = computed(() => store.fieldForecastAnalytics || null)
  const managementZones = computed(() => store.fieldManagementZones || null)
  const analyticsSummary = computed(() => dashboard.value?.analytics_summary || temporalAnalytics.value?.analytics_summary || {})
  const metricsDataStatus = computed(() => temporalAnalytics.value?.data_status || null)
  const forecastDataStatus = computed(() => forecastTemporalAnalytics.value?.data_status || null)
  const waterBalance = computed(() => forecastTemporalAnalytics.value?.water_balance || prediction.value?.water_balance || {})
  const riskSummary = computed(() => forecastTemporalAnalytics.value?.risk || prediction.value?.risk || {})
  const managementZoneSummary = computed(() =>
    prediction.value?.management_zone_summary || managementZones.value?.summary || dashboard.value?.zones_summary || {})
  const debugRuntime = computed(() => store.selectedDebugTileDetail?.runtime_meta || null)

  // Visible tabs
  const visibleTabs = computed(() => {
    if (isGroup.value || isPreviewField.value) {
      return [
        { id: 'overview', label: t('field.overview') },
        { id: 'metrics', label: t('field.metricsTab') },
      ]
    }
    return [
      { id: 'overview', label: t('field.overview') },
      { id: 'metrics', label: t('field.metricsTab') },
      { id: 'forecast', label: t('field.forecastTab') },
      { id: 'scenarios', label: t('field.scenarioTab') },
      { id: 'archive', label: t('field.archiveTab') },
      { id: 'events', label: t('field.eventsTab') },
    ]
  })

  watchEffect(() => {
    if (isPreviewField.value && ['forecast', 'scenarios', 'archive', 'events'].includes(store.activeFieldTab)) {
      store.setActiveFieldTab('overview')
    }
  })

  // Tab auto-load
  watch(() => store.activeFieldTab, (tab, prevTab) => {
    if (tab === prevTab) return
    const fieldId = store.selectedField?.field_id
    if (!fieldId || store.hasGroupSelection) return
    if (tab === 'metrics') {
      store.loadFieldTemporalAnalytics(fieldId, { target: 'metrics', preferExisting: true, autoBackfill: true, silent: true })
    } else if (tab === 'forecast') {
      store.loadFieldTemporalAnalytics(fieldId, { target: 'forecast', preferExisting: true, autoBackfill: true, silent: true })
    } else if (tab === 'events') {
      store.loadFieldEvents()
    }
  })

  watch(() => store.metricsDisplayMode, (mode) => {
    if (!['xy', 'timeline', 'anomalies', 'cards'].includes(mode)) return
    const fieldId = store.selectedField?.field_id
    if (!fieldId || store.hasGroupSelection || store.activeFieldTab !== 'metrics') return
    store.loadFieldTemporalAnalytics(fieldId, { target: 'metrics', preferExisting: true, autoBackfill: true, silent: true })
  })

  watch(
    () => [store.fieldDashboard, store.fieldForecastAnalytics, store.fieldTemporalAnalytics],
    () => { if (!store.useManualModeling) store.autoFillModelingFactors() },
    { deep: false },
  )

  // Date range helpers
  let _dateDebounceTimer = null
  function onDateRangeChange() {
    if (store.seriesDateFrom && store.seriesDateTo && store.seriesDateFrom > store.seriesDateTo) {
      const tmp = store.seriesDateFrom
      store.seriesDateFrom = store.seriesDateTo
      store.seriesDateTo = tmp
    }
    store.loadFieldTemporalAnalytics(undefined, { target: 'metrics', preferExisting: false, autoBackfill: true })
  }
  function clearDateRange() {
    store.seriesDateFrom = ''
    store.seriesDateTo = ''
    store.loadFieldTemporalAnalytics(undefined, { target: 'metrics', preferExisting: false, autoBackfill: true })
  }
  function onDateInput() {
    if (_dateDebounceTimer) clearTimeout(_dateDebounceTimer)
    _dateDebounceTimer = setTimeout(() => { _dateDebounceTimer = null; onDateRangeChange() }, 600)
  }

  // Field labels
  const sourceLabel = computed(() => getSourceLabel(dashboard.value?.field?.source || store.selectedField?.source || '') || '—')
  const qualityLabel = computed(() => {
    const v = dashboard.value?.field?.quality_score ?? store.selectedField?.quality_score
    return v === null || v === undefined ? t('field.qualityUnknown') : `${t('field.quality')}: ${Number(v).toFixed(2)}`
  })
  const qualityBandLabel = computed(() => {
    const label = dashboard.value?.field?.quality_label || store.selectedField?.quality_label || dashboard.value?.field?.quality_band || store.selectedField?.quality_band
    return label ? `${t('field.qualityBand')}: ${getQualityBandLabel(label)}` : t('field.qualityUnknown')
  })
  const qualityReasonLabel = computed(() =>
    formatReasonText(
      dashboard.value?.field?.quality_reason_code || store.selectedField?.quality_reason_code,
      dashboard.value?.field?.quality_reason || store.selectedField?.quality_reason,
      dashboard.value?.field?.quality_reason_params || store.selectedField?.quality_reason_params,
    ) || t('field.qualityReasonMissing'))
  const fieldOperationalTierLabel = computed(() => {
    const tier = dashboard.value?.field?.operational_tier || store.selectedField?.operational_tier
    return tier ? t(`field.operationalTier.${tier}`) : '—'
  })
  const fieldReviewRequiredLabel = computed(() => {
    const r = dashboard.value?.field?.review_required ?? store.selectedField?.review_required
    return r ? t('field.yes') : t('field.no')
  })
  const fieldReviewReason = computed(() =>
    formatReasonText(
      dashboard.value?.field?.review_reason_code || store.selectedField?.review_reason_code,
      dashboard.value?.field?.review_reason || store.selectedField?.review_reason,
      dashboard.value?.field?.review_reason_params || store.selectedField?.review_reason_params,
    ) || '')
  const totalAreaLabel = computed(() => {
    const areaM2 = isGroup.value
      ? Number(dashboard.value?.selection?.total_area_m2 || 0)
      : Number(dashboard.value?.field?.area_m2 || store.selectedField?.area_m2 || 0)
    return `${(areaM2 / 10000).toFixed(2)} га`
  })
  const perimeterLabel = computed(() => {
    const v = Number(dashboard.value?.field?.perimeter_m || store.selectedField?.perimeter_m || 0)
    return v ? `${Math.round(v)} м` : '—'
  })
  const observationCellsLabel = computed(() => {
    const v = dashboard.value?.kpis?.observation_cells ?? dashboard.value?.data_quality?.observation_cells
    return v ? `${Math.round(Number(v))}` : '—'
  })
  const availableMetricsLabel = computed(() => {
    const keys = dashboard.value?.data_quality?.metrics_available || []
    return keys.length ? keys.map(m => getLayerMeta(m).label).join(', ') : t('field.noMetrics')
  })
  const dataQualityText = computed(() => {
    if (!dashboard.value?.data_quality) return t('field.noDataQuality')
    if (isGroup.value) return `${t('field.groupSelection')}: ${store.selectedFieldCount} · ${t('field.metricCoverage')}: ${observationCellsLabel.value}`
    return dashboard.value?.prediction?.data_quality?.confidence_reason || t('field.dataQualityReady')
  })
  const overviewCards = computed(() => {
    if (!dashboard.value) return []
    if (isGroup.value) return [
      { label: t('field.selectedCount'), value: String(store.selectedFieldCount) },
      { label: t('field.area'), value: totalAreaLabel.value },
      { label: t('field.availableMetrics'), value: availableMetricsLabel.value },
      { label: t('field.metricCoverage'), value: observationCellsLabel.value },
    ]
    if (isPreviewField.value) return [
      { label: t('field.area'), value: totalAreaLabel.value },
      { label: t('field.perimeter'), value: perimeterLabel.value },
      { label: t('field.metricsTab'), value: availableMetricsLabel.value },
      { label: t('field.reviewNeeded'), value: fieldReviewRequiredLabel.value },
      { label: t('field.previewOnlyTitle'), value: t('field.previewOnly') },
      { label: t('field.forecastReady'), value: t('field.no') },
    ]
    return [
      { label: t('field.area'), value: totalAreaLabel.value },
      { label: t('field.perimeter'), value: perimeterLabel.value },
      { label: t('field.archiveTab'), value: String(dashboard.value?.kpis?.archive_count || 0) },
      { label: t('field.scenarioTab'), value: String(dashboard.value?.kpis?.scenario_count || 0) },
      { label: t('field.metricCoverage'), value: observationCellsLabel.value },
      { label: t('field.forecastReady'), value: dashboard.value?.kpis?.prediction_ready ? t('field.yes') : t('field.no') },
    ]
  })

  // Metric cards
  const metricCards = computed(() => {
    const metrics = dashboard.value?.current_metrics || {}
    return Object.entries(metrics).map(([id, payload]) => ({
      id,
      label: getFeatureLabel(id, { expertMode: store.expertMode }),
      mean: formatMetricValue(id, payload.mean),
      median: formatMetricValue(id, payload.median),
      range: `${formatMetricValue(id, payload.min)} … ${formatMetricValue(id, payload.max)}`,
      coverage: payload.coverage ? `${Math.round(Number(payload.coverage))}` : '—',
    }))
  })

  const seriesEntries = computed(() => {
    const entries = {}
    const temporalMetrics = temporalAnalytics.value?.seasonal_series?.metrics || []
    const dashSeries = dashboard.value?.series || {}
    for (const [id, items] of Object.entries(dashSeries)) {
      if (TEMPORAL_WEATHER_METRICS.has(id) && temporalMetrics.some(m => m.metric === id)) continue
      entries[id] = {
        id,
        label: getFeatureLabel(id, { expertMode: store.expertMode }),
        items,
        latest: items?.length ? formatMetricValue(id, items[items.length - 1].mean) : '—',
        color: SERIES_PALETTE[id] || '#21579c',
      }
    }
    for (const metricObj of temporalMetrics) {
      const id = metricObj.metric
      if (entries[id]) continue
      const rawPoints = metricObj.points || []
      const items = rawPoints.map(p => ({ mean: p.smoothed ?? p.value, min: p.value, max: p.value, observed_at: p.observed_at }))
      entries[id] = {
        id,
        label: metricObj.label || getFeatureLabel(id, { expertMode: store.expertMode }),
        items,
        latest: items.length ? formatMetricValue(id, items[items.length - 1].mean) : '—',
        color: SERIES_PALETTE[id] || '#21579c',
      }
    }
    return Object.values(entries).sort((a, b) => {
      const aW = TEMPORAL_WEATHER_METRICS.has(a.id)
      const bW = TEMPORAL_WEATHER_METRICS.has(b.id)
      if (aW !== bW) return aW ? 1 : -1
      return 0
    })
  })

  const histogramEntries = computed(() => {
    const palette = { ndvi: '#2f8a63', ndmi: '#1f6aa0', soil_moisture: '#3d8f7f', vpd: '#8e4fc6' }
    const histograms = dashboard.value?.histograms || {}
    return Object.entries(histograms).map(([id, histogram]) => ({
      id,
      label: getFeatureLabel(id, { expertMode: store.expertMode }),
      histogram,
      color: palette[id] || '#2f8a63',
    }))
  })

  // Seasonal metrics
  const seasonalMetricEntries = computed(() => temporalAnalytics.value?.seasonal_series?.metrics || [])
  const selectedSeasonalMetricId = computed(() => {
    const configured = store.metricsSelectedSeries
    if (seasonalMetricEntries.value.some(item => item.metric === configured)) return configured
    return seasonalMetricEntries.value[0]?.metric || 'ndvi'
  })
  const selectedSeasonalMetric = computed(() =>
    seasonalMetricEntries.value.find(item => item.metric === selectedSeasonalMetricId.value) || seasonalMetricEntries.value[0] || null)
  const seasonalMetricSelectorOptions = computed(() =>
    seasonalMetricEntries.value.map(item => ({
      id: item.metric,
      label: item.label || getFeatureLabel(item.metric, { expertMode: store.expertMode }),
    })))
  const metricsAnomalyItems = computed(() => temporalAnalytics.value?.anomalies || [])
  const forecastAnomalyItems = computed(() => forecastTemporalAnalytics.value?.anomalies || prediction.value?.anomalies || [])
  const selectedMetricAnomalies = computed(() => {
    const id = selectedSeasonalMetricId.value
    return metricsAnomalyItems.value.filter(item => !item?.metric ? id === 'ndvi' : String(item.metric).toLowerCase() === String(id).toLowerCase())
  })
  const seasonalAnomalyRows = computed(() =>
    metricsAnomalyItems.value.map((item, i) => ({
      key: `${item.kind || item.label || 'anomaly'}-${item.observed_at || i}`,
      label: _formatAnomalyLabel(item),
      severity: _formatAnomalySeverity(item.severity),
      date: formatUiDateTime(item.observed_at, { day: '2-digit', month: '2-digit', year: 'numeric' }),
      metric: getFeatureLabel(item.metric || 'ndvi', { expertMode: store.expertMode }),
      reason: _formatAnomalyReason(item),
    })))
  const seasonalMetricSeries = computed(() => {
    if (!selectedSeasonalMetric.value?.points?.length) return []
    return [{
      label: selectedSeasonalMetric.value.label || getFeatureLabel(selectedSeasonalMetric.value.metric, { expertMode: store.expertMode }),
      color: resolveMetricColor(selectedSeasonalMetric.value.metric),
      points: selectedSeasonalMetric.value.points.map(p => ({ date: p.observed_at, value: p.value })),
    }]
  })
  const selectedSeasonalMetricLabel = computed(() =>
    selectedSeasonalMetric.value?.label || getFeatureLabel(selectedSeasonalMetricId.value, { expertMode: store.expertMode }) || '—')
  const seasonalMetricPointCountLabel = computed(() => {
    const count = selectedSeasonalMetric.value?.points?.length || 0
    if (!count) return t('field.noSeries')
    return locale.value === 'ru'
      ? `${count} ${count === 1 ? 'точка' : count < 5 ? 'точки' : 'точек'}`
      : `${count} point${count === 1 ? '' : 's'}`
  })
  const seasonalMetricTimeline = computed(() =>
    (selectedSeasonalMetric.value?.points || []).map(p => ({
      observed_at: p.observed_at,
      value: formatMetricValue(selectedSeasonalMetric.value?.metric, p.value),
    })))

  // Temporal status messages
  const temporalTaskFailureCode = computed(() => {
    const state = store.temporalAnalyticsTaskState || {}
    const status = String(state.status || state.state || '').toLowerCase()
    if (status !== 'failed') return ''
    return state.stage_detail_code || state.result?.data_status?.message_code || ''
  })
  const metricsDataStatusMessage = computed(() => _resolveTemporalStatusMessage(metricsDataStatus.value, { failureCode: temporalTaskFailureCode.value }))
  const forecastDataStatusMessage = computed(() => _resolveTemporalStatusMessage(forecastDataStatus.value, { failureCode: temporalTaskFailureCode.value }))
  const seasonalMetricChartEmptyText = computed(() => {
    if (metricsDataStatusMessage.value) return metricsDataStatusMessage.value
    const pointCount = selectedSeasonalMetric.value?.points?.length || 0
    if (pointCount > 0 && pointCount < 2) return t('field.temporalSinglePoint')
    return t('field.noSeries')
  })

  // Progress indicators
  const predictionProgressActive = computed(() => store.predictionTaskProgress > 0 && store.predictionTaskProgress < 100)
  const scenarioProgressActive = computed(() => store.scenarioTaskProgress > 0 && store.scenarioTaskProgress < 100)
  const temporalProgressActive = computed(() => store.temporalAnalyticsTaskProgress > 0 && store.temporalAnalyticsTaskProgress < 100)
  const predictionProgressLabel = computed(() => formatUiProgress(store.predictionTaskProgress))
  const scenarioProgressLabel = computed(() => formatUiProgress(store.scenarioTaskProgress))
  const temporalProgressLabel = computed(() => formatUiProgress(store.temporalAnalyticsTaskProgress))
  const predictionTaskStage = computed(() => getTaskStageLabel(store.predictionTaskState, store.predictionTaskState?.stage_label || 'running'))
  const predictionTaskDetail = computed(() => getTaskStageDetail(store.predictionTaskState) || '')
  const predictionTaskTiming = computed(() => formatTaskTiming(store.predictionTaskState, locale.value))
  const scenarioTaskStage = computed(() => getTaskStageLabel(store.scenarioTaskState, store.scenarioTaskState?.stage_label || 'running'))
  const scenarioTaskDetail = computed(() => getTaskStageDetail(store.scenarioTaskState) || '')
  const scenarioTaskTiming = computed(() => formatTaskTiming(store.scenarioTaskState, locale.value))
  const temporalTaskStage = computed(() => getTaskStageLabel(store.temporalAnalyticsTaskState, store.temporalAnalyticsTaskState?.stage_label || 'running'))
  const temporalTaskDetail = computed(() => getTaskStageDetail(store.temporalAnalyticsTaskState) || '')
  const temporalTaskTiming = computed(() => formatTaskTiming(store.temporalAnalyticsTaskState, locale.value))
  const predictionTaskLogs = computed(() => store.predictionTaskState?.logs || [])
  const scenarioTaskLogs = computed(() => store.scenarioTaskState?.logs || [])

  // Prediction labels
  const yieldLabel = computed(() => formatYield(prediction.value?.estimated_yield_kg_ha))
  const confidenceLabel = computed(() => formatPercent(prediction.value?.confidence))
  const predictionDateLabel = computed(() => formatDateTime(prediction.value?.prediction_date))
  const predictionConfidenceTierLabel = computed(() => {
    if (!prediction.value?.confidence_tier) return ''
    return `${locale.value === 'ru' ? 'Контур доверия' : 'Confidence tier'}: ${getConfidenceTierLabel(prediction.value.confidence_tier)}`
  })
  const predictionOperationalTierLabel = computed(() => {
    const tier = prediction.value?.operational_tier
    return tier ? t(`field.operationalTier.${tier}`) : '—'
  })
  const predictionReviewRequiredLabel = computed(() => prediction.value?.review_required ? t('field.yes') : t('field.no'))
  const predictionReviewReason = computed(() => formatReasonText(prediction.value?.review_reason_code, prediction.value?.review_reason, prediction.value?.review_reason_params) || '')
  const predictionSupportReason = computed(() => formatReasonText(prediction.value?.support_reason_code, prediction.value?.support_reason, prediction.value?.support_reason_params) || '')
  const predictionDrivers = computed(() => {
    const drivers = prediction.value?.driver_breakdown || prediction.value?.explanation?.drivers || []
    return drivers.map(d => ({
      label: getFeatureLabel(d.input_key || d.driver_id || d.factor || d.label, { expertMode: store.expertMode }) || d.label,
      effect: d.effect_kg_ha !== null && d.effect_kg_ha !== undefined
        ? `${Number(d.effect_kg_ha) >= 0 ? '+' : ''}${Number(d.effect_kg_ha).toFixed(0)} кг/га`
        : formatSigned(d.effect_pct),
    }))
  })
  const featureEntries = computed(() => {
    const features = prediction.value?.input_features || {}
    return Object.entries(features)
      .filter(([key, value]) => {
        if (store.expertMode) return true
        if (String(key || '').startsWith('_')) return false
        return value !== null && value !== undefined && value !== ''
      })
      .map(([key, value]) => ({ label: getFeatureLabel(key, { expertMode: store.expertMode }), value: formatFeatureValue(key, value, store.expertMode) }))
  })
  const qualityEntries = computed(() =>
    Object.entries(prediction.value?.data_quality || {}).map(([key, value]) => ({
      label: getFeatureLabel(key, { expertMode: store.expertMode }),
      value: formatFeatureValue(key, value, store.expertMode),
    })))
  const suitabilityEntries = computed(() => {
    const SKIP = new Set(['reasons', 'support_reason', 'yield_factor', 'warnings', 'recommendation'])
    const SUITABILITY_LABELS = { high: 'Высокая — оптимально для культуры', moderate: 'Умеренная — возможны ограничения', low: 'Низкая — требует интенсивной агротехники', unsuitable: 'Не подходит — риски критические' }
    const cs = prediction.value?.crop_suitability || {}
    return Object.entries(cs)
      .filter(([key, value]) => !SKIP.has(key) && value !== null && value !== undefined && !(Array.isArray(value) && !value.length))
      .map(([key, value]) => ({ label: getFeatureLabel(key, { expertMode: store.expertMode }), value: key === 'status' ? SUITABILITY_LABELS[value] || String(value) : formatFeatureValue(key, value, store.expertMode) }))
  })

  // Scenario/modeling
  const modelingBaselineLabel = computed(() => formatYield(store.modelingResult?.baseline_yield_kg_ha))
  const modelingScenarioLabel = computed(() => formatYield(store.modelingResult?.scenario_yield_kg_ha))
  const modelingDeltaLabel = computed(() => formatDelta(store.modelingResult?.predicted_yield_change_pct))
  const modelingRiskLevel = computed(() => getRiskLevelLabel(store.modelingResult?.risk_summary?.level_code || store.modelingResult?.risk_summary?.level || ''))
  const modelingRiskComment = computed(() => store.modelingResult?.risk_summary?.comment || '')
  const assumptionEntries = computed(() =>
    Object.entries(store.modelingResult?.assumptions || {}).map(([key, value]) => ({ label: getFeatureLabel(key, { expertMode: store.expertMode }), value: formatFeatureValue(key, value, store.expertMode) })))
  const scenarioOperationalTierLabel = computed(() => {
    const tier = store.modelingResult?.operational_tier
    return tier ? t(`field.operationalTier.${tier}`) : '—'
  })
  const scenarioReviewRequiredLabel = computed(() => store.modelingResult?.review_required ? t('field.yes') : t('field.no'))
  const scenarioReviewReason = computed(() => formatReasonText(store.modelingResult?.review_reason_code, store.modelingResult?.review_reason, store.modelingResult?.review_reason_params) || '')
  const riskLevelClass = computed(() => {
    const level = String(store.modelingResult?.risk_summary?.level_code || store.modelingResult?.risk_summary?.level || '').toLowerCase()
    if (level === 'low' || level === 'низкий') return 'risk-low'
    if (level === 'moderate' || level === 'умеренный') return 'risk-moderate'
    if (level === 'elevated' || level === 'повышенный') return 'risk-elevated'
    if (level === 'high' || level === 'высокий') return 'risk-high'
    return ''
  })
  const factorBreakdown = computed(() => {
    const RAW_INPUT_KEYS = new Set(['irrigation_pct', 'fertilizer_pct', 'expected_rain_mm', 'temperature_delta_c', 'planting_density_pct', 'cloud_cover_factor'])
    const all = store.modelingResult?.comparison?.factor_breakdown || store.modelingResult?.driver_breakdown || []
    return all.filter(f => !RAW_INPUT_KEYS.has(f.input_key || f.driver_id || f.factor || ''))
  })
  const scenarioWarnings = computed(() =>
    (store.modelingResult?.constraint_warnings || []).map(warning => {
      if (warning === store.modelingResult?.support_reason) {
        return formatReasonText(store.modelingResult?.support_reason_code, warning, store.modelingResult?.support_reason_params) || warning
      }
      return warning
    }))
  const comparisonChart = computed(() => {
    const baseline = Number(store.modelingResult?.baseline_yield_kg_ha)
    const scenario = Number(store.modelingResult?.scenario_yield_kg_ha)
    if (!Number.isFinite(baseline) || !Number.isFinite(scenario)) return null
    const interval = prediction.value?.prediction_interval || {}
    const lower = Number(interval.lower)
    const upper = Number(interval.upper)
    const values = [baseline, scenario, ...(Number.isFinite(lower) ? [lower] : []), ...(Number.isFinite(upper) ? [upper] : [])]
    const min = Math.min(...values)
    const span = Math.max(Math.max(...values) - min, 1)
    const toPct = v => ((v - min) / span) * 100
    return {
      baselinePct: toPct(baseline),
      scenarioPct: toPct(scenario),
      intervalStartPct: Number.isFinite(lower) ? toPct(lower) : 0,
      intervalWidth: Number.isFinite(lower) && Number.isFinite(upper) ? Math.max(0, toPct(upper) - toPct(lower)) : 0,
      intervalLabel: Number.isFinite(lower) && Number.isFinite(upper) ? `${formatYield(lower)} … ${formatYield(upper)}` : '',
    }
  })

  // Charts — GDD, water balance, history, forecast curves
  const gddCumulativeSeries = computed(() => {
    const cumMetric = (temporalAnalytics.value?.seasonal_series?.metrics || []).find(item => item.metric === 'gdd_cumulative')
    if (!cumMetric?.points?.length) return []
    return [{
      label: locale.value === 'ru' ? 'ГСТ накоп., °C·день' : 'Cumul. GDD',
      color: '#c98b24',
      points: cumMetric.points.map(p => ({ date: p.observed_at, value: p.value })),
    }]
  })
  const gddCumulativeTotal = computed(() => {
    const series = gddCumulativeSeries.value
    if (!series.length || !series[0].points.length) return '—'
    return `${Math.round(series[0].points[series[0].points.length - 1].value)} °C·д`
  })
  const waterBalanceSeries = computed(() => {
    const rows = waterBalance.value?.series || []
    const series = []
    if (rows.length) {
      series.push({ label: t('field.rootZoneStorage'), color: '#2f8a63', points: rows.map(r => ({ date: r.observed_at, value: r.storage_mm })) })
      series.push({ label: t('field.rootZoneDeficit'), color: '#b24d2a', points: rows.map(r => ({ date: r.observed_at, value: r.deficit_mm })) })
    }
    return series
  })
  const historyTrend = computed(() => prediction.value?.history_trend || forecastTemporalAnalytics.value?.history_trend || temporalAnalytics.value?.history_trend || {})
  const historyTrendChartSeries = computed(() => {
    const points = historyTrend.value?.points || []
    const series = []
    const observed = points.map(p => ({ date: p.year, value: p.observed_yield_kg_ha }))
    const rolling = points.map(p => ({ date: p.year, value: p.rolling_mean_kg_ha }))
    if (observed.length) series.push({ label: t('field.historyObserved'), color: '#2f6b97', points: observed })
    if (rolling.length) series.push({ label: t('field.historyRollingMean'), color: '#8a6b18', points: rolling })
    return series
  })
  const historyTrendHasSeries = computed(() => historyTrendChartSeries.value.some(s => (s.points || []).length >= 2))
  const forecastTrendMarkers = computed(() => {
    if (!historyTrendHasSeries.value) return []
    const currentYear = new Date().getUTCFullYear()
    const markers = []
    if (prediction.value?.estimated_yield_kg_ha !== null && prediction.value?.estimated_yield_kg_ha !== undefined)
      markers.push({ label: t('field.currentForecastMarker'), date: currentYear, value: prediction.value.estimated_yield_kg_ha, color: '#3c9c64' })
    if (store.modelingResult?.scenario_yield_kg_ha !== null && store.modelingResult?.scenario_yield_kg_ha !== undefined)
      markers.push({ label: t('field.currentScenarioMarker'), date: currentYear, value: store.modelingResult.scenario_yield_kg_ha, color: '#c97e27' })
    return markers
  })
  const forecastTrendRanges = computed(() => {
    if (!historyTrendHasSeries.value) return []
    const { lower, upper } = prediction.value?.prediction_interval || {}
    if (!Number.isFinite(Number(lower)) || !Number.isFinite(Number(upper))) return []
    return [{ date: new Date().getUTCFullYear(), lower: Number(lower), upper: Number(upper), color: '#7b91c8' }]
  })

  // Forecast curve options (shared by predict + scenario tabs)
  const forecastCurveMetric = ref('gdd_cumulative')
  const scenarioForecastCurveMetric = ref('gdd_cumulative')
  const scenarioComparisonMetric = ref('ndvi')
  const forecastCurveMetricOptions = computed(() => [
    { value: 'temperature_mean_c', label: t('field.temperatureMean') },
    { value: 'precipitation_mm', label: t('field.precipitationMetric') },
    { value: 'gdd_daily', label: t('field.gddDaily') },
    { value: 'gdd_cumulative', label: t('field.gddCumulative') },
  ])
  const predictionForecastCurveSeries = computed(() => {
    const metricId = forecastCurveMetric.value
    const points = mapForecastCurvePoints(prediction.value?.forecast_curve?.points || [], metricId)
    if (!points.length) return []
    return [{ label: forecastCurveMetricOptions.value.find(i => i.value === metricId)?.label || metricId, color: resolveMetricColor(metricId), points }]
  })
  const predictionForecastCurveEmptyText = computed(() => prediction.value?.forecast_curve?.error || t('field.futureForecastUnavailable'))

  // Archive
  const archiveView = computed(() => store.selectedArchiveView?.snapshot || null)
  const archivePrediction = computed(() => archiveView.value?.prediction_snapshot?.estimated_yield_kg_ha)
  const archiveConfidence = computed(() => archiveView.value?.prediction_snapshot?.confidence)
  const archiveMetricsLabel = computed(() => {
    const metrics = archiveView.value?.metrics_snapshot?.current_metrics || {}
    return Object.keys(metrics).length ? Object.keys(metrics).map(k => getLayerMeta(k).label).join(', ') : '—'
  })
  const archiveScenarioCount = computed(() => String((archiveView.value?.scenario_snapshot?.items || []).length))

  // Management zones
  const managementZoneRows = computed(() => managementZones.value?.zones || [])
  const managementZonesSupported = computed(() => Boolean(managementZoneSummary.value?.supported))
  const managementZoneModeLabel = computed(() => {
    const mode = managementZoneSummary.value?.mode
    if (mode === 'yield') return t('field.zoneModeYield')
    if (mode === 'yield_potential') return t('field.zoneModePotential')
    return '—'
  })

  // Phenology
  const phenologySummaryEntries = computed(() => {
    const ph = forecastTemporalAnalytics.value?.phenology || prediction.value?.phenology || {}
    const lagValue = ph.lag_weeks_vs_norm === null || ph.lag_weeks_vs_norm === undefined
      ? '—'
      : locale.value === 'ru' ? `${Number(ph.lag_weeks_vs_norm).toFixed(1)} нед.` : `${Number(ph.lag_weeks_vs_norm).toFixed(1)} wk`
    return [
      { label: t('field.currentStage'), value: ph.stage_label || '—' },
      { label: t('field.stageLag'), value: lagValue },
      { label: t('field.peakDate'), value: ph.peak_date || '—' },
      { label: t('field.seasonAmplitude'), value: ph.seasonal_amplitude === null || ph.seasonal_amplitude === undefined ? '—' : Number(ph.seasonal_amplitude).toFixed(3) },
    ].filter(e => e.value !== '—' || e.label === t('field.currentStage'))
  })

  // Geometry diag
  const geometryDiagnosticsEntries = computed(() => {
    const field = dashboard.value?.field || store.selectedField || {}
    const runtime = debugRuntime.value || {}
    return [
      { label: t('field.geometryConfidence'), value: formatPercent(field.geometry_confidence) },
      { label: t('field.ttaConsensus'), value: formatPercent(field.tta_consensus) },
      { label: t('field.boundaryUncertainty'), value: formatPercent(field.boundary_uncertainty) },
      { label: t('field.componentsAfterGrow'), value: formatInteger(runtime.components_after_grow) },
      { label: t('field.componentsAfterGapClose'), value: formatInteger(runtime.components_after_gap_close) },
      { label: t('field.componentsAfterInfill'), value: formatInteger(runtime.components_after_infill) },
      { label: t('field.componentsAfterMerge'), value: formatInteger(runtime.components_after_merge) },
      { label: t('field.componentsAfterWatershed'), value: formatInteger(runtime.components_after_watershed) },
      { label: t('field.splitScoreP50'), value: formatDecimal(runtime.split_score_p50) },
      { label: t('field.splitScoreP90'), value: formatDecimal(runtime.split_score_p90) },
    ]
  })
  const geometryDebugStatus = computed(() => {
    const runtime = debugRuntime.value || {}
    if (!Object.keys(runtime).length) return ''
    if (runtime.watershed_rollback_reason) return `${t('field.watershedRollback')}: ${runtime.watershed_rollback_reason}`
    if (runtime.watershed_applied) return t('field.watershedApplied')
    if (runtime.watershed_skipped_reason) return `${t('field.watershedSkipped')}: ${runtime.watershed_skipped_reason}`
    return ''
  })
  const modelFoundationEntries = computed(() => [
    { label: t('field.modelHeads'), value: Array.isArray(analyticsSummary.value?.heads) ? analyticsSummary.value.heads.join(', ') : 'extent, boundary, distance' },
    { label: t('field.modelHeadCount'), value: String(analyticsSummary.value?.head_count || 3) },
    { label: t('field.ttaStandard'), value: analyticsSummary.value?.tta_standard || 'flip2' },
    { label: t('field.ttaQuality'), value: analyticsSummary.value?.tta_quality || 'rotate4' },
  ])
  const retrainDescription = computed(() => analyticsSummary.value?.retrain_description || t('field.retrainDescription'))
  const analyticsAlertEntries = computed(() => metricsAnomalyItems.value.slice(0, 6))

  // Sensitivity analysis
  const selectedSweepParam = ref('fertilizer_pct')
  const sensitivityParamLabel = computed(() => {
    const labels = {
      irrigation_pct: t('field.irrigation'),
      fertilizer_pct: t('field.fertilizer'),
      expected_rain_mm: t('field.expectedRain'),
      temperature_delta_c: t('field.temperatureDelta'),
    }
    return labels[selectedSweepParam.value] || ''
  })
  const currentSweepParamValue = computed(() => Number((store.modelingResult?.factors || {})[selectedSweepParam.value]) || 0)

  // Auto-fill badges
  const expectedRainAutoBadge = computed(() => {
    const source = store.modelingAutoSources?.expected_rain_mm || ''
    if (source === 'satellite_wetness_bsi' || source === 'satellite_wetness') return '🛰'
    if (source === 'forecast_curve') return '☁'
    if (source === 'observed_weather') return '☔'
    return '🛰'
  })
  const expectedRainAutoSourceTitle = computed(() => {
    const source = store.modelingAutoSources?.expected_rain_mm || ''
    if (source === 'satellite_wetness_bsi') return t('field.expectedRainFromSatelliteBsi')
    if (source === 'satellite_wetness') return t('field.expectedRainFromSatelliteWetness')
    if (source === 'forecast_curve') return t('field.expectedRainFromForecast')
    if (source === 'observed_weather') return t('field.expectedRainFromObservedWeather')
    return t('field.expectedRainAutoHint')
  })
  const soilCompactionAutoSourceTitle = computed(() => {
    const source = store.modelingAutoSources?.soil_compaction || ''
    if (source === 'satellite_soil_moisture_bsi') return t('field.soilCompactionFromSatelliteBsi')
    if (source === 'satellite_soil_moisture') return t('field.soilCompactionFromSatelliteMoisture')
    return t('field.soilCompactionAutoHint')
  })

  // Events management
  const currentYear = new Date().getFullYear()
  const eventForm = ref({ event_type: '', event_date: '', season_year: currentYear, amount: null, unit: '' })
  const editingEventId = ref(null)
  const eventFormError = ref('')
  const availableSeasonYears = computed(() => {
    const years = new Set(store.fieldEvents.map(e => e.season_year).filter(Boolean))
    for (let y = currentYear; y >= currentYear - 4; y--) years.add(y)
    return [...years].sort((a, b) => b - a)
  })
  function formatEventType(type) {
    if (!type) return '—'
    const labels = t('field.eventsTypeLabels') || {}
    return labels[type.toLowerCase()] || type
  }
  function formatEventDate(dateStr) {
    if (!dateStr) return '—'
    try {
      return new Date(dateStr).toLocaleDateString(locale.value === 'ru' ? 'ru-RU' : 'en-US', { day: '2-digit', month: '2-digit', year: 'numeric' })
    } catch { return dateStr }
  }
  function resetEventForm() {
    eventForm.value = { event_type: '', event_date: '', season_year: currentYear, amount: null, unit: '' }
    editingEventId.value = null
    eventFormError.value = ''
  }
  function openEditEvent(event) {
    editingEventId.value = event.id
    eventForm.value = { event_type: event.event_type || '', event_date: event.event_date ? String(event.event_date).slice(0, 10) : '', season_year: event.season_year || currentYear, amount: event.amount ?? null, unit: event.unit || '' }
    eventFormError.value = ''
  }
  function cancelEditEvent() { resetEventForm() }
  async function submitEventForm() {
    eventFormError.value = ''
    if (!eventForm.value.event_type.trim()) { eventFormError.value = t('field.eventsType') + ': обязательное поле'; return }
    if (!eventForm.value.event_date) { eventFormError.value = t('field.eventsDate') + ': обязательное поле'; return }
    const fieldId = store.selectedField?.field_id
    if (!fieldId) return
    const payload = {
      event_type: eventForm.value.event_type.trim(),
      event_date: new Date(eventForm.value.event_date).toISOString(),
      season_year: Number(eventForm.value.season_year),
      amount: eventForm.value.amount !== null && eventForm.value.amount !== '' ? Number(eventForm.value.amount) : null,
      unit: eventForm.value.unit.trim() || null,
      payload: {},
    }
    try {
      if (editingEventId.value) await store.updateFieldEvent(fieldId, editingEventId.value, payload)
      else await store.createFieldEvent(fieldId, payload)
      resetEventForm()
    } catch (err) { eventFormError.value = err?.response?.data?.detail || String(err) }
  }
  async function handleDeleteEvent(eventId) {
    if (!window.confirm(t('field.eventsDeleteConfirm'))) return
    const fieldId = store.selectedField?.field_id
    if (!fieldId) return
    try { await store.deleteFieldEvent(fieldId, eventId) }
    catch (err) { eventFormError.value = err?.response?.data?.detail || String(err) }
  }

  // ── Private helpers ─────────────────────────────────────────────────────────
  function _formatAnomalySeverity(value) {
    const n = String(value || '').toLowerCase()
    if (n === 'critical') return locale.value === 'ru' ? 'Критично' : 'Critical'
    if (n === 'warning') return locale.value === 'ru' ? 'Предупреждение' : 'Warning'
    if (n === 'info') return locale.value === 'ru' ? 'Инфо' : 'Info'
    return value || '—'
  }
  const ANOMALY_LABELS = {
    rapid_canopy_loss: { ru: 'Быстрая потеря зелёной массы', en: 'Rapid canopy loss' },
    possible_drought_stress: { ru: 'Возможный стресс засухи', en: 'Possible drought stress' },
    possible_waterlogging: { ru: 'Возможное переувлажнение', en: 'Possible waterlogging' },
    possible_disease: { ru: 'Возможное заболевание или вредитель', en: 'Possible disease or pest pressure' },
    delayed_development: { ru: 'Смещение развития', en: 'Delayed development' },
  }
  const ANOMALY_REASONS = {
    rapid_canopy_loss: { ru: 'NDVI падает быстрее ожидаемой сезонной динамики.', en: 'NDVI is dropping faster than the expected seasonal pattern.' },
    possible_drought_stress: { ru: 'Падение NDVI сопровождается сухим сигналом NDMI и похоже на дефицит влаги.', en: 'The NDVI drop is accompanied by a dry NDMI signal and looks like moisture stress.' },
    possible_waterlogging: { ru: 'Сигнал похож на переувлажнение и кислородный стресс в корневой зоне.', en: 'The signal is consistent with waterlogging and root-zone oxygen stress.' },
    possible_disease: { ru: 'Сигнал отклоняется от нормы и может указывать на болезнь или вредителя.', en: 'The signal departs from the norm and may indicate disease or pest pressure.' },
    delayed_development: { ru: 'Развитие культуры отстаёт от ожидаемой фенологической нормы.', en: 'Crop development is lagging behind the expected phenological norm.' },
  }
  function _formatAnomalyLabel(item) {
    const lang = locale.value === 'en' ? 'en' : 'ru'
    return ANOMALY_LABELS[item?.kind]?.[lang] || item?.label || '—'
  }
  function _formatAnomalyReason(item) {
    const lang = locale.value === 'en' ? 'en' : 'ru'
    return ANOMALY_REASONS[item?.kind]?.[lang] || item?.reason || '—'
  }
  function _resolveTemporalStatusMessage(status, options = {}) {
    const failureCode = options.failureCode || ''
    const statusCode = status?.message_code || status?.code || ''
    const messageCode = ['temporal_ready', 'ready'].includes(statusCode) ? statusCode : (failureCode || statusCode)
    if (!messageCode || messageCode === 'temporal_ready' || messageCode === 'ready') return ''
    const range = (() => {
      const r = status?.requested_range || status?.actual_range
      if (!r?.date_from && !r?.date_to) return ''
      const from = formatDateShort(r?.date_from)
      const to = formatDateShort(r?.date_to)
      return from && to ? `${from} - ${to}` : (from || to)
    })()
    const suffix = range ? (locale.value === 'ru' ? ` Диапазон: ${range}.` : ` Range: ${range}.`) : ''
    if (messageCode === 'temporal_backfill_required' || messageCode === 'backfill_required') return `${t('field.temporalBackfillRequired')}${suffix}`
    if (messageCode === 'temporal_range_exceeds_limit' || messageCode === 'range_exceeds_limit') return t('field.temporalRangeExceedsLimit')
    if (messageCode === 'temporal_historical_data_sparse' || messageCode === 'historical_data_sparse') return `${t('field.temporalHistoricalDataSparse')}${suffix}`
    if (messageCode === 'temporal_insufficient_points_current_season' || messageCode === 'insufficient_points_current_season') return t('field.temporalInsufficientPointsCurrentSeason')
    if (messageCode === 'temporal_no_history_available' || messageCode === 'no_history_available') return `${t('field.temporalNoHistoryAvailable')}${suffix}`
    if (messageCode === 'source_unavailable_quota') return t('field.temporalSourceUnavailableQuota')
    if (messageCode === 'backfill_delayed') return t('field.temporalBackfillDelayed')
    return status?.message || status?.detail || ''
  }

  return {
    // State
    store, isGroup, dashboard, isLoadingDashboard, isPreviewField, prediction,
    temporalAnalytics, forecastTemporalAnalytics, managementZones, analyticsSummary,
    waterBalance, riskSummary, managementZoneSummary, debugRuntime,
    // Tabs
    visibleTabs,
    // Date range
    onDateRangeChange, clearDateRange, onDateInput,
    // Field labels
    sourceLabel, qualityLabel, qualityBandLabel, qualityReasonLabel,
    fieldOperationalTierLabel, fieldReviewRequiredLabel, fieldReviewReason,
    totalAreaLabel, perimeterLabel, observationCellsLabel, availableMetricsLabel,
    dataQualityText, overviewCards,
    // Metrics
    metricCards, seriesEntries, histogramEntries,
    seasonalMetricEntries, selectedSeasonalMetricId, selectedSeasonalMetric,
    seasonalMetricSelectorOptions, metricsAnomalyItems, forecastAnomalyItems,
    selectedMetricAnomalies, seasonalAnomalyRows, seasonalMetricSeries,
    selectedSeasonalMetricLabel, seasonalMetricPointCountLabel, seasonalMetricTimeline,
    // Status messages
    metricsDataStatusMessage, forecastDataStatusMessage, seasonalMetricChartEmptyText,
    temporalTaskFailureCode,
    // Progress
    predictionProgressActive, scenarioProgressActive, temporalProgressActive,
    predictionProgressLabel, scenarioProgressLabel, temporalProgressLabel,
    predictionTaskStage, predictionTaskDetail, predictionTaskTiming,
    scenarioTaskStage, scenarioTaskDetail, scenarioTaskTiming,
    temporalTaskStage, temporalTaskDetail, temporalTaskTiming,
    predictionTaskLogs, scenarioTaskLogs,
    // Prediction
    yieldLabel, confidenceLabel, predictionDateLabel, predictionConfidenceTierLabel,
    predictionOperationalTierLabel, predictionReviewRequiredLabel, predictionReviewReason,
    predictionSupportReason, predictionDrivers, featureEntries, qualityEntries, suitabilityEntries,
    // Charts
    gddCumulativeSeries, gddCumulativeTotal, waterBalanceSeries,
    historyTrend, historyTrendChartSeries, forecastTrendMarkers, forecastTrendRanges,
    forecastCurveMetric, scenarioForecastCurveMetric, scenarioComparisonMetric, forecastCurveMetricOptions,
    predictionForecastCurveSeries, predictionForecastCurveEmptyText,
    // Scenario
    modelingBaselineLabel, modelingScenarioLabel, modelingDeltaLabel,
    modelingRiskLevel, modelingRiskComment, assumptionEntries,
    scenarioOperationalTierLabel, scenarioReviewRequiredLabel, scenarioReviewReason,
    riskLevelClass, factorBreakdown, scenarioWarnings, comparisonChart,
    // Sensitivity
    selectedSweepParam, sensitivityParamLabel, currentSweepParamValue,
    expectedRainAutoBadge, expectedRainAutoSourceTitle, soilCompactionAutoSourceTitle,
    // Archive
    archiveView, archivePrediction, archiveConfidence, archiveMetricsLabel, archiveScenarioCount,
    // Zones & Phenology & Geometry
    managementZoneRows, managementZonesSupported, managementZoneModeLabel,
    phenologySummaryEntries, analyticsAlertEntries,
    geometryDiagnosticsEntries, geometryDebugStatus, modelFoundationEntries, retrainDescription,
    // Events
    eventForm, editingEventId, eventFormError, availableSeasonYears,
    formatEventType, formatEventDate, openEditEvent, cancelEditEvent, submitEventForm, handleDeleteEvent,
    // Helper functions used in template
    riskLevelLabel: getRiskLevelLabel,
    scenarioRiskItemLabel: getRiskItemLabel,
    scenarioRiskItemReason: getRiskItemReason,
  }
}
