<template>
  <section class="window-shell control-panel">
    <div class="window-title">{{ t('control.title') }}</div>
    <div class="window-body control-body">
      <div class="section-block">
        <div class="section-caption">{{ t('control.zone') }}</div>
        <button class="btn-secondary" :class="{ active: store.isPickingSearchCenter }" :title="helpHints.pickCenter" data-btip="Кликните эту кнопку, затем кликните на карте, чтобы задать центр области поиска полей." @click="store.toggleSearchCenterPicking()">
          {{ store.isPickingSearchCenter ? t('control.cancelPick') : t('control.pickCenter') }}
        </button>
        <div class="input-grid two-columns">
          <label>
            <span>{{ t('control.lat') }}</span>
            <input type="number" v-model.number="store.centerLat" step="0.001" min="-90" max="90" data-btip="Широта центра области поиска (от -90 до 90). Можно вбить вручную или выбрать на карте." />
          </label>
          <label>
            <span>{{ t('control.lon') }}</span>
            <input type="number" v-model.number="store.centerLon" step="0.001" min="-180" max="180" data-btip="Долгота центра области поиска (от -180 до 180). Можно вбить вручную или выбрать на карте." />
          </label>
        </div>
        <label>
          <span>{{ t('control.radius') }}</span>
          <input
            type="number"
            v-model.number="store.radiusKm"
            step="1"
            min="1"
            :max="store.activeDetectionPreset.maxRadiusKm"
            :data-btip="radiusHelpText"
            data-testid="radius-km-input"
          />
        </label>
      </div>

      <div class="section-block">
        <div class="section-caption">{{ t('control.period') }}</div>
        <div class="input-grid two-columns">
          <label>
            <span>{{ t('control.start') }}</span>
            <input type="date" v-model="store.startDate" data-btip="Начало периода наблюдений. Система будет искать спутниковые снимки начиная с этой даты." />
          </label>
          <label>
            <span>{{ t('control.end') }}</span>
            <input type="date" v-model="store.endDate" data-btip="Конец периода наблюдений. Чем длиннее период — тем больше снимков и точнее результат, но дольше обработка." />
          </label>
        </div>
      </div>

      <div class="section-block">
        <div class="section-caption">{{ t('control.algorithm') }}</div>
        <div class="preset-strip">
          <span>{{ locale === 'ru' ? 'Профиль' : 'Preset' }}</span>
          <strong>{{ store.activeDetectionPreset.label }}</strong>
        </div>
        <div class="preset-description">{{ store.activeDetectionPreset.description }}</div>
        <div v-if="!store.expertMode" class="action-note">
          {{ t('control.presetLocked') }}
        </div>
        <div class="input-grid two-columns compact-grid">
          <label>
            <span>{{ t('control.clouds') }}</span>
            <input type="number" v-model.number="store.maxCloudPct" step="5" min="0" max="100" data-btip="Максимальный процент облачности снимка. Снимки с большей облачностью будут отброшены. Рекомендуется 20–40%." />
          </label>
          <label>
            <span>{{ t('control.dates') }}</span>
            <input type="number" v-model.number="store.targetDates" step="1" min="1" max="12" :disabled="!store.expertMode" />
          </label>
          <label>
            <span>{{ t('control.minArea') }}</span>
            <input type="number" v-model.number="store.minFieldAreaHa" step="0.1" min="0.1" max="10" :disabled="!store.expertMode" />
          </label>
          <label>
            <span>{{ t('control.resolution') }}</span>
            <input type="number" v-model.number="store.resolutionM" step="10" min="10" max="60" :disabled="!store.expertMode" />
          </label>
        </div>
        <label class="toggle-line">
          <input type="checkbox" v-model="store.useSam" :disabled="!store.expertMode" />
          <span>{{ t('control.samToggle') }}</span>
        </label>
        <div v-if="store.lastPreflight" class="info-card">
          <div class="info-card-title">{{ t('control.launchTier') }}</div>
          <div class="info-line">
            <span>{{ t('control.launchTier') }}</span>
            <strong>{{ launchTierLabel }}</strong>
          </div>
          <div class="info-line">
            <span>{{ t('control.reviewNeeded') }}</span>
            <strong>{{ store.lastPreflight.review_required ? t('field.yes') : t('field.no') }}</strong>
          </div>
          <div v-if="store.lastPreflight.review_reason" class="info-line">
            <span>{{ t('control.reviewReason') }}</span>
            <strong>{{ preflightReviewReason }}</strong>
          </div>
        </div>
      </div>

      <div class="section-block">
        <div class="section-caption">{{ t('control.actions') }}</div>
        <div class="button-stack">
          <button class="btn-primary" data-testid="start-detection" :title="helpHints.detect" :disabled="!canStartDetection" data-btip="Запускает автодетекцию полей по заданным параметрам. Система скачивает спутниковые снимки, строит признаки, запускает нейросеть и векторизует контуры полей." @click="store.startDetection">
            {{ store.isDetecting ? t('control.detecting') : t('control.startDetect') }}
          </button>
          <div v-if="validationErrors.length" class="validation-errors">
            <div v-for="(err, i) in validationErrors" :key="i" class="validation-line">{{ err }}</div>
          </div>
          <button class="btn-secondary" data-testid="refresh-all" data-btip="Обновляет погоду, статус системы, прогноз и другие данные без повторного запуска детекции полей." @click="store.refreshAll">
            {{ t('control.refreshAll') }}
          </button>
          <button class="btn-secondary" data-testid="refresh-weather" :disabled="store.isLoadingWeather" @click="store.loadWeather({ manual: true })">
            {{ t('control.syncWeather') }}
          </button>
          <button class="btn-secondary" data-testid="refresh-status" @click="store.loadSystemStatus({ manual: true })">
            {{ t('control.syncStatus') }}
          </button>
          <div class="action-note" :class="weatherSyncClass">
            {{ t('control.weatherUpdated') }}: {{ weatherUpdatedLabel }}<span v-if="weatherSyncLabel"> · {{ weatherSyncLabel }}</span>
          </div>
          <div class="action-note" :class="statusSyncClass">
            {{ t('control.statusUpdated') }}: {{ statusUpdatedLabel }}<span v-if="statusSyncLabel"> · {{ statusSyncLabel }}</span>
          </div>
          <button class="btn-secondary" :class="{ active: store.drawMode }" @click="store.drawMode = !store.drawMode">
            {{ store.drawMode ? t('control.cancelDraw') : t('control.drawField') }}
          </button>
        </div>
      </div>

      <div v-if="store.expertMode && (preflightDiagnostics.length || runtimeDiagnostics.length)" class="section-block">
        <div class="section-caption">{{ t('control.diagnostics') }}</div>
        <div v-if="preflightDiagnostics.length" class="info-card">
          <div class="info-card-title">{{ t('control.preflight') }}</div>
          <div v-for="item in preflightDiagnostics" :key="`pre-${item.label}`" class="info-line">
            <span>{{ item.label }}</span>
            <strong>{{ item.value }}</strong>
          </div>
        </div>
        <div v-if="runtimeDiagnostics.length" class="info-card">
          <div class="info-card-title">{{ t('control.runtime') }}</div>
          <div v-for="item in runtimeDiagnostics" :key="`run-${item.label}`" class="info-line">
            <span>{{ item.label }}</span>
            <strong>{{ item.value }}</strong>
          </div>
        </div>
      </div>

      <div class="section-block">
        <div class="section-caption">{{ t('control.layers') }}</div>
        <label class="toggle-line">
          <input type="checkbox" v-model="store.showSatelliteBrowse" />
          <span>{{ t('control.satelliteToggle') }}</span>
        </label>
        <div v-if="store.showSatelliteBrowse" class="input-grid two-columns compact-grid">
          <label>
            <span>{{ t('control.satelliteDate') }}</span>
            <input type="date" v-model="store.satelliteBrowseDate" />
          </label>
          <label>
            <span>{{ t('control.satelliteMode') }}</span>
            <input type="text" :value="t('control.satelliteModeValue')" readonly />
          </label>
        </div>
        <div v-if="store.showSatelliteBrowse" class="action-note">
          {{ satelliteSceneLabel }}
        </div>
        <div class="layer-group-label">{{ t('control.spectralGroup') }}</div>
        <div class="layer-list">
          <label v-for="layer in spectralLayers" :key="layer.id" class="layer-row" :data-btip="layer.btip">
            <input
              type="checkbox"
              :checked="store.activeLayers[layer.id]"
              @change="store.toggleLayer(layer.id)"
            />
            <span>{{ t(`layers.${layer.id}`) }}</span>
          </label>
        </div>
        <div class="layer-group-label">{{ t('control.weatherGroup') }}</div>
        <div class="layer-list">
          <label v-for="layer in weatherLayers" :key="layer.id" class="layer-row" :data-btip="layer.btip">
            <input
              type="checkbox"
              :checked="store.activeLayers[layer.id]"
              @change="store.toggleLayer(layer.id)"
            />
            <span>{{ t(`layers.${layer.id}`) }}</span>
          </label>
        </div>
        <div class="display-mode-switch">
          <button
            class="mode-btn"
            :class="{ active: store.showFieldsOnly }"
            data-btip="Показывать спектральные слои (NDVI, NDMI и т.д.) только на распознанных полях."
            @click="store.showFieldsOnly = true"
          >
            {{ t('control.onFields') }}
          </button>
          <button
            class="mode-btn"
            :class="{ active: !store.showFieldsOnly }"
            data-btip="Показывать спектральные слои на всей видимой карте, включая области вне полей."
            @click="store.showFieldsOnly = false"
          >
            {{ t('control.fullMap') }}
          </button>
        </div>
      </div>

      <div v-if="store.error" class="error-box">
        {{ store.error }}
      </div>
    </div>
  </section>
</template>

<script setup>
import { computed } from 'vue'
import { useMapStore } from '../store/map'
import { locale, t } from '../utils/i18n'
import { formatReasonText, formatUiDateTime, getTaskStageLabel } from '../utils/presentation'

const store = useMapStore()
const helpHints = {
  pickCenter: 'Укажите центр AOI прямо на карте, чтобы не вводить координаты вручную.',
  detect: 'Запускает поиск полей по спутниковым данным за выбранный период.',
}

const radiusHelpText = computed(() => {
  if (locale.value === 'ru') {
    return `Радиус области поиска в километрах. Для профиля ${store.activeDetectionPreset.label.toLowerCase()} рекомендовано до ${store.activeDetectionPreset.recommendedRadiusKm} км, жёсткий лимит — ${store.activeDetectionPreset.maxRadiusKm} км.`
  }
  return `Search radius in kilometers. For ${store.activeDetectionPreset.label.toLowerCase()} the recommended radius is up to ${store.activeDetectionPreset.recommendedRadiusKm} km, hard cap ${store.activeDetectionPreset.maxRadiusKm} km.`
})

const validationErrors = computed(() => {
  const errors = []
  if (!Number.isFinite(store.centerLat) || store.centerLat < -90 || store.centerLat > 90) {
    errors.push(t('validation.latRange'))
  }
  if (!Number.isFinite(store.centerLon) || store.centerLon < -180 || store.centerLon > 180) {
    errors.push(t('validation.lonRange'))
  }
  if (!Number.isFinite(store.radiusKm) || store.radiusKm < 1 || store.radiusKm > store.activeDetectionPreset.maxRadiusKm) {
    errors.push(
      locale.value === 'ru'
        ? `Радиус должен быть от 1 до ${store.activeDetectionPreset.maxRadiusKm} км для профиля ${store.activeDetectionPreset.label.toLowerCase()}.`
        : `Radius must be between 1 and ${store.activeDetectionPreset.maxRadiusKm} km for the ${store.activeDetectionPreset.label} preset.`
    )
  }
  if (!store.startDate || !store.endDate) {
    errors.push(t('validation.datesRequired'))
  } else if (store.startDate >= store.endDate) {
    errors.push(t('validation.datesOrder'))
  }
  if (store.maxCloudPct < 0 || store.maxCloudPct > 100) {
    errors.push(t('validation.cloudRange'))
  }
  return errors
})

const canStartDetection = computed(() => {
  return !store.isDetecting && validationErrors.value.length === 0
})

const preflightDiagnostics = computed(() => {
  const payload = store.lastPreflight || {}
  if (!payload || !Object.keys(payload).length) {
    return []
  }
  const seasonWindow = payload.season_window || {}
  const enabledStages = Array.isArray(payload.enabled_stages)
    ? payload.enabled_stages.map((stage) => getTaskStageLabel(stage)).join(', ')
    : ''
  return [
    { label: t('control.diagPreset'), value: store.activeDetectionPreset.label },
    { label: t('control.launchTier'), value: launchTierLabel.value },
    { label: t('control.diagTiles'), value: payload.estimated_tiles ?? '—' },
    { label: t('control.diagRuntime'), value: payload.estimated_runtime_class || '—' },
    { label: t('control.diagPipeline'), value: payload.pipeline_profile || '—' },
    { label: t('control.diagPreview'), value: payload.preview_only ? t('field.yes') : t('field.no') },
    { label: locale.value === 'ru' ? 'Режим вывода' : 'Output mode', value: payload.output_mode || '—' },
    { label: locale.value === 'ru' ? 'Операционный режим' : 'Operational eligible', value: payload.operational_eligible ? t('field.yes') : t('field.no') },
    { label: t('control.diagStages'), value: enabledStages || '—' },
    { label: t('control.diagMemory'), value: payload.estimated_ram_mb ? `${payload.estimated_ram_mb} MB` : '—' },
    { label: locale.value === 'ru' ? 'Радиус профиля' : 'Preset radius', value: payload.max_radius_km ? `≤${payload.max_radius_km} км` : '—' },
    { label: t('control.diagRegion'), value: payload.regional_profile || '—' },
    { label: t('control.diagSeason'), value: seasonWindow.start && seasonWindow.end ? `${seasonWindow.start} → ${seasonWindow.end}` : '—' },
    { label: t('control.diagS1'), value: payload.s1_planned ? t('field.yes') : t('field.no') },
    { label: t('control.diagTta'), value: payload.tta_mode || 'none' },
  ]
})

const launchTierLabel = computed(() => {
  const value = store.lastPreflight?.launch_tier
  if (!value) return '—'
  return t(`field.operationalTier.${value}`)
})

const preflightReviewReason = computed(() => {
  return formatReasonText(
    store.lastPreflight?.review_reason_code,
    store.lastPreflight?.review_reason,
    store.lastPreflight?.review_reason_params,
  ) || '—'
})

const runtimeDiagnostics = computed(() => {
  const runtime = store.runRuntime || {}
  if (!runtime || !Object.keys(runtime).length) {
    return []
  }
  const enabledStages = Array.isArray(runtime.enabled_stages)
    ? runtime.enabled_stages.map((stage) => getTaskStageLabel(stage)).join(', ')
    : ''
  return [
    { label: t('control.diagProvider'), value: runtime.sentinel_account_used || 'primary' },
    { label: t('control.diagFailover'), value: runtime.sentinel_failover_level ?? 0 },
    { label: t('control.diagS1'), value: runtime.s1_planned ? t('field.yes') : t('field.no') },
    { label: t('control.diagTta'), value: runtime.tta_mode || 'none' },
    { label: t('control.diagPipeline'), value: runtime.pipeline_profile || '—' },
    { label: t('control.diagPreview'), value: runtime.preview_only ? t('field.yes') : t('field.no') },
    { label: locale.value === 'ru' ? 'Режим вывода' : 'Output mode', value: runtime.output_mode || '—' },
    { label: locale.value === 'ru' ? 'Операционный режим' : 'Operational eligible', value: runtime.operational_eligible ? t('field.yes') : t('field.no') },
    { label: t('control.diagStages'), value: enabledStages || '—' },
    { label: t('control.diagCloud'), value: runtime.selected_date_confidence_mean !== undefined && runtime.selected_date_confidence_mean !== null ? Number(runtime.selected_date_confidence_mean).toFixed(2) : '—' },
    { label: t('control.diagBridge'), value: runtime.bridge_skipped_reason || t('control.diagBridgeUsed') },
    { label: t('control.diagProfile'), value: runtime.region_boundary_profile || runtime.date_selection_region_band || '—' },
  ]
    .filter((item) => item.value !== undefined && item.value !== null && item.value !== '')
})

const weatherUpdatedLabel = computed(() => {
  return formatUiDateTime(store.lastWeatherUpdatedAt, {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    day: '2-digit',
    month: '2-digit',
  })
})

const weatherSyncLabel = computed(() => store.lastWeatherSyncDetail || '')
const weatherSyncClass = computed(() => ({
  'is-ok': store.lastWeatherSyncState === 'ok',
  'is-error': store.lastWeatherSyncState === 'error',
}))

const statusUpdatedLabel = computed(() => {
  return formatUiDateTime(store.lastStatusUpdatedAt, {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    day: '2-digit',
    month: '2-digit',
  })
})

const statusSyncLabel = computed(() => store.lastStatusSyncDetail || '')
const statusSyncClass = computed(() => ({
  'is-ok': store.lastStatusSyncState === 'ok',
  'is-error': store.lastStatusSyncState === 'error',
}))

const satelliteSceneLabel = computed(() => {
  const scene = store.satelliteScene
  if (!scene) {
    return t('control.satelliteScenePending')
  }
  if (scene.status === 'no_data') {
    return t('control.satelliteSceneNoData')
  }
  const dateToken = scene.resolved_date || scene.requested_date || scene.requested_window?.start || 'auto'
  const modeLabel = scene.status === 'fallback_window'
    ? (locale.value === 'ru' ? 'автоподбор по окну' : 'window fallback')
    : ''
  const cloud = scene.cloud_cover_pct === null || scene.cloud_cover_pct === undefined ? '—' : `${Number(scene.cloud_cover_pct).toFixed(0)}%`
  return locale.value === 'ru'
    ? `${dateToken}${modeLabel ? ` · ${modeLabel}` : ''} · облачность ${cloud} · ${scene.provider_account || 'primary'}`
    : `${dateToken}${modeLabel ? ` · ${modeLabel}` : ''} · cloud ${cloud} · ${scene.provider_account || 'primary'}`
})

const spectralLayers = [
  { id: 'ndvi', btip: 'Индекс зелёной вегетации (NDVI). Зелёный = густая растительность, красный = голая почва или стресс.' },
  { id: 'ndwi', btip: 'Водный индекс (NDWI). Синий = вода/переувлажнение, коричневый = суша.' },
  { id: 'ndmi', btip: 'Влажность листьев (NDMI). Синий = высокая влажность, красный = засуха/стресс.' },
  { id: 'bsi', btip: 'Индекс голой почвы (BSI). Яркий = обнажённая почва, тёмный = покрытая растительностью.' },
]
const weatherLayers = [
  { id: 'precipitation', btip: 'Осадки за период. Интенсивность синего соответствует количеству осадков.' },
  { id: 'wind', btip: 'Направление и сила ветра. Стрелки показывают направление, длина — скорость.' },
  { id: 'soil_moisture', btip: 'Влажность почвы. Тёмно-синий = насыщенный, светлый = сухой.' },
  { id: 'gdd', btip: 'Градусо-дни роста (GDD). Тепловой ресурс сезона для оценки стадии развития культур.' },
  { id: 'vpd', btip: 'Дефицит давления пара (VPD). Высокий VPD = потенциальный тепловой/водный стресс.' },
]
</script>

<style scoped>
.control-panel {
  min-height: 0;
}

.control-body {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.section-block {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.section-caption {
  font-weight: 700;
  color: var(--text-main);
}

.input-grid {
  display: grid;
  gap: 8px;
}

.two-columns {
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

label {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

label span {
  color: var(--text-muted);
}

.compact-grid label span {
  font-size: 12px;
}

.toggle-line {
  flex-direction: row;
  align-items: center;
  gap: 8px;
}

.preset-strip {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  padding: 6px 8px;
  border: 1px solid var(--win-shadow-soft);
  background: rgba(191, 210, 230, 0.18);
  color: var(--text-main);
  font-size: 12px;
}

.preset-description {
  margin-top: 6px;
  color: var(--text-muted);
  font-size: 12px;
  line-height: 1.35;
}

.action-note {
  font-size: 11px;
  color: var(--text-muted);
  line-height: 1.35;
}

.action-note.is-ok {
  color: #2d6b3f;
}

.action-note.is-error {
  color: var(--error-color);
}

.button-stack {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.btn-secondary {
  min-height: 36px;
  border: 2px solid;
  border-color: var(--win-shadow-light) var(--win-shadow-dark) var(--win-shadow-dark) var(--win-shadow-light);
  background: var(--win-bg);
  color: var(--text-main);
  font-weight: 700;
  cursor: pointer;
}

.btn-secondary.active {
  background: #bfd2e6;
}

.layer-group-label {
  font-size: 11px;
  color: var(--text-muted);
  margin-top: 4px;
  font-style: italic;
}

.layer-list {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.display-mode-switch {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}

.mode-btn {
  min-height: 34px;
  border: 2px solid;
  border-color: var(--win-shadow-light) var(--win-shadow-dark) var(--win-shadow-dark) var(--win-shadow-light);
  background: var(--win-bg);
  color: var(--text-main);
  font-weight: 700;
  cursor: pointer;
}

.mode-btn.active {
  background: #bfd2e6;
}

.layer-row {
  flex-direction: row;
  align-items: center;
  gap: 8px;
  min-height: 28px;
}

.hint-text {
  color: var(--text-muted);
  font-size: 12px;
}

.error-box {
  padding: 10px;
  background: var(--error-bg);
  color: var(--error-color);
  border: 2px solid;
  border-color: var(--win-shadow-light) var(--error-border) var(--error-border) var(--win-shadow-light);
}

.validation-errors {
  padding: 6px 8px;
  background: #fff3cd;
  border: 1px solid #ffc107;
  font-size: 12px;
  color: #856404;
}

.validation-line {
  padding: 2px 0;
}

.info-card {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 8px;
  border: 1px solid var(--win-shadow-soft);
  background: rgba(191, 210, 230, 0.12);
}

.info-card-title {
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  color: var(--text-muted);
}

.info-line {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  font-size: 12px;
}

@media (max-width: 700px) {
  .two-columns {
    grid-template-columns: 1fr;
  }
}
</style>
