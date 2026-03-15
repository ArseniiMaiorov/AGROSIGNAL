<template>
  <section class="map-shell">
    <div class="map-toolbar">
      <div class="map-title">{{ t('map.title') }}</div>
      <div class="map-toolbar-meta">
        <span>{{ t('map.activeLayer') }}: {{ activeLayerLabel }}</span>
        <span v-if="store.visibleRunId">{{ t('map.run') }}: {{ shortVisibleRunId }}</span>
        <span v-if="store.showSatelliteBrowse && satelliteMetaLabel">{{ satelliteMetaLabel }}</span>
        <span v-if="store.activeRunId && store.activeRunId !== store.visibleRunId" class="map-run-pending">
          {{ t('map.pendingRun') }}
        </span>
      </div>
      <button
        class="toolbar-toggle-btn"
        :class="{ active: store.showFieldBoundaries }"
        :title="store.showFieldBoundaries ? t('map.hideBoundaries') : t('map.showBoundaries')"
        @click="store.showFieldBoundaries = !store.showFieldBoundaries"
      >
        {{ store.showFieldBoundaries ? t('map.hideBoundaries') : t('map.showBoundaries') }}
      </button>
    </div>
    <div ref="mapContainer" class="map-container"></div>
    <canvas ref="windCanvas" class="wind-canvas"></canvas>
    <div v-if="store.gridLayerStatus === 'loading'" class="map-overlay-status">
      {{ t('map.loadingLayer') }}
    </div>
    <div v-if="store.satelliteLoadStatus === 'loading'" class="map-overlay-status map-overlay-secondary">
      {{ t('map.loadingSatellite') }}
    </div>
    <div v-else-if="store.showSatelliteBrowse && store.satelliteLoadStatus === 'error'" class="map-overlay-status map-overlay-error">
      {{ t('map.satelliteError') }}
    </div>
    <div v-else-if="store.gridLayerStatus === 'no_data'" class="map-overlay-status map-overlay-warn">
      {{ t('map.noLayerData') }}
    </div>
    <div v-else-if="store.gridLayerStatus === 'error'" class="map-overlay-status map-overlay-error">
      {{ t('map.layerError') }}
    </div>
    <div v-else-if="windRenderState === 'speed_only'" class="map-overlay-status map-overlay-warn">
      {{ t('map.windFallback') }}
    </div>
    <section v-if="store.expertMode" class="map-debug-panel">
      <div class="map-debug-title">{{ t('map.segmentationDebugTitle') }}</div>
      <div class="map-debug-controls">
        <label>
          <span>{{ t('map.debugRun') }}</span>
          <select v-model="store.selectedDebugRunId">
            <option value="">{{ t('map.noDebugRuns') }}</option>
            <option v-for="run in debugRunOptions" :key="run.id" :value="run.id">
              {{ run.label }}
            </option>
          </select>
        </label>
        <label>
          <span>{{ t('map.debugTile') }}</span>
          <select v-model="store.selectedDebugTileId" :disabled="!debugTileOptions.length">
            <option value="">{{ t('map.noDebugTiles') }}</option>
            <option v-for="tile in debugTileOptions" :key="tile.id" :value="tile.id">
              {{ tile.label }}
            </option>
          </select>
        </label>
        <label>
          <span>{{ t('map.debugLayer') }}</span>
          <select v-model="store.selectedDebugLayerId" :disabled="!debugLayerOptions.length">
            <option value="">{{ t('map.noDebugLayers') }}</option>
            <option v-for="layer in debugLayerOptions" :key="layer.id" :value="layer.id">
              {{ layer.label }}
            </option>
          </select>
        </label>
      </div>
      <div class="map-debug-actions">
        <button class="toolbar-toggle-btn" :disabled="!store.selectedDebugRunId || store.isLoadingDebugTiles" @click="refreshDebugTiles">
          {{ t('map.refreshDebug') }}
        </button>
        <label class="map-debug-toggle">
          <input v-model="store.debugOverlayEnabled" type="checkbox" :disabled="!store.selectedDebugTileId || !store.selectedDebugLayerId" />
          <span>{{ t('map.debugOverlay') }}</span>
        </label>
      </div>
      <label class="map-debug-slider">
        <span>{{ t('map.debugOpacity') }}: {{ Math.round(store.debugOverlayOpacity * 100) }}%</span>
        <input v-model.number="store.debugOverlayOpacity" type="range" min="0.15" max="0.95" step="0.05" />
      </label>
      <div v-if="selectedDebugRuntimeMeta" class="map-debug-meta">
        <div>{{ t('map.debugWatershed') }}: {{ debugWatershedLabel }}</div>
        <div>{{ t('map.debugComponents') }}: {{ debugComponentSummary }}</div>
        <div>{{ t('map.debugSplitScores') }}: p50={{ debugSplitP50 }} · p90={{ debugSplitP90 }}</div>
      </div>
      <div v-else class="map-debug-empty">
        {{ t('map.noDebugLayers') }}
      </div>
    </section>
  </section>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useMapStore } from '../store/map'
import axios, { API_BASE } from '../services/api'
import { locale, t } from '../utils/i18n'
import Map from 'ol/Map'
import View from 'ol/View'
import TileLayer from 'ol/layer/Tile'
import ImageLayer from 'ol/layer/Image'
import VectorLayer from 'ol/layer/Vector'
import ImageStatic from 'ol/source/ImageStatic'
import VectorSource from 'ol/source/Vector'
import OSM from 'ol/source/OSM'
import GeoJSON from 'ol/format/GeoJSON'
import Overlay from 'ol/Overlay'
import { Draw } from 'ol/interaction'
import Point from 'ol/geom/Point'
import { fromLonLat, toLonLat, transformExtent } from 'ol/proj'
import { Circle as CircleStyle, Fill, Stroke, Style } from 'ol/style'
import { valueToColor } from '../utils/gradients'
import {
  formatReasonText,
  getDetectionPresetMeta,
  getLayerMeta,
  getQualityBandLabel,
  getTaskStageLabel,
} from '../utils/presentation'

const mapContainer = ref(null)
const windCanvas = ref(null)
const store = useMapStore()
const windRenderState = ref('off')

let map = null
let fieldsLayer = null
let fieldsSource = null
let managementZonesLayer = null
let managementZonesSource = null
let gridSource = null
let satelliteLayer = null
let debugOverlayLayer = null
let drawLayer = null
let drawSource = null
let drawInteraction = null
let suppressFieldFit = false
let resizeHandler = null
let gridLayers = []
let gridRequestController = null

const SPECTRAL_LAYERS = new Set(['ndvi', 'ndwi', 'ndmi', 'bsi'])
const LAYER_PROPERTY_MAP = {
  ndvi: 'ndvi_mean',
  ndwi: 'ndwi_mean',
  ndmi: 'ndmi_mean',
  bsi: 'bsi_mean',
  precipitation: 'precipitation_mm',
  wind: 'wind_speed_m_s',
  soil_moisture: 'soil_moist',
  gdd: 'gdd_sum',
  vpd: 'vpd_mean',
}

const manualFieldStyle = new Style({
  fill: new Fill({ color: 'rgba(33, 87, 156, 0)' }),
  stroke: new Stroke({ color: '#21579c', width: 2.8, lineDash: [8, 4] }),
})

const selectedManualFieldStyle = new Style({
  fill: new Fill({ color: 'rgba(33, 87, 156, 0.05)' }),
  stroke: new Stroke({ color: '#21579c', width: 3.6, lineDash: [8, 4] }),
})

const mergeSelectionStyle = new Style({
  fill: new Fill({ color: 'rgba(201, 126, 39, 0.06)' }),
  stroke: new Stroke({ color: '#c97e27', width: 3.2, lineDash: [10, 4] }),
})

const archiveMarkerImage = new CircleStyle({
  radius: 5,
  fill: new Fill({ color: '#c93a36' }),
  stroke: new Stroke({ color: '#fff4f0', width: 1.5 }),
})

const scenarioMarkerImage = new CircleStyle({
  radius: 5,
  fill: new Fill({ color: '#2563eb' }),
  stroke: new Stroke({ color: '#eff6ff', width: 1.5 }),
})

const bothMarkersImage = new CircleStyle({
  radius: 5,
  fill: new Fill({ color: '#d4a017' }),
  stroke: new Stroke({ color: '#fffbeb', width: 1.5 }),
})

const fieldHoverStyle = new Style({
  fill: new Fill({ color: 'rgba(36, 118, 187, 0)' }),
  stroke: new Stroke({ color: '#204f88', width: 3.5 }),
})

const drawStyle = new Style({
  fill: new Fill({ color: 'rgba(226, 167, 44, 0.18)' }),
  stroke: new Stroke({ color: '#a86e13', width: 2, lineDash: [6, 4] }),
})

const FIELD_STROKE_COLORS = {
  high: '#1e6a3a',
  medium: '#8a6b18',
  low: '#a04a24',
  unknown: '#6c7b88',
}

const FIELD_CONFIDENCE_COLORS = [
  '#8f2921',
  '#b85d22',
  '#c38b20',
  '#5f8e2f',
  '#1e8a4b',
]

const ZONE_STYLES = {
  high: new Style({
    fill: new Fill({ color: 'rgba(45, 143, 84, 0.24)' }),
    stroke: new Stroke({ color: '#226642', width: 1.6 }),
  }),
  medium: new Style({
    fill: new Fill({ color: 'rgba(207, 162, 41, 0.24)' }),
    stroke: new Stroke({ color: '#9a6f14', width: 1.6 }),
  }),
  low: new Style({
    fill: new Fill({ color: 'rgba(188, 98, 42, 0.24)' }),
    stroke: new Stroke({ color: '#92481f', width: 1.6 }),
  }),
}

const fieldStyleCache = new Map()
const selectedFieldStyleCache = new Map()

function clamp01(value) {
  return Math.max(0, Math.min(1, value))
}

function formatTooltipNumber(value, formatter) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return '—'
  }
  return formatter(Number(value))
}

function appendTooltipRow(container, label, value) {
  const row = document.createElement('div')
  row.textContent = `${label}: ${value}`
  container.appendChild(row)
}

function renderFieldTooltipContent(container, props) {
  const content = document.createDocumentFragment()
  const title = document.createElement('div')
  title.className = 'tooltip-title'
  title.textContent = t('tooltip.field')
  content.appendChild(title)

  appendTooltipRow(
    content,
    t('tooltip.area'),
    formatTooltipNumber(props.area_m2, (value) => `${(value / 10000).toFixed(2)} га`)
  )
  appendTooltipRow(
    content,
    t('tooltip.perimeter'),
    formatTooltipNumber(props.perimeter_m, (value) => `${Math.round(value)} м`)
  )
  appendTooltipRow(
    content,
    t('tooltip.quality'),
    formatTooltipNumber(props.quality_score, (value) => value.toFixed(2))
  )
  appendTooltipRow(
    content,
    t('field.geometryConfidence'),
    formatTooltipNumber(props.geometry_confidence, (value) => `${Math.round(value * 100)}%`)
  )
  appendTooltipRow(
    content,
    t('field.ttaConsensus'),
    formatTooltipNumber(props.tta_consensus, (value) => `${Math.round(value * 100)}%`)
  )
  appendTooltipRow(
    content,
    t('field.boundaryUncertainty'),
    formatTooltipNumber(props.boundary_uncertainty, (value) => `${Math.round(value * 100)}%`)
  )
  appendTooltipRow(
    content,
    t('tooltip.qualityBand'),
    getQualityBandLabel(props.quality_label || props.quality_band || 'unknown'),
  )
  appendTooltipRow(
    content,
    t('tooltip.qualityReason'),
    formatReasonText(props.quality_reason_code, props.quality_reason, props.quality_reason_params),
  )
  appendTooltipRow(
    content,
    t('tooltip.operationalTier'),
    props.operational_tier ? t(`field.operationalTier.${props.operational_tier}`) : '—'
  )
  appendTooltipRow(
    content,
    t('tooltip.reviewNeeded'),
    props.review_required ? t('field.yes') : t('field.no')
  )
  appendTooltipRow(
    content,
    t('field.source'),
    getSourceLabel(props.source || 'autodetect'),
  )

  container.replaceChildren(content)
}

function renderManagementZoneTooltipContent(container, props) {
  const content = document.createDocumentFragment()
  const title = document.createElement('div')
  title.className = 'tooltip-title'
  title.textContent = props.zone_label || t('field.managementZones')
  content.appendChild(title)
  appendTooltipRow(content, t('field.zoneScore'), formatTooltipNumber(props.zone_score, (value) => value.toFixed(3)))
  appendTooltipRow(
    content,
    t('field.zoneArea'),
    formatTooltipNumber(props.area_m2, (value) => `${(value / 10000).toFixed(2)} га`),
  )
  appendTooltipRow(
    content,
    t('field.zoneYield'),
    props.predicted_yield_kg_ha === null || props.predicted_yield_kg_ha === undefined
      ? (
        props.yield_potential_kg_ha === null || props.yield_potential_kg_ha === undefined
          ? t('field.yieldPotential')
          : `${Number(props.yield_potential_kg_ha).toFixed(0)} кг/га`
      )
      : `${Number(props.predicted_yield_kg_ha).toFixed(0)} кг/га`,
  )
  appendTooltipRow(
    content,
    t('field.confidence'),
    formatTooltipNumber(props.confidence, (value) => `${Math.round(value * 100)}%`),
  )
  container.replaceChildren(content)
}

function fieldQualityBand(feature) {
  const band = String(feature?.get('quality_band') || '')
  if (band === 'high' || band === 'medium' || band === 'low') {
    return band
  }
  return 'unknown'
}

function fieldBoundaryConfidence(feature) {
  const geometryConfidence = Number(feature?.get('geometry_confidence'))
  const ttaConsensus = Number(feature?.get('tta_consensus'))
  const boundaryUncertainty = Number(feature?.get('boundary_uncertainty'))
  const qualityConfidence = Number(feature?.get('quality_confidence'))
  const qualityScore = Number(feature?.get('quality_score'))

  let confidence = Number.isFinite(geometryConfidence)
    ? geometryConfidence
    : (Number.isFinite(qualityConfidence) ? qualityConfidence : qualityScore)
  if (!Number.isFinite(confidence)) {
    return null
  }
  confidence = clamp01(confidence)

  if (Number.isFinite(boundaryUncertainty)) {
    confidence = clamp01(confidence * (1 - 0.65 * clamp01(boundaryUncertainty)))
  }
  if (Number.isFinite(ttaConsensus)) {
    confidence = clamp01(confidence * 0.85 + clamp01(ttaConsensus) * 0.15)
  }
  return confidence
}

function fieldStrokeColor(feature, band) {
  if (String(feature?.get('source') || '').trim().toLowerCase() === 'autodetect_preview') {
    return '#b46a1f'
  }
  const confidence = fieldBoundaryConfidence(feature)
  if (confidence === null) {
    return FIELD_STROKE_COLORS[band] || FIELD_STROKE_COLORS.unknown
  }
  const bucket = Math.max(0, Math.min(FIELD_CONFIDENCE_COLORS.length - 1, Math.floor(confidence * FIELD_CONFIDENCE_COLORS.length)))
  return FIELD_CONFIDENCE_COLORS[bucket]
}

function getFieldStyleForFeature(feature, selected = false) {
  const band = fieldQualityBand(feature)
  const strokeColor = fieldStrokeColor(feature, band)
  const preview = String(feature?.get('source') || '').trim().toLowerCase() === 'autodetect_preview'
  const cacheKey = `${band}:${strokeColor}:${preview ? 'preview' : 'normal'}:${selected ? 'selected' : 'plain'}`
  const cache = selected ? selectedFieldStyleCache : fieldStyleCache
  const cached = cache.get(cacheKey)
  if (cached) {
    return cached
  }
  const style = selected
    ? [
        new Style({
          fill: new Fill({ color: preview ? 'rgba(180, 106, 31, 0.08)' : 'rgba(33, 87, 156, 0.04)' }),
          stroke: new Stroke({
            color: preview ? 'rgba(180, 106, 31, 0.82)' : 'rgba(33, 87, 156, 0.78)',
            width: 4.4,
            lineDash: preview ? [10, 5] : undefined,
          }),
        }),
        new Style({
          fill: new Fill({ color: preview ? 'rgba(180, 106, 31, 0.03)' : 'rgba(40, 142, 72, 0)' }),
          stroke: new Stroke({ color: strokeColor, width: 2.6, lineDash: preview ? [10, 5] : undefined }),
        }),
      ]
    : new Style({
        fill: new Fill({ color: preview ? 'rgba(180, 106, 31, 0.03)' : 'rgba(40, 142, 72, 0)' }),
        stroke: new Stroke({ color: strokeColor, width: 2.5, lineDash: preview ? [10, 5] : undefined }),
      })
  cache.set(cacheKey, style)
  return style
}

const _gridStyleCache = new Map()
function buildGridLayerStyle(layerId, order = 0) {
  const isSpectral = SPECTRAL_LAYERS.has(layerId)
  const strokeColor = isSpectral ? 'rgba(255, 255, 255, 0.12)' : 'rgba(255, 255, 255, 0.18)'
  return (feature) => {
    const propertyName = LAYER_PROPERTY_MAP[layerId] || layerId
    const rawFieldCoverage = feature.get('field_coverage')
    const hasFieldCoverage = rawFieldCoverage !== null && rawFieldCoverage !== undefined
    const fieldCoverage = hasFieldCoverage ? Number(rawFieldCoverage) : null
    if (store.showFieldsOnly && hasFieldCoverage && fieldCoverage <= 0) {
      return null
    }
    const value = feature.get(propertyName)
    if (value === undefined || value === null) {
      return null
    }
    const color = valueToColor(value, layerId)
    const cacheKey = color + strokeColor
    let style = _gridStyleCache.get(cacheKey)
    if (!style) {
      style = new Style({
        fill: new Fill({ color }),
        stroke: new Stroke({ color: strokeColor, width: 1 }),
      })
      if (_gridStyleCache.size > 2000) _gridStyleCache.clear()
      _gridStyleCache.set(cacheKey, style)
    }
    return style
  }
}

function syncGridLayers() {
  if (!map) {
    return
  }
  for (const layer of gridLayers) {
    map.removeLayer(layer)
  }
  gridLayers = []

  if (!store.activeLayerIds.length) {
    return
  }

  const sortedLayerIds = [...store.activeLayerIds].sort((left, right) => {
    const leftSpectral = SPECTRAL_LAYERS.has(left)
    const rightSpectral = SPECTRAL_LAYERS.has(right)
    if (leftSpectral === rightSpectral) {
      return 0
    }
    return leftSpectral ? -1 : 1
  })

  for (const [index, layerId] of sortedLayerIds.entries()) {
    const layer = new VectorLayer({
      source: gridSource,
      style: buildGridLayerStyle(layerId, index),
      zIndex: SPECTRAL_LAYERS.has(layerId) ? 2 : 3 + index,
    })
    gridLayers.push(layer)
    map.addLayer(layer)
  }
}

const activeLayerLabel = computed(() => {
  const layer = store.availableLayers.find((item) => item.id === store.primaryLayerId)
  return getLayerMeta(store.primaryLayerId, layer || {}).label
})

const combinedFieldsGeoJson = computed(() => {
  const autoFeatures = store.fieldsGeoJson?.features || []
  const manualFeatures = store.manualFieldsGeoJson?.features || []
  return {
    type: 'FeatureCollection',
    features: [...autoFeatures, ...manualFeatures],
  }
})

const shortVisibleRunId = computed(() => String(store.visibleRunId || '').slice(0, 8))
const satelliteMetaLabel = computed(() => {
  const scene = store.satelliteScene
  if (!store.showSatelliteBrowse || !scene) {
    return ''
  }
  const dateToken = scene.requested_date || scene.requested_window?.start || 'auto'
  const cloud = scene.cloud_cover_pct === null || scene.cloud_cover_pct === undefined
    ? (locale.value === 'ru' ? 'облачность —' : 'cloud —')
    : (locale.value === 'ru' ? `облачность ${Number(scene.cloud_cover_pct).toFixed(0)}%` : `cloud ${Number(scene.cloud_cover_pct).toFixed(0)}%`)
  return locale.value === 'ru'
    ? `Спутник: ${dateToken} · ${cloud} · ${scene.provider_account || 'primary'}`
    : `Satellite: ${dateToken} · ${cloud} · ${scene.provider_account || 'primary'}`
})

const debugRunOptions = computed(() => {
  return (store.runSummaries || []).map((run) => ({
    id: run.id,
    label: `${String(run.id).slice(0, 8)} · ${getTaskStageLabel(run.status, run.status)} · ${getDetectionPresetMeta(run.preset || 'standard').label}`,
  }))
})

const debugTileOptions = computed(() => {
  return (store.debugTilesCatalog || []).map((tile) => ({
    id: String(tile.tile_id || ''),
    label: String(tile.tile_id || ''),
  }))
})

const debugLayerOptions = computed(() => {
  return (store.selectedDebugTileDetail?.available_layers || []).map((layer) => ({
    id: String(layer.id || ''),
    label: layer.label || String(layer.id || ''),
  }))
})

const selectedDebugRuntimeMeta = computed(() => store.selectedDebugTileDetail?.runtime_meta || null)

const debugWatershedLabel = computed(() => {
  const runtime = selectedDebugRuntimeMeta.value || {}
  if (!runtime || !Object.keys(runtime).length) return '—'
  if (runtime.watershed_rollback_reason) {
    return locale.value === 'ru'
      ? `откат · ${runtime.watershed_rollback_reason}`
      : `rollback · ${runtime.watershed_rollback_reason}`
  }
  if (runtime.watershed_applied) return locale.value === 'ru' ? 'применён' : 'applied'
  if (runtime.watershed_skipped_reason) {
    return locale.value === 'ru'
      ? `пропущен · ${runtime.watershed_skipped_reason}`
      : `skipped · ${runtime.watershed_skipped_reason}`
  }
  return '—'
})

const debugComponentSummary = computed(() => {
  const runtime = selectedDebugRuntimeMeta.value || {}
  if (!runtime || !Object.keys(runtime).length) return '—'
  const parts = [
    `grow ${runtime.components_after_grow ?? 0}`,
    `gap ${runtime.components_after_gap_close ?? 0}`,
    `infill ${runtime.components_after_infill ?? 0}`,
    `merge ${runtime.components_after_merge ?? 0}`,
    `ws ${runtime.components_after_watershed ?? 0}`,
  ]
  return parts.join(' · ')
})

const debugSplitP50 = computed(() => {
  const value = Number(selectedDebugRuntimeMeta.value?.split_score_p50)
  return Number.isFinite(value) ? value.toFixed(3) : '—'
})

const debugSplitP90 = computed(() => {
  const value = Number(selectedDebugRuntimeMeta.value?.split_score_p90)
  return Number.isFinite(value) ? value.toFixed(3) : '—'
})

function syncSatelliteLayer(payload = store.satelliteScene) {
  if (!satelliteLayer) {
    return
  }
  if (!store.showSatelliteBrowse || !payload?.image_base64 || !Array.isArray(payload?.bbox)) {
    satelliteLayer.setVisible(false)
    satelliteLayer.setSource(null)
    return
  }
  const imageExtent = transformExtent(payload.bbox, 'EPSG:4326', 'EPSG:3857')
  satelliteLayer.setSource(
    new ImageStatic({
      url: `data:image/png;base64,${payload.image_base64}`,
      imageExtent,
      projection: 'EPSG:3857',
      interpolate: false,
    }),
  )
  satelliteLayer.setVisible(true)
}

function syncDebugOverlayLayer(payload = store.debugLayerPayload) {
  if (!debugOverlayLayer) {
    return
  }
  if (!store.expertMode || !store.debugOverlayEnabled || !payload?.image_base64 || !Array.isArray(payload?.bounds)) {
    debugOverlayLayer.setVisible(false)
    debugOverlayLayer.setSource(null)
    return
  }
  const imageExtent = transformExtent(payload.bounds, 'EPSG:4326', 'EPSG:3857')
  debugOverlayLayer.setOpacity(Math.min(0.95, Math.max(0.15, Number(store.debugOverlayOpacity || 0.5))))
  debugOverlayLayer.setSource(
    new ImageStatic({
      url: `data:image/png;base64,${payload.image_base64}`,
      imageExtent,
      projection: 'EPSG:3857',
      interpolate: false,
    }),
  )
  debugOverlayLayer.setVisible(true)
}

async function refreshDebugTiles() {
  if (!store.selectedDebugRunId) {
    return
  }
  await store.loadRunDebugTiles(store.selectedDebugRunId)
}

async function loadSatelliteLayer(options = {}) {
  if (!map || !store.showSatelliteBrowse) {
    syncSatelliteLayer(null)
    return
  }
  const mapSize = map.getSize()
  if (!mapSize) {
    return
  }
  const extent4326 = transformExtent(map.getView().calculateExtent(mapSize), 'EPSG:3857', 'EPSG:4326')
  const width = Math.max(256, Math.min(1280, Math.round(mapSize[0])))
  const height = Math.max(256, Math.min(1280, Math.round(mapSize[1])))
  const payload = await store.loadSatelliteScene({
    bbox: extent4326,
    width,
    height,
    manual: Boolean(options.manual),
  })
  syncSatelliteLayer(payload)
}

function resolveFieldStyle(feature) {
  if (store.mergeMode && store.mergeSelectionIds.includes(feature?.get('field_id'))) {
    return mergeSelectionStyle
  }
  const isSelected = store.selectedFieldIds.includes(feature?.get('field_id'))
  const isManual = feature?.get('source') === 'manual'
  const baseStyle = isSelected
    ? (isManual ? selectedManualFieldStyle : getFieldStyleForFeature(feature, true))
    : (isManual ? manualFieldStyle : getFieldStyleForFeature(feature, false))
  const styles = Array.isArray(baseStyle) ? [...baseStyle] : [baseStyle]
  const hasArchive = feature?.get('has_archive')
  const hasScenarios = feature?.get('has_scenarios')
  if (hasArchive || hasScenarios) {
    const geometry = feature.getGeometry()
    if (geometry) {
      const extent = geometry.getExtent()
      const markerPoint = new Point([
        extent[2] - Math.max((extent[2] - extent[0]) * 0.12, 6),
        extent[3] - Math.max((extent[3] - extent[1]) * 0.12, 6),
      ])
      const markerImage = hasArchive && hasScenarios
        ? bothMarkersImage
        : hasArchive
          ? archiveMarkerImage
          : scenarioMarkerImage
      styles.push(new Style({ geometry: markerPoint, image: markerImage }))
    }
  }
  return styles
}

function resolveZoneStyle(feature) {
  const zoneCode = String(feature?.get('zone_code') || 'medium')
  return ZONE_STYLES[zoneCode] || ZONE_STYLES.medium
}

function mapViewZoomToGridZoom() {
  const zoom = Number(map?.getView()?.getZoom?.() || 11)
  if (zoom < 10) return 0
  if (zoom < 11.5) return 1
  if (zoom < 13) return 2
  if (zoom < 14.5) return 3
  return 4
}

function syncWindCanvasSize() {
  if (!windCanvas.value || !mapContainer.value) {
    return
  }
  const width = Math.max(1, Math.round(mapContainer.value.clientWidth))
  const height = Math.max(1, Math.round(mapContainer.value.clientHeight))
  if (windCanvas.value.width !== width) windCanvas.value.width = width
  if (windCanvas.value.height !== height) windCanvas.value.height = height
}

function clearWindCanvas() {
  const ctx = windCanvas.value?.getContext?.('2d')
  if (!ctx || !windCanvas.value) {
    return
  }
  ctx.clearRect(0, 0, windCanvas.value.width, windCanvas.value.height)
}

function densityDivisor() {
  if (store.mapLabelDensity === 'full') return 1
  if (store.mapLabelDensity === 'compact') return 2
  return 4
}

function renderWindStreaks() {
  syncWindCanvasSize()
  clearWindCanvas()
  if (!store.activeLayers.wind || !windCanvas.value || !map || !gridSource) {
    windRenderState.value = 'off'
    return
  }

  const features = gridSource.getFeatures()
  if (!features.length) {
    windRenderState.value = 'off'
    return
  }

  const vectorFeatures = features.filter((feature) => {
    const u = Number(feature.get('u_wind_10m'))
    const v = Number(feature.get('v_wind_10m'))
    return Number.isFinite(u) && Number.isFinite(v)
  })
  if (!vectorFeatures.length) {
    const hasSpeedOnly = features.some((feature) => Number.isFinite(Number(feature.get('wind_speed_m_s'))))
    windRenderState.value = hasSpeedOnly ? 'speed_only' : 'off'
    return
  }

  windRenderState.value = 'vector'
  const ctx = windCanvas.value.getContext('2d')
  const zoom = Number(map.getView().getZoom() || 11)
  const divisor = densityDivisor()
  const lineWidth = store.animationDensity === 'low' ? 1 : 1.25

  ctx.save()
  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'

  vectorFeatures.forEach((feature, index) => {
    const row = Number(feature.get('row') || 0)
    const col = Number(feature.get('col') || 0)
    if (((row + col + index) % divisor) !== 0) {
      return
    }

    const geometry = feature.getGeometry()
    if (!geometry) {
      return
    }
    const extent = geometry.getExtent()
    const center = [(extent[0] + extent[2]) / 2, (extent[1] + extent[3]) / 2]
    const pixel = map.getPixelFromCoordinate(center)
    if (!pixel) {
      return
    }

    const u = Number(feature.get('u_wind_10m'))
    const v = Number(feature.get('v_wind_10m'))
    const speed = Number(feature.get('wind_speed_m_s'))
    const speedNorm = Number.isFinite(speed) ? speed : Math.sqrt(u * u + v * v)
    const magnitude = Math.sqrt(u * u + v * v)
    if (!Number.isFinite(magnitude) || magnitude <= 0.001) {
      return
    }

    const length = Math.max(10, Math.min(30, speedNorm * (2.4 + zoom * 0.12)))
    const dx = (u / magnitude) * length
    const dy = (-v / magnitude) * length
    const color = valueToColor(speedNorm, 'wind')

    ctx.strokeStyle = color
    ctx.lineWidth = lineWidth
    ctx.globalAlpha = 0.85
    ctx.beginPath()
    ctx.moveTo(pixel[0], pixel[1])
    ctx.lineTo(pixel[0] + dx, pixel[1] + dy)
    ctx.stroke()

    ctx.globalAlpha = 0.36
    ctx.beginPath()
    ctx.moveTo(pixel[0] - dx * 0.28, pixel[1] - dy * 0.28)
    ctx.lineTo(pixel[0], pixel[1])
    ctx.stroke()

    ctx.globalAlpha = 0.2
    ctx.beginPath()
    ctx.moveTo(pixel[0] - dx * 0.5, pixel[1] - dy * 0.5)
    ctx.lineTo(pixel[0] - dx * 0.18, pixel[1] - dy * 0.18)
    ctx.stroke()
  })

  ctx.restore()
}

onMounted(() => {
  fieldsSource = new VectorSource()
  managementZonesSource = new VectorSource()
  gridSource = new VectorSource()
  drawSource = new VectorSource()
  satelliteLayer = new ImageLayer({ source: null, visible: false, zIndex: 1.5, opacity: 0.94 })
  debugOverlayLayer = new ImageLayer({ source: null, visible: false, zIndex: 10.5, opacity: 0.5 })
  managementZonesLayer = new VectorLayer({ source: managementZonesSource, style: resolveZoneStyle, zIndex: 12, visible: false })
  fieldsLayer = new VectorLayer({ source: fieldsSource, style: resolveFieldStyle, zIndex: 20 })
  drawLayer = new VectorLayer({ source: drawSource, style: drawStyle, zIndex: 21 })

  map = new Map({
    target: mapContainer.value,
    layers: [new TileLayer({ source: new OSM() }), satelliteLayer, debugOverlayLayer, managementZonesLayer, fieldsLayer, drawLayer],
    view: new View({
      center: fromLonLat([store.centerLon, store.centerLat]),
      zoom: 11,
    }),
  })
  syncGridLayers()

  let _resizeRaf = null
  resizeHandler = () => {
    if (_resizeRaf) cancelAnimationFrame(_resizeRaf)
    _resizeRaf = requestAnimationFrame(() => {
      _resizeRaf = null
      if (map) map.updateSize()
      syncWindCanvasSize()
      renderWindStreaks()
    })
  }
  window.addEventListener('resize', resizeHandler)
  requestAnimationFrame(() => {
    if (map) {
      map.updateSize()
    }
    syncWindCanvasSize()
    renderWindStreaks()
  })

  const tooltipEl = document.createElement('div')
  tooltipEl.className = 'map-tooltip'
  const tooltip = new Overlay({
    element: tooltipEl,
    positioning: 'bottom-left',
    offset: [14, -10],
  })
  map.addOverlay(tooltip)

  let hoveredFeature = null
  map.on('pointermove', (event) => {
    if (hoveredFeature) {
      hoveredFeature.setStyle(undefined)
      hoveredFeature = null
    }
    const hoveredItems = []
    map.forEachFeatureAtPixel(event.pixel, (item) => {
      hoveredItems.push(item)
      return null
    })
    const feature =
      hoveredItems.find((item) => item.getProperties()?.field_id) ||
      hoveredItems.find((item) => item.getProperties()?.kind === 'management_zone')
    if (!feature) {
      tooltip.setPosition(undefined)
      return
    }
    const props = feature.getProperties()
    if (props.field_id) {
      hoveredFeature = feature
      feature.setStyle(fieldHoverStyle)
      renderFieldTooltipContent(tooltipEl, props)
    } else if (props.kind === 'management_zone') {
      renderManagementZoneTooltipContent(tooltipEl, props)
    } else {
      tooltip.setPosition(undefined)
      return
    }
    tooltip.setPosition(event.coordinate)
  })

  map.on('singleclick', (event) => {
    if (store.isPickingSearchCenter) {
      const [lon, lat] = toLonLat(event.coordinate)
      const confirmed = window.confirm(
        `Использовать эту точку как центр поиска?\nШирота: ${lat.toFixed(6)}\nДолгота: ${lon.toFixed(6)}`
      )
      if (confirmed) {
        store.applySearchCenter(lat, lon)
      } else {
        store.toggleSearchCenterPicking()
      }
      return
    }

    const clickedItems = []
    map.forEachFeatureAtPixel(event.pixel, (item) => {
      clickedItems.push(item)
      return null
    })
    const feature = clickedItems.find((item) => item.getProperties()?.field_id)
    if (feature?.getProperties()?.field_id) {
      const additive = Boolean(
        event.originalEvent?.ctrlKey ||
        event.originalEvent?.metaKey ||
        event.originalEvent?.shiftKey
      )
      if (store.mergeMode) {
        store.toggleMergeFieldSelection(feature.getProperties().field_id)
        fieldsLayer?.changed()
        return
      }
      store.selectField(feature.getProperties(), { additive })
      fieldsLayer?.changed()
      return
    }
    store.clearFieldSelection()
    fieldsLayer?.changed()
  })

  map.on('moveend', () => {
    loadSatelliteLayer()
    loadGridLayer()
    renderWindStreaks()
  })
})

onBeforeUnmount(() => {
  if (gridRequestController) {
    gridRequestController.abort()
    gridRequestController = null
  }
  if (resizeHandler) {
    window.removeEventListener('resize', resizeHandler)
    resizeHandler = null
  }
  if (map) {
    map.setTarget(undefined)
    map = null
  }
})

watch(
  () => combinedFieldsGeoJson.value,
  (geojson) => {
    if (!fieldsSource) {
      return
    }
    fieldsSource.clear()
    if (geojson?.features?.length) {
      const features = new GeoJSON().readFeatures(geojson, { featureProjection: 'EPSG:3857' })
      fieldsSource.addFeatures(features)
      if (!suppressFieldFit) {
        const extent = fieldsSource.getExtent()
        if (extent && Number.isFinite(extent[0])) {
          map.getView().fit(extent, {
            padding: [60, 60, 60, 60],
            duration: store.animationDensity === 'low' ? 180 : 600,
            maxZoom: 15,
          })
        }
      }
      suppressFieldFit = false
      loadGridLayer()
      fieldsLayer?.changed()
      renderWindStreaks()
    }
  },
  { immediate: true },
)

watch(
  () => store.fieldScenarios,
  (scenarios) => {
    if (!fieldsSource || !scenarios?.length) return
    const fieldId = scenarios[0]?.field_id
    if (!fieldId) return
    const feature = fieldsSource.getFeatures().find((f) => f.get('field_id') === fieldId)
    if (feature && !feature.get('has_scenarios')) {
      feature.set('has_scenarios', true)
      fieldsLayer?.changed()
    }
  },
  { deep: true },
)

watch(
  () => [store.fieldManagementZones, store.showManagementZonesOverlay],
  () => {
    if (!managementZonesSource || !managementZonesLayer) {
      return
    }
    managementZonesSource.clear()
    if (!store.showManagementZonesOverlay) {
      managementZonesLayer.setVisible(false)
      return
    }
    const geojson = store.fieldManagementZones?.geojson
    if (geojson?.features?.length) {
      const features = new GeoJSON().readFeatures(geojson, { featureProjection: 'EPSG:3857' })
      managementZonesSource.addFeatures(features)
      managementZonesLayer.setVisible(true)
      return
    }
    managementZonesLayer.setVisible(false)
  },
  { deep: true, immediate: true },
)

watch(
  () => [store.expertMode, store.visibleRunId],
  async ([expertModeEnabled, visibleRunId]) => {
    if (!expertModeEnabled) {
      store.clearDebugOverlay()
      return
    }
    const targetRunId = store.selectedDebugRunId || visibleRunId
    if (!targetRunId) {
      return
    }
    if (store.selectedDebugRunId !== targetRunId) {
      store.selectedDebugRunId = targetRunId
    }
    if (!store.debugTilesCatalog.length || store.selectedDebugRunId === targetRunId) {
      await store.loadRunDebugTiles(targetRunId)
    }
  },
  { immediate: true },
)

watch(
  () => store.selectedDebugRunId,
  async (runId, previousRunId) => {
    if (!store.expertMode) {
      return
    }
    if (!runId) {
      store.debugTilesCatalog = []
      store.selectedDebugTileId = ''
      store.selectedDebugLayerId = ''
      store.selectedDebugTileDetail = null
      store.clearDebugOverlay()
      return
    }
    if (runId !== previousRunId || !store.debugTilesCatalog.length) {
      await store.loadRunDebugTiles(runId)
    }
  },
)

watch(
  () => store.selectedDebugTileId,
  async (tileId, previousTileId) => {
    if (!store.expertMode || !store.selectedDebugRunId || !tileId || tileId === previousTileId) {
      return
    }
    await store.loadDebugTile(store.selectedDebugRunId, tileId, { silent: true })
  },
)

watch(
  () => [store.selectedDebugLayerId, store.debugOverlayEnabled, store.selectedDebugRunId, store.selectedDebugTileId],
  async ([layerId, enabled, runId, tileId]) => {
    if (!store.expertMode || !enabled) {
      syncDebugOverlayLayer(null)
      return
    }
    if (!runId || !tileId || !layerId) {
      syncDebugOverlayLayer(null)
      return
    }
    await store.loadDebugLayer(runId, tileId, layerId, { silent: true })
  },
  { deep: true },
)

watch(
  () => [store.debugLayerPayload, store.debugOverlayOpacity, store.debugOverlayEnabled, store.expertMode],
  () => {
    syncDebugOverlayLayer(store.debugLayerPayload)
  },
  { deep: true, immediate: true },
)

watch(
  () => [store.centerLat, store.centerLon],
  ([lat, lon]) => {
    if (!map) {
      return
    }
    suppressFieldFit = true
    map.getView().animate({
      center: fromLonLat([lon, lat]),
      duration: store.animationDensity === 'low' ? 160 : 500,
    })
  },
)

watch(
  () => store.isPickingSearchCenter,
  (enabled) => {
    if (!map?.getTargetElement()) {
      return
    }
    map.getTargetElement().style.cursor = enabled ? 'crosshair' : ''
  },
)

watch(
  () => [store.activeLayerIds, store.visibleRunId],
  () => {
    syncGridLayers()
    loadSatelliteLayer()
    loadGridLayer()
    renderWindStreaks()
  },
  { deep: true },
)

watch(
  () => [store.showSatelliteBrowse, store.satelliteBrowseDate, store.startDate, store.endDate, store.maxCloudPct],
  () => {
    if (!store.showSatelliteBrowse) {
      syncSatelliteLayer(null)
      return
    }
    loadSatelliteLayer({ manual: true })
  },
)

watch(
  () => store.showFieldsOnly,
  () => {
    syncGridLayers()
    for (const layer of gridLayers) {
      layer.changed()
    }
    renderWindStreaks()
  },
)

watch(
  () => store.showFieldBoundaries,
  (visible) => {
    if (fieldsLayer) {
      fieldsLayer.setVisible(visible)
    }
  },
)

watch(
  () => [store.drawMode, store.splitMode],
  ([drawEnabled, splitEnabled]) => {
    if (!map || !drawSource) {
      return
    }
    if (drawInteraction) {
      map.removeInteraction(drawInteraction)
      drawInteraction = null
    }
    drawSource.clear()
    if (!drawEnabled && !splitEnabled) {
      return
    }
    drawInteraction = new Draw({ source: drawSource, type: splitEnabled ? 'LineString' : 'Polygon' })
    drawInteraction.on('drawend', async (event) => {
      const geometry = new GeoJSON().writeFeatureObject(event.feature, {
        dataProjection: 'EPSG:4326',
        featureProjection: 'EPSG:3857',
      }).geometry
      if (splitEnabled) {
        await store.splitSelectedField(geometry)
      } else {
        const field = await store.createManualField(geometry)
        if (field) {
          event.feature.setProperties({
            field_id: field.id,
            aoi_run_id: field.aoi_run_id,
            area_m2: field.area_m2,
            perimeter_m: field.perimeter_m,
            quality_score: field.quality_score,
            quality_confidence: field.quality_score ?? 1,
            quality_band: 'manual',
            quality_label: 'manual',
            quality_reason: 'Контур подтверждён вручную.',
            source: field.source,
          })
          fieldsSource.addFeature(event.feature)
        }
      }
      drawSource.clear()
      if (drawInteraction) {
        map.removeInteraction(drawInteraction)
        drawInteraction = null
      }
    })
    map.addInteraction(drawInteraction)
  },
)

watch(
  () => [store.mergeMode, store.mergeSelectionIds, store.selectedFieldIds],
  () => {
    fieldsLayer?.changed()
  },
  { deep: true },
)

async function loadGridLayer() {
  if (!map || !gridSource) {
    return
  }
  if (gridRequestController) {
    gridRequestController.abort()
    gridRequestController = null
  }
  gridSource.clear()
  if (!store.visibleRunId || !store.activeLayerIds.length) {
    store.gridLayerStatus = 'idle'
    renderWindStreaks()
    return
  }
  store.gridLayerStatus = 'loading'
  const extent4326 = transformExtent(map.getView().calculateExtent(map.getSize()), 'EPSG:3857', 'EPSG:4326')
  const selectedZoom = mapViewZoomToGridZoom()
  gridRequestController = new AbortController()
  try {
    const response = await axios.get(`${API_BASE}/layers/${store.primaryLayerId}/grid`, {
      signal: gridRequestController.signal,
      params: {
        aoi_run_id: store.visibleRunId,
        zoom: selectedZoom,
        allow_run_fallback: false,
        minx: extent4326[0],
        miny: extent4326[1],
        maxx: extent4326[2],
        maxy: extent4326[3],
      },
    })
    if (response.data?.features?.length) {
      const features = new GeoJSON().readFeatures(response.data, { featureProjection: 'EPSG:3857' })
      gridSource.addFeatures(features)
      store.gridLayerStatus = 'ready'
      renderWindStreaks()
      return
    }
    if (selectedZoom !== 2) {
      const fallback = await axios.get(`${API_BASE}/layers/${store.primaryLayerId}/grid`, {
        signal: gridRequestController.signal,
        params: {
          aoi_run_id: store.visibleRunId,
          zoom: 2,
          allow_run_fallback: false,
          minx: extent4326[0],
          miny: extent4326[1],
          maxx: extent4326[2],
          maxy: extent4326[3],
        },
      })
      if (fallback.data?.features?.length) {
        const features = new GeoJSON().readFeatures(fallback.data, { featureProjection: 'EPSG:3857' })
        gridSource.addFeatures(features)
        store.gridLayerStatus = 'ready'
        renderWindStreaks()
        return
      }
    }
    store.gridLayerStatus = 'no_data'
    renderWindStreaks()
  } catch (error) {
    if (store.isAbortError?.(error) || error?.code === 'ERR_CANCELED') {
      return
    }
    store.gridLayerStatus = 'error'
    store.addLog(`Не удалось загрузить слой ${store.primaryLayerId}: сетка данных недоступна`)
  } finally {
    gridRequestController = null
  }
}
</script>

<style scoped>
.map-shell {
  position: absolute;
  inset: 8px;
  z-index: 1;
  overflow: hidden;
  border: 2px solid;
  border-color: var(--win-shadow-light) var(--win-shadow-dark) var(--win-shadow-dark) var(--win-shadow-light);
  background: var(--win-bg);
}

.map-toolbar {
  position: absolute;
  top: 6px;
  left: 12px;
  right: 12px;
  z-index: 3;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.map-title {
  font-weight: 700;
  color: var(--text-main);
}

.map-toolbar-meta {
  display: flex;
  gap: 12px;
  color: var(--text-muted);
  font-size: 12px;
}

.map-run-pending {
  color: #8a5a17;
}

.map-container {
  position: absolute;
  inset: 30px 8px 8px;
  border: 2px solid;
  border-color: var(--win-shadow-mid) var(--win-shadow-light) var(--win-shadow-light) var(--win-shadow-mid);
  background: #e3e0d8;
}

.wind-canvas {
  position: absolute;
  inset: 30px 8px 8px;
  z-index: 4;
  pointer-events: none;
}

:global(.map-tooltip) {
  background: var(--tooltip-bg);
  border: 2px solid;
  border-color: var(--win-shadow-light) var(--win-shadow-dark) var(--win-shadow-dark) var(--win-shadow-light);
  padding: 8px 10px;
  color: var(--text-main);
  font-size: 12px;
  min-width: 160px;
}

:global(.tooltip-title) {
  font-weight: 700;
  margin-bottom: 4px;
}

.map-overlay-status {
  position: absolute;
  bottom: 18px;
  left: 50%;
  transform: translateX(-50%);
  z-index: 10;
  padding: 6px 16px;
  border: 2px solid;
  border-color: var(--win-shadow-light) var(--win-shadow-dark) var(--win-shadow-dark) var(--win-shadow-light);
  background: var(--win-bg);
  color: var(--text-main);
  font-size: 12px;
  font-weight: 700;
  white-space: nowrap;
}

.map-overlay-warn {
  background: #fff3cd;
  color: #856404;
}

.map-overlay-secondary {
  bottom: 52px;
  background: #e8eef7;
  color: #2d5a80;
}

.map-overlay-error {
  background: #f8d7da;
  color: #721c24;
}

.toolbar-toggle-btn {
  padding: 2px 8px;
  font-size: 11px;
  font-family: inherit;
  cursor: pointer;
  border: 2px solid;
  border-color: var(--win-shadow-light) var(--win-shadow-dark) var(--win-shadow-dark) var(--win-shadow-light);
  background: var(--win-bg);
  color: var(--text-main);
  white-space: nowrap;
}

.toolbar-toggle-btn:hover {
  background: var(--btn-hover-bg, #d4d0c8);
}

.toolbar-toggle-btn.active {
  border-color: var(--win-shadow-dark) var(--win-shadow-light) var(--win-shadow-light) var(--win-shadow-dark);
  background: var(--btn-active-bg, #c8c4bc);
}

.map-debug-panel {
  position: absolute;
  top: 44px;
  right: 18px;
  z-index: 11;
  width: 320px;
  padding: 10px;
  border: 2px solid;
  border-color: var(--win-shadow-light) var(--win-shadow-dark) var(--win-shadow-dark) var(--win-shadow-light);
  background: rgba(228, 225, 217, 0.96);
  color: var(--text-main);
  font-size: 12px;
}

.map-debug-title {
  margin-bottom: 8px;
  font-weight: 700;
}

.map-debug-controls {
  display: grid;
  gap: 8px;
}

.map-debug-controls label,
.map-debug-slider {
  display: grid;
  gap: 4px;
}

.map-debug-controls select,
.map-debug-slider input {
  width: 100%;
}

.map-debug-actions {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-top: 8px;
}

.map-debug-toggle {
  display: flex;
  align-items: center;
  gap: 6px;
}

.map-debug-meta {
  margin-top: 8px;
  display: grid;
  gap: 4px;
  color: var(--text-muted);
}

.map-debug-empty {
  margin-top: 8px;
  color: var(--text-muted);
}
</style>
