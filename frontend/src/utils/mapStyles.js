import { Fill, Stroke, Style, Circle as CircleStyle } from 'ol/style'
import Point from 'ol/geom/Point'
import { valueToColor } from './gradients'

export const SPECTRAL_LAYERS = new Set(['ndvi', 'ndwi', 'ndmi', 'bsi'])
export const LAYER_PROPERTY_MAP = {
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

export const manualFieldStyle = new Style({
  fill: new Fill({ color: 'rgba(33, 87, 156, 0)' }),
  stroke: new Stroke({ color: '#21579c', width: 2.8, lineDash: [8, 4] }),
})

export const selectedManualFieldStyle = new Style({
  fill: new Fill({ color: 'rgba(33, 87, 156, 0.05)' }),
  stroke: new Stroke({ color: '#21579c', width: 3.6, lineDash: [8, 4] }),
})

export const mergeSelectionStyle = new Style({
  fill: new Fill({ color: 'rgba(201, 126, 39, 0.06)' }),
  stroke: new Stroke({ color: '#c97e27', width: 3.2, lineDash: [10, 4] }),
})

export const archiveMarkerImage = new CircleStyle({
  radius: 5,
  fill: new Fill({ color: '#c93a36' }),
  stroke: new Stroke({ color: '#fff4f0', width: 1.5 }),
})

export const scenarioMarkerImage = new CircleStyle({
  radius: 5,
  fill: new Fill({ color: '#2563eb' }),
  stroke: new Stroke({ color: '#eff6ff', width: 1.5 }),
})

export const bothMarkersImage = new CircleStyle({
  radius: 5,
  fill: new Fill({ color: '#d4a017' }),
  stroke: new Stroke({ color: '#fffbeb', width: 1.5 }),
})

export const fieldHoverStyle = new Style({
  fill: new Fill({ color: 'rgba(36, 118, 187, 0)' }),
  stroke: new Stroke({ color: '#204f88', width: 3.5 }),
})

export const drawStyle = new Style({
  fill: new Fill({ color: 'rgba(226, 167, 44, 0.18)' }),
  stroke: new Stroke({ color: '#a86e13', width: 2, lineDash: [6, 4] }),
})

export const FIELD_STROKE_COLORS = {
  high: '#1e6a3a',
  medium: '#8a6b18',
  low: '#a04a24',
  unknown: '#6c7b88',
}

export const FIELD_CONFIDENCE_COLORS = [
  '#8f2921',
  '#b85d22',
  '#c38b20',
  '#5f8e2f',
  '#1e8a4b',
]

export const ZONE_STYLES = {
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
const gridStyleCache = new Map()

export function clamp01(value) {
  return Math.max(0, Math.min(1, value))
}

export function fieldQualityBand(feature) {
  const band = String(feature?.get('quality_band') || '')
  if (band === 'high' || band === 'medium' || band === 'low') {
    return band
  }
  return 'unknown'
}

export function fieldBoundaryConfidence(feature) {
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

export function fieldStrokeColor(feature, band) {
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

export function getFieldStyleForFeature(feature, selected = false) {
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

export function buildGridLayerStyle(layerId) {
  const isSpectral = SPECTRAL_LAYERS.has(layerId)
  const strokeColor = isSpectral ? 'rgba(255, 255, 255, 0.12)' : 'rgba(255, 255, 255, 0.18)'
  return (feature) => {
    const propertyName = LAYER_PROPERTY_MAP[layerId] || layerId
    const value = feature.get(propertyName)
    if (value === undefined || value === null) {
      return null
    }
    const color = valueToColor(value, layerId)
    const cacheKey = color + strokeColor
    let style = gridStyleCache.get(cacheKey)
    if (!style) {
      style = new Style({
        fill: new Fill({ color }),
        stroke: new Stroke({ color: strokeColor, width: 1 }),
      })
      if (gridStyleCache.size > 2000) gridStyleCache.clear()
      gridStyleCache.set(cacheKey, style)
    }
    return style
  }
}

export function resolveFieldStyle(feature, store) {
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

export function resolveZoneStyle(feature) {
  const zoneCode = String(feature?.get('zone_code') || 'medium')
  return ZONE_STYLES[zoneCode] || ZONE_STYLES.medium
}
