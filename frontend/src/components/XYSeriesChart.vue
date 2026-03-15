<template>
  <div class="xy-chart-shell">
    <svg v-if="hasDrawableContent" :viewBox="`0 0 ${W} ${H}`" preserveAspectRatio="none" class="xy-chart-svg">
      <line :x1="padL" :y1="H - padB" :x2="W - padR" :y2="H - padB" class="axis-line" />
      <line :x1="padL" :y1="padT" :x2="padL" :y2="H - padB" class="axis-line" />

      <template v-for="tick in yTicks" :key="`y-${tick.value}`">
        <line :x1="padL" :y1="tick.y" :x2="W - padR" :y2="tick.y" class="grid-line" />
        <text :x="padL - 4" :y="tick.y + 3" class="tick-label" text-anchor="end">{{ tick.label }}</text>
      </template>

      <template v-for="range in plottedRanges" :key="`range-${range.key}`">
        <line :x1="range.x" :y1="range.y1" :x2="range.x" :y2="range.y2" :stroke="range.color" stroke-width="3" stroke-linecap="round" opacity="0.38" />
      </template>

      <template v-for="series in plottedSeries" :key="`area-${series.label}`">
        <path v-if="series.areaPath" :d="series.areaPath" :fill="series.color" fill-opacity="0.09" stroke="none" />
      </template>

      <template v-for="series in plottedSeries" :key="series.label">
        <path :d="series.path" :stroke="series.color" fill="none" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" />
        <circle v-for="point in series.points" :key="`${series.label}-${point.key}`" :cx="point.x" :cy="point.y" r="3.2" :fill="series.color" />
      </template>

      <template v-if="showPointLabels">
        <template v-for="series in plottedSeries" :key="`lbl-${series.label}`">
          <text
            v-for="point in series.points"
            :key="`lbl-${point.key}`"
            :x="point.x"
            :y="point.y - 7"
            :fill="series.color"
            class="point-label"
            text-anchor="middle"
          >{{ point.rawValue < 10 ? point.rawValue.toFixed(2) : Math.round(point.rawValue) }}</text>
        </template>
      </template>

      <template v-for="marker in plottedMarkers" :key="`marker-${marker.key}`">
        <line :x1="marker.x" :y1="padT" :x2="marker.x" :y2="H - padB" :stroke="marker.color" stroke-width="1.2" stroke-dasharray="4 3" />
        <circle v-if="marker.y !== null" :cx="marker.x" :cy="marker.y" r="4" :fill="marker.color" stroke="#f0ede4" stroke-width="1.2" />
      </template>

      <template v-for="anomaly in plottedAnomalies" :key="`anomaly-${anomaly.key}`">
        <line :x1="anomaly.x" :y1="padT + 4" :x2="anomaly.x" :y2="H - padB" :stroke="anomaly.color" stroke-width="1" stroke-dasharray="2 2" opacity="0.55" />
        <rect :x="anomaly.x - 3" :y="padT" width="6" height="6" :fill="anomaly.color" />
      </template>

      <template v-for="tick in xTicks" :key="`x-${tick.key}`">
        <text :x="tick.x" :y="H - 4" class="tick-label" text-anchor="middle">{{ tick.label }}</text>
      </template>
    </svg>
    <div v-else class="xy-chart-empty">{{ emptyText }}</div>

    <div v-if="legendEntries.length" class="xy-chart-legend">
      <span v-for="entry in legendEntries" :key="entry.label" class="legend-entry">
        <i class="legend-swatch" :style="{ background: entry.color }"></i>
        {{ entry.label }}
      </span>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { getUiLocale } from '../utils/presentation'

const W = 320
const H = 170
const padL = 34
const padR = 12
const padT = 14
const padB = 24

const props = defineProps({
  series: { type: Array, default: () => [] },
  markers: { type: Array, default: () => [] },
  anomalies: { type: Array, default: () => [] },
  ranges: { type: Array, default: () => [] },
  emptyText: { type: String, default: 'Нет данных' },
})

function parsePointKey(raw) {
  if (raw === null || raw === undefined || raw === '') return null
  if (raw instanceof Date && !Number.isNaN(raw.getTime())) {
    return { type: 'time', value: raw.getTime() }
  }
  if (typeof raw === 'number' && Number.isFinite(raw)) {
    return { type: 'number', value: raw }
  }
  const text = String(raw).trim()
  if (!text) return null
  if (/^\d{4}-\d{2}-\d{2}/.test(text) || text.includes('T') || /^\d{2}\/\d{2}\/\d{4}/.test(text)) {
    const parsed = new Date(text)
    if (!Number.isNaN(parsed.getTime())) {
      return { type: 'time', value: parsed.getTime() }
    }
  }
  if (/^-?\d+(\.\d+)?$/.test(text)) {
    return { type: 'number', value: Number(text) }
  }
  const parsed = new Date(text)
  if (!Number.isNaN(parsed.getTime())) {
    return { type: 'time', value: parsed.getTime() }
  }
  return { type: 'category', value: text }
}

function domainIdentity(point) {
  return `${point.type}:${String(point.value)}`
}

const xDomain = computed(() => {
  const entries = new Map()
  for (const series of props.series || []) {
    for (const point of series.points || []) {
      const parsed = parsePointKey(point.date ?? point.x ?? point.label)
      if (parsed) entries.set(domainIdentity(parsed), parsed)
    }
  }
  for (const marker of props.markers || []) {
    const parsed = parsePointKey(marker.date ?? marker.x ?? marker.label)
    if (parsed) entries.set(domainIdentity(parsed), parsed)
  }
  for (const anomaly of props.anomalies || []) {
    const parsed = parsePointKey(anomaly.observed_at ?? anomaly.date)
    if (parsed) entries.set(domainIdentity(parsed), parsed)
  }
  for (const range of props.ranges || []) {
    const parsed = parsePointKey(range.date ?? range.x ?? range.label)
    if (parsed) entries.set(domainIdentity(parsed), parsed)
  }
  const domain = [...entries.values()]
  const scaleType = domain.length && domain.every((item) => item.type === 'time')
    ? 'time'
    : domain.length && domain.every((item) => item.type === 'number')
      ? 'number'
      : 'category'
  domain.sort((a, b) => {
    if (scaleType !== 'category') return Number(a.value) - Number(b.value)
    return String(a.value).localeCompare(String(b.value))
  })
  return {
    entries: domain,
    scaleType,
  }
})

function toX(point) {
  const parsed = point?.type ? point : parsePointKey(point)
  const { entries, scaleType } = xDomain.value
  const width = W - padL - padR
  if (!parsed || !entries.length) return padL + width / 2
  if (entries.length === 1) return padL + width / 2
  if (scaleType !== 'category') {
    const min = Number(entries[0].value)
    const max = Number(entries[entries.length - 1].value)
    const span = Math.max(max - min, 0.001)
    return padL + ((Number(parsed.value) - min) / span) * width
  }
  const index = Math.max(0, entries.findIndex((item) => domainIdentity(item) === domainIdentity(parsed)))
  const span = Math.max(entries.length - 1, 1)
  return padL + (index / span) * width
}

const yBounds = computed(() => {
  const values = []
  for (const series of props.series || []) {
    for (const point of series.points || []) {
      const value = Number(point.value)
      if (Number.isFinite(value)) values.push(value)
    }
  }
  for (const range of props.ranges || []) {
    const lower = Number(range.lower)
    const upper = Number(range.upper)
    if (Number.isFinite(lower)) values.push(lower)
    if (Number.isFinite(upper)) values.push(upper)
  }
  for (const marker of props.markers || []) {
    const markerValue = Number(marker.value)
    if (Number.isFinite(markerValue)) values.push(markerValue)
  }
  if (!values.length) return { min: 0, max: 1 }
  const min = Math.min(...values)
  const max = Math.max(...values)
  if (Math.abs(max - min) < 1e-9) {
    const padding = Math.max(Math.abs(max) * 0.2, 0.1)
    return {
      min: min - padding,
      max: max + padding,
    }
  }
  const span = Math.max(max - min, 0.001)
  return {
    min: min - span * 0.08,
    max: max + span * 0.08,
  }
})

function toY(value) {
  const { min, max } = yBounds.value
  const span = Math.max(max - min, 0.001)
  return H - padB - ((value - min) / span) * (H - padB - padT)
}

const plottedSeries = computed(() => {
  return (props.series || [])
    .map((series) => {
      const points = (series.points || [])
        .map((point) => {
          const key = parsePointKey(point.date ?? point.x ?? point.label)
          const value = Number(point.value)
          if (key === null || !Number.isFinite(value)) return null
          return {
            key: domainIdentity(key),
            x: toX(key),
            y: toY(value),
            rawValue: value,
          }
        })
        .filter(Boolean)
      if (!points.length) return null
      const linePath = points
        .map((point, index) => `${index === 0 ? 'M' : 'L'} ${point.x.toFixed(1)} ${point.y.toFixed(1)}`)
        .join(' ')
      const areaPath = points.length >= 2
        ? `${linePath} L ${points[points.length - 1].x.toFixed(1)} ${(H - padB).toFixed(1)} L ${points[0].x.toFixed(1)} ${(H - padB).toFixed(1)} Z`
        : null
      return {
        label: series.label || 'Series',
        color: series.color || '#21579c',
        points,
        path: linePath,
        areaPath,
      }
    })
    .filter(Boolean)
})

const plottedMarkers = computed(() => {
  return (props.markers || [])
    .map((marker, index) => {
      const key = parsePointKey(marker.date ?? marker.x ?? marker.label)
      if (key === null) return null
      const value = Number(marker.value)
      return {
        key: `${domainIdentity(key)}-${index}`,
        x: toX(key),
        y: Number.isFinite(value) ? toY(value) : null,
        color: marker.color || '#2f6b97',
      }
    })
    .filter(Boolean)
})

const plottedRanges = computed(() => {
  return (props.ranges || [])
    .map((range, index) => {
      const key = parsePointKey(range.date ?? range.x ?? range.label)
      const lower = Number(range.lower)
      const upper = Number(range.upper)
      if (key === null || !Number.isFinite(lower) || !Number.isFinite(upper)) return null
      return {
        key: `${domainIdentity(key)}-${index}`,
        x: toX(key),
        y1: toY(lower),
        y2: toY(upper),
        color: range.color || '#5d86b9',
      }
    })
    .filter(Boolean)
})

const plottedAnomalies = computed(() => {
  return (props.anomalies || [])
    .map((item, index) => {
      const key = parsePointKey(item.observed_at ?? item.date)
      if (key === null) return null
      const severity = String(item.severity || '')
      const color = severity === 'critical' ? '#b43f2d' : severity === 'warning' ? '#cc8b19' : '#5d6fb3'
      return {
        key: `${domainIdentity(key)}-${index}`,
        x: toX(key),
        color,
      }
    })
    .filter(Boolean)
})

const xTicks = computed(() => {
  const { entries, scaleType } = xDomain.value
  if (!entries.length) return []
  const step = Math.max(1, Math.ceil(entries.length / 6))
  return entries
    .filter((_, index) => index % step === 0 || index === entries.length - 1)
    .map((entry) => {
      let label = String(entry.value)
      if (scaleType === 'time') {
        const parsed = new Date(Number(entry.value))
        label = parsed.toLocaleDateString(getUiLocale(), { month: '2-digit', day: '2-digit' })
      } else if (scaleType === 'number') {
        const numeric = Number(entry.value)
        label = Number.isInteger(numeric) ? String(numeric) : numeric.toFixed(Math.abs(numeric) < 10 ? 1 : 0)
      }
      return { key: domainIdentity(entry), x: toX(entry), label }
    })
})

const yTicks = computed(() => {
  const { min, max } = yBounds.value
  const step = (max - min) / 4
  return new Array(5).fill(null).map((_, index) => {
    const value = min + step * index
    return {
      value,
      y: toY(value),
      label: Number.isFinite(value) ? value.toFixed(Math.abs(value) < 10 ? 2 : 0) : '—',
    }
  })
})

const legendEntries = computed(() => {
  const entries = (props.series || []).map((series) => ({
    label: series.label,
    color: series.color || '#21579c',
  }))
  for (const marker of props.markers || []) {
    if (marker.label) {
      entries.push({ label: marker.label, color: marker.color || '#2f6b97' })
    }
  }
  return entries
})

const plottedPointCount = computed(() => {
  return plottedSeries.value.reduce((total, series) => total + series.points.length, 0)
})

const showPointLabels = computed(() => plottedPointCount.value <= 5)

const hasDrawableContent = computed(() => {
  return plottedSeries.value.some((series) => (series.points || []).length >= 2)
})
</script>

<style scoped>
.xy-chart-shell {
  display: grid;
  gap: 8px;
}

.xy-chart-svg {
  width: 100%;
  height: 170px;
  display: block;
  background: rgba(255, 255, 255, 0.36);
}

.point-label {
  font-size: 9px;
  font-family: inherit;
  font-weight: 600;
  pointer-events: none;
}

.xy-chart-empty {
  min-height: 120px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--text-muted);
  font-size: 12px;
}

.axis-line {
  stroke: rgba(33, 57, 87, 0.5);
  stroke-width: 1;
}

.grid-line {
  stroke: rgba(33, 57, 87, 0.12);
  stroke-width: 1;
}

.tick-label {
  fill: rgba(33, 57, 87, 0.72);
  font-size: 8px;
  font-family: inherit;
}

.xy-chart-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  font-size: 12px;
  color: var(--text-muted);
}

.legend-entry {
  display: inline-flex;
  align-items: center;
  gap: 4px;
}

.legend-swatch {
  width: 10px;
  height: 10px;
  display: inline-block;
}
</style>
