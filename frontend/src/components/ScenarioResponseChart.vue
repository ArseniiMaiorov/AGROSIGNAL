<template>
  <div class="response-chart-shell">
    <svg v-if="curvePoints.length >= 2" :viewBox="`0 0 ${W} ${H}`" preserveAspectRatio="none" class="response-chart-svg">
      <!-- Baseline dashed line -->
      <line
        :x1="pad" :y1="baselineY" :x2="W - pad" :y2="baselineY"
        stroke="#888" stroke-width="1" stroke-dasharray="4 3"
      />

      <!-- Response curve area fill -->
      <path :d="areaPath" :fill="areaFill" />

      <!-- Response curve line -->
      <path :d="curvePath" stroke="#21579c" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />

      <!-- Current value vertical marker -->
      <line
        v-if="currentMarkerX !== null"
        :x1="currentMarkerX" :y1="pad" :x2="currentMarkerX" :y2="H - padBottom"
        stroke="#c04040" stroke-width="1.5" stroke-dasharray="3 2"
      />
      <circle
        v-if="currentMarkerX !== null && currentMarkerY !== null"
        :cx="currentMarkerX" :cy="currentMarkerY"
        r="3.5" fill="#c04040" stroke="#fff" stroke-width="1"
      />

      <!-- Axis labels -->
      <text :x="pad" :y="H - 2" class="axis-label">{{ axisMin }}</text>
      <text :x="W - pad" :y="H - 2" class="axis-label" text-anchor="end">{{ axisMax }}</text>
      <text :x="W / 2" :y="H - 2" class="axis-label" text-anchor="middle">{{ paramLabel }}</text>

      <!-- Yield labels on Y axis -->
      <text :x="pad - 2" :y="baselineY - 3" class="axis-label" text-anchor="start">{{ baselineLabel }}</text>
    </svg>
    <div v-else class="response-chart-empty">{{ emptyText }}</div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const W = 260
const H = 100
const pad = 10
const padBottom = 18

const props = defineProps({
  points: { type: Array, default: () => [] },
  baselineYield: { type: Number, default: 0 },
  currentValue: { type: Number, default: null },
  paramLabel: { type: String, default: '' },
  emptyText: { type: String, default: 'Нет данных' },
})

const sortedPoints = computed(() => {
  return [...(props.points || [])].sort((a, b) => a.param_value - b.param_value)
})

const yieldValues = computed(() => sortedPoints.value.map((p) => p.yield_kg_ha))

const yMin = computed(() => {
  const values = [...yieldValues.value, props.baselineYield].filter(Number.isFinite)
  return values.length ? Math.min(...values) * 0.95 : 0
})

const yMax = computed(() => {
  const values = [...yieldValues.value, props.baselineYield].filter(Number.isFinite)
  return values.length ? Math.max(...values) * 1.05 : 1
})

const xMin = computed(() => sortedPoints.value.length ? sortedPoints.value[0].param_value : 0)
const xMax = computed(() => sortedPoints.value.length ? sortedPoints.value[sortedPoints.value.length - 1].param_value : 1)

function toSvgX(paramVal) {
  const range = Math.max(xMax.value - xMin.value, 0.001)
  return pad + ((paramVal - xMin.value) / range) * (W - 2 * pad)
}

function toSvgY(yieldVal) {
  const range = Math.max(yMax.value - yMin.value, 0.001)
  return (H - padBottom) - ((yieldVal - yMin.value) / range) * (H - padBottom - pad)
}

const curvePoints = computed(() => {
  return sortedPoints.value.map((p) => ({
    x: toSvgX(p.param_value),
    y: toSvgY(p.yield_kg_ha),
  }))
})

const curvePath = computed(() => {
  if (curvePoints.value.length < 2) return ''
  return curvePoints.value
    .map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`)
    .join(' ')
})

const areaPath = computed(() => {
  if (curvePoints.value.length < 2) return ''
  const first = curvePoints.value[0]
  const last = curvePoints.value[curvePoints.value.length - 1]
  const bottom = H - padBottom
  return `${curvePath.value} L ${last.x.toFixed(1)} ${bottom} L ${first.x.toFixed(1)} ${bottom} Z`
})

const areaFill = '#21579c18'

const baselineY = computed(() => toSvgY(props.baselineYield))

const currentMarkerX = computed(() => {
  if (props.currentValue === null || props.currentValue === undefined) return null
  return toSvgX(props.currentValue)
})

const currentMarkerY = computed(() => {
  if (currentMarkerX.value === null || !sortedPoints.value.length) return null
  // Interpolate yield at currentValue
  const pts = sortedPoints.value
  const cv = props.currentValue
  if (cv <= pts[0].param_value) return toSvgY(pts[0].yield_kg_ha)
  if (cv >= pts[pts.length - 1].param_value) return toSvgY(pts[pts.length - 1].yield_kg_ha)
  for (let i = 0; i < pts.length - 1; i++) {
    if (cv >= pts[i].param_value && cv <= pts[i + 1].param_value) {
      const t = (cv - pts[i].param_value) / Math.max(pts[i + 1].param_value - pts[i].param_value, 0.001)
      const y = pts[i].yield_kg_ha + t * (pts[i + 1].yield_kg_ha - pts[i].yield_kg_ha)
      return toSvgY(y)
    }
  }
  return null
})

const baselineLabel = computed(() => {
  if (!Number.isFinite(props.baselineYield)) return ''
  return `${(props.baselineYield / 100).toFixed(0)} ц/га`
})

const axisMin = computed(() => {
  if (!sortedPoints.value.length) return ''
  return `${sortedPoints.value[0].param_value}`
})

const axisMax = computed(() => {
  if (!sortedPoints.value.length) return ''
  return `${sortedPoints.value[sortedPoints.value.length - 1].param_value}`
})
</script>

<style scoped>
.response-chart-shell {
  width: 100%;
  min-height: 80px;
}

.response-chart-svg {
  width: 100%;
  height: 100px;
  display: block;
}

.response-chart-empty {
  min-height: 80px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--text-muted);
  font-size: 12px;
}

.axis-label {
  font-size: 8px;
  fill: #888;
  font-family: inherit;
}
</style>
