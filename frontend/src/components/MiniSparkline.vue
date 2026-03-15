<template>
  <div class="sparkline-shell">
    <svg v-if="pathData" viewBox="0 0 100 32" preserveAspectRatio="none" class="sparkline-svg">
      <path :d="areaData" :fill="fillColor" />
      <path :d="pathData" :stroke="color" fill="none" stroke-width="2.2" stroke-linecap="round" />
    </svg>
    <div v-else class="sparkline-empty">Нет ряда</div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  points: {
    type: Array,
    default: () => [],
  },
  color: {
    type: String,
    default: '#21579c',
  },
})

const normalizedPoints = computed(() => {
  const values = (props.points || [])
    .map((item) => Number(item?.mean))
    .filter((value) => Number.isFinite(value))
  if (!values.length) {
    return []
  }
  const min = Math.min(...values)
  const max = Math.max(...values)
  const isConstant = min === max
  const span = isConstant ? 1 : Math.max(max - min, 0.0001)
  if (values.length === 1) {
    // Draw a short flat segment instead of a single invisible move command.
    return [
      { x: 8, y: 16 },
      { x: 92, y: 16 },
    ]
  }
  return values.map((value, index) => ({
    x: (index / (values.length - 1)) * 100,
    // Constant series: draw line at middle (y=16) instead of bottom (y=28)
    y: isConstant ? 16 : 28 - ((value - min) / span) * 24,
  }))
})

const pathData = computed(() => {
  if (!normalizedPoints.value.length) {
    return ''
  }
  return normalizedPoints.value
    .map((point, index) => `${index === 0 ? 'M' : 'L'} ${point.x.toFixed(2)} ${point.y.toFixed(2)}`)
    .join(' ')
})

const areaData = computed(() => {
  if (!normalizedPoints.value.length) {
    return ''
  }
  const first = normalizedPoints.value[0]
  const last = normalizedPoints.value[normalizedPoints.value.length - 1]
  return `${pathData.value} L ${last.x.toFixed(2)} 31 L ${first.x.toFixed(2)} 31 Z`
})

const fillColor = computed(() => `${props.color}22`)
</script>

<style scoped>
.sparkline-shell {
  width: 100%;
  min-height: 40px;
}

.sparkline-svg {
  width: 100%;
  height: 42px;
  display: block;
}

.sparkline-empty {
  min-height: 42px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--text-muted);
  font-size: 12px;
}
</style>
