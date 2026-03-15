<template>
  <div v-if="rows.length" class="timeline-shell">
    <div v-for="row in rows" :key="row.key" class="timeline-row">
      <strong>{{ row.label }}</strong>
      <span>{{ row.value }}</span>
    </div>
  </div>
  <div v-else class="timeline-empty">{{ emptyText }}</div>
</template>

<script setup>
import { computed } from 'vue'
import { getUiLocale } from '../utils/presentation'

const props = defineProps({
  points: { type: Array, default: () => [] },
  formatter: { type: Function, default: null },
  emptyText: { type: String, default: 'Нет данных' },
})

const rows = computed(() => {
  return (props.points || []).map((point, index) => {
    const observedAt = point.observed_at || point.date || point.x || ''
    const parsed = new Date(observedAt)
    const label = !Number.isNaN(parsed.getTime())
      ? parsed.toLocaleDateString(getUiLocale(), { day: '2-digit', month: '2-digit' })
      : String(observedAt || index + 1)
    const value = props.formatter ? props.formatter(point.value, point) : String(point.value ?? '—')
    return {
      key: `${observedAt}-${index}`,
      label,
      value,
    }
  })
})
</script>

<style scoped>
.timeline-shell {
  display: grid;
  gap: 4px;
  max-height: 180px;
  overflow: auto;
  padding-right: 4px;
}

.timeline-row {
  display: flex;
  justify-content: space-between;
  gap: 8px;
  padding: 4px 6px;
  border: 1px solid rgba(33, 57, 87, 0.16);
  background: rgba(255, 255, 255, 0.34);
  font-size: 12px;
}

.timeline-empty {
  min-height: 80px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--text-muted);
  font-size: 12px;
}
</style>
