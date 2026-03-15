<template>
  <div v-if="bars.length" class="histogram-shell">
    <div class="bars">
      <div
        v-for="bar in bars"
        :key="bar.index"
        class="bar"
        :style="{ height: `${bar.height}%`, background: color }"
        :title="`${bar.label}: ${bar.count}`"
      ></div>
    </div>
    <div class="labels">
      <span>{{ minLabel }}</span>
      <span>{{ maxLabel }}</span>
    </div>
  </div>
  <div v-else class="histogram-empty">Недостаточно данных</div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  histogram: {
    type: Object,
    default: () => ({}),
  },
  color: {
    type: String,
    default: '#2f8a63',
  },
})

const bars = computed(() => {
  const counts = Array.isArray(props.histogram?.counts) ? props.histogram.counts : []
  const bins = Array.isArray(props.histogram?.bins) ? props.histogram.bins : []
  if (!counts.length || bins.length < 2) {
    return []
  }
  const max = Math.max(...counts, 1)
  return counts.map((count, index) => ({
    index,
    count,
    label: `${Number(bins[index]).toFixed(2)} … ${Number(bins[index + 1]).toFixed(2)}`,
    height: Math.max(8, Math.round((count / max) * 100)),
  }))
})

const minLabel = computed(() => {
  const bins = Array.isArray(props.histogram?.bins) ? props.histogram.bins : []
  return bins.length ? Number(bins[0]).toFixed(2) : '—'
})

const maxLabel = computed(() => {
  const bins = Array.isArray(props.histogram?.bins) ? props.histogram.bins : []
  return bins.length ? Number(bins[bins.length - 1]).toFixed(2) : '—'
})
</script>

<style scoped>
.histogram-shell {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.bars {
  min-height: 78px;
  display: flex;
  align-items: end;
  gap: 4px;
}

.bar {
  flex: 1;
  min-height: 8px;
  border: 1px solid rgba(0, 0, 0, 0.16);
}

.labels {
  display: flex;
  justify-content: space-between;
  color: var(--text-muted);
  font-size: 11px;
}

.histogram-empty {
  min-height: 78px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--text-muted);
  font-size: 12px;
}
</style>
