<template>
  <section class="window-shell legend-panel">
    <div class="window-title">{{ t('legend.title') }}</div>
    <div class="window-body">
      <div class="legend-name">{{ layerLabel }}</div>
      <div class="legend-bar" :style="{ background: gradientCss(store.primaryLayerId) }"></div>
      <div class="legend-scale">
        <span>{{ minLabel }}</span>
        <span>{{ maxLabel }}</span>
      </div>
    </div>
  </section>
</template>

<script setup>
import { computed } from 'vue'
import { useMapStore } from '../store/map'
import { getGradientForLayer, gradientCss } from '../utils/gradients'
import { t } from '../utils/i18n'
import { getLayerMeta } from '../utils/presentation'

const store = useMapStore()

const layerLabel = computed(() => {
  const layer = store.availableLayers.find((item) => item.id === store.primaryLayerId)
  return getLayerMeta(store.primaryLayerId, layer || {}).label
})

const minLabel = computed(() => {
  const gradient = getGradientForLayer(store.primaryLayerId)
  return gradient.format(gradient.range[0])
})

const maxLabel = computed(() => {
  const gradient = getGradientForLayer(store.primaryLayerId)
  return gradient.format(gradient.range[1])
})
</script>

<style scoped>
.legend-panel {
  min-height: 120px;
}

.legend-name {
  font-weight: 700;
  margin-bottom: 8px;
  color: var(--text-main);
}

.legend-bar {
  height: 18px;
  border: 2px solid;
  border-color: var(--win-shadow-mid) var(--win-shadow-light) var(--win-shadow-light) var(--win-shadow-mid);
}

.legend-scale {
  margin-top: 6px;
  display: flex;
  justify-content: space-between;
  color: var(--text-muted);
  font-size: 12px;
}
</style>
