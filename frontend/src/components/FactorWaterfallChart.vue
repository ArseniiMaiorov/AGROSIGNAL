<template>
  <div v-if="rows.length" class="waterfall-shell">
    <div class="waterfall-axis">
      <span>{{ minLabel }}</span>
      <span>0</span>
      <span>{{ maxLabel }}</span>
    </div>
    <div v-for="row in rows" :key="row.key" class="waterfall-row">
      <div class="waterfall-label" :title="row.description || undefined">{{ row.label }}</div>
      <div class="waterfall-track">
        <div class="waterfall-zero"></div>
        <div class="waterfall-bar" :class="row.direction" :style="row.style"></div>
      </div>
      <div class="waterfall-value">{{ row.value }}</div>
    </div>
  </div>
  <div v-else class="waterfall-empty">{{ emptyText }}</div>
</template>

<script setup>
import { computed } from 'vue'
import { getFeatureLabel } from '../utils/presentation'

const props = defineProps({
  factors: { type: Array, default: () => [] },
  emptyText: { type: String, default: 'Нет данных' },
})

const FACTOR_DESCRIPTIONS = {
  moisture_stress: 'Потери урожая от дефицита или избытка влаги в почве',
  water_stress: 'Потери урожая от дефицита или избытка влаги в почве',
  hydro_stress: 'Потери урожая из-за суммарного водного стресса: засуха, переувлажнение, слабая инфильтрация',
  vegetation_signal: 'Вегетационный индекс (NDVI): активность роста культуры',
  ndvi_signal: 'Вегетационный индекс (NDVI): активность роста культуры',
  wind_stress: 'Механические потери от сильного ветра: полегание, испарение',
  field_shape: 'Потери на краевых эффектах из-за формы и компактности поля',
  shape: 'Потери на краевых эффектах из-за формы и компактности поля',
  area_shape: 'Потери на краевых эффектах из-за формы и компактности поля',
  climate_suitability: 'Соответствие климата оптимуму для данной культуры',
  management: 'Вклад агрономических мероприятий (полив, удобрения) из истории поля',
  scenario_irrigation: 'Эффект сценарного орошения относительно базового прогноза',
  scenario_fertilizer: 'Эффект сценарного внесения удобрений относительно базового прогноза',
  scenario_rain: 'Эффект ожидаемых осадков с учётом кросс-факторных взаимодействий',
  scenario_rainfall: 'Эффект ожидаемых осадков с учётом кросс-факторных взаимодействий и риска переувлажнения',
  scenario_compaction: 'Эффект уплотнения почвы: снижение инфильтрации, аэрации и роста корней',
  soil_profile: 'Влияние механического состава и структуры почвы',
  solar_radiation: 'Влияние солнечной радиации (ERA5) на фотосинтез культуры',
  temperature: 'Эффект температурного режима на скорость развития культуры',
}

const normalized = computed(() => {
  return (props.factors || [])
    .map((factor, index) => ({
      key: `${factor.driver_id || factor.input_key || factor.factor || factor.label || 'factor'}-${index}`,
      label: getFeatureLabel(factor.input_key || factor.driver_id || factor.factor || factor.label) || factor.label || factor.factor || 'Фактор',
      effect: Number(factor.effect_kg_ha ?? factor.effect_pct ?? 0),
      description: FACTOR_DESCRIPTIONS[factor.driver_id || factor.input_key || factor.factor || ''] || '',
      raw: factor,
    }))
    .filter((factor) => Number.isFinite(factor.effect))
    .filter((factor) => Math.abs(factor.effect) >= 5)
})

const extent = computed(() => {
  const maxAbs = Math.max(1, ...normalized.value.map((item) => Math.abs(item.effect)))
  return maxAbs
})

const rows = computed(() => {
  // Sort: negatives first (baseline constraints), then positives (scenario gains)
  const sorted = [...normalized.value].sort((a, b) => a.effect - b.effect)
  return sorted.map((factor) => {
    const widthPct = Math.min(48, (Math.abs(factor.effect) / extent.value) * 48)
    const isPositive = factor.effect >= 0
    return {
      ...factor,
      direction: isPositive ? 'positive' : 'negative',
      style: isPositive
        ? { left: '50%', width: `${widthPct}%` }
        : { left: `${50 - widthPct}%`, width: `${widthPct}%` },
      value: factor.raw.effect_kg_ha !== null && factor.raw.effect_kg_ha !== undefined
        ? `${factor.effect >= 0 ? '+' : ''}${Number(factor.raw.effect_kg_ha).toFixed(0)} кг/га`
        : `${factor.effect >= 0 ? '+' : ''}${Number(factor.raw.effect_pct || 0).toFixed(2)}%`,
    }
  })
})

const minLabel = computed(() => `-${extent.value.toFixed(0)}`)
const maxLabel = computed(() => `+${extent.value.toFixed(0)}`)
</script>

<style scoped>
.waterfall-shell {
  display: grid;
  gap: 8px;
}

.waterfall-axis {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  font-size: 11px;
  color: var(--text-muted);
}

.waterfall-axis span:nth-child(2) {
  text-align: center;
}

.waterfall-axis span:last-child {
  text-align: right;
}

.waterfall-row {
  display: grid;
  grid-template-columns: minmax(120px, 1fr) 1.6fr minmax(96px, auto);
  gap: 8px;
  align-items: center;
}

.waterfall-label,
.waterfall-value {
  font-size: 12px;
}

.waterfall-track {
  position: relative;
  height: 18px;
  border: 1px solid rgba(33, 57, 87, 0.16);
  background: rgba(255, 255, 255, 0.34);
}

.waterfall-zero {
  position: absolute;
  left: 50%;
  top: 0;
  bottom: 0;
  width: 1px;
  background: rgba(33, 57, 87, 0.45);
}

.waterfall-bar {
  position: absolute;
  top: 2px;
  bottom: 2px;
}

.waterfall-bar.positive {
  background: rgba(46, 143, 89, 0.84);
}

.waterfall-bar.negative {
  background: rgba(184, 73, 41, 0.82);
}

.waterfall-empty {
  min-height: 80px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--text-muted);
  font-size: 12px;
}
</style>
