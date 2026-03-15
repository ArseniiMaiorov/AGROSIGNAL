<template>
  <section class="window-shell weather-panel" data-testid="weather-panel" :class="{ 'weather-bar': props.overlay }">
    <div class="window-title">{{ t('weather.title') }}</div>
    <div class="window-body">
      <div v-if="store.weatherCurrent" class="weather-current">
        <div class="weather-main">
          <div class="weather-temp">{{ formatTemp(store.weatherCurrent.temperature_c) }}</div>
          <div class="weather-meta">
            <div>{{ t('weather.feelsLike') }}: {{ formatTemp(store.weatherCurrent.apparent_temperature_c) }}</div>
            <div>{{ t('weather.wind') }}: {{ formatWind(store.weatherCurrent.wind_speed_m_s, store.weatherCurrent.wind_direction_deg) }}</div>
            <div>{{ t('weather.precipitation') }}: {{ formatMm(store.weatherCurrent.precipitation_mm) }}</div>
          </div>
        </div>
        <FreshnessBadge :meta="store.weatherCurrent.freshness" :compact="props.overlay" />
        <div v-if="store.weatherCurrent.error" class="weather-note">{{ store.weatherCurrent.error }}</div>
        <div class="weather-grid">
          <div class="weather-cell">
            <span>{{ t('weather.humidity') }}</span>
            <strong>{{ formatPct(store.weatherCurrent.humidity_pct) }}</strong>
          </div>
          <div class="weather-cell">
            <span>{{ t('weather.cloudCover') }}</span>
            <strong>{{ formatPct(store.weatherCurrent.cloud_cover_pct) }}</strong>
          </div>
          <div class="weather-cell">
            <span>{{ t('weather.pressure') }}</span>
            <strong>{{ formatPressure(store.weatherCurrent.pressure_hpa) }}</strong>
          </div>
          <div class="weather-cell">
            <span>{{ t('weather.soil') }}</span>
            <strong>{{ formatSoil(store.weatherCurrent.soil_moisture) }}</strong>
          </div>
        </div>
      </div>
      <div v-else class="placeholder">{{ t('weather.loading') }}</div>

      <div class="forecast-strip">
        <div v-for="day in visibleForecast" :key="day.date" class="forecast-card">
          <div class="forecast-date">{{ formatDate(day.date) }}</div>
          <div class="forecast-temp-range">{{ formatTemp(day.temp_min_c) }} / {{ formatTemp(day.temp_max_c) }}</div>
          <div class="forecast-extra">{{ formatMm(day.precipitation_mm) }}</div>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup>
import { computed } from 'vue'
import FreshnessBadge from './FreshnessBadge.vue'

import { useMapStore } from '../store/map'
import { locale, t } from '../utils/i18n'

const props = defineProps({
  overlay: {
    type: Boolean,
    default: false,
  },
})

const store = useMapStore()
const visibleForecast = computed(() => (props.overlay ? store.weatherForecast.slice(0, 3) : store.weatherForecast))

const COMPASS_POINTS = {
  ru: ['С', 'ССВ', 'СВ', 'ВСВ', 'В', 'ВЮВ', 'ЮВ', 'ЮЮВ', 'Ю', 'ЮЮЗ', 'ЮЗ', 'ЗЮЗ', 'З', 'ЗСЗ', 'СЗ', 'ССЗ'],
  en: ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'],
}

function formatTemp(value) {
  return value === null || value === undefined ? '—' : `${Number(value).toFixed(1)} °C`
}

function formatCompass(direction) {
  const numeric = Number(direction)
  if (!Number.isFinite(numeric)) return null
  const points = COMPASS_POINTS[locale.value] || COMPASS_POINTS.en
  const normalized = ((numeric % 360) + 360) % 360
  const index = Math.round(normalized / 22.5) % points.length
  return points[index]
}

function formatWind(value, direction) {
  if (value === null || value === undefined) return '—'
  const compass = formatCompass(direction)
  if (!compass) return `${Number(value).toFixed(1)} м/с`
  return `${Number(value).toFixed(1)} м/с · ${compass}`
}

function formatMm(value) {
  return value === null || value === undefined ? '—' : `${Number(value).toFixed(1)} мм`
}

function formatPct(value) {
  return value === null || value === undefined ? '—' : `${Number(value).toFixed(0)}%`
}

function formatPressure(value) {
  return value === null || value === undefined ? '—' : `${Number(value).toFixed(0)} гПа`
}

function formatSoil(value) {
  return value === null || value === undefined ? '—' : `${(Number(value) * 100).toFixed(0)}%`
}

function formatDate(value) {
  return new Date(value).toLocaleDateString('ru-RU', { day: '2-digit', month: '2-digit' })
}
</script>

<style scoped>
.weather-panel {
  min-height: 0;
}

.weather-bar {
  border-left: none;
  border-right: none;
  box-shadow: none;
}

.weather-bar :deep(.window-title) {
  padding: 4px 10px;
  font-size: 12px;
}

.weather-bar .weather-current {
  flex-direction: row;
  align-items: center;
  gap: 14px;
}

.weather-bar .weather-grid {
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 4px;
}

.weather-bar .forecast-strip {
  grid-template-columns: repeat(3, minmax(86px, 1fr));
  margin-top: 4px;
}

.weather-bar :deep(.window-body) {
  padding: 6px 10px;
}

.weather-bar .weather-main {
  min-width: 220px;
  gap: 12px;
}

.weather-bar .weather-temp {
  font-size: 22px;
}

.weather-bar .weather-meta {
  gap: 1px;
  font-size: 11px;
}

.weather-bar .weather-cell,
.weather-bar .forecast-card {
  padding: 4px 6px;
}

.weather-bar .weather-cell span,
.weather-bar .forecast-extra,
.weather-bar .forecast-temp-range,
.weather-bar .forecast-date {
  font-size: 10px;
}

.weather-bar .weather-cell strong {
  font-size: 15px;
}

.weather-current {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.weather-main {
  display: flex;
  gap: 16px;
  align-items: center;
}

.weather-temp {
  font-size: 28px;
  line-height: 1;
  color: var(--text-main);
  font-weight: 700;
}

.weather-meta {
  display: flex;
  flex-direction: column;
  gap: 2px;
  color: var(--text-main);
  font-size: 12px;
}

.weather-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 6px;
}

.weather-note {
  color: #8a5a17;
  font-size: 12px;
}

.weather-cell,
.forecast-card {
  border: 2px solid;
  border-color: var(--win-shadow-mid) var(--win-shadow-light) var(--win-shadow-light) var(--win-shadow-mid);
  background: var(--weather-cell-bg);
  padding: 6px 8px;
}

.weather-cell span {
  display: block;
  color: var(--text-muted);
  font-size: 11px;
  margin-bottom: 1px;
}

.forecast-strip {
  margin-top: 8px;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(80px, 1fr));
  gap: 6px;
}

.forecast-date {
  font-weight: 700;
  margin-bottom: 4px;
  font-size: 12px;
}

.forecast-temp-range {
  font-size: 12px;
}

.forecast-extra {
  color: var(--text-muted);
  margin-top: 2px;
  font-size: 11px;
}

@media (max-width: 900px) {
  .weather-bar .weather-current {
    flex-direction: column;
  }
  .weather-bar .weather-grid {
    grid-template-columns: repeat(2, 1fr);
  }
  .weather-bar .forecast-strip {
    grid-template-columns: repeat(3, 1fr);
  }
}
</style>
