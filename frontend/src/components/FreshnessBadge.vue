<template>
  <span
    v-if="store.showFreshnessBadges && meta"
    class="freshness-badge"
    :class="stateClass"
    :title="titleText"
  >
    {{ labelText }}
  </span>
</template>

<script setup>
import { computed } from 'vue'

import { useMapStore } from '../store/map'
import { locale } from '../utils/i18n'

const props = defineProps({
  meta: {
    type: Object,
    default: null,
  },
  compact: {
    type: Boolean,
    default: false,
  },
})

const store = useMapStore()

const stateClass = computed(() => {
  const state = String(props.meta?.freshness_state || 'unknown').toLowerCase()
  if (state === 'fresh') return 'state-fresh'
  if (state === 'stale') return 'state-stale'
  return 'state-unknown'
})

const referenceDate = computed(() => {
  const raw = props.meta?.cache_written_at || props.meta?.fetched_at || props.meta?.source_published_at
  if (!raw) return null
  const parsed = new Date(raw)
  return Number.isNaN(parsed.getTime()) ? null : parsed
})

const ageLabel = computed(() => {
  if (!referenceDate.value) {
    return locale.value === 'ru' ? 'время ?' : 'time ?'
  }
  const diffMs = Math.max(0, Date.now() - referenceDate.value.getTime())
  const diffMinutes = Math.round(diffMs / 60000)
  if (diffMinutes < 1) return locale.value === 'ru' ? 'только что' : 'just now'
  if (diffMinutes < 60) return locale.value === 'ru' ? `${diffMinutes} мин` : `${diffMinutes}m`
  const diffHours = Math.round(diffMinutes / 60)
  if (diffHours < 24) return locale.value === 'ru' ? `${diffHours} ч` : `${diffHours}h`
  const diffDays = Math.round(diffHours / 24)
  return locale.value === 'ru' ? `${diffDays} д` : `${diffDays}d`
})

const stateLabel = computed(() => {
  const state = String(props.meta?.freshness_state || 'unknown').toLowerCase()
  if (locale.value === 'ru') {
    if (state === 'fresh') return 'свежие'
    if (state === 'stale') return 'устарели'
    return null
  }
  if (state === 'fresh') return 'fresh'
  if (state === 'stale') return 'stale'
  return null
})

const providerLabel = computed(() => {
  const raw = props.meta?.provider ? String(props.meta.provider).trim() : ''
  if (!raw) return null
  const token = raw.toLowerCase()
  if (token === 'backend') return locale.value === 'ru' ? 'сервер' : 'server'
  if (token === 'openmeteo' || token === 'open-meteo') return 'Open-Meteo'
  if (token === 'era5') return 'ERA5'
  if (token === 'sentinelhub' || token === 'sentinel-hub') return 'Sentinel Hub'
  return raw.replace(/[_-]+/g, ' ')
})

const labelText = computed(() => {
  const fallbackLabel = locale.value === 'ru' ? 'обновлено' : 'updated'
  if (props.compact) {
    if (stateLabel.value) return `${stateLabel.value} · ${ageLabel.value}`
    if (providerLabel.value) return `${providerLabel.value} · ${ageLabel.value}`
    return `${fallbackLabel} · ${ageLabel.value}`
  }
  if (providerLabel.value && stateLabel.value) {
    return `${providerLabel.value} · ${stateLabel.value} · ${ageLabel.value}`
  }
  if (providerLabel.value) return `${providerLabel.value} · ${ageLabel.value}`
  if (stateLabel.value) return `${stateLabel.value} · ${ageLabel.value}`
  return `${fallbackLabel} · ${ageLabel.value}`
})

const titleText = computed(() => {
  const lines = []
  if (providerLabel.value) lines.push(`provider: ${providerLabel.value}`)
  if (!stateLabel.value) {
    lines.push(locale.value === 'ru' ? 'freshness: статус свежести не указан' : 'freshness: not specified')
  } else {
    lines.push(`freshness: ${stateLabel.value}`)
  }
  if (props.meta?.fetched_at) lines.push(`fetched: ${props.meta.fetched_at}`)
  if (props.meta?.cache_written_at) lines.push(`cache: ${props.meta.cache_written_at}`)
  if (props.meta?.source_published_at) lines.push(`source: ${props.meta.source_published_at}`)
  if (props.meta?.model_version) lines.push(`model: ${props.meta.model_version}`)
  if (props.meta?.dataset_version) lines.push(`dataset: ${props.meta.dataset_version}`)
  return lines.join('\n')
})
</script>

<style scoped>
.freshness-badge {
  display: inline-flex;
  align-items: center;
  min-height: 20px;
  padding: 2px 8px;
  border: 1px solid rgba(0, 0, 0, 0.2);
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  white-space: nowrap;
}

.state-fresh {
  background: rgba(35, 122, 77, 0.14);
  color: #15553a;
}

.state-stale {
  background: rgba(190, 126, 20, 0.14);
  color: #8a5610;
}

.state-unknown {
  background: rgba(80, 90, 112, 0.14);
  color: #41506a;
}
</style>
