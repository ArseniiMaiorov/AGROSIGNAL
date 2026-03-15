<template>
  <section class="boot-shell" data-testid="boot-shell">
    <div class="boot-card">
      <div class="boot-head">
        <div class="boot-kicker">AgroVision boot sequence</div>
        <div class="boot-version">
          <span>{{ payload?.build?.app_version || '1.0.0' }}</span>
          <span v-if="payload?.build?.model_version">model {{ payload.build.model_version }}</span>
          <span v-if="payload?.build?.train_data_version">dataset {{ payload.build.train_data_version }}</span>
        </div>
      </div>

      <div class="boot-console">
        <div v-if="isLoading" class="boot-line boot-loading">
          <span class="boot-state">[ .... ]</span>
          <span class="boot-name">bootstrap</span>
          <span class="boot-detail">contacting `/api/v1/bootstrap`</span>
        </div>
        <template v-else>
          <div
            v-for="entry in visibleEntries"
            :key="entry.name"
            class="boot-line"
            :class="`status-${entry.status}`"
          >
            <span class="boot-state">[ {{ formatStatus(entry.status) }} ]</span>
            <span class="boot-name">{{ entry.label }}</span>
            <span class="boot-detail">{{ entry.detail }}</span>
          </div>
        </template>
      </div>

      <div v-if="loadError" class="boot-banner boot-error">
        {{ locale === 'ru' ? 'Не удалось получить bootstrap-статус backend.' : 'Unable to load backend bootstrap status.' }}
      </div>
      <div v-else-if="hasCriticalFailure" class="boot-banner boot-error">
        {{ locale === 'ru' ? 'Критический компонент недоступен. Вход заблокирован до восстановления database/auth bootstrap.' : 'Critical component offline. Login is blocked until database/auth bootstrap recovers.' }}
      </div>
      <div v-else-if="hasWarnings" class="boot-banner boot-warn">
        {{ locale === 'ru' ? 'Есть деградации. Вход будет разрешён, но часть функций может работать ограниченно.' : 'Degraded components detected. Login is allowed, but some functions may be limited.' }}
      </div>
      <div v-else-if="isLoading" class="boot-banner">
        {{ locale === 'ru' ? 'Проверяем критические сервисы и build metadata…' : 'Checking critical services and build metadata…' }}
      </div>
      <div v-else class="boot-banner boot-ok">
        {{ locale === 'ru' ? 'Базовые проверки завершены. Передаю управление login shell…' : 'Bootstrap checks completed. Handing off to login shell…' }}
      </div>

      <div class="boot-actions">
        <button v-if="loadError" class="boot-btn" data-testid="boot-retry" @click="loadBootstrap">
          {{ locale === 'ru' ? 'Повторить' : 'Retry' }}
        </button>
        <button v-else-if="!hasCriticalFailure" class="boot-btn" data-testid="boot-continue" @click="continueToLogin">
          {{ locale === 'ru' ? 'Продолжить к входу' : 'Continue to Login' }}
        </button>
      </div>
    </div>
  </section>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'

import axios, { API_BASE } from '../services/api'
import { locale } from '../utils/i18n'

const emit = defineEmits(['ready'])

const payload = ref(null)
const isLoading = ref(false)
const loadError = ref('')
const startedAt = ref(0)
const revealedCount = ref(0)
let continueTimer = null
let revealTimer = null

const componentLabels = {
  database: 'database',
  redis: 'redis',
  auth_bootstrap: 'auth bootstrap',
  sentinel_model: 'boundary model',
  edge_refiner: 'edge refiner',
  classifier: 'quality scorer',
  weather_provider: 'weather provider',
  layer_catalog: 'layer catalog',
  scene_cache: 'scene cache',
  satellite_browse: 'satellite browse',
  prediction_engine: 'prediction engine',
  scenario_engine: 'scenario engine',
  archive_service: 'archive service',
  async_jobs: 'async jobs',
  release_smoke: 'release smoke',
  training_pipeline: 'training pipeline',
  training_commands: 'training commands',
  cpu_training_deps: 'cpu training deps',
  qa_matrix: 'qa matrix',
  qa_band_summary: 'qa band summary',
  docs: 'docs',
}

const componentEntries = computed(() => {
  const components = payload.value?.components || {}
  const baseEntries = Object.entries(components).map(([name, entry]) => ({
    name,
    label: componentLabels[name] || name,
    status: String(entry?.status || 'unknown'),
    detail: entry?.detail ? formatDetail(entry.detail) : '',
  }))
  if (!payload.value) {
    return baseEntries
  }
  const statuses = baseEntries.map((entry) => entry.status)
  const online = statuses.filter((status) => status === 'online').length
  const degraded = statuses.filter((status) => status === 'degraded').length
  const offline = statuses.filter((status) => status === 'offline').length
  return [
    ...baseEntries,
    {
      name: 'self_check_summary',
      label: 'self-check summary',
      status: offline > 0 ? 'degraded' : 'online',
      detail: `online=${online}, degraded=${degraded}, offline=${offline}`,
    },
    {
      name: 'build_info',
      label: 'build info',
      status: 'online',
      detail: [
        payload.value?.build?.app_version ? `app=${payload.value.build.app_version}` : null,
        payload.value?.build?.model_version ? `model=${payload.value.build.model_version}` : null,
        payload.value?.build?.train_data_version ? `dataset=${payload.value.build.train_data_version}` : null,
        payload.value?.build?.yield_model_version ? `yield=${payload.value.build.yield_model_version}` : null,
      ].filter(Boolean).join(', '),
    },
  ]
})

const visibleEntries = computed(() => componentEntries.value.slice(0, revealedCount.value || 0))

const hasCriticalFailure = computed(() => {
  const components = payload.value?.components || {}
  return ['database', 'auth_bootstrap'].some((name) => String(components[name]?.status || '') === 'offline')
})

const hasWarnings = computed(() => {
  if (!payload.value) return false
  return componentEntries.value.some((entry) => entry.status === 'offline' || entry.status === 'degraded')
})

function formatStatus(status) {
  if (status === 'online') return ' ok '
  if (status === 'degraded') return 'warn'
  if (status === 'offline') return 'fail'
  return '....'
}

function formatDetail(detail) {
  if (typeof detail === 'string') return detail
  if (!detail || typeof detail !== 'object') return String(detail ?? '')
  return Object.entries(detail)
    .map(([key, value]) => `${key}=${value}`)
    .join(', ')
}

function clearContinueTimer() {
  if (continueTimer) {
    clearTimeout(continueTimer)
    continueTimer = null
  }
}

function clearRevealTimer() {
  if (revealTimer) {
    clearInterval(revealTimer)
    revealTimer = null
  }
}

function continueToLogin() {
  clearContinueTimer()
  emit('ready', payload.value)
}

async function loadBootstrap() {
  clearContinueTimer()
  clearRevealTimer()
  isLoading.value = true
  loadError.value = ''
  startedAt.value = Date.now()
  revealedCount.value = 0
  try {
    const response = await axios.get(`${API_BASE}/bootstrap`)
    payload.value = response.data
    const totalEntries = componentEntries.value.length
    revealTimer = window.setInterval(() => {
      revealedCount.value = Math.min(totalEntries, revealedCount.value + 1)
      if (revealedCount.value >= totalEntries) {
        clearRevealTimer()
      }
    }, 220)
    if (!hasCriticalFailure.value) {
      const minVisibleMs = hasWarnings.value ? 6200 : 4800
      const elapsedMs = Date.now() - startedAt.value
      const waitMs = Math.max(900, minVisibleMs - elapsedMs)
      continueTimer = window.setTimeout(() => {
        continueToLogin()
      }, waitMs)
    }
  } catch (requestError) {
    loadError.value =
      requestError?.response?.data?.detail || requestError?.message || 'bootstrap request failed'
  } finally {
    isLoading.value = false
  }
}

onMounted(() => {
  loadBootstrap()
})

onBeforeUnmount(() => {
  clearContinueTimer()
  clearRevealTimer()
})
</script>

<style scoped>
.boot-shell {
  position: absolute;
  inset: 0;
  z-index: 22;
  display: grid;
  place-items: center;
  padding: 24px;
  background:
    radial-gradient(circle at top left, rgba(103, 142, 86, 0.2), transparent 28%),
    linear-gradient(180deg, rgba(9, 18, 18, 0.92), rgba(16, 24, 28, 0.92));
}

.boot-card {
  width: min(720px, 100%);
  padding: 22px;
  border: 2px solid rgba(225, 214, 178, 0.28);
  background: rgba(8, 15, 18, 0.92);
  color: #dce9d7;
  box-shadow: 0 30px 120px rgba(0, 0, 0, 0.42);
}

.boot-head {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 16px;
  padding-bottom: 10px;
  border-bottom: 1px solid rgba(225, 214, 178, 0.18);
}

.boot-kicker {
  font-size: 12px;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: #d7c473;
}

.boot-version {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  justify-content: flex-end;
  font-size: 11px;
  color: rgba(220, 233, 215, 0.68);
}

.boot-console {
  display: flex;
  flex-direction: column;
  gap: 6px;
  min-height: 320px;
  font-family: 'IBM Plex Mono', 'Courier New', monospace;
  font-size: 13px;
}

.boot-line {
  display: grid;
  grid-template-columns: 68px 160px 1fr;
  gap: 12px;
  align-items: baseline;
}

.boot-state {
  color: rgba(220, 233, 215, 0.9);
}

.boot-name {
  color: #f0ead8;
  text-transform: lowercase;
}

.boot-detail {
  color: rgba(220, 233, 215, 0.68);
  min-width: 0;
  overflow-wrap: anywhere;
}

.boot-loading .boot-state,
.status-online .boot-state {
  color: #8fd3a4;
}

.status-degraded .boot-state {
  color: #e0b45d;
}

.status-offline .boot-state {
  color: #ef8b7e;
}

.boot-banner {
  margin-top: 16px;
  padding: 10px 12px;
  border: 1px solid rgba(225, 214, 178, 0.18);
  color: rgba(220, 233, 215, 0.82);
}

.boot-ok {
  background: rgba(34, 93, 58, 0.12);
}

.boot-warn {
  background: rgba(131, 87, 23, 0.12);
}

.boot-error {
  background: rgba(122, 41, 36, 0.18);
}

.boot-actions {
  display: flex;
  justify-content: flex-end;
  margin-top: 14px;
}

.boot-btn {
  min-width: 180px;
  min-height: 38px;
  border: 1px solid rgba(225, 214, 178, 0.28);
  background: linear-gradient(135deg, #d9c471, #f5e9bf);
  color: #10221b;
  font: inherit;
  font-weight: 700;
  cursor: pointer;
}

@media (max-width: 760px) {
  .boot-head,
  .boot-line {
    grid-template-columns: 1fr;
  }

  .boot-head {
    flex-direction: column;
  }

  .boot-version {
    justify-content: flex-start;
  }
}
</style>
