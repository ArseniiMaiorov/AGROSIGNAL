<template>
  <div class="progress-shell window-shell" data-testid="progress-shell" :class="{ stale: store.runStaleRunning }">
    <div class="progress-title">{{ titleLabel }}</div>
    <div class="progress-track">
      <div class="progress-fill" :style="fillStyle"></div>
    </div>
    <div class="progress-text">
      <span>{{ progressLabel }}%</span>
      <span>·</span>
      <span>{{ stageLabel }}</span>
      <span v-if="store.runStaleRunning" class="progress-badge stale-badge">{{ staleLabel }}</span>
    </div>
    <div v-if="showDetail" class="progress-detail">
      <span>{{ detailLabel }}</span>
      <span v-if="visibilityHint" class="progress-meta">{{ visibilityHint }}</span>
      <span v-if="elapsedLabel" class="progress-meta">{{ elapsedLabel }}</span>
      <span v-if="etaLabel" class="progress-meta">{{ etaLabel }}</span>
      <span v-if="heartbeatLabel" class="progress-meta">{{ heartbeatLabel }}</span>
    </div>
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { useMapStore } from '../store/map'
import { locale } from '../utils/i18n'
import { formatUiProgress, formatUiTime, getTaskStageDetail, getTaskStageLabel } from '../utils/presentation'

const store = useMapStore()
const nowTs = ref(Date.now())
let timerId = null
const isRu = computed(() => locale.value === 'ru')
const titleLabel = computed(() => (isRu.value ? 'Выполнение задачи' : 'Task progress'))
const staleLabel = computed(() => (isRu.value ? 'зависло' : 'stalled'))
const progressLabel = computed(() => formatUiProgress(store.runProgress))

const stageLabel = computed(() => {
  if (store.runStageCode || store.runStageLabel) {
    return getTaskStageLabel({ stage_code: store.runStageCode, stage_label: store.runStageLabel, status: store.runStatus }, store.runStageLabel || store.runStatus)
  }
  if (!store.runStatus) return isRu.value ? 'Подготовка' : 'Preparing'
  return getTaskStageLabel({ status: store.runStatus }, store.runStatus)
})

const detailLabel = computed(() => getTaskStageDetail({
  stage_detail: store.runStageDetail,
  stage_detail_code: store.runStageDetailCode,
  stage_detail_params: store.runStageDetailParams,
}) || '')
const visibilityHint = computed(() => {
  if (!store.activeRunId || !store.visibleRunId || store.activeRunId === store.visibleRunId) return ''
  return isRu.value
    ? 'Карта показывает предыдущий завершённый результат.'
    : 'The map still shows the previous completed result.'
})
const showDetail = computed(
  () => store.progressVerbosity === 'detailed' && Boolean(detailLabel.value || visibilityHint.value || elapsedLabel.value || etaLabel.value || heartbeatLabel.value)
)
const elapsedLabel = computed(() => {
  if (!store.runStartedAt) return ''
  const started = Date.parse(store.runStartedAt)
  if (!Number.isFinite(started)) return ''
  const elapsedSeconds = Math.max(0, Math.round((nowTs.value - started) / 1000))
  if (elapsedSeconds < 60) return isRu.value ? `Прошло ${elapsedSeconds} с` : `Elapsed ${elapsedSeconds}s`
  return isRu.value ? `Прошло ${Math.floor(elapsedSeconds / 60)} мин` : `Elapsed ${Math.floor(elapsedSeconds / 60)}m`
})
const etaLabel = computed(() => {
  const raw = store.runEstimatedRemainingS
  if (raw === null || raw === undefined) return ''
  const seconds = Number(raw)
  if (!Number.isFinite(seconds) || seconds <= 0) return ''
  if (seconds < 60) return `ETA ${Math.round(seconds)}s`
  return isRu.value ? `ETA ${Math.round(seconds / 60)} мин` : `ETA ${Math.round(seconds / 60)}m`
})
const heartbeatLabel = computed(() => {
  if (!store.runLastHeartbeatTs || !store.expertMode) return ''
  return `hb ${formatUiTime(store.runLastHeartbeatTs)}`
})
const fillStyle = computed(() => ({
  width: `${store.runProgress}%`,
  transitionDuration: store.animationDensity === 'low' ? '0.12s' : '0.35s',
}))

onMounted(() => {
  timerId = window.setInterval(() => {
    nowTs.value = Date.now()
  }, 1000)
})

onBeforeUnmount(() => {
  if (timerId) {
    window.clearInterval(timerId)
  }
})
</script>

<style scoped>
.progress-shell {
  position: absolute;
  top: 92px;
  left: 50%;
  transform: translateX(-50%);
  width: min(460px, calc(100vw - 48px));
  z-index: 9;
}

.progress-title {
  font-weight: 700;
  margin-bottom: 5px;
  color: var(--text-main);
  font-size: 12px;
}

.progress-track {
  height: 14px;
  border: 2px solid;
  border-color: #7e7e7e #ffffff #ffffff #7e7e7e;
  background: #efefef;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #1f6aa0, #3c9c64);
  transition-property: width;
  transition-timing-function: ease;
}

.progress-text {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  margin-top: 4px;
  color: #385474;
  font-size: 12px;
}

.progress-detail {
  display: flex;
  justify-content: center;
  gap: 10px;
  margin-top: 4px;
  font-size: 11px;
  color: #536476;
  text-align: center;
}

.progress-meta {
  color: #73859a;
}

.progress-badge {
  padding: 1px 6px;
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
  border: 1px solid rgba(0, 0, 0, 0.18);
}

.stale-badge {
  background: rgba(201, 74, 68, 0.14);
  color: #8a221f;
}

.progress-shell.stale {
  border-color: rgba(201, 74, 68, 0.45);
}
</style>
