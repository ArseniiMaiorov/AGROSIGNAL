<template>
  <section class="window-shell status-panel" data-testid="status-panel">
    <div class="window-title">{{ t('status.title') }}</div>
    <div class="window-body">
      <div class="status-overview">
        <div class="status-main">
          <span class="status-pill" :class="store.systemStatus?.status || 'degraded'">
            {{ statusLabel }}
          </span>
          <span class="status-note">{{ timestampLabel }}</span>
        </div>
        <FreshnessBadge :meta="store.systemStatus?.freshness" compact />
      </div>
      <div class="status-list">
        <div v-for="(component, name) in components" :key="name" class="status-row">
          <div class="status-row-main">
            <span class="status-row-name">{{ t(`status.${name}`) }}</span>
            <span class="status-row-value" :class="component.status">{{ mapStatus(component.status) }}</span>
          </div>
          <div v-if="component.detail" class="status-row-detail">{{ component.detail }}</div>
        </div>
      </div>
      <div v-if="store.expertMode && modelTruth" class="status-truth">
        <div class="status-truth-title">{{ t('field.modelTruth') }}</div>
        <div>{{ t('field.modelHeadCount') }}: {{ modelTruth.head_count || '—' }}</div>
        <div>{{ t('field.modelHeads') }}: {{ modelTruth.heads?.join(', ') || '—' }}</div>
        <div>{{ t('field.ttaStandard') }}: {{ modelTruth.tta_standard || '—' }}</div>
        <div>{{ t('field.ttaQuality') }}: {{ modelTruth.tta_quality || '—' }}</div>
        <div class="status-truth-detail">
          {{ modelTruth.retrain_description || t('field.retrainDescription') }}
        </div>
      </div>
      <div v-if="store.expertMode && store.systemStatus?.build" class="status-build">
        <div>app {{ store.systemStatus.build.app_version || '—' }}</div>
        <div>model {{ store.systemStatus.build.model_version || '—' }}</div>
        <div>dataset {{ store.systemStatus.build.train_data_version || '—' }}</div>
      </div>
    </div>
  </section>
</template>

<script setup>
import { computed } from 'vue'
import FreshnessBadge from './FreshnessBadge.vue'
import { useMapStore } from '../store/map'
import { t } from '../utils/i18n'
import { formatUiDateTime } from '../utils/presentation'

const store = useMapStore()

const components = computed(() => store.systemStatus?.components || {})
const modelTruth = computed(() => store.systemStatus?.model_truth || null)

const statusLabel = computed(() => {
  const value = store.systemStatus?.status
  if (value === 'online') return t('status.online')
  if (value === 'offline') return t('status.offline')
  return t('status.degraded')
})

const timestampLabel = computed(() => {
  if (!store.systemStatus?.timestamp) {
    return t('status.updating')
  }
  return formatUiDateTime(store.systemStatus.timestamp)
})

function mapStatus(value) {
  if (value === 'online') return t('status.online')
  if (value === 'offline') return t('status.offline')
  return t('status.degraded')
}
</script>

<style scoped>
.status-panel {
  min-height: 0;
}

.status-main {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 6px;
}

.status-pill {
  padding: 4px 10px;
  font-weight: 700;
  border: 2px solid;
  border-color: var(--win-shadow-light) var(--win-shadow-dark) var(--win-shadow-dark) var(--win-shadow-light);
}

.status-pill.online { background: var(--status-online-bg); color: var(--status-online-color); }
.status-pill.degraded { background: var(--status-degraded-bg); color: var(--status-degraded-color); }
.status-pill.offline { background: var(--status-offline-bg); color: var(--status-offline-color); }

.status-note {
  color: var(--text-muted);
  font-size: 12px;
}

.status-list {
  margin-top: 10px;
  display: flex;
  flex-direction: column;
  gap: 4px;
  max-height: 146px;
  overflow-y: auto;
}

.status-row {
  display: flex;
  flex-direction: column;
  gap: 4px;
  border-bottom: 1px solid var(--win-shadow-soft);
  padding: 4px 0 6px;
}

.status-row-main {
  display: flex;
  justify-content: space-between;
  gap: 10px;
}

.status-row-name {
  color: var(--text-main);
}

.status-row-value {
  font-weight: 700;
}

.status-row-value.online { color: var(--status-online-color); }
.status-row-value.degraded { color: var(--status-degraded-color); }
.status-row-value.offline { color: var(--status-offline-color); }

.status-row-detail {
  color: var(--text-muted);
  font-size: 11px;
  line-height: 1.35;
}

.status-build {
  margin-top: 10px;
  padding-top: 8px;
  border-top: 1px solid var(--win-shadow-soft);
  display: grid;
  gap: 3px;
  color: var(--text-muted);
  font-size: 11px;
}

.status-truth {
  margin-top: 10px;
  padding-top: 8px;
  border-top: 1px solid var(--win-shadow-soft);
  display: grid;
  gap: 3px;
  color: var(--text-main);
  font-size: 11px;
}

.status-truth-title {
  font-weight: 700;
  color: var(--text-main);
}

.status-truth-detail {
  color: var(--text-muted);
  line-height: 1.35;
}
</style>
