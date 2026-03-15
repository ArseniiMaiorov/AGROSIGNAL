<template>
  <section class="window-shell log-panel">
    <div class="window-title">
      <span style="flex:1">{{ t('logs.title') }}</span>
      <div class="log-filters">
        <button
          v-for="filter in filters"
          :key="filter.id"
          class="filter-btn"
          :class="{ active: activeFilter === filter.id }"
          @click="activeFilter = filter.id"
        >
          {{ filter.label }}
        </button>
      </div>
      <button v-if="store.logs.length" class="clear-btn" @click="store.clearLogs()">{{ t('logs.clear') }}</button>
    </div>
    <div class="window-body log-body" ref="logContent">
      <div v-if="filteredLogs.length === 0" class="placeholder">{{ t('logs.empty') }}</div>
      <div
        v-for="(entry, index) in filteredLogs"
        :key="`${entry.ts}-${index}`"
        class="log-line"
        :class="logLineClass(entry)"
      >
        <span class="log-line-ts">[{{ formatEntryTime(entry) }}]</span>
        <span class="log-line-message">{{ entry.message }}</span>
      </div>
    </div>
  </section>
</template>

<script setup>
import { computed, nextTick, ref, watch } from 'vue'
import { useMapStore } from '../store/map'
import { t } from '../utils/i18n'

const store = useMapStore()
const logContent = ref(null)
const activeFilter = ref('all')

const filters = computed(() => [
  { id: 'all', label: t('logs.filterAll') },
  { id: 'warnings', label: t('logs.filterWarnings') },
  { id: 'errors', label: t('logs.filterErrors') },
])

function isError(entry) {
  return entry?.severity === 'error'
}

function isWarn(entry) {
  return entry?.severity === 'warning'
}

function logLineClass(entry) {
  if (isError(entry)) return 'log-error'
  if (isWarn(entry)) return 'log-warn'
  return ''
}

function formatEntryTime(entry) {
  try {
    return new Date(entry?.ts || Date.now()).toLocaleTimeString()
  } catch {
    return '--:--:--'
  }
}

const filteredLogs = computed(() => {
  if (activeFilter.value === 'all') return store.logs
  if (activeFilter.value === 'errors') return store.logs.filter(isError)
  if (activeFilter.value === 'warnings') return store.logs.filter((entry) => isError(entry) || isWarn(entry))
  return store.logs
})

watch(
  () => filteredLogs.value.length,
  async () => {
    const el = logContent.value
    const stickToBottom = el ? el.scrollTop + el.clientHeight >= el.scrollHeight - 24 : true
    await nextTick()
    if (stickToBottom && logContent.value) {
      logContent.value.scrollTop = logContent.value.scrollHeight
    }
  },
)
</script>

<style scoped>
.log-panel {
  min-height: 220px;
}

.log-body {
  max-height: 260px;
  overflow-y: auto;
}

.log-line {
  padding: 4px 0;
  border-bottom: 1px dotted var(--win-shadow-soft);
  color: var(--text-main);
  font-size: 12px;
  display: flex;
  gap: 6px;
  align-items: flex-start;
}

.log-line-ts {
  font-family: var(--font-mono);
  color: var(--text-muted);
  white-space: nowrap;
}

.log-line-message {
  min-width: 0;
  line-height: 1.4;
}

.log-error {
  color: #c0392b;
  font-weight: 700;
}

.log-warn {
  color: #856404;
}

.log-filters {
  display: flex;
  gap: 2px;
  margin-right: 8px;
}

.filter-btn {
  background: transparent;
  border: 1px solid transparent;
  color: var(--win-title-color);
  font-size: 11px;
  cursor: pointer;
  padding: 1px 6px;
  border-radius: 0;
}

.filter-btn.active {
  background: var(--win-bg);
  border: 1px solid var(--win-shadow-mid);
  color: var(--text-main);
  font-weight: 700;
}

.clear-btn {
  background: transparent;
  border: none;
  color: var(--win-title-color);
  font-size: 11px;
  cursor: pointer;
  padding: 0 6px;
}

.clear-btn:hover {
  text-decoration: underline;
}
</style>
