<template>
  <footer class="taskbar-shell">
    <div v-if="startMenuOpen" class="start-menu window-shell">
      <div class="window-title">{{ t('taskbar.menu') }}</div>
      <div class="window-body start-menu-body">
        <button
          v-for="item in startMenuItems"
          :key="item.id"
          class="start-menu-item"
          @click="handleStartItem(item)"
        >
          {{ item.label }}
        </button>
        <hr class="menu-separator" />
        <div class="menu-settings">
          <div class="setting-row">
            <span data-btip="Язык интерфейса приложения. RU — русский, EN — английский.">{{ locale === 'ru' ? 'Язык' : 'Language' }}</span>
            <div class="setting-buttons">
              <button :class="{ active: locale === 'ru' }" @click="switchLang('ru')">RU</button>
              <button :class="{ active: locale === 'en' }" @click="switchLang('en')">EN</button>
            </div>
          </div>
          <div class="setting-row">
            <span data-btip="Цветовая тема интерфейса. Светлая — для дневной работы, Тёмная — для ночи.">{{ locale === 'ru' ? 'Тема' : 'Theme' }}</span>
            <div class="setting-buttons">
              <button :class="{ active: theme === 'light' }" @click="switchTheme('light')">{{ locale === 'ru' ? 'Светлая' : 'Light' }}</button>
              <button :class="{ active: theme === 'dark' }" @click="switchTheme('dark')">{{ locale === 'ru' ? 'Тёмная' : 'Dark' }}</button>
            </div>
          </div>
          <div class="setting-row column-row">
            <span :data-btip="detectionPresetTooltip">{{ locale === 'ru' ? 'Детект-профиль' : 'Detection preset' }}</span>
            <select class="setting-select" :value="store.detectionPreset" @change="updatePreset">
              <option value="fast">{{ presetOptionLabel('fast') }}</option>
              <option value="standard">{{ presetOptionLabel('standard') }}</option>
              <option value="quality">{{ presetOptionLabel('quality') }}</option>
            </select>
            <small class="setting-hint">{{ store.activeDetectionPreset.description }}</small>
          </div>
          <div class="setting-row column-row">
            <span data-btip="Интервал автоматического обновления данных (погода, статус). 0 — автообновление отключено.">{{ locale === 'ru' ? 'Автообновление' : 'Auto refresh' }}</span>
            <select class="setting-select" :value="String(store.autoRefreshIntervalS)" @change="updateRefreshInterval">
              <option value="15">15s</option>
              <option value="30">30s</option>
              <option value="60">60s</option>
              <option value="0">{{ locale === 'ru' ? 'Выкл.' : 'Off' }}</option>
            </select>
          </div>
          <div class="setting-row column-row">
            <span data-btip="Детализация отображения прогресса детекции.
Кратко — только ключевые этапы.
Подробно — все шаги и проценты.">{{ locale === 'ru' ? 'Прогресс' : 'Progress' }}</span>
            <select class="setting-select" v-model="store.progressVerbosity">
              <option value="simple">{{ locale === 'ru' ? 'Кратко' : 'Simple' }}</option>
              <option value="detailed">{{ locale === 'ru' ? 'Подробно' : 'Detailed' }}</option>
            </select>
          </div>
          <div class="setting-row column-row">
            <span data-btip="Интенсивность анимаций интерфейса. Минимум — ускоряет работу на слабых машинах.">{{ locale === 'ru' ? 'Анимации' : 'Animations' }}</span>
            <select class="setting-select" v-model="store.animationDensity">
              <option value="low">{{ locale === 'ru' ? 'Минимум' : 'Low' }}</option>
              <option value="normal">{{ locale === 'ru' ? 'Обычные' : 'Normal' }}</option>
            </select>
          </div>
          <div class="setting-row column-row">
            <span data-btip="Показывать ли метки актуальности данных (источник, время получения, свежесть кэша) на виджетах погоды и слоях.">{{ locale === 'ru' ? 'Метки свежести' : 'Freshness badges' }}</span>
            <select class="setting-select" :value="String(store.showFreshnessBadges)" @change="updateBoolean('showFreshnessBadges', $event)">
              <option value="true">{{ locale === 'ru' ? 'Вкл.' : 'On' }}</option>
              <option value="false">{{ locale === 'ru' ? 'Выкл.' : 'Off' }}</option>
            </select>
          </div>
          <div class="setting-row column-row">
            <span data-btip="Подписи на полях карты.
Выкл — без подписей.
Компактно — ID поля.
Полно — ID + площадь + метрики.">{{ locale === 'ru' ? 'Плотность подписей' : 'Map label density' }}</span>
            <select class="setting-select" v-model="store.mapLabelDensity">
              <option value="off">{{ locale === 'ru' ? 'Выкл.' : 'Off' }}</option>
              <option value="compact">{{ locale === 'ru' ? 'Компактно' : 'Compact' }}</option>
              <option value="full">{{ locale === 'ru' ? 'Полно' : 'Full' }}</option>
            </select>
          </div>
          <div class="setting-row column-row">
            <span data-btip="Разблокирует расширенные параметры: число дат, минимальная площадь поля, разрешение, SAM. Также показывает диагностику детектора.">{{ locale === 'ru' ? 'Экспертный режим' : 'Expert mode' }}</span>
            <select class="setting-select" :value="String(store.expertMode)" @change="updateBoolean('expertMode', $event)">
              <option value="false">{{ locale === 'ru' ? 'Выкл.' : 'Off' }}</option>
              <option value="true">{{ locale === 'ru' ? 'Вкл.' : 'On' }}</option>
            </select>
          </div>
          <div class="setting-row column-row">
            <span data-btip="При включении — при наведении на любой элемент управления появляется подсказка с объяснением что это такое и как работает.">{{ locale === 'ru' ? 'Режим новичка' : 'Beginner mode' }}</span>
            <select class="setting-select" :value="String(store.beginnerMode)" @change="updateBoolean('beginnerMode', $event)">
              <option value="false">{{ locale === 'ru' ? 'Выкл.' : 'Off' }}</option>
              <option value="true">{{ locale === 'ru' ? 'Вкл.' : 'On' }}</option>
            </select>
            <small class="setting-hint">{{ locale === 'ru' ? 'Подсказки при наведении на все элементы управления' : 'Tooltips on hover for all controls' }}</small>
          </div>
        </div>
      </div>
    </div>

    <div class="taskbar window-shell">
      <button class="taskbar-start" data-testid="taskbar-start" @click.stop="toggleStartMenu">{{ t('taskbar.start') }}</button>

      <div class="taskbar-windows">
        <button
          v-for="item in windowButtons"
          :key="item.id"
          class="taskbar-window-btn"
          :class="{ active: store.uiWindows[item.id] }"
          :title="item.title"
          :data-btip="item.btip"
          @click="store.toggleWindow(item.id)"
        >
          {{ item.shortLabel }}
        </button>
      </div>

      <div class="taskbar-brand" title="AgroVision Production">{{ t('app.brand') }}</div>

      <div class="taskbar-status" :title="statusLabel">
        <span class="status-lamp" :class="statusClass"></span>
        <span class="taskbar-status-text">{{ statusLabel }}</span>
        <span class="taskbar-time">{{ clock }}</span>
      </div>
    </div>
  </footer>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { useMapStore } from '../store/map'
import { t, setLocale, locale } from '../utils/i18n'
import { setTheme, theme } from '../utils/theme'
import { formatUiTime, getDetectionPresetMeta } from '../utils/presentation'

const store = useMapStore()
const clock = ref(formatUiTime(new Date(), { hour: '2-digit', minute: '2-digit' }))
const startMenuOpen = ref(false)
let timerId = null

const startMenuItems = computed(() => [
  { id: 'help', label: t('taskbar.menuHelp'), type: 'window' },
  { id: 'refresh', label: t('taskbar.menuRefresh'), type: 'action' },
])

const windowButtons = computed(() => {
  const ru = locale.value === 'ru'
  return [
    { id: 'control', shortLabel: ru ? 'Управление' : 'Control', title: t('taskbar.menuControl'),
      btip: ru ? 'Панель управления детекцией: координаты, радиус, период, профиль, кнопка запуска.' : 'Detection control panel.' },
    { id: 'weather', shortLabel: ru ? 'Погода' : 'Weather', title: t('taskbar.menuWeather'),
      btip: ru ? 'Погода: ветер, осадки, температура, влажность воздуха и почвы.' : 'Weather panel.' },
    { id: 'status', shortLabel: ru ? 'Статус' : 'Status', title: t('taskbar.menuStatus'),
      btip: ru ? 'Статус детекции: прогресс, этапы, ошибки, время выполнения.' : 'Detection run status panel.' },
    { id: 'fieldActions', shortLabel: ru ? 'Поле' : 'Field', title: t('taskbar.menuField'),
      btip: ru ? 'Панель поля: прогноз урожайности, метрики NDVI/NDMI, история, сценарии, архив.' : 'Field panel: yield, metrics, history, scenarios.' },
    { id: 'legend', shortLabel: ru ? 'Слои' : 'Layers', title: t('taskbar.menuLegend'),
      btip: ru ? 'Слои карты: NDVI, NDMI, NDWI, BSI, ветер, осадки, спутниковый снимок.' : 'Map layer controls.' },
    { id: 'logs', shortLabel: ru ? 'Логи' : 'Logs', title: t('taskbar.menuLogs'),
      btip: ru ? 'Журнал событий: preflight, запуск, прогресс, ошибки, обновления данных.' : 'Event log.' },
    { id: 'help', shortLabel: ru ? 'Справка' : 'Help', title: t('taskbar.menuHelp'),
      btip: ru ? 'Справка и инструкция по работе с приложением.' : 'Help and user guide.' },
  ]
})

const statusLabel = computed(() => {
  const status = store.systemStatus?.status
  if (status === 'online') return t('taskbar.systemOnline')
  if (status === 'degraded') return t('taskbar.systemDegraded')
  if (status === 'offline') return t('taskbar.systemOffline')
  return t('taskbar.statusLoading')
})

const statusClass = computed(() => store.systemStatus?.status || 'degraded')
const detectionPresetTooltip = computed(() => (
  locale.value === 'ru'
    ? `Профиль детекции полей.
Быстрый: preview-only, укрупнённые сельхоз-контуры до 40 км.
Стандарт: основной рабочий режим точного детекта полей до 20 км.
Качество: максимальная геометрия с TTA и тяжёлым уточнением до 8 км.`
    : `Detection profile.
Fast: preview-only, coarse agricultural contours up to 40 km.
Standard: the main operational field-detection mode up to 20 km.
Quality: highest-fidelity geometry with TTA and heavy refinement up to 8 km.`
))

function toggleStartMenu() {
  startMenuOpen.value = !startMenuOpen.value
}

function switchLang(lang) {
  setLocale(lang)
}

function switchTheme(name) {
  setTheme(name)
}

function updatePreset(event) {
  store.applyDetectionPreset(event.target.value)
}

function updateRefreshInterval(event) {
  store.autoRefreshIntervalS = Number(event.target.value)
  store.restartSystemPolling()
}

function updateBoolean(key, event) {
  store[key] = event.target.value === 'true'
}

function presetOptionLabel(preset) {
  return getDetectionPresetMeta(preset).label
}

async function handleStartItem(item) {
  startMenuOpen.value = false
  if (item.type === 'window') {
    store.toggleWindow(item.id)
    return
  }
  if (item.id === 'refresh') {
    await store.refreshAll()
  }
}

function handleGlobalClick(event) {
  if (!(event.target instanceof Element) || !event.target.closest('.taskbar-shell')) {
    startMenuOpen.value = false
  }
}

onMounted(() => {
  timerId = window.setInterval(() => {
    clock.value = formatUiTime(new Date(), { hour: '2-digit', minute: '2-digit' })
  }, 10000)
  window.addEventListener('click', handleGlobalClick)
})

onBeforeUnmount(() => {
  if (timerId) {
    clearInterval(timerId)
  }
  window.removeEventListener('click', handleGlobalClick)
})
</script>

<style scoped>
.taskbar-shell {
  position: relative;
  flex-shrink: 0;
  height: 44px;
  z-index: 20;
}

.taskbar {
  display: flex;
  align-items: center;
  gap: 6px;
  height: 44px;
  padding: 6px 10px;
}

.taskbar-start,
.start-menu-item {
  border: 2px solid;
  border-color: var(--win-shadow-light) var(--win-shadow-dark) var(--win-shadow-dark) var(--win-shadow-light);
  background: var(--taskbar-btn-bg);
  color: var(--text-main);
  font-weight: 700;
  cursor: pointer;
  white-space: nowrap;
}

.taskbar-start {
  min-width: 72px;
  height: 28px;
  flex-shrink: 0;
}

.taskbar-brand {
  flex: 1;
  padding: 0 8px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: var(--text-main);
  font-weight: 700;
  font-size: 12px;
}

.taskbar-windows {
  display: flex;
  align-items: center;
  gap: 4px;
  flex-shrink: 0;
}

.taskbar-window-btn {
  min-width: 84px;
  height: 28px;
  padding: 0 10px;
  border: 2px solid;
  border-color: var(--win-shadow-light) var(--win-shadow-dark) var(--win-shadow-dark) var(--win-shadow-light);
  background: var(--taskbar-btn-bg);
  color: var(--text-main);
  font-weight: 700;
  cursor: pointer;
  white-space: nowrap;
  font-size: 12px;
}

.taskbar-window-btn.active {
  background: #bfd2e6;
}

.taskbar-status {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 0 10px;
  height: 28px;
  border: 2px solid;
  border-color: var(--win-shadow-mid) var(--win-shadow-light) var(--win-shadow-light) var(--win-shadow-mid);
  background: var(--win-bg);
  flex-shrink: 0;
}

.taskbar-status-text {
  max-width: 180px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: 12px;
}

.status-lamp {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  border: 1px solid rgba(0, 0, 0, 0.3);
  flex-shrink: 0;
}

.status-lamp.online { background: #0bb45f; }
.status-lamp.degraded { background: #d7a100; }
.status-lamp.offline { background: #c94a44; }

.taskbar-time {
  min-width: 42px;
  text-align: right;
  font-size: 12px;
}

.setting-hint {
  display: block;
  margin-top: 6px;
  color: var(--text-muted);
  font-size: 11px;
  line-height: 1.3;
}

.start-menu {
  position: absolute;
  left: 8px;
  bottom: 48px;
  width: 260px;
  z-index: 25;
}

.start-menu-body {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.start-menu-item {
  min-height: 30px;
  text-align: left;
  padding: 0 10px;
  font-size: 12px;
}

.menu-separator {
  border: none;
  border-top: 1px solid var(--win-shadow-soft);
  margin: 4px 0;
}

.menu-settings {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 4px 0;
}

.setting-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 0 4px;
}

.column-row {
  flex-direction: column;
  align-items: stretch;
}

.setting-row span {
  color: var(--text-muted);
  font-size: 12px;
}

.setting-buttons {
  display: flex;
  gap: 4px;
}

.setting-buttons button {
  border: 2px solid;
  border-color: var(--win-shadow-light) var(--win-shadow-dark) var(--win-shadow-dark) var(--win-shadow-light);
  background: var(--win-bg);
  color: var(--text-main);
  padding: 2px 10px;
  font-size: 11px;
  font-weight: 700;
  cursor: pointer;
  min-height: 24px;
}

.setting-buttons button.active {
  border-color: var(--win-shadow-dark) var(--win-shadow-light) var(--win-shadow-light) var(--win-shadow-dark);
  background: #bfd2e6;
}

.setting-select {
  width: 100%;
  min-height: 28px;
  border: 2px solid;
  border-color: var(--win-shadow-mid) var(--win-shadow-light) var(--win-shadow-light) var(--win-shadow-mid);
  background: var(--win-bg);
  color: var(--text-main);
  font: inherit;
}

@media (max-width: 900px) {
  .taskbar-windows {
    display: none;
  }

  .taskbar-status-text {
    display: none;
  }
}
</style>
