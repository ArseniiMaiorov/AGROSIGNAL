<template>
  <div class="desktop-shell" :class="themeState.theme">
    <div class="desktop-background"></div>

    <div v-if="auth.isAuthenticated" class="session-strip">
      <div class="session-meta">
        <strong>{{ auth.user?.organization_name || 'Organization' }}</strong>
        <span>{{ auth.user?.email }}</span>
      </div>
      <button class="session-exit" @click="handleLogout">Logout</button>
    </div>

    <header v-if="store.uiWindows.weather" class="top-weather-bar">
      <WeatherPanel :overlay="true" />
    </header>

    <main class="workspace" :class="{ 'has-weather': store.uiWindows.weather }">
      <MapView />

      <DraggableWindow
        v-if="store.uiWindows.control"
        window-id="control"
        :initial-x="14"
        :initial-y="14"
        :width="360"
        @close="store.hideWindow('control')"
      >
        <SidePanel />
      </DraggableWindow>

      <DraggableWindow
        v-if="store.uiWindows.status"
        window-id="status"
        :initial-x="-314"
        :initial-y="14"
        :width="300"
        @close="store.hideWindow('status')"
      >
        <StatusPanel />
      </DraggableWindow>

      <DraggableWindow
        v-if="store.uiWindows.fieldActions"
        window-id="fieldActions"
        :initial-x="-574"
        :initial-y="162"
        :width="560"
        @close="store.hideWindow('fieldActions')"
      >
        <FieldActionsPanel />
      </DraggableWindow>

      <DraggableWindow
        v-if="store.uiWindows.legend"
        window-id="legend"
        :initial-x="-334"
        :initial-y="-200"
        :width="320"
        @close="store.hideWindow('legend')"
      >
        <LegendPanel />
      </DraggableWindow>

      <DraggableWindow
        v-if="store.uiWindows.logs"
        window-id="logs"
        :initial-x="390"
        :initial-y="-200"
        :width="700"
        @close="store.hideWindow('logs')"
      >
        <LogPanel />
      </DraggableWindow>

      <DraggableWindow
        v-if="store.uiWindows.help"
        window-id="help"
        :initial-x="48"
        :initial-y="92"
        :width="640"
        @close="store.hideWindow('help')"
      >
        <HelpPanel />
      </DraggableWindow>

      <ProgressBar v-if="store.isDetecting" />
    </main>
    <Taskbar />

    <BootShell
      v-if="auth.restored && !auth.isAuthenticated && !bootCompleted"
      :key="bootSequenceKey"
      @ready="handleBootReady"
    />
    <AuthOverlay
      v-else-if="auth.restored && !auth.isAuthenticated"
      :initial-email="bootstrapHints?.auth?.bootstrap_admin_email || ''"
      :initial-organization-slug="bootstrapHints?.auth?.bootstrap_org_slug || ''"
      :initial-organization-name="bootstrapHints?.auth?.bootstrap_org_name || ''"
    />
  </div>
</template>

<script setup>
import { defineAsyncComponent, onBeforeUnmount, onMounted, ref, watch, watchEffect } from 'vue'
import AuthOverlay from './components/AuthOverlay.vue'
import BootShell from './components/BootShell.vue'
import DraggableWindow from './components/DraggableWindow.vue'
import MapView from './components/MapView.vue'
import ProgressBar from './components/ProgressBar.vue'
import Taskbar from './components/Taskbar.vue'
import { useAuthStore } from './store/auth'
import { useMapStore } from './store/map'
import { initTheme, themeState } from './utils/theme'

const SidePanel = defineAsyncComponent(() => import('./components/SidePanel.vue'))
const StatusPanel = defineAsyncComponent(() => import('./components/StatusPanel.vue'))
const FieldActionsPanel = defineAsyncComponent(() => import('./components/FieldActionsPanel.vue'))
const LegendPanel = defineAsyncComponent(() => import('./components/LegendPanel.vue'))
const LogPanel = defineAsyncComponent(() => import('./components/LogPanel.vue'))
const WeatherPanel = defineAsyncComponent(() => import('./components/WeatherPanel.vue'))
const HelpPanel = defineAsyncComponent(() => import('./components/HelpPanel.vue'))

const store = useMapStore()
const auth = useAuthStore()
const bootCompleted = ref(false)
const bootSequenceKey = ref(0)
const bootstrapHints = ref(null)

onMounted(() => {
  initTheme()
  if (auth.isAuthenticated) {
    bootCompleted.value = true
    store.initialize()
  }
})

onBeforeUnmount(() => {
  store.clearTimers()
})

watch(
  () => auth.isAuthenticated,
  (isAuthenticated, wasAuthenticated) => {
    if (isAuthenticated && !wasAuthenticated) {
      bootCompleted.value = true
      store.initialize()
      return
    }
    if (!isAuthenticated && wasAuthenticated) {
      bootCompleted.value = false
      bootSequenceKey.value += 1
      bootstrapHints.value = null
      store.clearTimers()
      store.clearFieldSelection()
    }
  }
)

function handleBootReady(payload) {
  bootstrapHints.value = payload || null
  bootCompleted.value = true
}

// Toggle body class for beginner-mode CSS tooltips
watchEffect(() => {
  document.body.classList.toggle('beginner-mode', !!store.beginnerMode)
})

async function handleLogout() {
  await auth.logout()
}
</script>

<style scoped>
.desktop-shell {
  position: relative;
  width: 100vw;
  height: 100vh;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.desktop-background {
  position: absolute;
  inset: 0;
  background: var(--desktop-bg);
  z-index: 0;
}

.session-strip {
  position: absolute;
  top: 12px;
  right: 16px;
  z-index: 8;
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 8px 12px;
  border: 1px solid rgba(255, 255, 255, 0.24);
  background: rgba(15, 24, 28, 0.76);
  color: #f5f0e3;
  backdrop-filter: blur(6px);
}

.session-meta {
  display: grid;
  gap: 2px;
}

.session-meta span {
  font-size: 12px;
  opacity: 0.78;
}

.session-exit {
  border: 0;
  padding: 8px 12px;
  font: inherit;
  font-weight: 700;
  color: #10221b;
  background: #d7c473;
  cursor: pointer;
}

.workspace {
  position: relative;
  flex: 1;
  overflow: hidden;
}

.workspace.has-weather {
  margin-top: 72px;
}

.top-weather-bar {
  position: absolute;
  top: 8px;
  left: 8px;
  right: 8px;
  z-index: 7;
}

@media (max-width: 900px) {
  .session-strip {
    left: 12px;
    right: 12px;
    top: auto;
    bottom: 58px;
    justify-content: space-between;
  }

  .workspace.has-weather {
    margin-top: 112px;
  }
}
</style>
