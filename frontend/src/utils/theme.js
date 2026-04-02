import { reactive, computed } from 'vue'

const THEME_KEY = 'terrainfo-theme'

const themes = {
  light: {
    '--win-bg': '#d4d0c8',
    '--win-panel': '#ece9d8',
    '--win-shadow-dark': '#5f5f5f',
    '--win-shadow-mid': '#7e7e7e',
    '--win-shadow-light': '#ffffff',
    '--win-shadow-soft': '#c7c7c7',
    '--text-main': '#203248',
    '--text-muted': '#5d6470',
    '--win-title-bg': 'linear-gradient(90deg, #0e4470 0%, #2a79a5 65%, #6bb0d3 100%)',
    '--win-title-color': '#f8fbff',
    '--btn-primary-bg': 'linear-gradient(180deg, #5db98c 0%, #2d8b64 100%)',
    '--btn-primary-color': '#f7fff9',
    '--input-bg': '#ffffff',
    '--input-color': '#1a2735',
    '--desktop-bg': 'linear-gradient(180deg, rgba(255,255,255,0.15), transparent 24%), linear-gradient(135deg, #0b6f77 0%, #138b91 38%, #3a9f7a 100%)',
    '--scrollbar-track': '#d8d3cb',
    '--scrollbar-thumb': '#a7a7a7',
    '--error-bg': '#f3cbc6',
    '--error-color': '#6e241e',
    '--error-border': '#8d4c47',
    '--status-online-bg': '#c9f0d8',
    '--status-online-color': '#124125',
    '--status-degraded-bg': '#f3e8b8',
    '--status-degraded-color': '#594500',
    '--status-offline-bg': '#f3cbc6',
    '--status-offline-color': '#68211c',
    '--taskbar-btn-bg': 'linear-gradient(180deg, #e9e9e9, #c7c7c7)',
    '--weather-cell-bg': '#efefef',
    '--tooltip-bg': '#fff6d6',
  },
  dark: {
    '--win-bg': '#2d2d2d',
    '--win-panel': '#363636',
    '--win-shadow-dark': '#1a1a1a',
    '--win-shadow-mid': '#444444',
    '--win-shadow-light': '#555555',
    '--win-shadow-soft': '#3a3a3a',
    '--text-main': '#e0e0e0',
    '--text-muted': '#999999',
    '--win-title-bg': 'linear-gradient(90deg, #1a3a5c 0%, #2a5a8c 65%, #3a7aac 100%)',
    '--win-title-color': '#e8eeff',
    '--btn-primary-bg': 'linear-gradient(180deg, #3d9960 0%, #1d7a44 100%)',
    '--btn-primary-color': '#e0ffe8',
    '--input-bg': '#3a3a3a',
    '--input-color': '#e0e0e0',
    '--desktop-bg': 'linear-gradient(180deg, rgba(0,0,0,0.2), transparent 24%), linear-gradient(135deg, #0a3a3e 0%, #0c4a4e 38%, #1a5a3a 100%)',
    '--scrollbar-track': '#2d2d2d',
    '--scrollbar-thumb': '#555555',
    '--error-bg': '#4a2020',
    '--error-color': '#f0a0a0',
    '--error-border': '#6a3030',
    '--status-online-bg': '#1a4a2a',
    '--status-online-color': '#90e0a0',
    '--status-degraded-bg': '#4a3a10',
    '--status-degraded-color': '#e0c060',
    '--status-offline-bg': '#4a2020',
    '--status-offline-color': '#f0a0a0',
    '--taskbar-btn-bg': 'linear-gradient(180deg, #444, #333)',
    '--weather-cell-bg': '#3a3a3a',
    '--tooltip-bg': '#3a3520',
  },
}

const state = reactive({
  theme: localStorage.getItem(THEME_KEY) || 'light',
})

function applyTheme(name) {
  const vars = themes[name]
  if (!vars) return
  const root = document.documentElement
  for (const [prop, value] of Object.entries(vars)) {
    root.style.setProperty(prop, value)
  }
}

function setTheme(name) {
  state.theme = name
  localStorage.setItem(THEME_KEY, name)
  applyTheme(name)
}

function initTheme() {
  applyTheme(state.theme)
}

const theme = computed(() => state.theme)

export { setTheme, initTheme, theme, state as themeState }
