<template>
  <div
    ref="windowEl"
    class="draggable-window"
    :style="windowStyle"
    @mousedown="bringToFront"
  >
    <slot />
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'

const props = defineProps({
  windowId: { type: String, required: true },
  initialX: { type: Number, default: 100 },
  initialY: { type: Number, default: 100 },
  width: { type: Number, default: 360 },
})

const emit = defineEmits(['close'])

const windowEl = ref(null)
const posX = ref(null)
const posY = ref(null)
const zIndex = ref(10)

let dragging = false
let startMouseX = 0
let startMouseY = 0
let startPosX = 0
let startPosY = 0

// Global z-index counter for stacking (capped well below taskbar at 9000)
const WINDOW_Z_MAX = 800
if (typeof window !== 'undefined') {
  window.__dragWindowZ = window.__dragWindowZ || 10
}

function bringToFront() {
  if (typeof window !== 'undefined') {
    window.__dragWindowZ = Math.min((window.__dragWindowZ || 10) + 1, WINDOW_Z_MAX)
    zIndex.value = window.__dragWindowZ
  }
}

function resolveInitialPosition() {
  if (posX.value !== null) return
  const parent = windowEl.value?.parentElement
  if (!parent) return
  const rect = parent.getBoundingClientRect()
  posX.value = props.initialX >= 0 ? props.initialX : rect.width + props.initialX
  posY.value = props.initialY >= 0 ? props.initialY : rect.height + props.initialY
}

const windowStyle = computed(() => {
  const style = {
    width: `${props.width}px`,
    zIndex: zIndex.value,
  }
  if (posX.value !== null) {
    style.left = `${posX.value}px`
    style.top = `${posY.value}px`
  }
  return style
})

function onMouseDown(event) {
  const title = event.target.closest('.window-title')
  if (!title) return
  if (event.target.closest('button')) return
  dragging = true
  startMouseX = event.clientX
  startMouseY = event.clientY
  startPosX = posX.value
  startPosY = posY.value
  bringToFront()
  document.addEventListener('mousemove', onMouseMove)
  document.addEventListener('mouseup', onMouseUp)
  event.preventDefault()
}

function onMouseMove(event) {
  if (!dragging) return
  const dx = event.clientX - startMouseX
  const dy = event.clientY - startMouseY
  const parent = windowEl.value?.parentElement
  const parentRect = parent?.getBoundingClientRect()
  const windowRect = windowEl.value?.getBoundingClientRect()
  const maxX = Math.max(0, (parentRect?.width || window.innerWidth) - (windowRect?.width || props.width) - 8)
  const maxY = Math.max(0, (parentRect?.height || window.innerHeight) - (windowRect?.height || 240) - 8)
  posX.value = Math.min(maxX, Math.max(0, startPosX + dx))
  posY.value = Math.min(maxY, Math.max(0, startPosY + dy))
}

function onMouseUp() {
  dragging = false
  document.removeEventListener('mousemove', onMouseMove)
  document.removeEventListener('mouseup', onMouseUp)
}

onMounted(() => {
  resolveInitialPosition()
  if (windowEl.value) {
    windowEl.value.addEventListener('mousedown', onMouseDown)
  }
})

onBeforeUnmount(() => {
  if (windowEl.value) {
    windowEl.value.removeEventListener('mousedown', onMouseDown)
  }
  document.removeEventListener('mousemove', onMouseMove)
  document.removeEventListener('mouseup', onMouseUp)
})
</script>

<style scoped>
.draggable-window {
  position: absolute;
  pointer-events: auto;
  max-width: calc(100vw - 28px);
}

.draggable-window :deep(.window-shell) {
  max-height: calc(100vh - 110px);
  overflow: hidden;
}

.draggable-window :deep(.window-body) {
  overflow-y: auto;
  max-height: calc(100vh - 150px);
}
</style>
