import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import ProgressBar from '../ProgressBar.vue'
import { useMapStore } from '../../store/map'

describe('ProgressBar', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('renders stage label and detail in detailed mode', () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useMapStore()
    store.runProgress = 47
    store.runStageLabel = 'boundary fill'
    store.runStageDetail = 'fetch window 4/7'
    store.runStartedAt = new Date(Date.now() - 65_000).toISOString()
    store.runEstimatedRemainingS = 120
    store.progressVerbosity = 'detailed'

    const wrapper = mount(ProgressBar, {
      global: {
        plugins: [pinia],
      },
    })

    expect(wrapper.text()).toContain('47,00%')
    expect(wrapper.text()).toContain('boundary fill')
    expect(wrapper.text()).toContain('fetch window 4/7')
    expect(wrapper.text()).toContain('Прошло 1 мин')
    expect(wrapper.text()).toContain('ETA 2 мин')
  })
})
