import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import Taskbar from '../Taskbar.vue'
import { useMapStore } from '../../store/map'

describe('Taskbar', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('updates release settings from the start menu', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useMapStore()
    const restartSpy = vi.spyOn(store, 'restartSystemPolling').mockImplementation(() => {})
    store.systemStatus = { status: 'online' }

    const wrapper = mount(Taskbar, {
      global: {
        plugins: [pinia],
      },
    })

    await wrapper.get('[data-testid="taskbar-start"]').trigger('click')

    expect(wrapper.text()).toContain('Хранение данных')
    expect(wrapper.text()).toContain('Стандартный')

    const selects = wrapper.findAll('select')
    await selects[0].setValue('quality')
    await selects[1].setValue('15')
    await selects[2].setValue('simple')
    await selects[4].setValue('false')
    await selects[6].setValue('true')

    expect(store.detectionPreset).toBe('quality')
    expect(store.useSam).toBe(true)
    expect(store.targetDates).toBe(9)
    expect(store.resolutionM).toBe(10)
    expect(store.autoRefreshIntervalS).toBe(15)
    expect(store.progressVerbosity).toBe('simple')
    expect(store.showFreshnessBadges).toBe(false)
    expect(store.expertMode).toBe(true)
    expect(restartSpy).toHaveBeenCalledTimes(1)
  })
})
