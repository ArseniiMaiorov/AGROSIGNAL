import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'

import StatusPanel from '../StatusPanel.vue'
import { useMapStore } from '../../store/map'

describe('StatusPanel', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('renders component health and build metadata in expert mode', () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useMapStore()
    store.expertMode = true
    store.systemStatus = {
      status: 'degraded',
      timestamp: '2026-03-08T10:00:00Z',
      components: {
        database: { status: 'online', detail: 'ok' },
        weather_provider: { status: 'degraded', detail: 'stale cache' },
      },
      build: {
        app_version: '2.0.0',
        model_version: 'boundary_unet_v2',
        train_data_version: 'train_v4',
      },
      model_truth: {
        head_count: 3,
        heads: ['extent', 'boundary', 'distance'],
        tta_standard: 'flip2',
        tta_quality: 'rotate4',
        retrain_description: 'Retrained model improves boundary quality.',
      },
      freshness: {
        freshness_state: 'stale',
        fetched_at: '2026-03-08T10:00:00Z',
      },
    }

    const wrapper = mount(StatusPanel, {
      global: {
        plugins: [pinia],
      },
    })

    expect(wrapper.text()).toContain('stale cache')
    expect(wrapper.text()).toContain('app 2.0.0')
    expect(wrapper.text()).toContain('model boundary_unet_v2')
    expect(wrapper.text()).toContain('dataset train_v4')
    expect(wrapper.text()).toContain('extent, boundary, distance')
    expect(wrapper.text()).toContain('flip2')
    expect(wrapper.text()).toContain('rotate4')
  })
})
