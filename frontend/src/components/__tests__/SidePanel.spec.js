import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'

import SidePanel from '../SidePanel.vue'
import { useMapStore } from '../../store/map'

describe('SidePanel', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('renders expert preflight and runtime diagnostics', () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useMapStore()
    store.expertMode = true
    store.lastPreflight = {
      estimated_tiles: 12,
      estimated_runtime_class: 'long',
      estimated_ram_mb: 1400,
      regional_profile: 'north_boundary',
      season_window: { start: '04-25', end: '10-25' },
      s1_planned: true,
      tta_mode: 'rotate4',
      launch_tier: 'experimental_rest',
      review_required: true,
      review_reason: 'Регион пока не входит в validated core.',
      review_reason_code: 'region_not_validated_core',
    }
    store.runRuntime = {
      sentinel_account_used: 'reserv',
      sentinel_failover_level: 1,
      s1_planned: true,
      tta_mode: 'rotate4',
      selected_date_confidence_mean: 0.73,
      bridge_skipped_reason: 'pair_budget_exceeded',
      region_boundary_profile: 'north_boundary',
    }

    const wrapper = mount(SidePanel, {
      global: {
        plugins: [pinia],
      },
    })

    expect(wrapper.text()).toContain('Диагностика')
    expect(wrapper.text()).toContain('north_boundary')
    expect(wrapper.text()).toContain('reserv')
    expect(wrapper.text()).toContain('rotate4')
    expect(wrapper.text()).toContain('Экспериментальный режим')
    expect(wrapper.text()).toContain('проверенный производственный контур')
  })

  it('shows sync outcome details for weather and status actions', () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useMapStore()
    store.lastWeatherUpdatedAt = '2026-03-09T00:41:03Z'
    store.lastStatusUpdatedAt = '2026-03-09T00:41:04Z'
    store.lastWeatherSyncState = 'ok'
    store.lastWeatherSyncDetail = 'openmeteo · облачность 40%'
    store.lastStatusSyncState = 'error'
    store.lastStatusSyncDetail = 'backend timeout'

    const wrapper = mount(SidePanel, {
      global: {
        plugins: [pinia],
      },
    })

    expect(wrapper.text()).toContain('openmeteo')
    expect(wrapper.text()).toContain('backend timeout')
  })

  it('locks preset tuning outside expert mode', () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useMapStore()
    store.expertMode = false

    const wrapper = mount(SidePanel, {
      global: {
        plugins: [pinia],
      },
    })

    expect(wrapper.text()).toContain('Параметры профиля зафиксированы')
    const disabledInputs = wrapper.findAll('input[disabled]')
    expect(disabledInputs.length).toBeGreaterThan(0)
  })

  it('keeps help access out of the side panel', () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useMapStore()
    store.expertMode = false

    const wrapper = mount(SidePanel, {
      global: {
        plugins: [pinia],
      },
    })

    const helpButton = wrapper.findAll('button').find((button) => button.text().includes('Справка'))
    expect(helpButton).toBeUndefined()
  })

  it('shows preset-specific radius guidance', () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useMapStore()
    store.applyDetectionPreset('quality')

    const wrapper = mount(SidePanel, {
      global: {
        plugins: [pinia],
      },
    })

    const radiusInput = wrapper.get('[data-testid="radius-km-input"]')
    expect(radiusInput.attributes('max')).toBe('8')
    expect(radiusInput.attributes('data-btip')).toContain('8 км')
  })
})
