import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'

import WeatherPanel from '../WeatherPanel.vue'
import { useMapStore } from '../../store/map'

describe('WeatherPanel', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('renders wind direction and forecast freshness context', () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useMapStore()
    store.weatherCurrent = {
      temperature_c: 12.4,
      apparent_temperature_c: 10.8,
      wind_speed_m_s: 4.2,
      wind_direction_deg: 215,
      precipitation_mm: 0.6,
      humidity_pct: 65,
      cloud_cover_pct: 22,
      pressure_hpa: 1008,
      soil_moisture: 0.34,
      freshness: {
        provider: 'era5',
        freshness_state: 'fresh',
        fetched_at: '2026-03-08T10:00:00Z',
      },
      error: 'Текущая погода временно недоступна',
    }
    store.weatherForecast = [
      { date: '2026-03-09', temp_min_c: 4, temp_max_c: 10, precipitation_mm: 1.1 },
      { date: '2026-03-10', temp_min_c: 5, temp_max_c: 12, precipitation_mm: 0.2 },
    ]

    const wrapper = mount(WeatherPanel, {
      global: {
        plugins: [pinia],
      },
    })

    expect(wrapper.text()).toContain('4.2 м/с · ЮЗ')
    expect(wrapper.text()).toContain('12.4 °C')
    expect(wrapper.text()).toContain('09.03')
    expect(wrapper.text()).toContain('Текущая погода временно недоступна')
  })
})
