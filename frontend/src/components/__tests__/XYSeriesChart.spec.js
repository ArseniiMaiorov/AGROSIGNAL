import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import XYSeriesChart from '../XYSeriesChart.vue'

describe('XYSeriesChart', () => {
  it('shows an empty state for a single isolated point', () => {
    const wrapper = mount(XYSeriesChart, {
      props: {
        series: [
          {
            label: 'NDVI',
            points: [{ date: '2025-03-09', value: 0.482 }],
          },
        ],
        emptyText: 'Недостаточно точек',
      },
    })

    expect(wrapper.text()).toContain('Недостаточно точек')
    expect(wrapper.find('svg').exists()).toBe(false)
  })

  it('renders a time-series chart when enough dated points are available', () => {
    const wrapper = mount(XYSeriesChart, {
      props: {
        series: [
          {
            label: 'NDVI',
            points: [
              { date: '2025-03-09', value: 0.482 },
              { date: '2025-04-13', value: 0.611 },
            ],
          },
        ],
      },
    })

    expect(wrapper.find('svg').exists()).toBe(true)
    expect(wrapper.findAll('circle').length).toBeGreaterThanOrEqual(2)
    expect(wrapper.text()).toContain('NDVI')
  })

  it('shows an empty state when only markers are provided without a line series', () => {
    const wrapper = mount(XYSeriesChart, {
      props: {
        series: [],
        markers: [
          { label: 'Forecast', date: 2026, value: 3374 },
        ],
        emptyText: 'Нет истории',
      },
    })

    expect(wrapper.text()).toContain('Нет истории')
    expect(wrapper.find('svg').exists()).toBe(false)
  })
})
