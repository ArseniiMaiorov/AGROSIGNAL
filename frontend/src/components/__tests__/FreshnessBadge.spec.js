import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import FreshnessBadge from '../FreshnessBadge.vue'

describe('FreshnessBadge', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2026-03-08T12:00:00Z'))
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('hides unknown freshness state behind a readable provider label', () => {
    const wrapper = mount(FreshnessBadge, {
      props: {
        compact: true,
        meta: {
          provider: 'backend',
          freshness_state: 'unknown',
          fetched_at: '2026-03-08T10:00:00Z',
        },
      },
    })

    expect(wrapper.text()).toContain('сервер · 2 ч')
    expect(wrapper.text()).not.toContain('unknown')
  })
})
