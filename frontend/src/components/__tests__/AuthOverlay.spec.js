import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'

import AuthOverlay from '../AuthOverlay.vue'

describe('AuthOverlay', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('prefills bootstrap email and organization slug', async () => {
    const wrapper = mount(AuthOverlay, {
      props: {
        initialEmail: 'admin@local',
        initialOrganizationSlug: 'default-organization',
        initialOrganizationName: 'Default Organization',
      },
      global: {
        plugins: [createPinia()],
      },
    })

    const inputs = wrapper.findAll('input')
    expect(inputs[0].element.value).toBe('admin@local')
    expect(inputs[2].element.value).toBe('default-organization')
    expect(wrapper.text()).toContain('Организация по умолчанию')
    expect(wrapper.text()).toContain('ТерраINFO')
  })
})
