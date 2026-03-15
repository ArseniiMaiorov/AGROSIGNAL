import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import HelpPanel from '../HelpPanel.vue'

describe('HelpPanel', () => {
  it('renders core help sections', () => {
    const wrapper = mount(HelpPanel)

    expect(wrapper.text()).toContain('Справка')
    expect(wrapper.text()).toContain('Автодетект полей')
    expect(wrapper.text()).toContain('Прогноз урожайности')
    expect(wrapper.text()).toContain('Графики и как их читать')
    expect(wrapper.text()).toContain('История и прогноз')
    expect(wrapper.text()).toContain('Сценарии')
  })
})
