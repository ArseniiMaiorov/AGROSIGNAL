import { mount } from '@vue/test-utils'
import { describe, expect, it, vi } from 'vitest'

vi.mock('../../services/api', () => ({
  API_BASE: '/api/v1',
  default: {
    get: vi.fn().mockResolvedValue({
      data: {
        status: 'online',
        components: {
          database: { status: 'online', detail: 1 },
          auth_bootstrap: { status: 'online', detail: 'ok' },
          classifier: { status: 'online', detail: 'ready' },
        },
        build: {
          app_version: '1.0.0',
          model_version: 'boundary_unet_v2',
          train_data_version: 'train_v3',
        },
        auth: {
          bootstrap_admin_email: 'admin@local',
          bootstrap_org_slug: 'default-organization',
          bootstrap_org_name: 'Default Organization',
        },
      },
    }),
  },
}))

import BootShell from '../BootShell.vue'

describe('BootShell', () => {
  it('emits ready after successful bootstrap', async () => {
    vi.useFakeTimers()
    const wrapper = mount(BootShell)
    await Promise.resolve()
    await Promise.resolve()
    vi.runAllTimers()

    expect(wrapper.emitted('ready')).toBeTruthy()
    vi.useRealTimers()
  })
})
