import { expect, test } from '@playwright/test'

function json(route, body, status = 200) {
  return route.fulfill({
    status,
    contentType: 'application/json',
    body: JSON.stringify(body),
  })
}

test.beforeEach(async ({ page }) => {
  const state = {
    detectSubmitted: false,
    detectFailed: false,
    statusPoll: 0,
  }

  await page.route('**/api/v1/**', async (route) => {
    const request = route.request()
    const url = new URL(request.url())
    const path = url.pathname
    const method = request.method()

    if (path.endsWith('/bootstrap')) {
      return json(route, {
        status: 'degraded',
        timestamp: '2026-03-08T12:00:00Z',
        components: {
          database: { status: 'online', detail: 'ok' },
          auth_bootstrap: { status: 'online', detail: 'ok' },
          weather_provider: { status: 'degraded', detail: 'stale cache' },
        },
        build: {
          app_version: '2.0.0',
          model_version: 'boundary_unet_v2',
          train_data_version: 'train_v4',
        },
        auth: {
          enabled: true,
          bootstrap_admin_email: 'admin@local',
          bootstrap_org_slug: 'default-organization',
          bootstrap_org_name: 'Default Organization',
        },
      })
    }

    if (path.endsWith('/auth/login') && method === 'POST') {
      return json(route, {
        access_token: 'access-1',
        refresh_token: 'refresh-1',
        token_type: 'bearer',
        expires_in: 3600,
        user: {
          id: 'user-1',
          email: 'admin@local',
          full_name: 'Bootstrap Admin',
          organization_id: 'org-1',
          organization_slug: 'default-organization',
          organization_name: 'Default Organization',
          roles: ['tenant_admin'],
          permissions: ['fields:read', 'fields:write'],
        },
      })
    }

    if (path.endsWith('/auth/logout')) {
      return json(route, { ok: true })
    }

    if (path.endsWith('/layers')) {
      return json(route, {
        layers: [
          { id: 'ndvi', name: 'NDVI' },
          { id: 'wind', name: 'Wind' },
        ],
        freshness: {
          freshness_state: 'fresh',
          fetched_at: '2026-03-08T12:00:00Z',
        },
      })
    }

    if (path.endsWith('/status')) {
      return json(route, {
        status: 'online',
        timestamp: '2026-03-08T12:00:00Z',
        components: {
          database: { status: 'online', detail: 'ok' },
          weather_provider: { status: 'online', detail: 'ok' },
        },
        runs: { running: state.detectSubmitted ? 1 : 0, total: 1 },
        build: {
          app_version: '2.0.0',
          model_version: 'boundary_unet_v2',
          train_data_version: 'train_v4',
        },
        freshness: {
          freshness_state: 'fresh',
          fetched_at: '2026-03-08T12:00:00Z',
        },
      })
    }

    if (path.endsWith('/weather/current')) {
      return json(route, {
        latitude: 45.2307,
        longitude: 38.7199,
        observed_at: '2026-03-08T12:00:00Z',
        provider: 'era5',
        cached: false,
        temperature_c: 11.2,
        apparent_temperature_c: 9.8,
        precipitation_mm: 0.4,
        wind_speed_m_s: 5.4,
        wind_direction_deg: 210,
        humidity_pct: 61,
        cloud_cover_pct: 22,
        pressure_hpa: 1007,
        soil_moisture: 0.33,
        freshness: {
          freshness_state: 'fresh',
          fetched_at: '2026-03-08T12:00:00Z',
        },
      })
    }

    if (path.endsWith('/weather/forecast')) {
      return json(route, {
        latitude: 45.2307,
        longitude: 38.7199,
        provider: 'era5',
        days: 5,
        forecast: [
          { date: '2026-03-09', temp_min_c: 4, temp_max_c: 11, precipitation_mm: 0.5 },
          { date: '2026-03-10', temp_min_c: 5, temp_max_c: 13, precipitation_mm: 0.1 },
        ],
        freshness: {
          freshness_state: 'fresh',
          fetched_at: '2026-03-08T12:00:00Z',
        },
      })
    }

    if (path.endsWith('/crops')) {
      return json(route, {
        crops: [
          { id: 1, code: 'wheat', name: 'Пшеница', category: 'grain', yield_baseline_kg_ha: 4200, ndvi_target: 0.72, base_temp_c: 5, description: null },
        ],
      })
    }

    if (path.endsWith('/fields/runs')) {
      return json(route, {
        runs: state.detectSubmitted
          ? [{ id: 'run-1', status: 'done', progress: 100, preset: 'standard' }]
          : [],
      })
    }

    if (path.endsWith('/fields/geojson')) {
      return json(route, {
        type: 'FeatureCollection',
        features: state.detectSubmitted && !state.detectFailed ? [
          {
            type: 'Feature',
            geometry: {
              type: 'Polygon',
              coordinates: [[[38.718, 45.229], [38.722, 45.229], [38.722, 45.232], [38.718, 45.232], [38.718, 45.229]]],
            },
            properties: {
              field_id: 'field-1',
              area_m2: 6400,
              source: 'autodetect',
              aoi_run_id: 'run-1',
            },
          },
        ] : [],
      })
    }

    if (path.endsWith('/manual/fields/geojson')) {
      return json(route, {
        type: 'FeatureCollection',
        features: [],
      })
    }

    if (path.endsWith('/fields/detect/preflight') && method === 'POST') {
      return json(route, {
        budget_ok: true,
        estimated_tiles: 1,
        estimated_runtime_class: 'short',
        recommended_preset: 'standard',
        warnings: [],
      })
    }

    if (path.endsWith('/fields/detect') && method === 'POST') {
      state.detectSubmitted = true
      state.statusPoll = 0
      state.detectFailed = request.headers()['x-fail-detect'] === '1'
      return json(route, {
        aoi_run_id: 'run-1',
      })
    }

    if (path.includes('/fields/status/run-1')) {
      state.statusPoll += 1
      if (state.detectFailed) {
        return json(route, {
          status: state.statusPoll < 2 ? 'running' : 'failed',
          progress: state.statusPoll < 2 ? 42 : 42,
          stage_label: state.statusPoll < 2 ? 'boundary fill' : 'failed',
          stage_detail: state.statusPoll < 2 ? 'fetch window 4/7' : 'provider timeout',
          stale_running: false,
          estimated_remaining_s: state.statusPoll < 2 ? 40 : null,
          error_msg: state.statusPoll < 2 ? null : 'provider timeout',
        })
      }
      if (state.statusPoll === 1) {
        return json(route, {
          status: 'queued',
          progress: 0,
          stage_label: 'queued',
          stage_detail: null,
          stale_running: false,
          estimated_remaining_s: 65,
        })
      }
      if (state.statusPoll === 2) {
        return json(route, {
          status: 'running',
          progress: 41,
          stage_label: 'boundary fill',
          stage_detail: 'fetch window 4/7',
          stale_running: false,
          estimated_remaining_s: 35,
        })
      }
      return json(route, {
        status: 'done',
        progress: 100,
        stage_label: 'archive',
        stage_detail: 'result ready',
        stale_running: false,
        estimated_remaining_s: 0,
      })
    }

    if (path.includes('/fields/result/run-1')) {
      return json(route, {
        status: 'done',
        progress: 100,
        runtime: 88.4,
        geojson: {
          type: 'FeatureCollection',
          features: [
            {
              type: 'Feature',
              geometry: {
                type: 'Polygon',
                coordinates: [[[38.718, 45.229], [38.722, 45.229], [38.722, 45.232], [38.718, 45.232], [38.718, 45.229]]],
              },
              properties: {
                field_id: 'field-1',
                area_m2: 6400,
                source: 'autodetect',
                aoi_run_id: 'run-1',
              },
            },
          ],
        },
      })
    }

    if (path.includes('/fields/') && path.endsWith('/dashboard')) {
      return json(route, {
        mode: 'single',
        field: {
          field_id: 'field-1',
          source: 'autodetect',
          area_m2: 6400,
        },
        kpis: {
          archive_count: 0,
          scenario_count: 0,
          prediction_ready: true,
        },
        current_metrics: {},
        series: {},
        histograms: {},
        prediction: null,
        archives: [],
        scenarios: [],
        data_quality: {
          metrics_available: ['ndvi', 'wind'],
          observation_cells: 4,
        },
      })
    }

    return json(route, { detail: `unhandled mock for ${method} ${path}` }, 404)
  })
})

test('boot, login, detect, and result retrieval complete without hanging', async ({ page }) => {
  await page.goto('/')

  await expect(page.getByTestId('boot-shell')).toBeVisible()
  await expect(page.getByTestId('auth-overlay')).toBeVisible()
  await expect(page.getByTestId('auth-email')).toHaveValue('admin@local')
  await expect(page.getByTestId('auth-org')).toHaveValue('default-organization')

  await page.getByTestId('auth-password').fill('admin12345')
  await page.getByTestId('auth-submit').click()

  await expect(page.getByTestId('weather-panel')).toBeVisible()
  await expect(page.getByTestId('status-panel')).toBeVisible()

  await page.getByTestId('start-detection').click()

  await expect(page.getByTestId('progress-shell')).toBeVisible()
  await expect(page.getByTestId('progress-shell')).toContainText('boundary fill')
  await expect(page.getByTestId('progress-shell')).toContainText('fetch window 4/7')
  await expect(page.getByText('Загружено полей: 1')).toBeVisible()
})
