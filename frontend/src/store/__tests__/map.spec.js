import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { nextTick } from 'vue'

const apiMock = vi.hoisted(() => ({
  get: vi.fn(),
  post: vi.fn(),
  patch: vi.fn(),
  delete: vi.fn(),
}))

vi.mock('../../services/api', () => ({
  API_BASE: '/api/v1',
  default: apiMock,
}))

import { useMapStore } from '../map'

describe('map store', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    apiMock.get.mockReset()
    apiMock.post.mockReset()
    apiMock.patch.mockReset()
    apiMock.delete.mockReset()
    window.localStorage.clear()
  })

  it('does not trigger event API calls during store creation', () => {
    useMapStore()

    expect(apiMock.post).not.toHaveBeenCalled()
    expect(apiMock.patch).not.toHaveBeenCalled()
    expect(apiMock.delete).not.toHaveBeenCalled()
  })

  it('applies release detection presets deterministically', () => {
    const store = useMapStore()

    store.applyDetectionPreset('fast')
    expect(store.detectionPreset).toBe('fast')
    expect(store.useSam).toBe(false)
    expect(store.targetDates).toBe(4)
    expect(store.resolutionM).toBe(10)
    expect(store.minFieldAreaHa).toBe(0.5)
    expect(store.activeDetectionPreset.maxRadiusKm).toBe(40)
    expect(store.activeDetectionPreset.recommendedRadiusKm).toBe(30)

    store.applyDetectionPreset('quality')
    expect(store.detectionPreset).toBe('quality')
    expect(store.useSam).toBe(true)
    expect(store.targetDates).toBe(9)
    expect(store.resolutionM).toBe(10)
    expect(store.minFieldAreaHa).toBe(0.1)
    expect(store.activeDetectionPreset.maxRadiusKm).toBe(8)
    expect(store.activeDetectionPreset.recommendedRadiusKm).toBe(8)

    store.applyDetectionPreset('standard')
    expect(store.detectionPreset).toBe('standard')
    expect(store.useSam).toBe(false)
    expect(store.targetDates).toBe(7)
    expect(store.resolutionM).toBe(10)
    expect(store.minFieldAreaHa).toBe(0.25)
    expect(store.activeDetectionPreset.maxRadiusKm).toBe(20)
    expect(store.activeDetectionPreset.recommendedRadiusKm).toBe(20)
  })

  it('clamps the search radius to preset-safe limits', () => {
    const store = useMapStore()

    store.radiusKm = 37
    store.applyDetectionPreset('quality')
    expect(store.radiusKm).toBe(8)

    store.radiusKm = 29
    store.applyDetectionPreset('standard')
    expect(store.radiusKm).toBe(20)

    store.radiusKm = 12
    store.applyDetectionPreset('fast')
    expect(store.radiusKm).toBe(30)
  })

  it('restores versioned UI settings from local storage', async () => {
    apiMock.get.mockResolvedValue({ data: {} })
    window.localStorage.setItem(
      'agrovision-ui-settings-v2',
      JSON.stringify({
        centerLat: 55.75,
        centerLon: 37.61,
        detectionPreset: 'quality',
        autoRefreshIntervalS: 15,
        progressVerbosity: 'simple',
        showFreshnessBadges: false,
        uiWindows: {
          status: false,
          weather: false,
        },
      })
    )

    const store = useMapStore()
    await store.initialize()
    store.clearTimers()

    expect(store.centerLat).toBe(55.75)
    expect(store.centerLon).toBe(37.61)
    expect(store.detectionPreset).toBe('quality')
    expect(store.autoRefreshIntervalS).toBe(15)
    expect(store.progressVerbosity).toBe('simple')
    expect(store.showFreshnessBadges).toBe(false)
    expect(store.uiWindows.status).toBe(false)
    expect(store.uiWindows.weather).toBe(false)
    expect(store.uiWindows.control).toBe(true)
  })

  it('normalizes scenario inputs before API submission', async () => {
    apiMock.post.mockResolvedValueOnce({
      data: {
        task_id: 'scenario-job-1',
        status: 'queued',
        progress: 0,
      },
    })
    apiMock.get.mockResolvedValueOnce({
      data: {
        task_id: 'scenario-job-1',
        status: 'done',
        progress: 100,
        stage_label: 'done',
        result_ready: true,
      },
    })
    apiMock.get.mockResolvedValueOnce({
      data: {
        task_id: 'scenario-job-1',
        status: 'done',
        progress: 100,
        result: {
          field_id: 'field-1',
          baseline_yield_kg_ha: 4200,
          scenario_yield_kg_ha: 4368,
          predicted_yield_change_pct: 4,
        },
      },
    })
    apiMock.get.mockResolvedValueOnce({
      data: {
        mode: 'single',
        field: {
          field_id: 'field-1',
        },
        prediction: {
          estimated_yield_kg_ha: 4300,
          prediction_date: '2026-03-09T00:00:00Z',
        },
        scenarios: [],
        archives: [],
      },
    })

    const store = useMapStore()
    store.selectedField = { field_id: 'field-1' }
    store.useManualModeling = true
    store.modelingForm = {
      irrigation_pct: 140,
      fertilizer_pct: -180,
      expected_rain_mm: 900,
      temperature_delta_c: 20,
      planting_density_pct: 180,
      tillage_type: 9,
      pest_pressure: 8,
      soil_compaction: 2,
    }

    await store.simulateScenario()

    expect(apiMock.post).toHaveBeenCalledWith(
      '/api/v1/modeling/jobs',
      expect.objectContaining({
        field_id: 'field-1',
        irrigation_pct: 100,
        fertilizer_pct: -100,
        expected_rain_mm: 500,
        temperature_delta_c: 10,
        planting_density_pct: 100,
        tillage_type: 3,
        pest_pressure: 3,
        soil_compaction: 1,
      }),
      expect.any(Object)
    )
    expect(store.modelingForm.irrigation_pct).toBe(100)
    expect(store.modelingForm.fertilizer_pct).toBe(-100)
    expect(store.modelingForm.expected_rain_mm).toBe(500)
    expect(store.modelingForm.temperature_delta_c).toBe(10)
    expect(store.modelingForm.planting_density_pct).toBe(100)
    expect(store.modelingForm.tillage_type).toBe(3)
    expect(store.modelingForm.pest_pressure).toBe(3)
    expect(store.modelingForm.soil_compaction).toBe(1)
  })

  it('replaces stale manual scenario values with satellite-derived auto values when auto mode is enabled', async () => {
    const store = useMapStore()
    store.useManualModeling = true
    store.modelingForm = {
      irrigation_pct: 14,
      fertilizer_pct: 9,
      expected_rain_mm: 137,
      temperature_delta_c: 0,
      planting_density_pct: 0,
      tillage_type: null,
      pest_pressure: null,
      soil_compaction: 0.91,
      cloud_cover_factor: 1.0,
    }
    store.fieldTemporalAnalytics = {
      seasonal_series: {
        metrics: [
          { metric: 'soil_moisture', points: [{ value: 0.32 }, { value: 0.31 }] },
          { metric: 'ndwi', points: [{ value: -0.05 }] },
          { metric: 'ndmi', points: [{ value: 0.18 }] },
          { metric: 'bsi', points: [{ value: 0.06 }] },
        ],
      },
    }

    store.enableAutoModeling()
    await nextTick()

    expect(store.modelingForm.expected_rain_mm).toBe(46)
    expect(store.modelingAutoSources.expected_rain_mm).toBe('satellite_wetness_bsi')
    expect(store.modelingForm.soil_compaction).toBe(0.48)
    expect(store.modelingAutoSources.soil_compaction).toBe('satellite_soil_moisture_bsi')
  })

  it('runs prediction refresh through async jobs with polling', async () => {
    apiMock.post.mockResolvedValueOnce({
      data: {
        task_id: 'prediction-job-1',
        status: 'queued',
        progress: 0,
      },
    })
    apiMock.get.mockResolvedValueOnce({
      data: {
        task_id: 'prediction-job-1',
        status: 'done',
        progress: 100,
        stage_label: 'done',
        result_ready: true,
      },
    })
    apiMock.get.mockResolvedValueOnce({
      data: {
        task_id: 'prediction-job-1',
        status: 'done',
        progress: 100,
        result: {
          field_id: 'field-1',
          estimated_yield_kg_ha: 4300,
          prediction_date: '2026-03-09T00:00:00Z',
        },
      },
    })
    apiMock.get.mockResolvedValueOnce({
      data: {
        mode: 'single',
        field: {
          field_id: 'field-1',
        },
        scenarios: [],
        archives: [],
      },
    })

    const store = useMapStore()
    store.selectedField = { field_id: 'field-1' }

    const result = await store.refreshPrediction(true)

    expect(apiMock.post).toHaveBeenCalledWith(
      '/api/v1/predictions/field/field-1/jobs',
      null,
      expect.objectContaining({
        params: expect.objectContaining({
          refresh: true,
        }),
      })
    )
    expect(result.estimated_yield_kg_ha).toBe(4300)
  })

  it('clears stale temporal analytics when a new date-range load fails', async () => {
    apiMock.get.mockRejectedValueOnce(new Error('network down'))

    const store = useMapStore()
    store.selectedField = { field_id: 'field-1' }
    store.fieldTemporalAnalytics = {
      seasonal_series: { metrics: [{ metric: 'ndvi', points: [{ observed_at: '2025-03-09', value: 0.4 }] }] },
    }

    const result = await store.loadFieldTemporalAnalytics('field-1', {
      target: 'metrics',
      dateFrom: '2025-03-01',
      dateTo: '2025-08-31',
      preferExisting: false,
      silent: true,
    })

    expect(result).toBeNull()
    expect(store.fieldTemporalAnalytics).toBeNull()
  })

  it('keeps the previous visible run until a new detect result is fully loaded', async () => {
    apiMock.get
      .mockResolvedValueOnce({ data: { runs: [{ id: 'run-old', status: 'done', progress: 100 }] } })
      .mockResolvedValueOnce({ data: {} })
      .mockResolvedValueOnce({ data: {} })
      .mockResolvedValueOnce({ data: { forecast: [] } })
      .mockResolvedValueOnce({ data: { crops: [] } })
      .mockResolvedValueOnce({
        data: {
          type: 'FeatureCollection',
          features: [
            {
              type: 'Feature',
              geometry: null,
              properties: {
                field_id: 'field-old',
                aoi_run_id: 'run-old',
              },
            },
          ],
        },
      })
      .mockResolvedValueOnce({ data: { type: 'FeatureCollection', features: [] } })
      .mockResolvedValueOnce({
        data: {
          status: 'running',
          progress: 24,
          stage_label: 'fetch',
          stage_detail: 'windows 2/4',
        },
      })
      .mockResolvedValueOnce({
        data: {
          aoi_run_id: 'run-new',
          status: 'done',
          progress: 100,
          stage_label: 'complete',
          geojson: {
            type: 'FeatureCollection',
            features: [
              {
                type: 'Feature',
                geometry: null,
                properties: {
                  field_id: 'field-new',
                  aoi_run_id: 'run-new',
                },
              },
            ],
          },
        },
      })
      .mockResolvedValueOnce({
        data: {
          type: 'FeatureCollection',
          features: [
            {
              type: 'Feature',
              geometry: null,
              properties: {
                field_id: 'field-new',
                aoi_run_id: 'run-new',
              },
            },
          ],
        },
      })
      .mockResolvedValueOnce({ data: { type: 'FeatureCollection', features: [] } })
      .mockResolvedValueOnce({ data: { runs: [{ id: 'run-new', status: 'done', progress: 100 }] } })
    apiMock.post
      .mockResolvedValueOnce({
        data: {
          budget_ok: true,
          estimated_tiles: 2,
          estimated_runtime_class: 'medium',
          recommended_preset: 'standard',
          warnings: [],
        },
      })
      .mockResolvedValueOnce({
        data: {
          aoi_run_id: 'run-new',
          status: 'queued',
        },
      })

    const store = useMapStore()
    await store.initialize()
    expect(store.visibleRunId).toBe('run-old')

    await store.startDetection()
    expect(store.activeRunId).toBe('run-new')
    expect(store.visibleRunId).toBe('run-old')

    expect(store.visibleRunId).toBe('run-old')

    await store.fetchResult('run-new')
    expect(store.visibleRunId).toBe('run-new')
    expect(store.lastCompletedRunId).toBe('run-new')
  })

  it('allows a soft-budget quality run when preflight is not hard-blocked', async () => {
    apiMock.post
      .mockResolvedValueOnce({
        data: {
          budget_ok: false,
          hard_block: false,
          launch_tier: 'review_needed',
          review_required: true,
          review_reason: 'Quality run still needs operator review.',
          estimated_tiles: 15,
          estimated_runtime_class: 'extreme',
          estimated_ram_mb: 2400,
          regional_profile: 'south_boundary',
          tta_mode: 'rotate4',
          s1_planned: true,
          reason: 'Запуск разрешён, но расчёт может занять заметно больше времени.',
          warnings: ['long run'],
        },
      })
      .mockResolvedValueOnce({
        data: {
          aoi_run_id: 'run-quality',
          status: 'queued',
        },
      })
    apiMock.get
      .mockResolvedValueOnce({
        data: {
          status: 'running',
          progress: 5,
          stage_label: 'fetch',
          stage_detail: 'windows 0/9',
        },
      })
      .mockResolvedValue({ data: { runs: [] } })

    const store = useMapStore()
    store.applyDetectionPreset('quality')

    await store.startDetection()
    store.clearTimers()

    expect(store.activeRunId).toBe('run-quality')
    expect(store.lastPreflight.hard_block).toBe(false)
    expect(apiMock.post).toHaveBeenCalledTimes(2)
    expect(store.logs.some((entry) => entry.message.includes('Запуск разрешён'))).toBe(true)
    expect(store.logs.some((entry) => entry.message.includes('review'))).toBe(true)
  })

  it('locks detect payload to preset values outside expert mode', async () => {
    apiMock.post
      .mockResolvedValueOnce({
        data: {
          budget_ok: true,
          hard_block: false,
          launch_tier: 'validated_core',
          review_required: false,
          estimated_tiles: 2,
          estimated_runtime_class: 'medium',
          warnings: [],
        },
      })
      .mockResolvedValueOnce({
        data: {
          aoi_run_id: 'run-standard',
          status: 'queued',
        },
      })
    apiMock.get.mockResolvedValue({ data: { runs: [] } })

    const store = useMapStore()
    store.expertMode = false
    store.applyDetectionPreset('standard')
    store.targetDates = 12
    store.resolutionM = 60
    store.minFieldAreaHa = 3
    store.useSam = true

    await store.startDetection()
    store.clearTimers()

    const [, payload] = apiMock.post.mock.calls[0]
    expect(payload.target_dates).toBe(7)
    expect(payload.resolution_m).toBe(10)
    expect(payload.min_field_area_ha).toBe(0.25)
  })

  it('asks for confirmation before deleting a field', async () => {
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(false)
    const store = useMapStore()
    store.selectedField = { field_id: 'field-1' }

    const result = await store.deleteSelectedField()

    expect(result).toBe(false)
    expect(confirmSpy).toHaveBeenCalled()
    expect(apiMock.delete).not.toHaveBeenCalled()
    expect(store.logs.some((entry) => entry.message.includes('Удаление поля отменено'))).toBe(true)
    confirmSpy.mockRestore()
  })

  it('deletes a field only after explicit confirmation', async () => {
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true)
    apiMock.delete.mockResolvedValueOnce({ data: {} })
    const store = useMapStore()
    store.selectedField = { field_id: 'field-2' }
    store.fieldsGeoJson = {
      type: 'FeatureCollection',
      features: [{ type: 'Feature', geometry: null, properties: { field_id: 'field-2' } }],
    }

    const result = await store.deleteSelectedField()

    expect(result).toBe(true)
    expect(apiMock.delete).toHaveBeenCalledWith('/api/v1/fields/field-2')
    confirmSpy.mockRestore()
  })

  it('loads debug tile catalog, tile detail and layer payload for expert diagnostics', async () => {
    apiMock.get
      .mockResolvedValueOnce({
        data: {
          run_id: 'run-debug',
          tiles: [
            {
              tile_id: 'tile-01',
              available_layers: [{ id: 'after_merge', label: 'После merge' }],
            },
          ],
        },
      })
      .mockResolvedValueOnce({
        data: {
          tile_id: 'tile-01',
          available_layers: [{ id: 'after_merge', label: 'После merge' }],
          runtime_meta: {
            components_after_merge: 5,
            components_after_watershed: 6,
            split_score_p50: 0.42,
          },
        },
      })
      .mockResolvedValueOnce({
        data: {
          layer_name: 'after_merge',
          image_base64: 'ZmFrZS1wbmc=',
          bounds: [30, 50, 31, 51],
          opacity_default: 0.55,
        },
      })

    const store = useMapStore()

    const tiles = await store.loadRunDebugTiles('run-debug')
    expect(tiles).toHaveLength(1)
    expect(store.selectedDebugRunId).toBe('run-debug')
    expect(store.selectedDebugTileId).toBe('tile-01')
    expect(store.selectedDebugLayerId).toBe('after_merge')
    expect(store.selectedDebugTileDetail.runtime_meta.components_after_merge).toBe(5)

    const layer = await store.loadDebugLayer('run-debug', 'tile-01', 'after_merge', { silent: true })
    expect(layer.layer_name).toBe('after_merge')
    expect(store.debugLayerPayload.bounds).toEqual([30, 50, 31, 51])
  })

  it('loads field dashboard temporal analytics from dedicated temporal endpoints', async () => {
    apiMock.get
      .mockResolvedValueOnce({
        data: {
          mode: 'single',
          field: { field_id: 'field-1' },
          prediction: {
            estimated_yield_kg_ha: 4200,
            seasonal_series: {
              metrics: [
                {
                  metric: 'ndvi',
                  points: [{ observed_at: '1999-01-01', value: 0.1 }],
                },
              ],
            },
          },
          scenarios: [],
          archives: [],
        },
      })
      .mockResolvedValueOnce({
        data: {
          seasonal_series: {
            metrics: [
              {
                metric: 'ndvi',
                points: [
                  { observed_at: '2025-03-01', value: 0.41 },
                  { observed_at: '2025-03-15', value: 0.53 },
                ],
              },
            ],
          },
          data_status: { code: 'ready', message_code: 'temporal_ready' },
        },
      })
      .mockResolvedValueOnce({
        data: {
          seasonal_series: {
            metrics: [
              {
                metric: 'ndvi',
                points: [{ observed_at: '2026-03-09', value: 0.48 }],
              },
            ],
          },
          data_status: {
            code: 'insufficient_points_current_season',
            message_code: 'temporal_insufficient_points_current_season',
          },
        },
      })
      .mockResolvedValueOnce({
        data: {
          summary: { supported: false },
          zones: [],
        },
      })

    const store = useMapStore()
    const result = await store.loadFieldDashboard('field-1')

    expect(result.field.field_id).toBe('field-1')
    expect(store.fieldTemporalAnalytics.seasonal_series.metrics[0].points).toHaveLength(2)
    expect(store.fieldTemporalAnalytics.seasonal_series.metrics[0].points[0].observed_at).toBe('2025-03-01')
    expect(store.fieldForecastAnalytics.seasonal_series.metrics[0].points).toHaveLength(1)
    expect(apiMock.get).toHaveBeenNthCalledWith(2, '/api/v1/fields/field-1/temporal-analytics', expect.any(Object))
  })

  it('starts temporal backfill for historical ranges and refreshes the series after job completion', async () => {
    apiMock.get
      .mockResolvedValueOnce({
        data: {
          seasonal_series: {
            metrics: [
              {
                metric: 'ndvi',
                points: [],
              },
            ],
          },
          data_status: {
            code: 'backfill_required',
            message_code: 'temporal_backfill_required',
            backfill_required: true,
            requested_range: {
              date_from: '2025-03-01',
              date_to: '2025-08-31',
            },
          },
        },
      })
      .mockResolvedValueOnce({
        data: {
          task_id: 'temporal-job-1',
          status: 'done',
          progress: 100,
          stage_label: 'done',
          result_ready: true,
        },
      })
      .mockResolvedValueOnce({
        data: {
          task_id: 'temporal-job-1',
          status: 'done',
          progress: 100,
          result: {
            data_status: {
              code: 'ready',
              message_code: 'temporal_ready',
            },
          },
        },
      })
      .mockResolvedValueOnce({
        data: {
          seasonal_series: {
            metrics: [
              {
                metric: 'ndvi',
                points: [
                  { observed_at: '2025-03-01', value: 0.41 },
                  { observed_at: '2025-03-15', value: 0.53 },
                ],
              },
            ],
          },
          data_status: {
            code: 'ready',
            message_code: 'temporal_ready',
            backfill_required: false,
          },
        },
      })
    apiMock.post.mockResolvedValueOnce({
      data: {
        task_id: 'temporal-job-1',
        status: 'queued',
        progress: 0,
      },
    })

    const store = useMapStore()
    store.selectedField = { field_id: 'field-1' }

    await store.loadFieldTemporalAnalytics('field-1', {
      dateFrom: '2025-03-01',
      dateTo: '2025-08-31',
      preferExisting: false,
      autoBackfill: true,
      silent: true,
    })

    expect(apiMock.post).toHaveBeenCalledWith(
      '/api/v1/fields/field-1/temporal-analytics/jobs',
      null,
      expect.objectContaining({
        params: expect.objectContaining({
          date_from: '2025-03-01',
          date_to: '2025-08-31',
        }),
      }),
    )
    expect(store.fieldTemporalAnalytics.seasonal_series.metrics[0].points).toHaveLength(2)
  })

  it('blocks forecast, scenarios and archive for preview contours', async () => {
    const store = useMapStore()
    store.selectedField = { field_id: 'preview-1', source: 'autodetect_preview' }
    store.fieldDashboard = { field: { field_id: 'preview-1', source: 'autodetect_preview' } }

    await expect(store.refreshPrediction()).resolves.toBeNull()
    await expect(store.simulateScenario()).resolves.toBeNull()
    await expect(store.createArchiveForSelectedField()).resolves.toBeNull()

    expect(apiMock.post).not.toHaveBeenCalled()
    expect(store.logs.some((entry) => entry.message.includes('Standard') || entry.message.includes('field boundaries'))).toBe(true)
  })
})
