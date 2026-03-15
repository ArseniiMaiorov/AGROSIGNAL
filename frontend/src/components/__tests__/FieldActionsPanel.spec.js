import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import FieldActionsPanel from '../FieldActionsPanel.vue'
import { useMapStore } from '../../store/map'

describe('FieldActionsPanel', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('renders forecast trust metadata for a selected field', () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useMapStore()
    store.selectedFieldIds = ['field-1']
    store.selectedField = {
      field_id: 'field-1',
      source: 'autodetect',
      quality_score: 0.84,
      quality_label: 'high',
      quality_reason: 'Контур стабилен.',
      operational_tier: 'validated_core',
      review_required: false,
    }
    store.activeFieldTab = 'forecast'
    store.fieldDashboard = {
      field: store.selectedField,
      kpis: { prediction_ready: true, archive_count: 0, scenario_count: 0, observation_cells: 12 },
      data_quality: { metrics_available: ['ndvi'] },
      prediction: {
        estimated_yield_kg_ha: 4200,
        confidence: 0.78,
        model_version: 'agronomy_tabular_v3',
        prediction_date: '2026-03-09T00:00:00Z',
        confidence_tier: 'global_baseline',
        operational_tier: 'experimental_rest',
        review_required: true,
        review_reason: 'Нужна проверка агрономом.',
        review_reason_code: 'global_baseline_requires_review',
        crop_suitability: { status: 'low', score: 0.41 },
        freshness: null,
        explanation: { summary: 'Тестовое объяснение.', drivers: [] },
        input_features: {},
        data_quality: {},
      },
      scenarios: [],
      archives: [],
    }

    const wrapper = mount(FieldActionsPanel, {
      global: {
        plugins: [pinia],
        stubs: {
          FreshnessBadge: true,
          MiniHistogram: true,
          MiniSparkline: true,
        },
      },
    })

    expect(wrapper.text()).toContain('Операционный контур')
    expect(wrapper.text()).toContain('Экспериментальный режим')
    expect(wrapper.text()).toContain('Использован общий baseline без локальной калибровки')
  })

  it('renders extended scenario controls and async logs', () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useMapStore()
    store.selectedFieldIds = ['field-1']
    store.selectedField = {
      field_id: 'field-1',
      source: 'autodetect',
      quality_score: 0.7,
      quality_label: 'medium',
    }
    store.activeFieldTab = 'scenarios'
    store.crops = [{ code: 'wheat', name: 'Пшеница' }]
    store.fieldDashboard = {
      field: store.selectedField,
      kpis: { prediction_ready: true, archive_count: 0, scenario_count: 0, observation_cells: 8 },
      data_quality: { metrics_available: ['ndvi'] },
      prediction: {
        estimated_yield_kg_ha: 4100,
        confidence: 0.72,
        prediction_interval: { lower: 3800, upper: 4400 },
      },
      scenarios: [],
      archives: [],
    }
    store.scenarioTaskProgress = 44
    store.scenarioTaskState = {
      stage_label: 'running',
      stage_detail: 'counterfactual scoring',
      elapsed_s: 25,
      estimated_remaining_s: 11,
      logs: ['Сценарий: 12% (prepare)', 'Сценарий: 44% (running)'],
    }
    store.modelingResult = {
      baseline_yield_kg_ha: 4100,
      scenario_yield_kg_ha: 3980,
      predicted_yield_change_pct: -2.93,
      risk_summary: { level: 'умеренный', level_code: 'moderate', comment: 'Тестовый риск.' },
      assumptions: {},
      comparison: { factor_breakdown: [] },
      operational_tier: 'review_needed',
      review_required: true,
      review_reason: 'Сценарий выходит на границу применимости.',
      review_reason_code: 'outside_training_envelope',
      freshness: null,
      constraint_warnings: [],
    }

    const wrapper = mount(FieldActionsPanel, {
      global: {
        plugins: [pinia],
        stubs: {
          FreshnessBadge: true,
          MiniHistogram: true,
          MiniSparkline: true,
        },
      },
    })

    expect(wrapper.text()).toContain('Температура, delta C')
    expect(wrapper.text()).toContain('Обработка почвы')
    expect(wrapper.text()).toContain('Давление вредителей')
    expect(wrapper.text()).toContain('Сценарий: 44% (running)')
    expect(wrapper.text()).toContain('Нужна проверка')
  })

  it('renders future weather and cumulative GDD sections for forecast and scenario views', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useMapStore()
    store.selectedFieldIds = ['field-1']
    store.selectedField = {
      field_id: 'field-1',
      source: 'autodetect',
      quality_score: 0.82,
      quality_label: 'high',
    }
    store.fieldDashboard = {
      field: store.selectedField,
      kpis: { prediction_ready: true, archive_count: 0, scenario_count: 0, observation_cells: 12 },
      data_quality: { metrics_available: ['ndvi'] },
      prediction: {
        estimated_yield_kg_ha: 4200,
        confidence: 0.78,
        model_version: 'agronomy_tabular_v3',
        prediction_date: '2026-03-09T00:00:00Z',
        confidence_tier: 'global_baseline',
        operational_tier: 'validated_core',
        review_required: false,
        crop_suitability: { status: 'moderate', score: 0.64 },
        explanation: { summary: 'Тестовое объяснение.', drivers: [] },
        input_features: {},
        data_quality: {},
        forecast_curve: {
          points: [
            { date: '2026-03-12', temperature_mean_c: 11.0, precipitation_mm: 2.0, gdd_daily: 6.0, gdd_cumulative: 6.0 },
            { date: '2026-03-13', temperature_mean_c: 12.0, precipitation_mm: 1.0, gdd_daily: 7.0, gdd_cumulative: 13.0 },
          ],
        },
      },
      scenarios: [],
      archives: [],
    }

    store.activeFieldTab = 'forecast'
    let wrapper = mount(FieldActionsPanel, {
      global: {
        plugins: [pinia],
        stubs: {
          FreshnessBadge: true,
          MiniHistogram: true,
          MiniSparkline: true,
        },
      },
    })

    expect(wrapper.text()).toContain('Будущая погода и накопленные температуры')

    store.activeFieldTab = 'scenarios'
    store.modelingResult = {
      baseline_yield_kg_ha: 4100,
      scenario_yield_kg_ha: 4250,
      predicted_yield_change_pct: 3.66,
      risk_summary: { level: 'низкий', level_code: 'low', comment: 'Тестовый риск.' },
      assumptions: {},
      comparison: { factor_breakdown: [] },
      operational_tier: 'validated_core',
      review_required: false,
      constraint_warnings: [],
      forecast_curve: {
        baseline_points: [
          { date: '2026-03-12', temperature_mean_c: 11.0, precipitation_mm: 2.0, gdd_daily: 6.0, gdd_cumulative: 6.0 },
        ],
        scenario_points: [
          { date: '2026-03-12', temperature_mean_c: 13.0, precipitation_mm: 3.0, gdd_daily: 8.0, gdd_cumulative: 8.0 },
        ],
      },
    }
    await wrapper.vm.$nextTick()

    expect(wrapper.text()).toContain('Будущая погода и накопленные температуры сценария')
  })

  it('formats prediction interval and hides internal empty features outside expert mode', () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useMapStore()
    store.selectedFieldIds = ['field-1']
    store.selectedField = {
      field_id: 'field-1',
      source: 'autodetect',
      quality_score: 0.82,
      quality_label: 'high',
    }
    store.activeFieldTab = 'forecast'
    store.fieldDashboard = {
      field: store.selectedField,
      kpis: { prediction_ready: true, archive_count: 0, scenario_count: 0, observation_cells: 12 },
      data_quality: { metrics_available: ['ndvi'] },
      prediction: {
        estimated_yield_kg_ha: 3374,
        confidence: 0.5,
        model_version: 'global_baseline_v3',
        prediction_date: '2026-03-12T00:00:00Z',
        confidence_tier: 'global_baseline',
        operational_tier: 'experimental_rest',
        review_required: true,
        review_reason_code: 'global_baseline_requires_review',
        crop_suitability: { status: 'high', score: 1.0 },
        explanation: { summary: 'Тестовое объяснение.', drivers: [] },
        input_features: {
          current_ndvi_mean: 0.46,
          seasonal_gdd_sum: null,
          _mgmt_fertilizer: null,
        },
        data_quality: {
          prediction_interval: { lower: 2900, upper: 3800 },
        },
      },
      scenarios: [],
      archives: [],
    }

    const wrapper = mount(FieldActionsPanel, {
      global: {
        plugins: [pinia],
        stubs: {
          FreshnessBadge: true,
          MiniHistogram: true,
          MiniSparkline: true,
        },
      },
    })

    expect(wrapper.text()).toContain('2900 … 3800 кг/га')
    expect(wrapper.text()).not.toContain('[object Object]')
    expect(wrapper.text()).toContain('Текущий NDVI')
    expect(wrapper.text()).not.toContain('Mgmt fertilizer')
    expect(wrapper.text()).not.toContain('Сумма GDD за сезон—')
  })

  it('renders expert-only geometry diagnostics when debug runtime is available', () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useMapStore()
    store.expertMode = true
    store.selectedFieldIds = ['field-1']
    store.selectedField = {
      field_id: 'field-1',
      source: 'autodetect',
      quality_score: 0.91,
      quality_label: 'high',
      geometry_confidence: 0.83,
      tta_consensus: 0.88,
      boundary_uncertainty: 0.12,
    }
    store.activeFieldTab = 'forecast'
    store.selectedDebugTileDetail = {
      runtime_meta: {
        components_after_grow: 12,
        components_after_gap_close: 9,
        components_after_infill: 8,
        components_after_merge: 5,
        components_after_watershed: 6,
        split_score_p50: 0.41,
        split_score_p90: 0.77,
        watershed_skipped_reason: 'low_split_score',
      },
    }
    store.fieldDashboard = {
      field: store.selectedField,
      analytics_summary: {
        heads: ['extent', 'boundary', 'distance'],
        head_count: 3,
        tta_standard: 'flip2',
        tta_quality: 'rotate4',
      },
      kpis: { prediction_ready: true, archive_count: 0, scenario_count: 0, observation_cells: 12 },
      data_quality: { metrics_available: ['ndvi'] },
      prediction: {
        estimated_yield_kg_ha: 4200,
        confidence: 0.78,
        model_version: 'agronomy_tabular_v3',
        prediction_date: '2026-03-09T00:00:00Z',
        confidence_tier: 'global_baseline',
        operational_tier: 'experimental_rest',
        review_required: true,
        review_reason: 'Нужна проверка агрономом.',
        review_reason_code: 'global_baseline_requires_review',
        crop_suitability: { status: 'low', score: 0.41 },
        freshness: null,
        explanation: { summary: 'Тестовое объяснение.', drivers: [] },
        input_features: {},
        data_quality: {},
      },
      scenarios: [],
      archives: [],
    }

    const wrapper = mount(FieldActionsPanel, {
      global: {
        plugins: [pinia],
        stubs: {
          FreshnessBadge: true,
          MiniHistogram: true,
          MiniSparkline: true,
        },
      },
    })

    expect(wrapper.text()).toContain('Геометрия и сегментация')
    expect(wrapper.text()).toContain('Уверенность геометрии')
    expect(wrapper.text()).toContain('83%')
    expect(wrapper.text()).toContain('После merge')
    expect(wrapper.text()).toContain('5')
    expect(wrapper.text()).toContain('Watershed пропущен: low_split_score')
  })

  it('restores seasonal analytics modes and renders anomaly rows', () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useMapStore()
    store.selectedFieldIds = ['field-1']
    store.selectedField = {
      field_id: 'field-1',
      source: 'autodetect',
      quality_score: 0.88,
      quality_label: 'high',
    }
    store.activeFieldTab = 'metrics'
    store.metricsDisplayMode = 'anomalies'
    store.metricsSelectedSeries = 'ndvi'
    store.fieldDashboard = {
      field: store.selectedField,
      kpis: { prediction_ready: true, archive_count: 0, scenario_count: 0, observation_cells: 14 },
      data_quality: { metrics_available: ['ndvi', 'ndmi'] },
      current_metrics: {
        ndvi: { mean: 0.62, median: 0.63, min: 0.41, max: 0.79, coverage: 84 },
      },
      series: {
        ndvi: [{ observed_at: '2025-03-09', mean: 0.48 }],
      },
      histograms: {},
      prediction: null,
      scenarios: [],
      archives: [],
    }
    store.fieldTemporalAnalytics = {
      seasonal_series: {
        metrics: [
          {
            metric: 'ndvi',
            label: 'NDVI',
            points: [
              { observed_at: '2025-03-09', value: 0.482 },
            ],
          },
        ],
      },
      anomalies: [
        {
          metric: 'ndvi',
          kind: 'possible_drought_stress',
          severity: 'warning',
          observed_at: '2025-03-09',
        },
      ],
    }

    const wrapper = mount(FieldActionsPanel, {
      global: {
        plugins: [pinia],
        stubs: {
          FreshnessBadge: true,
          MiniHistogram: true,
          MiniSparkline: true,
        },
      },
    })

    expect(wrapper.text()).toContain('XY')
    expect(wrapper.text()).toContain('Линейка')
    expect(wrapper.text()).toContain('Аномалии')
    expect(wrapper.text()).toContain('Возможный стресс засухи')
    expect(wrapper.text()).toContain('Предупреждение')
  })

  it('prefers temporal weather series over a single dashboard snapshot in seasonal cards', () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useMapStore()
    store.selectedFieldIds = ['field-1']
    store.selectedField = {
      field_id: 'field-1',
      source: 'autodetect',
      quality_score: 0.88,
      quality_label: 'high',
    }
    store.activeFieldTab = 'metrics'
    store.metricsDisplayMode = 'cards'
    store.fieldDashboard = {
      field: store.selectedField,
      kpis: { prediction_ready: true, archive_count: 0, scenario_count: 0, observation_cells: 14 },
      data_quality: { metrics_available: ['ndvi', 'precipitation'] },
      current_metrics: {
        precipitation: { mean: 0, median: 0, min: 0, max: 0, coverage: 84 },
      },
      series: {
        precipitation: [{ observed_at: '2025-08-31', mean: 0 }],
      },
      prediction: null,
      scenarios: [],
      archives: [],
    }
    store.fieldTemporalAnalytics = {
      seasonal_series: {
        metrics: [
          {
            metric: 'precipitation',
            label: 'Осадки',
            points: [
              { observed_at: '2025-08-24', value: 3.5 },
              { observed_at: '2025-08-31', value: 0.0 },
            ],
          },
        ],
      },
    }

    const wrapper = mount(FieldActionsPanel, {
      global: {
        plugins: [pinia],
        stubs: {
          FreshnessBadge: true,
          MiniHistogram: true,
          MiniSparkline: {
            props: ['points'],
            template: '<div class="sparkline-points">{{ points.length }}</div>',
          },
        },
      },
    })

    expect(wrapper.text()).toContain('Осадки')
    expect(wrapper.findAll('.sparkline-points').some((node) => node.text() === '2')).toBe(true)
  })

  it('reloads temporal analytics instead of dashboard when the date range changes', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useMapStore()
    store.selectedFieldIds = ['field-1']
    store.selectedField = {
      field_id: 'field-1',
      source: 'autodetect',
      quality_score: 0.88,
      quality_label: 'high',
    }
    store.activeFieldTab = 'metrics'
    store.seriesDateFrom = '2025-03-01'
    store.seriesDateTo = '2025-08-31'
    store.fieldDashboard = {
      field: store.selectedField,
      kpis: { prediction_ready: true, archive_count: 0, scenario_count: 0, observation_cells: 14 },
      data_quality: { metrics_available: ['ndvi'] },
      current_metrics: {},
      prediction: null,
      scenarios: [],
      archives: [],
    }
    store.loadFieldDashboard = vi.fn()
    store.loadFieldTemporalAnalytics = vi.fn().mockResolvedValue(null)

    const wrapper = mount(FieldActionsPanel, {
      global: {
        plugins: [pinia],
        stubs: {
          FreshnessBadge: true,
          MiniHistogram: true,
          MiniSparkline: true,
        },
      },
    })

    const inputs = wrapper.findAll('input[type="date"]')
    await inputs[0].trigger('change')

    expect(store.loadFieldTemporalAnalytics).toHaveBeenCalledWith(
      undefined,
      expect.objectContaining({
        target: 'metrics',
        preferExisting: false,
        autoBackfill: true,
      }),
    )
    expect(store.loadFieldDashboard).not.toHaveBeenCalled()
  })

  it('renders a historical-backfill status instead of a fake one-point line', () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useMapStore()
    store.selectedFieldIds = ['field-1']
    store.selectedField = {
      field_id: 'field-1',
      source: 'autodetect',
      quality_score: 0.88,
      quality_label: 'high',
    }
    store.activeFieldTab = 'metrics'
    store.metricsDisplayMode = 'xy'
    store.metricsSelectedSeries = 'ndvi'
    store.seriesDateFrom = '2025-03-01'
    store.seriesDateTo = '2025-08-31'
    store.fieldDashboard = {
      field: store.selectedField,
      kpis: { prediction_ready: true, archive_count: 0, scenario_count: 0, observation_cells: 14 },
      data_quality: { metrics_available: ['ndvi'] },
      current_metrics: {},
      prediction: null,
      scenarios: [],
      archives: [],
    }
    store.fieldTemporalAnalytics = {
      seasonal_series: {
        metrics: [
          {
            metric: 'ndvi',
            label: 'NDVI',
            points: [],
          },
        ],
      },
      data_status: {
        code: 'backfill_required',
        message_code: 'temporal_backfill_required',
        requested_range: {
          date_from: '2025-03-01',
          date_to: '2025-08-31',
        },
      },
    }

    const wrapper = mount(FieldActionsPanel, {
      global: {
        plugins: [pinia],
        stubs: {
          FreshnessBadge: true,
          MiniHistogram: true,
          MiniSparkline: true,
        },
      },
    })

    expect(wrapper.text()).toContain('Исторический диапазон ещё не подготовлен')
    expect(wrapper.text()).toContain('01.03.2025 - 31.08.2025')
  })

  it('limits preview contours to overview and metrics tabs', () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useMapStore()
    store.selectedFieldIds = ['preview-1']
    store.selectedField = {
      field_id: 'preview-1',
      source: 'autodetect_preview',
      quality_score: 0.61,
      quality_label: 'medium',
      review_required: true,
      review_reason_code: 'preview_contour_requires_confirmation',
    }
    store.fieldDashboard = {
      field: store.selectedField,
      kpis: { prediction_ready: false, archive_count: 0, scenario_count: 0, observation_cells: 5 },
      data_quality: { metrics_available: ['ndvi'] },
      current_metrics: {},
      prediction: null,
      scenarios: [],
      archives: [],
    }
    store.activeFieldTab = 'forecast'

    const wrapper = mount(FieldActionsPanel, {
      global: {
        plugins: [pinia],
        stubs: {
          FreshnessBadge: true,
          MiniHistogram: true,
          MiniSparkline: true,
        },
      },
    })

    expect(wrapper.text()).toContain('Предварительный контур')
    expect(wrapper.text()).toContain('Preview-only')
    expect(wrapper.text()).not.toContain('Сценарии')
    expect(wrapper.text()).not.toContain('Архив')
    expect(store.activeFieldTab).toBe('overview')
  })
})
