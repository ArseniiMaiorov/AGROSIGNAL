<template>
  <section class="window-shell field-actions-panel">
    <div class="window-title">{{ t('field.title') }}</div>
    <div class="window-body panel-body">
      <div v-if="store.selectedFieldCount === 0" class="placeholder">
        {{ t('field.selectHint') }}
      </div>

      <template v-else>
        <div class="selection-strip">
          <div>
            <div class="selection-title">
              {{ isGroup ? t('field.groupSelection') : t('field.currentField') }}
            </div>
            <div class="selection-subtitle">
              <template v-if="isGroup">
                {{ t('field.selectedCount') }}: {{ store.selectedFieldCount }} · {{ totalAreaLabel }}
              </template>
              <template v-else>
                {{ t('field.source') }}: {{ sourceLabel }} · {{ qualityLabel }}
              </template>
            </div>
          </div>
          <div class="button-row compact-row">
            <button class="btn-secondary" @click="store.clearFieldSelection()">
              {{ t('field.clearSelection') }}
            </button>
          </div>
        </div>

        <div class="tab-strip">
          <button
            v-for="tab in visibleTabs"
            :key="tab.id"
            class="tab-btn"
            :class="{ active: store.activeFieldTab === tab.id }"
            @click="store.setActiveFieldTab(tab.id)"
          >
            {{ tab.label }}
          </button>
        </div>

        <div v-if="store.activeFieldTab === 'overview'" class="section-block">
          <div class="section-caption">{{ t('field.tools') }}</div>
          <div class="button-row">
            <button
              v-if="!store.mergeMode"
              class="btn-secondary"
              :disabled="store.selectedFieldCount < 2 && !store.selectedField?.field_id"
              @click="store.startMergeMode()"
            >
              {{ t('field.mergeMode') }}
            </button>
            <template v-else>
              <button
                class="btn-primary"
                :disabled="store.mergeSelectionIds.length < 2"
                @click="store.mergeSelectedFields()"
              >
                {{ t('field.mergeApply') }}
              </button>
              <button class="btn-secondary" @click="store.cancelMergeMode()">
                {{ t('field.mergeCancel') }}
              </button>
            </template>

            <button
              v-if="!isGroup && !store.splitMode"
              class="btn-secondary"
              @click="store.startSplitMode()"
            >
              {{ t('field.splitMode') }}
            </button>
            <button
              v-else-if="!isGroup && store.splitMode"
              class="btn-secondary"
              @click="store.cancelSplitMode()"
            >
              {{ t('field.splitCancel') }}
            </button>
            <button
              v-if="!isGroup"
              class="btn-secondary btn-danger"
              @click="store.deleteSelectedField()"
            >
              {{ t('field.deleteField') }}
            </button>
          </div>
          <div v-if="store.mergeMode" class="info-box">
            <div>{{ t('field.mergeHint') }}</div>
            <strong>{{ t('field.mergeSelected') }}: {{ store.mergeSelectionIds.length }}</strong>
          </div>
          <div v-if="store.splitMode" class="info-box">
            {{ t('field.splitHint') }}
          </div>
          <div v-if="isPreviewField" class="info-box">
            <div><strong>{{ t('field.previewOnlyTitle') }}</strong> · {{ t('field.previewOnly') }}</div>
            <div>{{ t('field.previewOnlyMessage') }}</div>
            <div>{{ t('field.previewOnlyActionHint') }}</div>
          </div>
        </div>

        <div v-if="isLoadingDashboard && !dashboard" class="placeholder">{{ t('field.loadingDashboard') }}</div>

        <template v-else-if="dashboard">
          <div v-if="store.activeFieldTab === 'overview'" class="section-stack">
            <div class="section-block">
              <div class="section-caption">{{ t('field.overview') }}</div>
              <div class="summary-grid">
                <div v-for="card in overviewCards" :key="card.label" class="summary-card">
                  <span>{{ card.label }}</span>
                  <strong>{{ card.value }}</strong>
                </div>
              </div>
            </div>

            <div v-if="!isGroup" class="section-block">
              <div class="section-caption">{{ t('field.quality') }}</div>
              <div class="info-box">
                <div>{{ qualityBandLabel }}</div>
                <div class="meta-row">
                  <span>{{ t('field.operationalTierLabel') }}: {{ fieldOperationalTierLabel }}</span>
                </div>
                <div class="meta-row">
                  <span>{{ t('field.reviewNeeded') }}: {{ fieldReviewRequiredLabel }}</span>
                </div>
                <div class="meta-row">
                  <span>{{ qualityReasonLabel }}</span>
                </div>
                <div v-if="fieldReviewReason" class="meta-row">
                  <span>{{ fieldReviewReason }}</span>
                </div>
              </div>
            </div>

            <div class="section-block">
              <div class="section-caption">{{ t('field.dataQuality') }}</div>
              <div class="info-box">
                <div>{{ dataQualityText }}</div>
                <div class="meta-row">
                  <span>{{ t('field.availableMetrics') }}: {{ availableMetricsLabel }}</span>
                </div>
              </div>
            </div>

            <div v-if="analyticsSummary.current_stage || analyticsAlertEntries.length || managementZoneSummary.zone_count" class="section-block">
              <div class="section-caption">{{ t('field.analyticsHighlights') }}</div>
              <div class="feature-grid compact-grid">
                <div v-if="analyticsSummary.current_stage" class="feature-item">
                  <span>{{ t('field.currentStage') }}</span>
                  <strong>{{ analyticsSummary.current_stage }}</strong>
                </div>
                <div v-if="analyticsSummary.lag_weeks_vs_norm !== null && analyticsSummary.lag_weeks_vs_norm !== undefined" class="feature-item">
                  <span>{{ t('field.stageLag') }}</span>
                  <strong>{{ Number(analyticsSummary.lag_weeks_vs_norm).toFixed(1) }} нед.</strong>
                </div>
                <div class="feature-item">
                  <span>{{ t('field.activeAlerts') }}</span>
                  <strong>{{ analyticsSummary.active_alert_count || 0 }}</strong>
                </div>
                <div v-if="managementZoneSummary.zone_count" class="feature-item">
                  <span>{{ t('field.zoneCount') }}</span>
                  <strong>{{ managementZoneSummary.zone_count }}</strong>
                </div>
              </div>
            </div>
          </div>

          <div v-else-if="store.activeFieldTab === 'metrics'" class="section-stack">
            <div class="section-block">
              <div class="section-caption">{{ t('field.currentMetrics') }}</div>
              <div v-if="metricCards.length" class="metrics-grid">
                <div v-for="metric in metricCards" :key="metric.id" class="metric-card">
                  <div class="metric-card-head">
                    <strong>{{ metric.label }}</strong>
                    <span>{{ metric.mean }}</span>
                  </div>
                  <div class="metric-card-meta">
                    <span>{{ t('field.metricMedian') }}: {{ metric.median }}</span>
                    <span>{{ t('field.metricRange') }}: {{ metric.range }}</span>
                    <span>{{ t('field.metricCoverage') }}: {{ metric.coverage }}</span>
                  </div>
                </div>
              </div>
              <div v-else class="placeholder">{{ t('field.noMetrics') }}</div>
            </div>

            <div class="section-block">
              <div class="section-caption">{{ t('field.seasonalAnalytics') }}</div>
              <div class="button-row compact-row">
                <button class="btn-secondary" :class="{ active: store.metricsDisplayMode === 'cards' }" @click="store.metricsDisplayMode = 'cards'">
                  {{ t('field.metricModeCards') }}
                </button>
                <button class="btn-secondary" :class="{ active: store.metricsDisplayMode === 'xy' }" @click="store.metricsDisplayMode = 'xy'">
                  {{ t('field.metricModeXY') }}
                </button>
                <button class="btn-secondary" :class="{ active: store.metricsDisplayMode === 'timeline' }" @click="store.metricsDisplayMode = 'timeline'">
                  {{ t('field.metricModeTimeline') }}
                </button>
                <button class="btn-secondary" :class="{ active: store.metricsDisplayMode === 'anomalies' }" @click="store.metricsDisplayMode = 'anomalies'">
                  {{ t('field.metricModeAnomalies') }}
                </button>
                <button class="btn-secondary" :class="{ active: store.metricsDisplayMode === 'zones' }" @click="store.metricsDisplayMode = 'zones'">
                  {{ t('field.metricModeZones') }}
                </button>
              </div>
              <div class="date-range-row">
                <label>
                  <span>{{ t('field.dateFrom') }}</span>
                  <input type="date" v-model="store.seriesDateFrom" @change="onDateRangeChange" @input="onDateInput" />
                </label>
                <label>
                  <span>{{ t('field.dateTo') }}</span>
                  <input type="date" v-model="store.seriesDateTo" @change="onDateRangeChange" @input="onDateInput" />
                </label>
                <button
                  v-if="store.seriesDateFrom || store.seriesDateTo"
                  class="btn-secondary btn-sm"
                  @click="clearDateRange"
                >
                  {{ t('field.clearDateRange') }}
                </button>
                <label v-if="seasonalMetricSelectorOptions.length">
                  <span>{{ t('field.metricSelector') }}</span>
                  <select v-model="store.metricsSelectedSeries">
                    <option v-for="item in seasonalMetricSelectorOptions" :key="item.id" :value="item.id">
                      {{ item.label }}
                    </option>
                  </select>
                </label>
              </div>
              <div v-if="temporalProgressActive && store.metricsDisplayMode !== 'zones'" class="task-progress-card">
                <div class="task-progress-head">
                  <strong>{{ t('field.taskInProgress') }}</strong>
                  <span>{{ temporalProgressLabel }}%</span>
                </div>
                <div class="task-progress-track">
                  <div class="task-progress-fill" :style="{ width: `${store.temporalAnalyticsTaskProgress}%` }"></div>
                </div>
                <div class="task-progress-meta">
                  <span>{{ temporalTaskStage }}</span>
                  <span v-if="temporalTaskDetail">{{ temporalTaskDetail }}</span>
                  <span>{{ temporalTaskTiming }}</span>
                </div>
              </div>
              <div v-else-if="metricsDataStatusMessage && store.metricsDisplayMode !== 'zones'" class="info-box">
                <div>{{ metricsDataStatusMessage }}</div>
              </div>
              <div v-if="store.metricsDisplayMode === 'cards' && seriesEntries.length" class="chart-grid">
                <div v-for="entry in seriesEntries" :key="entry.id" class="chart-card">
                  <div class="chart-head">
                    <strong>{{ entry.label }}</strong>
                    <span>{{ entry.latest }}</span>
                  </div>
                  <MiniSparkline :points="entry.items" :color="entry.color" />
                </div>
              </div>
              <div v-else-if="store.metricsDisplayMode === 'xy'" class="chart-card chart-card-wide">
                <div class="chart-head">
                  <strong>{{ selectedSeasonalMetricLabel }}</strong>
                  <span>{{ seasonalMetricPointCountLabel }}</span>
                </div>
                <XYSeriesChart
                  :series="seasonalMetricSeries"
                  :anomalies="selectedMetricAnomalies"
                  :empty-text="seasonalMetricChartEmptyText"
                />
              </div>
              <div v-else-if="store.metricsDisplayMode === 'timeline'" class="chart-card chart-card-wide">
                <div class="chart-head">
                  <strong>{{ selectedSeasonalMetricLabel }}</strong>
                  <span>{{ seasonalMetricPointCountLabel }}</span>
                </div>
                <NumericTimeline
                  :points="seasonalMetricTimeline"
                  :empty-text="seasonalMetricChartEmptyText"
                />
              </div>
              <div v-else-if="store.metricsDisplayMode === 'anomalies'" class="section-stack">
                <div v-if="seasonalAnomalyRows.length" class="list-stack">
                  <div v-for="entry in seasonalAnomalyRows" :key="entry.key" class="list-item">
                    <div class="list-item-head">
                      <strong>{{ entry.label }}</strong>
                      <span>{{ entry.severity }}</span>
                    </div>
                    <div class="meta-row">
                      <span>{{ entry.date }}</span>
                      <span>{{ entry.metric }}</span>
                    </div>
                    <div class="meta-row">
                      <span>{{ entry.reason }}</span>
                    </div>
                  </div>
                </div>
                <div v-else class="placeholder">{{ t('field.noAnomalies') }}</div>
              </div>
              <div v-else-if="store.metricsDisplayMode === 'zones'" class="section-stack">
                <div class="info-box">
                  <div class="meta-row">
                    <span>{{ t('field.zoneMode') }}: {{ managementZoneModeLabel }}</span>
                    <span>{{ t('field.zoneCount') }}: {{ managementZoneSummary.zone_count || 0 }}</span>
                  </div>
                  <div class="button-row compact-row">
                    <button class="btn-secondary" :disabled="!managementZonesSupported" @click="store.showManagementZonesOverlay = !store.showManagementZonesOverlay">
                      {{ store.showManagementZonesOverlay ? t('field.hideZonesOnMap') : t('field.showZonesOnMap') }}
                    </button>
                  </div>
                </div>
                <div v-if="managementZoneRows.length" class="list-stack">
                  <div v-for="zone in managementZoneRows" :key="zone.zone_code" class="list-item">
                    <div class="list-item-head">
                      <strong>{{ zone.label }}</strong>
                      <span>{{ zone.area_share_pct }}%</span>
                    </div>
                    <div class="meta-row">
                      <span>{{ t('field.zoneArea') }}: {{ (Number(zone.area_m2 || 0) / 10000).toFixed(2) }} га</span>
                      <span>{{ t('field.zoneScore') }}: {{ Number(zone.mean_score || 0).toFixed(3) }}</span>
                    </div>
                    <div class="meta-row">
                      <span>{{ t('field.zoneYield') }}: {{ zone.predicted_yield_kg_ha ? formatYield(zone.predicted_yield_kg_ha) : t('field.yieldPotential') }}</span>
                      <span>{{ t('field.confidence') }}: {{ formatPercent(zone.confidence) }}</span>
                    </div>
                  </div>
                </div>
                <div v-else class="placeholder">{{ t('field.noZones') }}</div>
              </div>
              <div v-else class="placeholder">{{ t('field.noSeries') }}</div>
            </div>

            <div v-if="gddCumulativeSeries.length" class="section-block">
              <div class="section-caption">{{ locale === 'ru' ? 'Накопленные температуры (ГСТ)' : 'Accumulated Heat Units (GDD)' }}</div>
              <div class="info-box">
                <div class="meta-row">
                  <span>{{ locale === 'ru' ? 'Сумма ГСТ' : 'GDD Sum' }}: {{ gddCumulativeTotal }}</span>
                  <span>{{ locale === 'ru' ? 'Наблюдений' : 'Weeks' }}: {{ gddCumulativeSeries[0]?.points?.length || 0 }}</span>
                </div>
              </div>
              <div class="chart-card chart-card-wide">
                <XYSeriesChart
                  :series="gddCumulativeSeries"
                  :empty-text="locale === 'ru' ? 'Нет данных о температуре' : 'No temperature data'"
                />
              </div>
            </div>

            <div v-if="store.metricsDisplayMode === 'cards'" class="section-block">
              <div class="section-caption">{{ t('field.distribution') }}</div>
              <div v-if="histogramEntries.length" class="chart-grid">
                <div v-for="entry in histogramEntries" :key="entry.id" class="chart-card">
                  <div class="chart-head">
                    <strong>{{ entry.label }}</strong>
                  </div>
                  <MiniHistogram :histogram="entry.histogram" :color="entry.color" />
                </div>
              </div>
              <div v-else class="placeholder">{{ t('field.noDistribution') }}</div>
            </div>
          </div>

          <div v-else-if="store.activeFieldTab === 'forecast'" class="section-stack">
            <template v-if="!isGroup">
              <div class="section-block">
                <div class="section-caption">{{ t('field.cropAndForecast') }}</div>
                <div class="input-grid two-col">
                  <label>
                    <span>{{ t('field.crop') }}</span>
                    <select v-model="store.selectedCropCode">
                      <option v-for="crop in store.crops" :key="crop.code" :value="crop.code">
                        {{ crop.name }}
                      </option>
                    </select>
                  </label>
                  <label>
                    <span>{{ t('field.lastCalculation') }}</span>
                    <input :value="predictionDateLabel" type="text" readonly />
                  </label>
                </div>
                <div class="button-row">
                  <button class="btn-secondary" :disabled="store.isRefreshingPrediction" @click="store.refreshPrediction(true)">
                    {{ store.isRefreshingPrediction ? t('field.refreshingPrediction') : t('field.refreshPrediction') }}
                  </button>
                  <button class="btn-secondary" :class="{ active: store.showForecastGraphs }" @click="store.showForecastGraphs = !store.showForecastGraphs">
                    {{ store.showForecastGraphs ? t('field.hideGraphs') : t('field.showGraphs') }}
                  </button>
                </div>
                <div v-if="predictionProgressActive" class="task-progress-card">
                  <div class="task-progress-head">
                    <strong>{{ t('field.taskInProgress') }}</strong>
                    <span>{{ predictionProgressLabel }}%</span>
                  </div>
                  <div class="task-progress-track">
                    <div class="task-progress-fill" :style="{ width: `${store.predictionTaskProgress}%` }"></div>
                  </div>
                  <div class="task-progress-meta">
                    <span>{{ predictionTaskStage }}</span>
                    <span v-if="predictionTaskDetail">{{ predictionTaskDetail }}</span>
                    <span>{{ predictionTaskTiming }}</span>
                  </div>
                  <div v-if="predictionTaskLogs.length" class="task-log-list">
                    <div v-for="(line, index) in predictionTaskLogs" :key="`prediction-log-${index}`" class="task-log-line">{{ line }}</div>
                  </div>
                </div>
              </div>

              <div v-if="prediction" class="section-block">
                <div class="forecast-hero">
                  <div class="forecast-main">{{ yieldLabel }}</div>
                  <div class="forecast-meta">
                    <span>{{ t('field.confidence') }}: {{ confidenceLabel }}</span>
                    <span>{{ t('field.model') }}: {{ prediction.model_version }}</span>
                    <span v-if="prediction.confidence_tier">{{ predictionConfidenceTierLabel }}</span>
                  </div>
                </div>
                <FreshnessBadge :meta="prediction.freshness" />
                <div class="info-box">
                  <div class="meta-row">
                    <span>{{ t('field.operationalTierLabel') }}: {{ predictionOperationalTierLabel }}</span>
                  </div>
                  <div class="meta-row">
                    <span>{{ t('field.reviewNeeded') }}: {{ predictionReviewRequiredLabel }}</span>
                  </div>
                  <div v-if="predictionReviewReason" class="meta-row">
                    <span>{{ t('field.reviewReason') }}: {{ predictionReviewReason }}</span>
                  </div>
                </div>
                <div v-if="predictionSupportReason || suitabilityEntries.length || prediction?.crop_suitability?.recommendation" class="info-box">
                  <div v-if="prediction?.crop_suitability?.recommendation" class="explain-summary" style="margin-bottom:6px">
                    {{ prediction.crop_suitability.recommendation }}
                  </div>
                  <div v-if="predictionSupportReason" class="explain-summary">{{ predictionSupportReason }}</div>
                  <div v-if="suitabilityEntries.length" class="feature-grid compact-grid">
                    <div v-for="entry in suitabilityEntries" :key="entry.label" class="feature-item">
                      <span>{{ entry.label }}</span>
                      <strong>{{ entry.value }}</strong>
                    </div>
                  </div>
                </div>
                <div class="info-box">
                  <div class="explain-summary">{{ prediction.explanation?.summary || t('field.noExplanation') }}</div>
                  <div class="drivers-list">
                    <div v-for="driver in predictionDrivers" :key="driver.label" class="driver-row">
                      <span>{{ driver.label }}</span>
                      <strong>{{ driver.effect }}</strong>
                    </div>
                  </div>
                </div>
              </div>

              <div v-if="prediction" class="section-block">
                <div class="section-caption">{{ t('field.modelTruth') }}</div>
                <div class="info-box">
                  <div class="feature-grid compact-grid">
                    <div v-for="entry in modelFoundationEntries" :key="entry.label" class="feature-item">
                      <span>{{ entry.label }}</span>
                      <strong>{{ entry.value }}</strong>
                    </div>
                  </div>
                  <div class="meta-row">
                    <span>{{ retrainDescription }}</span>
                  </div>
                </div>
              </div>

              <div v-if="prediction && store.expertMode" class="section-block">
                <div class="section-caption">{{ t('field.geometrySegmentation') }}</div>
                <div class="info-box">
                  <div class="feature-grid compact-grid">
                    <div v-for="entry in geometryDiagnosticsEntries" :key="entry.label" class="feature-item">
                      <span>{{ entry.label }}</span>
                      <strong>{{ entry.value }}</strong>
                    </div>
                  </div>
                  <div v-if="geometryDebugStatus" class="meta-row">
                    <span>{{ geometryDebugStatus }}</span>
                  </div>
                  <div v-if="!debugRuntime" class="meta-row">
                    <span>{{ t('field.noGeometryDebug') }}</span>
                  </div>
                </div>
              </div>

              <div v-if="prediction && phenologySummaryEntries.length" class="section-block">
                <div class="section-caption">{{ t('field.phenology') }}</div>
                <div class="feature-grid compact-grid">
                  <div v-for="entry in phenologySummaryEntries" :key="entry.label" class="feature-item">
                    <span>{{ entry.label }}</span>
                    <strong>{{ entry.value }}</strong>
                  </div>
                </div>
              </div>

              <div v-if="prediction && store.showForecastGraphs" class="section-block">
                <div class="section-caption">{{ t('field.historyAndForecast') }}</div>
                <div class="chart-card chart-card-wide">
                  <XYSeriesChart
                    :series="historyTrendChartSeries"
                    :markers="forecastTrendMarkers"
                    :ranges="forecastTrendRanges"
                    :empty-text="t('field.noTrendHistory')"
                  />
                </div>
              </div>

              <div v-if="prediction && store.showForecastGraphs" class="section-block">
                <div class="section-caption">{{ t('field.seasonCurves') }}</div>
                <div v-if="temporalProgressActive" class="task-progress-card">
                  <div class="task-progress-head">
                    <strong>{{ t('field.taskInProgress') }}</strong>
                    <span>{{ temporalProgressLabel }}%</span>
                  </div>
                  <div class="task-progress-track">
                    <div class="task-progress-fill" :style="{ width: `${store.temporalAnalyticsTaskProgress}%` }"></div>
                  </div>
                  <div class="task-progress-meta">
                    <span>{{ temporalTaskStage }}</span>
                    <span v-if="temporalTaskDetail">{{ temporalTaskDetail }}</span>
                    <span>{{ temporalTaskTiming }}</span>
                  </div>
                </div>
                <div v-else-if="forecastDataStatusMessage" class="info-box">
                  <div>{{ forecastDataStatusMessage }}</div>
                </div>
                <div class="chart-card chart-card-wide">
                  <XYSeriesChart
                    :series="forecastSeasonalOverlaySeries"
                    :anomalies="forecastAnomalyItems"
                    :empty-text="forecastSeasonalChartEmptyText"
                  />
                </div>
              </div>

              <div v-if="prediction && store.showForecastGraphs" class="section-block">
                <div class="section-caption">{{ t('field.futureWeatherAndGdd') }}</div>
                <div class="chart-card chart-card-wide">
                  <div class="chart-head">
                    <strong>{{ t('field.futureWeatherAndGdd') }}</strong>
                    <select v-model="forecastCurveMetric" class="sensitivity-select" style="font-size:11px;padding:1px 4px">
                      <option v-for="option in forecastCurveMetricOptions" :key="option.value" :value="option.value">
                        {{ option.label }}
                      </option>
                    </select>
                  </div>
                  <XYSeriesChart
                    :series="predictionForecastCurveSeries"
                    :empty-text="predictionForecastCurveEmptyText"
                  />
                </div>
              </div>

              <div v-if="prediction && (waterBalanceSeries.length || forecastDataStatusMessage || temporalProgressActive)" class="section-block">
                <div class="section-caption">{{ t('field.waterBalance') }}</div>
                <div class="info-box">
                  <div class="meta-row">
                    <span>{{ t('field.waterBalanceModel') }}: {{ waterBalance.model || 'FAO-lite' }}</span>
                    <span>{{ t('field.stressClass') }}: {{ waterBalance.summary?.stress_class || '—' }}</span>
                  </div>
                  <div class="meta-row">
                    <span>{{ t('field.irrigationNeed') }}: {{ waterBalance.summary?.irrigation_need_class || '—' }}</span>
                    <span>{{ t('field.rootZoneStorage') }}: {{ waterBalance.summary?.root_zone_storage_mm ?? '—' }} мм</span>
                  </div>
                </div>
                <div v-if="forecastDataStatusMessage && !waterBalanceSeries.length" class="info-box">
                  <div>{{ forecastDataStatusMessage }}</div>
                </div>
                <div class="chart-card chart-card-wide">
                  <XYSeriesChart :series="waterBalanceSeries" :empty-text="forecastDataStatusMessage || t('field.noWaterBalance')" />
                </div>
              </div>

              <div v-if="prediction && riskSummary?.items?.length" class="section-block">
                <div class="section-caption">{{ t('field.riskIndicators') }}</div>
                <div class="list-stack">
                  <div v-for="item in riskSummary.items" :key="item.id" class="list-item">
                    <div class="list-item-head">
                      <strong>{{ scenarioRiskItemLabel(item) }}</strong>
                      <span>{{ riskLevelLabel(item.level_code || item.level) }}</span>
                    </div>
                    <div class="meta-row">
                      <span>{{ t('field.score') }}: {{ Number(item.score || 0).toFixed(2) }}</span>
                      <span>{{ scenarioRiskItemReason(item) }}</span>
                    </div>
                  </div>
                </div>
              </div>

              <div v-if="prediction && (prediction.driver_breakdown?.length || predictionDrivers.length)" class="section-block">
                <div class="section-caption">{{ t('field.factorBreakdown') }}</div>
                <div class="chart-card chart-card-wide">
                  <FactorWaterfallChart :factors="prediction.driver_breakdown || []" :empty-text="t('field.noExplanation')" />
                </div>
              </div>

              <div v-if="featureEntries.length" class="section-block">
                <div class="section-caption">{{ t('field.inputFeatures') }}</div>
                <div class="feature-grid">
                  <div v-for="entry in featureEntries" :key="entry.label" class="feature-item">
                    <span>{{ entry.label }}</span>
                    <strong>{{ entry.value }}</strong>
                  </div>
                </div>
              </div>

              <div v-if="qualityEntries.length" class="section-block">
                <div class="section-caption">{{ t('field.dataQuality') }}</div>
                <div class="feature-grid">
                  <div v-for="entry in qualityEntries" :key="entry.label" class="feature-item">
                    <span>{{ entry.label }}</span>
                    <strong>{{ entry.value }}</strong>
                  </div>
                </div>
              </div>

              <div v-if="!prediction" class="placeholder">{{ t('field.noPrediction') }}</div>
            </template>
            <div v-else class="placeholder">{{ t('field.singleOnlyForecast') }}</div>
          </div>

          <div v-else-if="store.activeFieldTab === 'scenarios'" class="section-stack">
            <template v-if="!isGroup">
              <div class="section-block">
                <div class="section-caption">{{ t('field.scenario') }}</div>
                <div class="button-row compact-row" style="margin-bottom:6px">
                  <button
                    class="btn-secondary"
                    :class="{ active: !store.useManualModeling }"
                    :title="t('field.autoFillHint') || 'Заполнить факторы из спутниковых данных поля'"
                    @click="store.enableAutoModeling()"
                  >
                    {{ t('field.autoFromSatellite') || '🛰 Авто со спутника' }}
                  </button>
                  <button
                    class="btn-secondary"
                    :class="{ active: store.useManualModeling }"
                    :title="t('field.manualInputHint') || 'Вводить факторы вручную'"
                    @click="store.enableManualModeling()"
                  >
                    {{ t('field.manualInput') || '✏ Ручной ввод' }}
                  </button>
                </div>
                <div class="input-grid two-col">
                  <label>
                    <span>{{ t('field.scenarioName') }}</span>
                    <input v-model="store.scenarioName" type="text" :placeholder="t('field.scenarioNamePlaceholder')" />
                  </label>
                  <label>
                    <span>{{ t('field.crop') }}</span>
                    <select v-model="store.selectedCropCode">
                      <option v-for="crop in store.crops" :key="crop.code" :value="crop.code">
                        {{ crop.name }}
                      </option>
                    </select>
                  </label>
                </div>
                <div class="input-grid three-col">
                  <label>
                    <span>{{ t('field.irrigation') }}</span>
                    <input v-model.number="store.modelingForm.irrigation_pct" type="number" step="1" min="-100" max="100" :disabled="!store.useManualModeling" />
                  </label>
                  <label>
                    <span>{{ t('field.fertilizer') }}</span>
                    <input v-model.number="store.modelingForm.fertilizer_pct" type="number" step="1" min="-100" max="100" :disabled="!store.useManualModeling" />
                  </label>
                  <label :class="{ 'satellite-derived': !store.useManualModeling }">
                    <span>
                      {{ t('field.expectedRain') }}
                      <span
                        v-if="!store.useManualModeling"
                        class="sat-badge weather-badge"
                        :title="expectedRainAutoSourceTitle"
                      >{{ expectedRainAutoBadge }}</span>
                    </span>
                    <input v-model.number="store.modelingForm.expected_rain_mm" type="number" step="1" min="0" max="500" :disabled="!store.useManualModeling" />
                    <span v-if="!store.useManualModeling && store.modelingAutoSources?.expected_rain_mm" class="auto-source-hint">{{ expectedRainAutoSourceTitle }}</span>
                  </label>
                </div>
                <div class="input-grid three-col">
                  <label>
                    <span>{{ t('field.temperatureDelta') }}</span>
                    <input v-model.number="store.modelingForm.temperature_delta_c" type="number" step="0.5" min="-10" max="10" />
                  </label>
                  <label>
                    <span>{{ t('field.plantingDensity') }}</span>
                    <input v-model.number="store.modelingForm.planting_density_pct" type="number" step="1" min="-80" max="100" :disabled="!store.useManualModeling" />
                  </label>
                  <label :class="{ 'satellite-derived': !store.useManualModeling }">
                    <span>
                      {{ t('field.soilCompaction') }}
                      <span
                        v-if="!store.useManualModeling"
                        class="sat-badge"
                        :title="soilCompactionAutoSourceTitle"
                      >🛰</span>
                    </span>
                    <input v-model.number="store.modelingForm.soil_compaction" type="number" step="0.05" min="0" max="1" :disabled="!store.useManualModeling" />
                    <span v-if="!store.useManualModeling && store.modelingAutoSources?.soil_compaction" class="auto-source-hint">{{ soilCompactionAutoSourceTitle }}</span>
                  </label>
                  <label :title="t('field.cloudCoverFactorHint')">
                    <span>{{ t('field.cloudCoverFactor') }} <span class="sat-badge" title="Использует ERA5 солнечную радиацию">ERA5</span></span>
                    <input v-model.number="store.modelingForm.cloud_cover_factor" type="number" step="0.05" min="0.1" max="3.0" />
                  </label>
                </div>
                <div class="input-grid two-col">
                  <label>
                    <span>{{ t('field.tillageType') }}</span>
                    <select v-model="store.modelingForm.tillage_type" :disabled="!store.useManualModeling">
                      <option :value="null">{{ t('field.scenarioAuto') }}</option>
                      <option :value="0">{{ t('field.tillage.none') }}</option>
                      <option :value="1">{{ t('field.tillage.minimum') }}</option>
                      <option :value="2">{{ t('field.tillage.conventional') }}</option>
                      <option :value="3">{{ t('field.tillage.deep') }}</option>
                    </select>
                  </label>
                  <label>
                    <span>{{ t('field.pestPressure') }}</span>
                    <select v-model="store.modelingForm.pest_pressure" :disabled="!store.useManualModeling">
                      <option :value="null">{{ t('field.scenarioAuto') }}</option>
                      <option :value="0">{{ t('field.pest.none') }}</option>
                      <option :value="1">{{ t('field.pest.low') }}</option>
                      <option :value="2">{{ t('field.pest.medium') }}</option>
                      <option :value="3">{{ t('field.pest.high') }}</option>
                    </select>
                  </label>
                </div>
                <button class="btn-primary" :disabled="store.isSimulatingScenario" @click="store.simulateScenario()">
                  {{ store.isSimulatingScenario ? t('field.simulating') : t('field.simulate') }}
                </button>
                <div class="button-row compact-row">
                  <button class="btn-secondary" :class="{ active: store.showScenarioGraphs }" @click="store.showScenarioGraphs = !store.showScenarioGraphs">
                    {{ store.showScenarioGraphs ? t('field.hideGraphs') : t('field.showGraphs') }}
                  </button>
                  <button class="btn-secondary" :class="{ active: store.showScenarioFactors }" @click="store.showScenarioFactors = !store.showScenarioFactors">
                    {{ store.showScenarioFactors ? t('field.hideFactors') : t('field.showFactors') }}
                  </button>
                  <button class="btn-secondary" :class="{ active: store.showScenarioRisks }" @click="store.showScenarioRisks = !store.showScenarioRisks">
                    {{ store.showScenarioRisks ? t('field.hideRisks') : t('field.showRisks') }}
                  </button>
                </div>
                <div v-if="scenarioProgressActive" class="task-progress-card">
                  <div class="task-progress-head">
                    <strong>{{ t('field.taskInProgress') }}</strong>
                    <span>{{ scenarioProgressLabel }}%</span>
                  </div>
                  <div class="task-progress-track">
                    <div class="task-progress-fill" :style="{ width: `${store.scenarioTaskProgress}%` }"></div>
                  </div>
                  <div class="task-progress-meta">
                    <span>{{ scenarioTaskStage }}</span>
                    <span v-if="scenarioTaskDetail">{{ scenarioTaskDetail }}</span>
                    <span>{{ scenarioTaskTiming }}</span>
                  </div>
                  <div v-if="scenarioTaskLogs.length" class="task-log-list">
                    <div v-for="(line, index) in scenarioTaskLogs" :key="`scenario-log-${index}`" class="task-log-line">{{ line }}</div>
                  </div>
                </div>
              </div>

              <div v-if="store.modelingResult" class="section-block">
                <div class="section-caption">{{ t('field.latestScenario') }}</div>
                <FreshnessBadge :meta="store.modelingResult.freshness" />
                <div class="scenario-box">
                  <div class="summary-grid">
                    <div class="summary-card">
                      <span>{{ t('field.baseline') }}</span>
                      <strong>{{ modelingBaselineLabel }}</strong>
                    </div>
                    <div class="summary-card">
                      <span>{{ t('field.scenarioResult') }}</span>
                      <strong>{{ modelingScenarioLabel }}</strong>
                    </div>
                    <div class="summary-card">
                      <span>{{ t('field.change') }}</span>
                      <strong>{{ modelingDeltaLabel }}</strong>
                    </div>
                    <div class="summary-card">
                      <span>{{ t('field.riskLevel') }}</span>
                      <strong :class="riskLevelClass">{{ modelingRiskLevel }}</strong>
                    </div>
                  </div>

                  <div class="info-box">
                    <div class="meta-row">
                      <span>{{ t('field.operationalTierLabel') }}: {{ scenarioOperationalTierLabel }}</span>
                    </div>
                    <div class="meta-row">
                      <span>{{ t('field.reviewNeeded') }}: {{ scenarioReviewRequiredLabel }}</span>
                    </div>
                    <div v-if="scenarioReviewReason" class="meta-row">
                      <span>{{ t('field.reviewReason') }}: {{ scenarioReviewReason }}</span>
                    </div>
                  </div>

                  <div v-if="comparisonChart" class="info-box comparison-box">
                    <div class="section-caption comparison-title">{{ t('field.baselineVsScenario') }}</div>
                    <div class="comparison-track">
                      <div
                        v-if="comparisonChart.intervalWidth > 0"
                        class="comparison-interval"
                        :style="{
                          left: `${comparisonChart.intervalStartPct}%`,
                          width: `${comparisonChart.intervalWidth}%`,
                        }"
                      ></div>
                      <div class="comparison-marker baseline-marker" :style="{ left: `${comparisonChart.baselinePct}%` }"></div>
                      <div class="comparison-marker scenario-marker" :style="{ left: `${comparisonChart.scenarioPct}%` }"></div>
                    </div>
                    <div class="comparison-legend">
                      <span><i class="legend-dot baseline-dot"></i>{{ t('field.baselineLine') }}: {{ modelingBaselineLabel }}</span>
                      <span><i class="legend-dot scenario-dot"></i>{{ t('field.scenarioLine') }}: {{ modelingScenarioLabel }}</span>
                      <span v-if="comparisonChart.intervalLabel"><i class="legend-band"></i>{{ t('field.intervalBand') }}: {{ comparisonChart.intervalLabel }}</span>
                    </div>
                  </div>

                  <div v-if="store.showScenarioGraphs" class="chart-card chart-card-wide">
                    <div class="chart-head">
                      <strong>{{ t('field.historyAndScenario') }}</strong>
                      <span>{{ scenarioOperationalTierLabel }}</span>
                    </div>
                    <XYSeriesChart
                      v-if="store.scenarioGraphMode === 'xy'"
                      :series="historyTrendChartSeries"
                      :markers="forecastTrendMarkers"
                      :ranges="forecastTrendRanges"
                      :empty-text="t('field.noTrendHistory')"
                    />
                    <NumericTimeline
                      v-else
                      :points="historyTrend?.points || []"
                      :formatter="(_, point) => point?.observed_yield_kg_ha ? formatYield(point.observed_yield_kg_ha) : '—'"
                      :empty-text="t('field.noTrendHistory')"
                    />
                  </div>

                  <div v-if="store.modelingResult?.scenario_time_series" class="chart-card chart-card-wide">
                    <div class="chart-head">
                      <strong>{{ t('field.baselineVsScenario') || 'Базовый vs Сценарий' }}</strong>
                      <select v-model="scenarioComparisonMetric" class="sensitivity-select" style="font-size:11px;padding:1px 4px">
                        <option value="ndvi">NDVI</option>
                        <option value="ndmi">NDMI</option>
                        <option value="ndwi">NDWI</option>
                        <option value="soil_moisture">{{ t('field.soilMoisture') || 'Влажность' }}</option>
                      </select>
                    </div>
                    <XYSeriesChart
                      :series="scenarioTimeSeriesChart"
                      :empty-text="t('field.noScenarioSeries') || 'Нет данных временного ряда для этого сценария'"
                    />
                    <div v-if="scenarioSeriesIdentical" class="meta-row" style="justify-content:center;color:var(--text-muted);font-size:11px;margin-top:4px">
                      Базовый и сценарный прогноз полностью совпадают — влияние выбранных параметров на динамику незначительно
                    </div>
                  </div>

                  <div class="chart-card chart-card-wide">
                    <div class="chart-head">
                      <strong>{{ t('field.futureScenarioWeatherAndGdd') }}</strong>
                      <select v-model="scenarioForecastCurveMetric" class="sensitivity-select" style="font-size:11px;padding:1px 4px">
                        <option v-for="option in forecastCurveMetricOptions" :key="`scenario-${option.value}`" :value="option.value">
                          {{ option.label }}
                        </option>
                      </select>
                    </div>
                    <XYSeriesChart
                      :series="scenarioForecastCurveSeries"
                      :empty-text="scenarioForecastCurveEmptyText"
                    />
                    <div v-if="scenarioForecastCurveIdentical" class="meta-row" style="justify-content:center;color:var(--text-muted);font-size:11px;margin-top:4px">
                      Базовый и сценарный прогноз погоды полностью совпадают
                    </div>
                  </div>

                  <div class="info-box sensitivity-box">
                    <div class="section-caption" style="font-size:12px">{{ t('field.sensitivityAnalysis') || 'Анализ чувствительности' }}</div>
                    <div class="meta-row explain-summary">{{ t('field.sensitivityExplanation') }}</div>
                    <div class="sensitivity-controls">
                      <select v-model="selectedSweepParam" class="sensitivity-select">
                        <option value="irrigation_pct">{{ t('field.irrigation') }}</option>
                        <option value="fertilizer_pct">{{ t('field.fertilizer') }}</option>
                        <option value="expected_rain_mm">{{ t('field.expectedRain') }}</option>
                        <option value="temperature_delta_c">{{ t('field.temperatureDelta') || 'Температура' }}</option>
                      </select>
                      <button
                        class="btn-secondary"
                        :disabled="store.isLoadingSensitivity"
                        @click="store.fetchSensitivitySweep(selectedSweepParam)"
                      >
                        {{ store.isLoadingSensitivity ? '...' : t('field.showCurve') || 'Кривая отклика' }}
                      </button>
                    </div>
                    <ScenarioResponseChart
                      v-if="store.sensitivityData?.points?.length"
                      :points="store.sensitivityData.points"
                      :baseline-yield="store.sensitivityData.baseline_yield_kg_ha"
                      :current-value="currentSweepParamValue"
                      :param-label="sensitivityParamLabel"
                    />
                  </div>

                  <div v-if="store.showScenarioFactors && factorBreakdown.length" class="chart-card chart-card-wide">
                    <div class="section-caption" style="font-size:12px">{{ t('field.factorBreakdown') }}</div>
                    <div class="meta-row explain-summary">{{ t('field.scenarioFactorBreakdownExplanation') }}</div>
                    <FactorWaterfallChart :factors="factorBreakdown" :empty-text="t('field.noExplanation')" />
                  </div>

                  <div v-if="store.showScenarioRisks && (store.modelingResult?.scenario_risk_projection?.items?.length || store.modelingResult?.risk_summary)" class="section-block">
                    <div class="section-caption">{{ t('field.riskIndicators') }}</div>
                    <div class="list-stack">
                      <div
                        v-for="item in (store.modelingResult?.scenario_risk_projection?.items || [])"
                        :key="item.id"
                        class="list-item"
                      >
                        <div class="list-item-head">
                          <strong>{{ scenarioRiskItemLabel(item) }}</strong>
                          <span>{{ riskLevelLabel(item.level_code || item.level) }}</span>
                        </div>
                        <div class="meta-row">
                          <span>{{ t('field.score') }}: {{ Number(item.score || 0).toFixed(2) }}</span>
                          <span>{{ scenarioRiskItemReason(item) }}</span>
                        </div>
                      </div>
                      <div v-if="!(store.modelingResult?.scenario_risk_projection?.items || []).length" class="list-item">
                        <div class="list-item-head">
                          <strong>{{ t('field.riskLevel') }}</strong>
                          <span>{{ modelingRiskLevel }}</span>
                        </div>
                        <div class="meta-row">
                          <span>{{ modelingRiskComment }}</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div class="info-box">
                    <div>{{ modelingRiskComment }}</div>
                    <div v-if="store.modelingResult?.risk_summary?.confidence_note" class="meta-row">
                      <span style="font-style:italic">{{ store.modelingResult.risk_summary.confidence_note }}</span>
                    </div>
                    <div v-if="assumptionEntries.length" class="feature-grid compact-grid">
                      <div v-for="entry in assumptionEntries" :key="entry.label" class="feature-item">
                        <span>{{ entry.label }}</span>
                        <strong>{{ entry.value }}</strong>
                      </div>
                    </div>
                  </div>
                  <div v-if="scenarioSeasonalConstraints.length" class="info-box">
                    <div class="section-caption" style="font-size:12px">{{ t('field.seasonalConstraints') || 'Ограничения сезона' }}</div>
                    <div class="feature-grid compact-grid">
                      <div v-for="entry in scenarioSeasonalConstraints" :key="entry.label" class="feature-item">
                        <span>{{ entry.label }}</span>
                        <strong>{{ entry.value }}</strong>
                      </div>
                    </div>
                  </div>
                  <div v-if="scenarioWarnings.length" class="info-box">
                    <div class="section-caption" style="font-size:12px">{{ t('field.constraintWarnings') }}</div>
                    <div v-for="warning in scenarioWarnings" :key="warning" class="driver-row">
                      <span>{{ warning }}</span>
                    </div>
                  </div>
                </div>
              </div>

              <div class="section-block">
                <div class="section-caption">{{ t('field.savedScenarios') }}</div>
                <div v-if="store.fieldScenarios.length" class="list-stack">
                  <div v-for="scenario in store.fieldScenarios" :key="scenario.id" class="list-item">
                    <div class="list-item-head">
                      <strong>{{ scenario.scenario_name }}</strong>
                      <span>{{ formatDateTime(scenario.created_at) }}</span>
                    </div>
                    <div class="meta-row">
                      <span>{{ t('field.model') }}: {{ scenario.model_version || '—' }}</span>
                      <span>{{ t('field.change') }}: {{ formatDelta(scenario.delta_pct) }}</span>
                    </div>
                    <FreshnessBadge :meta="scenario.freshness" compact />
                  </div>
                </div>
                <div v-else class="placeholder">{{ t('field.noScenarios') }}</div>
              </div>
            </template>
            <div v-else class="placeholder">{{ t('field.singleOnlyScenarios') }}</div>
          </div>

          <div v-else-if="store.activeFieldTab === 'archive'" class="section-stack">
            <template v-if="!isGroup">
              <div class="section-block">
                <div class="section-caption">{{ t('field.archives') }}</div>
                <div class="button-row">
                  <button class="btn-secondary" :disabled="store.isCreatingArchive" @click="store.createArchiveForSelectedField()">
                    {{ store.isCreatingArchive ? t('field.creatingArchive') : t('field.createArchive') }}
                  </button>
                </div>
                <div v-if="store.fieldArchives.length" class="list-stack">
                  <div v-for="archive in store.fieldArchives" :key="archive.id" class="list-item">
                    <div class="list-item-head">
                      <strong>#{{ archive.id }}</strong>
                      <span>{{ formatDateTime(archive.created_at || archive.date_to) }}</span>
                    </div>
                    <div class="meta-row">
                      <span>{{ t('field.archiveStatus') }}: {{ archive.status }}</span>
                      <span>{{ t('field.archiveLayers') }}: {{ archive.meta?.layers?.join(', ') || '—' }}</span>
                    </div>
                    <FreshnessBadge :meta="archive.freshness" compact />
                    <div class="button-row compact-row">
                      <button class="btn-secondary" @click="store.loadArchiveView(archive.id)">
                        {{ t('field.openArchive') }}
                      </button>
                      <a
                        class="archive-link btn-secondary"
                        :href="`/api/v1/archive/${archive.id}/download`"
                        target="_blank"
                        rel="noreferrer"
                      >
                        {{ t('field.downloadArchive') }}
                      </a>
                    </div>
                  </div>
                </div>
                <div v-else class="placeholder">{{ t('field.noArchives') }}</div>
              </div>

              <div v-if="archiveView" class="section-block">
                <div class="section-caption">{{ t('field.archiveSnapshot') }}</div>
                <div class="summary-grid">
                  <div class="summary-card">
                    <span>{{ t('field.archivePrediction') }}</span>
                    <strong>{{ formatYield(archivePrediction) }}</strong>
                  </div>
                  <div class="summary-card">
                    <span>{{ t('field.confidence') }}</span>
                    <strong>{{ formatPercent(archiveConfidence) }}</strong>
                  </div>
                  <div class="summary-card">
                    <span>{{ t('field.availableMetrics') }}</span>
                    <strong>{{ archiveMetricsLabel }}</strong>
                  </div>
                  <div class="summary-card">
                    <span>{{ t('field.savedScenarios') }}</span>
                    <strong>{{ archiveScenarioCount }}</strong>
                  </div>
                </div>
              </div>

              <div v-if="store.isLoadingArchiveView" class="placeholder">{{ t('field.loadingArchive') }}</div>
            </template>
            <div v-else class="placeholder">{{ t('field.singleOnlyArchive') }}</div>
          </div>

          <!-- ── EVENTS TAB ───────────────────────────────────── -->
          <div v-else-if="store.activeFieldTab === 'events'" class="section-stack">
            <!-- Season filter -->
            <div class="section-block">
              <div class="section-caption">{{ t('field.eventsSeasonFilter') }}</div>
              <div class="button-row compact-row">
                <select class="input-select" v-model="store.selectedEventSeasonYear" @change="store.loadFieldEvents()">
                  <option :value="null">{{ t('field.eventsAllSeasons') }}</option>
                  <option v-for="yr in availableSeasonYears" :key="yr" :value="yr">{{ yr }}</option>
                </select>
                <button class="btn-secondary" @click="store.loadFieldEvents()">{{ t('field.eventsRefresh') }}</button>
              </div>
            </div>

            <!-- Event list -->
            <div class="section-block">
              <div class="section-caption">
                {{ t('field.eventsList') }}
                <span v-if="store.fieldEventsTotal > 0" class="section-count">({{ store.fieldEventsTotal }})</span>
              </div>
              <div v-if="store.isLoadingEvents" class="placeholder">{{ t('field.loadingDashboard') }}</div>
              <div v-else-if="store.fieldEvents.length" class="list-stack">
                <div v-for="event in store.fieldEvents" :key="event.id" class="list-item">
                  <div class="list-item-head">
                    <strong>{{ formatEventType(event.event_type) }}</strong>
                    <span>{{ formatEventDate(event.event_date) }}</span>
                  </div>
                  <div class="meta-row">
                    <span v-if="event.amount !== null && event.amount !== undefined">
                      {{ t('field.eventsAmount') }}: {{ event.amount }} {{ event.unit || '' }}
                    </span>
                    <span>{{ t('field.eventsSeasonYear') }}: {{ event.season_year }}</span>
                    <span class="source-badge" :class="`source-${event.source}`">{{ event.source }}</span>
                  </div>
                  <div v-if="event.source === 'manual'" class="button-row compact-row">
                    <button class="btn-secondary" @click="openEditEvent(event)">{{ t('field.eventsEdit') }}</button>
                    <button class="btn-secondary btn-danger" @click="handleDeleteEvent(event.id)">{{ t('field.eventsDelete') }}</button>
                  </div>
                </div>
              </div>
              <div v-else class="placeholder">{{ t('field.noEvents') }}</div>
            </div>

            <!-- Add / Edit form -->
            <div class="section-block">
              <div class="section-caption">{{ editingEventId ? t('field.eventsEditTitle') : t('field.eventsAdd') }}</div>
              <div class="events-form-grid">
                <div class="events-row1">
                  <div class="feature-item">
                    <span>{{ t('field.eventsType') }}</span>
                    <input class="input-field" v-model="eventForm.event_type" :placeholder="t('field.eventsTypePlaceholder')" list="event-type-suggestions" />
                    <datalist id="event-type-suggestions">
                      <option value="irrigation" />
                      <option value="fertilization" />
                      <option value="tillage" />
                      <option value="sowing" />
                      <option value="harvest" />
                      <option value="pesticide" />
                    </datalist>
                  </div>
                  <div class="feature-item">
                    <span>{{ t('field.eventsDate') }}</span>
                    <input type="date" class="input-field" v-model="eventForm.event_date" />
                  </div>
                </div>
                <div class="events-row2">
                  <div class="feature-item">
                    <span>{{ t('field.eventsAmount') }}</span>
                    <input type="number" class="input-field" v-model.number="eventForm.amount" :placeholder="t('field.eventsAmount')" />
                  </div>
                  <div class="feature-item">
                    <span>{{ t('field.eventsUnit') }}</span>
                    <input class="input-field" v-model="eventForm.unit" :placeholder="t('field.eventsUnitPlaceholder')" />
                  </div>
                  <div class="feature-item">
                    <span>{{ t('field.eventsSeasonYear') }}</span>
                    <input type="number" class="input-field" v-model.number="eventForm.season_year" min="2000" max="2100" />
                  </div>
                </div>
              </div>
              <div class="button-row">
                <button
                  class="btn-secondary"
                  :disabled="store.isSubmittingEvent || !eventForm.event_type || !eventForm.event_date"
                  @click="submitEventForm"
                >
                  {{ store.isSubmittingEvent ? t('field.eventsSubmitting') : t('field.eventsSave') }}
                </button>
                <button v-if="editingEventId" class="btn-secondary" @click="cancelEditEvent">
                  {{ t('field.eventsCancelEdit') }}
                </button>
              </div>
              <div v-if="eventFormError" class="info-box info-box-warn">{{ eventFormError }}</div>
            </div>
          </div>
        </template>
      </template>
    </div>
  </section>
</template>

<script setup>
/**
 * FieldActionsPanel — thin presenter component.
 *
 * All reactive data & business logic live in the useFieldPanelData composable.
 * This component is responsible only for rendering and wiring up async sub-components.
 */
import { defineAsyncComponent } from 'vue'
import FreshnessBadge from './FreshnessBadge.vue'
import MiniHistogram from './MiniHistogram.vue'
import MiniSparkline from './MiniSparkline.vue'
import ScenarioResponseChart from './ScenarioResponseChart.vue'
import { useFieldPanelData, formatYield, formatPercent, formatDelta, formatInteger, formatDecimal, formatDateTime, resolveMetricColor, formatMetricValue, seriesAreIdentical, mapForecastCurvePoints, formatTaskTiming } from '../composables/useFieldPanelData'
import { locale, t } from '../utils/i18n'

const XYSeriesChart = defineAsyncComponent(() => import('./XYSeriesChart.vue'))
const NumericTimeline = defineAsyncComponent(() => import('./NumericTimeline.vue'))
const FactorWaterfallChart = defineAsyncComponent(() => import('./FactorWaterfallChart.vue'))

const {
  store, isGroup, dashboard, isLoadingDashboard, isPreviewField, prediction,
  temporalAnalytics, forecastTemporalAnalytics, managementZones, analyticsSummary,
  waterBalance, riskSummary, managementZoneSummary, debugRuntime,
  visibleTabs,
  onDateRangeChange, clearDateRange, onDateInput,
  sourceLabel, qualityLabel, qualityBandLabel, qualityReasonLabel,
  fieldOperationalTierLabel, fieldReviewRequiredLabel, fieldReviewReason,
  totalAreaLabel, perimeterLabel, observationCellsLabel, availableMetricsLabel,
  dataQualityText, overviewCards,
  metricCards, seriesEntries, histogramEntries,
  seasonalMetricEntries, selectedSeasonalMetricId, selectedSeasonalMetric,
  seasonalMetricSelectorOptions, metricsAnomalyItems, forecastAnomalyItems,
  selectedMetricAnomalies, seasonalAnomalyRows, seasonalMetricSeries,
  selectedSeasonalMetricLabel, seasonalMetricPointCountLabel, seasonalMetricTimeline,
  metricsDataStatusMessage, forecastDataStatusMessage, seasonalMetricChartEmptyText,
  temporalTaskFailureCode,
  predictionProgressActive, scenarioProgressActive, temporalProgressActive,
  predictionProgressLabel, scenarioProgressLabel, temporalProgressLabel,
  predictionTaskStage, predictionTaskDetail, predictionTaskTiming,
  scenarioTaskStage, scenarioTaskDetail, scenarioTaskTiming,
  temporalTaskStage, temporalTaskDetail, temporalTaskTiming,
  predictionTaskLogs, scenarioTaskLogs,
  yieldLabel, confidenceLabel, predictionDateLabel, predictionConfidenceTierLabel,
  predictionOperationalTierLabel, predictionReviewRequiredLabel, predictionReviewReason,
  predictionSupportReason, predictionDrivers, featureEntries, qualityEntries, suitabilityEntries,
  gddCumulativeSeries, gddCumulativeTotal, waterBalanceSeries,
  historyTrend, historyTrendChartSeries, forecastTrendMarkers, forecastTrendRanges,
  forecastCurveMetric, scenarioForecastCurveMetric, scenarioComparisonMetric, forecastCurveMetricOptions,
  predictionForecastCurveSeries, predictionForecastCurveEmptyText,
  modelingBaselineLabel, modelingScenarioLabel, modelingDeltaLabel,
  modelingRiskLevel, modelingRiskComment, assumptionEntries,
  scenarioOperationalTierLabel, scenarioReviewRequiredLabel, scenarioReviewReason,
  riskLevelClass, factorBreakdown, scenarioWarnings, comparisonChart,
  selectedSweepParam, sensitivityParamLabel, currentSweepParamValue,
  expectedRainAutoBadge, expectedRainAutoSourceTitle, soilCompactionAutoSourceTitle,
  archiveView, archivePrediction, archiveConfidence, archiveMetricsLabel, archiveScenarioCount,
  managementZoneRows, managementZonesSupported, managementZoneModeLabel,
  phenologySummaryEntries, analyticsAlertEntries,
  geometryDiagnosticsEntries, geometryDebugStatus, modelFoundationEntries, retrainDescription,
  eventForm, editingEventId, eventFormError, availableSeasonYears,
  formatEventType, formatEventDate, openEditEvent, cancelEditEvent, submitEventForm, handleDeleteEvent,
  riskLevelLabel, scenarioRiskItemLabel, scenarioRiskItemReason,
} = useFieldPanelData()

</script>


<style scoped>
.date-range-row {
  display: flex;
  gap: 8px;
  align-items: flex-end;
  margin-bottom: 8px;
}
.date-range-row label {
  display: flex;
  flex-direction: column;
  gap: 2px;
  font-size: 11px;
}
.date-range-row input[type="date"] {
  font-size: 11px;
  padding: 2px 4px;
}
.btn-sm {
  font-size: 11px;
  padding: 2px 6px;
}
.field-actions-panel {
  min-height: 0;
}

.panel-body {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.selection-strip,
.info-box,
.forecast-hero,
.metric-card,
.chart-card,
.summary-card,
.feature-item,
.list-item,
.scenario-box {
  border: 2px solid;
  border-color: var(--win-shadow-mid) var(--win-shadow-light) var(--win-shadow-light) var(--win-shadow-mid);
  background: var(--weather-cell-bg);
}

.selection-strip,
.forecast-hero,
.info-box,
.metric-card,
.chart-card,
.summary-card,
.feature-item,
.list-item,
.scenario-box {
  padding: 10px;
}

.selection-strip {
  display: flex;
  align-items: start;
  justify-content: space-between;
  gap: 10px;
}

.selection-title {
  font-weight: 700;
  margin-bottom: 4px;
}

.selection-subtitle,
.meta-row,
.feature-item span,
.summary-card span {
  color: var(--text-muted);
  font-size: 12px;
}

.task-progress-card {
  border: 2px solid;
  border-color: var(--win-shadow-mid) var(--win-shadow-light) var(--win-shadow-light) var(--win-shadow-mid);
  background: rgba(191, 210, 230, 0.22);
  padding: 8px 10px;
  display: grid;
  gap: 6px;
}

.task-progress-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  font-size: 12px;
}

.task-progress-track {
  height: 12px;
  border: 2px solid;
  border-color: var(--win-shadow-mid) var(--win-shadow-light) var(--win-shadow-light) var(--win-shadow-mid);
  background: #efefef;
  overflow: hidden;
}

.task-progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #2f6b97, #3c9c64);
  transition: width 0.25s ease;
}

.task-progress-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  font-size: 11px;
  color: var(--text-muted);
}

.task-log-list {
  display: grid;
  gap: 3px;
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px solid var(--win-shadow-soft);
}

.task-log-line {
  font-size: 11px;
  color: var(--text-muted);
}

.tab-strip {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.tab-btn,
.btn-secondary,
.archive-link {
  min-height: 34px;
  border: 2px solid;
  border-color: var(--win-shadow-light) var(--win-shadow-dark) var(--win-shadow-dark) var(--win-shadow-light);
  background: var(--win-bg);
  color: var(--text-main);
  font-weight: 700;
  cursor: pointer;
  text-decoration: none;
}

.tab-btn {
  padding: 0 10px;
}

.tab-btn.active {
  background: #bfd2e6;
}

.btn-secondary.active {
  background: #bfd2e6;
}

.section-stack,
.section-block {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.section-caption {
  font-weight: 700;
}

.summary-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 8px;
}

.metrics-grid,
.feature-grid,
.chart-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}

.metric-card-head,
.chart-head,
.list-item-head {
  display: flex;
  justify-content: space-between;
  gap: 8px;
  margin-bottom: 6px;
}

.metric-card-meta {
  display: flex;
  flex-direction: column;
  gap: 4px;
  color: var(--text-muted);
  font-size: 12px;
}

.input-grid {
  display: grid;
  gap: 10px;
}

.input-grid.two-col {
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
}

.input-grid.three-col {
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
}

label {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
}

label span {
  color: var(--text-muted);
  font-size: 12px;
}

.button-row {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.button-row > * {
  min-width: 0;
}

.compact-row {
  align-items: center;
}

.forecast-main {
  font-size: 28px;
  font-weight: 700;
  color: var(--text-main);
}

.forecast-meta {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  color: var(--text-muted);
}

.drivers-list,
.list-stack {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.chart-card-wide {
  min-height: 220px;
}

.compact-grid {
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
}

.events-form-grid {
  display: grid;
  gap: 6px;
}

.events-row1 {
  display: grid;
  grid-template-columns: 1fr 148px;
  gap: 6px;
}

.events-row2 {
  display: grid;
  grid-template-columns: 1fr 1fr 88px;
  gap: 6px;
}

.driver-row,
.meta-row {
  display: flex;
  justify-content: space-between;
  gap: 8px;
  flex-wrap: wrap;
}

.archive-link {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 0 10px;
}

.comparison-box {
  display: grid;
  gap: 10px;
}

.comparison-title {
  font-size: 12px;
}

.comparison-track {
  position: relative;
  height: 22px;
  border: 2px solid;
  border-color: var(--win-shadow-mid) var(--win-shadow-light) var(--win-shadow-light) var(--win-shadow-mid);
  background: linear-gradient(90deg, rgba(47, 107, 151, 0.08), rgba(60, 156, 100, 0.08));
}

.comparison-interval {
  position: absolute;
  top: 3px;
  bottom: 3px;
  background: rgba(63, 126, 232, 0.18);
  border: 1px solid rgba(63, 126, 232, 0.36);
}

.comparison-marker {
  position: absolute;
  top: 1px;
  bottom: 1px;
  width: 3px;
  transform: translateX(-50%);
}

.baseline-marker {
  background: #2f6b97;
}

.scenario-marker {
  background: #3c9c64;
}

.comparison-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  font-size: 12px;
  color: var(--text-muted);
}

.legend-dot,
.legend-band {
  display: inline-block;
  width: 10px;
  height: 10px;
  margin-right: 4px;
  vertical-align: middle;
}

.baseline-dot {
  background: #2f6b97;
}

.scenario-dot {
  background: #3c9c64;
}

.legend-band {
  background: rgba(63, 126, 232, 0.18);
  border: 1px solid rgba(63, 126, 232, 0.36);
}

.risk-low { color: #27ae60; }
.risk-moderate { color: #f39c12; }
.risk-elevated { color: #e67e22; }
.risk-high { color: #c0392b; }

.explain-summary {
  margin-bottom: 6px;
}

@media (max-width: 720px) {
  .summary-grid,
  .metrics-grid,
  .feature-grid,
  .chart-grid,
  .input-grid.two-col,
  .input-grid.three-col {
    grid-template-columns: 1fr;
  }

  .selection-strip {
    flex-direction: column;
  }
}

.sensitivity-box {
  display: grid;
  gap: 8px;
}

.sensitivity-controls {
  display: flex;
  gap: 6px;
  align-items: center;
  flex-wrap: wrap;
}

.sensitivity-select {
  flex: 1;
  min-width: 0;
}

.satellite-derived input,
.satellite-derived select {
  background: rgba(47, 107, 151, 0.07);
  border-color: rgba(47, 107, 151, 0.35);
  color: inherit;
}

.sat-badge {
  font-size: 9px;
  margin-left: 3px;
  vertical-align: middle;
  opacity: 0.75;
}

.weather-badge {
  color: #2f6b97;
}

.auto-source-hint {
  font-size: 10px;
  color: var(--text-muted);
  font-style: italic;
  margin-top: 2px;
}

.section-count {
  font-weight: normal;
  font-size: 11px;
  margin-left: 4px;
  opacity: 0.65;
}

.source-badge {
  display: inline-block;
  font-size: 9px;
  padding: 1px 5px;
  border-radius: 2px;
  background: rgba(33, 57, 87, 0.1);
  color: var(--text-muted, rgba(33, 57, 87, 0.7));
  text-transform: lowercase;
}

.source-badge.source-manual {
  background: rgba(47, 107, 151, 0.15);
  color: #2f6b97;
}

.btn-danger {
  color: #b43f2d;
  border-color: rgba(180, 63, 45, 0.35);
}

.btn-danger:hover {
  background: rgba(180, 63, 45, 0.08);
}

.info-box-warn {
  background: rgba(204, 139, 25, 0.1);
  border: 1px solid rgba(204, 139, 25, 0.4);
  color: #7a5000;
  padding: 6px 8px;
  font-size: 11px;
  margin-top: 6px;
}

.input-select {
  font-family: inherit;
  font-size: 11px;
  padding: 3px 5px;
  border: 1px solid rgba(33, 57, 87, 0.35);
  background: #f8f5ee;
  color: var(--text-primary);
  cursor: pointer;
}

.input-field,
label input,
label select,
label textarea,
.feature-item input,
.feature-item select,
.feature-item textarea {
  width: 100%;
  min-width: 0;
  max-width: 100%;
  box-sizing: border-box;
}

.feature-item {
  min-width: 0;
}
</style>
