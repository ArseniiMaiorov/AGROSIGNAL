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
import { computed, defineAsyncComponent, ref, watch, watchEffect } from 'vue'
import FreshnessBadge from './FreshnessBadge.vue'
import MiniHistogram from './MiniHistogram.vue'
import MiniSparkline from './MiniSparkline.vue'
import ScenarioResponseChart from './ScenarioResponseChart.vue'
import { useMapStore } from '../store/map'
import { locale, t } from '../utils/i18n'
import {
  formatDisplayValue,
  formatReasonText,
  formatUiDateTime,
  formatUiProgress,
  getConfidenceTierLabel,
  getFeatureLabel,
  getLayerMeta,
  getQualityBandLabel,
  getRiskItemLabel,
  getRiskItemReason,
  getRiskLevelLabel,
  getSourceLabel,
  getTaskStageDetail,
  getTaskStageLabel,
} from '../utils/presentation'

const store = useMapStore()
const XYSeriesChart = defineAsyncComponent(() => import('./XYSeriesChart.vue'))
const NumericTimeline = defineAsyncComponent(() => import('./NumericTimeline.vue'))
const FactorWaterfallChart = defineAsyncComponent(() => import('./FactorWaterfallChart.vue'))

const selectedSweepParam = ref('fertilizer_pct')
const sensitivityParamLabel = computed(() => {
  const labels = {
    irrigation_pct: t('field.irrigation'),
    fertilizer_pct: t('field.fertilizer'),
    expected_rain_mm: t('field.expectedRain'),
    temperature_delta_c: t('field.temperatureDelta'),
  }
  return labels[selectedSweepParam.value] || ''
})
const currentSweepParamValue = computed(() => {
  const factors = store.modelingResult?.factors || {}
  return Number(factors[selectedSweepParam.value]) || 0
})

const expectedRainAutoBadge = computed(() => {
  const source = store.modelingAutoSources?.expected_rain_mm || ''
  if (source === 'satellite_wetness_bsi' || source === 'satellite_wetness') return '🛰'
  if (source === 'forecast_curve') return '☁'
  if (source === 'observed_weather') return '☔'
  return '🛰'
})

const expectedRainAutoSourceTitle = computed(() => {
  const source = store.modelingAutoSources?.expected_rain_mm || ''
  if (source === 'satellite_wetness_bsi') {
    return t('field.expectedRainFromSatelliteBsi')
  }
  if (source === 'satellite_wetness') {
    return t('field.expectedRainFromSatelliteWetness')
  }
  if (source === 'forecast_curve') {
    return t('field.expectedRainFromForecast')
  }
  if (source === 'observed_weather') {
    return t('field.expectedRainFromObservedWeather')
  }
  return t('field.expectedRainAutoHint')
})

const soilCompactionAutoSourceTitle = computed(() => {
  const source = store.modelingAutoSources?.soil_compaction || ''
  if (source === 'satellite_soil_moisture_bsi') {
    return t('field.soilCompactionFromSatelliteBsi')
  }
  if (source === 'satellite_soil_moisture') {
    return t('field.soilCompactionFromSatelliteMoisture')
  }
  return t('field.soilCompactionAutoHint')
})

function onDateRangeChange() {
  // Auto-swap if from > to to prevent reversed ranges
  if (store.seriesDateFrom && store.seriesDateTo && store.seriesDateFrom > store.seriesDateTo) {
    const tmp = store.seriesDateFrom
    store.seriesDateFrom = store.seriesDateTo
    store.seriesDateTo = tmp
  }
  store.loadFieldTemporalAnalytics(undefined, {
    target: 'metrics',
    preferExisting: false,
    autoBackfill: true,
  })
}

function clearDateRange() {
  store.seriesDateFrom = ''
  store.seriesDateTo = ''
  store.loadFieldTemporalAnalytics(undefined, {
    target: 'metrics',
    preferExisting: false,
    autoBackfill: true,
  })
}

const isGroup = computed(() => store.hasGroupSelection)
const dashboard = computed(() => store.activeDashboard)
const isLoadingDashboard = computed(() => store.isLoadingFieldDashboard || store.isLoadingGroupDashboard)
const isPreviewField = computed(() => !isGroup.value && store.selectedFieldIsPreviewOnly)

const visibleTabs = computed(() => {
  if (isGroup.value || isPreviewField.value) {
    return [
      { id: 'overview', label: t('field.overview') },
      { id: 'metrics', label: t('field.metricsTab') },
    ]
  }
  return [
    { id: 'overview', label: t('field.overview') },
    { id: 'metrics', label: t('field.metricsTab') },
    { id: 'forecast', label: t('field.forecastTab') },
    { id: 'scenarios', label: t('field.scenarioTab') },
    { id: 'archive', label: t('field.archiveTab') },
    { id: 'events', label: t('field.eventsTab') },
  ]
})

watchEffect(() => {
  if (isPreviewField.value && ['forecast', 'scenarios', 'archive', 'events'].includes(store.activeFieldTab)) {
    store.setActiveFieldTab('overview')
  }
})

// Auto-reload temporal analytics when switching to metrics/forecast tab
watch(
  () => store.activeFieldTab,
  (tab, prevTab) => {
    if (tab === prevTab) return
    const fieldId = store.selectedField?.field_id
    if (!fieldId || store.hasGroupSelection) return
    if (tab === 'metrics') {
      store.loadFieldTemporalAnalytics(fieldId, {
        target: 'metrics',
        preferExisting: true,
        autoBackfill: true,
        silent: true,
      })
    } else if (tab === 'forecast') {
      store.loadFieldTemporalAnalytics(fieldId, {
        target: 'forecast',
        preferExisting: true,
        autoBackfill: true,
        silent: true,
      })
    }
  },
)

// Auto-reload temporal analytics when switching display modes that need series data
watch(
  () => store.metricsDisplayMode,
  (mode) => {
    if (!['xy', 'timeline', 'anomalies', 'cards'].includes(mode)) return
    const fieldId = store.selectedField?.field_id
    if (!fieldId || store.hasGroupSelection) return
    if (store.activeFieldTab === 'metrics') {
      store.loadFieldTemporalAnalytics(fieldId, {
        target: 'metrics',
        preferExisting: true,
        autoBackfill: true,
        silent: true,
      })
    }
  },
)

// Re-run auto-fill when satellite data arrives (fieldDashboard / temporal analytics change)
// so values are updated even if data loaded after the button click
watch(
  () => [store.fieldDashboard, store.fieldForecastAnalytics, store.fieldTemporalAnalytics],
  () => {
    if (!store.useManualModeling) {
      store.autoFillModelingFactors()
    }
  },
  { deep: false },
)

// Debounced date range reload: reload 600ms after the user stops typing a date
let _dateDebounceTimer = null
function onDateInput() {
  if (_dateDebounceTimer) clearTimeout(_dateDebounceTimer)
  _dateDebounceTimer = setTimeout(() => {
    _dateDebounceTimer = null
    onDateRangeChange()
  }, 600)
}

const sourceLabel = computed(() => {
  const source = dashboard.value?.field?.source || store.selectedField?.source
  if (!source) return '—'
  return getSourceLabel(source)
})

const qualityLabel = computed(() => {
  const value = dashboard.value?.field?.quality_score ?? store.selectedField?.quality_score
  return value === null || value === undefined ? t('field.qualityUnknown') : `${t('field.quality')}: ${Number(value).toFixed(2)}`
})

const qualityBandLabel = computed(() => {
  const label = dashboard.value?.field?.quality_label || store.selectedField?.quality_label || dashboard.value?.field?.quality_band || store.selectedField?.quality_band
  if (!label) {
    return t('field.qualityUnknown')
  }
  return `${t('field.qualityBand')}: ${getQualityBandLabel(label)}`
})

const qualityReasonLabel = computed(() => {
  return formatReasonText(
    dashboard.value?.field?.quality_reason_code || store.selectedField?.quality_reason_code,
    dashboard.value?.field?.quality_reason || store.selectedField?.quality_reason,
    dashboard.value?.field?.quality_reason_params || store.selectedField?.quality_reason_params,
  ) || t('field.qualityReasonMissing')
})

const fieldOperationalTierLabel = computed(() => {
  const tier = dashboard.value?.field?.operational_tier || store.selectedField?.operational_tier
  return tier ? t(`field.operationalTier.${tier}`) : '—'
})

const fieldReviewRequiredLabel = computed(() => {
  const reviewRequired = dashboard.value?.field?.review_required ?? store.selectedField?.review_required
  return reviewRequired ? t('field.yes') : t('field.no')
})

const fieldReviewReason = computed(() => {
  return formatReasonText(
    dashboard.value?.field?.review_reason_code || store.selectedField?.review_reason_code,
    dashboard.value?.field?.review_reason || store.selectedField?.review_reason,
    dashboard.value?.field?.review_reason_params || store.selectedField?.review_reason_params,
  ) || ''
})

const totalAreaLabel = computed(() => {
  if (isGroup.value) {
    const areaM2 = Number(dashboard.value?.selection?.total_area_m2 || 0)
    return `${(areaM2 / 10000).toFixed(2)} га`
  }
  const areaM2 = Number(dashboard.value?.field?.area_m2 || store.selectedField?.area_m2 || 0)
  return `${(areaM2 / 10000).toFixed(2)} га`
})

const overviewCards = computed(() => {
  if (!dashboard.value) return []
  if (isGroup.value) {
    return [
      { label: t('field.selectedCount'), value: String(store.selectedFieldCount) },
      { label: t('field.area'), value: totalAreaLabel.value },
      { label: t('field.availableMetrics'), value: availableMetricsLabel.value },
      { label: t('field.metricCoverage'), value: observationCellsLabel.value },
    ]
  }
  if (isPreviewField.value) {
    return [
      { label: t('field.area'), value: totalAreaLabel.value },
      { label: t('field.perimeter'), value: perimeterLabel.value },
      { label: t('field.metricsTab'), value: availableMetricsLabel.value },
      { label: t('field.reviewNeeded'), value: fieldReviewRequiredLabel.value },
      { label: t('field.previewOnlyTitle'), value: t('field.previewOnly') },
      { label: t('field.forecastReady'), value: t('field.no') },
    ]
  }
  return [
    { label: t('field.area'), value: totalAreaLabel.value },
    { label: t('field.perimeter'), value: perimeterLabel.value },
    { label: t('field.archiveTab'), value: String(dashboard.value?.kpis?.archive_count || 0) },
    { label: t('field.scenarioTab'), value: String(dashboard.value?.kpis?.scenario_count || 0) },
    { label: t('field.metricCoverage'), value: observationCellsLabel.value },
    { label: t('field.forecastReady'), value: dashboard.value?.kpis?.prediction_ready ? t('field.yes') : t('field.no') },
  ]
})

const perimeterLabel = computed(() => {
  const value = Number(dashboard.value?.field?.perimeter_m || store.selectedField?.perimeter_m || 0)
  return value ? `${Math.round(value)} м` : '—'
})

const observationCellsLabel = computed(() => {
  const value = dashboard.value?.kpis?.observation_cells ?? dashboard.value?.data_quality?.observation_cells
  return value ? `${Math.round(Number(value))}` : '—'
})

const availableMetricsLabel = computed(() => {
  const keys = dashboard.value?.data_quality?.metrics_available || []
  return keys.length ? keys.map((metric) => getLayerMeta(metric).label).join(', ') : t('field.noMetrics')
})

const dataQualityText = computed(() => {
  if (!dashboard.value?.data_quality) return t('field.noDataQuality')
  if (isGroup.value) {
    return `${t('field.groupSelection')}: ${store.selectedFieldCount} · ${t('field.metricCoverage')}: ${observationCellsLabel.value}`
  }
  return dashboard.value?.prediction?.data_quality?.confidence_reason || t('field.dataQualityReady')
})

const metricCards = computed(() => {
  const metrics = dashboard.value?.current_metrics || {}
  return Object.entries(metrics).map(([id, payload]) => ({
    id,
    label: getFeatureLabel(id, { expertMode: store.expertMode }),
    mean: formatMetricValue(id, payload.mean),
    median: formatMetricValue(id, payload.median),
    range: `${formatMetricValue(id, payload.min)} … ${formatMetricValue(id, payload.max)}`,
    coverage: payload.coverage ? `${Math.round(Number(payload.coverage))}` : '—',
  }))
})

const SERIES_PALETTE = {
  ndvi: '#2f8a63',
  ndmi: '#1f6aa0',
  ndwi: '#1b87b7',
  bsi: '#b37632',
  gdd: '#c98b24',
  gdd_daily: '#c98b24',
  gdd_cumulative: '#d28a1f',
  vpd: '#8e4fc6',
  soil_moisture: '#3d8f7f',
  precipitation: '#3f7ee8',
  precipitation_mm: '#3f7ee8',
  temperature_mean_c: '#c45b2d',
  wind: '#5d6fb3',
}

// Weather/climate metrics sourced from temporal analytics (FieldFeatureWeekly),
// not from FieldMetricSeries (satellite snapshots only). These need to be pulled
// from temporalAnalytics.seasonal_series and mapped to {mean} for MiniSparkline.
const TEMPORAL_WEATHER_METRICS = new Set(['precipitation', 'soil_moisture', 'vpd', 'wind', 'gdd', 'temperature_mean_c'])

const seriesEntries = computed(() => {
  const entries = {}
  const temporalMetrics = temporalAnalytics.value?.seasonal_series?.metrics || []
  const temporalWeatherEntries = new Map()

  // Weather/climate series from temporal analytics (FieldFeatureWeekly) should
  // drive the chart for the selected period. Dashboard weather series are often
  // just the latest run snapshot, so they are only used as a fallback.
  for (const metricObj of temporalMetrics) {
    const id = metricObj.metric
    const rawPoints = metricObj.points || []
    const items = rawPoints.map((p) => ({
      mean: p.smoothed ?? p.value,
      min: p.value,
      max: p.value,
      observed_at: p.observed_at,
    }))
    if (!TEMPORAL_WEATHER_METRICS.has(id) || !items.length) continue
    temporalWeatherEntries.set(id, {
      id,
      label: metricObj.label || getFeatureLabel(id, { expertMode: store.expertMode }),
      items,
      latest: formatMetricValue(id, items[items.length - 1].mean),
      color: SERIES_PALETTE[id] || '#21579c',
    })
  }

  // 1. Satellite snapshot series from FieldMetricSeries (NDVI, NDMI, NDWI, BSI, ...)
  const dashSeries = dashboard.value?.series || {}
  for (const [id, items] of Object.entries(dashSeries)) {
    if (TEMPORAL_WEATHER_METRICS.has(id) && temporalWeatherEntries.has(id)) {
      continue
    }
    entries[id] = {
      id,
      label: getFeatureLabel(id, { expertMode: store.expertMode }),
      items,
      latest: items?.length ? formatMetricValue(id, items[items.length - 1].mean) : '—',
      color: SERIES_PALETTE[id] || '#21579c',
    }
  }

  // 2. Weather/climate series from temporal analytics (FieldFeatureWeekly).
  // Points use {value, smoothed} — aliased to {mean} for MiniSparkline compatibility.
  for (const metricObj of temporalMetrics) {
    const id = metricObj.metric
    if (entries[id]) continue // satellite snapshot takes priority
    const rawPoints = metricObj.points || []
    const items = rawPoints.map((p) => ({ mean: p.smoothed ?? p.value, min: p.value, max: p.value, observed_at: p.observed_at }))
    const latest = items.length ? formatMetricValue(id, items[items.length - 1].mean) : '—'
    entries[id] = {
      id,
      label: metricObj.label || getFeatureLabel(id, { expertMode: store.expertMode }),
      items,
      latest,
      color: SERIES_PALETTE[id] || '#21579c',
    }
  }

  // Sort: satellite indices first, then weather metrics
  return Object.values(entries).sort((a, b) => {
    const aIsWeather = TEMPORAL_WEATHER_METRICS.has(a.id)
    const bIsWeather = TEMPORAL_WEATHER_METRICS.has(b.id)
    if (aIsWeather !== bIsWeather) return aIsWeather ? 1 : -1
    return 0
  })
})

const histogramEntries = computed(() => {
  const palette = {
    ndvi: '#2f8a63',
    ndmi: '#1f6aa0',
    soil_moisture: '#3d8f7f',
    vpd: '#8e4fc6',
  }
  const histograms = dashboard.value?.histograms || {}
  return Object.entries(histograms).map(([id, histogram]) => ({
    id,
    label: getFeatureLabel(id, { expertMode: store.expertMode }),
    histogram,
    color: palette[id] || '#2f8a63',
  }))
})

const prediction = computed(() => store.selectedFieldPrediction || dashboard.value?.prediction)
const temporalAnalytics = computed(() => store.fieldTemporalAnalytics || null)
const forecastTemporalAnalytics = computed(() => store.fieldForecastAnalytics || null)
const managementZones = computed(() => store.fieldManagementZones || null)
const analyticsSummary = computed(() => dashboard.value?.analytics_summary || temporalAnalytics.value?.analytics_summary || {})
const supportedSections = computed(() => dashboard.value?.supported_sections || temporalAnalytics.value?.supported_sections || {})
const metricsDataStatus = computed(() => temporalAnalytics.value?.data_status || null)
const forecastDataStatus = computed(() => forecastTemporalAnalytics.value?.data_status || null)

const seasonalMetricEntries = computed(() => temporalAnalytics.value?.seasonal_series?.metrics || [])
const selectedSeasonalMetricId = computed(() => {
  const configured = store.metricsSelectedSeries
  if (seasonalMetricEntries.value.some((item) => item.metric === configured)) {
    return configured
  }
  return seasonalMetricEntries.value[0]?.metric || 'ndvi'
})
const selectedSeasonalMetric = computed(() => {
  return seasonalMetricEntries.value.find((item) => item.metric === selectedSeasonalMetricId.value) || seasonalMetricEntries.value[0] || null
})
const seasonalMetricSelectorOptions = computed(() => {
  return seasonalMetricEntries.value.map((item) => ({
    id: item.metric,
    label: item.label || getFeatureLabel(item.metric, { expertMode: store.expertMode }),
  }))
})
const metricsAnomalyItems = computed(() => temporalAnalytics.value?.anomalies || [])
const forecastAnomalyItems = computed(() => forecastTemporalAnalytics.value?.anomalies || prediction.value?.anomalies || [])
const historyTrend = computed(() => prediction.value?.history_trend || forecastTemporalAnalytics.value?.history_trend || temporalAnalytics.value?.history_trend || {})
const waterBalance = computed(() => forecastTemporalAnalytics.value?.water_balance || prediction.value?.water_balance || {})
const riskSummary = computed(() => forecastTemporalAnalytics.value?.risk || prediction.value?.risk || {})
const managementZoneSummary = computed(() => prediction.value?.management_zone_summary || managementZones.value?.summary || dashboard.value?.zones_summary || {})
const debugRuntime = computed(() => store.selectedDebugTileDetail?.runtime_meta || null)

const yieldLabel = computed(() => formatYield(prediction.value?.estimated_yield_kg_ha))
const confidenceLabel = computed(() => formatPercent(prediction.value?.confidence))
const predictionDateLabel = computed(() => formatDateTime(prediction.value?.prediction_date))

const predictionDrivers = computed(() => {
  const drivers = prediction.value?.driver_breakdown || prediction.value?.explanation?.drivers || []
  return drivers.map((driver) => ({
    label: getFeatureLabel(driver.input_key || driver.driver_id || driver.factor || driver.label, { expertMode: store.expertMode }) || driver.label,
    effect: driver.effect_kg_ha !== null && driver.effect_kg_ha !== undefined
      ? `${Number(driver.effect_kg_ha) >= 0 ? '+' : ''}${Number(driver.effect_kg_ha).toFixed(0)} кг/га`
      : formatSigned(driver.effect_pct),
  }))
})

const predictionConfidenceTierLabel = computed(() => {
  if (!prediction.value?.confidence_tier) return ''
  return `${locale.value === 'ru' ? 'Контур доверия' : 'Confidence tier'}: ${getConfidenceTierLabel(prediction.value.confidence_tier)}`
})

const predictionOperationalTierLabel = computed(() => {
  const tier = prediction.value?.operational_tier
  return tier ? t(`field.operationalTier.${tier}`) : '—'
})

const predictionReviewRequiredLabel = computed(() => {
  return prediction.value?.review_required ? t('field.yes') : t('field.no')
})

const predictionReviewReason = computed(() => {
  return formatReasonText(
    prediction.value?.review_reason_code,
    prediction.value?.review_reason,
    prediction.value?.review_reason_params,
  ) || ''
})

const predictionSupportReason = computed(() => {
  return formatReasonText(
    prediction.value?.support_reason_code,
    prediction.value?.support_reason,
    prediction.value?.support_reason_params,
  ) || ''
})

const featureEntries = computed(() => {
  const features = prediction.value?.input_features || {}
  return Object.entries(features)
    .filter(([key, value]) => {
      if (store.expertMode) return true
      if (String(key || '').startsWith('_')) return false
      return value !== null && value !== undefined && value !== ''
    })
    .map(([key, value]) => ({
      label: getFeatureLabel(key, { expertMode: store.expertMode }),
      value: formatFeatureValue(key, value),
    }))
})

const qualityEntries = computed(() => {
  const quality = prediction.value?.data_quality || {}
  return Object.entries(quality).map(([key, value]) => ({
    label: getFeatureLabel(key, { expertMode: store.expertMode }),
    value: formatFeatureValue(key, value),
  }))
})

const SUITABILITY_STATUS_LABELS = {
  high: 'Высокая — оптимально для культуры',
  moderate: 'Умеренная — возможны ограничения',
  low: 'Низкая — требует интенсивной агротехники',
  unsuitable: 'Не подходит — риски критические',
}
const suitabilityEntries = computed(() => {
  const cs = prediction.value?.crop_suitability || {}
  // Fields to skip entirely (internal or covered separately)
  const SKIP = new Set(['reasons', 'support_reason', 'yield_factor', 'warnings', 'recommendation'])
  const entries = []
  for (const [key, value] of Object.entries(cs)) {
    if (SKIP.has(key)) continue
    // Skip null/undefined seasonal data — don't show "—" for missing weather
    if (value === null || value === undefined) continue
    // Skip empty arrays
    if (Array.isArray(value) && value.length === 0) continue
    let displayValue
    if (key === 'status') {
      displayValue = SUITABILITY_STATUS_LABELS[value] || String(value)
    } else {
      displayValue = formatFeatureValue(key, value)
    }
    entries.push({ label: getFeatureLabel(key, { expertMode: store.expertMode }), value: displayValue })
  }
  return entries
})

const modelingBaselineLabel = computed(() => formatYield(store.modelingResult?.baseline_yield_kg_ha))
const modelingScenarioLabel = computed(() => formatYield(store.modelingResult?.scenario_yield_kg_ha))
const modelingDeltaLabel = computed(() => formatDelta(store.modelingResult?.predicted_yield_change_pct))
const modelingRiskLevel = computed(() => getRiskLevelLabel(store.modelingResult?.risk_summary?.level_code || store.modelingResult?.risk_summary?.level || ''))
const modelingRiskComment = computed(() => store.modelingResult?.risk_summary?.comment || '')
const assumptionEntries = computed(() => {
  const assumptions = store.modelingResult?.assumptions || {}
  return Object.entries(assumptions).map(([key, value]) => ({
    label: getFeatureLabel(key, { expertMode: store.expertMode }),
    value: formatFeatureValue(key, value),
  }))
})

const factorBreakdown = computed(() => {
  // Prefer scenario delta breakdown; fall back to absolute driver breakdown
  const all = store.modelingResult?.comparison?.factor_breakdown
    || store.modelingResult?.driver_breakdown
    || []
  // Remove raw scenario percentage inputs — they clutter the chart with ~0 bars
  const RAW_INPUT_KEYS = new Set([
    'irrigation_pct', 'fertilizer_pct', 'expected_rain_mm',
    'temperature_delta_c', 'planting_density_pct', 'cloud_cover_factor',
  ])
  return all.filter((f) => {
    const key = f.input_key || f.driver_id || f.factor || ''
    return !RAW_INPUT_KEYS.has(key)
  })
})

const scenarioWarnings = computed(() => {
  const warnings = store.modelingResult?.constraint_warnings || []
  return warnings.map((warning) => {
    if (warning === store.modelingResult?.support_reason) {
      return formatReasonText(
        store.modelingResult?.support_reason_code,
        warning,
        store.modelingResult?.support_reason_params,
      ) || warning
    }
    return warning
  })
})
const scenarioSeasonalConstraints = computed(() => {
  const cs = store.modelingResult?.crop_suitability || {}
  const entries = []
  if (cs.seasonal_gdd_sum != null)
    entries.push({ label: t('field.seasonalGdd') || 'Сумма ГСТ за сезон', value: `${Number(cs.seasonal_gdd_sum).toFixed(0)} °C·д` })
  if (cs.seasonal_precipitation_mm != null)
    entries.push({ label: t('field.seasonalPrecip') || 'Осадки за сезон', value: `${Number(cs.seasonal_precipitation_mm).toFixed(0)} мм` })
  if (cs.seasonal_temperature_mean_c != null)
    entries.push({ label: t('field.seasonalTemp') || 'Средняя температура', value: `${Number(cs.seasonal_temperature_mean_c).toFixed(1)} °C` })
  if (cs.score != null) {
    const statusLabels = { high: 'Высокая', moderate: 'Умеренная', low: 'Низкая', unsuitable: 'Не подходит' }
    const statusLabel = statusLabels[cs.status] || cs.status || ''
    entries.push({ label: t('field.cropSuitabilityScore'), value: statusLabel ? `${statusLabel} (${Number(cs.score * 100).toFixed(0)}%)` : `${Number(cs.score * 100).toFixed(0)}%` })
  }
  return entries
})

const predictionTaskLogs = computed(() => store.predictionTaskState?.logs || [])
const scenarioTaskLogs = computed(() => store.scenarioTaskState?.logs || [])
const scenarioOperationalTierLabel = computed(() => {
  const tier = store.modelingResult?.operational_tier
  return tier ? t(`field.operationalTier.${tier}`) : '—'
})
const scenarioReviewRequiredLabel = computed(() => {
  return store.modelingResult?.review_required ? t('field.yes') : t('field.no')
})
const scenarioReviewReason = computed(() => {
  return formatReasonText(
    store.modelingResult?.review_reason_code,
    store.modelingResult?.review_reason,
    store.modelingResult?.review_reason_params,
  ) || ''
})

const riskLevelClass = computed(() => {
  const level = String(store.modelingResult?.risk_summary?.level_code || store.modelingResult?.risk_summary?.level || '').toLowerCase()
  if (level === 'low' || level === 'низкий') return 'risk-low'
  if (level === 'moderate' || level === 'умеренный') return 'risk-moderate'
  if (level === 'elevated' || level === 'повышенный') return 'risk-elevated'
  if (level === 'high' || level === 'высокий') return 'risk-high'
  return ''
})

const archiveView = computed(() => store.selectedArchiveView?.snapshot || null)
const archivePrediction = computed(() => archiveView.value?.prediction_snapshot?.estimated_yield_kg_ha)
const archiveConfidence = computed(() => archiveView.value?.prediction_snapshot?.confidence)
const archiveMetricsLabel = computed(() => {
  const metrics = archiveView.value?.metrics_snapshot?.current_metrics || {}
  return Object.keys(metrics).length ? Object.keys(metrics).map((key) => getLayerMeta(key).label).join(', ') : '—'
})
const archiveScenarioCount = computed(() => {
  const items = archiveView.value?.scenario_snapshot?.items || []
  return String(items.length)
})

const predictionProgressActive = computed(() => store.predictionTaskProgress > 0 && store.predictionTaskProgress < 100)
const scenarioProgressActive = computed(() => store.scenarioTaskProgress > 0 && store.scenarioTaskProgress < 100)
const temporalProgressActive = computed(() => store.temporalAnalyticsTaskProgress > 0 && store.temporalAnalyticsTaskProgress < 100)
const predictionProgressLabel = computed(() => formatUiProgress(store.predictionTaskProgress))
const scenarioProgressLabel = computed(() => formatUiProgress(store.scenarioTaskProgress))
const temporalProgressLabel = computed(() => formatUiProgress(store.temporalAnalyticsTaskProgress))
const predictionTaskStage = computed(() => getTaskStageLabel(store.predictionTaskState, store.predictionTaskState?.stage_label || 'running'))
const predictionTaskDetail = computed(() => getTaskStageDetail(store.predictionTaskState) || '')
const predictionTaskTiming = computed(() => formatTaskTiming(store.predictionTaskState))
const scenarioTaskStage = computed(() => getTaskStageLabel(store.scenarioTaskState, store.scenarioTaskState?.stage_label || 'running'))
const scenarioTaskDetail = computed(() => getTaskStageDetail(store.scenarioTaskState) || '')
const scenarioTaskTiming = computed(() => formatTaskTiming(store.scenarioTaskState))
const temporalTaskStage = computed(() => getTaskStageLabel(store.temporalAnalyticsTaskState, store.temporalAnalyticsTaskState?.stage_label || 'running'))
const temporalTaskDetail = computed(() => getTaskStageDetail(store.temporalAnalyticsTaskState) || '')
const temporalTaskTiming = computed(() => formatTaskTiming(store.temporalAnalyticsTaskState))
const temporalTaskFailureCode = computed(() => {
  const state = store.temporalAnalyticsTaskState || {}
  const status = String(state.status || state.state || '').toLowerCase()
  if (status !== 'failed') return ''
  return state.stage_detail_code || state.result?.data_status?.message_code || state.result?.data_status?.code || ''
})

const comparisonChart = computed(() => {
  const baseline = Number(store.modelingResult?.baseline_yield_kg_ha)
  const scenario = Number(store.modelingResult?.scenario_yield_kg_ha)
  if (!Number.isFinite(baseline) || !Number.isFinite(scenario)) {
    return null
  }
  const interval = prediction.value?.prediction_interval || {}
  const lower = Number(interval.lower)
  const upper = Number(interval.upper)
  const values = [baseline, scenario]
  if (Number.isFinite(lower)) values.push(lower)
  if (Number.isFinite(upper)) values.push(upper)
  const min = Math.min(...values)
  const max = Math.max(...values)
  const span = Math.max(max - min, 1)
  const toPct = (value) => ((value - min) / span) * 100
  return {
    baselinePct: toPct(baseline),
    scenarioPct: toPct(scenario),
    intervalStartPct: Number.isFinite(lower) ? toPct(lower) : 0,
    intervalWidth: Number.isFinite(lower) && Number.isFinite(upper) ? Math.max(0, toPct(upper) - toPct(lower)) : 0,
    intervalLabel:
      Number.isFinite(lower) && Number.isFinite(upper)
        ? `${formatYield(lower)} … ${formatYield(upper)}`
        : '',
  }
})

const seasonalMetricSeries = computed(() => {
  if (!selectedSeasonalMetric.value?.points?.length) return []
  return [
    {
      label: selectedSeasonalMetric.value.label || getFeatureLabel(selectedSeasonalMetric.value.metric, { expertMode: store.expertMode }),
      color: resolveMetricColor(selectedSeasonalMetric.value.metric),
      points: selectedSeasonalMetric.value.points.map((point) => ({
        date: point.observed_at,
        value: point.value,
      })),
    },
  ]
})

const selectedSeasonalMetricLabel = computed(() => {
  return selectedSeasonalMetric.value?.label || getFeatureLabel(selectedSeasonalMetricId.value, { expertMode: store.expertMode }) || '—'
})

const seasonalMetricPointCountLabel = computed(() => {
  const count = selectedSeasonalMetric.value?.points?.length || 0
  if (!count) return t('field.noSeries')
  return locale.value === 'ru'
    ? `${count} ${count === 1 ? 'точка' : count < 5 ? 'точки' : 'точек'}`
    : `${count} point${count === 1 ? '' : 's'}`
})

function formatDateShort(value) {
  if (!value) return ''
  return formatUiDateTime(value, {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
  })
}

function formatTemporalRangeLabel(range) {
  if (!range?.date_from && !range?.date_to) return ''
  const from = formatDateShort(range?.date_from)
  const to = formatDateShort(range?.date_to)
  if (from && to) return `${from} - ${to}`
  return from || to
}

function resolveTemporalStatusMessage(status, options = {}) {
  const failureCode = options.failureCode || ''
  const statusCode = status?.message_code || status?.code || ''
  const messageCode = ['temporal_ready', 'ready'].includes(statusCode) ? statusCode : (failureCode || statusCode)
  if (!messageCode || messageCode === 'temporal_ready' || messageCode === 'ready') {
    return ''
  }
  const range = formatTemporalRangeLabel(status?.requested_range || status?.actual_range)
  const suffix = range
    ? (locale.value === 'ru' ? ` Диапазон: ${range}.` : ` Range: ${range}.`)
    : ''
  if (messageCode === 'temporal_backfill_required' || messageCode === 'backfill_required') {
    return `${t('field.temporalBackfillRequired')}${suffix}`
  }
  if (messageCode === 'temporal_range_exceeds_limit' || messageCode === 'range_exceeds_limit') {
    return t('field.temporalRangeExceedsLimit')
  }
  if (messageCode === 'temporal_historical_data_sparse' || messageCode === 'historical_data_sparse') {
    return `${t('field.temporalHistoricalDataSparse')}${suffix}`
  }
  if (messageCode === 'temporal_insufficient_points_current_season' || messageCode === 'insufficient_points_current_season') {
    return t('field.temporalInsufficientPointsCurrentSeason')
  }
  if (messageCode === 'temporal_no_history_available' || messageCode === 'no_history_available') {
    return `${t('field.temporalNoHistoryAvailable')}${suffix}`
  }
  if (messageCode === 'source_unavailable_quota') {
    return t('field.temporalSourceUnavailableQuota')
  }
  if (messageCode === 'backfill_delayed') {
    return t('field.temporalBackfillDelayed')
  }
  return status?.message || status?.detail || ''
}

const metricsDataStatusMessage = computed(() => {
  return resolveTemporalStatusMessage(metricsDataStatus.value, {
    failureCode: temporalTaskFailureCode.value,
  })
})

const forecastDataStatusMessage = computed(() => {
  return resolveTemporalStatusMessage(forecastDataStatus.value, {
    failureCode: temporalTaskFailureCode.value,
  })
})

const seasonalMetricChartEmptyText = computed(() => {
  if (metricsDataStatusMessage.value) {
    return metricsDataStatusMessage.value
  }
  const pointCount = selectedSeasonalMetric.value?.points?.length || 0
  if (pointCount > 0 && pointCount < 2) {
    return t('field.temporalSinglePoint')
  }
  return t('field.noSeries')
})

const seasonalMetricTimeline = computed(() => {
  return (selectedSeasonalMetric.value?.points || []).map((point) => ({
    observed_at: point.observed_at,
    value: formatMetricValue(selectedSeasonalMetric.value?.metric, point.value),
  }))
})

const selectedMetricAnomalies = computed(() => {
  const metricId = selectedSeasonalMetricId.value
  return metricsAnomalyItems.value.filter((item) => {
    if (!item?.metric) return metricId === 'ndvi'
    return String(item.metric).toLowerCase() === String(metricId).toLowerCase()
  })
})

const seasonalAnomalyRows = computed(() => {
  return metricsAnomalyItems.value.map((item, index) => ({
    key: `${item.kind || item.label || 'anomaly'}-${item.observed_at || index}`,
    label: formatAnomalyLabel(item),
    severity: formatAnomalySeverity(item.severity),
    date: formatUiDateTime(item.observed_at, {
      day: '2-digit',
      month: '2-digit',
      year: 'numeric',
    }),
    metric: getFeatureLabel(item.metric || 'ndvi', { expertMode: store.expertMode }),
    reason: formatAnomalyReason(item),
  }))
})

const historyTrendChartSeries = computed(() => {
  const points = historyTrend.value?.points || []
  const observed = points.map((point) => ({
    date: point.year,
    value: point.observed_yield_kg_ha,
  }))
  const rolling = points.map((point) => ({
    date: point.year,
    value: point.rolling_mean_kg_ha,
  }))
  const series = []
  if (observed.length) {
    series.push({ label: t('field.historyObserved'), color: '#2f6b97', points: observed })
  }
  if (rolling.length) {
    series.push({ label: t('field.historyRollingMean'), color: '#8a6b18', points: rolling })
  }
  return series
})

const historyTrendHasSeries = computed(() => historyTrendChartSeries.value.some((series) => (series.points || []).length >= 2))

const forecastTrendMarkers = computed(() => {
  if (!historyTrendHasSeries.value) return []
  const markers = []
  const currentYear = new Date().getUTCFullYear()
  if (prediction.value?.estimated_yield_kg_ha !== null && prediction.value?.estimated_yield_kg_ha !== undefined) {
    markers.push({
      label: t('field.currentForecastMarker'),
      date: currentYear,
      value: prediction.value.estimated_yield_kg_ha,
      color: '#3c9c64',
    })
  }
  if (store.modelingResult?.scenario_yield_kg_ha !== null && store.modelingResult?.scenario_yield_kg_ha !== undefined) {
    markers.push({
      label: t('field.currentScenarioMarker'),
      date: currentYear,
      value: store.modelingResult.scenario_yield_kg_ha,
      color: '#c97e27',
    })
  }
  return markers
})

const forecastTrendRanges = computed(() => {
  if (!historyTrendHasSeries.value) return []
  const interval = prediction.value?.prediction_interval || {}
  if (!Number.isFinite(Number(interval.lower)) || !Number.isFinite(Number(interval.upper))) {
    return []
  }
  return [
    {
      date: new Date().getUTCFullYear(),
      lower: Number(interval.lower),
      upper: Number(interval.upper),
      color: '#7b91c8',
    },
  ]
})

const scenarioComparisonMetric = ref('ndvi')
const forecastCurveMetric = ref('gdd_cumulative')
const scenarioForecastCurveMetric = ref('gdd_cumulative')

const forecastCurveMetricOptions = computed(() => [
  { value: 'temperature_mean_c', label: t('field.temperatureMean') },
  { value: 'precipitation_mm', label: t('field.precipitationMetric') },
  { value: 'gdd_daily', label: t('field.gddDaily') },
  { value: 'gdd_cumulative', label: t('field.gddCumulative') },
])

function mapForecastCurvePoints(points, metricId) {
  return (points || [])
    .map((point) => ({
      date: point?.date,
      value: point?.[metricId],
    }))
    .filter((point) => point.date && point.value !== null && point.value !== undefined && !Number.isNaN(Number(point.value)))
}

const predictionForecastCurveSeries = computed(() => {
  const metricId = forecastCurveMetric.value
  const points = mapForecastCurvePoints(prediction.value?.forecast_curve?.points || [], metricId)
  if (!points.length) return []
  return [
    {
      label: forecastCurveMetricOptions.value.find((item) => item.value === metricId)?.label || metricId,
      color: resolveMetricColor(metricId),
      points,
    },
  ]
})

const predictionForecastCurveEmptyText = computed(() => {
  const curve = prediction.value?.forecast_curve || {}
  return curve.error || t('field.futureForecastUnavailable')
})

const scenarioForecastCurveSeries = computed(() => {
  const metricId = scenarioForecastCurveMetric.value
  const curve = store.modelingResult?.forecast_curve || {}
  const baselinePoints = mapForecastCurvePoints(curve.baseline_points || [], metricId)
  const scenarioPoints = mapForecastCurvePoints(curve.scenario_points || [], metricId)
  const identical = _seriesAreIdentical(baselinePoints, scenarioPoints)
  const series = []
  if (baselinePoints.length) {
    series.push({
      label: t('field.baselineLine'),
      color: '#2f6b97',
      points: baselinePoints,
    })
  }
  if (scenarioPoints.length && !identical) {
    series.push({
      label: t('field.scenarioLine'),
      color: '#c97e27',
      points: scenarioPoints,
    })
  }
  return series
})
const scenarioForecastCurveIdentical = computed(() => {
  const metricId = scenarioForecastCurveMetric.value
  const curve = store.modelingResult?.forecast_curve || {}
  const bp = mapForecastCurvePoints(curve.baseline_points || [], metricId)
  const sp = mapForecastCurvePoints(curve.scenario_points || [], metricId)
  return bp.length > 0 && sp.length > 0 && _seriesAreIdentical(bp, sp)
})

const scenarioForecastCurveEmptyText = computed(() => {
  const curve = store.modelingResult?.forecast_curve || {}
  return curve.error || t('field.futureScenarioForecastUnavailable')
})

// Returns true when baseline and scenario series are nearly identical (all values within 2%)
function _seriesAreIdentical(a, b) {
  if (!a?.length || !b?.length || a.length !== b.length) return false
  return a.every((ap, i) => {
    const bv = Number(b[i]?.value ?? b[i]?.y)
    const av = Number(ap?.value ?? ap?.y)
    if (!Number.isFinite(av) || !Number.isFinite(bv)) return true
    const denominator = Math.max(Math.abs(av), Math.abs(bv), 1e-9)
    return Math.abs(av - bv) / denominator <= 0.02
  })
}
const scenarioSeriesIdentical = computed(() => {
  const sts = store.modelingResult?.scenario_time_series
  if (!sts) return false
  const metricId = scenarioComparisonMetric.value
  const bp = (sts.baseline?.metrics || []).find((m) => m.metric === metricId)?.points || []
  const sp = (sts.scenario?.metrics || []).find((m) => m.metric === metricId)?.points || []
  return _seriesAreIdentical(bp, sp)
})
const scenarioTimeSeriesChart = computed(() => {
  const sts = store.modelingResult?.scenario_time_series
  if (!sts) return []
  const baselineMetrics = sts.baseline?.metrics || []
  const scenarioMetrics = sts.scenario?.metrics || []
  const metricId = scenarioComparisonMetric.value
  const baselineMetric = baselineMetrics.find((m) => m.metric === metricId)
  const scenarioMetric = scenarioMetrics.find((m) => m.metric === metricId)
  const series = []
  if (baselineMetric?.points?.length) {
    series.push({
      label: scenarioSeriesIdentical.value
        ? (t('field.baselineLine') || 'Базовый сценарий')
        : (t('field.baselineLine') || 'Базовый сценарий'),
      color: '#2f6b97',
      points: baselineMetric.points.map((p) => ({ date: p.observed_at, value: p.value })),
    })
  }
  // Only add scenario series if it differs meaningfully from baseline
  if (scenarioMetric?.points?.length && !scenarioSeriesIdentical.value) {
    series.push({
      label: t('field.scenarioLine') || 'Сценарий',
      color: '#c97e27',
      points: scenarioMetric.points.map((p) => ({ date: p.observed_at, value: p.value })),
    })
  }
  return series
})

const forecastSeasonalOverlaySeries = computed(() => {
  const metricMap = new Map((forecastTemporalAnalytics.value?.seasonal_series?.metrics || []).map((item) => [item.metric, item]))
  const series = []
  for (const metricId of ['ndvi', 'ndmi', 'soil_moisture']) {
    const metric = metricMap.get(metricId)
    if (!metric?.points?.length) continue
    series.push({
      label: metric.label || getFeatureLabel(metricId, { expertMode: store.expertMode }),
      color: resolveMetricColor(metricId),
      points: metric.points.map((point) => ({ date: point.observed_at, value: point.value })),
    })
  }
  return series
})

const forecastSeasonalChartEmptyText = computed(() => {
  if (forecastDataStatusMessage.value) {
    return forecastDataStatusMessage.value
  }
  const maxPoints = Math.max(...forecastSeasonalOverlaySeries.value.map((series) => series.points.length), 0)
  if (maxPoints > 0 && maxPoints < 2) {
    return t('field.temporalSinglePoint')
  }
  return t('field.noSeries')
})

const gddCumulativeSeries = computed(() => {
  const metricEntries = temporalAnalytics.value?.seasonal_series?.metrics || []
  const cumMetric = metricEntries.find((item) => item.metric === 'gdd_cumulative')
  if (!cumMetric?.points?.length) return []
  return [{
    label: locale.value === 'ru' ? 'ГСТ накоп., °C·день' : 'Cumul. GDD',
    color: '#c98b24',
    points: cumMetric.points.map((point) => ({ date: point.observed_at, value: point.value })),
  }]
})

const gddCumulativeTotal = computed(() => {
  const series = gddCumulativeSeries.value
  if (!series.length || !series[0].points.length) return '—'
  const last = series[0].points[series[0].points.length - 1]
  return `${Math.round(last.value)} °C·д`
})

const waterBalanceSeries = computed(() => {
  const rows = waterBalance.value?.series || []
  const storage = rows.map((row) => ({ date: row.observed_at, value: row.storage_mm }))
  const deficit = rows.map((row) => ({ date: row.observed_at, value: row.deficit_mm }))
  const series = []
  if (storage.length) series.push({ label: t('field.rootZoneStorage'), color: '#2f8a63', points: storage })
  if (deficit.length) series.push({ label: t('field.rootZoneDeficit'), color: '#b24d2a', points: deficit })
  return series
})

const phenologySummaryEntries = computed(() => {
  const phenology = forecastTemporalAnalytics.value?.phenology || prediction.value?.phenology || {}
  const lagValue = phenology.lag_weeks_vs_norm === null || phenology.lag_weeks_vs_norm === undefined
    ? '—'
    : locale.value === 'ru'
      ? `${Number(phenology.lag_weeks_vs_norm).toFixed(1)} нед.`
      : `${Number(phenology.lag_weeks_vs_norm).toFixed(1)} wk`
  return [
    { label: t('field.currentStage'), value: phenology.stage_label || '—' },
    { label: t('field.stageLag'), value: lagValue },
    { label: t('field.peakDate'), value: phenology.peak_date || '—' },
    { label: t('field.seasonAmplitude'), value: phenology.seasonal_amplitude === null || phenology.seasonal_amplitude === undefined ? '—' : Number(phenology.seasonal_amplitude).toFixed(3) },
  ].filter((entry) => entry.value !== '—' || entry.label === t('field.currentStage'))
})

const analyticsAlertEntries = computed(() => metricsAnomalyItems.value.slice(0, 6))
const managementZoneRows = computed(() => managementZones.value?.zones || [])
const managementZonesSupported = computed(() => Boolean(managementZoneSummary.value?.supported))
const managementZoneModeLabel = computed(() => {
  const mode = managementZoneSummary.value?.mode
  if (mode === 'yield') return t('field.zoneModeYield')
  if (mode === 'yield_potential') return t('field.zoneModePotential')
  return '—'
})
const modelFoundationEntries = computed(() => {
  return [
    { label: t('field.modelHeads'), value: Array.isArray(analyticsSummary.value?.heads) ? analyticsSummary.value.heads.join(', ') : 'extent, boundary, distance' },
    { label: t('field.modelHeadCount'), value: String(analyticsSummary.value?.head_count || 3) },
    { label: t('field.ttaStandard'), value: analyticsSummary.value?.tta_standard || 'flip2' },
    { label: t('field.ttaQuality'), value: analyticsSummary.value?.tta_quality || 'rotate4' },
  ]
})
const retrainDescription = computed(() => analyticsSummary.value?.retrain_description || t('field.retrainDescription'))
const geometryDiagnosticsEntries = computed(() => {
  const field = dashboard.value?.field || store.selectedField || {}
  const runtime = debugRuntime.value || {}
  return [
    { label: t('field.geometryConfidence'), value: formatPercent(field.geometry_confidence) },
    { label: t('field.ttaConsensus'), value: formatPercent(field.tta_consensus) },
    { label: t('field.boundaryUncertainty'), value: formatPercent(field.boundary_uncertainty) },
    { label: t('field.componentsAfterGrow'), value: formatInteger(runtime.components_after_grow) },
    { label: t('field.componentsAfterGapClose'), value: formatInteger(runtime.components_after_gap_close) },
    { label: t('field.componentsAfterInfill'), value: formatInteger(runtime.components_after_infill) },
    { label: t('field.componentsAfterMerge'), value: formatInteger(runtime.components_after_merge) },
    { label: t('field.componentsAfterWatershed'), value: formatInteger(runtime.components_after_watershed) },
    { label: t('field.splitScoreP50'), value: formatDecimal(runtime.split_score_p50) },
    { label: t('field.splitScoreP90'), value: formatDecimal(runtime.split_score_p90) },
  ]
})
const geometryDebugStatus = computed(() => {
  const runtime = debugRuntime.value || {}
  if (!Object.keys(runtime).length) return ''
  if (runtime.watershed_rollback_reason) {
    return `${t('field.watershedRollback')}: ${runtime.watershed_rollback_reason}`
  }
  if (runtime.watershed_applied) {
    return t('field.watershedApplied')
  }
  if (runtime.watershed_skipped_reason) {
    return `${t('field.watershedSkipped')}: ${runtime.watershed_skipped_reason}`
  }
  return ''
})

function formatMetricValue(metricId, value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return '—'
  }
  const numeric = Number(value)
  if (metricId === 'precipitation') return `${numeric.toFixed(1)} мм`
  if (metricId === 'wind') return `${numeric.toFixed(1)} м/с`
  if (metricId === 'gdd') return numeric.toFixed(0)
  if (metricId === 'soil_moisture') return `${(numeric * 100).toFixed(0)}%`
  if (metricId === 'vpd') return numeric.toFixed(2)
  return numeric.toFixed(3)
}

function formatAnomalySeverity(value) {
  const normalized = String(value || '').toLowerCase()
  if (normalized === 'critical') return locale.value === 'ru' ? 'Критично' : 'Critical'
  if (normalized === 'warning') return locale.value === 'ru' ? 'Предупреждение' : 'Warning'
  if (normalized === 'info') return locale.value === 'ru' ? 'Инфо' : 'Info'
  return value || '—'
}

function formatAnomalyLabel(item) {
  const labels = {
    rapid_canopy_loss: {
      ru: 'Быстрая потеря зелёной массы',
      en: 'Rapid canopy loss',
    },
    possible_drought_stress: {
      ru: 'Возможный стресс засухи',
      en: 'Possible drought stress',
    },
    possible_waterlogging: {
      ru: 'Возможное переувлажнение',
      en: 'Possible waterlogging',
    },
    possible_disease: {
      ru: 'Возможное заболевание или вредитель',
      en: 'Possible disease or pest pressure',
    },
    delayed_development: {
      ru: 'Смещение развития',
      en: 'Delayed development',
    },
  }
  const lang = locale.value === 'en' ? 'en' : 'ru'
  return labels[item?.kind]?.[lang] || item?.label || '—'
}

function formatAnomalyReason(item) {
  const reasons = {
    rapid_canopy_loss: {
      ru: 'NDVI падает быстрее ожидаемой сезонной динамики.',
      en: 'NDVI is dropping faster than the expected seasonal pattern.',
    },
    possible_drought_stress: {
      ru: 'Падение NDVI сопровождается сухим сигналом NDMI и похоже на дефицит влаги.',
      en: 'The NDVI drop is accompanied by a dry NDMI signal and looks like moisture stress.',
    },
    possible_waterlogging: {
      ru: 'Сигнал похож на переувлажнение и кислородный стресс в корневой зоне.',
      en: 'The signal is consistent with waterlogging and root-zone oxygen stress.',
    },
    possible_disease: {
      ru: 'Сигнал отклоняется от нормы и может указывать на болезнь или вредителя.',
      en: 'The signal departs from the norm and may indicate disease or pest pressure.',
    },
    delayed_development: {
      ru: 'Развитие культуры отстаёт от ожидаемой фенологической нормы.',
      en: 'Crop development is lagging behind the expected phenological norm.',
    },
  }
  const lang = locale.value === 'en' ? 'en' : 'ru'
  return reasons[item?.kind]?.[lang] || item?.reason || '—'
}

function resolveMetricColor(metricId) {
  const palette = {
    ndvi: '#2f8a63',
    ndmi: '#1f6aa0',
    ndwi: '#1b87b7',
    bsi: '#b37632',
    gdd: '#c98b24',
    vpd: '#8e4fc6',
    soil_moisture: '#3d8f7f',
    precipitation: '#3f7ee8',
    wind: '#5d6fb3',
  }
  return palette[metricId] || '#21579c'
}

function formatYield(value) {
  return value === null || value === undefined ? '—' : `${Number(value).toFixed(0)} кг/га`
}

function formatPercent(value) {
  return value === null || value === undefined ? '—' : `${(Number(value) * 100).toFixed(0)}%`
}

function formatSigned(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—'
  const numeric = Number(value)
  return `${numeric >= 0 ? '+' : ''}${numeric.toFixed(3)}`
}

function formatDelta(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—'
  const numeric = Number(value)
  return `${numeric >= 0 ? '+' : ''}${numeric.toFixed(2)}%`
}

function formatInteger(value) {
  return Number.isFinite(Number(value)) ? String(Math.round(Number(value))) : '—'
}

function formatDecimal(value) {
  return Number.isFinite(Number(value)) ? Number(value).toFixed(3) : '—'
}

function formatFeatureValue(key, value) {
  return formatDisplayValue(key, value, { expertMode: store.expertMode })
}

function formatDateTime(value) {
  return formatUiDateTime(value, {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

function formatTaskTiming(payload) {
  if (!payload) return '—'
  const elapsed = Number(payload.elapsed_s)
  const eta = Number(payload.estimated_remaining_s)
  const elapsedLabel = Number.isFinite(elapsed)
    ? (locale.value === 'ru' ? `Прошло ${formatSeconds(elapsed)}` : `Elapsed ${formatSeconds(elapsed)}`)
    : null
  const etaLabel = Number.isFinite(eta) ? `ETA ${formatSeconds(Math.max(0, eta))}` : null
  return [elapsedLabel, etaLabel].filter(Boolean).join(' · ') || '—'
}

function formatSeconds(value) {
  const total = Math.max(0, Math.round(Number(value) || 0))
  const minutes = Math.floor(total / 60)
  const seconds = total % 60
  if (minutes >= 60) {
    const hours = Math.floor(minutes / 60)
    const restMinutes = minutes % 60
    return locale.value === 'ru' ? `${hours}ч ${restMinutes}м` : `${hours}h ${restMinutes}m`
  }
  if (minutes > 0) {
    return locale.value === 'ru' ? `${minutes}м ${seconds}с` : `${minutes}m ${seconds}s`
  }
  return locale.value === 'ru' ? `${seconds}с` : `${seconds}s`
}

function riskLevelLabel(value) {
  return getRiskLevelLabel(value)
}

function scenarioRiskItemLabel(item) {
  return getRiskItemLabel(item)
}

function scenarioRiskItemReason(item) {
  return getRiskItemReason(item)
}

// ── Management Events ────────────────────────────────────────────────────────

const currentYear = new Date().getFullYear()

const eventForm = ref({
  event_type: '',
  event_date: '',
  season_year: currentYear,
  amount: null,
  unit: '',
})
const editingEventId = ref(null)
const eventFormError = ref('')

const availableSeasonYears = computed(() => {
  const years = new Set()
  for (const ev of store.fieldEvents) {
    if (ev.season_year) years.add(ev.season_year)
  }
  // Always include last 5 years as options
  for (let y = currentYear; y >= currentYear - 4; y--) years.add(y)
  return [...years].sort((a, b) => b - a)
})

function formatEventType(type) {
  if (!type) return '—'
  const labels = t('field.eventsTypeLabels') || {}
  return labels[type.toLowerCase()] || type
}

function formatEventDate(dateStr) {
  if (!dateStr) return '—'
  try {
    return new Date(dateStr).toLocaleDateString(locale.value === 'ru' ? 'ru-RU' : 'en-US', {
      day: '2-digit', month: '2-digit', year: 'numeric',
    })
  } catch {
    return dateStr
  }
}

function resetEventForm() {
  eventForm.value = { event_type: '', event_date: '', season_year: currentYear, amount: null, unit: '' }
  editingEventId.value = null
  eventFormError.value = ''
}

function openEditEvent(event) {
  editingEventId.value = event.id
  const d = event.event_date ? String(event.event_date).slice(0, 10) : ''
  eventForm.value = {
    event_type: event.event_type || '',
    event_date: d,
    season_year: event.season_year || currentYear,
    amount: event.amount ?? null,
    unit: event.unit || '',
  }
  eventFormError.value = ''
}

function cancelEditEvent() {
  resetEventForm()
}

async function submitEventForm() {
  eventFormError.value = ''
  if (!eventForm.value.event_type.trim()) {
    eventFormError.value = t('field.eventsType') + ': обязательное поле'
    return
  }
  if (!eventForm.value.event_date) {
    eventFormError.value = t('field.eventsDate') + ': обязательное поле'
    return
  }
  const fieldId = store.selectedField?.field_id
  if (!fieldId) return
  const payload = {
    event_type: eventForm.value.event_type.trim(),
    event_date: new Date(eventForm.value.event_date).toISOString(),
    season_year: Number(eventForm.value.season_year),
    amount: eventForm.value.amount !== null && eventForm.value.amount !== '' ? Number(eventForm.value.amount) : null,
    unit: eventForm.value.unit.trim() || null,
    payload: {},
  }
  try {
    if (editingEventId.value) {
      await store.updateFieldEvent(fieldId, editingEventId.value, payload)
    } else {
      await store.createFieldEvent(fieldId, payload)
    }
    resetEventForm()
  } catch (err) {
    eventFormError.value = err?.response?.data?.detail || String(err)
  }
}

async function handleDeleteEvent(eventId) {
  if (!window.confirm(t('field.eventsDeleteConfirm'))) return
  const fieldId = store.selectedField?.field_id
  if (!fieldId) return
  try {
    await store.deleteFieldEvent(fieldId, eventId)
  } catch (err) {
    eventFormError.value = err?.response?.data?.detail || String(err)
  }
}

// Load events when switching to the events tab
watch(
  () => store.activeFieldTab,
  (tab) => {
    if (tab === 'events') {
      store.loadFieldEvents()
    }
  },
)
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
