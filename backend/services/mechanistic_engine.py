"""Weekly mechanistic crop model inspired by AquaCrop/WOFOST.

Computes yield through a stage-wise process: canopy → transpiration → biomass → yield.
Each week, the model steps through:
1. GDD accumulation & phenological stage update
2. Root-zone water balance (rainfall + irrigation - ET - drainage)
3. Soil nitrogen availability (inputs - uptake - leaching)
4. Canopy cover update (assimilating satellite NDVI when available)
5. Stress computation (water, heat, VPD, nutrient)
6. Biomass accumulation via radiation use efficiency × stress
7. Yield potential tracking

References:
- AquaCrop (FAO): canopy cover → transpiration → biomass → yield
- WOFOST: dynamic explanatory model, weather → soil → crop → yield
- Monteith (1977): radiation use efficiency
- Allen et al. (1998): FAO-56 ET₀ framework
- Lobell et al. (2011): VPD heat stress on crop yields
"""
from __future__ import annotations

from datetime import date, datetime
from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np

from core.logging import get_logger

logger = get_logger(__name__)


# --- Crop phenology parameters ---
CROP_PARAMS: dict[str, dict[str, Any]] = {
    "wheat": {
        "base_temp_c": 0.0, "gdd_emergence": 120, "gdd_flowering": 1050,
        "gdd_maturity": 1650, "rue_max": 1.35, "harvest_index_max": 0.45,
        "canopy_max": 0.90, "root_depth_max_m": 1.2,
        "heat_crit_c": 32.0, "heat_lethal_c": 38.0,
        "vpd_crit_kpa": 2.5, "vpd_lethal_kpa": 4.5,
        "n_demand_kg_ha": 180, "water_stress_p": 0.55,
    },
    "corn": {
        "base_temp_c": 8.0, "gdd_emergence": 80, "gdd_flowering": 900,
        "gdd_maturity": 1550, "rue_max": 1.70, "harvest_index_max": 0.50,
        "canopy_max": 0.95, "root_depth_max_m": 1.5,
        "heat_crit_c": 35.0, "heat_lethal_c": 42.0,
        "vpd_crit_kpa": 3.0, "vpd_lethal_kpa": 5.0,
        "n_demand_kg_ha": 220, "water_stress_p": 0.50,
    },
    "soy": {
        "base_temp_c": 8.0, "gdd_emergence": 90, "gdd_flowering": 750,
        "gdd_maturity": 1350, "rue_max": 1.15, "harvest_index_max": 0.40,
        "canopy_max": 0.88, "root_depth_max_m": 1.0,
        "heat_crit_c": 33.0, "heat_lethal_c": 40.0,
        "vpd_crit_kpa": 2.2, "vpd_lethal_kpa": 4.0,
        "n_demand_kg_ha": 80, "water_stress_p": 0.50,
        "photoperiod_mode": "short_day", "pp_base_h": 11.0, "pp_critical_h": 14.5,
    },
    "barley": {
        "base_temp_c": 0.0, "gdd_emergence": 100, "gdd_flowering": 900,
        "gdd_maturity": 1400, "rue_max": 1.25, "harvest_index_max": 0.48,
        "canopy_max": 0.85, "root_depth_max_m": 1.0,
        "heat_crit_c": 30.0, "heat_lethal_c": 36.0,
        "vpd_crit_kpa": 2.3, "vpd_lethal_kpa": 4.2,
        "n_demand_kg_ha": 150, "water_stress_p": 0.55,
    },
    "sunflower": {
        "base_temp_c": 6.0, "gdd_emergence": 100, "gdd_flowering": 1100,
        "gdd_maturity": 1800, "rue_max": 1.30, "harvest_index_max": 0.35,
        "canopy_max": 0.92, "root_depth_max_m": 2.0,
        "heat_crit_c": 34.0, "heat_lethal_c": 40.0,
        "vpd_crit_kpa": 2.8, "vpd_lethal_kpa": 5.0,
        "n_demand_kg_ha": 130, "water_stress_p": 0.45,
        "photoperiod_mode": "short_day", "pp_base_h": 11.0, "pp_critical_h": 14.2,
    },
    "rapeseed": {
        "base_temp_c": 0.0, "gdd_emergence": 100, "gdd_flowering": 1000,
        "gdd_maturity": 1550, "rue_max": 1.20, "harvest_index_max": 0.30,
        "canopy_max": 0.88, "root_depth_max_m": 1.3,
        "heat_crit_c": 30.0, "heat_lethal_c": 37.0,
        "vpd_crit_kpa": 2.2, "vpd_lethal_kpa": 4.0,
        "n_demand_kg_ha": 200, "water_stress_p": 0.55,
        "photoperiod_mode": "long_day", "pp_base_h": 10.5, "pp_critical_h": 13.8,
    },
}

_CROP_ALIASES: dict[str, str] = {
    "maize": "corn", "кукуруза": "corn",
    "soybean": "soy", "соя": "soy",
    "ячмень": "barley", "oats": "barley", "овес": "barley",
    "canola": "rapeseed", "рапс": "rapeseed",
    "пшеница": "wheat",
    "подсолнечник": "sunflower",
}

DEFAULT_PARAMS = CROP_PARAMS["wheat"]


def _get_crop_params(crop_code: str) -> dict[str, Any]:
    code = str(crop_code or "").strip().lower()
    code = _CROP_ALIASES.get(code, code)
    return CROP_PARAMS.get(code, DEFAULT_PARAMS)


# --- Phenological stages ---

STAGE_NAMES = {
    0: "pre_emergence",
    1: "emergence_to_vegetative",
    2: "vegetative",
    3: "reproductive",
    4: "grain_fill",
    5: "maturity",
}


def _thermal_progress_to_stage(thermal_progress: float, params: dict) -> int:
    gdd_emerg = float(params["gdd_emergence"])
    gdd_flower = float(params["gdd_flowering"])
    gdd_mat = float(params["gdd_maturity"])
    if thermal_progress < gdd_emerg:
        return 0
    elif thermal_progress < gdd_emerg + (gdd_flower - gdd_emerg) * 0.4:
        return 1
    elif thermal_progress < gdd_flower:
        return 2
    elif thermal_progress < gdd_flower + (gdd_mat - gdd_flower) * 0.5:
        return 3
    elif thermal_progress < gdd_mat:
        return 4
    else:
        return 5


# --- Weekly state ---

@dataclass
class WeeklyState:
    week: int = 0
    stage: int = 0
    stage_name: str = "pre_emergence"
    thermal_progress: float = 0.0
    photoperiod_hours: float = 12.0
    root_depth_m: float = 0.1
    surface_layer_water_mm: float = 15.0
    subsoil_water_mm: float = 65.0
    root_zone_water_mm: float = 80.0
    soil_n_available: float = 0.0
    canopy_cover: float = 0.0
    ndvi_assimilated: float = 0.0
    biomass_proxy: float = 0.0
    storage_biomass_proxy: float = 0.0
    gross_assimilation: float = 0.0
    maintenance_respiration: float = 0.0
    heat_stress: float = 0.0
    vpd_stress: float = 0.0
    water_stress: float = 0.0
    nutrient_stress: float = 0.0
    yield_potential_remaining: float = 1.0
    cumulative_stress: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "week": self.week,
            "stage": self.stage,
            "stage_name": self.stage_name,
            "thermal_progress": round(self.thermal_progress, 1),
            "photoperiod_hours": round(self.photoperiod_hours, 2),
            "root_depth_m": round(self.root_depth_m, 3),
            "surface_layer_water_mm": round(self.surface_layer_water_mm, 1),
            "subsoil_water_mm": round(self.subsoil_water_mm, 1),
            "root_zone_water_mm": round(self.root_zone_water_mm, 1),
            "soil_n_available": round(self.soil_n_available, 2),
            "canopy_cover": round(self.canopy_cover, 4),
            "ndvi_assimilated": round(self.ndvi_assimilated, 4),
            "biomass_proxy": round(self.biomass_proxy, 2),
            "storage_biomass_proxy": round(self.storage_biomass_proxy, 2),
            "gross_assimilation": round(self.gross_assimilation, 2),
            "maintenance_respiration": round(self.maintenance_respiration, 2),
            "heat_stress": round(self.heat_stress, 4),
            "vpd_stress": round(self.vpd_stress, 4),
            "water_stress": round(self.water_stress, 4),
            "nutrient_stress": round(self.nutrient_stress, 4),
            "yield_potential_remaining": round(self.yield_potential_remaining, 4),
            "cumulative_stress": round(self.cumulative_stress, 4),
        }


@dataclass
class WeeklyInput:
    """Input data for one week of the growing season."""
    week: int
    tmean_c: float = 15.0
    tmax_c: float = 22.0
    tmin_c: float = 8.0
    precipitation_mm: float = 10.0
    vpd_kpa: float = 1.0
    solar_radiation_mj: float = 15.0  # MJ/m²/day equivalent
    soil_moisture: float | None = None
    wind_speed_m_s: float = 2.0
    ndvi: float | None = None
    ndre: float | None = None
    ndmi: float | None = None
    irrigation_mm: float = 0.0
    n_applied_kg_ha: float = 0.0
    week_start: date | None = None
    season_year: int | None = None
    previous_crop_code: str | None = None


@dataclass
class MechanisticResult:
    baseline_yield_kg_ha: float
    harvest_index: float
    total_biomass_proxy: float
    trace: list[dict[str, Any]]
    final_state: WeeklyState
    params_used: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_yield_kg_ha": round(self.baseline_yield_kg_ha, 2),
            "harvest_index": round(self.harvest_index, 4),
            "total_biomass_proxy": round(self.total_biomass_proxy, 2),
            "trace": self.trace,
            "weeks_simulated": len(self.trace),
        }


def run_mechanistic_baseline(
    *,
    crop_code: str,
    crop_baseline_kg_ha: float,
    weekly_inputs: list[WeeklyInput],
    soil_ph: float | None = None,
    soil_om_pct: float | None = None,
    soil_n_ppm: float | None = None,
    soil_texture_code: float | None = None,
    field_area_ha: float | None = None,
    compactness: float | None = None,
    latitude: float | None = None,
    previous_crop_code: str | None = None,
) -> MechanisticResult:
    """Run the weekly mechanistic model over a growing season.

    Returns baseline yield estimate and weekly trace for explainability.
    """
    params = _get_crop_params(crop_code)
    state = WeeklyState()
    resolved_previous_crop = (
        previous_crop_code
        or next((inp.previous_crop_code for inp in weekly_inputs if getattr(inp, "previous_crop_code", None)), None)
    )

    # Initialize soil nitrogen from soil profile
    if soil_n_ppm is not None:
        # Convert ppm to approximate kg/ha available (0-30cm layer, bulk density ~1.3)
        state.soil_n_available = float(soil_n_ppm) * 1.3 * 0.3 * 10.0 / 1000.0 * 10000.0
    else:
        state.soil_n_available = 60.0  # default moderate

    # Initialize water and root state
    state.root_depth_m = 0.12
    total_initial_water = _initial_root_zone_water(soil_texture_code)
    surface_capacity = _surface_available_water(soil_texture_code)
    state.surface_layer_water_mm = min(surface_capacity, total_initial_water * 0.20)
    state.subsoil_water_mm = max(total_initial_water - state.surface_layer_water_mm, 0.0)
    state.root_zone_water_mm = state.surface_layer_water_mm + state.subsoil_water_mm

    predecessor = _previous_crop_effect(resolved_previous_crop, crop_code)
    state.soil_n_available += predecessor["n_credit_kg_ha"]
    state.yield_potential_remaining *= predecessor["yield_factor"]

    trace: list[dict[str, Any]] = []

    for wk_input in weekly_inputs:
        state.week = wk_input.week

        # 1. Phenology with temperature and photoperiod
        state.photoperiod_hours = _photoperiod_hours(
            latitude=latitude,
            week=wk_input.week,
            season_year=wk_input.season_year,
            week_start=wk_input.week_start,
        )
        photoperiod_factor = _photoperiod_modifier(params, state.photoperiod_hours)
        gdd = max(0.0, wk_input.tmean_c - float(params["base_temp_c"])) * 7.0 * photoperiod_factor
        state.thermal_progress += gdd

        # 2. Stage update
        state.stage = _thermal_progress_to_stage(state.thermal_progress, params)
        state.stage_name = STAGE_NAMES.get(state.stage, "unknown")

        # 3. Dynamic root growth and two-layer water balance
        current_taw = _total_available_water(state.root_depth_m, soil_texture_code)
        water_fraction = state.root_zone_water_mm / max(current_taw, 1.0)
        state.root_depth_m = _update_root_depth(
            prev_depth_m=state.root_depth_m,
            max_depth_m=float(params["root_depth_max_m"]),
            tmean_c=wk_input.tmean_c,
            stage=state.stage,
            texture_code=soil_texture_code,
            water_fraction=water_fraction,
        )
        et0_daily = _estimate_et0(
            wk_input.tmean_c,
            wk_input.tmax_c,
            wk_input.tmin_c,
            wk_input.solar_radiation_mj,
            latitude,
            wind_speed_m_s=wk_input.wind_speed_m_s,
            vpd_kpa=wk_input.vpd_kpa,
        )
        kc = _crop_coefficient(state.stage, state.canopy_cover)
        et_crop = et0_daily * kc * 7.0  # weekly ET
        water_in = wk_input.precipitation_mm + wk_input.irrigation_mm
        taw = _total_available_water(state.root_depth_m, soil_texture_code)
        surface_capacity = _surface_available_water(soil_texture_code)
        state.surface_layer_water_mm, state.subsoil_water_mm = _update_water_balance(
            surface_mm=state.surface_layer_water_mm,
            subsoil_mm=state.subsoil_water_mm,
            water_in=water_in,
            et_crop=et_crop,
            canopy_cover=state.canopy_cover,
            surface_capacity_mm=surface_capacity,
            root_capacity_mm=max(taw - surface_capacity, taw * 0.5),
            observed_soil_moisture=wk_input.soil_moisture,
        )
        state.root_zone_water_mm = state.surface_layer_water_mm + state.subsoil_water_mm

        # 4. Soil nitrogen
        n_uptake = _nitrogen_uptake(state.stage, state.canopy_cover, float(params["n_demand_kg_ha"]))
        n_leach = _nitrogen_leaching(state.root_zone_water_mm, taw, soil_texture_code)
        state.soil_n_available = max(0.0, state.soil_n_available + wk_input.n_applied_kg_ha - n_uptake - n_leach)

        # 5. Canopy cover update (assimilate NDVI if available)
        state.canopy_cover = _update_canopy(
            prev_cc=state.canopy_cover,
            stage=state.stage,
            gdd_progress=state.thermal_progress / max(float(params["gdd_maturity"]), 1.0),
            sat_ndvi=wk_input.ndvi,
            canopy_max=float(params["canopy_max"]),
            water_stress=state.water_stress,
        )
        if wk_input.ndvi is not None:
            state.ndvi_assimilated = wk_input.ndvi

        # 6. Stress computation
        state.water_stress = _water_stress(
            state.root_zone_water_mm, taw, float(params["water_stress_p"]),
        )
        state.heat_stress = _heat_stress(
            wk_input.tmax_c, float(params["heat_crit_c"]), float(params["heat_lethal_c"]),
            state.stage,
        )
        state.vpd_stress = _vpd_stress(
            wk_input.vpd_kpa, float(params["vpd_crit_kpa"]), float(params["vpd_lethal_kpa"]),
        )
        state.nutrient_stress = _nutrient_stress(
            state.soil_n_available, state.stage, float(params["n_demand_kg_ha"]),
        )

        # 7. Biomass accumulation
        stress_mult = _combine_stresses(
            state.water_stress, state.heat_stress, state.vpd_stress, state.nutrient_stress,
        )
        apar = _absorbed_par(state.canopy_cover, wk_input.solar_radiation_mj * 7.0)
        rue = float(params["rue_max"]) * _rue_temperature_modifier(wk_input.tmean_c, float(params["base_temp_c"]))
        gross_assimilation = apar * rue * stress_mult * 10.0  # g/m² → kg/ha approximation
        maintenance_respiration = _maintenance_respiration(state.biomass_proxy, wk_input.tmean_c, state.stage)
        delta_biomass = max(0.0, gross_assimilation - maintenance_respiration)
        state.gross_assimilation = gross_assimilation
        state.maintenance_respiration = maintenance_respiration
        state.biomass_proxy += delta_biomass
        state.storage_biomass_proxy += delta_biomass * _storage_partition_fraction(state.stage)

        # 8. Yield potential tracking
        # Stress during reproductive/grain-fill permanently reduces yield potential
        if state.stage in (3, 4):
            stress_penalty = 1.0 - stress_mult
            state.yield_potential_remaining *= (1.0 - stress_penalty * 0.15)
        state.cumulative_stress += (1.0 - stress_mult)

        trace.append(state.to_dict())

    # Final yield computation
    hi = _estimate_harvest_index(state, params)
    storage_yield = state.storage_biomass_proxy
    baseline_yield = max(storage_yield, state.biomass_proxy * hi) * state.yield_potential_remaining

    # Scale to match crop baseline (the mechanistic model produces relative yield)
    if crop_baseline_kg_ha > 0 and state.biomass_proxy > 0:
        # Blend mechanistic estimate with baseline prior
        mech_ratio = baseline_yield / max(crop_baseline_kg_ha, 1.0)
        # If mechanistic estimate is reasonable (0.3-1.5×baseline), use it
        # Otherwise, anchor toward baseline
        if 0.3 <= mech_ratio <= 1.5:
            final_yield = baseline_yield
        else:
            # Shrink toward baseline
            final_yield = crop_baseline_kg_ha * (0.6 + 0.4 * float(np.clip(mech_ratio, 0.3, 1.5)))
    else:
        final_yield = baseline_yield

    # Area/shape penalty
    if field_area_ha is not None and field_area_ha < 2.0:
        edge_loss = max(0.0, 1.0 - field_area_ha / 2.0) * 0.10
        final_yield *= (1.0 - edge_loss)
    if compactness is not None and compactness < 0.5:
        shape_loss = max(0.0, 0.5 - compactness) * 0.08
        final_yield *= (1.0 - shape_loss)

    return MechanisticResult(
        baseline_yield_kg_ha=round(max(0.0, final_yield), 2),
        harvest_index=round(hi, 4),
        total_biomass_proxy=round(state.biomass_proxy, 2),
        trace=trace,
        final_state=state,
        params_used={**params, "previous_crop_code": resolved_previous_crop, "feature_model": "wofost_lite_v1"},
    )


# --- Helper functions ---

def _estimate_et0(
    tmean: float,
    tmax: float,
    tmin: float,
    solar_mj: float,
    latitude: float | None,
    *,
    wind_speed_m_s: float = 2.0,
    vpd_kpa: float | None = None,
) -> float:
    """FAO Penman-Monteith inspired ET₀ (mm/day).

    Uses available weather inputs:
    - temperature
    - net-radiation proxy from solar radiation
    - wind speed
    - vapor pressure deficit
    """
    if solar_mj <= 0:
        solar_mj = 14.0
    if vpd_kpa is None or vpd_kpa <= 0:
        vpd_kpa = 0.8
    wind = float(np.clip(wind_speed_m_s or 2.0, 0.1, 12.0))
    tmean = float(tmean)
    tmax = float(max(tmax, tmean))
    tmin = float(min(tmin, tmean))

    es_tmax = 0.6108 * math.exp((17.27 * tmax) / max(tmax + 237.3, 1.0))
    es_tmin = 0.6108 * math.exp((17.27 * tmin) / max(tmin + 237.3, 1.0))
    es = (es_tmax + es_tmin) / 2.0
    ea = max(0.0, es - float(vpd_kpa))
    delta = 4098.0 * (0.6108 * math.exp((17.27 * tmean) / max(tmean + 237.3, 1.0))) / max((tmean + 237.3) ** 2, 1.0)
    gamma = 0.066

    # Net radiation proxy: shortwave net + modest latitude adjustment.
    lat_factor = 1.0
    if latitude is not None:
        lat_factor = float(np.clip(1.05 - abs(float(latitude)) / 180.0, 0.75, 1.05))
    rn = max(0.0, solar_mj * 0.77 * lat_factor)
    numerator = 0.408 * delta * rn + gamma * (900.0 / max(tmean + 273.0, 1.0)) * wind * max(es - ea, 0.0)
    denominator = delta + gamma * (1.0 + 0.34 * wind)
    et0 = numerator / max(denominator, 0.05)
    return float(np.clip(et0, 0.4, 11.5))


def _crop_coefficient(stage: int, canopy_cover: float) -> float:
    """Crop coefficient Kc based on FAO-56."""
    base_kc = {0: 0.3, 1: 0.4, 2: 0.8, 3: 1.15, 4: 1.0, 5: 0.5}
    kc = float(base_kc.get(stage, 0.7))
    # Adjust by actual canopy cover
    kc *= (0.3 + 0.7 * canopy_cover)
    return float(np.clip(kc, 0.15, 1.25))


def _update_root_depth(
    *,
    prev_depth_m: float,
    max_depth_m: float,
    tmean_c: float,
    stage: int,
    texture_code: float | None,
    water_fraction: float,
) -> float:
    """SUCROS-like dynamic root extension."""
    if stage >= 5:
        return float(np.clip(prev_depth_m, 0.08, max_depth_m))
    temp_factor = float(np.clip((tmean_c - 2.0) / 18.0, 0.15, 1.0))
    if texture_code is None:
        soil_strength_factor = 0.9
    else:
        tc = float(texture_code)
        soil_strength_factor = 1.0 if tc <= 3.0 else 0.88 if tc <= 6.0 else 0.75
    moisture_factor = float(np.clip(0.45 + 0.65 * water_fraction, 0.35, 1.0))
    stage_factor = 1.0 if stage <= 2 else 0.6 if stage == 3 else 0.25
    max_rate_m_week = 0.14
    growth = max_rate_m_week * temp_factor * soil_strength_factor * moisture_factor * stage_factor
    growth *= max(0.0, 1.0 - prev_depth_m / max(max_depth_m, 0.1))
    return float(np.clip(prev_depth_m + growth, 0.08, max_depth_m))


def _total_available_water(root_depth_m: float, texture_code: float | None) -> float:
    """Total available water in root zone (mm).

    TAW = (FC - WP) × root_depth × 1000
    Texture affects FC and WP.
    """
    if texture_code is None:
        awc_mm_per_m = 150.0  # default loam
    else:
        tc = float(texture_code)
        if tc <= 1.5:
            awc_mm_per_m = 80.0   # sand
        elif tc <= 3.0:
            awc_mm_per_m = 120.0  # sandy loam
        elif tc <= 5.5:
            awc_mm_per_m = 170.0  # loam/silt loam
        elif tc <= 7.0:
            awc_mm_per_m = 160.0  # clay loam
        else:
            awc_mm_per_m = 130.0  # heavy clay (lower due to strong retention)
    return awc_mm_per_m * root_depth_m


def _surface_available_water(texture_code: float | None) -> float:
    if texture_code is None:
        return 18.0
    tc = float(texture_code)
    if tc <= 2.0:
        return 10.0
    if tc <= 5.5:
        return 18.0
    return 22.0


def _update_water_balance(
    *,
    surface_mm: float,
    subsoil_mm: float,
    water_in: float,
    et_crop: float,
    canopy_cover: float,
    surface_capacity_mm: float,
    root_capacity_mm: float,
    observed_soil_moisture: float | None,
) -> tuple[float, float]:
    """Two-layer FAO-56 inspired bucket model.

    Surface layer handles evaporation and quick infiltration.
    Subsoil/root layer handles transpiration and deeper storage.
    """
    surface = surface_mm + water_in
    infiltrated = max(surface - surface_capacity_mm, 0.0)
    surface = min(surface, surface_capacity_mm)

    subsoil = subsoil_mm + infiltrated
    deep_drainage = max(subsoil - root_capacity_mm, 0.0) * 0.55
    subsoil = min(subsoil, root_capacity_mm) - deep_drainage

    surface_evap_fraction = float(np.clip(0.60 - 0.40 * canopy_cover, 0.18, 0.70))
    surface_et = et_crop * surface_evap_fraction
    transpiration = max(et_crop - surface_et, 0.0)

    surface = max(surface - min(surface, surface_et), 0.0)
    subsoil = max(subsoil - min(subsoil, transpiration), 0.0)

    if observed_soil_moisture is not None:
        moisture = float(np.clip(observed_soil_moisture, 0.0, 1.0))
        target_total = (surface_capacity_mm + root_capacity_mm) * moisture
        total = surface + subsoil
        adjustment = (target_total - total) * 0.12
        if adjustment > 0:
            subsoil = min(root_capacity_mm, subsoil + adjustment)

    floor_surface = surface_capacity_mm * 0.03
    floor_subsoil = root_capacity_mm * 0.05
    return max(surface, floor_surface), max(subsoil, floor_subsoil)


def _nitrogen_uptake(stage: int, canopy_cover: float, n_demand: float) -> float:
    """Weekly N uptake based on stage and canopy cover."""
    # Most uptake during vegetative and early reproductive
    stage_frac = {0: 0.01, 1: 0.05, 2: 0.12, 3: 0.08, 4: 0.03, 5: 0.01}
    weekly_frac = float(stage_frac.get(stage, 0.03))
    return n_demand * weekly_frac * canopy_cover


def _nitrogen_leaching(water_mm: float, taw: float, texture_code: float | None) -> float:
    """N leaching when soil is saturated. Sandy soils leach more."""
    if water_mm < taw * 0.8:
        return 0.0
    excess_frac = (water_mm - taw * 0.8) / max(taw * 0.2, 1.0)
    base_leach = 2.0 * excess_frac
    # Sandy soils leach more
    if texture_code is not None and float(texture_code) <= 2.0:
        base_leach *= 1.5
    return base_leach


def _update_canopy(*, prev_cc: float, stage: int, gdd_progress: float,
                    sat_ndvi: float | None, canopy_max: float,
                    water_stress: float) -> float:
    """Update canopy cover with optional NDVI assimilation."""
    # Model-based canopy trajectory
    if stage == 0:
        model_cc = 0.0
    elif stage <= 2:
        # Logistic growth
        model_cc = canopy_max * (1.0 - np.exp(-4.0 * gdd_progress))
    elif stage <= 4:
        model_cc = canopy_max
    else:
        # Senescence
        model_cc = canopy_max * max(0.2, 1.0 - (gdd_progress - 1.0) * 2.0)

    model_cc = float(np.clip(model_cc, 0.0, canopy_max))

    # Assimilate satellite NDVI if available
    if sat_ndvi is not None and sat_ndvi > 0.1:
        # NDVI ≈ 0.1 + 0.9×CC (rough linear mapping)
        sat_cc = float(np.clip((sat_ndvi - 0.1) / 0.8, 0.0, 1.0)) * canopy_max
        # Blend: 60% satellite, 40% model (satellite is observed truth)
        cc = 0.60 * sat_cc + 0.40 * model_cc
    else:
        cc = model_cc

    # Water stress reduces canopy
    if water_stress > 0.3:
        cc *= (1.0 - (water_stress - 0.3) * 0.3)

    return float(np.clip(cc, 0.0, canopy_max))


def _photoperiod_hours(
    *,
    latitude: float | None,
    week: int,
    season_year: int | None,
    week_start: date | None,
) -> float:
    if latitude is None:
        return 12.0
    if week_start is None:
        resolved_year = int(season_year or datetime.utcnow().year)
        try:
            week_start = date.fromisocalendar(resolved_year, max(min(int(week), 53), 1), 1)
        except ValueError:
            week_start = date(resolved_year, 6, 15)
    day_of_year = int(week_start.timetuple().tm_yday)
    lat_rad = math.radians(float(latitude))
    decl = math.radians(23.45 * math.sin(math.radians((360.0 / 365.0) * (284 + day_of_year))))
    cos_omega = -math.tan(lat_rad) * math.tan(decl)
    cos_omega = float(np.clip(cos_omega, -1.0, 1.0))
    return float((24.0 / math.pi) * math.acos(cos_omega))


def _photoperiod_modifier(params: dict[str, Any], photoperiod_h: float) -> float:
    mode = str(params.get("photoperiod_mode") or "").strip().lower()
    if not mode:
        return 1.0
    base_h = float(params.get("pp_base_h") or 10.0)
    critical_h = float(params.get("pp_critical_h") or base_h)
    if mode == "long_day":
        if photoperiod_h >= critical_h:
            return 1.0
        return float(np.clip((photoperiod_h - base_h) / max(critical_h - base_h, 0.1), 0.35, 1.0))
    if mode == "short_day":
        if photoperiod_h <= critical_h:
            return 1.0
        ceiling_h = max(base_h, critical_h + 3.5)
        return float(np.clip((ceiling_h - photoperiod_h) / max(ceiling_h - critical_h, 0.1), 0.35, 1.0))
    return 1.0


def _water_stress(water_mm: float, taw: float, p: float) -> float:
    """Water stress coefficient Ks (0=no stress, 1=max stress).

    FAO-56: stress begins when soil water drops below p×TAW (readily available water).
    """
    if taw <= 0:
        return 0.0
    raw = taw * p  # Readily available water threshold
    if water_mm >= raw:
        return 0.0
    if water_mm <= taw * 0.05:
        return 1.0  # Below wilting point
    return float(1.0 - (water_mm - taw * 0.05) / max(raw - taw * 0.05, 1.0))


def _heat_stress(tmax: float, t_crit: float, t_lethal: float, stage: int) -> float:
    """Heat stress (0=none, 1=max). Critical during reproductive/grain-fill."""
    if tmax <= t_crit:
        return 0.0
    # Reproductive stages are 2× more sensitive
    sensitivity = 2.0 if stage in (3, 4) else 1.0
    raw = (tmax - t_crit) / max(t_lethal - t_crit, 1.0)
    return float(np.clip(raw * sensitivity, 0.0, 1.0))


def _vpd_stress(vpd: float, vpd_crit: float, vpd_lethal: float) -> float:
    """VPD stress (stomatal closure, reduced transpiration)."""
    if vpd <= vpd_crit:
        return 0.0
    return float(np.clip((vpd - vpd_crit) / max(vpd_lethal - vpd_crit, 0.5), 0.0, 1.0))


def _nutrient_stress(n_available: float, stage: int, n_demand: float) -> float:
    """Nutrient stress based on remaining N vs demand."""
    if stage in (0, 5):
        return 0.0  # N not limiting before emergence or after maturity
    # Remaining demand fraction
    stage_demand_frac = {1: 0.90, 2: 0.70, 3: 0.30, 4: 0.10}
    remaining_demand = n_demand * float(stage_demand_frac.get(stage, 0.5))
    if remaining_demand <= 0:
        return 0.0
    sufficiency = n_available / max(remaining_demand, 1.0)
    if sufficiency >= 1.0:
        return 0.0
    return float(np.clip(1.0 - sufficiency, 0.0, 0.8))


def _combine_stresses(water: float, heat: float, vpd: float, nutrient: float) -> float:
    """Combine stresses using Liebig's Law: worst stress dominates.

    Returns multiplier 0-1 (1 = no stress).
    """
    individual = [
        1.0 - water,
        1.0 - heat,
        1.0 - vpd,
        1.0 - nutrient,
    ]
    # Weighted: 70% worst factor, 30% geometric mean
    worst = min(individual)
    geometric = float(np.prod(np.asarray(individual, dtype=float)) ** (1.0 / len(individual)))
    return float(np.clip(0.70 * worst + 0.30 * geometric, 0.05, 1.0))


def _maintenance_respiration(biomass_proxy: float, tmean_c: float, stage: int) -> float:
    if biomass_proxy <= 0:
        return 0.0
    stage_coeff = {0: 0.002, 1: 0.004, 2: 0.006, 3: 0.007, 4: 0.006, 5: 0.003}
    q10 = 2.0 ** ((tmean_c - 20.0) / 10.0)
    return float(max(0.0, biomass_proxy * float(stage_coeff.get(stage, 0.005)) * q10))


def _storage_partition_fraction(stage: int) -> float:
    fractions = {0: 0.0, 1: 0.0, 2: 0.08, 3: 0.28, 4: 0.55, 5: 0.25}
    return float(fractions.get(stage, 0.10))


def _initial_root_zone_water(texture_code: float | None) -> float:
    if texture_code is None:
        return 80.0
    tc = float(texture_code)
    if tc <= 2.0:
        return 50.0
    if tc <= 5.0:
        return 90.0
    return 110.0


def _previous_crop_effect(previous_crop_code: str | None, crop_code: str) -> dict[str, float]:
    previous = str(previous_crop_code or "").strip().lower()
    current = str(crop_code or "").strip().lower()
    legumes = {"soy", "soybean", "pea", "beans", "горох", "соя"}
    cereals = {"wheat", "barley", "corn", "maize", "пшеница", "ячмень", "кукуруза"}
    oilseeds = {"rapeseed", "canola", "sunflower", "рапс", "подсолнечник"}

    n_credit = 0.0
    yield_factor = 1.0
    if previous in legumes:
        n_credit += 20.0
        if current in cereals | oilseeds:
            yield_factor *= 1.05
    if previous and previous == current:
        yield_factor *= 0.92
    if previous in {"rapeseed", "canola", "рапс"} and current in {"wheat", "barley", "пшеница", "ячмень"}:
        yield_factor *= 1.08
    return {
        "n_credit_kg_ha": n_credit,
        "yield_factor": float(np.clip(yield_factor, 0.80, 1.12)),
    }


def _absorbed_par(canopy_cover: float, solar_mj_week: float) -> float:
    """Absorbed photosynthetically active radiation (MJ/m²/week).

    PAR ≈ 0.48 × total solar radiation
    Absorbed fraction ≈ 1 - exp(-k × LAI), approximated by canopy cover
    """
    par = solar_mj_week * 0.48
    # Beer-Lambert: absorbed = PAR × (1 - exp(-k×LAI))
    # With canopy cover as proxy: absorbed ≈ PAR × CC
    return par * canopy_cover


def _rue_temperature_modifier(tmean: float, base_temp: float) -> float:
    """Temperature modifier for RUE. Optimal around 20-25°C for most crops."""
    if tmean <= base_temp:
        return 0.1
    opt_low = base_temp + 12.0
    opt_high = base_temp + 20.0
    if tmean < opt_low:
        return float(0.3 + 0.7 * (tmean - base_temp) / max(opt_low - base_temp, 1.0))
    elif tmean <= opt_high:
        return 1.0
    else:
        return float(np.clip(1.0 - (tmean - opt_high) * 0.06, 0.3, 1.0))


def _estimate_harvest_index(state: WeeklyState, params: dict) -> float:
    """Harvest index adjusted by reproductive-stage stress.

    HI is reduced by stress during grain fill (Borrás et al. 2004).
    """
    hi_max = float(params["harvest_index_max"])
    # Stress penalty during grain fill reduces HI
    hi = hi_max * state.yield_potential_remaining
    # Extreme stress floors HI
    if state.cumulative_stress > 15.0:
        hi *= 0.70
    elif state.cumulative_stress > 10.0:
        hi *= 0.85
    return float(np.clip(hi, 0.10, hi_max))
