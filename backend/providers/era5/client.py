import asyncio
import hashlib
import json
import os
import shutil
import tempfile
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from core.config import get_settings
from core.logging import get_logger

logger = get_logger(__name__)


class ERA5Client:
    VARIABLES = {
        "temperature_2m": "2m_temperature",
        "dewpoint_2m": "2m_dewpoint_temperature",
        "u_wind_10m": "10m_u_component_of_wind",
        "v_wind_10m": "10m_v_component_of_wind",
        "total_precipitation": "total_precipitation",
        "soil_water_l1": "volumetric_soil_water_layer_1",
        "total_cloud_cover": "total_cloud_cover",
    }

    def __init__(self):
        self.settings = get_settings()

    @staticmethod
    def _empty_result(variables: list[str]) -> dict[str, Any]:
        return {v: [] for v in variables}

    @staticmethod
    def _should_skip_httpx_fallback(error: Exception | str) -> tuple[bool, str | None]:
        message = str(error).lower()
        if "required licences not accepted" in message or "required licenses not accepted" in message:
            return True, "licenses_not_accepted"
        if "accept the required licence" in message or "accept the required license" in message:
            return True, "licenses_not_accepted"
        if "unknown file format" in message:
            return True, "unsupported_payload_format"
        if "403 client error: forbidden" in message:
            return True, "forbidden"
        if "401 client error" in message or "unauthorized" in message:
            return True, "unauthorized"
        return False, None

    def _cache_path(self, lat: float, lon: float, variables: list[str], date_from: date, date_to: date) -> Path:
        payload = json.dumps({
            "lat": round(lat, 3), "lon": round(lon, 3),
            "variables": sorted(variables),
            "from": str(date_from), "to": str(date_to),
        }, sort_keys=True)
        key = hashlib.sha256(payload.encode()).hexdigest()
        cache_dir = Path(self.settings.ERA5_CACHE_DIR)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{key}.json"

    def _read_cache(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        age_hours = (time.time() - path.stat().st_mtime) / 3600
        if age_hours > self.settings.ERA5_CACHE_TTL_HOURS:
            path.unlink(missing_ok=True)
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _write_cache(self, path: Path, data: dict[str, Any]) -> None:
        try:
            path.write_text(json.dumps(data), encoding="utf-8")
        except Exception:
            pass

    async def get_timeseries(
        self,
        lat: float,
        lon: float,
        variables: list[str],
        date_from: date,
        date_to: date,
    ) -> dict[str, Any]:
        """Fetch ERA5 time series for a point.

        Preferred path: official `cdsapi` + `~/.cdsapirc` or `ERA5_CDS_KEY`.
        Legacy HTTP fallback is optional because the public CDS endpoint changed
        and old `/v1/retrieve` flows now frequently return 404.
        """
        logger.info(
            "era5_fetch",
            lat=lat,
            lon=lon,
            variables=variables,
            date_from=str(date_from),
            date_to=str(date_to),
        )

        if not self.settings.ERA5_CDS_KEY and not (Path.home() / ".cdsapirc").exists():
            logger.warning("era5_credentials_missing", detail="Set ERA5_CDS_KEY or ~/.cdsapirc")
            return self._empty_result(variables)

        cache_file = self._cache_path(lat, lon, variables, date_from, date_to)
        cached = self._read_cache(cache_file)
        if cached is not None:
            logger.info("era5_cache_hit", lat=lat, lon=lon)
            return cached

        cds_vars = [self.VARIABLES.get(v, v) for v in variables]
        request = {
            "product_type": "reanalysis",
            "variable": cds_vars,
            "year": sorted(set(str(y) for y in range(date_from.year, date_to.year + 1))),
            "month": sorted(set(f"{d.month:02d}" for d in _date_range(date_from, date_to))),
            "day": sorted(set(f"{d.day:02d}" for d in _date_range(date_from, date_to))),
            "time": ["12:00"],
            "area": [lat + 0.125, lon - 0.125, lat - 0.125, lon + 0.125],
            # CDS API v2 keys
            "data_format": "netcdf",
            "download_format": "unarchived",
        }

        t0 = time.time()
        _cds_exc: Exception | None = None
        try:
            timeout_s = max(5.0, float(getattr(self.settings, "ERA5_CDS_TIMEOUT_S", 45.0)))
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    self._retrieve_and_parse_netcdf,
                    cds_vars=cds_vars,
                    variables=variables,
                    request=request,
                ),
                timeout=timeout_s,
            )
            logger.info("era5_response", status=200, latency_s=round(time.time() - t0, 2), mode="cdsapi")
            self._write_cache(cache_file, result)
            return result
        except asyncio.TimeoutError:
            logger.warning(
                "era5_cdsapi_timeout",
                latency_s=round(time.time() - t0, 1),
                detail=f"CDS API request exceeded {round(timeout_s, 1)} s",
            )
            _cds_exc = TimeoutError(f"CDS API timeout after {round(timeout_s, 1)} s")
        except Exception as _e:
            _cds_exc = _e

        assert _cds_exc is not None
        skip_fallback, reason = self._should_skip_httpx_fallback(_cds_exc)
        fallback_enabled = bool(getattr(self.settings, "ERA5_HTTP_FALLBACK_ENABLED", False))
        logger.warning(
            "era5_cdsapi_failed",
            error=str(_cds_exc),
            fallback="httpx" if (fallback_enabled and not skip_fallback) else None,
            reason=reason,
        )
        if skip_fallback or not fallback_enabled:
            if reason == "licenses_not_accepted":
                logger.warning(
                    "era5_licenses_not_accepted",
                    detail=(
                        "Accept the ERA5 dataset licence at "
                        "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download#manage-licences"
                    ),
                )
            elif not fallback_enabled:
                logger.info("era5_httpx_fallback_disabled")
            return self._empty_result(variables)
        result = await self._fallback_httpx_timeseries(
            lat=lat,
            lon=lon,
            variables=variables,
            cds_vars=cds_vars,
            date_from=date_from,
            date_to=date_to,
        )
        logger.info("era5_response", status=200, latency_s=round(time.time() - t0, 2), mode="httpx_fallback")
        self._write_cache(cache_file, result)
        return result

    async def _fallback_httpx_timeseries(
        self,
        *,
        lat: float,
        lon: float,
        variables: list[str],
        cds_vars: list[str],
        date_from: date,
        date_to: date,
    ) -> dict[str, Any]:
        body = {
            "dataset_short_name": "reanalysis-era5-single-levels",
            "product_type": "reanalysis",
            "variable": cds_vars,
            "year": sorted(set(str(y) for y in range(date_from.year, date_to.year + 1))),
            "month": sorted(set(f"{d.month:02d}" for d in _date_range(date_from, date_to))),
            "day": sorted(set(f"{d.day:02d}" for d in _date_range(date_from, date_to))),
            "time": ["12:00"],
            "area": [lat + 0.125, lon - 0.125, lat - 0.125, lon + 0.125],
            "format": "json",
        }

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.settings.ERA5_CDS_URL}/v1/retrieve",
                json=body,
                headers={"PRIVATE-TOKEN": self.settings.ERA5_CDS_KEY} if self.settings.ERA5_CDS_KEY else {},
                timeout=max(5.0, min(float(getattr(self.settings, "ERA5_CDS_TIMEOUT_S", 45.0)), 60.0)),
            )

        if resp.status_code != 200:
            logger.error("era5_error", status=resp.status_code, body=resp.text[:500])
            return self._empty_result(variables)

        payload = resp.json()
        if isinstance(payload, dict):
            return payload
        return self._empty_result(variables)

    def _retrieve_and_parse_netcdf(
        self,
        *,
        cds_vars: list[str],
        variables: list[str],
        request: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            import cdsapi  # type: ignore
            from netCDF4 import Dataset, num2date  # type: ignore
        except Exception as exc:
            raise RuntimeError("cdsapi/netCDF4 are not installed") from exc

        client_kwargs: dict[str, Any] = {"quiet": True, "progress": False}
        if self.settings.ERA5_CDS_URL:
            client_kwargs["url"] = self.settings.ERA5_CDS_URL
        if self.settings.ERA5_CDS_KEY:
            client_kwargs["key"] = self.settings.ERA5_CDS_KEY

        client = cdsapi.Client(**client_kwargs)

        fd, target_path = tempfile.mkstemp(prefix="era5_", suffix=".nc")
        os.close(fd)
        extract_dir = tempfile.mkdtemp(prefix="era5_extract_")
        try:
            client.retrieve(
                "reanalysis-era5-single-levels",
                request,
                target_path,
            )
            resolved_path = self._resolve_download_payload(
                target_path=target_path,
                extract_dir=extract_dir,
            )

            return self._parse_netcdf_to_timeseries(
                target_path=resolved_path,
                cds_vars=cds_vars,
                variables=variables,
                Dataset=Dataset,
                num2date=num2date,
            )
        finally:
            try:
                os.remove(target_path)
            except OSError:
                pass
            shutil.rmtree(extract_dir, ignore_errors=True)

    def _resolve_download_payload(self, *, target_path: str, extract_dir: str) -> str:
        if not zipfile.is_zipfile(target_path):
            return target_path

        with zipfile.ZipFile(target_path) as archive:
            members = [name for name in archive.namelist() if not name.endswith("/")]
            preferred = next(
                (
                    name for name in members
                    if name.lower().endswith((".nc", ".nc4", ".cdf"))
                ),
                None,
            )
            if preferred is None:
                raise RuntimeError(
                    f"Unsupported ERA5 archive payload: {members[:5]}"
                )
            payload = archive.read(preferred)

        output_name = Path(preferred).name or "era5_payload.nc"
        output_path = Path(extract_dir) / output_name
        output_path.write_bytes(payload)
        return str(output_path)

    def _parse_netcdf_to_timeseries(
        self,
        *,
        target_path: str,
        cds_vars: list[str],
        variables: list[str],
        Dataset,
        num2date,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {v: [] for v in variables}
        reverse_map = {self.VARIABLES.get(v, v): v for v in variables}

        with Dataset(target_path, mode="r") as ds:
            time_var = ds.variables.get("time") or ds.variables.get("valid_time")
            if time_var is None:
                return result

            calendar = getattr(time_var, "calendar", "standard")
            dates = num2date(time_var[:], units=time_var.units, calendar=calendar)

            for cds_name in cds_vars:
                out_name = reverse_map.get(cds_name, cds_name)
                data_var = ds.variables.get(cds_name)
                if data_var is None:
                    continue

                values = np.asarray(data_var[:], dtype=np.float64)
                if values.ndim == 1:
                    series = values
                else:
                    # ERA5 point request returns (time, lat, lon); average spatial dims.
                    series = values.reshape(values.shape[0], -1).mean(axis=1)

                for dt, val in zip(dates, series, strict=False):
                    result[out_name].append(
                        {
                            "date": dt.isoformat() if hasattr(dt, "isoformat") else str(dt),
                            "value": float(val),
                        }
                    )
        return result

    @staticmethod
    def kelvin_to_celsius(t_k: float) -> float:
        return t_k - 273.15

    @staticmethod
    def compute_gdd(t_max_c: float, t_min_c: float, t_base: float = 10.0) -> float:
        t_mean = (t_max_c + t_min_c) / 2.0
        return max(0.0, t_mean - t_base)

    @staticmethod
    def compute_vpd(t_c: float, td_c: float) -> float:
        es = 0.6108 * np.exp(17.27 * t_c / (t_c + 237.3))
        ea = 0.6108 * np.exp(17.27 * td_c / (td_c + 237.3))
        return max(0.0, es - ea)

    @staticmethod
    def compute_wind_speed(u10: float, v10: float) -> float:
        return float(np.sqrt(u10**2 + v10**2))

    @staticmethod
    def compute_wind_direction(u10: float, v10: float) -> float:
        return float((270 - np.degrees(np.arctan2(v10, u10))) % 360)


def _date_range(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)
