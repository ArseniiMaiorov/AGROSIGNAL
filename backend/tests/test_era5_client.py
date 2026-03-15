"""Tests for ERA5 client utility functions."""
import zipfile
from pathlib import Path

import numpy as np
import pytest

from providers.era5.client import ERA5Client


class TestKelvinToCelsius:
    def test_freezing_point(self):
        assert ERA5Client.kelvin_to_celsius(273.15) == pytest.approx(0.0)

    def test_boiling_point(self):
        assert ERA5Client.kelvin_to_celsius(373.15) == pytest.approx(100.0)

    def test_negative(self):
        assert ERA5Client.kelvin_to_celsius(253.15) == pytest.approx(-20.0)


class TestComputeGDD:
    def test_normal_gdd(self):
        gdd = ERA5Client.compute_gdd(t_max_c=30.0, t_min_c=20.0, t_base=10.0)
        assert gdd == pytest.approx(15.0)

    def test_below_base(self):
        gdd = ERA5Client.compute_gdd(t_max_c=8.0, t_min_c=2.0, t_base=10.0)
        assert gdd == 0.0

    def test_zero_gdd(self):
        gdd = ERA5Client.compute_gdd(t_max_c=10.0, t_min_c=10.0, t_base=10.0)
        assert gdd == 0.0


class TestComputeVPD:
    def test_positive_vpd(self):
        vpd = ERA5Client.compute_vpd(t_c=25.0, td_c=15.0)
        assert vpd > 0

    def test_saturated(self):
        vpd = ERA5Client.compute_vpd(t_c=20.0, td_c=20.0)
        assert vpd == pytest.approx(0.0, abs=0.01)

    def test_known_value(self):
        # At 25C, es ≈ 3.17 kPa; at 15C dewpoint, ea ≈ 1.70 kPa
        vpd = ERA5Client.compute_vpd(t_c=25.0, td_c=15.0)
        assert 1.0 < vpd < 2.0


class TestWindComputation:
    def test_wind_speed(self):
        speed = ERA5Client.compute_wind_speed(3.0, 4.0)
        assert speed == pytest.approx(5.0)

    def test_zero_wind(self):
        speed = ERA5Client.compute_wind_speed(0.0, 0.0)
        assert speed == 0.0

    def test_wind_direction_east(self):
        direction = ERA5Client.compute_wind_direction(1.0, 0.0)
        assert 260 <= direction <= 280  # approximately west wind (blowing from west)

    def test_wind_direction_north(self):
        direction = ERA5Client.compute_wind_direction(0.0, 1.0)
        assert 170 <= direction <= 190  # south wind


class TestEra5FallbackPolicy:
    def test_skip_httpx_fallback_for_unaccepted_licenses(self):
        skip, reason = ERA5Client._should_skip_httpx_fallback(
            "403 Client Error: Forbidden - required licences not accepted"
        )
        assert skip is True
        assert reason == "licenses_not_accepted"

    def test_skip_httpx_fallback_for_forbidden(self):
        skip, reason = ERA5Client._should_skip_httpx_fallback(
            "403 Client Error: Forbidden for url"
        )
        assert skip is True
        assert reason == "forbidden"

    def test_keep_httpx_fallback_for_transient_errors(self):
        skip, reason = ERA5Client._should_skip_httpx_fallback(
            "Temporary network failure"
        )
        assert skip is False
        assert reason is None


class TestResolveDownloadPayload:
    def test_extracts_netcdf_from_zip_archive(self, tmp_path):
        client = ERA5Client()
        target_path = tmp_path / "era5_payload.nc"
        extract_dir = tmp_path / "extract"
        extract_dir.mkdir()

        with zipfile.ZipFile(target_path, mode="w") as archive:
            archive.writestr("nested/payload.nc", b"fake-netcdf-content")

        resolved_path = client._resolve_download_payload(
            target_path=str(target_path),
            extract_dir=str(extract_dir),
        )

        assert resolved_path != str(target_path)
        assert Path(resolved_path).exists()
        assert Path(resolved_path).read_bytes() == b"fake-netcdf-content"
