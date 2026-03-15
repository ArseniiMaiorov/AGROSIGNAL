"""Tests for API schemas and validation."""
import pytest
from pydantic import ValidationError

from api.schemas import AoiInput, DetectRequest, TimeRange


class TestAoiInput:
    def test_defaults(self):
        aoi = AoiInput()
        assert aoi.lat == pytest.approx(58.689077)
        assert aoi.lon == pytest.approx(29.892103)
        assert aoi.radius_km == 15.0

    def test_valid_custom(self):
        aoi = AoiInput(lat=55.0, lon=30.0, radius_km=10.0)
        assert aoi.lat == 55.0

    def test_valid_bbox(self):
        aoi = AoiInput(type="bbox", bbox=[29.0, 58.0, 30.0, 59.0])
        assert aoi.type == "bbox"
        assert aoi.bbox == [29.0, 58.0, 30.0, 59.0]

    def test_valid_polygon(self):
        aoi = AoiInput(
            type="polygon",
            polygon=[[29.0, 58.0], [30.0, 58.0], [30.0, 59.0], [29.0, 58.0]],
        )
        assert aoi.type == "polygon"
        assert len(aoi.polygon) == 4

    def test_invalid_lat(self):
        with pytest.raises(ValidationError):
            AoiInput(lat=95.0, lon=30.0, radius_km=10.0)

    def test_invalid_lon(self):
        with pytest.raises(ValidationError):
            AoiInput(lat=55.0, lon=200.0, radius_km=10.0)

    def test_radius_too_large(self):
        with pytest.raises(ValidationError):
            AoiInput(lat=55.0, lon=30.0, radius_km=45.0)

    def test_radius_negative(self):
        with pytest.raises(ValidationError):
            AoiInput(lat=55.0, lon=30.0, radius_km=-5.0)

    def test_invalid_bbox_payload(self):
        with pytest.raises(ValidationError):
            AoiInput(type="bbox", bbox=[29.0, 58.0, 29.5])

    def test_invalid_polygon_payload(self):
        with pytest.raises(ValidationError):
            AoiInput(type="polygon", polygon=[[29.0, 58.0], [30.0, 58.0]])


class TestTimeRange:
    def test_defaults(self):
        tr = TimeRange()
        assert tr.start_date.month == 5
        assert tr.end_date.month == 8

    def test_end_before_start(self):
        from datetime import date
        with pytest.raises(ValidationError):
            TimeRange(start_date=date(2025, 8, 1), end_date=date(2025, 5, 1))


class TestDetectRequest:
    def test_defaults(self):
        req = DetectRequest()
        assert req.resolution_m == 10
        assert req.max_cloud_pct == 40

    def test_custom(self):
        req = DetectRequest(
            aoi=AoiInput(lat=59.0, lon=30.0, radius_km=5.0),
            resolution_m=20,
            max_cloud_pct=30,
            min_field_area_ha=0.5,
            config={"AUTO_DETECT_VERSION": 3},
        )
        assert req.aoi.radius_km == 5.0
        assert req.resolution_m == 20
        assert req.config["AUTO_DETECT_VERSION"] == 3

    def test_invalid_resolution(self):
        with pytest.raises(ValidationError):
            DetectRequest(resolution_m=5)

    def test_invalid_seed_mode(self):
        with pytest.raises(ValidationError):
            DetectRequest(seed_mode="invalid")

    def test_valid_seed_modes(self):
        req1 = DetectRequest(seed_mode="auto")
        req2 = DetectRequest(seed_mode="grid")
        req3 = DetectRequest(seed_mode="edges")
        req4 = DetectRequest(seed_mode="distance")
        assert req1.seed_mode == "auto"
        assert req2.seed_mode == "grid"
        assert req3.seed_mode == "edges"
        assert req4.seed_mode == "distance"

    def test_custom_seed_mode_requires_points(self):
        with pytest.raises(ValidationError):
            DetectRequest(seed_mode="custom")

    def test_custom_seed_mode_with_points(self):
        req = DetectRequest(
            seed_mode="custom",
            seed_points=[[29.91, 58.70], [29.93, 58.71]],
        )
        assert req.seed_mode == "custom"
        assert len(req.seed_points or []) == 2

    def test_json_serializable(self):
        req = DetectRequest(config={"FRAMEWORK_SAM_FIELD_DET": True})
        data = req.model_dump(mode="json")
        assert "aoi" in data
        assert "time_range" in data
        assert "config" in data
        assert data["aoi"]["lat"] == pytest.approx(58.689077)
        assert data["config"]["FRAMEWORK_SAM_FIELD_DET"] is True
