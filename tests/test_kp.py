"""Tests for the fixed definitive GFZ Kp series and frame matching."""

from datetime import datetime, timezone

import numpy as np
import pytest

from icbuilder.kp import DEFAULT_KP_PATH, GFZ_KP_SHA256
from icbuilder.kp import load_gfz_kp, match_gfz_kp


def test_bundled_kp_data_are_complete_and_definitive():
    series = load_gfz_kp()

    assert len(series["time"]) == 5848
    assert series["time"][0] == np.datetime64("2000-01-01T00:00:00")
    assert series["time"][-1] == np.datetime64("2001-12-31T21:00:00")
    assert np.all(np.diff(series["time"]) == np.timedelta64(3, "h"))
    assert np.all(series["status"] == "def")
    assert np.all((series["kp"] >= 0) & (series["kp"] <= 9))
    np.testing.assert_allclose(
        series["kp"] * 3,
        np.round(series["kp"] * 3),
        atol=0.0011,
        rtol=0,
    )
    assert series["provenance"]["doi"] == "10.5880/Kp.0001"
    assert series["provenance"]["licence"] == "CC BY 4.0"
    assert series["provenance"]["sha256"] == GFZ_KP_SHA256


def test_modified_bundled_response_fails_checksum_validation(tmp_path):
    modified = tmp_path / "modified_kp.json"
    modified.write_bytes(DEFAULT_KP_PATH.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="checksum"):
        load_gfz_kp(modified)


def _small_series():
    return {
        "time": np.array(
            [
                "2000-01-01T00:00:00",
                "2000-01-01T03:00:00",
                "2000-01-01T06:00:00",
            ],
            dtype="datetime64[s]",
        ),
        "kp": np.array([1.0, 2.333, 4.667]),
        "status": np.array(["def", "def", "def"]),
    }


def test_matching_uses_half_open_three_hour_intervals():
    frames = [
        datetime(2000, 1, 1, 0, 0),
        datetime(2000, 1, 1, 2, 59, 59),
        datetime(2000, 1, 1, 3, 0),
        datetime(2000, 1, 1, 6, 0),
    ]
    matched = match_gfz_kp(frames, _small_series())

    np.testing.assert_allclose(matched["kp"], [1.0, 1.0, 2.333, 4.667])
    np.testing.assert_array_equal(
        matched["interval_start"],
        np.array(
            [
                "2000-01-01T00:00:00",
                "2000-01-01T00:00:00",
                "2000-01-01T03:00:00",
                "2000-01-01T06:00:00",
            ],
            dtype="datetime64[s]",
        ),
    )


def test_midnight_boundary_and_repeated_frames():
    series = {
        "time": np.array(
            ["2000-01-01T21:00:00", "2000-01-02T00:00:00"],
            dtype="datetime64[s]",
        ),
        "kp": np.array([1.0, 3.0]),
        "status": np.array(["def", "def"]),
    }
    frames = [
        datetime(2000, 1, 2, tzinfo=timezone.utc),
        datetime(2000, 1, 2, tzinfo=timezone.utc),
    ]

    matched = match_gfz_kp(frames, series)
    np.testing.assert_allclose(matched["kp"], [3.0, 3.0])


def test_gap_and_out_of_range_times_fail():
    gap = _small_series()
    gap["time"][1] = np.datetime64("2000-01-01T04:00:00")
    with pytest.raises(ValueError, match="uninterrupted"):
        match_gfz_kp([datetime(2000, 1, 1, 1)], gap)

    with pytest.raises(ValueError, match="no definitive"):
        match_gfz_kp([datetime(1999, 12, 31, 23, 59)], _small_series())
    with pytest.raises(ValueError, match="no definitive"):
        match_gfz_kp([datetime(2000, 1, 1, 9, 0)], _small_series())
