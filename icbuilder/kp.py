"""Load definitive GFZ Kp and match IMAGE frames to three-hour intervals."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


DEFAULT_KP_PATH = Path(__file__).parent / "data" / "gfz_kp_2000_2001.json"
GFZ_KP_QUERY = (
    "https://kp.gfz.de/app/json/"
    "?start=2000-01-01T00:00:00Z"
    "&end=2001-12-31T23:59:59Z"
    "&index=Kp&status=def"
)
GFZ_KP_DOI = "10.5880/Kp.0001"
GFZ_KP_ACQUIRED = "2026-07-30"
GFZ_KP_SHA256 = (
    "259cc539ac6578510ea9f54691bbe2f15913a19060db3e390f4490c49226f91e"
)


def _utc_datetime64(values):
    """Convert IMAGE datetimes to timezone-free UTC seconds.

    IMAGE times in the current orbit files are naive datetimes. They are
    interpreted as UTC because the source files do not carry a timezone.
    Timezone-aware inputs are converted to UTC before the timezone is removed.
    """

    converted = []
    for value in np.atleast_1d(values):
        if isinstance(value, datetime):
            if value.tzinfo is not None:
                value = value.astimezone(timezone.utc).replace(tzinfo=None)
            converted.append(np.datetime64(value, "s"))
        else:
            if isinstance(value, (str, np.str_)) and value.endswith("Z"):
                value = value[:-1]
            converted.append(np.datetime64(value, "s"))
    return np.asarray(converted, dtype="datetime64[s]")


def validate_gfz_kp(times, kp, status, require_complete_interval=False):
    """Check Kp arrays before they enter the IMAGE processing pipeline."""

    times = np.asarray(times, dtype="datetime64[s]")
    kp = np.asarray(kp, dtype=float)
    status = np.asarray(status)

    if times.ndim != 1 or kp.ndim != 1 or status.ndim != 1:
        raise ValueError("Kp time, value, and status arrays must be one-dimensional")
    if not (len(times) == len(kp) == len(status)):
        raise ValueError("Kp time, value, and status arrays must have equal length")
    if len(times) == 0:
        raise ValueError("Kp data are empty")
    if np.any(np.isnat(times)):
        raise ValueError("Kp timestamps must be finite")
    if np.any(np.diff(times) != np.timedelta64(3, "h")):
        raise ValueError("Kp data must have uninterrupted three-hour spacing")
    if np.any(status != "def"):
        raise ValueError("only definitive GFZ Kp values are accepted")
    if np.any(~np.isfinite(kp)) or np.any((kp < 0) | (kp > 9)):
        raise ValueError("Kp values must be finite and between 0 and 9")

    # GFZ reports thirds as decimal 0.333 and 0.667 values.
    if not np.allclose(kp * 3, np.round(kp * 3), atol=0.0011, rtol=0):
        raise ValueError("Kp values must lie on the traditional one-third scale")

    if require_complete_interval:
        if len(times) != 5848:
            raise ValueError("the complete 2000-2001 series must have 5,848 values")
        if times[0] != np.datetime64("2000-01-01T00:00:00"):
            raise ValueError("the Kp series must start at 2000-01-01 00:00 UTC")
        if times[-1] != np.datetime64("2001-12-31T21:00:00"):
            raise ValueError("the Kp series must end at 2001-12-31 21:00 UTC")


def validate_gfz_kp_checksum(content):
    """Confirm that the local file is the documented GFZ API response."""

    checksum = hashlib.sha256(content).hexdigest()
    if checksum != GFZ_KP_SHA256:
        raise ValueError(
            "GFZ Kp file checksum does not match the documented source response"
        )


def load_gfz_kp(path=DEFAULT_KP_PATH):
    """Load the fixed, definitive GFZ Kp series used by the IMAGE pipeline."""

    path = Path(path)
    content = path.read_bytes()
    validate_gfz_kp_checksum(content)
    source = json.loads(content)

    times = _utc_datetime64(source["datetime"])
    kp = np.asarray(source["Kp"], dtype=float)
    status = np.asarray(source["status"])
    validate_gfz_kp(times, kp, status, require_complete_interval=True)

    meta = source.get("meta", {})
    if meta.get("source") != "GFZ Potsdam":
        raise ValueError("unexpected Kp source")
    if meta.get("license") != "CC BY 4.0":
        raise ValueError("unexpected Kp licence")

    return {
        "time": times,
        "kp": kp,
        "status": status,
        "provenance": {
            "source": meta["source"],
            "status": "def",
            "doi": GFZ_KP_DOI,
            "licence": meta["license"],
            "query": GFZ_KP_QUERY,
            "acquired": GFZ_KP_ACQUIRED,
            "sha256": GFZ_KP_SHA256,
            "image_time_interpretation": (
                "Naive IMAGE frame datetimes are interpreted as UTC."
            ),
        },
    }


def match_gfz_kp(frame_times, kp_series):
    """Match frames to enclosing half-open GFZ intervals [start, start + 3 h)."""

    frame_times = _utc_datetime64(frame_times)
    kp_times = np.asarray(kp_series["time"], dtype="datetime64[s]")
    kp = np.asarray(kp_series["kp"], dtype=float)
    status = np.asarray(kp_series["status"])
    validate_gfz_kp(kp_times, kp, status)

    indices = np.searchsorted(kp_times, frame_times, side="right") - 1
    outside = (indices < 0) | (indices >= len(kp_times))
    safe_indices = np.clip(indices, 0, len(kp_times) - 1)
    outside |= frame_times >= kp_times[safe_indices] + np.timedelta64(3, "h")
    if np.any(outside):
        bad_time = frame_times[np.flatnonzero(outside)[0]]
        raise ValueError(f"no definitive GFZ Kp interval contains {bad_time}")

    return {
        "kp": kp[indices],
        "interval_start": kp_times[indices],
    }
