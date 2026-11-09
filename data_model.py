"""Validated inputs and reproducible, explicitly synthetic sensor scenarios."""
from pathlib import Path

import numpy as np
import pandas as pd

ARCHIVE_URL = ("https://static.nhtsa.gov/odi/ffdd/sgo-2021-01/Archive-2021-2025/"
               "SGO-2021-01_Incident_Reports_ADS.csv")
SOURCE_URL = "https://www.nhtsa.gov/laws-regulations/standing-general-order-crash-reporting"
GROUP = "session_id"
LIGHT = "lighting_conditions"
REASON = "automation_disengagement_reason"
CAMERA, RADAR, LIDAR = SENSORS = ["camera_confidence", "radar_confidence", "lidar_confidence"]
LUX = "illumination_lux"
NOISE = "sensor_noise"
LATENCY = "safety_driver_takeover_latency"
SPEED = "vehicle_speed_kmh"
GAP = "following_distance_m"
FAILURE = "emergency_braking_failure"
UNSAFE = "unsafe_takeover"
FEATURES = [*SENSORS, LUX, NOISE, SPEED, GAP]
FIELDS = [GROUP, LIGHT, REASON, *FEATURES, LATENCY, FAILURE, UNSAFE]
UNITS = ["Stable run / trip group", "Documented category", "Documented reason", "0–1", "0–1",
         "0–1", "lux", "0–1", "km/h", "metres", "seconds", "0 / 1", "0 / 1"]


def normalize(frame, mapping=None):
    """Unknowns remain missing. A positive component establishes the composite outcome."""
    if mapping is None:
        mapping = {name: name for name in FIELDS}
        if "Lighting" in frame and LIGHT not in frame:
            mapping[LIGHT] = "Lighting"
    out = pd.DataFrame(index=frame.index)
    invalid = {}
    for name in FIELDS:
        column = mapping.get(name)
        raw = (frame[column].astype("string").str.strip() if column in frame
               else pd.Series(pd.NA, index=frame.index, dtype="string"))
        raw = raw.mask(raw.str.lower().isin(["", "unknown", "n/a", "na", "null", "not reported"]))
        if name in [GROUP, LIGHT, REASON]:
            out[name] = raw
        else:
            value = pd.to_numeric(raw, errors="coerce").astype(float)
            valid = np.isfinite(value) & (value >= 0)
            if name in [*SENSORS, NOISE]:
                valid &= value <= 1
            if name in [FAILURE, UNSAFE]:
                valid &= value.isin([0, 1])
            invalid[name] = int((raw.notna() & ~valid).sum())
            out[name] = value.where(valid)
    out[LIGHT] = out[LIGHT].replace({"Dark - Not Lighted": "Dark-No-Lights"})
    positive = out[[FAILURE, UNSAFE]].eq(1).any(axis=1)
    negative = out[[FAILURE, UNSAFE]].eq(0).all(axis=1)
    out["outcome"] = np.select([positive, negative], [1., 0.], default=np.nan)
    return out, invalid


def dark_rows(data):
    return data[LIGHT].eq("Dark-No-Lights").fillna(False)


def cohort_rows(data):
    return dark_rows(data) & data[REASON].eq("Camera_Blind").fillna(False)


def cohort_available(data):
    return bool((data[LIGHT].notna() & data[REASON].notna()).any())


def official_archive():
    return pd.read_csv(Path(__file__).parent / "data" / "ads-lighting.csv")


def demo(seed=24021):
    rng = np.random.default_rng(seed)
    groups, per_group = 150, 28
    n = groups * per_group
    trip = np.repeat(np.arange(groups), per_group)
    trip_noise = np.repeat(rng.normal(0, .065, groups), per_group)
    dark = rng.random(n) < .64
    lux = np.where(dark, rng.uniform(.2, 5.6, n), rng.uniform(8, 48, n))
    noise = np.clip(rng.beta(2.4, 2.1, n) + trip_noise, .02, .98)
    shared = rng.normal(0, .085, n)
    clip = lambda x: np.clip(x, .01, .99)
    camera = clip(.76 + .0045*lux - .51*noise + .003*lux*noise + shared + rng.normal(0, .09, n))
    radar = clip(.88 - .20*noise + .65*shared + rng.normal(0, .075, n))
    lidar = clip(.84 - .19*noise + .70*shared + rng.normal(0, .08, n))
    speed, gap = rng.uniform(20, 76, n), rng.uniform(6, 48, n)
    blind = dark & ((camera < .51) | (rng.random(n) < .17))
    braking_logit = 3.6 - 8.2*camera - 2.4*radar - 1.1*lidar + .88*noise + .028*speed - .075*gap + 2*trip_noise
    fail = rng.random(n) < 1/(1+np.exp(-braking_logit))
    unsafe = rng.random(n) < .008+.052*(1-camera)**2
    return pd.DataFrame({
        GROUP: [f"sim-{x:03}" for x in trip],
        LIGHT: np.where(dark, "Dark-No-Lights", "Dark - Lighted"),
        REASON: np.where(blind, "Camera_Blind", "Other"),
        CAMERA: camera, RADAR: radar, LIDAR: lidar, LUX: lux, NOISE: noise,
        LATENCY: .40+1.9*(1-camera)+rng.uniform(0, .7, n), SPEED: speed, GAP: gap,
        FAILURE: fail.astype(int), UNSAFE: unsafe.astype(int),
    })
