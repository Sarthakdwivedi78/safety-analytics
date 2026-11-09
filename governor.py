"""Illustrative authority and brake-request logic. Does not control hardware."""
from dataclasses import dataclass, replace
from math import isfinite


@dataclass(frozen=True)
class Scenario:
    camera: float | None = .72
    radar: float | None = .83
    lidar: float | None = .77
    camera_healthy: bool = True
    radar_healthy: bool = True
    lidar_healthy: bool = True
    camera_gate: float = .60
    speed: float = 45
    gap: float = 18
    obstacle: bool = True
    previous_request: bool = False
    feedback_delay: float = .20


def evaluate(s):
    def valid(score, healthy, gate):
        return healthy and score is not None and isfinite(score) and gate <= score <= 1
    camera = valid(s.camera, s.camera_healthy, s.camera_gate)
    radar = valid(s.radar, s.radar_healthy, .50)
    lidar = valid(s.lidar, s.lidar_healthy, .50)
    if camera:
        authority = "Normal fusion"
    elif radar:
        authority = "Radar primary"
    elif lidar:
        authority = "Reduced automation"
    else:
        authority = "Minimum-risk maneuver"
    ttc = s.gap/(s.speed/3.6) if s.obstacle and s.speed > 0 else None
    immediate = ttc is not None and ttc <= 3
    retain = s.previous_request and s.obstacle
    request = (immediate or retain) and (camera or radar or lidar)
    delayed = request and s.feedback_delay > .35
    if not (camera or radar or lidar):
        action = "Simulate controlled deceleration"
    elif delayed:
        action = "Flag delayed brake response"
    elif retain:
        action = "Maintain emergency brake"
    elif request:
        action = "Request emergency brake"
    else:
        action = "Monitor gap"
    return {"authority": authority, "action": action, "camera": camera, "radar": radar,
            "lidar": lidar, "ttc": ttc, "request": request, "delayed": delayed}


BASE = Scenario()
PRESETS = {
    "Normal fusion": (BASE, "Normal fusion", "Request emergency brake"),
    "Low camera with obstacle": (replace(BASE, camera=.33), "Radar primary", "Request emergency brake"),
    "No obstacle": (replace(BASE, obstacle=False), "Normal fusion", "Monitor gap"),
    "Late actuator feedback": (replace(BASE, camera=.36, feedback_delay=.65), "Radar primary", "Flag delayed brake response"),
    "Obstacle still present after request": (replace(BASE, camera=.35, previous_request=True, gap=55), "Radar primary", "Maintain emergency brake"),
    "Camera and radar degraded": (replace(BASE, camera=.30, radar_healthy=False), "Reduced automation", "Request emergency brake"),
    "All sensors unavailable": (replace(BASE, camera=None, radar=None, lidar=None), "Minimum-risk maneuver", "Simulate controlled deceleration"),
    "Exact camera threshold": (replace(BASE, camera=BASE.camera_gate), "Normal fusion", "Request emergency brake"),
    "Missing camera reading": (replace(BASE, camera=None), "Radar primary", "Request emergency brake"),
}
STPA = [
    ("UCA-01", "Brake omitted for obstacle", "L1 / H1", "Camera blindness masks the obstacle.",
     "Transfer authority to usable radar when the camera gate fails.", "Low camera with obstacle"),
    ("UCA-02", "Brake issued unnecessarily", "L1, L2 / H2", "A ghost detection is accepted as a real obstacle.",
     "Normal emergency requests require obstacle evidence.", "No obstacle"),
    ("UCA-03", "Brake issued too late", "L1 / H1", "Actuator feedback is delayed after handover.",
     "Request braking at the demand gate and flag late actuator feedback.", "Late actuator feedback"),
    ("UCA-04", "Brake released too soon", "L1 / H3", "A noisy gap estimate clears the original demand.",
     "Maintain the request while the obstacle remains.", "Obstacle still present after request"),
]
