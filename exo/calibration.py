"""The calibration file written by homing (calibration/joint_cal.json).

It records where the two hard stops are in the motor's frame and where mid-stroke (joint angle 0)
is, plus the friction and gravity values identified by homing. The motor's frame shifts by whole
turns at every power-up, so the stop positions are only valid until the motor is switched off:
after a power cycle, home again (the scripts check this).
"""

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import JointConfig
from .logs import ROOT

CAL_FILE = ROOT / "calibration" / "joint_cal.json"


class CalibrationError(RuntimeError):
    """The calibration is missing or does not match the motor; home again."""


@dataclass
class Calibration:
    created: str  # when the stops were found
    method: str  # "auto" or "manual"
    stroke_deg: float  # travel between the stops
    stops_motor_rad: list  # [lower, upper] stop in the motor's frame (this power session only)
    mid_motor_rad: float  # joint angle 0 (mid-stroke) in the motor's frame
    angle_offsets: dict  # the motor's zero-offset registers at homing (a re-saved zero changes them)
    breakaway: dict = field(default_factory=dict)  # auto homing step 1 (static friction, gravity there)
    compensation: dict = field(default_factory=dict)  # identified friction and gravity (exo.compensation)

    @classmethod
    def from_stops(cls, stop_a_rad: float, stop_b_rad: float, method: str, angle_offsets: dict,
                   config: JointConfig) -> "Calibration":
        """Calibration from the two stops (motor frame, either order). Refuses an implausible stroke."""
        lower, upper = sorted((stop_a_rad, stop_b_rad))
        stroke = math.degrees(upper - lower)
        if abs(stroke - config.design_stroke_deg) > config.stroke_tol_deg:
            raise CalibrationError(f"stroke {stroke:.1f} deg is more than {config.stroke_tol_deg:g} deg off the "
                                   f"design value {config.design_stroke_deg:g} deg; not saved")
        return cls(created=datetime.now().astimezone().isoformat(timespec="seconds"), method=method,
                   stroke_deg=stroke, stops_motor_rad=[lower, upper], mid_motor_rad=(lower + upper) / 2,
                   angle_offsets=angle_offsets)

    @property
    def limit_deg(self) -> float:
        """The stops are at +/- this joint angle."""
        return self.stroke_deg / 2

    def save(self, path: Optional[Path] = None) -> None:
        path = path or CAL_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Calibration":
        path = path or CAL_FILE
        if not path.exists():
            raise CalibrationError(f"no calibration at {path}; run homing first (pixi run home-auto)")
        try:
            return cls(**json.loads(path.read_text()))
        except TypeError as e:
            raise CalibrationError(f"{path} is in an older format; run homing again") from e
