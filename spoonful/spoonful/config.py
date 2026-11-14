"""config.yaml -> validated pydantic model. Every module reads tunables from here (R5).

Models are mutable so the operator drawer can change thresholds live (`set_path`).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).resolve().parent.parent
JOINTS = ["base", "shoulder", "elbow", "wrist_pitch", "wrist_roll", "gripper"]
ARM_JOINTS = JOINTS[:5]  # the gripper is never part of a trajectory


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Speeds(_M):
    default: float = 40
    near_face: float = 12


class ArmCfg(_M):
    backend: Literal["sim", "xarm", "raw"] = "sim"
    servo_map: dict[str, int]
    units_per_deg: float = 4.1667
    center_units: int = 500
    soft_limits_deg: dict[str, tuple[float, float]]
    home_pose: dict[str, float]
    park_pose: dict[str, float]
    max_speed_deg_s: Speeds = Speeds()
    max_accel_deg_s2: float = 120
    stream_hz: float = 50
    read_hz: float = 30
    read_median_window: int = 3
    contact_error_deg: float = 4.8
    contact_hold_s: float = 0.3
    contact_grace_deg: float = 2.0
    stop_torque_off_after_s: float = 1.0
    gripper_close_deg: Optional[float] = None
    heartbeat_timeout_s: float = 0.25
    trajectories_dir: str = "data/trajectories"
    sim_trajectories_dir: str = "data/trajectories_sim"
    blend_s: float = 0.4
    teach_downsample_deg: float = 2.0
    teach_median_window: int = 5
    mouth_arrive_mm: float = 6

    @model_validator(mode="after")
    def _check(self):
        for name in ("servo_map", "soft_limits_deg"):
            missing = set(JOINTS) - set(getattr(self, name))
            if missing:
                raise ValueError(f"arm.{name} missing joints {sorted(missing)}")
        if len(set(self.servo_map.values())) != len(self.servo_map):
            raise ValueError("arm.servo_map has duplicate servo ids")
        for j, (lo, hi) in self.soft_limits_deg.items():
            if lo >= hi:
                raise ValueError(f"soft limit for {j}: lo >= hi")
        for pose_name in ("home_pose", "park_pose"):
            pose = getattr(self, pose_name)
            for j, v in pose.items():
                lo, hi = self.soft_limits_deg[j]
                if not lo <= v <= hi:
                    raise ValueError(f"arm.{pose_name}.{j}={v} outside soft limits {lo}..{hi}")
        return self


class Box(_M):
    x: tuple[float, float]
    y: tuple[float, float]
    z: tuple[float, float]


class KinCfg(_M):
    base_height_mm: float
    upper_arm_mm: float
    forearm_mm: float
    tool_mm: float
    tool_offset_mm: float = 0
    sign: dict[str, float]
    zero_deg: dict[str, float]
    face_zone: Box


class CameraCfg(_M):
    index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    mirror: bool = False
    fallback_width: int = 640
    fallback_height: int = 360
    min_fps_before_fallback: float = 15
    jpeg_quality: int = 70
    hfov_deg: float = 70            # horizontal field of view; only used until calibration measures focal_px
    api: Literal["auto", "any", "dshow", "msmf", "avfoundation", "v4l2"] = "auto"   # auto = DirectShow on Windows


class Window(_M):
    x: tuple[float, float]
    y: tuple[float, float]


class FaceCfg(_M):
    model_path: str
    mouth_open_ratio_threshold: float = 0.08
    hysteresis_fraction: float = 0.7
    yaw_max_deg: float = 15
    pitch_max_deg: float = 20
    pitch_offset_deg: float = 0
    presence_hold_s: float = 0.5
    mouth_open_hold_s: float = 0.3
    mouth_closed_taken_s: float = 0.5
    frame_max_age_ms: float = 150
    delivery_window_px: Window
    ema_alpha: float = Field(0.4, gt=0, le=1)


class OffsetsCfg(_M):
    px_per_mm_at_nominal: float = 3.0
    max_correction_mm: float = 30
    rate_limit_mm_s: float = 10
    joint_map: str = "data/calibration/offset_map.json"
    joints: list[str] = ["base", "shoulder", "elbow", "wrist_pitch"]
    depth: bool = True                  # follow the diner leaning in/back (needs an 'away' calibration sample)
    focal_px: float = 0                 # in 1280-wide coords; 0 = derive from camera.hfov_deg
    iris_mm: float = 11.7               # adult iris diameter, 11.7 +- 0.5 mm across people
    eyes_behind_lips_mm: float = 20     # depth is measured at the eyes; the lips are this much closer
    max_depth_correction_mm: float = 40
    depth_deadband_mm: float = 8
    depth_person_reset_s: float = 3     # face gone this long = maybe a new person: relearn their face width


class GateCfg(_M):
    backoff_mm: float = 50
    retreat_after_s: float = 5


class FsmCfg(_M):
    tick_hz: float = 30
    pre_delivery_timeout_s: float = 20
    max_aborts_per_bite: int = 2
    stop_release_window_s: float = 30
    camera_fault_s: float = 2.0


class PaceCfg(_M):
    min_interval_s: float = 4
    default_interval_s: float = 9
    min_bites_for_model: int = 5
    swallow_threshold: float = 300
    swallow_offset_s: float = 2.0
    model: str = "data/calibration/pace.json"
    auto_offer: bool = False


class SpeechCfg(_M):
    backend: Literal["elevenlabs", "say", "none"] = "say"
    voice: str = "Samantha"
    elevenlabs_voice_id: str = ""
    elevenlabs_model: str = "eleven_turbo_v2_5"
    rate: float = 1.0
    cooldown_s: float = 3
    cache_dir: str = "data/tts_cache"
    enabled: bool = True


class VoiceCfg(_M):
    enabled: bool = False
    wake: str = "spoonful"
    wake_aliases: list[str] = ["spoonful"]
    backend: Literal["faster_whisper", "openai"] = "faster_whisper"
    model: str = "small"
    min_confidence: float = 0.45
    bare_stop: bool = True
    sample_rate: int = 16000
    vad_rms: float = 0.015
    commands: dict[str, list[str]]


class Esp32Cfg(_M):
    enabled: bool = True
    port: str = "auto"
    baud: int = 115200
    heartbeat_timeout_s: float = 3


class FoodCfg(_M):
    enabled: bool = False
    backend: Literal["gemini"] = "gemini"
    model: str = "gemini-2.5-flash"
    interval_s: float = 5
    bowl_roi: tuple[int, int, int, int] = (0, 0, 1280, 720)


class AppCfg(_M):
    host: str = "127.0.0.1"
    port: int = 8000
    ws_hz: float = 10
    stream_fps: float = 15


class PhotonCfg(_M):
    enabled: bool = False
    sidecar_url: str = "http://127.0.0.1:7777"
    caregiver_number: str = ""


class ProfilesCfg(_M):
    dir: str = "data/profiles"
    default: str = "demo"


class LogCfg(_M):
    db: str = "data/spoonful.db"
    servo_log_hz: float = 10


class CalibCfg(_M):
    dir: str = "data/calibration"
    max_age_h: float = 24


class Config(_M):
    arm: ArmCfg
    kinematics: KinCfg
    camera: CameraCfg = CameraCfg()
    face: FaceCfg
    offsets: OffsetsCfg = OffsetsCfg()
    gate: GateCfg = GateCfg()
    fsm: FsmCfg = FsmCfg()
    pace: PaceCfg = PaceCfg()
    speech: SpeechCfg = SpeechCfg()
    voice: VoiceCfg
    esp32: Esp32Cfg = Esp32Cfg()
    food: FoodCfg = FoodCfg()
    app: AppCfg = AppCfg()
    photon: PhotonCfg = PhotonCfg()
    profiles: ProfilesCfg = ProfilesCfg()
    log: LogCfg = LogCfg()
    calibration: CalibCfg = CalibCfg()

    # ---- helpers ----
    def path(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else ROOT / p

    def get_path(self, dotted: str) -> Any:
        obj: Any = self
        for part in dotted.split("."):
            obj = obj[part] if isinstance(obj, dict) else getattr(obj, part)
        return obj.model_dump() if isinstance(obj, BaseModel) else obj

    def set_path(self, dotted: str, value: Any) -> None:
        """Live edit, validated by pydantic (validate_assignment). Only leaf fields of sections."""
        *parents, leaf = dotted.split(".")
        obj: Any = self
        for part in parents:
            obj = getattr(obj, part)
        if isinstance(obj, dict):
            obj[leaf] = value
        else:
            setattr(obj, leaf, value)

    def save(self, path: Optional[Path] = None) -> None:
        path = path or ROOT / "config.yaml"
        path.write_text(yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False))


# Fields the operator drawer may edit live (keep this list short and boring).
LIVE_EDITABLE = [
    "face.mouth_open_ratio_threshold", "face.hysteresis_fraction", "face.yaw_max_deg", "face.pitch_max_deg",
    "face.pitch_offset_deg", "face.presence_hold_s", "face.mouth_open_hold_s", "face.frame_max_age_ms",
    "face.ema_alpha", "face.delivery_window_px", "gate.backoff_mm", "fsm.pre_delivery_timeout_s",
    "arm.max_speed_deg_s.default", "arm.max_speed_deg_s.near_face", "offsets.max_correction_mm",
    "offsets.rate_limit_mm_s", "offsets.depth", "offsets.max_depth_correction_mm", "pace.auto_offer", "speech.enabled",
]


def load(path: Optional[str | Path] = None) -> Config:
    path = Path(path) if path else ROOT / "config.yaml"
    data = yaml.safe_load(path.read_text())
    return Config.model_validate(data)


def load_env(path: Optional[Path] = None) -> None:
    """Minimal .env reader (no extra dependency). Existing environment wins."""
    path = path or ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
