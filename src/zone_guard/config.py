"""ZoneGuard configuration — Pydantic Settings + YAML."""
import os
from pathlib import Path
from typing import Any
import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class CameraConfig(BaseModel):
    id: str
    name: str
    url: str
    camera_type: str = "webcam"
    fps_cap: int = 30
    enabled: bool = True


class ZoneConfig(BaseModel):
    id: str
    name: str
    camera_id: str
    zone_type: str = "restricted"
    polygon: list[list[float]]
    dwell_time_seconds: float = 2.0
    cooldown_seconds: int = 60
    max_occupancy: int = 0
    alert_channels: list[str] = []


class AlertConfig(BaseModel):
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    slack_webhook_url: str = ""
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_from: str = ""
    smtp_to: list[str] = []
    rate_limit_per_zone_seconds: int = 60


class EventConfig(BaseModel):
    dwell_frames: int = 3
    cooldown_seconds: int = 60
    snapshot_quality: int = 85


class StorageConfig(BaseModel):
    local_base_path: str = "data"
    retention_days: int = 30


class Settings(BaseSettings):
    app_name: str = "ZoneGuard"
    log_level: str = "INFO"
    database_url: str = "postgresql+asyncpg://zoneguard:secret@localhost:5432/zoneguard"
    redis_url: str = "redis://localhost:6379/0"
    model_path: str = "models/yolov13n.pt"
    confidence_threshold: float = 0.4
    device: str = "cpu"
    jwt_secret: str = "CHANGE_ME_IN_PRODUCTION"
    admin_username: str = "admin"
    admin_password: str = "admin"
    api_port: int = 8000
    cameras: list[CameraConfig] = []
    zones: list[ZoneConfig] = []
    alerts: AlertConfig = AlertConfig()
    events: EventConfig = EventConfig()
    storage: StorageConfig = StorageConfig()

    model_config = {"env_prefix": "ZG_", "env_nested_delimiter": "__"}

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._load_yaml()

    def _load_yaml(self):
        cfg_dir = Path(os.getenv("ZG_CONFIG_DIR", "configs"))
        for fname, key, cls in [("cameras.yaml","cameras",CameraConfig), ("zones.yaml","zones",ZoneConfig)]:
            fp = cfg_dir / fname
            if fp.exists():
                with open(fp) as f:
                    data = yaml.safe_load(f) or {}
                if key in data:
                    setattr(self, key, [cls(**c) for c in data[key]])
        alerts_f = cfg_dir / "alerts.yaml"
        if alerts_f.exists():
            with open(alerts_f) as f:
                data = yaml.safe_load(f) or {}
            if "alerts" in data:
                self.alerts = AlertConfig(**data["alerts"])

    def resolve_device(self):
        if self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
