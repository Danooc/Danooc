"""
Global configuration for the Danooc call center system.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Config:
    # --- server ---
    host: str = "0.0.0.0"
    port: int = 8000
    base_url: str = "http://localhost:8000"

    # --- NLP / intent classifier ---
    model_path: str = "./data/intent_model.json"
    training_data_path: str = "./data/training_data.json"
    confidence_threshold: float = 0.40

    # --- IVR ---
    call_flow_path: str = "./data/call_flow.json"
    default_language: str = "es"
    max_retries: int = 3
    input_timeout: int = 5
    speech_timeout: str = "auto"

    # --- TTS ---
    tts_voice: str = "Polly.Mia"
    tts_language: str = "es-MX"

    @classmethod
    def from_json(cls, path: str) -> Config:
        data = json.loads(Path(path).read_text())
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_json(self, path: str) -> None:
        from dataclasses import asdict
        Path(path).write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False))
