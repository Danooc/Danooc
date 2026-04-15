from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")


def substitute_env(value: Any, context: dict[str, Any]) -> Any:
    """Reemplaza ${VAR} por variables de entorno o valores del contexto del flujo."""
    if isinstance(value, str):

        def repl(match: re.Match[str]) -> str:
            key = match.group(1).strip()
            if key in context:
                return str(context[key])
            env = os.environ.get(key)
            if env is None:
                raise KeyError(
                    f"Variable '{key}' no definida en contexto ni en entorno"
                )
            return env

        return _VAR_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: substitute_env(v, context) for k, v in value.items()}
    if isinstance(value, list):
        return [substitute_env(item, context) for item in value]
    return value


@dataclass
class RequestSpec:
    method: str
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    body: Any | None = None
    json_body: Any | None = None


@dataclass
class StepSpec:
    id: str
    name: str
    request: RequestSpec
    expect: dict[str, Any] = field(default_factory=dict)
    capture: dict[str, str] = field(default_factory=dict)


@dataclass
class FlowSpec:
    name: str
    description: str
    defaults: dict[str, Any]
    steps: list[StepSpec]


def load_flow(path: Path) -> FlowSpec:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("El flujo YAML debe ser un objeto en la raíz")

    defaults = raw.get("defaults") or {}
    steps_raw = raw.get("steps") or []
    steps: list[StepSpec] = []
    for i, s in enumerate(steps_raw):
        if not isinstance(s, dict):
            raise ValueError(f"Paso {i}: debe ser un objeto")
        req = s.get("request") or {}
        steps.append(
            StepSpec(
                id=str(s.get("id") or f"step_{i + 1}"),
                name=str(s.get("name") or s.get("id") or f"Paso {i + 1}"),
                request=RequestSpec(
                    method=str(req.get("method", "GET")).upper(),
                    url=str(req.get("url", "")),
                    headers=dict(req.get("headers") or {}),
                    body=req.get("body"),
                    json_body=req.get("json"),
                ),
                expect=dict(s.get("expect") or {}),
                capture=dict(s.get("capture") or {}),
            )
        )

    return FlowSpec(
        name=str(raw.get("name") or path.stem),
        description=str(raw.get("description") or ""),
        defaults=defaults,
        steps=steps,
    )
