from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from automation_bot.config import FlowSpec, StepSpec, substitute_env


def _get_by_path(obj: Any, path: str) -> Any:
    cur: Any = obj
    for part in path.split("."):
        if part == "":
            continue
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            raise KeyError(f"No se puede acceder a '{part}' en {type(cur).__name__}")
    return cur


@dataclass
class StepResult:
    step_id: str
    name: str
    ok: bool
    duration_ms: float
    request_method: str
    request_url: str
    status_code: int | None
    error: str | None = None
    response_headers: dict[str, str] = field(default_factory=dict)
    response_body_preview: str | None = None
    captured: dict[str, Any] = field(default_factory=dict)


def _preview_body(text: str, max_len: int = 2000) -> str:
    text = text.strip()
    if len(text) <= max_len:
        return text
    return text[:max_len] + "\n… [truncado]"


def run_step(
    client: httpx.Client,
    step: StepSpec,
    context: dict[str, Any],
    timeout: float,
) -> StepResult:
    req = step.request
    url = substitute_env(req.url, context)
    headers = substitute_env(dict(req.headers), context)
    body = substitute_env(req.body, context) if req.body is not None else None
    json_body = substitute_env(req.json_body, context) if req.json_body is not None else None

    t0 = time.perf_counter()
    status_code: int | None = None
    resp_headers: dict[str, str] = {}
    preview: str | None = None
    parsed_json: Any = None
    err: str | None = None

    json_payload: Any | None = json_body
    if json_payload is None and isinstance(body, dict):
        json_payload = body
        body = None

    try:
        resp = client.request(
            req.method,
            url,
            headers=headers or None,
            content=body if isinstance(body, (str, bytes)) else None,
            json=json_payload,
            timeout=timeout,
        )
        status_code = resp.status_code
        resp_headers = {k: v for k, v in resp.headers.items()}
        text = resp.text
        preview = _preview_body(text)
        ct = resp.headers.get("content-type", "")
        if "application/json" in ct.lower():
            try:
                parsed_json = resp.json()
            except json.JSONDecodeError:
                parsed_json = None

        allowed = step.expect.get("status")
        if allowed is not None:
            if isinstance(allowed, int):
                allowed_list = [allowed]
            else:
                allowed_list = list(allowed)
            if status_code not in allowed_list:
                err = f"Status {status_code} no está en {allowed_list}"

    except Exception as e:
        err = f"{type(e).__name__}: {e}"

    duration_ms = (time.perf_counter() - t0) * 1000.0
    captured: dict[str, Any] = {}
    if err is None and parsed_json is not None and step.capture:
        for key, json_path in step.capture.items():
            try:
                captured[key] = _get_by_path(parsed_json, json_path)
            except (KeyError, TypeError) as e:
                err = f"capture.{key}: {e}"

    ok = err is None
    return StepResult(
        step_id=step.id,
        name=step.name,
        ok=ok,
        duration_ms=duration_ms,
        request_method=req.method,
        request_url=url,
        status_code=status_code,
        error=err,
        response_headers=resp_headers,
        response_body_preview=preview,
        captured=captured,
    )


def run_flow(flow: FlowSpec) -> tuple[list[StepResult], dict[str, Any]]:
    timeout = float(flow.defaults.get("timeout_seconds", 30))
    results: list[StepResult] = []
    context: dict[str, Any] = {}

    with httpx.Client(follow_redirects=True) as client:
        for step in flow.steps:
            result = run_step(client, step, context, timeout=timeout)
            results.append(result)
            if result.captured:
                context.update(result.captured)
            if not result.ok:
                break

    return results, context
