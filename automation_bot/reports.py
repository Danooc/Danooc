from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from automation_bot.executor import StepResult


def build_report_dict(
    flow_name: str,
    flow_description: str,
    results: list[StepResult],
    context: dict[str, Any],
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "flow": {"name": flow_name, "description": flow_description},
        "summary": {
            "total_steps": len(results),
            "passed": sum(1 for r in results if r.ok),
            "failed": sum(1 for r in results if not r.ok),
        },
        "context": context,
        "steps": [
            {
                "id": r.step_id,
                "name": r.name,
                "ok": r.ok,
                "duration_ms": round(r.duration_ms, 2),
                "request": {"method": r.request_method, "url": r.request_url},
                "status_code": r.status_code,
                "error": r.error,
                "captured": r.captured,
                "response_preview": r.response_body_preview,
            }
            for r in results
        ],
    }


def write_json_report(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_markdown_summary(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Reporte: {data['flow']['name']}",
        "",
        f"- Generado: `{data['generated_at']}`",
        f"- Pasos: {data['summary']['total_steps']} · OK: {data['summary']['passed']} · Fallidos: {data['summary']['failed']}",
        "",
    ]
    if data["flow"].get("description"):
        lines.extend([data["flow"]["description"], ""])

    for step in data["steps"]:
        estado = "OK" if step["ok"] else "FALLO"
        lines.append(f"## [{estado}] {step['name']} (`{step['id']}`)")
        lines.append(f"- Duración: {step['duration_ms']} ms")
        lines.append(f"- `{step['request']['method']}` {step['request']['url']}")
        if step.get("status_code") is not None:
            lines.append(f"- HTTP: {step['status_code']}")
        if step.get("error"):
            lines.append(f"- **Error:** {step['error']}")
        if step.get("captured"):
            lines.append(f"- Capturado: `{step['captured']}`")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
