from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from automation_bot import __version__
from automation_bot.config import load_flow
from automation_bot.executor import run_flow
from automation_bot.reports import build_report_dict, write_json_report, write_markdown_summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ejecuta flujos de peticiones HTTP y genera reportes (JSON/Markdown)."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "flow",
        type=Path,
        help="Ruta al archivo YAML del flujo (ej. flows/ejemplo.yaml)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports"),
        help="Directorio de salida para reportes (default: reports/)",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Solo imprimir resultado en consola, sin escribir archivos",
    )
    args = parser.parse_args(argv)

    console = Console()
    if not args.flow.is_file():
        console.print(f"[red]No existe el archivo:[/red] {args.flow}")
        return 2

    flow = load_flow(args.flow)
    results, context = run_flow(flow)
    data = build_report_dict(flow.name, flow.description, results, context)

    table = Table(title=f"Flujo: {flow.name}")
    table.add_column("Paso", style="cyan")
    table.add_column("Estado")
    table.add_column("ms", justify="right")
    table.add_column("HTTP")
    for r in results:
        table.add_row(
            r.name,
            "[green]OK[/green]" if r.ok else "[red]FALLO[/red]",
            f"{r.duration_ms:.0f}",
            str(r.status_code) if r.status_code is not None else "—",
        )
    console.print(table)

    if not args.no_report:
        stem = flow.name.replace(" ", "_")
        json_path = args.out_dir / f"{stem}_report.json"
        md_path = args.out_dir / f"{stem}_summary.md"
        write_json_report(json_path, data)
        write_markdown_summary(md_path, data)
        console.print(f"Reporte JSON: [bold]{json_path}[/bold]")
        console.print(f"Resumen MD:   [bold]{md_path}[/bold]")

    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
