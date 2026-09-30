"""Rich console visualizer for DecisionBench benchmark runs."""

from typing import Dict, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
from decision_bench.evaluator import BenchmarkReport


console = Console()


def print_report_to_console(report: BenchmarkReport) -> None:
    """Print beautifully styled benchmark summary tables and metrics to console."""
    # Header Banner
    header = Text()
    header.append("🎯 DecisionBench Evaluation Results\n", style="bold cyan")
    header.append(f"Model: {report.model_name} | Date: {report.timestamp}\n", style="dim")
    header.append(f"Platform: {report.environment.get('platform', 'N/A')} ({report.environment.get('processor', 'N/A')})", style="dim")
    console.print(Panel(header, box=box.ROUNDED, expand=False))

    # Overall Scores Panel
    d_idx = report.decision_index
    idx_color = "green" if d_idx >= 75 else ("yellow" if d_idx >= 60 else "red")

    summary_grid = Table.grid(padding=1)
    summary_grid.add_column(justify="center", min_width=25)
    summary_grid.add_column(justify="center", min_width=25)
    summary_grid.add_column(justify="center", min_width=25)

    summary_grid.add_row(
        Panel(f"[bold]{report.overall_quality_score:.1f} / 100[/bold]\n[dim]Quality Score[/dim]", style="blue", box=box.ROUNDED),
        Panel(f"[bold]{report.overall_speed_score:.1f} / 100[/bold]\n[dim]Speed Score[/dim]", style="magenta", box=box.ROUNDED),
        Panel(f"[bold {idx_color}]{report.decision_index:.1f} / 100[/bold {idx_color}]\n[dim]Decision Index (D-Index)[/dim]", style=idx_color, box=box.ROUNDED)
    )
    console.print(summary_grid)
    console.print("")

    # Task Table
    table = Table(
        title="📋 Task Performance Breakdown",
        box=box.MINIMAL_DOUBLE_HEAD,
        header_style="bold cyan"
    )
    table.add_column("Task Key", style="dim")
    table.add_column("Task Name", style="bold")
    table.add_column("Primitive", justify="center")
    table.add_column("Samples", justify="right")
    table.add_column("Primary Metric", justify="center", style="bold green")
    table.add_column("P50 Latency", justify="right")
    table.add_column("P95 Latency", justify="right")
    table.add_column("Throughput", justify="right", style="cyan")

    for k, row in report.summary_table.items():
        table.add_row(
            k,
            row["task_name"],
            f"[yellow]{row['type']}[/yellow]",
            str(row["samples"]),
            row["primary_metric"],
            f"{row['p50_ms']} ms",
            f"{row['p95_ms']} ms",
            f"{row['qps']} req/s"
        )

    console.print(table)
    console.print("")
