"""Command Line Interface for DecisionBench."""

import sys
import json
from pathlib import Path
from typing import Optional
import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich import box

from decision_bench.dataset.loader import load_dataset, get_dataset_statistics
from decision_bench.dataset.schema import DifficultyLevel
from decision_bench.models import create_model
from decision_bench.evaluator import DecisionBenchEvaluator
from decision_bench.tasks import list_tasks, TASK_REGISTRY
from decision_bench.reporting.console import print_report_to_console
from decision_bench.reporting.markdown import save_markdown_report
from decision_bench.reporting.json_export import export_json_report


console = Console()


@click.group()
@click.version_option(version="1.0.0", prog_name="decision-bench")
def main():
    """DecisionBench: Standardized Benchmark Suite for System One & Decision Models."""
    pass


@main.command(name="list-tasks")
def cmd_list_tasks():
    """List all available benchmark evaluation tasks."""
    tasks = list_tasks()
    table = Table(title="📋 Available DecisionBench Tasks", box=box.ROUNDED)
    table.add_column("Task Key", style="cyan bold")
    table.add_column("Name", style="bold")
    table.add_column("Primitive Type", style="yellow")
    table.add_column("Description", style="white")

    for t in tasks:
        table.add_row(t["key"], t["name"], t["task_type"], t["description"])

    console.print(table)


@main.command(name="dataset-info")
@click.option("--dataset-path", type=str, default=None, help="Custom dataset JSON path")
def cmd_dataset_info(dataset_path: Optional[str]):
    """Display statistics and vertical track distribution of the benchmark dataset."""
    path = Path(dataset_path) if dataset_path else None
    samples = load_dataset(path=path)
    stats = get_dataset_statistics(samples)

    console.print(f"[bold cyan]📦 DecisionBench Dataset Statistics[/bold cyan]")
    console.print(f"Total Samples: [bold green]{stats['total_samples']}[/bold green]")
    console.print(f"Clickbait Samples: [bold yellow]{stats['clickbait_count']}[/bold yellow] ({stats['clickbait_ratio'] * 100:.1f}%)")
    console.print(f"Commercial Ad Samples: [bold magenta]{stats['commercial_ad_count']}[/bold magenta] ({stats['commercial_ad_ratio'] * 100:.1f}%)")
    console.print("")

    # Track table
    track_table = Table(title="Vertical Domain Tracks Distribution", box=box.SIMPLE)
    track_table.add_column("Category Track", style="bold")
    track_table.add_column("Count", justify="right")
    track_table.add_column("Share", justify="right", style="dim")

    for track, count in stats["track_distribution"].items():
        share = (count / stats["total_samples"]) * 100
        track_table.add_row(track, str(count), f"{share:.1f}%")

    console.print(track_table)


@main.command(name="run")
@click.option(
    "--model", "-m",
    required=True,
    help="Target model: URL (e.g. http://127.0.0.1:8700), 'laya', 'mock', or local path"
)
@click.option(
    "--tasks", "-t",
    default="all",
    help="Comma-separated list of tasks to run (e.g. 'choice,clickbait') or 'all'"
)
@click.option(
    "--limit", "-l",
    type=int,
    default=None,
    help="Limit number of test samples (useful for quick smoke tests)"
)
@click.option(
    "--difficulty", "-d",
    type=click.Choice(["easy", "medium", "hard", "adversarial"], case_sensitive=False),
    default=None,
    help="Filter benchmark samples by difficulty"
)
@click.option(
    "--output-md",
    type=str,
    default=None,
    help="Path to save Markdown benchmark report (e.g. reports/benchmark.md)"
)
@click.option(
    "--output-json",
    type=str,
    default=None,
    help="Path to save JSON report (e.g. reports/benchmark.json)"
)
@click.option(
    "--include-samples",
    is_flag=True,
    default=False,
    help="Include per-sample raw predictions in JSON report"
)
def cmd_run(
    model: str,
    tasks: str,
    limit: Optional[int],
    difficulty: Optional[str],
    output_md: Optional[str],
    output_json: Optional[str],
    include_samples: bool
):
    """Execute benchmark battery on a Decision Model or System One API."""
    console.print(f"[bold green]🚀 Initializing DecisionBench...[/bold green]")
    console.print(f"Target Model: [bold cyan]{model}[/bold cyan]")

    # 1. Initialize model adapter
    try:
        model_instance = create_model(model)
        console.print(f"Initialized adapter: [bold]{model_instance.name}[/bold]")
    except Exception as e:
        console.print(f"[bold red]Failed to initialize model '{model}': {e}[/bold red]")
        sys.exit(1)

    # 2. Parse task list
    if tasks.lower().strip() == "all":
        selected_tasks = list(TASK_REGISTRY.keys())
    else:
        selected_tasks = [t.strip().lower() for t in tasks.split(",") if t.strip()]

    diff_enum = DifficultyLevel(difficulty.lower()) if difficulty else None

    # 3. Load Evaluator
    evaluator = DecisionBenchEvaluator()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        overall_progress = progress.add_task("[bold cyan]Benchmarking Tasks...", total=len(selected_tasks))
        current_task_bar = progress.add_task("[dim]Running samples...", total=100, visible=False)

        def progress_callback(task_key: str, current: int, total: int, pred: any):
            progress.update(
                current_task_bar,
                total=total,
                completed=current,
                description=f"[yellow]{task_key}[/yellow] ({current}/{total})",
                visible=True
            )

        report = evaluator.run(
            model=model_instance,
            tasks=selected_tasks,
            limit=limit,
            difficulty=diff_enum,
            progress_callback=progress_callback
        )
        progress.update(overall_progress, completed=len(selected_tasks))

    # 4. Display results
    print_report_to_console(report)

    # 5. Export reports if requested
    if output_md:
        save_markdown_report(report, output_md)
        console.print(f"📄 Markdown report saved to: [bold]{output_md}[/bold]")

    if output_json:
        export_json_report(report, output_json, include_samples=include_samples)
        console.print(f"💾 JSON report saved to: [bold]{output_json}[/bold]")


@main.command(name="compare")
@click.argument("report_a_path", type=click.Path(exists=True))
@click.argument("report_b_path", type=click.Path(exists=True))
def cmd_compare(report_a_path: str, report_b_path: str):
    """Compare two benchmark JSON reports side-by-side."""
    with open(report_a_path, "r", encoding="utf-8") as f:
        rep_a = json.load(f)
    with open(report_b_path, "r", encoding="utf-8") as f:
        rep_b = json.load(f)

    model_a = rep_a.get("model_name", "Model A")
    model_b = rep_b.get("model_name", "Model B")

    table = Table(title=f"⚖️ Comparison: {model_a} vs {model_b}", box=box.ROUNDED)
    table.add_column("Metric", style="bold")
    table.add_column(model_a, justify="right", style="cyan")
    table.add_column(model_b, justify="right", style="magenta")
    table.add_column("Delta (B - A)", justify="right")

    # Overall D-Index
    d_a = rep_a.get("decision_index", 0.0)
    d_b = rep_b.get("decision_index", 0.0)
    delta_d = d_b - d_a
    d_color = "green" if delta_d >= 0 else "red"
    table.add_row(
        "Decision Index (D-Index)",
        f"{d_a:.1f}",
        f"{d_b:.1f}",
        f"[{d_color}]{delta_d:+.1f}[/{d_color}]"
    )

    q_a = rep_a.get("overall_quality_score", 0.0)
    q_b = rep_b.get("overall_quality_score", 0.0)
    delta_q = q_b - q_a
    q_color = "green" if delta_q >= 0 else "red"
    table.add_row(
        "Quality Score",
        f"{q_a:.1f}",
        f"{q_b:.1f}",
        f"[{q_color}]{delta_q:+.1f}[/{q_color}]"
    )

    s_a = rep_a.get("overall_speed_score", 0.0)
    s_b = rep_b.get("overall_speed_score", 0.0)
    delta_s = s_b - s_a
    s_color = "green" if delta_s >= 0 else "red"
    table.add_row(
        "Speed Score",
        f"{s_a:.1f}",
        f"{s_b:.1f}",
        f"[{s_color}]{delta_s:+.1f}[/{s_color}]"
    )

    console.print(table)


if __name__ == "__main__":
    main()
