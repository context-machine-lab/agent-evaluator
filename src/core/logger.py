"""Centralized logging module using Rich for academic/research output."""

from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from typing import Optional
import time
from datetime import datetime


# Verbosity levels
QUIET = 0      # Only errors and final results
NORMAL = 1     # Standard output with key steps
VERBOSE = 2    # Detailed output with tool calls and results
DEBUG = 3      # All output including model reasoning and debug info


class ResearchLogger:
    """Logger for academic/research projects using Rich formatting."""

    def __init__(self, verbose: bool = True, verbosity_level: int = NORMAL):
        """Initialize the research logger.

        Args:
            verbose: Legacy flag - if False, sets QUIET mode
            verbosity_level: Verbosity level (QUIET=0, NORMAL=1, VERBOSE=2, DEBUG=3)
        """
        self.console = Console()
        # Handle legacy verbose flag
        if not verbose:
            self.verbosity_level = QUIET
        else:
            self.verbosity_level = verbosity_level
        self.verbose = verbose  # Keep for backward compatibility
        self.start_time = time.time()

    def info(self, message: str, prefix: str = "", min_level: int = NORMAL) -> None:
        """Log an info message.

        Args:
            message: The message to log
            prefix: Optional prefix for the message
            min_level: Minimum verbosity level required to show this message
        """
        if self.verbosity_level >= min_level:
            if prefix:
                self.console.print(f"{prefix} {message}")
            else:
                self.console.print(message)

    def success(self, message: str, min_level: int = QUIET) -> None:
        """Log a success message (shown at all levels by default)."""
        if self.verbosity_level >= min_level:
            self.console.print(Text(f"✅ {message}", style="green"))

    def warning(self, message: str, min_level: int = QUIET) -> None:
        """Log a warning message (shown at all levels by default)."""
        if self.verbosity_level >= min_level:
            self.console.print(Text(f"⚠️  {message}", style="yellow"))

    def error(self, message: str, prefix: str = "", min_level: int = QUIET) -> None:
        """Log an error message (shown at all levels by default)."""
        if self.verbosity_level >= min_level:
            if prefix:
                self.console.print(Text(f"{prefix} {message}", style="red"))
            else:
                self.console.print(Text(f"❌ {message}", style="red"))

    def research(self, message: str) -> None:
        """Log a research-specific message."""
        self.console.print(Text(f"🔬 {message}", style="cyan"))

    def data(self, message: str) -> None:
        """Log a data-related message."""
        self.console.print(Text(f"📊 {message}", style="blue"))

    def agent(self, message: str) -> None:
        """Log an agent-related message."""
        self.console.print(Text(f"🤖 {message}", style="magenta"))

    def file(self, message: str) -> None:
        """Log a file-related message."""
        self.console.print(Text(f"📄 {message}", style="dim"))

    def section(self, title: str, content: Optional[str] = None) -> None:
        """Print a section with optional content."""
        if content:
            self.console.print(Panel(content, title=title, border_style="cyan"))
        else:
            self.console.print(f"\n[bold cyan]{'=' * 60}[/bold cyan]")
            self.console.print(f"[bold cyan]{title}[/bold cyan]")
            self.console.print(f"[bold cyan]{'=' * 60}[/bold cyan]\n")

    def table(self, title: str, columns: list, rows: list) -> None:
        """Print a formatted table."""
        table = Table(title=title)
        for col in columns:
            table.add_column(col, style="cyan", no_wrap=True)
        for row in rows:
            table.add_row(*[str(item) for item in row])
        self.console.print(table)

    def metrics(self, metrics: dict) -> None:
        """Log metrics in a formatted way."""
        table = Table(title="Metrics", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        for key, value in metrics.items():
            table.add_row(key, str(value))
        self.console.print(table)

    def timestamp(self, message: str) -> None:
        """Log a message with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.console.print(f"[dim]{timestamp}[/dim] {message}")

    def elapsed_time(self) -> str:
        """Get elapsed time since logger creation."""
        elapsed = time.time() - self.start_time
        if elapsed < 60:
            return f"{elapsed:.1f}s"
        elif elapsed < 3600:
            return f"{elapsed/60:.1f}m"
        else:
            return f"{elapsed/3600:.1f}h"

    def phase_start(self, phase_name: str, phase_num: Optional[int] = None,
                    total_phases: Optional[int] = None) -> None:
        """Log the start of a phase."""
        if phase_num and total_phases:
            msg = f"Phase {phase_num}/{total_phases}: {phase_name}"
        else:
            msg = f"Phase: {phase_name}"
        self.console.print(f"\n[bold cyan]🔍 {msg}[/bold cyan]")
        self.console.print(f"[dim]{'─' * 50}[/dim]")

    def phase_complete(self, phase_name: str) -> None:
        """Log the completion of a phase."""
        self.console.print(f"[green]✓ Completed: {phase_name}[/green]\n")

    def cost_report(self, input_tokens: int, output_tokens: int, cost: float) -> None:
        """Display a cost report."""
        table = Table(title="Cost Report", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow")
        table.add_row("Input Tokens", f"{input_tokens:,}")
        table.add_row("Output Tokens", f"{output_tokens:,}")
        table.add_row("Total Cost", f"${cost:.4f}")
        self.console.print(table)

    def debug(self, message: str, data: Optional[dict] = None) -> None:
        """Log debug information (only shown at DEBUG level)."""
        if self.verbosity_level >= DEBUG:
            self.console.print(f"[dim]DEBUG: {message}[/dim]")
            if data:
                for key, value in data.items():
                    self.console.print(f"[dim]  {key}: {value}[/dim]")

    def is_verbose(self) -> bool:
        """Check if verbosity is at VERBOSE level or higher."""
        return self.verbosity_level >= VERBOSE

    def is_debug(self) -> bool:
        """Check if verbosity is at DEBUG level."""
        return self.verbosity_level >= DEBUG

    def print_raw(self, content: str) -> None:
        """Print raw content without formatting."""
        self.console.print(content)


# Global logger instance
logger = ResearchLogger()