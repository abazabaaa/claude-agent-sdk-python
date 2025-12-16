#!/usr/bin/env python3
"""
Claude Agent SDK Infinite Loop Runner
======================================

A script that runs Claude Code in an infinite loop with:
- Rich console visualization of events
- Security hooks to prevent dangerous commands
- Opus 4.5 model with 64k thinking tokens
- Session persistence and resume capability

Usage:
    python runner.py --prompt "Your task here"
    python runner.py --prompt-file task.txt
    python runner.py --prompt "Your task" --max-iterations 5
    python runner.py --resume <session_id>

Environment Variables:
    ANTHROPIC_API_KEY: Your Anthropic API key (required)
    HARNESS_PROJECT_DIR: Working directory for the agent (default: current dir)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from console_renderer import ConsoleRenderer
from security_hooks import get_security_hooks, set_project_dir

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    CLIConnectionError,
    ProcessError,
    ResultMessage,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("harness.runner")

# Model configuration
MODEL_ID = "claude-opus-4-5-20251101"
MAX_THINKING_TOKENS = 64000

# Default configuration
DEFAULT_MAX_ITERATIONS = None  # Infinite
DEFAULT_DELAY_BETWEEN_SESSIONS = 2.0  # seconds
DEFAULT_AUTO_CONTINUE_PROMPT = "Continue with the next task. If all tasks are complete, summarize what was accomplished."


class GracefulExitError(Exception):
    """Exception raised for graceful shutdown."""

    pass


def setup_signal_handlers() -> None:
    """Set up signal handlers for graceful shutdown."""

    def signal_handler(signum: int, frame: Any) -> None:
        raise GracefulExitError()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


async def run_single_session(
    client: ClaudeSDKClient,
    prompt: str,
    renderer: ConsoleRenderer,
    iteration: int,
) -> tuple[str, str | None, dict[str, Any] | None]:
    """
    Run a single agent session.

    Args:
        client: The Claude SDK client.
        prompt: The prompt to send.
        renderer: The console renderer.
        iteration: The current iteration number.

    Returns:
        Tuple of (status, session_id, session_info) where:
        - status: "continue", "complete", or "error"
        - session_id: The session ID if available
        - session_info: Additional session information
    """
    renderer.print_session_header(iteration, prompt)

    try:
        # Send the query
        await client.query(prompt)

        # Process response
        session_id: str | None = None
        session_info: dict[str, Any] = {}

        async for message in client.receive_response():
            # Render the message
            renderer.handle(message)

            # Extract session info from result
            if isinstance(message, ResultMessage):
                session_id = message.session_id
                session_info = {
                    "session_id": session_id,
                    "cost_usd": message.total_cost_usd,
                    "turns": message.num_turns,
                    "duration_ms": message.duration_ms,
                    "is_error": message.is_error,
                    "usage": message.usage,
                }

                if message.is_error:
                    return "error", session_id, session_info

        return "continue", session_id, session_info

    except ProcessError as e:
        logger.error("Process error: %s", e)
        renderer.print_error(f"Process error (exit code {e.exit_code}): {e.stderr}")
        return "error", None, {"error": str(e)}

    except CLIConnectionError as e:
        logger.error("Connection error: %s", e)
        renderer.print_error(f"Connection error: {e}")
        return "error", None, {"error": str(e)}

    except Exception as e:
        logger.exception("Unexpected error: %s", e)
        renderer.print_error(f"Unexpected error: {e}")
        return "error", None, {"error": str(e)}


async def run_infinite_loop(
    initial_prompt: str,
    project_dir: Path,
    max_iterations: int | None = None,
    delay_between_sessions: float = DEFAULT_DELAY_BETWEEN_SESSIONS,
    auto_continue_prompt: str = DEFAULT_AUTO_CONTINUE_PROMPT,
    resume_session_id: str | None = None,
    permission_mode: str = "default",
) -> None:
    """
    Run the agent in an infinite loop.

    Args:
        initial_prompt: The initial prompt to send.
        project_dir: The working directory for the agent.
        max_iterations: Maximum number of iterations (None for infinite).
        delay_between_sessions: Delay between session iterations.
        auto_continue_prompt: Prompt to use for continuation.
        resume_session_id: Session ID to resume from.
        permission_mode: Permission mode for the agent.
    """
    console = Console()
    renderer = ConsoleRenderer(console=console, model_name=MODEL_ID)

    # Set project directory for security hooks
    set_project_dir(str(project_dir))

    # Print startup banner
    console.print()
    console.print(
        Panel(
            Text.from_markup(
                f"[bold cyan]Claude Agent SDK Runner[/bold cyan]\n\n"
                f"[white]Model:[/white] [green]{MODEL_ID}[/green]\n"
                f"[white]Thinking Tokens:[/white] [green]{MAX_THINKING_TOKENS:,}[/green]\n"
                f"[white]Project Dir:[/white] [blue]{project_dir}[/blue]\n"
                f"[white]Max Iterations:[/white] [yellow]{max_iterations or 'Infinite'}[/yellow]\n"
                f"[white]Permission Mode:[/white] [magenta]{permission_mode}[/magenta]"
            ),
            title="[bold]Configuration[/bold]",
            border_style="cyan",
        )
    )

    # Configure client options
    options = ClaudeAgentOptions(
        model=MODEL_ID,
        max_thinking_tokens=MAX_THINKING_TOKENS,
        cwd=str(project_dir),
        permission_mode=permission_mode,  # type: ignore[arg-type]
        hooks=get_security_hooks(),
        resume=resume_session_id,
    )

    iteration = 0
    current_prompt = initial_prompt
    last_session_id: str | None = resume_session_id
    total_cost: float = 0.0

    try:
        while True:
            iteration += 1

            # Check iteration limit
            if max_iterations and iteration > max_iterations:
                console.print(
                    f"\n[yellow]Reached max iterations ({max_iterations})[/yellow]"
                )
                break

            # Create client for this iteration
            # Use resume if we have a previous session and it's not the first iteration
            iter_options = options
            if iteration > 1 and last_session_id:
                iter_options = ClaudeAgentOptions(
                    model=MODEL_ID,
                    max_thinking_tokens=MAX_THINKING_TOKENS,
                    cwd=str(project_dir),
                    permission_mode=permission_mode,  # type: ignore[arg-type]
                    hooks=get_security_hooks(),
                    resume=last_session_id,
                )

            async with ClaudeSDKClient(options=iter_options) as client:
                status, session_id, session_info = await run_single_session(
                    client=client,
                    prompt=current_prompt,
                    renderer=renderer,
                    iteration=iteration,
                )

                # Track session ID for continuation
                if session_id:
                    last_session_id = session_id

                # Track costs
                if session_info and session_info.get("cost_usd"):
                    total_cost += session_info["cost_usd"]

                # Handle status
                if status == "error":
                    console.print(
                        "\n[red]Session encountered an error. "
                        f"Retrying in {delay_between_sessions}s...[/red]"
                    )
                    await asyncio.sleep(delay_between_sessions)
                    # Retry with same prompt
                    continue

                elif status == "complete":
                    console.print("\n[green]All tasks completed![/green]")
                    break

            # Prepare for next iteration
            current_prompt = auto_continue_prompt

            # Delay between sessions
            if max_iterations is None or iteration < max_iterations:
                console.print(
                    f"\n[dim]Next session in {delay_between_sessions}s... "
                    f"(Press Ctrl+C to stop)[/dim]"
                )
                await asyncio.sleep(delay_between_sessions)

    except GracefulExitError:
        console.print("\n[yellow]Graceful shutdown requested...[/yellow]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")

    finally:
        # Print final summary
        console.print()
        console.print(
            Panel(
                Text.from_markup(
                    f"[white]Total Iterations:[/white] [cyan]{iteration}[/cyan]\n"
                    f"[white]Total Cost:[/white] [green]${total_cost:.6f}[/green]\n"
                    f"[white]Last Session ID:[/white] [blue]{last_session_id or 'N/A'}[/blue]"
                ),
                title="[bold]Session Summary[/bold]",
                border_style="green",
            )
        )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Claude Code in an infinite loop with rich visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with a simple prompt
    python runner.py --prompt "Help me refactor this codebase"

    # Run with a prompt from a file
    python runner.py --prompt-file task.txt

    # Run with a maximum number of iterations
    python runner.py --prompt "Implement feature X" --max-iterations 10

    # Resume a previous session
    python runner.py --resume abc123 --prompt "Continue"

    # Run with bypass permissions (use with caution)
    python runner.py --prompt "Task" --permission-mode bypassPermissions
        """,
    )

    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument(
        "--prompt",
        type=str,
        help="The initial prompt to send to the agent",
    )
    prompt_group.add_argument(
        "--prompt-file",
        type=Path,
        help="Path to a file containing the prompt",
    )

    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path.cwd(),
        help="Working directory for the agent (default: current directory)",
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Maximum number of iterations (default: infinite)",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY_BETWEEN_SESSIONS,
        help=f"Delay between sessions in seconds (default: {DEFAULT_DELAY_BETWEEN_SESSIONS})",
    )

    parser.add_argument(
        "--continue-prompt",
        type=str,
        default=DEFAULT_AUTO_CONTINUE_PROMPT,
        help="Prompt to use for continuation between sessions",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Session ID to resume from",
    )

    parser.add_argument(
        "--permission-mode",
        type=str,
        choices=["default", "acceptEdits", "bypassPermissions"],
        default="default",
        help="Permission mode for the agent (default: default)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Set up signal handlers
    setup_signal_handlers()

    # Get prompt
    if args.prompt_file:
        if not args.prompt_file.exists():
            print(f"Error: Prompt file not found: {args.prompt_file}", file=sys.stderr)
            sys.exit(1)
        prompt = args.prompt_file.read_text().strip()
    else:
        prompt = args.prompt

    if not prompt:
        print("Error: Empty prompt", file=sys.stderr)
        sys.exit(1)

    # Validate project directory
    project_dir = args.project_dir.resolve()
    if not project_dir.exists():
        print(f"Error: Project directory not found: {project_dir}", file=sys.stderr)
        sys.exit(1)

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "Warning: ANTHROPIC_API_KEY not set. Make sure it's configured.",
            file=sys.stderr,
        )

    # Run the loop
    asyncio.run(
        run_infinite_loop(
            initial_prompt=prompt,
            project_dir=project_dir,
            max_iterations=args.max_iterations,
            delay_between_sessions=args.delay,
            auto_continue_prompt=args.continue_prompt,
            resume_session_id=args.resume,
            permission_mode=args.permission_mode,
        )
    )


if __name__ == "__main__":
    main()
