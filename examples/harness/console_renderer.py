"""Rich console rendering for Claude Agent SDK events.

This module provides the ConsoleRenderer class which buffers and renders
agent events as grouped panels, providing a clean visualization of
multi-turn agent conversations.

Event types from Claude Agent SDK:
- AssistantMessage: Contains TextBlock, ThinkingBlock, ToolUseBlock
- UserMessage: Contains ToolResultBlock for tool responses
- SystemMessage: System-level messages
- ResultMessage: Final result with cost/usage info
- StreamEvent: Partial streaming updates (when include_partial_messages=True)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from claude_agent_sdk.types import (
    AssistantMessage,
    Message,
    ResultMessage,
    StreamEvent,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)


def format_json(data: Any, max_length: int = 50000) -> str:
    """Format data as JSON string."""
    try:
        json_str = json.dumps(data, indent=2, default=str)
        if len(json_str) > max_length:
            return json_str[:max_length] + "\n... (truncated)"
        return json_str
    except Exception:
        return str(data)


@dataclass
class TurnBuffer:
    """Buffers for a single turn's content."""

    thoughts: list[str] = field(default_factory=list)
    text_content: list[str] = field(default_factory=list)
    tool_calls: list[ToolUseBlock] = field(default_factory=list)
    tool_results: list[ToolResultBlock] = field(default_factory=list)
    model: str | None = None

    def clear(self) -> None:
        """Clear all buffers."""
        self.thoughts.clear()
        self.text_content.clear()
        self.tool_calls.clear()
        self.tool_results.clear()
        self.model = None


class ConsoleRenderer:
    """Renders Claude agent events to the console with buffering and grouping.

    This renderer processes messages from the Claude Agent SDK and renders them
    as cohesive grouped panels showing:
    - Model response (thinking + text + tool calls) in one panel
    - Tool results in a separate panel
    - Final result summary with cost information

    The renderer manages its own state based on the event stream.
    """

    def __init__(
        self,
        console: Console | None = None,
        model_name: str = "claude-opus-4-5-20251101",
        show_stream_events: bool = False,
        truncate_thinking: int = 2000,
        truncate_content: int = 3000,
        truncate_tool_output: int = 1000,
    ):
        """Initialize the console renderer.

        Args:
            console: Optional Rich console instance. Creates one if not provided.
            model_name: Model name to display in output panels.
            show_stream_events: Whether to show streaming events (default False).
            truncate_thinking: Max characters for thinking blocks.
            truncate_content: Max characters for content blocks.
            truncate_tool_output: Max characters for tool output.
        """
        self.console = console or Console()
        self.model_name = model_name
        self.show_stream_events = show_stream_events
        self.truncate_thinking = truncate_thinking
        self.truncate_content = truncate_content
        self.truncate_tool_output = truncate_tool_output

        self._turn_number = 0
        self._buffer = TurnBuffer()
        self._total_cost_usd: float = 0.0
        self._total_turns: int = 0
        self._stream_text_buffer: str = ""

    def handle(self, message: Message) -> None:
        """Handle a message from the Claude Agent SDK.

        Args:
            message: The message to handle.
        """
        if isinstance(message, StreamEvent):
            self._handle_stream_event(message)
        elif isinstance(message, AssistantMessage):
            self._handle_assistant_message(message)
        elif isinstance(message, UserMessage):
            self._handle_user_message(message)
        elif isinstance(message, SystemMessage):
            self._handle_system_message(message)
        elif isinstance(message, ResultMessage):
            self._handle_result_message(message)

    def _handle_stream_event(self, event: StreamEvent) -> None:
        """Handle a streaming event for partial updates."""
        if not self.show_stream_events:
            return

        # Extract text delta if present
        raw_event = event.event
        if raw_event.get("type") == "content_block_delta":
            delta = raw_event.get("delta", {})
            if delta.get("type") == "text_delta":
                text = delta.get("text", "")
                self._stream_text_buffer += text
                # Print incrementally
                self.console.print(text, end="", style="white")

    def _handle_assistant_message(self, message: AssistantMessage) -> None:
        """Handle an assistant message with content blocks."""
        # Clear stream buffer if we were streaming
        if self._stream_text_buffer:
            self.console.print()  # Newline after streaming
            self._stream_text_buffer = ""

        # Capture model name
        if message.model:
            self._buffer.model = message.model

        # Process content blocks
        for block in message.content:
            if isinstance(block, ThinkingBlock):
                self._buffer.thoughts.append(block.thinking)
            elif isinstance(block, TextBlock):
                self._buffer.text_content.append(block.text)
            elif isinstance(block, ToolUseBlock):
                self._buffer.tool_calls.append(block)

        # Render the assistant response immediately
        self._render_assistant_response()

    def _handle_user_message(self, message: UserMessage) -> None:
        """Handle a user message (often contains tool results)."""
        # Check if this contains tool results
        if isinstance(message.content, list):
            for block in message.content:
                if isinstance(block, ToolResultBlock):
                    self._buffer.tool_results.append(block)

        # Render tool results if we have any
        if self._buffer.tool_results:
            self._render_tool_results()

    def _handle_system_message(self, message: SystemMessage) -> None:
        """Handle a system message."""
        # Display system messages as info panels
        if message.subtype == "init":
            # Initialization message - can be verbose, skip
            return

        panel = Panel(
            Text(f"[{message.subtype}] {message.data}", style="dim"),
            title="[dim]System[/dim]",
            border_style="dim",
            expand=False,
        )
        self.console.print(panel)

    def _handle_result_message(self, result: ResultMessage) -> None:
        """Handle a result message (end of response)."""
        self._turn_number += 1
        self._total_turns = result.num_turns

        if result.total_cost_usd is not None:
            self._total_cost_usd = result.total_cost_usd

        # Render result summary
        self._render_result_summary(result)

        # Clear buffers for next turn
        self._buffer.clear()

    def _render_assistant_response(self) -> None:
        """Render buffered assistant response as a single grouped panel."""
        # Count content blocks
        block_count = (
            len(self._buffer.thoughts)
            + len(self._buffer.text_content)
            + len(self._buffer.tool_calls)
        )
        if block_count == 0:
            return

        # Build main tree
        model_display = self._buffer.model or self.model_name
        tree = Tree("[bold cyan]Claude Response[/bold cyan] ([green]assistant[/green])")

        # Metadata nodes
        tree.add(f"[dim white]Model:[/dim white] {model_display}")
        tree.add(f"[dim white]Turn:[/dim white] {self._turn_number + 1}")

        # Content blocks
        content_tree = tree.add(
            f"[bold white]Content[/bold white] ({block_count} block{'s' if block_count != 1 else ''})"
        )

        block_num = 1

        # Render thought blocks
        for thought in self._buffer.thoughts:
            block_tree = content_tree.add(f"[dim white]Block {block_num}[/dim white]")
            block_num += 1

            thought_node = block_tree.add("[italic magenta]Thinking[/italic magenta]")
            text = thought
            if len(text) > self.truncate_thinking:
                text = text[: self.truncate_thinking] + "\n... (truncated)"
            thought_node.add(Text(text, style="italic dim white"))

        # Render text content blocks
        for content in self._buffer.text_content:
            block_tree = content_tree.add(f"[dim white]Block {block_num}[/dim white]")
            block_num += 1

            text_node = block_tree.add("[cyan]Text[/cyan]")
            text = content
            if len(text) > self.truncate_content:
                text = text[: self.truncate_content] + "\n... (truncated)"
            text_node.add(Text(text, style="white"))

        # Render tool calls
        for call in self._buffer.tool_calls:
            block_tree = content_tree.add(f"[dim white]Block {block_num}[/dim white]")
            block_num += 1

            tool_node = block_tree.add(
                f"[yellow]Tool Use:[/yellow] [bold yellow]{call.name}[/bold yellow]"
            )

            if call.id:
                tool_node.add(f"[dim white]ID:[/dim white] {call.id}")

            if call.input:
                input_node = tool_node.add("[green]Input:[/green]")
                json_syntax = Syntax(
                    format_json(call.input),
                    "json",
                    theme="monokai",
                    line_numbers=False,
                )
                input_node.add(json_syntax)

        # Create panel
        panel = Panel(
            tree,
            title=f"[bold]Claude API Response (Turn {self._turn_number + 1})[/bold]",
            border_style="cyan",
            expand=False,
        )

        self.console.print(panel)

        # Clear content buffers (keep tool calls for matching with results)
        self._buffer.thoughts.clear()
        self._buffer.text_content.clear()

    def _render_tool_results(self) -> None:
        """Render tool results as a grouped panel."""
        if not self._buffer.tool_results:
            return

        # Build tree for tool response
        tree = Tree("[bold green]Tool Response[/bold green] ([green]user[/green])")

        tree.add(f"[dim white]Turn:[/dim white] {self._turn_number + 1}")

        # Count successes/errors
        errors = sum(1 for r in self._buffer.tool_results if r.is_error)
        successes = len(self._buffer.tool_results) - errors

        results_tree = tree.add(
            f"[bold white]Results[/bold white] "
            f"({len(self._buffer.tool_results)} result{'s' if len(self._buffer.tool_results) != 1 else ''}: "
            f"[green]{successes} ok[/green], [red]{errors} error{'s' if errors != 1 else ''}[/red])"
        )

        for i, result in enumerate(self._buffer.tool_results, 1):
            color = "red" if result.is_error else "green"
            status = "ERROR" if result.is_error else "OK"

            result_tree = results_tree.add(f"[dim white]Result {i}[/dim white]")

            # Find matching tool name from tool calls if possible
            tool_name = "unknown"
            for call in self._buffer.tool_calls:
                if call.id == result.tool_use_id:
                    tool_name = call.name
                    break

            tool_node = result_tree.add(
                f"[{color}]Tool Result:[/{color}] "
                f"[bold {color}]{tool_name}[/bold {color}] "
                f"[dim][{status}][/dim]"
            )

            if result.tool_use_id:
                tool_node.add(f"[dim white]ID:[/dim white] {result.tool_use_id}")

            # Format output
            output_node = tool_node.add(f"[{color}]Output:[/{color}]")
            if isinstance(result.content, (dict, list)):
                json_str = format_json(result.content)
                if len(json_str) > self.truncate_tool_output:
                    json_str = (
                        json_str[: self.truncate_tool_output] + "\n... (truncated)"
                    )
                json_syntax = Syntax(
                    json_str,
                    "json",
                    theme="monokai",
                    line_numbers=False,
                )
                output_node.add(json_syntax)
            else:
                text = str(result.content) if result.content else "(empty)"
                if len(text) > self.truncate_tool_output:
                    text = text[: self.truncate_tool_output] + "... (truncated)"
                output_node.add(Text(text, style="white"))

        # Create panel
        panel = Panel(
            tree,
            title="[bold]Tool Execution Results[/bold]",
            border_style="green",
            expand=False,
        )

        self.console.print(panel)

        # Clear tool buffers
        self._buffer.tool_calls.clear()
        self._buffer.tool_results.clear()

    def _render_result_summary(self, result: ResultMessage) -> None:
        """Render a result message with cost and usage information."""
        # Build usage string
        usage_parts = []
        if result.usage:
            input_tokens = result.usage.get("input_tokens", 0)
            output_tokens = result.usage.get("output_tokens", 0)
            cache_read = result.usage.get("cache_read_input_tokens", 0)
            cache_write = result.usage.get("cache_creation_input_tokens", 0)

            if input_tokens:
                usage_parts.append(f"[cyan]{input_tokens:,}[/cyan] in")
            if output_tokens:
                usage_parts.append(f"[green]{output_tokens:,}[/green] out")
            if cache_read:
                usage_parts.append(f"[yellow]{cache_read:,}[/yellow] cache_read")
            if cache_write:
                usage_parts.append(f"[magenta]{cache_write:,}[/magenta] cache_write")

        # Build summary table
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Metric", style="dim")
        table.add_column("Value")

        table.add_row("Session ID", result.session_id)
        table.add_row("Turns", str(result.num_turns))
        table.add_row("Duration", f"{result.duration_ms:,} ms")
        table.add_row("API Time", f"{result.duration_api_ms:,} ms")

        if result.total_cost_usd is not None:
            table.add_row("Cost", f"${result.total_cost_usd:.6f}")

        if usage_parts:
            table.add_row("Tokens", " | ".join(usage_parts))

        if result.is_error:
            table.add_row("Status", "[red]ERROR[/red]")
        else:
            table.add_row("Status", "[green]OK[/green]")

        # Create panel
        border_color = "red" if result.is_error else "blue"
        panel = Panel(
            table,
            title="[bold]Response Complete[/bold]",
            border_style=border_color,
            expand=False,
        )

        self.console.print(panel)

    def print_session_header(self, iteration: int, prompt: str) -> None:
        """Print a header for a new session iteration.

        Args:
            iteration: The iteration number (1-based).
            prompt: The prompt being sent.
        """
        self.console.print()
        self.console.rule(f"[bold cyan]Session {iteration}[/bold cyan]", style="cyan")
        self.console.print()

        # Show prompt in a panel
        prompt_display = prompt
        if len(prompt_display) > 500:
            prompt_display = prompt_display[:500] + "..."

        panel = Panel(
            Text(prompt_display, style="white"),
            title="[bold]User Prompt[/bold]",
            border_style="blue",
            expand=False,
        )
        self.console.print(panel)
        self.console.print()

    def print_summary(self) -> None:
        """Print session summary statistics."""
        table = Table(title="Session Summary", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Turns", str(self._total_turns))
        table.add_row("Total Cost", f"${self._total_cost_usd:.6f}")

        self.console.print()
        self.console.print(table)

    def print_error(self, error: str | Exception) -> None:
        """Print an error message.

        Args:
            error: The error to display.
        """
        panel = Panel(
            Text(str(error), style="bold white"),
            title="[bold red]Error[/bold red]",
            border_style="red",
            expand=False,
        )
        self.console.print(panel)

    def reset(self) -> None:
        """Reset the renderer state for a new session."""
        self._turn_number = 0
        self._buffer.clear()
        self._stream_text_buffer = ""
