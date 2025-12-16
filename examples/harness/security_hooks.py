"""
Security Hooks for Claude Agent SDK
====================================

Pre-tool-use hooks that validate bash commands for security.
Uses a blocklist approach to prevent dangerous commands while allowing
general development workflows.
"""

from __future__ import annotations

import logging
import re
import shlex
from pathlib import Path
from typing import Any

from claude_agent_sdk.types import (
    HookContext,
    HookInput,
    HookJSONOutput,
)

logger = logging.getLogger("harness.security")


# Dangerous command patterns that should always be blocked
DANGEROUS_PATTERNS: list[tuple[str, str]] = [
    # Recursive deletion
    (r"\brm\s+(-[a-zA-Z]*r[a-zA-Z]*\s+|.*\s+-[a-zA-Z]*r)", "Recursive rm is blocked"),
    (r"\brm\s+-rf\b", "rm -rf is blocked"),
    (r"\brm\s+-fr\b", "rm -fr is blocked"),
    (r"\brm\s+--recursive\b", "rm --recursive is blocked"),
    # Force removal without confirmation
    (r"\brm\s+-f\s+/", "Forced removal of root paths is blocked"),
    # Disk formatting
    (r"\bmkfs\b", "Disk formatting commands are blocked"),
    (r"\bfdisk\b", "Disk partitioning commands are blocked"),
    (r"\bparted\b", "Disk partitioning commands are blocked"),
    # System destruction
    (r">\s*/dev/sd[a-z]", "Writing to disk devices is blocked"),
    (r"\bdd\s+.*of=/dev/", "dd to devices is blocked"),
    # Fork bombs and resource exhaustion
    (r":\(\)\{.*:\|:.*\}", "Fork bombs are blocked"),
    (r"\bforkbomb\b", "Fork bombs are blocked"),
    # Dangerous redirects
    (r">\s*/dev/null\s*2>&1\s*&", "Backgrounded silent commands are suspicious"),
    # Network exfiltration patterns
    (r"\bcurl\b.*\|\s*bash", "Piping curl to bash is blocked"),
    (r"\bwget\b.*\|\s*bash", "Piping wget to bash is blocked"),
    (r"\bcurl\b.*\|\s*sh", "Piping curl to sh is blocked"),
    (r"\bwget\b.*\|\s*sh", "Piping wget to sh is blocked"),
    # Reverse shells
    (r"\b/dev/tcp/", "TCP device access is blocked"),
    (r"\b/dev/udp/", "UDP device access is blocked"),
    (r"\bnc\s+-[a-zA-Z]*e\b", "Netcat with execute is blocked"),
    (r"\bncat\s+-[a-zA-Z]*e\b", "Ncat with execute is blocked"),
    # Privilege escalation
    (r"\bsudo\s+su\b", "sudo su is blocked"),
    (r"\bsudo\s+-i\b", "sudo -i is blocked"),
    (r"\bsudo\s+bash\b", "sudo bash is blocked"),
    # Cron manipulation
    (r"\bcrontab\s+-r\b", "Removing crontab is blocked"),
    (r"\bcrontab\s+-e\b", "Editing crontab is blocked"),
    # System file modification
    (r"\b(cat|echo|tee)\b.*>\s*/etc/passwd", "Modifying /etc/passwd is blocked"),
    (r"\b(cat|echo|tee)\b.*>\s*/etc/shadow", "Modifying /etc/shadow is blocked"),
    (r"\b(cat|echo|tee)\b.*>\s*/etc/sudoers", "Modifying /etc/sudoers is blocked"),
    # History manipulation (hiding tracks)
    (r"\bhistory\s+-c\b", "Clearing history is blocked"),
    (r"\bunset\s+HISTFILE\b", "Unsetting HISTFILE is blocked"),
    (r"\bexport\s+HISTSIZE=0\b", "Disabling history is blocked"),
    # Shutdown/reboot
    (r"\bshutdown\b", "Shutdown commands are blocked"),
    (r"\breboot\b", "Reboot commands are blocked"),
    (r"\bhalt\b", "Halt commands are blocked"),
    (r"\bpoweroff\b", "Poweroff commands are blocked"),
    (r"\binit\s+[06]\b", "Init runlevel changes are blocked"),
    # Package manager dangerous operations
    (r"\bapt\s+.*--purge\s+.*\*", "Wildcard purge is blocked"),
    (r"\byum\s+remove\s+\*", "Wildcard yum remove is blocked"),
    # Kernel module manipulation
    (r"\binsmod\b", "Loading kernel modules is blocked"),
    (r"\brmmod\b", "Removing kernel modules is blocked"),
    (r"\bmodprobe\b", "Modprobe is blocked"),
]

# Commands that are completely blocked
BLOCKED_COMMANDS: set[str] = {
    "mkfs",
    "fdisk",
    "parted",
    "dd",
    "shutdown",
    "reboot",
    "halt",
    "poweroff",
    "insmod",
    "rmmod",
    "modprobe",
    "iptables",
    "ip6tables",
    "nft",
    "ufw",
    "firewall-cmd",
}

# Paths that should never be deleted (rm operations)
# Note: Reading/navigating these paths is allowed
PROTECTED_DELETE_PATHS: list[str] = [
    "/",
    "/bin",
    "/boot",
    "/dev",
    "/etc",
    "/lib",
    "/lib64",
    "/proc",
    "/root",
    "/sbin",
    "/sys",
    "/usr",
]

# Critical system files that should never be written to
PROTECTED_WRITE_FILES: list[str] = [
    "/etc/passwd",
    "/etc/shadow",
    "/etc/sudoers",
    "/etc/hosts",
    "/etc/fstab",
    "/etc/ssh/sshd_config",
    "/boot/grub/grub.cfg",
]

# Track project directory for path validation
_project_dir: str | None = None


def set_project_dir(project_dir: str) -> None:
    """Set the project directory for path validation."""
    global _project_dir
    _project_dir = str(Path(project_dir).resolve())


def is_delete_protected(path: str) -> bool:
    """Check if a path is protected from deletion (rm).

    Args:
        path: The path to check.

    Returns:
        True if the path should not be deleted.
    """
    # Normalize the path
    normalized = str(Path(path))

    # Check against protected delete paths
    for protected in PROTECTED_DELETE_PATHS:
        if normalized == protected or normalized.startswith(protected + "/"):
            # Allow if it's within project directory
            return not (_project_dir and normalized.startswith(_project_dir))

    return False


def is_write_protected(path: str) -> bool:
    """Check if a file is protected from writes.

    Only blocks critical system files, not general directories.

    Args:
        path: The path to check.

    Returns:
        True if the file should not be written to.
    """
    # Normalize the path
    try:
        normalized = str(Path(path).resolve())
    except (OSError, ValueError):
        normalized = str(Path(path))

    # Check against protected write files
    for protected in PROTECTED_WRITE_FILES:
        if normalized == protected or normalized.endswith(protected):
            return True

    return False


def extract_commands(command_string: str) -> list[str]:
    """
    Extract command names from a shell command string.

    Handles pipes, command chaining (&&, ||, ;), subshells, and command substitution.
    Returns the base command names (without paths).

    Args:
        command_string: The full shell command.

    Returns:
        List of command names found in the string.
    """
    commands: list[str] = []

    # Extract commands from $() substitutions
    for match in re.finditer(r"\$\((\w+)", command_string):
        commands.append(match.group(1))

    # Extract commands from backtick substitutions
    for match in re.finditer(r"`(\w+)", command_string):
        commands.append(match.group(1))

    # Use shlex.split to properly handle quoted strings
    try:
        tokens = shlex.split(command_string)
    except ValueError:
        # Malformed command - return empty to trigger further validation
        return commands if commands else []

    if not tokens:
        return commands if commands else []

    # Shell operators that indicate a new command follows
    operators = {"|", "||", "&&", "&", ";"}

    # Shell keywords that precede commands
    shell_keywords = {
        "if",
        "then",
        "else",
        "elif",
        "fi",
        "for",
        "while",
        "until",
        "do",
        "done",
        "case",
        "esac",
        "in",
        "!",
        "{",
        "}",
    }

    expect_command = True

    for token in tokens:
        # Check for operators
        if token in operators:
            expect_command = True
            continue

        # Handle semicolon attached to end of token
        if token.endswith(";"):
            base_token = token[:-1]
            if (
                base_token
                and expect_command
                and not base_token.startswith("-")
                and ("=" not in base_token or base_token.startswith("="))
            ):
                cmd = Path(base_token).name
                commands.append(cmd)
            expect_command = True
            continue

        # Skip shell keywords
        if token in shell_keywords:
            continue

        # Skip flags/options
        if token.startswith("-"):
            continue

        # Skip variable assignments
        if "=" in token and not token.startswith("="):
            continue

        if expect_command:
            cmd = Path(token).name
            commands.append(cmd)
            expect_command = False

    return commands


def validate_rm_command(command_string: str) -> tuple[bool, str]:
    """
    Validate rm commands specifically.

    Args:
        command_string: The command string.

    Returns:
        Tuple of (is_allowed, reason_if_blocked).
    """
    # Check for recursive flags
    if re.search(r"\brm\s+.*-[a-zA-Z]*r", command_string):
        return False, "Recursive rm (-r, -R, --recursive) is blocked for safety"

    # Check for force removal of protected paths
    try:
        tokens = shlex.split(command_string)
        for i, token in enumerate(tokens):
            if token == "rm":
                # Check subsequent tokens for protected paths
                for path_token in tokens[i + 1 :]:
                    if not path_token.startswith("-") and is_delete_protected(
                        path_token
                    ):
                        return (
                            False,
                            f"rm on protected path '{path_token}' is blocked",
                        )
    except ValueError:
        pass

    return True, ""


def validate_chmod_chown_command(command_string: str) -> tuple[bool, str]:
    """
    Validate chmod/chown commands.

    Args:
        command_string: The command string.

    Returns:
        Tuple of (is_allowed, reason_if_blocked).
    """
    # Block recursive chmod/chown on protected paths
    if re.search(
        r"\b(chmod|chown)\s+-R\s+.*(/etc|/usr|/bin|/sbin|/lib)", command_string
    ):
        return False, "Recursive chmod/chown on system paths is blocked"

    return True, ""


async def bash_security_hook(
    input_data: HookInput | dict[str, Any],
    tool_use_id: str | None = None,
    context: HookContext | None = None,
) -> HookJSONOutput:
    """
    Pre-tool-use hook that validates bash commands for security.

    This hook blocks dangerous commands that could harm the system,
    while allowing normal development workflows.

    Args:
        input_data: Dict containing tool_name and tool_input.
        tool_use_id: Optional tool use ID.
        context: Optional hook context.

    Returns:
        Empty dict to allow, or blocking response to deny.
    """
    # Handle both HookInput and dict
    if isinstance(input_data, dict):
        tool_name = input_data.get("tool_name")
        tool_input = input_data.get("tool_input", {})
    else:
        tool_name = input_data.get("tool_name")
        tool_input = input_data.get("tool_input", {})

    # Only check Bash commands
    if tool_name != "Bash":
        return {}

    command = tool_input.get("command", "") if isinstance(tool_input, dict) else ""
    if not command:
        return {}

    # Check against dangerous patterns first
    for pattern, reason in DANGEROUS_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            logger.warning(
                "Blocked dangerous command: %s (reason: %s)", command, reason
            )
            return {
                "reason": reason,
                "systemMessage": f"Command blocked: {reason}",
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                },
            }

    # Extract and check individual commands
    commands = extract_commands(command)

    for cmd in commands:
        if cmd in BLOCKED_COMMANDS:
            reason = f"Command '{cmd}' is blocked for security reasons"
            logger.warning("Blocked command: %s", cmd)
            return {
                "reason": reason,
                "systemMessage": f"Command blocked: {reason}",
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                },
            }

    # Special validation for rm
    if "rm" in commands:
        allowed, reason = validate_rm_command(command)
        if not allowed:
            logger.warning("Blocked rm command: %s (reason: %s)", command, reason)
            return {
                "reason": reason,
                "systemMessage": f"Command blocked: {reason}",
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                },
            }

    # Special validation for chmod/chown
    if "chmod" in commands or "chown" in commands:
        allowed, reason = validate_chmod_chown_command(command)
        if not allowed:
            logger.warning(
                "Blocked chmod/chown command: %s (reason: %s)", command, reason
            )
            return {
                "reason": reason,
                "systemMessage": f"Command blocked: {reason}",
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                },
            }

    # Command passed all checks
    return {}


async def write_security_hook(
    input_data: HookInput | dict[str, Any],
    tool_use_id: str | None = None,
    context: HookContext | None = None,
) -> HookJSONOutput:
    """
    Pre-tool-use hook that validates Write/Edit operations.

    Blocks writes to system files and other protected locations.

    Args:
        input_data: Dict containing tool_name and tool_input.
        tool_use_id: Optional tool use ID.
        context: Optional hook context.

    Returns:
        Empty dict to allow, or blocking response to deny.
    """
    # Handle both HookInput and dict
    if isinstance(input_data, dict):
        tool_name = input_data.get("tool_name")
        tool_input = input_data.get("tool_input", {})
    else:
        tool_name = input_data.get("tool_name")
        tool_input = input_data.get("tool_input", {})

    # Only check Write and Edit commands
    if tool_name not in ("Write", "Edit", "MultiEdit"):
        return {}

    file_path = tool_input.get("file_path", "") if isinstance(tool_input, dict) else ""
    if not file_path:
        return {}

    # Check if file is a critical system file
    if is_write_protected(file_path):
        reason = f"Writing to critical system file '{file_path}' is blocked"
        logger.warning("Blocked write to protected file: %s", file_path)
        return {
            "reason": reason,
            "systemMessage": f"Write blocked: {reason}",
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            },
        }

    return {}


def get_security_hooks() -> dict[str, list[Any]]:
    """
    Get the security hooks configuration for ClaudeAgentOptions.

    Returns:
        Dict mapping hook events to hook matchers.
    """
    from claude_agent_sdk.types import HookMatcher

    return {
        "PreToolUse": [
            HookMatcher(matcher="Bash", hooks=[bash_security_hook]),
            HookMatcher(matcher="Write|Edit|MultiEdit", hooks=[write_security_hook]),
        ],
    }
