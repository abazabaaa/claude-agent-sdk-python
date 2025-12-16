"""
Claude Agent SDK Harness
========================

A harness for running Claude Code in an infinite loop with:
- Rich console visualization
- Security hooks
- Opus 4.5 with 64k thinking tokens

Usage:
    python -m harness.runner --prompt "Your task here"
"""

from .console_renderer import ConsoleRenderer
from .security_hooks import (
    bash_security_hook,
    get_security_hooks,
    is_delete_protected,
    is_write_protected,
    set_project_dir,
    write_security_hook,
)

__all__ = [
    "ConsoleRenderer",
    "bash_security_hook",
    "write_security_hook",
    "get_security_hooks",
    "set_project_dir",
    "is_delete_protected",
    "is_write_protected",
]
