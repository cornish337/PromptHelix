import sys
import pytest


def is_interactive() -> bool:
    """Return True if the session is connected to a TTY."""
    return sys.stdin.isatty() and sys.stdout.isatty()


def require_interactive():
    """Skip the calling test when not running interactively."""
    if not is_interactive():
        pytest.skip("Interactive test skipped: no TTY available")


def prompt_user(text: str, default: str | None = None) -> str | None:
    """Prompt the user for input if interactive, otherwise return ``default``."""
    if not is_interactive():
        return default
    return input(text)
