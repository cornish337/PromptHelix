import unittest

class InteractiveTestCase(unittest.TestCase):
    """Base class for interactive tests.

    Provides helper methods for prompting the user or simulating simple UI
    actions. These tests are meant to be run manually and may pause to wait
    for user interaction.
    """

    def prompt(self, message: str) -> str:
        """Display a message and return the user's input."""
        return input(message)

    def confirm(self, message: str) -> bool:
        """Prompt the user with a yes/no question and return True if confirmed."""
        response = input(f"{message} [y/N]: ").strip().lower()
        return response in {"y", "yes"}

    def click(self, description: str) -> None:
        """Simulate clicking a UI element by waiting for ENTER."""
        input(f"Press ENTER to {description}...")

