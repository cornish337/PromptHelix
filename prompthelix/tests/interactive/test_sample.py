from .base import InteractiveTestCase

class TestSampleInteractive(InteractiveTestCase):
    """Example interactive test using InteractiveTestCase."""

    def test_user_confirmation(self):
        name = self.prompt("Enter your name: ")
        if self.confirm(f"Did you enter '{name}' correctly?"):
            print("Thanks for confirming!")
        else:
            self.fail("User did not confirm input")

