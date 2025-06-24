import unittest

class TestDummyInteractiveGeneral(unittest.TestCase):
    def test_general_interactive_pass(self):
        """A dummy interactive test in the general 'tests/interactive' directory that passes."""
        self.assertEqual(1, 1)

    def test_general_interactive_fail(self):
        """A dummy interactive test in the general 'tests/interactive' directory that fails."""
        self.assertEqual(1, 0, "This test is designed to fail.")

if __name__ == '__main__':
    unittest.main()
