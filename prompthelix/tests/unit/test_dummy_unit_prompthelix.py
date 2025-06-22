import unittest

class TestDummyUnitPromptHelix(unittest.TestCase):
    def test_prompthelix_unit_pass(self):
        """A dummy unit test in 'prompthelix/tests/unit' that passes."""
        self.assertEqual(5 * 5, 25)

    def test_prompthelix_unit_another_pass(self):
        """Another dummy unit test in 'prompthelix/tests/unit' that also passes."""
        self.assertIn("hello", "hello world")

if __name__ == '__main__':
    unittest.main()
