import importlib
import sys
import unittest
import logging
# Removed builtins import as we'll use sys.modules manipulation

# Import the module that contains textstat so we can reload it.
from prompthelix.evaluation import metrics as evaluation_metrics_module

class TestMetricsFallback(unittest.TestCase):

    def setUp(self):
        self.metrics_module_name = 'prompthelix.evaluation.metrics' # Used for logging
        self.textstat_module_name = 'textstat'

        # Store original textstat if it's already loaded
        self.original_textstat_module = sys.modules.get(self.textstat_module_name)

        # Ensure textstat is not in sys.modules to simulate ImportError on next import/reload
        if self.textstat_module_name in sys.modules:
            del sys.modules[self.textstat_module_name]

        # Reload the metrics module. During this reload, 'import textstat' should fail.
        # The try-except block within metrics.py should then set its local 'textstat' to None.
        importlib.reload(evaluation_metrics_module)
        self.metrics = evaluation_metrics_module

    def tearDown(self):
        # Restore textstat to sys.modules if it was originally there
        if self.original_textstat_module is not None:
            sys.modules[self.textstat_module_name] = self.original_textstat_module
        elif self.textstat_module_name in sys.modules: # If it somehow got added back during test (shouldn't happen)
            del sys.modules[self.textstat_module_name]

        # Reload the metrics module again to ensure it's in a clean state for other tests
        # (textstat should now import successfully if it's installed in the environment)
        try:
            importlib.reload(evaluation_metrics_module)
        except ImportError: # pragma: no cover
            # This might happen if textstat was never installed and self.original_textstat_module was None
            logging.warning(f"Could not reload {self.metrics_module_name} in tearDown, possibly textstat is not installed.")


    def test_fallback_metrics(self):
        # After reload in setUp, evaluation_metrics_module.textstat should be None
        self.assertIsNone(self.metrics.textstat,
                          f"self.metrics.textstat should be None after simulating ImportError. Value: {self.metrics.textstat}")
        clarity = self.metrics.calculate_clarity_score("A simple sentence.")
        specificity = self.metrics.calculate_specificity_score("Explain quantum mechanics.")
        self.assertIsInstance(clarity, float)
        self.assertIsInstance(specificity, float)
        self.assertGreaterEqual(clarity, 0.0)
        self.assertLessEqual(clarity, 1.0)
        self.assertGreaterEqual(specificity, 0.0)
        self.assertLessEqual(specificity, 1.0)


if __name__ == "__main__":
    unittest.main()
