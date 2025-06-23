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
        self.textstat_was_present = self.textstat_module_name in sys.modules

        # Temporarily make textstat unimportable or appear as None
        sys.modules[self.textstat_module_name] = None # This will cause `import textstat` to import None

        # Reload the metrics module. During this reload, 'import textstat' will get None,
        # and the try-except block in metrics.py should catch an error if it tries to use None as a module,
        # or the `textstat = None` line in the except block will ensure its internal textstat is None.
        try:
            importlib.reload(evaluation_metrics_module)
        except Exception as e:
            # This reload might fail if textstat=None causes issues deeper in metrics.py's import-time code
            # For now, we assume the try-except in metrics.py handles it gracefully.
            logging.error(f"Error reloading metrics module during setUp: {e}")
        self.metrics = evaluation_metrics_module

    def tearDown(self):
        # Restore textstat to sys.modules
        if self.textstat_was_present:
            if self.original_textstat_module is not None:
                sys.modules[self.textstat_module_name] = self.original_textstat_module
            else:
                # This case should ideally not happen if textstat_was_present is true,
                # but as a safeguard:
                if self.textstat_module_name in sys.modules: # pragma: no cover
                    del sys.modules[self.textstat_module_name]
        else: # textstat was not originally in sys.modules
            if self.textstat_module_name in sys.modules:
                 del sys.modules[self.textstat_module_name]


        # Reload the metrics module again to ensure it's in a clean state for other tests
        try:
            importlib.reload(evaluation_metrics_module)
        except Exception: # pragma: no cover
            logging.warning(f"Could not reload {self.metrics_module_name} in tearDown.")


    def test_fallback_metrics(self):
        # After reload in setUp, evaluation_metrics_module.textstat should be None
        # due to the try-except block in metrics.py
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
