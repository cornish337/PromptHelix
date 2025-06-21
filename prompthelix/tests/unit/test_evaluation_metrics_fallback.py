import builtins
import importlib
import sys
import unittest

class TestMetricsFallback(unittest.TestCase):
    def setUp(self):
        self.real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "textstat":
                raise ImportError("No textstat")
            return self.real_import(name, globals, locals, fromlist, level)

        builtins.__import__ = fake_import

        # Ensure a clean import state for the metrics module and textstat
        if 'prompthelix.evaluation.metrics' in sys.modules:
            del sys.modules['prompthelix.evaluation.metrics']
        if 'textstat' in sys.modules:
            del sys.modules['textstat']

        # Import the module; it will be processed by fake_import
        from prompthelix.evaluation import metrics as freshly_imported_metrics

        # No real need to reload again if it's freshly imported after fake_import is set up.
        # The import itself will trigger the textstat import attempt.
        self.metrics = freshly_imported_metrics

    def tearDown(self):
        builtins.__import__ = self.real_import

        module_name = 'prompthelix.evaluation.metrics'

        # Remove the module that might have been affected by fake_import
        if module_name in sys.modules:
            del sys.modules[module_name]

        # Import it fresh using the original import mechanism
        # This ensures it's available for other tests in a clean state
        import prompthelix.evaluation.metrics

        # Reload it to be absolutely sure it's initialized correctly,
        # as just importing might return the version from before del if not careful,
        # though del then import should be clean. Reload is for safety.
        if module_name in sys.modules:
            # Get the definitive reference from sys.modules after import
            module_to_reload = sys.modules[module_name]
            importlib.reload(module_to_reload)
        else:
            # This would be unexpected if the import above worked
            logging.warning(f"Module {module_name} was not found in sys.modules after attempting re-import in tearDown.")


    def test_fallback_metrics(self):
        self.assertIsNone(self.metrics.textstat)
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
