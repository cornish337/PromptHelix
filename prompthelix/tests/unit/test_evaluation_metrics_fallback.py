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
        sys.modules.pop('prompthelix.evaluation.metrics', None)
        from prompthelix.evaluation import metrics as metrics_mod
        importlib.reload(metrics_mod)
        self.metrics = metrics_mod

    def tearDown(self):
        builtins.__import__ = self.real_import
        importlib.reload(self.metrics)

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
