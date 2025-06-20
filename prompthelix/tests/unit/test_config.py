import unittest
import os
import importlib
from unittest.mock import patch

# Import the config module from prompthelix to be tested
from prompthelix import config

class TestConfigSettings(unittest.TestCase):

    def setUp(self):
        # Store original environment variables that we might modify
        self.original_environ = os.environ.copy()

    def tearDown(self):
        # Restore original environment variables
        os.environ.clear()
        os.environ.update(self.original_environ)
        # Reload config to reset its state for subsequent tests
        importlib.reload(config)

    def test_default_knowledge_dir(self):
        """Test that KNOWLEDGE_DIR defaults correctly when env var is not set."""
        with patch.dict(os.environ, {}, clear=True):
            # Ensure KNOWLEDGE_DIR is not set
            if "KNOWLEDGE_DIR" in os.environ:
                del os.environ["KNOWLEDGE_DIR"]
            importlib.reload(config)
            self.assertEqual(config.KNOWLEDGE_DIR, "knowledge")

    def test_override_knowledge_dir(self):
        """Test that KNOWLEDGE_DIR is correctly overridden by env var."""
        custom_knowledge_dir = "/custom/knowledge/path"
        with patch.dict(os.environ, {"KNOWLEDGE_DIR": custom_knowledge_dir}, clear=True):
            importlib.reload(config)
            self.assertEqual(config.KNOWLEDGE_DIR, custom_knowledge_dir)

    def test_default_population_persistence_path(self):
        """Test DEFAULT_POPULATION_PERSISTENCE_PATH defaults correctly."""
        with patch.dict(os.environ, {}, clear=True):
            # Ensure relevant env vars are not set
            if "DEFAULT_POPULATION_PERSISTENCE_PATH" in os.environ:
                del os.environ["DEFAULT_POPULATION_PERSISTENCE_PATH"]
            if "KNOWLEDGE_DIR" in os.environ: # Control KNOWLEDGE_DIR for predictable default
                del os.environ["KNOWLEDGE_DIR"]

            importlib.reload(config) # Reload to apply unset env vars

            # KNOWLEDGE_DIR will default to "knowledge"
            expected_default_knowledge_dir = "knowledge"
            expected_path = os.path.join(expected_default_knowledge_dir, "ga_population.json")
            self.assertEqual(config.settings.DEFAULT_POPULATION_PERSISTENCE_PATH, expected_path)

    def test_default_population_persistence_path_with_custom_knowledge_dir(self):
        """Test DEFAULT_POPULATION_PERSISTENCE_PATH with custom KNOWLEDGE_DIR via env var."""
        custom_knowledge_dir = "my_custom_knowledge"
        with patch.dict(os.environ, {"KNOWLEDGE_DIR": custom_knowledge_dir}, clear=True):
            # Ensure DEFAULT_POPULATION_PERSISTENCE_PATH is not set
            if "DEFAULT_POPULATION_PERSISTENCE_PATH" in os.environ:
                del os.environ["DEFAULT_POPULATION_PERSISTENCE_PATH"]

            importlib.reload(config)

            expected_path = os.path.join(custom_knowledge_dir, "ga_population.json")
            self.assertEqual(config.settings.DEFAULT_POPULATION_PERSISTENCE_PATH, expected_path)

    def test_override_population_persistence_path(self):
        """Test DEFAULT_POPULATION_PERSISTENCE_PATH is overridden by env var."""
        custom_path = "/custom/persistence/path.json"
        with patch.dict(os.environ, {"DEFAULT_POPULATION_PERSISTENCE_PATH": custom_path}, clear=True):
            # KNOWLEDGE_DIR override does not matter if DEFAULT_POPULATION_PERSISTENCE_PATH is set directly
            if "KNOWLEDGE_DIR" in os.environ:
                del os.environ["KNOWLEDGE_DIR"]

            importlib.reload(config)
            self.assertEqual(config.settings.DEFAULT_POPULATION_PERSISTENCE_PATH, custom_path)

    def test_default_save_population_frequency(self):
        """Test DEFAULT_SAVE_POPULATION_FREQUENCY defaults to 10."""
        with patch.dict(os.environ, {}, clear=True):
            # Ensure relevant env var is not set
            if "DEFAULT_SAVE_POPULATION_FREQUENCY" in os.environ:
                del os.environ["DEFAULT_SAVE_POPULATION_FREQUENCY"]

            importlib.reload(config)
            self.assertEqual(config.settings.DEFAULT_SAVE_POPULATION_FREQUENCY, 10)

    def test_override_save_population_frequency(self):
        """Test DEFAULT_SAVE_POPULATION_FREQUENCY is overridden by env var."""
        custom_frequency = "25"
        with patch.dict(os.environ, {"DEFAULT_SAVE_POPULATION_FREQUENCY": custom_frequency}, clear=True):
            importlib.reload(config)
            self.assertEqual(config.settings.DEFAULT_SAVE_POPULATION_FREQUENCY, int(custom_frequency))

    def test_save_population_frequency_is_int(self):
        """Test that DEFAULT_SAVE_POPULATION_FREQUENCY is loaded as an int."""
        with patch.dict(os.environ, {"DEFAULT_SAVE_POPULATION_FREQUENCY": "15"}, clear=True):
            importlib.reload(config)
            self.assertIsInstance(config.settings.DEFAULT_SAVE_POPULATION_FREQUENCY, int)

if __name__ == '__main__':
    unittest.main()
