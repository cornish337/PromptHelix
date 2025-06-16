import unittest
import tempfile
import os
import json
from unittest.mock import patch, mock_open # For MetaLearner, logging checks

# Import Agents
from prompthelix.agents.domain_expert import DomainExpertAgent
from prompthelix.agents.architect import PromptArchitectAgent
from prompthelix.agents.critic import PromptCriticAgent
from prompthelix.agents.style_optimizer import StyleOptimizerAgent
from prompthelix.agents.results_evaluator import ResultsEvaluatorAgent
from prompthelix.agents.meta_learner import MetaLearnerAgent
from prompthelix.message_bus import MessageBus # MetaLearner needs a bus

class TestAgentPersistence(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory to store knowledge files
        self.temp_dir = tempfile.TemporaryDirectory()
        # MetaLearnerAgent requires a message bus
        self.mock_bus = MessageBus()

    def tearDown(self):
        # Clean up the temporary directory
        self.temp_dir.cleanup()

    def _get_temp_filepath(self, filename_suffix):
        return os.path.join(self.temp_dir.name, f"test_{filename_suffix}.json")

    # Generic test methods to be reused by specific agent tests where possible
    def _test_default_load_and_initial_save(self, agent_class, default_knowledge_attr, **kwargs):
        temp_file = self._get_temp_filepath(f"{agent_class.__name__}_defaults")

        # Ensure file does not exist initially for a clean test of initial save behavior
        if os.path.exists(temp_file):
            os.remove(temp_file)

        agent = agent_class(knowledge_file_path=temp_file, **kwargs)
        default_knowledge = getattr(agent, default_knowledge_attr)
        self.assertTrue(default_knowledge, f"Default knowledge for {agent_class.__name__} should not be empty.")

        # Check if save_knowledge was called (file should now exist with default content)
        self.assertTrue(os.path.exists(temp_file), f"Knowledge file {temp_file} should have been created with defaults.")
        with open(temp_file, 'r') as f:
            loaded_from_file = json.load(f)
        self.assertEqual(default_knowledge, loaded_from_file, f"File content should match default knowledge for {agent_class.__name__}.")

    def _test_save_and_reload_modified_knowledge(self, agent_class, knowledge_attr, modification_func, **kwargs):
        temp_file = self._get_temp_filepath(f"{agent_class.__name__}_modified")

        agent1 = agent_class(knowledge_file_path=temp_file, **kwargs)

        # Perform modification
        modification_func(getattr(agent1, knowledge_attr))
        modified_knowledge_agent1 = getattr(agent1, knowledge_attr)

        agent1.save_knowledge()

        agent2 = agent_class(knowledge_file_path=temp_file, **kwargs)
        reloaded_knowledge_agent2 = getattr(agent2, knowledge_attr)

        self.assertEqual(modified_knowledge_agent1, reloaded_knowledge_agent2,
                         f"Reloaded knowledge should match modified knowledge for {agent_class.__name__}.")

    def _test_corrupted_file_fallback(self, agent_class, default_knowledge_attr, **kwargs):
        temp_file = self._get_temp_filepath(f"{agent_class.__name__}_corrupted")

        # Create a dummy valid file first to ensure agent would normally load it
        agent_dummy = agent_class(knowledge_file_path=temp_file, **kwargs)
        agent_dummy.save_knowledge() # Save defaults

        # Now corrupt the file
        with open(temp_file, 'w') as f:
            f.write("this is not valid json")

        with patch('logging.error') as mock_log_error:
            agent = agent_class(knowledge_file_path=temp_file, **kwargs)
            default_knowledge = getattr(agent, default_knowledge_attr)
            original_default_knowledge = agent._get_default_knowledge() if hasattr(agent, '_get_default_knowledge') else \
                                         agent._get_default_templates() if hasattr(agent, '_get_default_templates') else \
                                         agent._get_default_critique_rules() if hasattr(agent, '_get_default_critique_rules') else \
                                         agent._get_default_style_rules() if hasattr(agent, '_get_default_style_rules') else \
                                         agent._get_default_metrics_config() if hasattr(agent, '_get_default_metrics_config') else {} # Add MetaLearner specific one if needed

            self.assertEqual(default_knowledge, original_default_knowledge,
                             f"Agent {agent_class.__name__} should fall back to default knowledge on corrupted file.")
            mock_log_error.assert_called() # Check that an error was logged

    # --- DomainExpertAgent Tests ---
    def test_dea_default_load(self):
        self._test_default_load_and_initial_save(DomainExpertAgent, 'knowledge_base')

    def test_dea_save_reload_modified(self):
        def modify_dea_knowledge(kb):
            kb["new_domain"] = {"keywords": ["test1"]}
        self._test_save_and_reload_modified_knowledge(DomainExpertAgent, 'knowledge_base', modify_dea_knowledge)

    def test_dea_corrupted_fallback(self):
        self._test_corrupted_file_fallback(DomainExpertAgent, 'knowledge_base')

    # --- PromptArchitectAgent Tests ---
    def test_paa_default_load(self):
        self._test_default_load_and_initial_save(PromptArchitectAgent, 'templates')

    def test_paa_save_reload_modified(self):
        def modify_paa_templates(tpl):
            tpl["new_template_test"] = {"instruction": "test instruction"}
        self._test_save_and_reload_modified_knowledge(PromptArchitectAgent, 'templates', modify_paa_templates)

    def test_paa_corrupted_fallback(self):
        self._test_corrupted_file_fallback(PromptArchitectAgent, 'templates')

    # --- PromptCriticAgent Tests ---
    def test_pca_default_load(self):
        self._test_default_load_and_initial_save(PromptCriticAgent, 'critique_rules')

    def test_pca_save_reload_modified(self):
        def modify_pca_rules(rules):
            if isinstance(rules, list) and rules: # Default rules are a list
                 rules[0]["message"] = "Modified test message"
            else: # If it became a dict somehow or empty
                rules.append({"name": "TestRule", "message": "Modified test message"})

        self._test_save_and_reload_modified_knowledge(PromptCriticAgent, 'critique_rules', modify_pca_rules)

    def test_pca_corrupted_fallback(self):
        self._test_corrupted_file_fallback(PromptCriticAgent, 'critique_rules')

    # --- StyleOptimizerAgent Tests ---
    def test_soa_default_load(self):
        self._test_default_load_and_initial_save(StyleOptimizerAgent, 'style_rules')

    def test_soa_save_reload_modified(self):
        def modify_soa_rules(rules):
            rules["new_style_test"] = {"replace": {"old": "new"}}
        self._test_save_and_reload_modified_knowledge(StyleOptimizerAgent, 'style_rules', modify_soa_rules)

    def test_soa_corrupted_fallback(self):
        self._test_corrupted_file_fallback(StyleOptimizerAgent, 'style_rules')

    # --- ResultsEvaluatorAgent Tests ---
    def test_rea_default_load(self):
        self._test_default_load_and_initial_save(ResultsEvaluatorAgent, 'evaluation_metrics_config')

    def test_rea_save_reload_modified(self):
        def modify_rea_config(config):
            config["new_metric_test"] = "test_value"
        self._test_save_and_reload_modified_knowledge(ResultsEvaluatorAgent, 'evaluation_metrics_config', modify_rea_config)

    def test_rea_corrupted_fallback(self):
        self._test_corrupted_file_fallback(ResultsEvaluatorAgent, 'evaluation_metrics_config')

    # --- MetaLearnerAgent Tests ---
    # MetaLearnerAgent's _get_default_knowledge initializes knowledge_base with specific keys.
    # Its load_knowledge also initializes these keys if they are missing after loading from file.
    # So, the "default_knowledge_attr" is 'knowledge_base', but the comparison needs to be against
    # what _get_default_knowledge() returns.

    def test_mla_default_load(self):
         self._test_default_load_and_initial_save(MetaLearnerAgent, 'knowledge_base', message_bus=self.mock_bus)

    def test_mla_save_reload_modified(self):
        def modify_mla_knowledge(kb):
            kb["successful_prompt_features"]["new_feature_test"] = 10
        self._test_save_and_reload_modified_knowledge(MetaLearnerAgent, 'knowledge_base', modify_mla_knowledge, message_bus=self.mock_bus)

    def test_mla_corrupted_fallback(self):
        # Special handling for MetaLearnerAgent's default knowledge structure
        temp_file = self._get_temp_filepath(f"{MetaLearnerAgent.__name__}_corrupted_mla")
        agent_dummy = MetaLearnerAgent(knowledge_file_path=temp_file, message_bus=self.mock_bus)
        agent_dummy.save_knowledge()

        with open(temp_file, 'w') as f:
            f.write("this is not valid json")

        with patch('logging.error') as mock_log_error:
            agent = MetaLearnerAgent(knowledge_file_path=temp_file, message_bus=self.mock_bus)
            # Compare with the structure _get_default_knowledge provides
            expected_default_kb = agent._get_default_knowledge()
            self.assertEqual(agent.knowledge_base, expected_default_kb,
                             "MetaLearnerAgent should fall back to default knowledge structure on corrupted file.")
            mock_log_error.assert_called()

if __name__ == '__main__':
    unittest.main()
