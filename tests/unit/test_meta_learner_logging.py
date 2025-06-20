import unittest
import sys
import types
import importlib

class TestMetaLearnerLogging(unittest.TestCase):
    def setUp(self):
        # Stub modules required by meta_learner to avoid heavy dependencies
        self.original_modules = {}
        stub_base = types.ModuleType('prompthelix.agents.base')
        class BaseAgent:
            def __init__(self, agent_id, message_bus=None, settings=None):
                self.agent_id = agent_id
                self.message_bus = message_bus
                self.settings = settings or {}
        stub_base.BaseAgent = BaseAgent
        self._inject_module('prompthelix.agents.base', stub_base)

        stub_engine = types.ModuleType('prompthelix.genetics.engine')
        class PromptChromosome:
            def __init__(self, genes=None):
                self.genes = genes or []
                self.id = 'test'
        stub_engine.PromptChromosome = PromptChromosome
        self._inject_module('prompthelix.genetics.engine', stub_engine)

        stub_llm_utils = types.ModuleType('prompthelix.utils.llm_utils')
        def call_llm_api(prompt, provider=None, model=None):
            return "[]"
        stub_llm_utils.call_llm_api = call_llm_api
        self._inject_module('prompthelix.utils.llm_utils', stub_llm_utils)

        stub_config = types.ModuleType('prompthelix.config')
        stub_config.AGENT_SETTINGS = {"MetaLearnerAgent": {}}
        stub_config.KNOWLEDGE_DIR = "."
        self._inject_module('prompthelix.config', stub_config)

        self.meta_module = importlib.import_module('prompthelix.agents.meta_learner')
        self.MetaLearnerAgent = self.meta_module.MetaLearnerAgent
        # Avoid file operations during tests
        self.original_load = self.MetaLearnerAgent.load_knowledge
        self.MetaLearnerAgent.load_knowledge = lambda self: None

    def tearDown(self):
        # Restore patched modules and methods
        self.MetaLearnerAgent.load_knowledge = self.original_load
        for name, mod in self.original_modules.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod

    def _inject_module(self, name, module):
        self.original_modules[name] = sys.modules.get(name)
        sys.modules[name] = module

    def test_update_knowledge_base_logging(self):
        agent = self.MetaLearnerAgent(knowledge_file_path="dummy.json")
        agent.persist_on_update = False
        with self.assertLogs(self.meta_module.__name__, level='DEBUG') as cm:
            agent._update_knowledge_base('successful_prompt_features', {'feature': 'log_test'})
        self.assertTrue(any('log_test' in message for message in cm.output))


if __name__ == '__main__':
    unittest.main()
