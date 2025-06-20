
import pytest
from prompthelix.tests.conftest import *  # re-export fixtures for root tests

@pytest.fixture(autouse=True)
def mock_llm_api(monkeypatch):
    """Return a constant response for all llm_api calls during tests."""
    def _mock_call(*args, **kwargs):
        return "MOCK_RESPONSE"

    # Patch the central utility
    monkeypatch.setattr("prompthelix.utils.llm_utils.call_llm_api", _mock_call)
    
    # Also patch already-imported agent modules
    modules = [
        "prompthelix.agents.architect",
        "prompthelix.agents.domain_expert",
        "prompthelix.agents.results_evaluator",
        "prompthelix.agents.style_optimizer",
    ]
    for mod in modules:
        monkeypatch.setattr(f"{mod}.call_llm_api", _mock_call, raising=False)


