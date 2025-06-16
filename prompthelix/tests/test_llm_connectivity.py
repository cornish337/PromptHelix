from prompthelix.utils.llm_utils import call_llm_api
import pytest
import argparse

def test_llm_connectivity(llm_provider: str, model: str):
    """
    Tests connectivity to the LLM API and checks if a response is received.
    """
    prompt = "Hello, this is a test."
    try:
        response = call_llm_api(prompt, llm_provider, model)
        print(f"LLM Provider: {llm_provider}")
        print(f"Model: {model}")
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
        assert response is not None, "Response should not be None"
        assert response != "", "Response should not be empty"
    except Exception as e:
        print(f"Error during LLM API call: {e}")
        pytest.fail(f"LLM API call failed for {llm_provider} - {model}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Tests connectivity to a specified LLM provider and model.",
        epilog="""Examples:
  python test_llm_connectivity.py --provider openai --model gpt-3.5-turbo
  python test_llm_connectivity.py --provider claude --model claude-2"""
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        help="LLM provider (e.g., openai, claude). This specifies which LLM provider to use.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-3.5-turbo",
        help="LLM model name (e.g., gpt-3.5-turbo, claude-2). This specifies which model of the provider to use.",
    )
    args = parser.parse_args()
    test_llm_connectivity(args.provider, args.model)
