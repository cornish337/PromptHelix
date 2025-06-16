import argparse
from prompthelix.utils.llm_utils import call_llm_api


def main():
    parser = argparse.ArgumentParser(description="Test connectivity to an LLM provider")
    parser.add_argument("--provider", default="openai", help="LLM provider name")
    parser.add_argument("--model", help="Model name for the provider")
    args = parser.parse_args()

    test_prompt = "Connectivity test"
    try:
        response = call_llm_api(prompt=test_prompt, provider=args.provider, model=args.model)
        print(response)
    except Exception as exc:
        print(f"ERROR: {exc}")


if __name__ == "__main__":
    main()
