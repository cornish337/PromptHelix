from .interactive_runner import require_interactive, prompt_user


def test_basic_flow():
    """Simple interactive check requiring the user to confirm input."""
    require_interactive()
    answer = prompt_user("Type 'ok' to pass this test: ")
    assert answer is not None
    assert answer.strip().lower() == "ok"
