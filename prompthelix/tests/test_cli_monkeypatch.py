import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

from prompthelix import cli


def run_cli(args, monkeypatch):
    exit_code = None
    monkeypatch.setattr(sys, "argv", ["prompthelix"] + args)
    try:
        cli.main_cli()
    except SystemExit as e:
        exit_code = e.code
    return exit_code


def test_cli_test_command(monkeypatch, capsys):
    mock_loader = MagicMock()
    mock_runner = MagicMock()
    mock_result = MagicMock()
    mock_result.wasSuccessful.return_value = True
    mock_runner.run.return_value = mock_result
    monkeypatch.setattr(cli, "unittest", SimpleNamespace(TestLoader=lambda: mock_loader, TextTestRunner=lambda verbosity: mock_runner))
    mock_loader.discover.return_value = "suite"
    exit_code = run_cli(["test"], monkeypatch)
    captured = capsys.readouterr().out
    assert exit_code == 0
    assert "CLI: Running all tests..." in captured


def test_cli_test_command_custom_path(monkeypatch, capsys):
    mock_loader = MagicMock()
    mock_runner = MagicMock()
    mock_result = MagicMock()
    mock_result.wasSuccessful.return_value = True
    mock_runner.run.return_value = mock_result
    monkeypatch.setattr(
        cli,
        "unittest",
        SimpleNamespace(TestLoader=lambda: mock_loader, TextTestRunner=lambda verbosity: mock_runner),
    )
    mock_loader.discover.return_value = "suite"
    exit_code = run_cli(["test", "--path", "tests/unit"], monkeypatch)
    captured = capsys.readouterr().out
    assert exit_code == 0
    mock_loader.discover.assert_called_with(start_dir="tests/unit")


def test_cli_run_command(monkeypatch, capsys):
    mock_loop = MagicMock(return_value=SimpleNamespace(fitness_score=1.0, to_prompt_string=lambda: "best"))
    monkeypatch.setattr("prompthelix.orchestrator.main_ga_loop", mock_loop)
    exit_code = run_cli(["run", "ga"], monkeypatch)
    captured = capsys.readouterr().out
    assert exit_code is None or exit_code == 0
    assert "Best prompt fitness" in captured


def test_cli_check_llm(monkeypatch, capsys):
    from prompthelix.utils import llm_utils
    monkeypatch.setattr(llm_utils, "call_llm_api", MagicMock(return_value="ok"))
    exit_code = run_cli(["check-llm"], monkeypatch)
    captured = capsys.readouterr().out
    assert exit_code is None or exit_code == 0
    assert "ok" in captured
