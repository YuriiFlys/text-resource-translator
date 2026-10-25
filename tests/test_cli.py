from __future__ import annotations

from typer.testing import CliRunner

from trt import __version__
from trt.cli import app

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_help_lists_all_pipeline_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in (
        "check",
        "migrate",
        "seed",
        "fetch",
        "ingest",
        "embed",
        "search",
        "translate",
        "run-all",
        "schedule",
    ):
        assert cmd in result.output, cmd
