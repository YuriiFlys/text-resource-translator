from __future__ import annotations

from typer.testing import CliRunner

from trt import __version__
from trt.cli import app

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_unimplemented_commands_exit_2():
    for args in (["ingest"], ["embed"], ["search", "x"], ["translate", "--locale", "uk"]):
        result = runner.invoke(app, args)
        assert result.exit_code == 2, args
        assert "Not implemented" in result.output
