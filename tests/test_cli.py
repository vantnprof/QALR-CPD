"""The command-line interface."""
import pytest

from src.cli import COMMANDS, STAGES, main


def test_help_lists_every_command(capsys):
    main(["--help"])
    out = capsys.readouterr().out
    for cmd in COMMANDS:
        assert cmd in out


def test_run_help_lists_every_stage(capsys):
    with pytest.raises(SystemExit) as e:
        main(["run", "--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    for stage in STAGES:
        assert stage in out


def test_every_stage_exists():
    from src.experiments.stages import STAGES as IMPL

    for stage in STAGES:
        assert stage in IMPL or stage == "qaoa-bank"


def test_unknown_command():
    with pytest.raises(SystemExit):
        main(["no-such-command"])
