"""Tests for fedmaq-lit CLI."""

from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

from fedmaq_literature.cli import main


def test_cli_help() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0


def test_cli_convert_missing_args() -> None:
    code = main(["convert"])
    assert code == 1


@patch("fedmaq_literature.cli._cmd_convert")
def test_cli_convert_slug(mock_convert: MagicMock) -> None:
    mock_convert.return_value = 0
    code = main(["convert", "--slug", "some-slug"])
    assert code == 0
    mock_convert.assert_called_once()
    args = mock_convert.call_args[0][0]
    assert args.slug == "some-slug"
    assert not args.all


@patch("fedmaq_literature.cli._cmd_convert")
def test_cli_convert_all(mock_convert: MagicMock) -> None:
    mock_convert.return_value = 0
    code = main(["convert", "--all"])
    assert code == 0
    mock_convert.assert_called_once()
    args = mock_convert.call_args[0][0]
    assert args.all
    assert args.slug is None
