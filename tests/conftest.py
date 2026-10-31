"""Test setup: import the code from src/ without installing anything; `pytest --smoke` adds the 2-minute smoke run."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def pytest_addoption(parser):
    parser.addoption("--smoke", action="store_true", help="also run every stage on stand-in data (about 2 minutes)")


def pytest_configure(config):
    config.addinivalue_line("markers", "smoke: runs every stage on stand-in data; enabled with --smoke")
    config.addinivalue_line("filterwarnings", "ignore::DeprecationWarning")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--smoke"):
        return
    skip = pytest.mark.skip(reason="smoke run: use pytest --smoke")
    for item in items:
        if "smoke" in item.keywords:
            item.add_marker(skip)
