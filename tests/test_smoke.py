"""Every stage on tiny stand-in tensors (about 2 minutes): pytest --smoke."""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.smoke
def test_smoke():
    assert subprocess.run([sys.executable, str(ROOT / "main.py"), "smoke"]).returncode == 0
