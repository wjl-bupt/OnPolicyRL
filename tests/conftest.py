"""Pytest bootstrap: put `src/` on sys.path.

The rebuild uses namespace packages (no __init__.py), so `common.*` is importable
only with src/ on the path; every test module imports through this.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
