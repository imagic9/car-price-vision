"""Make the src-layout package importable without requiring `pip install -e .`.

Inserts src/ onto sys.path so `import car_price_vision...` works when running
`pytest` straight from a repo checkout. Also inserts the repo root so
`import serving.app` works (serving/ has no __init__.py, so this relies on
Python 3's implicit namespace packages -- it just needs the repo root on
sys.path, same as running `uvicorn serving.app:app` from there).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
