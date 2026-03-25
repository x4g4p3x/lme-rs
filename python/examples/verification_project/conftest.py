"""Allow `import parity` / `import paths` when running pytest from the `python/` tree."""

import sys
from pathlib import Path

_dir = Path(__file__).resolve().parent
if str(_dir) not in sys.path:
    sys.path.insert(0, str(_dir))
