"""Root conftest – makes ``smart_contract_rag`` importable from ``staging/src``."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure ``src/`` is on the import path so ``import smart_contract_rag`` works
# regardless of where pytest is invoked from.
_src = str(Path(__file__).resolve().parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)
