"""Minimal .env loader (no third-party dependency).

Rules: ``KEY=VALUE`` lines only; ``#`` comments and blank lines skipped;
``export `` prefix tolerated; single/double quotes stripped; existing
environment variables win unless ``override=True``. Unquoted trailing
`` # comment`` is stripped; quoted values keep everything inside.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

_KEY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def load_dotenv(path: str | Path = ".env", override: bool = False) -> dict:
    """Parse a .env file into ``os.environ``; return what was set."""
    found: dict = {}
    p = Path(path)
    if not p.exists():
        return found
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if not _KEY_RE.fullmatch(key):
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].rstrip()
        if override or key not in os.environ:
            os.environ[key] = value
            found[key] = value
    return found
