"""Configurable runtime data directories.

`data_dir` in config (or FACETRAK_DATA env) routes snapshots,
recordings and CSV exports to a persistent volume. Default is the
process working directory (backward compatible).
"""

import os
from pathlib import Path


def data_dir(cfg: dict) -> Path:
    raw = os.environ.get("FACETRAK_DATA", "") or cfg.get("data_dir", "") or "."
    path = Path(raw).expanduser()
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        return Path(".")
    return path
