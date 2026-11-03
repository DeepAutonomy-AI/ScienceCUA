"""Filesystem paths for PDF staging and recovery resources."""

from __future__ import annotations

import os
from pathlib import Path


def filesystem_path(value: str | Path) -> Path:
    """Resolve a path and enable long filenames on Windows.

    PDF assets use full SHA-256 filenames. Even a valid run directory can make
    their staging paths exceed Windows' legacy 260-character limit. The
    extended namespace also lets temporary-directory cleanup reach these files.
    """
    path = Path(value).resolve()
    if os.name != "nt":
        return path
    absolute = str(path)
    if absolute.startswith("\\\\?\\"):
        return path
    if absolute.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + absolute[2:])
    return Path("\\\\?\\" + absolute)
