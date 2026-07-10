"""Small, dependency-free helpers for safely persisting project data."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_json(path: str | os.PathLike[str], data: Any) -> None:
    """Write JSON next to *path* and atomically replace the destination.

    Keeping the temporary file in the destination directory makes ``os.replace``
    atomic on the filesystems normally used by the project.  A failed dump or
    replace therefore leaves the previous JSON file untouched.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path: str | None = None
    existing_mode = None
    try:
        existing_mode = destination.stat().st_mode & 0o777
    except FileNotFoundError:
        pass

    try:
        fd, temp_path = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        )
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        if existing_mode is not None:
            os.chmod(temp_path, existing_mode)
        os.replace(temp_path, destination)
        temp_path = None
    finally:
        if temp_path is not None:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass
