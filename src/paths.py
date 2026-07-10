"""Project-relative paths that do not depend on the process working directory."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def data_path(filename: str) -> Path:
    return DATA_DIR / filename
