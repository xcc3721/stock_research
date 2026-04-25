from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def display_path(value: str | Path | None) -> str:
    if value is None:
        return ""
    path = Path(value).expanduser()
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)
