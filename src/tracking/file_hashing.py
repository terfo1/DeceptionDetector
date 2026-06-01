"""File hashing and metadata helpers."""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

import pandas as pd


def calculate_file_hash(path: str) -> str:
    """Calculate SHA256 for a file, returning an empty string if missing."""
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return ""
    hasher = hashlib.sha256()
    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_file_size(path: str) -> int:
    """Return file size in bytes, or 0 if missing."""
    file_path = Path(path)
    return int(file_path.stat().st_size) if file_path.exists() and file_path.is_file() else 0


def get_file_modified_time(path: str) -> str:
    """Return ISO modified time, or empty string if missing."""
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return ""
    return datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(timespec="seconds")


def collect_file_metadata(paths: list[str]) -> pd.DataFrame:
    """Collect existence, size, modified time, and SHA256 for paths."""
    rows = []
    for path in paths:
        file_path = Path(path)
        exists = file_path.exists() and file_path.is_file()
        rows.append(
            {
                "path": path,
                "exists": exists,
                "size_bytes": get_file_size(path),
                "modified_at": get_file_modified_time(path),
                "sha256": calculate_file_hash(path),
            }
        )
    return pd.DataFrame(rows, columns=["path", "exists", "size_bytes", "modified_at", "sha256"])
