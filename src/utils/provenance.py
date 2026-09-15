"""Shared validation and provenance helpers for the national pipeline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, *, role: str, rows: int | None = None) -> dict:
    """Describe one input/output artifact for a reproducibility manifest."""
    path = Path(path)
    record = {
        "path": str(path),
        "role": role,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if rows is not None:
        record["rows"] = int(rows)
    return record


def write_json(path: Path, payload: dict) -> None:
    """Write stable, human-readable JSON metadata."""
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def assert_unique_keys(df: pd.DataFrame, keys: Iterable[str], *, label: str) -> None:
    """Fail closed when a supposedly one-row-per-key table contains duplicates."""
    keys = list(keys)
    missing = [key for key in keys if key not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing key columns: {missing}")
    duplicate_count = int(df.duplicated(keys).sum())
    if duplicate_count:
        raise ValueError(
            f"{label} contains {duplicate_count} duplicate rows for keys {keys}; "
            "refusing to continue with an ambiguous merge."
        )


def assert_keys_subset(
    required: pd.DataFrame,
    available: pd.DataFrame,
    key: str,
    *,
    required_label: str,
    available_label: str,
) -> None:
    """Require every key in ``required`` to exist in ``available``."""
    required_keys = set(required[key].astype(str))
    available_keys = set(available[key].astype(str))
    missing = sorted(required_keys - available_keys)
    if missing:
        raise ValueError(
            f"{required_label} contains {len(missing)} keys absent from "
            f"{available_label}; refusing to continue. Examples: {missing[:5]}"
        )


def assert_same_keys(
    left: pd.DataFrame,
    right: pd.DataFrame,
    key: str,
    *,
    left_label: str,
    right_label: str,
) -> None:
    """Require exact key coverage before combining independently-produced tables."""
    left_keys = set(left[key].astype(str))
    right_keys = set(right[key].astype(str))
    missing_right = sorted(left_keys - right_keys)
    missing_left = sorted(right_keys - left_keys)
    if missing_right or missing_left:
        raise ValueError(
            f"Key mismatch between {left_label} and {right_label}: "
            f"{len(missing_right)} missing from {right_label}, "
            f"{len(missing_left)} missing from {left_label}."
        )
