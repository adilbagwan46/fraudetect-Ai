from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

HISTORY_COLUMNS = (
    "transaction_id",
    "step",
    "transaction_type",
    "amount",
    "origin_balance_before",
    "customer_id",
)
TRANSACTION_TYPES = {"CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"}


class BehavioralIndexBuildError(ValueError):
    """Raised when prepared data cannot produce a safe behavioral index."""


def _resolve_split_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    project_relative = Path.cwd() / path
    if project_relative.is_file():
        return project_relative
    return manifest_path.parent / path.name


def _validate_chunk(chunk: pd.DataFrame, *, source: Path) -> None:
    if chunk.loc[:, HISTORY_COLUMNS].isna().any().any():
        raise BehavioralIndexBuildError(f"Missing behavioral values in {source}")
    numeric = chunk.loc[:, ["step", "amount", "origin_balance_before"]]
    if not numeric.map(math.isfinite).all().all():
        raise BehavioralIndexBuildError(f"Non-finite behavioral values in {source}")
    if (numeric < 0).any().any():
        raise BehavioralIndexBuildError(f"Negative behavioral values in {source}")
    if not set(chunk["transaction_type"]).issubset(TRANSACTION_TYPES):
        raise BehavioralIndexBuildError(f"Unexpected transaction type in {source}")
    if (chunk["transaction_id"].astype(str).str.len() == 0).any() or (
        chunk["customer_id"].astype(str).str.len() == 0
    ).any():
        raise BehavioralIndexBuildError(f"Empty internal identifier in {source}")
    step_values = chunk["step"].astype(float)
    if (step_values % 1 != 0).any():
        raise BehavioralIndexBuildError(f"Non-integral PaySim step in {source}")


def build_behavioral_index(
    manifest_path: Path,
    output_path: Path,
    *,
    chunksize: int = 250_000,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build a label-free SQLite lookup with one sequential pass over prepared splits."""

    if chunksize <= 0:
        raise ValueError("chunksize must be positive")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Behavioral index already exists: {output_path}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BehavioralIndexBuildError("Prepared dataset manifest is unavailable") from error

    split_entries = manifest.get("splits", {})
    expected_splits = ("train", "validation", "test")
    if any(name not in split_entries for name in expected_splits):
        raise BehavioralIndexBuildError("Manifest must contain train, validation, and test splits")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.building")
    if temporary_path.exists():
        temporary_path.unlink()

    row_count = 0
    try:
        with sqlite3.connect(temporary_path) as connection:
            connection.execute("PRAGMA journal_mode = OFF")
            connection.execute("PRAGMA synchronous = OFF")
            connection.execute(
                """
                CREATE TABLE transactions (
                    transaction_reference TEXT PRIMARY KEY,
                    step INTEGER NOT NULL,
                    transaction_type TEXT NOT NULL,
                    amount REAL NOT NULL,
                    origin_balance_before REAL NOT NULL,
                    origin_key TEXT NOT NULL
                ) WITHOUT ROWID
                """
            )
            for split_name in expected_splits:
                source = _resolve_split_path(
                    manifest_path,
                    str(split_entries[split_name]["path"]),
                )
                if not source.is_file():
                    raise BehavioralIndexBuildError(f"Prepared split is unavailable: {source}")
                for chunk in pd.read_csv(source, usecols=HISTORY_COLUMNS, chunksize=chunksize):
                    _validate_chunk(chunk, source=source)
                    rows = (
                        (
                            str(row.transaction_id),
                            int(row.step),
                            str(row.transaction_type),
                            float(row.amount),
                            float(row.origin_balance_before),
                            str(row.customer_id),
                        )
                        for row in chunk.itertuples(index=False)
                    )
                    connection.executemany(
                        "INSERT INTO transactions VALUES (?, ?, ?, ?, ?, ?)",
                        rows,
                    )
                    row_count += len(chunk)
                    connection.commit()
            connection.execute(
                "CREATE INDEX transactions_origin_step_idx "
                "ON transactions (origin_key, step, transaction_reference)"
            )
            connection.execute(
                "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            metadata = {
                "schema_version": "1",
                "rows": str(row_count),
                "source_dataset_sha256": str(manifest["source"]["sha256"]),
                "causal_boundary": "historical.step < current.step",
                "labels_stored": "false",
            }
            connection.executemany(
                "INSERT INTO metadata VALUES (?, ?)",
                metadata.items(),
            )
            connection.commit()
        if output_path.exists():
            output_path.unlink()
        temporary_path.replace(output_path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise

    return {
        "path": str(output_path),
        "rows": row_count,
        "source_dataset_sha256": manifest["source"]["sha256"],
        "causal_boundary": "historical.step < current.step",
        "labels_stored": False,
    }
