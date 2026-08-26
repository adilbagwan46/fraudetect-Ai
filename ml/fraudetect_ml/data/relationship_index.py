from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

RELATIONSHIP_COLUMNS = (
    "transaction_id",
    "step",
    "amount",
    "customer_id",
    "counterparty_id",
)


class RelationshipIndexBuildError(ValueError):
    """Raised when prepared data cannot produce a safe relationship index."""


def _resolve_split_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    project_relative = Path.cwd() / path
    if project_relative.is_file():
        return project_relative
    return manifest_path.parent / path.name


def _validate_chunk(chunk: pd.DataFrame, *, source: Path) -> None:
    if chunk.loc[:, RELATIONSHIP_COLUMNS].isna().any().any():
        raise RelationshipIndexBuildError(f"Missing relationship values in {source}")
    numeric = chunk.loc[:, ["step", "amount"]]
    if not numeric.map(math.isfinite).all().all():
        raise RelationshipIndexBuildError(f"Non-finite relationship values in {source}")
    if (numeric < 0).any().any():
        raise RelationshipIndexBuildError(f"Negative relationship values in {source}")
    step_values = chunk["step"].astype(float)
    if (step_values % 1 != 0).any():
        raise RelationshipIndexBuildError(f"Non-integral PaySim step in {source}")
    for column in ("transaction_id", "customer_id", "counterparty_id"):
        if (chunk[column].astype(str).str.len() == 0).any():
            raise RelationshipIndexBuildError(f"Empty internal identifier in {source}")


def build_relationship_index(
    manifest_path: Path,
    output_path: Path,
    *,
    chunksize: int = 250_000,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build an ignored, label-free SQLite relationship lookup from prepared splits."""

    if chunksize <= 0:
        raise ValueError("chunksize must be positive")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Relationship index already exists: {output_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RelationshipIndexBuildError("Prepared dataset manifest is unavailable") from error

    split_entries = manifest.get("splits", {})
    expected_splits = ("train", "validation", "test")
    if any(name not in split_entries for name in expected_splits):
        raise RelationshipIndexBuildError(
            "Manifest must contain train, validation, and test splits"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.building")
    if temporary_path.exists():
        temporary_path.unlink()

    row_count = 0
    try:
        with sqlite3.connect(temporary_path) as connection:
            connection.execute("PRAGMA journal_mode = OFF")
            connection.execute("PRAGMA synchronous = OFF")
            connection.execute("PRAGMA temp_store = FILE")
            connection.execute(
                """
                CREATE TABLE relationship_transactions (
                    transaction_reference TEXT PRIMARY KEY,
                    step INTEGER NOT NULL,
                    amount REAL NOT NULL,
                    origin_key TEXT NOT NULL,
                    destination_key TEXT NOT NULL
                ) WITHOUT ROWID
                """
            )
            for split_name in expected_splits:
                source = _resolve_split_path(
                    manifest_path,
                    str(split_entries[split_name]["path"]),
                )
                if not source.is_file():
                    raise RelationshipIndexBuildError(
                        f"Prepared split is unavailable: {source}"
                    )
                for chunk in pd.read_csv(
                    source,
                    usecols=RELATIONSHIP_COLUMNS,
                    chunksize=chunksize,
                ):
                    _validate_chunk(chunk, source=source)
                    connection.executemany(
                        "INSERT INTO relationship_transactions VALUES (?, ?, ?, ?, ?)",
                        (
                            (
                                str(row.transaction_id),
                                int(row.step),
                                float(row.amount),
                                str(row.customer_id),
                                str(row.counterparty_id),
                            )
                            for row in chunk.itertuples(index=False)
                        ),
                    )
                    row_count += len(chunk)
                    connection.commit()
            connection.execute(
                "CREATE INDEX relationship_pair_step_idx ON relationship_transactions "
                "(origin_key, destination_key, step, transaction_reference)"
            )
            connection.execute(
                "CREATE INDEX relationship_origin_step_idx ON relationship_transactions "
                "(origin_key, step, destination_key)"
            )
            connection.execute(
                "CREATE INDEX relationship_destination_step_idx ON relationship_transactions "
                "(destination_key, step, origin_key)"
            )
            connection.execute(
                "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            metadata = {
                "schema_version": "1",
                "rows": str(row_count),
                "source_dataset_sha256": str(manifest["source"]["sha256"]),
                "causal_boundary": "historical.step < current.step",
                "labels_loaded": "false",
                "labels_stored": "false",
            }
            connection.executemany(
                "INSERT INTO metadata VALUES (?, ?)", metadata.items()
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
        "labels_loaded": False,
        "labels_stored": False,
    }
