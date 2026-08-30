from __future__ import annotations

import json
import sqlite3
import zipfile
from pathlib import Path

import pytest

from scripts.showcase_runtime import (
    MAX_EXPANDED_BYTES,
    MODEL_FILES,
    ShowcaseRuntimeError,
    install_runtime,
    package_runtime,
    prepare_case_store,
    sha256_file,
    validate_runtime,
)


def _database(path: Path, table: str, rows: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute(f"CREATE TABLE {table} (value TEXT)")
        connection.executemany(
            f"INSERT INTO {table} (value) VALUES (?)", [(str(index),) for index in range(rows)]
        )


def _runtime(root: Path) -> None:
    version = "test-model"
    model_dir = root / "models" / version
    model_dir.mkdir(parents=True)
    (root / "models" / "latest.json").write_text(
        json.dumps({"model_version": version}), encoding="utf-8"
    )
    for filename in MODEL_FILES:
        (model_dir / filename).write_bytes(filename.encode())
    _database(root / "demo" / "behavior.sqlite", "transactions")
    _database(root / "demo" / "relationship.sqlite", "relationship_transactions")
    _database(root / "demo" / "cases.sqlite", "cases", rows=3)


def test_private_runtime_archive_is_reproducible_and_installable(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _runtime(source)
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"

    package_runtime(source, first)
    package_runtime(source, second)

    assert sha256_file(first) == sha256_file(second)
    installed = tmp_path / "installed"
    install_runtime(installed, first)
    validate_runtime(installed)


def test_runtime_archive_rejects_unexpected_members(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _runtime(source)
    archive_path = tmp_path / "runtime.zip"
    package_runtime(source, archive_path)
    with zipfile.ZipFile(archive_path, "a") as archive:
        archive.writestr("unexpected.txt", "not allowed")

    with pytest.raises(ShowcaseRuntimeError, match="unexpected file set"):
        install_runtime(tmp_path / "installed", archive_path)


def test_runtime_archive_rejects_oversize_before_read_or_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    _runtime(source)
    archive_path = tmp_path / "runtime.zip"
    package_runtime(source, archive_path)
    with zipfile.ZipFile(archive_path, "a") as archive:
        archive.writestr("oversized.bin", b"0" * MAX_EXPANDED_BYTES)

    def unexpected_read(*_args, **_kwargs):
        raise AssertionError("archive member was decompressed before the size check")

    monkeypatch.setattr(zipfile.ZipFile, "read", unexpected_read)
    destination = tmp_path / "installed"
    with pytest.raises(ShowcaseRuntimeError, match="expanded runtime artifact"):
        install_runtime(destination, archive_path)

    assert not destination.exists()


@pytest.mark.parametrize("unsafe_name", ["../escape", "/absolute", "models\\escape"])
def test_runtime_archive_rejects_unsafe_member_paths(
    tmp_path: Path, unsafe_name: str
) -> None:
    source = tmp_path / "source"
    _runtime(source)
    archive_path = tmp_path / "runtime.zip"
    package_runtime(source, archive_path)
    with zipfile.ZipFile(archive_path, "a") as archive:
        archive.writestr(unsafe_name, "not allowed")

    destination = tmp_path / "installed"
    with pytest.raises(ShowcaseRuntimeError, match="unsafe file metadata"):
        install_runtime(destination, archive_path)

    assert not destination.exists()


def test_prepare_case_store_initializes_once_without_overwrite(tmp_path: Path) -> None:
    seed = tmp_path / "seed.sqlite"
    destination = tmp_path / "disk" / "cases.sqlite"
    _database(seed, "cases", rows=3)

    prepare_case_store(seed, destination)
    initial_digest = sha256_file(destination)
    with sqlite3.connect(destination) as connection:
        connection.execute("INSERT INTO cases (value) VALUES ('analyst-change')")
    changed_digest = sha256_file(destination)

    prepare_case_store(seed, destination)

    assert changed_digest != initial_digest
    assert sha256_file(destination) == changed_digest
