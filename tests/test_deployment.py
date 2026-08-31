from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import sys
import urllib.request
import zipfile
from pathlib import Path

import pytest

from scripts.showcase_runtime import (
    MAX_EXPANDED_BYTES,
    MODEL_FILES,
    ShowcaseRuntimeError,
    _download_runtime_archive,
    _SafeRedirectHandler,
    install_runtime,
    package_runtime,
    parse_args,
    prepare_case_store,
    sha256_file,
    validate_runtime,
)


def _redirect(
    request: urllib.request.Request, target: str
) -> urllib.request.Request | None:
    return _SafeRedirectHandler().redirect_request(
        request,
        None,
        302,
        "Found",
        {},
        target,
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


def test_public_https_redirect_is_allowed_without_bearer_token() -> None:
    request = urllib.request.Request("https://github.com/example/release.zip")

    redirected = _redirect(
        request,
        "https://objects.githubusercontent.com/release-assets/runtime.zip",
    )

    assert redirected is not None
    assert redirected.full_url.startswith("https://objects.githubusercontent.com/")
    assert redirected.get_header("Authorization") is None


def test_http_redirect_is_rejected() -> None:
    request = urllib.request.Request("https://downloads.example/runtime.zip")

    assert _redirect(request, "http://downloads.example/runtime.zip") is None


def test_bearer_token_is_not_forwarded_to_a_different_redirect_origin() -> None:
    request = urllib.request.Request("https://downloads.example/runtime.zip")
    request.add_header("Authorization", "Bearer deployment-secret")

    redirected = _redirect(request, "https://storage.example/runtime.zip")

    assert redirected is not None
    assert redirected.get_header("Authorization") is None


def test_download_verifies_checksum_after_response_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"downloaded-runtime-archive"
    destination = tmp_path / "runtime.zip"
    monkeypatch.setenv(
        "FRAUDETECT_RUNTIME_ARTIFACT_URL",
        "https://downloads.example/runtime.zip",
    )
    monkeypatch.setenv(
        "FRAUDETECT_RUNTIME_ARTIFACT_SHA256",
        hashlib.sha256(b"different-archive").hexdigest(),
    )
    monkeypatch.delenv("FRAUDETECT_RUNTIME_ARTIFACT_TOKEN", raising=False)

    class FakeOpener:
        def open(self, request, timeout):  # noqa: ANN001
            assert request.full_url.startswith("https://")
            assert timeout == 120
            return io.BytesIO(payload)

    def fake_build_opener(handler):  # noqa: ANN001
        assert handler is _SafeRedirectHandler
        return FakeOpener()

    monkeypatch.setattr(urllib.request, "build_opener", fake_build_opener)

    with pytest.raises(ShowcaseRuntimeError, match="checksum does not match"):
        _download_runtime_archive(destination)

    assert destination.read_bytes() == payload


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
    destination = tmp_path / "ephemeral" / "cases.sqlite"
    _database(seed, "cases", rows=3)
    seed_digest = sha256_file(seed)

    prepare_case_store(seed, destination)
    initial_digest = sha256_file(destination)
    with sqlite3.connect(destination) as connection:
        connection.execute("INSERT INTO cases (value) VALUES ('analyst-change')")
    changed_digest = sha256_file(destination)

    prepare_case_store(seed, destination)

    assert changed_digest != initial_digest
    assert sha256_file(destination) == changed_digest
    assert sha256_file(seed) == seed_digest


def test_prepare_case_store_reinitializes_after_ephemeral_store_is_lost(tmp_path: Path) -> None:
    seed = tmp_path / "seed.sqlite"
    destination = tmp_path / "ephemeral" / "cases.sqlite"
    _database(seed, "cases", rows=3)
    seed_digest = sha256_file(seed)

    prepare_case_store(seed, destination)
    with sqlite3.connect(destination) as connection:
        connection.execute("INSERT INTO cases (value) VALUES ('temporary-change')")
    destination.unlink()

    prepare_case_store(seed, destination)

    with sqlite3.connect(destination) as connection:
        assert connection.execute("SELECT COUNT(*) FROM cases").fetchone()[0] == 3
    assert sha256_file(seed) == seed_digest


def test_prepare_case_store_defaults_to_free_render_ephemeral_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FRAUDETECT_CASE_DATABASE", raising=False)
    monkeypatch.setattr(sys, "argv", ["showcase_runtime.py", "prepare-case-store"])

    assert parse_args().destination == Path("/tmp/fraudetect/cases.sqlite")


def test_render_blueprint_uses_free_ephemeral_showcase_storage() -> None:
    blueprint = Path("render.yaml").read_text(encoding="utf-8")

    assert "    plan: free\n" in blueprint
    assert "\n    disk:\n" not in blueprint
    assert "/var/data" not in blueprint
    assert (
        "      - key: FRAUDETECT_CASE_DATABASE\n"
        "        value: /tmp/fraudetect/cases.sqlite\n"
    ) in blueprint
    assert (
        "uvicorn backend.app.deployment:app --host 0.0.0.0 --port $PORT" in blueprint
    )
    assert "--reload" not in blueprint
    assert "artifacts/runtime/demo/behavior.sqlite" in blueprint
    assert "artifacts/runtime/demo/relationship.sqlite" in blueprint
    assert "artifacts/runtime/demo/cases.sqlite" in blueprint
