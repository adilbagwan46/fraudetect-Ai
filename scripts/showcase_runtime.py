from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import stat
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

MODEL_FILES = (
    "model.joblib",
    "reference-profile.json",
    "metadata.json",
    "threshold-policy.json",
    "validation-metrics.json",
    "test-metrics.json",
    "candidate-comparison.json",
    "operating-points.json",
)
DEMO_DATABASES = {
    "demo/behavior.sqlite": "transactions",
    "demo/cases.sqlite": "cases",
    "demo/relationship.sqlite": "relationship_transactions",
}
MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024
MAX_EXPANDED_BYTES = 12 * 1024 * 1024


class ShowcaseRuntimeError(RuntimeError):
    """A safe deployment-artifact error that contains no credential or URL."""


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Keep an optional bearer token on the configured artifact origin only."""

    def redirect_request(self, request, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


def _active_model_version(root: Path) -> str:
    pointer = root / "models" / "latest.json"
    try:
        payload = json.loads(pointer.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ShowcaseRuntimeError("The model pointer is missing or invalid.") from error
    version = payload.get("model_version")
    if not isinstance(version, str) or not version or Path(version).name != version:
        raise ShowcaseRuntimeError("The model pointer contains an invalid version.")
    return version


def _required_files(root: Path) -> dict[str, Path]:
    version = _active_model_version(root)
    files = {"models/latest.json": root / "models" / "latest.json"}
    files.update(
        {
            f"models/{version}/{filename}": root / "models" / version / filename
            for filename in MODEL_FILES
        }
    )
    files.update({name: root / name for name in DEMO_DATABASES})
    return files


def _validate_sqlite(path: Path, table: str, *, expected_cases: int | None = None) -> None:
    try:
        with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as connection:
            exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
                (table,),
            ).fetchone()
            count = (
                connection.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
                if expected_cases is not None and exists
                else None
            )
    except sqlite3.Error as error:
        raise ShowcaseRuntimeError("A showcase database is unreadable.") from error
    if not exists:
        raise ShowcaseRuntimeError("A showcase database has an unexpected schema.")
    if expected_cases is not None and count != expected_cases:
        raise ShowcaseRuntimeError("The showcase case store must contain exactly three cases.")


def validate_runtime(root: Path) -> None:
    for relative, path in _required_files(root).items():
        if not path.is_file():
            raise ShowcaseRuntimeError(f"Required runtime artifact is missing: {relative}")
    for relative, table in DEMO_DATABASES.items():
        _validate_sqlite(
            root / relative,
            table,
            expected_cases=3 if relative == "demo/cases.sqlite" else None,
        )


def package_runtime(source: Path, output: Path) -> None:
    validate_runtime(source)
    if output.exists():
        raise ShowcaseRuntimeError("The output archive already exists.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for relative, path in sorted(_required_files(source).items()):
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compresslevel=9)
    print(f"Created private runtime archive: {output}")
    print(f"SHA-256: {sha256_file(output)}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_runtime_archive(destination: Path) -> None:
    url = os.getenv("FRAUDETECT_RUNTIME_ARTIFACT_URL", "")
    expected_digest = os.getenv("FRAUDETECT_RUNTIME_ARTIFACT_SHA256", "").lower()
    token = os.getenv("FRAUDETECT_RUNTIME_ARTIFACT_TOKEN")
    if not url.startswith("https://"):
        raise ShowcaseRuntimeError("A private HTTPS runtime artifact URL is required.")
    if len(expected_digest) != 64 or any(c not in "0123456789abcdef" for c in expected_digest):
        raise ShowcaseRuntimeError("A valid runtime artifact SHA-256 is required.")

    request = urllib.request.Request(url, headers={"Accept": "application/octet-stream"})
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        opener = urllib.request.build_opener(_NoRedirectHandler)
        with opener.open(request, timeout=120) as response, destination.open("wb") as output:
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_DOWNLOAD_BYTES:
                    raise ShowcaseRuntimeError("The runtime artifact exceeds the size limit.")
                output.write(chunk)
    except ShowcaseRuntimeError:
        raise
    except (OSError, urllib.error.URLError) as error:
        raise ShowcaseRuntimeError("The private runtime artifact download failed.") from error
    if sha256_file(destination) != expected_digest:
        raise ShowcaseRuntimeError("The runtime artifact checksum does not match.")


def _validate_archive(archive: zipfile.ZipFile) -> tuple[set[str], int]:
    entries = archive.infolist()
    if any(item.is_dir() for item in entries):
        raise ShowcaseRuntimeError("The runtime archive contains an unexpected file set.")
    names = {item.filename for item in entries}
    if len(entries) != len(names):
        raise ShowcaseRuntimeError("The runtime archive contains duplicate files.")
    for item in entries:
        path = PurePosixPath(item.filename)
        mode = item.external_attr >> 16
        file_type = stat.S_IFMT(mode)
        if (
            path.is_absolute()
            or path.as_posix() != item.filename
            or "\\" in item.filename
            or any(part in {"", ".", ".."} for part in path.parts)
            or item.file_size < 0
            or item.compress_size < 0
            or item.header_offset < 0
            or item.flag_bits & 0x1
            or item.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
            or file_type not in {0, stat.S_IFREG}
        ):
            raise ShowcaseRuntimeError("The runtime archive contains unsafe file metadata.")
    expanded_size = sum(item.file_size for item in entries)
    if expanded_size > MAX_EXPANDED_BYTES:
        raise ShowcaseRuntimeError("The expanded runtime artifact exceeds the size limit.")

    # Reading a member decompresses it. All metadata and expanded-size checks
    # therefore remain above the first archive.read call.
    try:
        latest = json.loads(archive.read("models/latest.json"))
    except (KeyError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ShowcaseRuntimeError("The runtime archive has an invalid model pointer.") from error
    version = latest.get("model_version")
    if not isinstance(version, str) or not version or PurePosixPath(version).name != version:
        raise ShowcaseRuntimeError("The runtime archive has an invalid model version.")
    expected = {"models/latest.json", *DEMO_DATABASES}
    expected.update(f"models/{version}/{filename}" for filename in MODEL_FILES)
    if names != expected:
        raise ShowcaseRuntimeError("The runtime archive contains an unexpected file set.")
    return expected, expanded_size


def install_runtime(destination: Path, archive_path: Path | None = None) -> None:
    destination_parent = destination.parent.resolve()
    destination_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination_parent) as temporary:
        temporary_root = Path(temporary)
        downloaded = temporary_root / "runtime.zip"
        if archive_path is None:
            _download_runtime_archive(downloaded)
        else:
            shutil.copyfile(archive_path, downloaded)
        staged = temporary_root / "staged"
        staged.mkdir()
        try:
            with zipfile.ZipFile(downloaded) as archive:
                expected, _ = _validate_archive(archive)
                for relative in sorted(expected):
                    target = staged / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(archive.read(relative))
        except (OSError, zipfile.BadZipFile) as error:
            raise ShowcaseRuntimeError("The runtime archive is invalid.") from error
        validate_runtime(staged)
        if destination.exists():
            shutil.rmtree(destination)
        staged.rename(destination)
    print("Installed the verified showcase runtime artifacts.")


def prepare_case_store(seed: Path, destination: Path) -> None:
    _validate_sqlite(seed, "cases", expected_cases=3)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        _validate_sqlite(destination, "cases")
        print("Using the existing persistent showcase case store.")
        return
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        shutil.copyfile(seed, temporary)
        _validate_sqlite(temporary, "cases", expected_cases=3)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print("Initialized the persistent showcase case store.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage private showcase runtime artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    package = subparsers.add_parser("package")
    package.add_argument("--source", type=Path, default=Path("artifacts"))
    package.add_argument("--output", type=Path, required=True)

    install = subparsers.add_parser("install")
    install.add_argument("--destination", type=Path, default=Path("artifacts/runtime"))
    install.add_argument("--archive", type=Path)

    prepare = subparsers.add_parser("prepare-case-store")
    prepare.add_argument(
        "--seed",
        type=Path,
        default=Path(
            os.getenv(
                "FRAUDETECT_SHOWCASE_CASE_SEED",
                "artifacts/runtime/demo/cases.sqlite",
            )
        ),
    )
    prepare.add_argument(
        "--destination",
        type=Path,
        default=Path(os.getenv("FRAUDETECT_CASE_DATABASE", "/var/data/fraudetect/cases.sqlite")),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "package":
            package_runtime(args.source, args.output)
        elif args.command == "install":
            install_runtime(args.destination, args.archive)
        else:
            prepare_case_store(args.seed, args.destination)
    except ShowcaseRuntimeError as error:
        print(f"Showcase runtime error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
