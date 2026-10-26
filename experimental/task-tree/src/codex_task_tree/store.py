from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sqlite3
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .models import RunState, WorkspaceLease


def canonical(path: Path) -> Path:
    return path.expanduser().resolve()


def overlaps(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def file_hash(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def atomic_write(path: Path, content: bytes, *, immutable: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if immutable:
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.read_bytes() != content:
                    raise ValueError(f"immutable artifact already exists: {path}") from None
        else:
            os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


class RunStore:
    def __init__(self, root: Path, run_id: str):
        if not run_id or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for c in run_id
        ):
            raise ValueError("invalid run ID")
        self.root = canonical(root)
        self.run_id = run_id
        self.directory = self.root / "runs" / run_id
        self.directory.mkdir(parents=True, exist_ok=True)
        self.database = self.directory / "state.sqlite"
        with self.connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS run_state (run_id TEXT PRIMARY KEY, revision INTEGER NOT NULL, schema_version INTEGER NOT NULL, state_json TEXT NOT NULL, updated_at REAL NOT NULL)"
            )

    def connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.database, timeout=10)
        db.execute("PRAGMA synchronous=FULL")
        return db

    @contextmanager
    def locked(self) -> Iterator[None]:
        with (self.directory / "run.lock").open("a+") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("another controller owns this run") from None
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def load(self) -> RunState:
        with self.connect() as db:
            row = db.execute(
                "SELECT state_json FROM run_state WHERE run_id=?", (self.run_id,)
            ).fetchone()
        if row is None:
            raise FileNotFoundError(f"unknown run: {self.run_id}")
        state = RunState.model_validate_json(row[0])
        if state.schema_version not in {1, 2}:
            raise ValueError("unsupported run-state schema version")
        return state

    def save(self, state: RunState, expected_revision: int | None) -> None:
        if state.run_id != self.run_id:
            raise ValueError("run identity mismatch")
        next_revision = 0 if expected_revision is None else expected_revision + 1
        snapshot = state.model_copy(deep=True)
        snapshot.revision = next_revision
        with self.connect() as db:
            if expected_revision is None:
                try:
                    db.execute(
                        "INSERT INTO run_state VALUES (?,?,?,?,?)",
                        (
                            state.run_id,
                            next_revision,
                            state.schema_version,
                            snapshot.model_dump_json(),
                            time.time(),
                        ),
                    )
                except sqlite3.IntegrityError:
                    raise RuntimeError("run already exists") from None
            else:
                cursor = db.execute(
                    "UPDATE run_state SET revision=?,schema_version=?,state_json=?,updated_at=? WHERE run_id=? AND revision=?",
                    (
                        next_revision,
                        state.schema_version,
                        snapshot.model_dump_json(),
                        time.time(),
                        state.run_id,
                        expected_revision,
                    ),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("stale state revision; reload before committing")
        state.revision = next_revision

    def write_json(self, relative: str, value: object) -> str:
        content = json.dumps(
            value, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2
        ).encode()
        return self.write_bytes(relative, content)

    def write_bytes(self, relative: str, content: bytes) -> str:
        path = canonical(self.directory / relative)
        if self.directory not in path.parents:
            raise ValueError("artifact escapes run directory")
        atomic_write(path, content)
        return str(path)

    def snapshot(self, source: Path) -> tuple[str, str]:
        directory = self.directory / "artifacts"
        directory.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=directory)
        checksum = hashlib.sha256()
        try:
            with source.open("rb") as original, os.fdopen(fd, "wb") as target:
                for block in iter(lambda: original.read(1024 * 1024), b""):
                    checksum.update(block)
                    target.write(block)
                target.flush()
                os.fsync(target.fileno())
            sha = checksum.hexdigest()
            if file_hash(source) != sha:
                raise RuntimeError("artifact changed while its immutable snapshot was captured")
            path = directory / sha
            try:
                os.link(temporary, path)
            except FileExistsError:
                if file_hash(path) != sha:
                    raise RuntimeError("immutable artifact snapshot was modified") from None
            handle = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(handle)
            finally:
                os.close(handle)
            return sha, str(path)
        finally:
            Path(temporary).unlink(missing_ok=True)

    @contextmanager
    def lease_registry(self) -> Iterator[dict]:
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / "workspace.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = self.root / "workspace-leases.json"
            leases = json.loads(path.read_text()) if path.exists() else {}
            try:
                yield leases
                atomic_write(path, json.dumps(leases, sort_keys=True).encode(), immutable=False)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def acquire_workspace(self, workspace: Path) -> WorkspaceLease:
        workspace = canonical(workspace)
        if overlaps(workspace, self.root):
            raise ValueError("controller storage must be outside worker write roots")
        with self.lease_registry() as leases:
            for run_id, lease in leases.items():
                if run_id != self.run_id and any(
                    overlaps(workspace, Path(p)) for p in lease["canonical_write_roots"]
                ):
                    raise RuntimeError(
                        f"workspace is reserved by run {run_id}; recover/release it first"
                    )
            old = leases.get(self.run_id)
            if old and old["canonical_write_roots"] != [str(workspace)]:
                raise ValueError("run workspace cannot change")
            lease = (
                WorkspaceLease.model_validate(old)
                if old
                else WorkspaceLease(
                    run_id=self.run_id,
                    canonical_write_roots=[str(workspace)],
                    owner_ref=str(self.directory),
                )
            )
            leases[self.run_id] = lease.model_dump(mode="json")
        return lease

    def execution_reservation(self, refs: list[str]) -> None:
        with self.lease_registry() as leases:
            if self.run_id not in leases:
                raise RuntimeError("workspace lease missing")
            leases[self.run_id]["unresolved_execution_refs"] = refs

    def release_workspace(self) -> None:
        with self.lease_registry() as leases:
            lease = leases.get(self.run_id)
            if lease and lease["unresolved_execution_refs"]:
                raise RuntimeError("cannot release workspace with unresolved execution")
            leases.pop(self.run_id, None)
