#!/usr/bin/env python3
"""Run local CI with disposable, exclusively managed Cargo target generations."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

WEEK = 7 * 24 * 60 * 60
BUCKET = re.compile(r"generation-[0-9]+-[0-9]+\Z")
MARKER = ".local-ci-owner.json"


def _is_link(path: Path) -> bool:
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def _valid_timestamp(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return value >= 0 and math.isfinite(value)
    except OverflowError:
        return False


@contextmanager
def _lock(path: Path):
    # Never unlink this file: another process might otherwise lock a different inode.
    if os.path.lexists(path) and _is_link(path):
        raise RuntimeError("CI retention lock is a link")
    with path.open("a+b") as handle:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            if handle.seek(0, os.SEEK_END) == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("another local CI run owns the artifact cache") from exc
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("another local CI run owns the artifact cache") from exc
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


def _write_marker(bucket: Path, root: Path, *, completed: bool, now: float) -> None:
    marker = bucket / MARKER
    if os.path.lexists(marker) and _is_link(marker):
        raise RuntimeError("CI retention marker is a link")
    marker.write_text(
        json.dumps({"version": 1, "root": str(root), "completed": completed, "last_used": now}),
        encoding="utf-8",
    )


def prune(cache: Path, root: Path, current: Path, now: float) -> list[Path]:
    """Remove only completed, owned generations unused for seven full days."""
    resolved_cache = cache.resolve(strict=True)
    if _is_link(cache) or not resolved_cache.is_relative_to(root.resolve(strict=True)):
        raise RuntimeError("CI retention directory is outside its owning checkout")
    removed = []
    for bucket in cache.iterdir():
        if bucket == current or not BUCKET.fullmatch(bucket.name) or _is_link(bucket):
            continue
        if not bucket.is_dir():
            continue
        marker = bucket / MARKER
        try:
            if _is_link(marker) or not marker.is_file():
                continue
            data = json.loads(marker.read_text(encoding="utf-8"))
            last_used = data["last_used"]
            if (
                type(data.get("version")) is not int
                or data["version"] != 1
                or data.get("root") != str(root)
                or data.get("completed") is not True
                or not _valid_timestamp(last_used)
                or now - last_used < WEEK
            ):
                continue
        except (OSError, ValueError, KeyError, TypeError):
            continue
        resolved_bucket = bucket.resolve(strict=True)
        if _is_link(bucket) or resolved_bucket.parent != resolved_cache:
            continue
        shutil.rmtree(resolved_bucket)
        removed.append(bucket)
    return removed


def run(root: Path, command: list[str], additional_workspaces: tuple[Path, ...] = ()) -> int:
    root = root.resolve(strict=True)
    if not command:
        raise RuntimeError("missing CI command")
    # Hosted CI and explicitly configured target/build paths retain their existing behavior.
    if (
        os.environ.get("CI")
        or os.environ.get("CARGO_TARGET_DIR")
        or os.environ.get("CARGO_BUILD_TARGET_DIR")
        or os.environ.get("CARGO_BUILD_BUILD_DIR")
    ):
        return subprocess.call(command, cwd=root)
    target = root / "target"
    for workspace in (root, *additional_workspaces):
        workspace = workspace.resolve(strict=True)
        try:
            metadata = subprocess.run(
                [
                    "cargo",
                    "metadata",
                    "--format-version",
                    "1",
                    "--locked",
                    "--offline",
                    "--no-deps",
                ],
                cwd=workspace,
                text=True,
                capture_output=True,
                check=True,
            )
            info = json.loads(metadata.stdout)
        except (OSError, ValueError, subprocess.SubprocessError):
            print("CI retention skipped: Cargo metadata unavailable offline", flush=True)
            return subprocess.call(command, cwd=root)
        default_target = workspace / "target"
        try:
            is_default = (
                Path(info["workspace_root"]).resolve() == workspace
                and Path(info["target_directory"]).resolve() == default_target
                and (
                    "build_directory" not in info
                    or Path(info["build_directory"]).resolve() == default_target
                )
            )
        except (KeyError, TypeError, OSError):
            is_default = False
        if not is_default:
            print("CI retention skipped: non-default or invalid Cargo configuration", flush=True)
            return subprocess.call(command, cwd=root)
    if os.path.lexists(target) and _is_link(target):
        raise RuntimeError("default Cargo target is a link")
    # Keep CI generations outside the ordinary target so unrelated cargo clean
    # cannot remove an active managed build.
    cache = root / ".local-ci-generations"
    if os.path.lexists(cache) and _is_link(cache):
        raise RuntimeError("CI retention directory is a link")
    cache.mkdir(exist_ok=True)
    with _lock(cache / ".lock"):
        now = time.time()
        generation = int(now // WEEK)
        # An interrupted generation is retained indefinitely; create a new one
        # rather than reusing or deleting a directory that may have orphaned work.
        for number in range(1000):
            bucket = cache / f"generation-{generation}-{number}"
            if not os.path.lexists(bucket):
                bucket.mkdir()
                break
            if _is_link(bucket) or not bucket.is_dir():
                continue
            marker = bucket / MARKER
            try:
                if _is_link(marker):
                    continue
                data = json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if (
                isinstance(data, dict)
                and data.get("root") == str(root)
                and type(data.get("version")) is int
                and data["version"] == 1
                and data.get("completed") is True
                and _valid_timestamp(data.get("last_used"))
            ):
                break
        else:
            raise RuntimeError("no available local CI target generation")
        _write_marker(bucket, root, completed=False, now=now)
        env = os.environ.copy()
        env["CARGO_TARGET_DIR"] = str(bucket)
        print(f"local CI target: {bucket}", flush=True)
        try:
            result = subprocess.call(command, cwd=root, env=env)
        except BaseException:
            # A killed wrapper may leave descendants: never age out an interrupted generation.
            raise
        try:
            if os.path.lexists(bucket):
                if (
                    _is_link(bucket)
                    or not bucket.is_dir()
                    or bucket.resolve(strict=True).parent != cache.resolve(strict=True)
                ):
                    raise RuntimeError("CI target generation was replaced during the run")
                _write_marker(bucket, root, completed=True, now=time.time())
            else:
                print("CI removed its own target generation", flush=True)
            for removed in prune(cache, root, bucket, time.time()):
                print(f"removed unused local CI generation: {removed}", flush=True)
        except (OSError, RuntimeError) as exc:
            print(f"CI artifact cleanup failed: {exc}", file=sys.stderr, flush=True)
            return result or 1
        return result


if __name__ == "__main__":
    try:
        arguments = sys.argv[1:]
        additional = []
        while arguments and arguments[0] == "--additional-workspace":
            if len(arguments) < 2:
                raise RuntimeError("missing path after --additional-workspace")
            additional.append(Path(arguments[1]))
            arguments = arguments[2:]
        raise SystemExit(run(Path.cwd(), arguments, tuple(additional)))
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"CI artifact retention: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
