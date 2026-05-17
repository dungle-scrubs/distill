"""Shared source probing, fingerprinting, and output-root helpers."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

from .errors import MediaIngestError
from .progress import ProgressReporter

CONTENT_HASH_LIMIT_BYTES = 5 * 1024 * 1024 * 1024
FINGERPRINT_SAMPLE_BYTES = 64 * 1024
SENSITIVE_COMPONENTS = {
    ".ssh",
    ".gnupg",
    ".aws",
    ".config/1password",
    "library/keychains",
}


def run_json_command(command: list[str], *, stage: str = "source") -> dict:
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise MediaIngestError(
            "E_COMMAND",
            stage,
            f"command failed: {command[0]}",
            {"stderr": proc.stderr.strip(), "command": command},
        )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise MediaIngestError(
            "E_COMMAND",
            stage,
            f"command returned invalid JSON: {command[0]}",
            {"stdout": proc.stdout[:500]},
        ) from exc


def probe_duration(path: Path, *, stage: str = "source") -> float:
    data = run_json_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        stage=stage,
    )
    try:
        return float(data["format"]["duration"])
    except (KeyError, TypeError, ValueError) as exc:
        raise MediaIngestError(
            "E_FFPROBE",
            stage,
            "could not read media duration",
        ) from exc


def ensure_duration_allowed(duration_sec: float, max_duration_sec: float) -> None:
    if duration_sec > max_duration_sec:
        raise MediaIngestError(
            "E_DURATION_CAP",
            "source",
            "media exceeds max_duration_sec",
            {"duration_sec": duration_sec, "max_duration_sec": max_duration_sec},
        )


def local_fingerprint(
    path: Path,
    cache_mode: str,
    progress: ProgressReporter | None = None,
) -> str:
    stat = path.stat()
    if cache_mode == "content":
        if stat.st_size > CONTENT_HASH_LIMIT_BYTES:
            raise MediaIngestError(
                "E_CONTENT_HASH_TOO_LARGE",
                "source",
                "content cache mode refuses files over 5 GB",
                {"size_bytes": stat.st_size},
            )
        digest = hashlib.sha256()
        bytes_read = 0
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                bytes_read += len(chunk)
                if progress:
                    progress.update(
                        "source_fingerprint",
                        percent=(bytes_read / max(1, stat.st_size)) * 100,
                        detail={
                            "cache_mode": "content",
                            "bytes_read": bytes_read,
                            "total_bytes": stat.st_size,
                        },
                    )
        if progress and stat.st_size == 0:
            progress.complete(
                "source_fingerprint",
                detail={"cache_mode": "content", "total_bytes": 0},
            )
        return digest.hexdigest()

    digest = hashlib.sha256()
    digest.update(str(stat.st_size).encode())
    digest.update(str(stat.st_mtime_ns).encode())
    sample_total = 2 if stat.st_size > FINGERPRINT_SAMPLE_BYTES else 1
    sample_done = 0
    with path.open("rb") as handle:
        digest.update(handle.read(FINGERPRINT_SAMPLE_BYTES))
        sample_done += 1
        if progress:
            progress.update(
                "source_fingerprint",
                percent=(sample_done / sample_total) * 100,
                detail={
                    "cache_mode": "fingerprint",
                    "sample": "first",
                    "samples_done": sample_done,
                    "samples_total": sample_total,
                },
            )
        if stat.st_size > FINGERPRINT_SAMPLE_BYTES:
            handle.seek(max(0, stat.st_size - FINGERPRINT_SAMPLE_BYTES))
            digest.update(handle.read(FINGERPRINT_SAMPLE_BYTES))
            sample_done += 1
            if progress:
                progress.update(
                    "source_fingerprint",
                    percent=(sample_done / sample_total) * 100,
                    detail={
                        "cache_mode": "fingerprint",
                        "sample": "last",
                        "samples_done": sample_done,
                        "samples_total": sample_total,
                    },
                )
    return digest.hexdigest()


def source_hash(source_fingerprint: str, opts_hash: str) -> str:
    return hashlib.sha256(f"{source_fingerprint}:{opts_hash}".encode()).hexdigest()


def validate_output_root(output_dir: str | None, *, default_name: str) -> Path:
    root = Path(output_dir or Path.home() / ".cache" / default_name).expanduser()
    resolved = root.resolve(strict=False)
    allowed_roots = [
        Path.home().resolve(),
        Path(tempfile.gettempdir()).resolve(),
        Path("/tmp").resolve(strict=False),
    ]
    try:
        if not any(resolved.is_relative_to(allowed) for allowed in allowed_roots):
            raise ValueError
    except ValueError as exc:
        raise MediaIngestError(
            "E_BAD_OUTPUT_DIR",
            "source",
            "output_dir must be under HOME or a system temp directory",
            {"output_dir": str(root)},
        ) from exc
    lowered = str(resolved).lower()
    home = str(Path.home().resolve()).lower()
    relative = lowered.removeprefix(home).strip(os.sep)
    for component in SENSITIVE_COMPONENTS:
        if relative == component or relative.startswith(component + os.sep):
            raise MediaIngestError(
                "E_BAD_OUTPUT_DIR",
                "source",
                "output_dir must not be inside a sensitive directory",
                {"output_dir": str(root), "blocked_component": component},
            )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved
