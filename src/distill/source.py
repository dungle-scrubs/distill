"""Source acquisition and fingerprinting for Distill.

This module owns local path resolution, duration probing, safe output root
validation, YouTube id lookup/download, disk checks, and source fingerprints.
It does not write final bundle artifacts.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import parse_qs, parse_qsl, urlencode, urlparse, urlunparse

from .errors import DistillError, warning
from .options import DistillOptions
from .progress import ProgressReporter

CONTENT_HASH_LIMIT_BYTES = 5 * 1024 * 1024 * 1024
FINGERPRINT_SAMPLE_BYTES = 64 * 1024
YOUTUBE_DISK_FLOOR_BYTES = 1024 * 1024 * 1024
# Wall-clock ceilings so a wedged tool or a stalled network call cannot hang the
# whole run. yt-dlp additionally gets `--socket-timeout` so it aborts a stalled
# connection on its own rather than blocking until the outer timeout fires.
FFPROBE_TIMEOUT_SEC = 60.0
YTDLP_METADATA_TIMEOUT_SEC = 120.0
YTDLP_SOCKET_TIMEOUT_SEC = 30
SENSITIVE_COMPONENTS = {
    ".ssh",
    ".gnupg",
    ".aws",
    ".config/1password",
    "library/keychains",
}
YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "youtu.be",
    "music.youtube.com",
}

BYTE_UNITS = {
    "B": 1,
    "KB": 1000,
    "MB": 1000**2,
    "GB": 1000**3,
    "TB": 1000**4,
    "KiB": 1024,
    "MiB": 1024**2,
    "GiB": 1024**3,
    "TiB": 1024**4,
}


@dataclass(frozen=True)
class SourceInfo:
    source_type: str
    resolved_path: Path
    duration_sec: float
    source_fingerprint: str
    source_hash: str
    warnings: list[dict[str, str]]
    youtube_video_id: str | None = None
    youtube_lock_key: str | None = None
    related_links: list[dict[str, str]] | None = None


@dataclass(frozen=True)
class SourceRequest:
    value: str
    options: DistillOptions
    output_root: Path | None = None
    progress: ProgressReporter | None = None


@dataclass(frozen=True)
class SourceResolution:
    source: SourceInfo
    output_root: Path | None
    progress: ProgressReporter | None = None


@dataclass(frozen=True)
class YouTubeMetadata:
    video_id: str
    description: str
    warnings: list[dict[str, str]]


class YouTubeDownloaderProtocol(Protocol):
    last_warnings: list[dict[str, str]]

    def download(
        self,
        url: str,
        lock_key: str,
        progress: ProgressReporter | None = None,
    ) -> Path: ...


def run_json_command(command: list[str], *, timeout: float = FFPROBE_TIMEOUT_SEC) -> dict:
    try:
        proc = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=timeout
        )
    except FileNotFoundError as exc:
        raise DistillError(
            "E_MISSING_TOOL",
            "source",
            f"required tool is not installed: {command[0]}",
            {"tool": command[0]},
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise DistillError(
            "E_COMMAND",
            "source",
            f"command timed out: {command[0]}",
            {"timeout_sec": timeout, "command": command},
        ) from exc
    if proc.returncode != 0:
        raise DistillError(
            "E_COMMAND",
            "source",
            f"command failed: {command[0]}",
            {"stderr": proc.stderr.strip(), "command": command},
        )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise DistillError(
            "E_COMMAND",
            "source",
            f"command returned invalid JSON: {command[0]}",
            {"stdout": proc.stdout[:500]},
        ) from exc


def probe_duration(path: Path) -> float:
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
        ]
    )
    try:
        duration = float(data["format"]["duration"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DistillError("E_FFPROBE", "source", "could not read video duration") from exc
    return duration


def ensure_duration_allowed(duration_sec: float, max_duration_sec: float) -> None:
    if duration_sec > max_duration_sec:
        raise DistillError(
            "E_DURATION_CAP",
            "source",
            "video exceeds max_duration_sec",
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
            raise DistillError(
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


class LocalSourceProvider:
    def resolve(self, request: SourceRequest) -> SourceInfo:
        path_text = request.value
        options = request.options
        progress = request.progress
        if not path_text:
            raise DistillError("E_BAD_SOURCE", "source", "path is required")
        original = Path(path_text).expanduser()
        if not original.exists() or not original.is_file():
            raise DistillError(
                "E_BAD_SOURCE", "source", "local video does not exist", {"path": path_text}
            )
        resolved = original.resolve()
        warnings: list[dict[str, str]] = []
        if resolved != original.absolute():
            warnings.append(
                warning(
                    "source",
                    "symlink_resolved",
                    f"source path resolved to {resolved}",
                )
            )
        if progress:
            progress.update("duration_probe", status="running")
        duration = probe_duration(resolved)
        if progress:
            progress.complete("duration_probe", detail={"duration_sec": duration})
        ensure_duration_allowed(duration, options.max_duration_sec)
        fingerprint = local_fingerprint(resolved, options.cache_mode, progress)
        opts_hash = options.opts_hash("local")
        return SourceInfo(
            source_type="local",
            resolved_path=resolved,
            duration_sec=duration,
            source_fingerprint=fingerprint,
            source_hash=source_hash(fingerprint, opts_hash),
            warnings=warnings,
        )


def resolve_local_source(
    path_text: str,
    options: DistillOptions,
    progress: ProgressReporter | None = None,
) -> SourceInfo:
    return LocalSourceProvider().resolve(SourceRequest(path_text, options, progress=progress))


def validate_output_root(output_dir: str | None) -> Path:
    root = Path(output_dir).expanduser() if output_dir else Path.home() / ".cache" / "distill"
    root = root.resolve()
    home = Path.home().resolve()
    temp = Path(tempfile.gettempdir()).resolve()
    if not (root in (home, temp) or root.is_relative_to(home) or root.is_relative_to(temp)):
        raise DistillError(
            "E_BAD_OUTPUT_DIR",
            "bundle",
            "output_dir must be under $HOME or the system temp directory",
            {"output_dir": str(root)},
        )

    sensitive = sensitive_path_match(root)
    if sensitive:
        raise DistillError(
            "E_BAD_OUTPUT_DIR",
            "bundle",
            "output_dir points inside a sensitive path",
            {"output_dir": str(root), "matched": sensitive},
        )
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink():
        raise DistillError("E_BAD_OUTPUT_DIR", "bundle", "output_dir must not be a symlink")
    return root


def sensitive_path_match(
    root: Path,
    *,
    case_sensitive: bool | None = None,
) -> str | None:
    if case_sensitive is None:
        case_sensitive = sys.platform != "darwin"
    root_text = str(root) if case_sensitive else str(root).lower()
    for sensitive in SENSITIVE_COMPONENTS:
        candidate = sensitive if case_sensitive else sensitive.lower()
        if candidate in root_text:
            return sensitive
    return None


YOUTUBE_STRIP_QUERY_KEYS = {"t", "start", "end", "time_continue"}


def normalize_youtube_url(url: str) -> str:
    """Drop player-state query params (e.g. `t=900s`) so the full video is consumed."""
    parsed = urlparse(url)
    if parsed.netloc.lower() not in YOUTUBE_HOSTS:
        return url
    pairs = parse_qsl(parsed.query, keep_blank_values=True)
    kept = [
        (k, v)
        for k, v in pairs
        if k not in YOUTUBE_STRIP_QUERY_KEYS
    ]
    if len(kept) == len(pairs):
        return url
    new_query = urlencode(kept)
    fragment = "" if parsed.fragment.startswith("t=") else parsed.fragment
    return urlunparse(parsed._replace(query=new_query, fragment=fragment))


def parse_youtube_url(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if host not in YOUTUBE_HOSTS:
        raise DistillError("E_BAD_URL", "youtube", "only YouTube URLs are supported")
    if host == "youtu.be":
        video_id = parsed.path.strip("/").split("/")[0]
    elif any(parsed.path.startswith(prefix) for prefix in ("/shorts/", "/embed/", "/live/", "/v/")):
        video_id = parsed.path.strip("/").split("/")[1]
    else:
        video_id = parse_qs(parsed.query).get("v", [""])[0]
    if not video_id:
        raise DistillError(
            "E_BAD_URL",
            "youtube",
            f"could not find YouTube video id in URL: {url}",
        )
    return video_id


def _ytdlp_command(extra_args: list[str], url: str) -> list[str]:
    """Build a yt-dlp argv with a stall guard and a `--` terminator.

    The `--` before the URL stops a value that begins with `-` from being parsed
    as a yt-dlp option (argument injection), and `--socket-timeout` lets yt-dlp
    abort a stalled connection on its own.
    """
    return [
        "yt-dlp",
        "--socket-timeout",
        str(YTDLP_SOCKET_TIMEOUT_SEC),
        *extra_args,
        "--",
        url,
    ]


def _run_ytdlp(
    extra_args: list[str],
    url: str,
    *,
    timeout: float = YTDLP_METADATA_TIMEOUT_SEC,
) -> subprocess.CompletedProcess[str]:
    command = _ytdlp_command(extra_args, url)
    try:
        return subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=timeout
        )
    except FileNotFoundError as exc:
        raise DistillError(
            "E_MISSING_TOOL", "youtube", "yt-dlp is not installed", {"tool": "yt-dlp"}
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise DistillError(
            "E_YTDLP", "youtube", "yt-dlp timed out", {"timeout_sec": timeout}
        ) from exc


def youtube_description(url: str) -> tuple[str, list[dict[str, str]]]:
    try:
        proc = subprocess.run(
            _ytdlp_command(["--skip-download", "--print", "%(description)s"], url),
            capture_output=True,
            text=True,
            check=False,
            timeout=YTDLP_METADATA_TIMEOUT_SEC,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        # Description is best-effort metadata; degrade rather than fail the run.
        return "", [
            warning(
                "youtube",
                "metadata_unavailable",
                "yt-dlp could not read YouTube description",
            )
        ]
    if proc.returncode != 0:
        return "", [
            warning(
                "youtube",
                "metadata_unavailable",
                "yt-dlp could not read YouTube description",
            )
        ]
    return proc.stdout.strip(), []


def youtube_metadata(url: str) -> YouTubeMetadata:
    proc = _run_ytdlp(["--skip-download", "--dump-json"], url)
    if proc.returncode == 0:
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError:
            payload = {}
        video_id = str(payload.get("id") or "").strip()
        if video_id:
            return YouTubeMetadata(
                video_id=video_id,
                description=str(payload.get("description") or "").strip(),
                warnings=[],
            )

    video_id = canonical_youtube_id(url)
    description, metadata_warnings = youtube_description(url)
    return YouTubeMetadata(video_id=video_id, description=description, warnings=metadata_warnings)


def canonical_youtube_id(url: str) -> str:
    parse_youtube_url(url)
    proc = _run_ytdlp(["--simulate", "--print", "id"], url)
    if proc.returncode != 0:
        raise DistillError(
            "E_YTDLP",
            "youtube",
            "yt-dlp could not resolve video id",
            {"stderr": proc.stderr.strip()},
        )
    video_id = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    if not video_id:
        raise DistillError("E_YTDLP", "youtube", "yt-dlp returned an empty video id")
    return video_id


def youtube_lock_key(video_id: str) -> str:
    return hashlib.sha256(video_id.encode()).hexdigest()


def check_disk_floor(path: Path) -> None:
    usage = shutil.disk_usage(path)
    if usage.free < YOUTUBE_DISK_FLOOR_BYTES:
        raise DistillError(
            "E_DISK_SPACE",
            "youtube",
            "at least 1 GB free disk space is required",
            {"free_bytes": usage.free},
        )


def parse_byte_amount(value: str, unit: str) -> int:
    multiplier = BYTE_UNITS.get(unit)
    if multiplier is None:
        raise ValueError(f"unknown byte unit: {unit}")
    return int(float(value) * multiplier)


def parse_ytdlp_progress(line: str) -> dict[str, float | int] | None:
    if "[download]" not in line:
        return None
    result: dict[str, float | int] = {}
    percent = re.search(r"(?P<percent>\d+(?:\.\d+)?)%", line)
    if percent:
        result["percent"] = float(percent.group("percent"))

    downloaded = re.search(
        r"(?P<downloaded>\d+(?:\.\d+)?)\s*(?P<downloaded_unit>[KMGT]?i?B)\s+of\s+"
        r"(?P<total>\d+(?:\.\d+)?)\s*(?P<total_unit>[KMGT]?i?B)",
        line,
    )
    if downloaded:
        result["downloaded_bytes"] = parse_byte_amount(
            downloaded.group("downloaded"),
            downloaded.group("downloaded_unit"),
        )
        result["total_bytes"] = parse_byte_amount(
            downloaded.group("total"),
            downloaded.group("total_unit"),
        )
        if "percent" not in result and result["total_bytes"]:
            result["percent"] = (
                float(result["downloaded_bytes"]) / float(result["total_bytes"])
            ) * 100
    return result or {"indeterminate": 1}


class YoutubeDownloader:
    def __init__(
        self,
        output_root: Path,
        stale_sec: float = 180.0,
        lock_wait_sec: float = 0.0,
        lock_poll_sec: float = 0.25,
        lock_warn_after_sec: float = 5.0,
    ) -> None:
        self.output_root = output_root
        self.stale_sec = stale_sec
        self.lock_wait_sec = lock_wait_sec
        self.lock_poll_sec = lock_poll_sec
        self.lock_warn_after_sec = lock_warn_after_sec
        self.last_warnings: list[dict[str, str]] = []

    def download(
        self,
        url: str,
        lock_key: str,
        progress: ProgressReporter | None = None,
    ) -> Path:
        locks = self.output_root / "_youtube_locks"
        locks.mkdir(parents=True, exist_ok=True)
        lock = locks / f"{lock_key}.lock"
        video_dir = self.output_root / "_youtube_sources" / lock_key
        video_dir.mkdir(parents=True, exist_ok=True)
        if progress:
            progress.update("youtube_download", status="running", detail={"step": "lock"})
        acquired, lock_warnings = self._acquire(lock)
        self.last_warnings = lock_warnings
        if not acquired:
            raise DistillError("E_LOCKED", "youtube", "YouTube source is locked by another process")
        try:
            out_template = str(video_dir / "source.%(ext)s")
            command = [
                "yt-dlp",
                "-f",
                "best[ext=mp4][height<=720]/best[height<=720]/best[ext=mp4]/best/bv*+ba/b",
                "--newline",
                "--socket-timeout",
                "30",
                "--retries",
                "3",
                "--fragment-retries",
                "3",
                "--concurrent-fragments",
                "8",
                "--merge-output-format",
                "mp4",
                "-o",
                out_template,
                "--",
                url,
            ]
            proc = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stderr_lines: list[str] = []
            if proc.stderr:
                for line in proc.stderr:
                    stderr_lines.append(line)
                    parsed = parse_ytdlp_progress(line)
                    if parsed and progress:
                        progress.update(
                            "youtube_download",
                            percent=(float(parsed["percent"]) if "percent" in parsed else None),
                            detail=parsed,
                        )
            stdout, stderr_tail = proc.communicate()
            if stderr_tail:
                stderr_lines.append(stderr_tail)
            if proc.returncode != 0:
                raise DistillError(
                    "E_YTDLP",
                    "youtube",
                    "yt-dlp download failed",
                    {"stderr": "".join(stderr_lines).strip(), "stdout": stdout.strip()},
                )
            candidates = sorted(video_dir.glob("source.*"))
            if not candidates:
                raise DistillError("E_YTDLP", "youtube", "yt-dlp did not produce a source file")
            if progress:
                progress.complete("youtube_download", detail={"path": str(candidates[0])})
            return candidates[0]
        finally:
            lock.unlink(missing_ok=True)

    def _acquire(self, lock: Path) -> tuple[bool, list[dict[str, str]]]:
        started = time.monotonic()
        warnings: list[dict[str, str]] = []
        while True:
            acquired = self._try_acquire_once(lock)
            if acquired:
                waited = time.monotonic() - started
                if waited >= self.lock_warn_after_sec:
                    warnings.append(
                        warning(
                            "youtube",
                            "long_lock_wait",
                            f"waited {waited:.1f}s for YouTube source lock",
                        )
                    )
                return True, warnings
            if time.monotonic() - started >= self.lock_wait_sec:
                return False, warnings
            time.sleep(self.lock_poll_sec)

    def _try_acquire_once(self, lock: Path) -> bool:
        now = time.monotonic()
        payload = {
            "pid": os.getpid(),
            "created_wall": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "created_monotonic": now,
            "last_heartbeat_monotonic": now,
        }
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                current = json.loads(lock.read_text())
                heartbeat = float(current.get("last_heartbeat_monotonic", 0))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                heartbeat = 0
            if now - heartbeat <= self.stale_sec:
                return False
            lock.unlink(missing_ok=True)
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, sort_keys=True)
        return True


class YouTubeSourceProvider:
    def cached(
        self,
        request: SourceRequest,
        metadata: YouTubeMetadata | None = None,
    ) -> SourceInfo | None:
        """Return a SourceInfo from a published manifest, or None on miss."""
        if request.output_root is None:
            raise DistillError("E_BAD_OUTPUT_DIR", "youtube", "output_root is required")
        video_id = metadata.video_id if metadata is not None else canonical_youtube_id(request.value)
        fingerprint = hashlib.sha256(video_id.encode()).hexdigest()
        sh = source_hash(fingerprint, request.options.opts_hash("youtube"))
        manifest_path = request.output_root / sh / "_manifest.json"
        if not manifest_path.exists():
            return None
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        duration = manifest.get("duration_sec")
        resolved_path = manifest.get("source_resolved_path")
        if not isinstance(duration, (int, float)) or not isinstance(resolved_path, str):
            return None
        return SourceInfo(
            source_type="youtube",
            resolved_path=Path(resolved_path),
            duration_sec=float(duration),
            source_fingerprint=fingerprint,
            source_hash=sh,
            warnings=[],
            youtube_video_id=video_id,
            youtube_lock_key=youtube_lock_key(video_id),
            related_links=list(manifest.get("related_links", []))
            if isinstance(manifest.get("related_links"), list)
            else None,
        )

    def resolve(
        self,
        request: SourceRequest,
        downloader: YouTubeDownloaderProtocol | None = None,
        metadata: YouTubeMetadata | None = None,
    ) -> SourceInfo:
        if request.output_root is None:
            raise DistillError("E_BAD_OUTPUT_DIR", "youtube", "output_root is required")
        options = request.options
        output_root = request.output_root
        progress = request.progress
        if progress:
            progress.update("youtube_download", status="running", detail={"step": "disk_precheck"})
        check_disk_floor(output_root)
        if progress:
            progress.update("youtube_download", status="running", detail={"step": "resolve_id"})
        metadata = metadata or youtube_metadata(request.value)
        video_id = metadata.video_id
        lock_key = youtube_lock_key(video_id)
        fingerprint = hashlib.sha256(video_id.encode()).hexdigest()
        source = source_hash(fingerprint, options.opts_hash("youtube"))
        from .links import extract_relevant_links

        downloader = downloader or YoutubeDownloader(output_root)
        downloaded = downloader.download(request.value, lock_key, progress)
        warnings = [*metadata.warnings, *list(getattr(downloader, "last_warnings", []))]
        if progress:
            progress.update("duration_probe", status="running")
        duration = probe_duration(downloaded)
        if progress:
            progress.complete("duration_probe", detail={"duration_sec": duration})
        ensure_duration_allowed(duration, options.max_duration_sec)
        if progress:
            progress.update("youtube_download", status="running", detail={"step": "disk_postcheck"})
        check_disk_floor(output_root)
        if progress:
            progress.complete(
                "youtube_download",
                detail={"path": str(downloaded.resolve()), "step": "complete"},
            )
        return SourceInfo(
            source_type="youtube",
            resolved_path=downloaded.resolve(),
            duration_sec=duration,
            source_fingerprint=fingerprint,
            source_hash=source,
            warnings=warnings,
            youtube_video_id=video_id,
            youtube_lock_key=lock_key,
            related_links=extract_relevant_links(
                metadata.description,
                source="youtube_description",
            ),
        )


class SourceResolver:
    def __init__(
        self,
        local: LocalSourceProvider | None = None,
        youtube: YouTubeSourceProvider | None = None,
    ) -> None:
        self.local = local or LocalSourceProvider()
        self.youtube = youtube or YouTubeSourceProvider()

    def local_source(
        self,
        path_text: str,
        options: DistillOptions,
        progress: ProgressReporter | None = None,
    ) -> SourceInfo:
        return self.local.resolve(SourceRequest(path_text, options, progress=progress))

    def cached_youtube_source(
        self,
        url: str,
        options: DistillOptions,
        output_root: Path,
        metadata: YouTubeMetadata | None = None,
    ) -> SourceInfo | None:
        return self.youtube.cached(
            SourceRequest(url, options, output_root=output_root),
            metadata=metadata,
        )

    def youtube_source(
        self,
        url: str,
        options: DistillOptions,
        output_root: Path,
        downloader: YouTubeDownloaderProtocol | None = None,
        progress: ProgressReporter | None = None,
        metadata: YouTubeMetadata | None = None,
    ) -> SourceInfo:
        return self.youtube.resolve(
            SourceRequest(url, options, output_root=output_root, progress=progress),
            downloader=downloader,
            metadata=metadata,
        )

    def resolve(
        self,
        source_type: str,
        value: str,
        options: DistillOptions,
        *,
        progress: ProgressReporter | None = None,
        downloader: YouTubeDownloaderProtocol | None = None,
    ) -> SourceResolution:
        if source_type == "local":
            return SourceResolution(
                self.local_source(value, options, progress=progress),
                output_root=None,
                progress=progress,
            )
        if source_type != "youtube":
            raise DistillError(
                "E_BAD_SOURCE",
                "source",
                "source_type must be 'local' or 'youtube'",
                {"source_type": source_type},
            )

        root = validate_output_root(options.output_dir)
        url = normalize_youtube_url(value)
        # Reject non-YouTube hosts (and option-injection values) before yt-dlp runs.
        parse_youtube_url(url)
        metadata = youtube_metadata(url)
        if not options.force_reprocess:
            cached = self.cached_youtube_source(url, options, root, metadata=metadata)
            if cached is not None:
                if progress:
                    progress.skip_cached(
                        "youtube_download",
                        detail={
                            "source": "cached_manifest",
                            "video_id": cached.youtube_video_id,
                        },
                    )
                return SourceResolution(cached, output_root=root, progress=progress)
        return SourceResolution(
            self.youtube_source(
                url,
                options,
                root,
                downloader=downloader,
                progress=progress,
                metadata=metadata,
            ),
            output_root=root,
            progress=progress,
        )


def cached_youtube_source(
    url: str,
    options: DistillOptions,
    output_root: Path,
) -> SourceInfo | None:
    return SourceResolver().cached_youtube_source(url, options, output_root)


def youtube_source_info(
    url: str,
    options: DistillOptions,
    output_root: Path,
    downloader: YouTubeDownloaderProtocol | None = None,
    progress: ProgressReporter | None = None,
) -> SourceInfo:
    return SourceResolver().youtube_source(url, options, output_root, downloader, progress)


def resolve_source_for_processing(
    source_type: str,
    value: str,
    options: DistillOptions,
    *,
    progress: ProgressReporter | None = None,
    downloader: YouTubeDownloaderProtocol | None = None,
) -> SourceResolution:
    return SourceResolver().resolve(
        source_type,
        value,
        options,
        progress=progress,
        downloader=downloader,
    )
