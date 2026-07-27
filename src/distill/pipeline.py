"""Distill pipeline orchestration.

This module owns the package-facing processing functions used by the CLI.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .bundle_store import (
    BATCH_ITEM_LOCK_WAIT_SEC,
    DEFAULT_KEEP_GENERATIONS,
    SINGLE_SOURCE_LOCK_WAIT_SEC,
    BundleRun,
    BundleSnapshot,
    BundleStore,
    PrunePolicy,
    ensure_safe_directory,
)
from .cache_doctor import inspect_cache
from .errors import DistillError
from .frame_selection import select_keyframes
from .job_store import JobOutcome, JobStore
from .local_vision import (
    FrameInterpreter,
    local_vision_config_from_args,
    probe_local_vision,
    try_interpret_image,
)
from .ocr import ocr_frames
from .options import DistillOptions
from .progress import (
    TERMINAL_PROGRESS_STATUSES,
    OverallProgressAggregator,
    ProgressCounter,
    ProgressHeartbeat,
    ProgressReporter,
)
from .redact_secrets import redact_text
from .render import render_markdown
from .response import manifest_document, run_response
from .source import (
    normalize_youtube_url,
    release_acquisition_lease,
    resolve_source_for_processing,
    validate_output_root,
)

TOOLS = {
    "process_local_video": "Process a local video into a transcript/keyframe markdown bundle",
    "process_youtube_video": "Download and process one YouTube video into a transcript/keyframe markdown bundle",
    "process_video_directory": "Process video files in a local directory into Distill bundles",
    "process_youtube_playlist": "Process videos from a YouTube playlist or channel URL",
    "cleanup_cache": "Prune old Distill cache bundles and generations",
    "cache_doctor": "Report bundles, markers, generations, locks and a prune preview for an output root",
    "get_job_status": "Read a Distill job status record by job id",
}
DEFAULT_CONFIGURED_TIMEOUT_MS = 5_400_000

TIMEOUT_ENV = "DISTILL_EFFECTIVE_TIMEOUT_MS"
LONG_TIMEOUT_PROBE_ENV = "DISTILL_ENABLE_LONG_TIMEOUT_PROBE"
TIMEOUT_PROBE_LIMIT_MS = 1_000


@dataclass(frozen=True)
class ToolSpec:
    description: str
    handler: Callable[[dict[str, Any]], dict[str, Any]]


class DistillSession:
    def call_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        try:
            result = call_registered_tool(name, args)
            return {"result": {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}}
        except DistillError as exc:
            return {"error": {"message": exc.to_json_text()}}
        except Exception as exc:
            error = DistillError("E_INTERNAL", "internal", str(exc))
            return {"error": {"message": error.to_json_text()}}


def process_local_video(
    args: dict[str, Any], *, lock_wait_sec: float = SINGLE_SOURCE_LOCK_WAIT_SEC
) -> dict[str, Any]:
    options = DistillOptions.from_args(args)
    resolution = resolve_source_for_processing("local", str(args.get("path", "")), options)
    return process_resolved_source(
        resolution.source,
        options,
        resolution.output_root,
        progress=resolution.progress,
        tool="process_local_video",
        lock_wait_sec=lock_wait_sec,
    )


def process_youtube_video(
    args: dict[str, Any], *, lock_wait_sec: float = SINGLE_SOURCE_LOCK_WAIT_SEC
) -> dict[str, Any]:
    options = DistillOptions.from_args({**args, "cache_mode": "fingerprint"})
    progress = ProgressReporter(emitter=progress_emitter(options.job_id))
    resolution = resolve_source_for_processing(
        "youtube",
        str(args.get("url", "")),
        options,
        progress=progress,
    )
    return process_resolved_source(
        resolution.source,
        options,
        resolution.output_root,
        progress=resolution.progress,
        tool="process_youtube_video",
        lock_wait_sec=lock_wait_sec,
    )


def progress_emitter(job_id: str) -> Any:
    def emit(event: Any) -> None:
        payload = event.to_dict()
        payload["job_id"] = job_id
        print(json.dumps(payload, sort_keys=True), file=sys.stderr, flush=True)

    return emit


def cache_hit_progress_summary(
    store: BundleStore, snapshot: BundleSnapshot
) -> tuple[BundleSnapshot, dict[str, Any]]:
    """The progress summary a cache hit reports, amending the manifest if it must.

    A bundle published before progress was recorded terminally has no summary a
    poller can use, so one is synthesized and written back through the store's
    amendment path (R-14) rather than by rewriting the file here - a **manifest**
    is the **bundle marker**, and a reader catching it half-rewritten sees a
    directory that is briefly not a bundle at all.
    """
    recorded = snapshot.manifest.get("progress")
    if isinstance(recorded, dict) and progress_summary_is_terminal(recorded):
        return snapshot, recorded

    summary = OverallProgressAggregator().cached_summary({"source": "cache"})
    return store.patch_published(snapshot, {"progress": summary}), summary


def progress_summary_is_terminal(progress_summary: dict[str, Any]) -> bool:
    if progress_summary.get("overall_percent") != 100.0:
        return False
    mechanisms = progress_summary.get("mechanisms")
    if not isinstance(mechanisms, dict) or not mechanisms:
        return False
    return all(
        isinstance(state, dict) and state.get("status") in TERMINAL_PROGRESS_STATUSES
        for state in mechanisms.values()
    )


def _abandon_reason(exc: BaseException) -> str:
    """Why a run gave up, in the terms the rest of Distill reports failures in.

    A `DistillError`'s code is the identity an operator correlates on, and
    `str(exc)` does not carry it; anything else falls back to its type, because
    an uncoded exception's message alone rarely says which stage produced it.
    """
    if isinstance(exc, DistillError):
        return f"{exc.code}: {exc}"
    return f"{type(exc).__name__}: {exc}"


def _as_stage_payload(name: str, recorded: Any) -> dict[str, Any] | None:
    """A recorded **stage result** in the shape this run expects, or `None`.

    Anything else is treated as absent rather than as an error: a stage result
    is scratch, so a shape a newer run does not recognize costs a recomputation
    and nothing more. The one accommodation is a bare list under `frames`, which
    is how an older run recorded that stage before payloads carried warnings.
    """
    if isinstance(recorded, dict):
        return recorded
    if name == "frames" and isinstance(recorded, list):
        return {"frames": recorded, "warnings": []}
    return None


@dataclass
class ProcessingRun:
    source: Any
    options: DistillOptions
    output_root: Path
    progress: ProgressReporter
    tool: str
    lock_wait_sec: float = SINGLE_SOURCE_LOCK_WAIT_SEC
    """How long this run waits for a contended **bundle key** (D-044).

    The run a user is watching waits five minutes, because the likely holder is
    another run of the same video that is nearly done. One item of a batch or a
    playlist waits five seconds and then fails `E_LOCKED`: serializing a
    25-item playlist behind another run's 40-minute video would cost hours of
    blocked waiting, and `continue_on_error` already defaults true, so the item
    fails, the batch proceeds, and re-running picks it up as a cache hit.
    """

    def execute(self) -> dict[str, Any]:
        """Produce this run's **generation**, or hand back the one already published.

        The run lock is taken before anything under the **bundle key** is
        created and held until the generation is committed (R-08), so the whole
        of a run - the download, the vision pass, the publish - happens under
        exclusion rather than only the staging that opens it. `begin` answers
        the cache question from *under* that lock, which is what makes a waiter
        coalesce onto the winner's result instead of redoing its work.
        """
        store = BundleStore.open(self.output_root)
        began = store.begin(
            self.source.source_hash,
            wait_sec=self.lock_wait_sec,
            resume=self.options.resume_partial,
            reuse_active=not self.options.force_reprocess,
        )
        if isinstance(began, BundleSnapshot):
            return self._cached_response(store, began)

        heartbeat = ProgressHeartbeat(self.progress.counter).start()
        try:
            with began as run:
                try:
                    return self._produce_generation(run, heartbeat)
                except BaseException as exc:
                    # A bundle that did not change is otherwise indistinguishable
                    # from a run that never happened: the reason is the record.
                    # The previous **active generation** is untouched and the
                    # **staging directory** stays for the next run to resume.
                    run.abandon(_abandon_reason(exc))
                    raise
        finally:
            heartbeat.stop()

    def _cached_response(self, store: BundleStore, snapshot: BundleSnapshot) -> dict[str, Any]:
        snapshot, progress_summary = cache_hit_progress_summary(store, snapshot)
        manifest = snapshot.manifest
        return run_response(
            snapshot,
            self.source,
            list(manifest.get("frames", [])),
            bool(manifest.get("transcript_present")),
            list(manifest.get("warnings", [])),
            cached=True,
            progress=progress_summary,
            job_id=self.options.job_id,
        )

    def _run_stage(
        self,
        run: BundleRun,
        heartbeat: ProgressHeartbeat,
        warnings: list[dict[str, str]],
        name: str,
        skipped_mechanisms: tuple[str, ...],
        producer: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]:
        """Recover a **stage result** or produce one, recording it either way.

        Resume is the store's answer, not this module's: `read_stage` reports
        `None` for every reason a recorded result cannot be used - the run is
        not resuming, nothing was recorded, or what was recorded is unreadable -
        and every one of them means the same thing here, which is compute it.
        """
        payload = _as_stage_payload(name, run.read_stage(name))
        if payload is not None:
            for mechanism in skipped_mechanisms:
                self.progress.skip_cached(mechanism, detail={"source": "partial_resume"})
        else:
            payload = producer()
            heartbeat.check()
            run.write_stage(name, payload)
        warnings.extend(list(payload.get("warnings", [])))
        return payload

    def _produce_ocr(self, frames: list[dict]) -> dict[str, Any]:
        ocr = self.options.ocr_config()
        ocr_frames_result, ocr_warnings = ocr_frames(
            frames,
            ocr.language,
            ocr.enabled,
            self.progress,
            ocr.preprocess,
        )
        return {"frames": ocr_frames_result, "warnings": ocr_warnings}

    def _produce_redaction(self, frames: list[dict]) -> dict[str, Any]:
        redaction_warnings: list[dict[str, str]] = []
        redacted_frames = []
        for index, frame in enumerate(frames):
            copied = dict(frame)
            result = redact_text(str(copied.get("ocr_text", "")))
            copied["ocr_text"] = result.text
            redaction_warnings.extend(result.warnings)
            redacted_frames.append(copied)
            self.progress.update(
                "redaction",
                percent=((index + 1) / max(1, len(frames))) * 100,
                detail={"frame": index + 1, "frames": len(frames)},
            )
        self.progress.complete("redaction", detail={"frames": len(frames)})
        return {"frames": redacted_frames, "warnings": redaction_warnings}

    def _produce_local_vision(self, frames: list[dict]) -> dict[str, Any]:
        vision_frames, vision_warnings = interpret_frames_with_local_vision(
            frames,
            self.options,
            self.progress,
        )
        return {"frames": vision_frames, "warnings": vision_warnings}

    def _produce_generation(
        self,
        run: BundleRun,
        heartbeat: ProgressHeartbeat,
    ) -> dict[str, Any]:
        warnings = list(self.source.warnings)

        def produce_transcript() -> dict[str, Any]:
            transcript, transcript_warnings = transcribe_with_imports(
                self.source.resolved_path,
                run.scratch_dir,
                self.options,
                self.progress,
                duration_sec=self.source.duration_sec,
            )
            return {"transcript": transcript, "warnings": transcript_warnings}

        transcript_payload = self._run_stage(
            run,
            heartbeat,
            warnings,
            "transcript",
            ("transcription", "audio_extraction"),
            produce_transcript,
        )
        transcript = transcript_payload.get("transcript")

        def produce_frames() -> dict[str, Any]:
            frame_selection = self.options.frame_selection_config()
            frames, frame_warnings = select_keyframes(
                self.source.resolved_path,
                run.frames_dir,
                self.source.duration_sec,
                frame_selection.max_keyframes,
                frame_selection.min_interval_sec,
                frame_selection.max_static_window_sec,
                self.progress,
            )
            return {"frames": frames, "warnings": frame_warnings}

        frames_payload = self._run_stage(
            run,
            heartbeat,
            warnings,
            "frames",
            ("frame_selection",),
            produce_frames,
        )
        frames = frames_payload["frames"]

        ocr_payload = self._run_stage(
            run,
            heartbeat,
            warnings,
            "ocr",
            ("ocr",),
            lambda: self._produce_ocr(frames),
        )
        frames = ocr_payload["frames"]

        if self.options.redact_secrets:
            redaction_payload = self._run_stage(
                run,
                heartbeat,
                warnings,
                "redaction",
                ("redaction",),
                lambda: self._produce_redaction(frames),
            )
            frames = redaction_payload["frames"]
        else:
            self.progress.skip_cached("redaction", detail={"reason": "disabled"})

        if self.options.caption_frames:
            vision_payload = self._run_stage(
                run,
                heartbeat,
                warnings,
                "local_vision",
                ("local_vision",),
                lambda: self._produce_local_vision(frames),
            )
            frames = vision_payload["frames"]
        else:
            self.progress.skip_cached("local_vision", detail={"reason": "disabled"})

        self.progress.update("rendering", status="running")
        markdown = render_markdown(
            str(self.source.resolved_path),
            self.source.duration_sec,
            transcript,
            frames,
            warnings,
            getattr(self.source, "related_links", None),
        )
        self.progress.complete("rendering")
        self.progress.update("bundle_publish", status="running")
        run.write_render(markdown)
        if transcript is not None:
            run.write_transcript(transcript)
        manifest = manifest_document(
            self.source,
            self.options,
            transcript_present=transcript is not None,
            frames=frames,
            warnings=warnings,
        )
        self.progress.complete("bundle_publish", detail={"generation": run.generation_name})
        progress_summary = self.progress.aggregator.terminal_summary(self.progress.states)
        manifest["progress"] = progress_summary
        snapshot = run.commit(manifest)

        final_frames = []
        for frame in frames:
            copied = dict(frame)
            copied["path"] = str(snapshot.frames / Path(str(frame["path"])).name)
            final_frames.append(copied)
        return run_response(
            snapshot,
            self.source,
            final_frames,
            transcript is not None,
            warnings,
            cached=False,
            progress=progress_summary,
            job_id=self.options.job_id,
        )


def record_job(
    store: JobStore,
    job_id: str,
    tool: str,
    work: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    """Run `work` with a **job record** covering both ways it can end (R-17).

    One helper rather than a `start`/`finish` pair written out at each call site.
    Finding 12 was a missing failure path: the record was written where the
    result was, so every way of not producing a result wrote nothing. A pair a
    caller assembles by hand is a pair a caller can assemble half of, and the
    half that gets forgotten is the one nobody exercises.
    """
    store.start(job_id, tool)
    try:
        result = work()
    except BaseException as exc:
        store.finish(job_id, JobOutcome.failure(exc))
        raise
    store.finish(job_id, JobOutcome.success(result))
    return result


def process_resolved_source(
    source: Any,
    options: DistillOptions,
    output_root: Path | None = None,
    progress: ProgressReporter | None = None,
    *,
    tool: str = "process_local_video",
    lock_wait_sec: float = SINGLE_SOURCE_LOCK_WAIT_SEC,
) -> dict[str, Any]:
    try:
        output_root = output_root or validate_output_root(options.output_dir)
        progress = progress or ProgressReporter(emitter=progress_emitter(options.job_id))
        run = ProcessingRun(source, options, output_root, progress, tool, lock_wait_sec)
        return record_job(JobStore.open(output_root), options.job_id, tool, run.execute)
    finally:
        # The end of the media file's read lifetime (R-36). Transcription and
        # keyframe selection both read `source.resolved_path`, so the
        # **acquisition lease** taken to fetch it is only safe to release once
        # this returns - until then another run sharing the **lock key** but not
        # the **bundle key** could promote a replacement underneath the read.
        release_acquisition_lease(source)


VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}


def bool_arg(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


@dataclass
class BatchRunner:
    """The per-item loop of a batch tool. Owns no record of its own.

    The batch's **job record** is the parent job's, written around the whole
    batch by `record_job`, so an item that fails the batch is a batch that
    records a failure. `job_id` is carried only to derive each item's own job
    identifier.
    """

    job_id: str
    tool: str
    item_key: str
    items: list[str]
    continue_on_error: bool

    def run(
        self, process_item: Callable[[str, int], dict[str, Any]]
    ) -> tuple[list[dict], list[dict]]:
        results: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        for index, item in enumerate(self.items, start=1):
            try:
                result = process_item(item, index)
                result["batch_index"] = index
                results.append(result)
            except Exception as exc:
                errors.append({self.item_key: item, "message": str(exc)})
                if not self.continue_on_error:
                    raise
        return results, errors


def process_video_directory(args: dict[str, Any]) -> dict[str, Any]:
    directory = Path(str(args.get("path", ""))).expanduser()
    if not directory.exists() or not directory.is_dir():
        raise DistillError(
            "E_BAD_SOURCE", "source", "directory does not exist", {"path": str(directory)}
        )
    options = DistillOptions.from_args(args)
    max_items = int(args.get("max_items", 50))
    recursive = bool(args.get("recursive", False))
    pattern = "**/*" if recursive else "*"
    files = [
        path
        for path in sorted(directory.glob(pattern))
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ][:max_items]
    root = validate_output_root(options.output_dir)

    def process_item(path_text: str, index: int) -> dict[str, Any]:
        child_args = {**args, "path": path_text, "job_id": f"{options.job_id}-{index}"}
        return process_local_video(child_args, lock_wait_sec=BATCH_ITEM_LOCK_WAIT_SEC)

    runner = BatchRunner(
        job_id=options.job_id,
        tool="process_video_directory",
        item_key="path",
        items=[str(path) for path in files],
        continue_on_error=bool_arg(args.get("continue_on_error"), True),
    )

    def work() -> dict[str, Any]:
        results, errors = runner.run(process_item)
        return {
            "job_id": options.job_id,
            "directory": str(directory.resolve()),
            "video_count": len(files),
            "processed_count": len(results),
            "error_count": len(errors),
            "results": results,
            "errors": errors,
        }

    return record_job(JobStore.open(root), options.job_id, runner.tool, work)


def youtube_playlist_urls(url: str, max_items: int) -> list[str]:
    from .source import _run_ytdlp

    # _run_ytdlp adds `--socket-timeout`, a `--` terminator before the URL, and
    # maps a missing/hung yt-dlp onto clean errors.
    proc = _run_ytdlp(["--flat-playlist", "--print", "webpage_url"], url)
    if proc.returncode != 0:
        raise DistillError(
            "E_YTDLP",
            "youtube",
            "yt-dlp could not list playlist videos",
            {"stderr": proc.stderr.strip()},
        )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()][:max_items]


def playlist_folder_name(url: str) -> str:
    parsed = urlparse(url)
    playlist_id = parse_qs(parsed.query).get("list", [""])[0]
    raw_name = playlist_id or f"url-{hashlib.sha256(url.encode()).hexdigest()[:16]}"
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", raw_name).strip(".-")
    return safe_name[:120] or f"url-{hashlib.sha256(url.encode()).hexdigest()[:16]}"


def process_youtube_playlist(args: dict[str, Any]) -> dict[str, Any]:
    url = normalize_youtube_url(str(args.get("url", "")))
    if not url:
        raise DistillError("E_BAD_URL", "youtube", "url is required")
    from .source import ensure_youtube_host

    # Reject non-YouTube hosts / option-injection values before any yt-dlp call.
    ensure_youtube_host(url)
    options = DistillOptions.from_args({**args, "cache_mode": "fingerprint"})
    root = validate_output_root(options.output_dir)
    playlist_root = root / "playlists" / playlist_folder_name(url)
    ensure_safe_directory(playlist_root, root)
    max_items = int(args.get("max_items", 25))
    urls = youtube_playlist_urls(url, max_items)

    def process_item(video_url: str, index: int) -> dict[str, Any]:
        child_args = {
            **args,
            "url": video_url,
            "output_dir": str(playlist_root),
            "job_id": f"{options.job_id}-{index}",
        }
        return process_youtube_video(child_args, lock_wait_sec=BATCH_ITEM_LOCK_WAIT_SEC)

    runner = BatchRunner(
        job_id=options.job_id,
        tool="process_youtube_playlist",
        item_key="url",
        items=urls,
        continue_on_error=bool_arg(args.get("continue_on_error"), True),
    )

    def work() -> dict[str, Any]:
        results, errors = runner.run(process_item)
        summary = {
            "job_id": options.job_id,
            "playlist_url": url,
            "playlist_output_dir": str(playlist_root),
            "video_count": len(urls),
            "processed_count": len(results),
            "error_count": len(errors),
            "results": results,
            "errors": errors,
        }
        playlist_summary_path = playlist_root / "playlist.json"
        summary["playlist_summary_path"] = str(playlist_summary_path)
        playlist_summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        return summary

    return record_job(JobStore.open(root), options.job_id, runner.tool, work)


def _prune_policy(args: dict[str, Any]) -> PrunePolicy:
    """The **prune** policy a tool call names, validated on construction (R-03).

    `keep_generations=0` is finding 2's input and is refused where the policy is
    built rather than reinterpreted where it is used; `max_age_days=None` means
    no **bundle expiry** at all, which is not the same as a horizon of zero days.
    """
    return PrunePolicy(
        keep_generations=int(args.get("keep_generations", DEFAULT_KEEP_GENERATIONS)),
        max_age_days=float(args["max_age_days"]) if args.get("max_age_days") is not None else None,
    )


def cleanup_distill_cache(args: dict[str, Any]) -> dict[str, Any]:
    """Prune the cache under an output root, or report what pruning would remove.

    A dry run is the plan alone: a `PrunePlan` is advisory (D-023), so producing
    one and declining to apply it is exactly what a preview is, and there is no
    second code path whose answer could differ from what a real run would do.

    The payload reports what was skipped and how many directories were
    considered alongside what was deleted, so an empty result says which kind of
    empty it is (R-57). `cache-doctor` is the surface for asking that question
    without the ability to answer it destructively.
    """
    root = validate_output_root(args.get("output_dir"))
    store = BundleStore.open(root)
    plan = store.plan_prune(_prune_policy(args))
    dry_run = bool_arg(args.get("dry_run"), True)
    payload: dict[str, Any] = {
        "root": str(store.root),
        "dry_run": dry_run,
        "candidate_count": len(plan.targets),
        "candidates": [str(target.path) for target in plan.targets],
        "considered": plan.considered,
        "skipped": [skip.to_dict() for skip in plan.skipped],
        "skipped_count": len(plan.skipped),
    }
    if dry_run:
        return {**payload, "deleted_count": 0, "deleted": [], "results": []}

    outcome = store.apply_prune(plan)
    return {
        **payload,
        "deleted_count": len(outcome.deleted),
        "deleted": [str(path) for path in outcome.deleted],
        "results": [result.to_dict() for result in outcome.results],
    }


def cache_doctor(args: dict[str, Any]) -> dict[str, Any]:
    """Report the state of an output root, changing nothing under it (R-57).

    The root is validated but not created: a command whose whole purpose is to
    be safe to run first has to be safe to run against a path the user mistyped,
    and creating a directory to report that it is empty is a mutation.
    """
    root = validate_output_root(args.get("output_dir"), create=False)
    policy = _prune_policy(args)
    return inspect_cache(
        root,
        keep_generations=policy.keep_generations,
        max_age_days=policy.max_age_days,
    )


def get_job_status(args: dict[str, Any]) -> dict[str, Any]:
    root = validate_output_root(args.get("output_dir"))
    job_id = str(args.get("job_id", ""))
    # An identifier outside the domain names no record, so it is refused here
    # rather than mapped onto whichever record it happens to resemble (R-18).
    record = JobStore.open(root).read(job_id)
    if record is None:
        raise DistillError("E_JOB_NOT_FOUND", "job", "job status not found", {"job_id": job_id})
    return record.to_dict()


def transcribe_with_imports(
    video_path: Path,
    work_dir: Path,
    options: DistillOptions,
    progress: ProgressCounter | ProgressReporter,
    duration_sec: float,
) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    # No default: SourceInfo.duration_sec is always present, and omitting it here
    # would silently disable ffmpeg -progress instead of failing loudly.
    from .transcript import transcribe_video

    return transcribe_video(
        video_path,
        work_dir,
        options.whisper_model,
        options.whisper_language,
        options.vad_filter,
        progress,
        duration_sec,
    )


def interpret_frames_with_local_vision(
    frames: list[dict],
    options: DistillOptions,
    progress: ProgressReporter | None = None,
) -> tuple[list[dict], list[dict[str, str]]]:
    interpreter = FrameInterpreter(
        config=options.local_vision_config(),
        redact_secrets=options.redact_secrets,
        progress=progress,
        probe=probe_local_vision,
        try_interpret=try_interpret_image,
    )
    return interpreter.interpret(frames)


def tool_registry() -> dict[str, ToolSpec]:
    return {
        "process_local_video": ToolSpec(
            TOOLS["process_local_video"],
            process_local_video,
        ),
        "process_youtube_video": ToolSpec(
            TOOLS["process_youtube_video"],
            process_youtube_video,
        ),
        "process_video_directory": ToolSpec(
            TOOLS["process_video_directory"],
            process_video_directory,
        ),
        "process_youtube_playlist": ToolSpec(
            TOOLS["process_youtube_playlist"],
            process_youtube_playlist,
        ),
        "cleanup_cache": ToolSpec(TOOLS["cleanup_cache"], cleanup_distill_cache),
        "cache_doctor": ToolSpec(TOOLS["cache_doctor"], cache_doctor),
        "get_job_status": ToolSpec(TOOLS["get_job_status"], get_job_status),
    }


def call_registered_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    spec = tool_registry().get(name)
    if spec is None:
        raise DistillError("E_UNKNOWN_TOOL", "protocol", f"Unknown tool: {name}")
    return spec.handler(args)


def list_tools() -> None:
    print(json.dumps({"tools": sorted(tool_registry())}, indent=2))


def configured_timeout_ms() -> int:
    return DEFAULT_CONFIGURED_TIMEOUT_MS


def timeout_diagnostics(effective_timeout_ms: int | None = None) -> dict[str, Any]:
    configured = configured_timeout_ms()
    effective = effective_timeout_ms
    source = "argument"
    if effective is None:
        raw_effective = os.environ.get(TIMEOUT_ENV)
        if raw_effective:
            effective = int(raw_effective)
            source = TIMEOUT_ENV
        else:
            effective = configured
            source = "app.config.json"
    return {
        "configured_timeout_ms": configured,
        "effective_timeout_ms": effective,
        "effective_timeout_source": source,
        "effective_meets_configured": effective >= configured,
        "assumption": "A-004",
    }


def run_timeout_probe(probe_ms: int) -> dict[str, Any]:
    if probe_ms < 0:
        raise DistillError(
            "E_BAD_ARGUMENT",
            "timeout",
            "timeout probe duration must be non-negative",
            {"probe_ms": probe_ms},
        )
    long_probe_enabled = os.environ.get(LONG_TIMEOUT_PROBE_ENV) == "1"
    if probe_ms > TIMEOUT_PROBE_LIMIT_MS and not long_probe_enabled:
        raise DistillError(
            "E_BAD_ARGUMENT",
            "timeout",
            "long timeout probes require DISTILL_ENABLE_LONG_TIMEOUT_PROBE=1",
            {"probe_ms": probe_ms, "limit_ms": TIMEOUT_PROBE_LIMIT_MS},
        )
    started = time.monotonic()
    time.sleep(probe_ms / 1000)
    elapsed_ms = int((time.monotonic() - started) * 1000)
    return {
        **timeout_diagnostics(),
        "probe_requested_ms": probe_ms,
        "probe_elapsed_ms": elapsed_ms,
        "long_probe_enabled": long_probe_enabled,
    }


def local_vision_diagnostics(args: dict[str, Any] | None = None) -> dict[str, Any]:
    config = local_vision_config_from_args(args or {})
    probe = probe_local_vision(config)
    return {
        "config": config.public_dict(),
        "probe": {
            "available": probe.available,
            "backend": probe.backend,
            "model": probe.model,
            "base_url": probe.base_url,
            "code": probe.code,
            "message": probe.message,
            "detail": probe.detail,
        },
        "setup_command": f"rapid-mlx serve {config.model}",
        "rapid_mlx_note": "Distill posts OpenAI-compatible chat completions to the configured base_url (default http://127.0.0.1:8000/v1). The server must already be running.",
    }
