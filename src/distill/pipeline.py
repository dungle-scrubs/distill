"""Distill pipeline orchestration.

This module owns the package-facing processing functions used by the CLI.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .artifacts import FrameArtifact, Transcript
from .bundle_store import (
    BATCH_ITEM_LOCK_WAIT_SEC,
    DEFAULT_KEEP_GENERATIONS,
    SINGLE_SOURCE_LOCK_WAIT_SEC,
    BundleStore,
    PrunePolicy,
    atomic_write_text,
    ensure_safe_directory,
)
from .cache_doctor import inspect_cache
from .config import resolve_options
from .errors import DistillError, WarningRecord
from .filtered_view import filtered_view_markdown
from .frame_selection import select_keyframes
from .job_store import JobOutcome, JobStore
from .local_vision import (
    MAX_SOCKET_TIMEOUT_SEC,
    FrameInterpreter,
    local_vision_config_from_args,
    probe_local_vision,
    try_interpret_image_after_probe,
)
from .ocr import ocr_frames
from .options import (
    GENERAL_OPTION_NAMES,
    DistillOptions,
    validated_count,
    validated_number,
)
from .progress import (
    ProgressCounter,
    ProgressReporter,
)
from .render import render_markdown

# Re-exported for monkeypatch tests — used via distill.pipeline.<name>
__all__ = [
    "select_keyframes",
    "ocr_frames",
    "render_markdown",
    "ProcessingRun",
    "REKEY_BOUND_REASON",
    "cache_hit_progress_summary",
    "progress_summary_is_terminal",
    "revalidation_is_owed",
]
from .run_orchestrator import (  # noqa: F401  re-exported — deep module
    REKEY_BOUND_REASON,
    ProcessingRun,
    cache_hit_progress_summary,
    progress_summary_is_terminal,
    revalidation_is_owed,
)
from .source import (  # noqa: F401  re-exported for test seam
    ChainRevalidation,
    candidate_in_hand,
    normalize_youtube_url,
    release_acquisition_lease,
    resolve_source_for_processing,
    revalidate_chain,
    source_path_kind,
    validate_output_root,
)
from .vision_selection import (  # noqa: F401  deep vision chain walk
    SelectionOutcome,
    VisionSelection,
)
from .youtube import ensure_youtube_host, youtube_playlist_urls

TOOLS = {
    "process_local_video": "Process a local video into a transcript/keyframe markdown bundle",
    "process_youtube_video": "Download and process one YouTube video into a transcript/keyframe markdown bundle",
    "process_video_directory": "Process video files in a local directory into Distill bundles",
    "process_youtube_playlist": "Process videos from a YouTube playlist or channel URL",
    "cleanup_cache": "Prune old Distill cache bundles and generations",
    "cache_doctor": "Report bundles, markers, generations, locks and a prune preview for an output root",
    "get_job_status": "Read a Distill job status record by job id",
}
LOGGER = logging.getLogger(__name__)

PIPELINE_EVENT_TYPE = "distill.pipeline"


def _pipeline_log(event: str, **detail: Any) -> None:
    """Emit one run-orchestration event: a decision `execute` took about a run.

    Metadata only, in the shape `bundle_store` and `source` use, so one log
    stream answers "what happened to this **bundle key**" across the three
    modules that act on one. `bundle_key` is the correlation field for that
    reason: an event here sits between the store's `lock_acquired` and its
    `generation_committed` for the same key, and an operator reading why a run
    behaved as it did needs the three in one order.

    Nothing a stage produced is recorded here. No **extracted text** has passed
    a **redaction sink** by the time these fire, and none of these decisions is
    about content anyway.
    """
    LOGGER.debug(
        json.dumps(
            {
                "type": PIPELINE_EVENT_TYPE,
                "event": event,
                "detail": {"pid": os.getpid(), **detail},
            },
            sort_keys=True,
        )
    )


DEFAULT_CONFIGURED_TIMEOUT_MS = 5_400_000

TIMEOUT_ENV = "DISTILL_EFFECTIVE_TIMEOUT_MS"
LONG_TIMEOUT_PROBE_ENV = "DISTILL_ENABLE_LONG_TIMEOUT_PROBE"
TIMEOUT_PROBE_LIMIT_MS = 1_000

TIMEOUT_PROBE_CEILING_MS = int(MAX_SOCKET_TIMEOUT_SEC * 1000)
"""The longest sleep that can be asked for, as opposed to the longest one wanted.

`TIMEOUT_PROBE_LIMIT_MS` is a policy - a probe longer than a second wants asking
for twice - and the opt-out lifts it. This is not a policy: CPython holds a
duration as nanoseconds in a signed 64-bit integer, so past this point
`time.sleep` answers `OverflowError` rather than sleeping, whatever anybody
opted into. It is the same fact about the same representation that gives
`--local-vision-timeout-sec` its ceiling, which is why it is derived from that
constant rather than written out again.
"""


@dataclass(frozen=True)
class ToolSpec:
    description: str
    handler: Callable[[dict[str, Any]], dict[str, Any]]


class DistillSession:
    """One tool call, answered as an MCP-style envelope rather than by raising.

    The error half is the **fatal error** record itself, field for field (R-46).
    It used to be that record serialized into a `message` string, so every
    reader that wanted the code had to know the message was secretly JSON and
    parse it back out - a code and a stage that travelled the whole run as
    structured fields, flattened at the last surface before a reader.
    """

    def call_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        try:
            result = call_registered_tool(name, args)
            return {"result": {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}}
        except DistillError as exc:
            return {"error": exc.to_dict()}
        except Exception as exc:
            return {"error": DistillError.from_unexpected(exc).to_dict()}


def acquire_and_process(
    source_type: str,
    value: str,
    options: DistillOptions,
    root: Path,
    *,
    progress: ProgressReporter | None,
    tool: str,
    lock_wait_sec: float,
) -> dict[str, Any]:
    """Acquire the source and produce its **generation**: everything a run does.

    Both halves belong to one job, which is why they are one function: the
    **job record** wraps this, and a record that started after acquisition
    described the second half of a run as though it were the whole of it.

    A YouTube resolution reports the **output root** it validated and a local
    one reports none, so the caller's root stands in - the same path, validated
    before the record was opened rather than in the middle of the work.

    `lock_wait_sec` is spent twice, at the two locks a run takes: the
    **acquisition lease** on the source and the run lock on the **bundle key**.
    Both are the same caller's decision (D-044) - a run a user is watching waits
    for either, and a batch item gives up on either - and acquisition is the one
    that a second run of the same video reaches first (finding 4-opus).
    """
    resolution = resolve_source_for_processing(
        source_type,
        value,
        options,
        progress=progress,
        lock_wait_sec=lock_wait_sec,
    )
    return process_resolved_source(
        resolution.source,
        options,
        resolution.output_root or root,
        progress=resolution.progress,
        tool=tool,
        lock_wait_sec=lock_wait_sec,
    )


def process_local_video(
    args: dict[str, Any], *, lock_wait_sec: float = SINGLE_SOURCE_LOCK_WAIT_SEC
) -> dict[str, Any]:
    options = DistillOptions.from_args(args)
    root = validate_output_root(options.output_dir)

    def work() -> dict[str, Any]:
        return acquire_and_process(
            "local",
            str(args.get("path", "")),
            options,
            root,
            progress=None,
            tool="process_local_video",
            lock_wait_sec=lock_wait_sec,
        )

    return record_job(JobStore.open(root), options.job_id, "process_local_video", work)


def process_youtube_video(
    args: dict[str, Any], *, lock_wait_sec: float = SINGLE_SOURCE_LOCK_WAIT_SEC
) -> dict[str, Any]:
    options = DistillOptions.from_args({**args, "cache_mode": "fingerprint"})
    root = validate_output_root(options.output_dir)
    progress = ProgressReporter(emitter=progress_emitter(options.job_id))

    def work() -> dict[str, Any]:
        return acquire_and_process(
            "youtube",
            str(args.get("url", "")),
            options,
            root,
            progress=progress,
            tool="process_youtube_video",
            lock_wait_sec=lock_wait_sec,
        )

    return record_job(JobStore.open(root), options.job_id, "process_youtube_video", work)


def progress_emitter(job_id: str) -> Any:
    def emit(event: Any) -> None:
        payload = event.to_dict()
        payload["job_id"] = job_id
        print(json.dumps(payload, sort_keys=True), file=sys.stderr, flush=True)

    return emit


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

    One envelope per tool call, opened as early as a record can be written -
    after the arguments and the **output root** are validated, because an
    unusable identifier or an unusable root leaves nowhere to write, and before
    anything else. Everything after that point is work that can fail: a probe, a
    download, a directory scan, a playlist listing. Wrapped further in, the
    record covered only what happened after a source already existed, which is
    finding 12 again for the whole acquisition half of a run. A second envelope
    nested inside this one is refused by `JobStore.start` rather than trusted
    not to exist.
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
    """Produce a **generation** from a source that has already been acquired.

    Records nothing: the **job record** belongs to the tool call, which began
    before the source existed. An envelope here would be the second one over the
    same run - and, since a held identifier cannot be started twice, would make
    every run fail rather than merely mis-report one.
    """
    # The lease is released at the end of the media file's read lifetime (R-36).
    # Transcription and keyframe selection both read `source.resolved_path`, so
    # the **acquisition lease** taken to fetch it is only safe to release once
    # this returns - until then another run sharing the **lock key** but not the
    # **bundle key** could promote a replacement underneath the read.
    #
    # Two arms rather than one `finally`, because the two cases owe different
    # things. On the way out of a failure the release must not raise, or it
    # replaces the failure it is cleaning up after; on the way out of a success
    # it must, because nothing else would report a lease that was not given up.
    try:
        # <!-- P3-D-015 --> The options the **endpoint chain** walk settled on,
        # if it ran. The source's **bundle key** was derived from these, so a
        # run that went on carrying its original options would build a
        # **manifest** naming whichever endpoint the chain listed first while
        # publishing under the key of the one that answered.
        options = getattr(source, "resolved_options", None) or options
        output_root = output_root or validate_output_root(options.output_dir)
        progress = progress or ProgressReporter(emitter=progress_emitter(options.job_id))
        run = ProcessingRun(source, options, output_root, progress, tool, lock_wait_sec)
        result = run.execute()
    except BaseException as exc:
        release_acquisition_lease(source, during=exc)
        raise
    release_acquisition_lease(source)
    return result


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
        """Every item, and the whole **fatal error** record for each one that failed.

        R-46: an item's failure is reported as the record every other surface
        reports, code and stage included. It was flattened to `str(exc)`, so a
        batch of twenty-five was a list of sentences - nothing could tell an
        `E_LOCKED` item that a re-run would pick up from an `E_BAD_MEDIA` item
        that will fail the same way forever, which is the one distinction the
        report exists to make.

        `batch_index` on the error for the same reason it is on the result: with
        `continue_on_error` the two lists are neither the same length nor in
        step, so position is not a handle and the index has to be carried.

        An item that failed without a code is converted here rather than left
        uncoded, on the same terms as the CLI boundary and through the same
        mapping - a report where some entries carry a code and others do not is
        a report a caller still has to branch on twice.
        """
        results: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        for index, item in enumerate(self.items, start=1):
            try:
                result = process_item(item, index)
                result["batch_index"] = index
                results.append(result)
            except Exception as exc:
                failure = (
                    exc if isinstance(exc, DistillError) else DistillError.from_unexpected(exc)
                )
                errors.append({self.item_key: item, "batch_index": index, **failure.to_dict()})
                if not self.continue_on_error:
                    raise
        return results, errors


def process_video_directory(args: dict[str, Any]) -> dict[str, Any]:
    options = DistillOptions.from_args(args)
    root = validate_output_root(options.output_dir)
    max_items = validated_count("max_items", args.get("max_items", 50))
    recursive = bool(args.get("recursive", False))
    continue_on_error = bool_arg(args.get("continue_on_error"), True)

    def process_item(path_text: str, index: int) -> dict[str, Any]:
        child_args = {**args, "path": path_text, "job_id": f"{options.job_id}-{index}"}
        return process_local_video(child_args, lock_wait_sec=BATCH_ITEM_LOCK_WAIT_SEC)

    def work() -> dict[str, Any]:
        # Inside the record, because scanning a directory is work: the path may
        # not be one, and a batch that finds nothing to do still ran.
        directory = Path(str(args.get("path", ""))).expanduser()
        if source_path_kind(directory) != "directory":
            raise DistillError(
                "E_BAD_SOURCE", "source", "directory does not exist", {"path": str(directory)}
            )
        pattern = "**/*" if recursive else "*"
        files = [
            path
            for path in sorted(directory.glob(pattern))
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        ][:max_items]
        runner = BatchRunner(
            job_id=options.job_id,
            tool="process_video_directory",
            item_key="path",
            items=[str(path) for path in files],
            continue_on_error=continue_on_error,
        )
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

    return record_job(JobStore.open(root), options.job_id, "process_video_directory", work)


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
    # Reject non-YouTube hosts / option-injection values before any yt-dlp call.
    ensure_youtube_host(url)
    options = DistillOptions.from_args({**args, "cache_mode": "fingerprint"})
    root = validate_output_root(options.output_dir)
    playlist_root = root / "playlists" / playlist_folder_name(url)
    max_items = validated_count("max_items", args.get("max_items", 25))
    continue_on_error = bool_arg(args.get("continue_on_error"), True)

    def process_item(video_url: str, index: int) -> dict[str, Any]:
        child_args = {
            **args,
            "url": video_url,
            "output_dir": str(playlist_root),
            "job_id": f"{options.job_id}-{index}",
        }
        return process_youtube_video(child_args, lock_wait_sec=BATCH_ITEM_LOCK_WAIT_SEC)

    def work() -> dict[str, Any]:
        # Inside the record: creating the playlist root writes, and listing the
        # playlist is a yt-dlp call - the parent job's first real step and the
        # first thing that can fail without a single item ever being tried.
        ensure_safe_directory(playlist_root, root)
        urls = youtube_playlist_urls(url, max_items)
        runner = BatchRunner(
            job_id=options.job_id,
            tool="process_youtube_playlist",
            item_key="url",
            items=urls,
            continue_on_error=continue_on_error,
        )
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
        # State a reader outside this process picks up, written through the one
        # checked atomic writer (R-14, R-16). A bare `write_text` here followed a
        # link pre-created at `playlist.json` and wrote over whatever it named,
        # and it was the last durable write in the tree asking nothing.
        atomic_write_text(
            playlist_summary_path,
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            root=root,
        )
        return summary

    return record_job(JobStore.open(root), options.job_id, "process_youtube_playlist", work)


def _prune_policy(args: dict[str, Any]) -> PrunePolicy:
    """The **prune** policy a tool call names, validated on construction (R-03).

    `keep_generations=0` is finding 2's input and is refused where the policy is
    built rather than reinterpreted where it is used; `max_age_days=None` means
    no **bundle expiry** at all, which is not the same as a horizon of zero days.

    Handed over unconverted, which is the whole of R-46's half here. The `int()`
    and `float()` that used to stand in front of the policy did two wrong things
    at once: they raised a bare `ValueError` on text no number could be made of
    - `--args '{"max_age_days": "soon"}'` was a traceback - and where they *did*
    convert, the policy was shown a number the caller never wrote, so its own
    validation was judging this function's guess. The values arrive as the
    caller sent them and `PrunePolicy` is the single place that says what a
    retention policy may be (R-03).
    """
    return PrunePolicy(
        keep_generations=args.get("keep_generations", DEFAULT_KEEP_GENERATIONS),
        max_age_days=args.get("max_age_days"),
    )


def configured_args(args: dict[str, Any]) -> dict[str, Any]:
    """`args` with the configured layers folded in: CLI > env > file > default.

    A processing tool gets this inside `DistillOptions.from_args`, where the
    options it resolves are then cast and validated. The tools that read an
    output root without building options - prune, doctor, job status - ask for
    it here, so `distill cache-doctor` reports on the root a run would publish
    into rather than on the default one while runs go somewhere else.
    """
    return resolve_options(args, general_keys=GENERAL_OPTION_NAMES)


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
    root = validate_output_root(configured_args(args).get("output_dir"))
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
    root = validate_output_root(configured_args(args).get("output_dir"), create=False)
    policy = _prune_policy(args)
    return inspect_cache(
        root,
        keep_generations=policy.keep_generations,
        max_age_days=policy.max_age_days,
    )


def filtered_view(args: dict[str, Any]) -> dict[str, Any]:
    """The non-authoritative filtered view of a published **bundle** (D-006).

    The root is resolved here rather than in `filtered_view.py` for the reason
    `cache_doctor` is: the read-only module is handed a root the option layers
    already settled, so `distill filtered-view` reads the root a run would have
    published into rather than the default one while runs go elsewhere. Not
    created, either - a command that only reads has to be safe to point at a
    path the operator mistyped, and creating a directory to report that no
    bundle is under it is a mutation.

    JSON, and the document under `markdown`, because R-46's contract is one
    shape on stdout: a command that printed a bare document would be the one an
    operator's parser has to special-case. The root and key travel beside it so
    a redirected view says which bundle it is a view of - the banner says what
    the document omits and nothing about where it came from.
    """
    root = validate_output_root(configured_args(args).get("output_dir"), create=False)
    bundle_key = str(args.get("bundle_key", ""))
    return {
        "root": str(root),
        "bundle_key": bundle_key,
        "markdown": filtered_view_markdown(root, bundle_key),
    }


def get_job_status(args: dict[str, Any]) -> dict[str, Any]:
    root = validate_output_root(configured_args(args).get("output_dir"))
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
) -> tuple[dict[str, Any] | None, list[WarningRecord]]:
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
    frames: list[FrameArtifact],
    options: DistillOptions,
    progress: ProgressReporter | None = None,
    *,
    transcript: Transcript | None = None,
) -> tuple[list[FrameArtifact], list[WarningRecord]]:
    """Interpret every frame, under the **redaction** policy the frames carry.

    The interpreter is told nothing about redaction. `--no-redact-secrets` is
    recorded on each **frame artifact** by `select_keyframes` and travels with
    it, so the model's words are redacted where they enter the carrier (R-19)
    rather than by a helper the vision pass had to remember to call.

    The transcript is the salience context (D-003): each frame is judged
    against the speech around its timestamp when `frame_salience` is on. A
    missing transcript means absent salience, never a judgment against
    nothing.
    """
    interpreter = FrameInterpreter(
        config=options.local_vision_config(),
        progress=progress,
        probe=probe_local_vision,
        try_interpret=try_interpret_image_after_probe,
        frame_salience=options.frame_salience,
    )
    return interpreter.interpret(
        frames,
        transcript_segments=None if transcript is None else transcript.segments,
    )


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


def list_tools() -> dict[str, Any]:
    """The registered tool names. Returned, not printed.

    Printing from here put one command's output outside the CLI's own emission
    path, so whatever that path promises about stdout - the flush inside the
    error boundary, the answer to a caller that stopped reading - held for every
    command except this one.
    """
    return {"tools": sorted(tool_registry())}


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
    """Sleep for `probe_ms`, or say why that is not a duration.

    Three refusals, all `E_BAD_ARGUMENT` at the `timeout` stage, because all
    three are the operator's own number and none of them is a defect in Distill:
    below zero is not a duration, above `TIMEOUT_PROBE_LIMIT_MS` is a wait long
    enough to want asking for twice, and above `TIMEOUT_PROBE_CEILING_MS` is a
    duration no clock can represent.

    The ceiling is the same escape `NumericDomain.ceiling` closed at the option
    boundary, at the one door that does not pass through it: the long-probe
    opt-out lifts the *policy* limit, and past that the number reached
    `time.sleep` and came back as `OverflowError: timestamp out of range for
    platform time_t` - reported by the CLI's catch-all as an internal fault,
    with the argument's name thrown away.
    """
    if probe_ms < 0:
        raise DistillError(
            "E_BAD_ARGUMENT",
            "timeout",
            "timeout probe duration must be non-negative",
            {"probe_ms": probe_ms},
        )
    if probe_ms >= TIMEOUT_PROBE_CEILING_MS:
        raise DistillError(
            "E_BAD_ARGUMENT",
            "timeout",
            "timeout probe duration is longer than a sleep can be asked for",
            {"probe_ms": probe_ms, "ceiling_ms": TIMEOUT_PROBE_CEILING_MS},
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
    """What the configured vision endpoint resolves to, and whether it answers.

    A timeout named *here* is validated before the config layer sees it, on the
    same terms as the run path and against the same domain. The config layer's
    contract is to coerce - a config file naming an unusable timeout should not
    stop a run - and that is the wrong contract for a number an operator just
    typed at the command whose whole purpose is to report what their arguments
    resolve to: `-5`, `0`, `nan` and `1e300` all printed the default and exited
    0, which told the operator their setting was in force.

    Only the override. A value read from a config file still takes the config
    layer's answer, because that scoping is the recorded decision and not an
    oversight.
    """
    args = dict(args or {})
    if "local_vision_timeout_sec" in args:
        args["local_vision_timeout_sec"] = validated_number(
            "local_vision_timeout_sec", args["local_vision_timeout_sec"]
        )
    config = local_vision_config_from_args(args)
    chain = config.endpoints or (config,)
    entries: list[dict[str, Any]] = []
    selected: Any = None
    for index, endpoint in enumerate(chain):
        # Every entry is asked, even after one has answered. This command is
        # for diagnosing a chain, and "why is the third endpoint never used"
        # is unanswerable if the walk stopped at the first that worked.
        probe = probe_local_vision(endpoint)
        if probe.available and selected is None:
            selected, outcome = probe, "selected"
        else:
            outcome = "available" if probe.available else "unavailable"
        entries.append(
            {
                "entry": index,
                "model": endpoint.model,
                "base_url": endpoint.base_url,
                "outcome": outcome,
                # The name, never the value. A name is diagnostics and this
                # command's output is meant to be pasteable; a value is a
                # secret, and D-007 keeps it out of every text form.
                "credential_env": endpoint.credential_env,
                "credential_set": endpoint.credential is not None,
                "code": probe.code,
                "message": probe.message,
                "detail": probe.detail,
            }
        )
    answered = selected is not None
    # An endpoint that rejected a credential *answered*: something is listening
    # and it said no. Telling an operator to start a server points at the wrong
    # problem, and does so at the moment they are trying to work out what the
    # right one is. `local_vision_backend_unsupported` is the same shape - a
    # configuration answer, not a missing process.
    responded = any(
        entry["code"] in {"local_vision_auth_rejected", "local_vision_backend_unsupported"}
        for entry in entries
    )
    report: dict[str, Any] = {
        "config": config.public_dict(),
        "endpoints": entries,
        "probe": {
            "available": answered,
            "backend": (selected or probe).backend,
            "model": (selected or probe).model,
            "base_url": (selected or probe).base_url,
            "code": (selected or probe).code,
            "message": (selected or probe).message,
            "detail": (selected or probe).detail,
        },
    }
    if not answered and not responded:
        # Only when nothing answered. Telling an operator whose endpoint replied
        # - and whose credential was rejected - to start a server points at the
        # wrong diagnosis, which is the failure Gate 3 asks about.
        report["setup_command"] = f"rapid-mlx serve {chain[0].model}"
        report["rapid_mlx_note"] = (
            "Distill posts OpenAI-compatible chat completions to the configured "
            "base_url (default http://127.0.0.1:8000/v1). The server must already "
            "be running."
        )
    return report
