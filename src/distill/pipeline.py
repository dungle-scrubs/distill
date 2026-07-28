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
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .artifacts import FrameArtifact, RedactionState, Transcript
from .bundle_store import (
    BATCH_ITEM_LOCK_WAIT_SEC,
    DEFAULT_KEEP_GENERATIONS,
    SINGLE_SOURCE_LOCK_WAIT_SEC,
    BundleRun,
    BundleSnapshot,
    BundleStore,
    PrunePolicy,
    atomic_write_text,
    ensure_safe_directory,
)
from .cache_doctor import inspect_cache
from .errors import DistillError, WarningRecord, aggregate_warnings
from .frame_selection import select_keyframes
from .job_store import JobOutcome, JobStore
from .local_vision import (
    MAX_SOCKET_TIMEOUT_SEC,
    FrameInterpreter,
    local_vision_config_from_args,
    probe_local_vision,
    try_interpret_image,
)
from .ocr import ocr_frames
from .options import DistillOptions, validated_count, validated_number
from .progress import (
    TERMINAL_PROGRESS_STATUSES,
    OverallProgressAggregator,
    ProgressCounter,
    ProgressHeartbeat,
    ProgressReporter,
)
from .render import render_markdown
from .response import manifest_document, response_frames, run_response
from .source import (
    normalize_youtube_url,
    release_acquisition_lease,
    resolve_source_for_processing,
    source_path_kind,
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
                    run.abandon(_abandon_reason(exc), during=exc)
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

    def _run_stage[StageValue](
        self,
        run: BundleRun,
        heartbeat: ProgressHeartbeat,
        warnings: list[WarningRecord],
        name: str,
        skipped_mechanisms: tuple[str, ...],
        producer: Callable[[], dict[str, Any]],
        revive: Callable[[dict[str, Any]], _Recovered[StageValue] | None],
    ) -> StageValue:
        """Recover a **stage result** or produce one, recording it either way.

        What a caller gets back is what the stage *speaks* - carriers - and
        never the payload they were read out of, which is what makes the
        recovery total. `revive` is the stage's own answer to "is this the shape
        I produce", asked of a resumed document and of a fresh one alike, and
        the only caller of it: nothing downstream subscripts a payload, so there
        is no field left for a document to be missing.

        Two reasons a stage runs, and this is deliberately one branch. `None`
        from `read_stage` is the store's: the run is not resuming, nothing was
        recorded, what was recorded is unreadable, or it failed the checks R-23
        puts a resumed document through. `None` from `revive` is this module's:
        the envelope held up but the payload inside it is not the shape this
        stage reads - an older Distill's spelling, a truncated write, scratch
        somebody edited. D-030 answers both the same way, which is compute it. A
        resume that could *end* a run would be worse than the trust it replaced
        (RV-8), and a shape check that raised would be exactly that with an
        extra step (boundary review, finding 2).

        A payload this run just produced is a different matter. If the stage's
        own output is not the shape the stage reads, nothing about resuming is
        wrong and recomputing would not fix it, so it is reported as the defect
        it is rather than retried - and reported before it is recorded, because
        scratch this run cannot read is scratch no later run could read either.
        """
        recorded = run.read_stage(name)
        if recorded is not None:
            recovered = _revived(recorded, revive)
            if recovered is not None:
                for mechanism in skipped_mechanisms:
                    self.progress.skip_cached(mechanism, detail={"source": "partial_resume"})
                warnings.extend(_stage_warnings(recorded))
                return recovered.value
            run.discard_stage(name, "payload_shape_unusable")
        payload = producer()
        heartbeat.check()
        produced = _revived(payload, revive)
        if produced is None:
            raise DistillError(
                "E_BAD_STAGE_PAYLOAD",
                "pipeline",
                f"the {name} stage produced a payload of a shape it does not read",
                {"stage": name},
            )
        run.write_stage(name, payload, redaction=self._redaction_policy)
        warnings.extend(_stage_warnings(payload))
        return produced.value

    def _recovered_frames(
        self, payload: dict[str, Any]
    ) -> _Recovered[list[FrameArtifact]] | None:
        """The **frame artifacts** `payload` holds, or `None` if it holds none.

        A stage that just ran hands back carriers. A stage a **resume**
        recovered hands back the documents `bundle_store` serialized them into,
        because a stage result is JSON on disk and nothing else. Both are the
        same frames, so the difference is settled here rather than by every
        consumer downstream.

        Total over both, which is the point: this is handed a document another
        process wrote, so `frames` may be absent, may not be a list, may hold
        strings, and may hold mappings that are missing a field a frame has.
        None of those is a run's business - the frames stage can always select
        the keyframes again - so each of them is `None` here and a recomputation
        upstream.

        `redaction` is this run's policy, applied to a rebuilt frame whatever
        the document says about itself: R-23's premise is that a stage result's
        claims are input rather than facts.
        """
        items = payload.get("frames")
        if not isinstance(items, list | tuple):
            return None
        frames: list[FrameArtifact] = []
        for item in items:
            if isinstance(item, FrameArtifact):
                frames.append(item)
                continue
            if not isinstance(item, Mapping):
                return None
            try:
                frames.append(
                    FrameArtifact.from_document(item, redaction=self._redaction_policy)
                )
            except Exception:
                # Deliberately broad, for the reason `read_stage_result`'s guard
                # is: every way a document can fail to become a carrier - a
                # missing field, a type a carrier cannot freeze, a nesting depth
                # that exhausts the walk - costs the same recomputation, and
                # there is no way to fail here that is worth ending a run over.
                return None
        return _Recovered(frames)

    def _recovered_transcript(self, payload: dict[str, Any]) -> _Recovered[Transcript | None] | None:
        """The **transcript** carrier `payload` holds, on the same terms.

        A run that produced no transcript carries nothing at all, which is why
        the answer is wrapped: `_Recovered(None)` is a stage whose value is
        legitimately absent, and a bare `None` is a payload this stage cannot
        read. Collapsing the two would make "the video had no speech" and "the
        scratch is unusable" the same answer.

        Which is also why the field has to be *there*. A recorded `null` is this
        stage saying it found no speech; a payload with no `transcript` key at
        all is a payload that has not answered, and reading it as "no speech"
        would publish a **generation** with no transcript for a video that has
        one, having skipped the transcription that would have found it.
        """
        if "transcript" not in payload:
            return None
        item = payload["transcript"]
        if item is None or isinstance(item, Transcript):
            return _Recovered(item)
        if not isinstance(item, Mapping):
            return None
        try:
            return _Recovered(Transcript.from_document(item, redaction=self._redaction_policy))
        except Exception:
            return None

    def _produce_ocr(self, frames: list[FrameArtifact]) -> dict[str, Any]:
        """Read every **keyframe**'s image text onto the artifact that names it.

        No translation step is left here. `ocr_frames` takes the carriers
        `select_keyframes` produced and hands back carriers, so the **redaction**
        policy runs where the text enters (R-19) and the **stage result** this
        payload becomes cannot hold it raw - which is finding 4, closed at the
        seam it was hiding in rather than downstream of it.
        """
        ocr = self.options.ocr_config()
        read, ocr_warnings = ocr_frames(
            frames,
            ocr.language,
            ocr.enabled,
            self.progress,
            ocr.preprocess,
        )
        return {"frames": read, "warnings": ocr_warnings}

    @property
    def _redaction_policy(self) -> RedactionState:
        """The **redaction** policy this run's carriers are constructed under.

        `DISABLED` is `--no-redact-secrets` and is recorded in what the run
        publishes; anything else means the policy runs (D-020).
        """
        return (
            RedactionState.NOT_APPLIED
            if self.options.redact_secrets
            else RedactionState.DISABLED
        )

    def _carry_transcript(self, transcript: Any) -> tuple[Transcript | None, list[dict]]:
        """Put the **transcript** through a carrier on the same terms (R-21).

        Finding 15: no stage covered the transcript, so a secret somebody said
        out loud was published verbatim in the transcript the store writes. It
        is extracted text like any other - the person who recorded the video
        chose the words - and the carrier is what makes "the same terms" mean
        the same policy rather than a second implementation of it.

        A run with no transcript carries nothing: `None` is the absence of the
        document, not an empty one, and inventing an empty transcript here would
        publish a transcript for a video that has none.

        The translation stays because `transcript.py` has not been migrated -
        it produces the mapping faster-whisper gave it, and this is where that
        mapping becomes a carrier. The frames need no equivalent any more.
        """
        if transcript is None:
            return None, []
        carrier = Transcript(
            language=str(transcript.get("language", "")),
            language_probability=float(transcript.get("language_probability", 0.0)),
            segments=tuple(transcript.get("segments", ())),
            redaction=self._redaction_policy,
        )
        return carrier, [dict(item) for item in carrier.warnings]

    def _produce_local_vision(self, frames: list[FrameArtifact]) -> dict[str, Any]:
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
            carried, carrier_warnings = self._carry_transcript(transcript)
            return {
                "transcript": carried,
                "warnings": [*transcript_warnings, *carrier_warnings],
            }

        transcript = self._run_stage(
            run,
            heartbeat,
            warnings,
            "transcript",
            ("transcription", "audio_extraction"),
            produce_transcript,
            self._recovered_transcript,
        )

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
                redaction=self._redaction_policy,
            )
            return {"frames": frames, "warnings": frame_warnings}

        frames = self._run_stage(
            run,
            heartbeat,
            warnings,
            "frames",
            ("frame_selection",),
            produce_frames,
            self._recovered_frames,
        )

        frames = self._run_stage(
            run,
            heartbeat,
            warnings,
            "ocr",
            ("ocr",),
            lambda: self._produce_ocr(frames),
            self._recovered_frames,
        )

        # No redaction stage: the policy ran inside `_produce_ocr`, where the
        # text entered its carrier (D-019). A stage here would be a stage that
        # runs after `write_stage` has already put the raw text on disk, which
        # is finding 4, and it would report a progress mechanism for work no
        # longer done in one place.

        if self.options.caption_frames:
            frames = self._run_stage(
                run,
                heartbeat,
                warnings,
                "local_vision",
                ("local_vision",),
                lambda: self._produce_local_vision(frames),
                self._recovered_frames,
            )
        else:
            self.progress.skip_cached("local_vision", detail={"reason": "disabled"})

        # R-41's fold, at the one point that sees every stage's warnings
        # together. A stage can only aggregate what it produced; what a reader
        # of a published **generation** gets is decided here, and the folded
        # record with its count *is* the warning carried, so ADR-0002's promise
        # that every warning reaches the **manifest** is kept by the count
        # rather than by the repetition. The **render**, the manifest and the
        # response are handed the same folded list, so no reader of a bundle
        # sees a different account of the run than another.
        warnings = aggregate_warnings(warnings)

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
            frames=response_frames(frames),
            warnings=warnings,
        )
        self.progress.complete("bundle_publish", detail={"generation": run.generation_name})
        progress_summary = self.progress.aggregator.terminal_summary(self.progress.states)
        manifest["progress"] = progress_summary
        snapshot = run.commit(manifest)

        # The publish renamed the **staging directory**, so the image path each
        # artifact recorded while staging is not where a reader will find it.
        published = [
            frame.relocated(str(snapshot.frames / Path(frame.path).name)) for frame in frames
        ]
        return run_response(
            snapshot,
            self.source,
            response_frames(published),
            transcript is not None,
            warnings,
            cached=False,
            progress=progress_summary,
            job_id=self.options.job_id,
        )


@dataclass(frozen=True, slots=True)
class _Recovered[StageValue]:
    """What a stage payload turned into, kept distinct from having failed to.

    One wrapper rather than a bare optional, because a stage's value can itself
    be `None`: a run whose video has no speech carries no **transcript**, and a
    reviver returning `None` for both that and "this payload is not the shape I
    read" would make `_run_stage` recompute a transcript that does not exist,
    every time, forever.
    """

    value: StageValue


def _revived[StageValue](
    payload: dict[str, Any],
    revive: Callable[[dict[str, Any]], _Recovered[StageValue] | None],
) -> _Recovered[StageValue] | None:
    """`payload` as the value its stage speaks, or `None` if it is not one.

    The common half of every reviver: the **warnings** every stage payload
    carries. A reviver is left owning only its own stage's field, and the
    envelope both of them share is checked once.

    Warnings are checked here rather than tolerated downstream because they are
    published: a resumed `"warnings": "truncated"` was iterated as a string, so
    a **generation**'s **manifest** recorded eleven warnings, one per character.
    """
    recorded = payload.get("warnings", ())
    if not isinstance(recorded, list | tuple):
        return None
    if not all(isinstance(item, Mapping) for item in recorded):
        return None
    return revive(payload)


def _stage_warnings(payload: dict[str, Any]) -> list[WarningRecord]:
    """The **warnings** a stage payload carries, as plain documents.

    Copied out rather than passed through: a carrier's warnings are
    `MappingProxyType`, which `json` cannot encode, and these end up in a
    **manifest**. Reached only for a payload `_revived` accepted, so every item
    is a mapping.
    """
    return [dict(item) for item in payload.get("warnings", ())]


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
                failure = exc if isinstance(exc, DistillError) else DistillError.from_unexpected(exc)
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


def youtube_playlist_urls(url: str, max_items: int) -> list[str]:
    from .source import _run_ytdlp

    # _run_ytdlp adds `--socket-timeout`, a `--` terminator before the URL, and
    # maps a missing/hung yt-dlp onto clean errors. `names_one_video=False` is
    # the one call in Distill whose subject really is a playlist: the default
    # would have this enumerate a single video.
    proc = _run_ytdlp(
        ["--flat-playlist", "--print", "webpage_url"], url, names_one_video=False
    )
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
) -> tuple[list[FrameArtifact], list[WarningRecord]]:
    """Interpret every frame, under the **redaction** policy the frames carry.

    The interpreter is told nothing about redaction. `--no-redact-secrets` is
    recorded on each **frame artifact** by `select_keyframes` and travels with
    it, so the model's words are redacted where they enter the carrier (R-19)
    rather than by a helper the vision pass had to remember to call.
    """
    interpreter = FrameInterpreter(
        config=options.local_vision_config(),
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
