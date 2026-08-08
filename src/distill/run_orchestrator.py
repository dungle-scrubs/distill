"""Run orchestration — the deep module that hides sequencing.

This module owns the **run** as a whole: how a source becomes a **generation**,
or how a cached one is served instead. It is the deep counterpart to the shallow
`pipeline.py` dispatch: its interface is `ProcessingRun(source, options, ...) →
dict`, and behind that seam it hides lock acquisition, wait accounting,
candidate **bundle key** settlement (including the single-rekey bound D-005 and
revalidation D-004), heartbeat, stage sequencing with resume, and publish.

Stages are adapters against a narrow ``StageContext`` — they receive carriers
and a progress reporter and return carriers — not direct imports in the caller.
Only this module decides the order and the retry semantics.

This is Candidate 01 of the architecture deepening review: pipeline.py was the
shallow god module whose interface was as wide as its implementation. By
concentrating sequencing here, the deletion test concentrates: deleting this
module would require reimplementing cache coalescing, re-keying, and stage
recovery in every caller. The interface is the test surface — cache-hit,
re-key, revalidation, and resume are unit tests against a fake ``BundleStore``.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .artifact import artifact_entry_name, emit_artifact, resolve_artifact_dir
from .artifacts import FrameArtifact, RedactionState, Transcript
from .bundle_store import (
    SINGLE_SOURCE_LOCK_WAIT_SEC,
    BundleRun,
    BundleSnapshot,
    BundleStore,
    confined_path,
)
from .errors import DistillError, WarningRecord, aggregate_warnings
from .local_vision import MAX_SOCKET_TIMEOUT_SEC
from .options import DistillOptions
from .progress import (
    TERMINAL_PROGRESS_STATUSES,
    OverallProgressAggregator,
    ProgressHeartbeat,
    ProgressReporter,
)
from .response import manifest_document, response_frames, run_response
from .source import ChainRevalidation, candidate_in_hand, revalidate_chain
from .vision_chain import REVALIDATE_AFTER_WAIT_SEC

LOGGER = logging.getLogger("distill.pipeline")

PIPELINE_EVENT_TYPE = "distill.pipeline"


def _pipeline_log(event: str, **detail: Any) -> None:
    """Emit one run-orchestration event: a decision ``execute`` took about a run."""
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


def _transcribe(video_path, work_dir, options, progress, duration_sec):
    """Indirect through pipeline so tests patching pipeline.transcribe_with_imports take effect."""
    from . import pipeline as _pipeline

    return _pipeline.transcribe_with_imports(
        video_path, work_dir, options, progress, duration_sec=duration_sec
    )


def _interpret_frames(frames, options, progress, transcript):
    from . import pipeline as _pipeline

    return _pipeline.interpret_frames_with_local_vision(
        frames, options, progress, transcript=transcript
    )


def _select_keyframes(*args, **kwargs):
    from . import pipeline as _pipeline

    return _pipeline.select_keyframes(*args, **kwargs)


def _ocr_frames(*args, **kwargs):
    from . import pipeline as _pipeline

    return _pipeline.ocr_frames(*args, **kwargs)


def _render_markdown(*args, **kwargs):
    from . import pipeline as _pipeline

    return _pipeline.render_markdown(*args, **kwargs)


DEFAULT_CONFIGURED_TIMEOUT_MS = 5_400_000
TIMEOUT_ENV = "DISTILL_EFFECTIVE_TIMEOUT_MS"
LONG_TIMEOUT_PROBE_ENV = "DISTILL_ENABLE_LONG_TIMEOUT_PROBE"
TIMEOUT_PROBE_LIMIT_MS = 1_000
TIMEOUT_PROBE_CEILING_MS = int(MAX_SOCKET_TIMEOUT_SEC * 1000)


def cache_hit_progress_summary(
    store: BundleStore, snapshot: BundleSnapshot
) -> tuple[BundleSnapshot, dict[str, Any]]:
    """The progress summary a cache hit reports, amending the manifest if it must."""
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


def _manifest_progress_summary(progress_summary: dict[str, Any]) -> dict[str, Any]:
    """Project live progress into a durable summary without machine-local paths."""
    portable = dict(progress_summary)
    mechanisms = progress_summary.get("mechanisms")
    if not isinstance(mechanisms, dict):
        return portable
    portable["mechanisms"] = {
        name: {
            **state,
            "detail": {
                key: value for key, value in state.get("detail", {}).items() if key != "path"
            },
        }
        if isinstance(state, dict) and isinstance(state.get("detail"), dict)
        else state
        for name, state in mechanisms.items()
    }
    return portable


def _abandon_reason(exc: BaseException) -> str:
    """Why a run gave up, in the terms the rest of Distill reports failures in."""
    if isinstance(exc, DistillError):
        return f"{exc.code}: {exc}"
    return f"{type(exc).__name__}: {exc}"


REKEY_BOUND_REASON = "one_rekey_per_run"


def revalidation_is_owed(waited_sec: float) -> bool:
    """Whether a run that waited this long owes its **endpoint chain** a second walk."""
    return waited_sec >= REVALIDATE_AFTER_WAIT_SEC


class StageRunner:
    """Runs one pipeline stage with resume support.

    A narrow adapter behind the revalidation seam: callers provide a ``producer``
    that creates the stage payload and a ``revive`` that turns a persisted
    payload (or a fresh one) into typed carriers. The runner owns the decision
    to reuse a ``stage result`` vs recompute, the warning propagation, and the
    heartbeat check. Every stage speaks carriers — no downstream subscripts a
    raw payload — so the recovery is total and a document that does not hold
    up simply triggers recomputation.
    """

    def __init__(
        self,
        run: BundleRun,
        heartbeat: ProgressHeartbeat,
        progress: ProgressReporter,
        warnings: list[WarningRecord],
        redaction: RedactionState,
    ) -> None:
        self._run = run
        self._heartbeat = heartbeat
        self._progress = progress
        self._warnings = warnings
        self._redaction = redaction

    def run_stage[StageValue](
        self,
        name: str,
        skipped_mechanisms: tuple[str, ...],
        producer: Callable[[], dict[str, Any]],
        revive: Callable[[dict[str, Any]], _Recovered[StageValue] | None],
    ) -> StageValue:
        recorded = self._run.read_stage(name)
        if recorded is not None:
            recovered = _revived(recorded, revive)
            if recovered is not None:
                for mechanism in skipped_mechanisms:
                    self._progress.skip_cached(mechanism, detail={"source": "partial_resume"})
                self._warnings.extend(_stage_warnings(recorded))
                return recovered.value
            self._run.discard_stage(name, "payload_shape_unusable")
        payload = producer()
        self._heartbeat.check()
        produced = _revived(payload, revive)
        if produced is None:
            raise DistillError(
                "E_BAD_STAGE_PAYLOAD",
                "pipeline",
                f"the {name} stage produced a payload of a shape it does not read",
                {"stage": name},
            )
        self._run.write_stage(name, payload, redaction=self._redaction)
        self._warnings.extend(_stage_warnings(payload))
        return produced.value


class KeySettlement:
    """Settles which **bundle key** the run will publish under after waiting.

    The wait is the store's measurement and the threshold is a statement about
    the vision memo (D-016 / D-004). Settlement happens under the lock and
    before any stage runs, so a re-key moves no staging. It asks at most twice
    and re-keys at most once (D-005).
    """

    def __init__(
        self,
        options: DistillOptions,
        source: Any,
        output_root: Path,
        lock_wait_sec: float,
        waited_sec_ref: list[float],
    ) -> None:
        self.options = options
        self.source = source
        self.output_root = output_root
        self.lock_wait_sec = lock_wait_sec
        self._waited = waited_sec_ref
        self.rekeyed_from: str | None = None

    def settle(
        self, store: BundleStore, run: BundleRun
    ) -> BundleRun | BundleSnapshot:
        held: BundleRun | None = run
        try:
            if not revalidation_is_owed(run.waited_sec):
                return run
            _pipeline_log(
                "chain_revalidated",
                bundle_key=run.bundle_key,
                waited_sec=run.waited_sec,
                revalidate_after_sec=REVALIDATE_AFTER_WAIT_SEC,
            )
            diverged = self._divergence(run)
            if diverged is None:
                return run
            remaining = max(0.0, self.lock_wait_sec - run.waited_sec)
            self._leave_key_for(run, diverged)
            held = None
            begun = self._begin(store, diverged.bundle_key, remaining)
            if isinstance(begun, BundleSnapshot):
                return begun
            held = begun
            declined = self._divergence(begun)
            if declined is not None:
                self._log_divergence(
                    "chain_divergence_declined", begun, declined, reason=REKEY_BOUND_REASON
                )
            return begun
        except BaseException as exc:
            if held is not None:
                held.abandon(_abandon_reason(exc), during=exc)
            raise

    def _begin(
        self, store: BundleStore, bundle_key: str, wait_sec: float
    ) -> BundleRun | BundleSnapshot:
        began = store.begin(
            bundle_key,
            wait_sec=wait_sec,
            resume=self.options.resume_partial,
            reuse_active=not self.options.force_reprocess,
        )
        self._waited[0] += began.waited_sec
        return began

    def _divergence(self, run: BundleRun) -> ChainRevalidation | None:
        # Test seam: ``tests/test_revalidation.py`` monkeypatches
        # ``pipeline.revalidate_chain``; honor it via lazy lookup so a
        # patched ``refuse`` that raises is not swallowed by the fallback.
        _revalidate = None
        try:
            from . import pipeline as _pipeline  # noqa: PLC0415

            _revalidate = getattr(_pipeline, "revalidate_chain", None)
        except ImportError:
            _revalidate = None
        # Use pipeline's version only when it is a different object than the
        # source's (i.e. monkeypatched); otherwise use the direct import.
        if callable(_revalidate) and _revalidate is not revalidate_chain:
            revalidated = _revalidate(
                self.options,
                self.source.source_fingerprint,
                self.source.source_type,
                self.output_root,
            )
        else:
            revalidated = revalidate_chain(
                self.options,
                self.source.source_fingerprint,
                self.source.source_type,
                self.output_root,
            )
        if revalidated.bundle_key == run.bundle_key:
            return None
        return revalidated

    def _leave_key_for(self, run: BundleRun, revalidated: ChainRevalidation) -> None:
        self._log_divergence("chain_diverged", run, revalidated)
        run.abandon(f"chain_rekeyed: {revalidated.bundle_key}")
        self.rekeyed_from = run.bundle_key
        self.options = revalidated.resolution.options
        self.source = replace(
            self.source,
            source_hash=revalidated.bundle_key,
            resolved_options=revalidated.resolution.options,
        )

    def _log_divergence(
        self,
        event: str,
        run: BundleRun,
        revalidated: ChainRevalidation,
        **detail: Any,
    ) -> None:
        held = candidate_in_hand(self.options, self.source.source_type)
        _pipeline_log(
            event,
            bundle_key=run.bundle_key,
            entry=None if held is None else held.entry,
            vision_mode=self.options.vision_mode,
            new_bundle_key=revalidated.bundle_key,
            new_entry=revalidated.resolution.entry,
            new_vision_mode=revalidated.resolution.vision_mode,
            **detail,
        )


@dataclass
class ProcessingRun:
    """One run's attempt to produce (or serve) its **generation**."""

    source: Any
    options: DistillOptions
    output_root: Path
    progress: ProgressReporter
    tool: str
    lock_wait_sec: float = SINGLE_SOURCE_LOCK_WAIT_SEC

    waited_sec: float = field(default=0.0, init=False)
    rekeyed_from: str | None = field(default=None, init=False)

    def execute(self) -> dict[str, Any]:
        store = BundleStore.open(self.output_root)
        began = self._begin(store, self.source.source_hash, self.lock_wait_sec)
        if isinstance(began, BundleRun):
            began = self._settled_after_wait(store, began)
        if isinstance(began, BundleSnapshot):
            return self._cached_response(store, began)
        heartbeat = ProgressHeartbeat(self.progress.counter).start()
        try:
            with began as run:
                try:
                    return self._produce_generation(run, heartbeat)
                except BaseException as exc:
                    run.abandon(_abandon_reason(exc), during=exc)
                    raise
        finally:
            heartbeat.stop()

    def _begin(
        self, store: BundleStore, bundle_key: str, wait_sec: float
    ) -> BundleRun | BundleSnapshot:
        began = store.begin(
            bundle_key,
            wait_sec=wait_sec,
            resume=self.options.resume_partial,
            reuse_active=not self.options.force_reprocess,
        )
        self.waited_sec += began.waited_sec
        return began

    def _settled_after_wait(
        self, store: BundleStore, run: BundleRun
    ) -> BundleRun | BundleSnapshot:
        waited_ref = [self.waited_sec]
        settlement = KeySettlement(
            self.options, self.source, self.output_root, self.lock_wait_sec, waited_ref
        )
        result = settlement.settle(store, run)
        # propagate mutations back
        self.options = settlement.options
        self.source = settlement.source
        self.waited_sec = waited_ref[0]
        if settlement.rekeyed_from is not None:
            self.rekeyed_from = settlement.rekeyed_from
        return result

    def _divergence(self, run: BundleRun) -> ChainRevalidation | None:
        # Test seam: ``tests/test_revalidation.py`` monkeypatches
        # ``pipeline.revalidate_chain``; honor it via lazy lookup so a
        # patched ``refuse`` that raises is not swallowed by the fallback.
        _revalidate = None
        try:
            from . import pipeline as _pipeline  # noqa: PLC0415

            _revalidate = getattr(_pipeline, "revalidate_chain", None)
        except ImportError:
            _revalidate = None
        # Use pipeline's version only when it is a different object than the
        # source's (i.e. monkeypatched); otherwise use the direct import.
        if callable(_revalidate) and _revalidate is not revalidate_chain:
            revalidated = _revalidate(
                self.options,
                self.source.source_fingerprint,
                self.source.source_type,
                self.output_root,
            )
        else:
            revalidated = revalidate_chain(
                self.options,
                self.source.source_fingerprint,
                self.source.source_type,
                self.output_root,
            )
        if revalidated.bundle_key == run.bundle_key:
            return None
        return revalidated

    def _leave_key_for(self, run: BundleRun, revalidated: ChainRevalidation) -> None:
        self._log_divergence("chain_diverged", run, revalidated)
        run.abandon(f"chain_rekeyed: {revalidated.bundle_key}")
        self.rekeyed_from = run.bundle_key
        self.options = revalidated.resolution.options
        self.source = replace(
            self.source,
            source_hash=revalidated.bundle_key,
            resolved_options=revalidated.resolution.options,
        )

    def _log_divergence(
        self,
        event: str,
        run: BundleRun,
        revalidated: ChainRevalidation,
        **detail: Any,
    ) -> None:
        held = candidate_in_hand(self.options, self.source.source_type)
        _pipeline_log(
            event,
            bundle_key=run.bundle_key,
            entry=None if held is None else held.entry,
            vision_mode=self.options.vision_mode,
            new_bundle_key=revalidated.bundle_key,
            new_entry=revalidated.resolution.entry,
            new_vision_mode=revalidated.resolution.vision_mode,
            **detail,
        )

    def _cached_response(self, store: BundleStore, snapshot: BundleSnapshot) -> dict[str, Any]:
        snapshot, progress_summary = cache_hit_progress_summary(store, snapshot)
        manifest = snapshot.manifest
        frames = [
            {
                **frame,
                "path": str(
                    confined_path(
                        snapshot.generation / Path(frame["relative_path"]),
                        snapshot.generation,
                    )
                ),
            }
            for frame in manifest.get("frames", [])
        ]
        return run_response(
            snapshot,
            self.source,
            frames,
            bool(manifest.get("transcript_present")),
            list(manifest.get("warnings", [])),
            cached=True,
            artifact_path=self._emit_artifact(snapshot),
            progress=progress_summary,
            job_id=self.options.job_id,
            waited_sec=self.waited_sec,
            rekeyed_from=self.rekeyed_from,
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
        runner = StageRunner(run, heartbeat, self.progress, warnings, self._redaction_policy)
        return runner.run_stage(name, skipped_mechanisms, producer, revive)

    def _recovered_frames(self, payload: dict[str, Any]) -> _Recovered[list[FrameArtifact]] | None:
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
                frames.append(FrameArtifact.from_document(item, redaction=self._redaction_policy))
            except Exception:
                return None
        return _Recovered(frames)

    def _recovered_transcript(
        self, payload: dict[str, Any]
    ) -> _Recovered[Transcript | None] | None:
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
        ocr = self.options.ocr_config()
        read, ocr_warnings = _ocr_frames(
            frames,
            ocr.language,
            ocr.enabled,
            self.progress,
            ocr.preprocess,
        )
        return {"frames": read, "warnings": ocr_warnings}

    @property
    def _redaction_policy(self) -> RedactionState:
        return (
            RedactionState.NOT_APPLIED if self.options.redact_secrets else RedactionState.DISABLED
        )

    def _carry_transcript(self, transcript: Any) -> tuple[Transcript | None, list[dict]]:
        if transcript is None:
            return None, []
        carrier = Transcript(
            language=str(transcript.get("language", "")),
            language_probability=float(transcript.get("language_probability", 0.0)),
            segments=tuple(transcript.get("segments", ())),
            redaction=self._redaction_policy,
        )
        return carrier, [dict(item) for item in carrier.warnings]

    def _emit_artifact(self, snapshot: BundleSnapshot) -> str | None:
        try:
            artifact_dir = resolve_artifact_dir(
                explicit=self.options.artifact_dir,
                env=dict(os.environ),
                cwd=Path.cwd(),
            )
            entry = artifact_entry_name(
                self.source.youtube_video_id,
                self.source.source_fingerprint,
                Path(self.source.resolved_path).stem,
            )
            return str(
                emit_artifact(
                    snapshot.self_contained_markdown,
                    artifact_dir,
                    entry,
                    output_root=self.output_root,
                )
            )
        except (OSError, ValueError, DistillError) as exc:
            LOGGER.warning("artifact not written: %s", exc)
            return None

    def _produce_local_vision(
        self, frames: list[FrameArtifact], transcript: Transcript | None
    ) -> dict[str, Any]:
        vision_frames, vision_warnings = _interpret_frames(
            frames,
            self.options,
            self.progress,
            transcript=transcript,
        )
        return {"frames": vision_frames, "warnings": vision_warnings}

    def _produce_generation(
        self,
        run: BundleRun,
        heartbeat: ProgressHeartbeat,
    ) -> dict[str, Any]:
        warnings = list(self.source.warnings)

        def produce_transcript() -> dict[str, Any]:
            transcript, transcript_warnings = _transcribe(
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
            frames, frame_warnings = _select_keyframes(
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

        if self.options.caption_frames:
            frames = self._run_stage(
                run,
                heartbeat,
                warnings,
                "local_vision",
                ("local_vision",),
                lambda: self._produce_local_vision(frames, transcript),
                self._recovered_frames,
            )
        else:
            self.progress.skip_cached("local_vision", detail={"reason": "disabled"})

        warnings = aggregate_warnings(warnings)
        self.progress.update("rendering", status="running")
        markdown = _render_markdown(
            str(self.source.resolved_path),
            self.source.duration_sec,
            transcript,
            frames,
            warnings,
            getattr(self.source, "related_links", None),
        )
        if self.source.provenance is None:
            raise AssertionError("current self-contained render requires provenance")
        self_contained_markdown = _render_markdown(
            str(self.source.resolved_path),
            self.source.duration_sec,
            transcript,
            frames,
            warnings,
            getattr(self.source, "related_links", None),
            provenance=self.source.provenance,
            include_frame_links=False,
        )
        self.progress.complete("rendering")
        self.progress.update("bundle_publish", status="running")
        run.write_render(markdown)
        run.write_self_contained_render(self_contained_markdown)
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
        manifest["progress"] = _manifest_progress_summary(progress_summary)
        snapshot = run.commit(manifest)
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
            artifact_path=self._emit_artifact(snapshot),
            progress=progress_summary,
            job_id=self.options.job_id,
            waited_sec=self.waited_sec,
            rekeyed_from=self.rekeyed_from,
        )


@dataclass(frozen=True, slots=True)
class _Recovered[StageValue]:
    value: StageValue


def _revived[StageValue](
    payload: dict[str, Any],
    revive: Callable[[dict[str, Any]], _Recovered[StageValue] | None],
) -> _Recovered[StageValue] | None:
    recorded = payload.get("warnings", ())
    if not isinstance(recorded, list | tuple):
        return None
    if not all(isinstance(item, Mapping) for item in recorded):
        return None
    return revive(payload)


def _stage_warnings(payload: dict[str, Any]) -> list[WarningRecord]:
    return [dict(item) for item in payload.get("warnings", ())]
