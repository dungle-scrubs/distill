"""Tests for redaction at the point **extracted text** enters a carrier.

The invariant under test is R-19 and R-21 (D-019): the **redaction** policy runs
where extracted text enters a carrier, not in a pipeline stage that runs after
some of that text is already on disk. Finding 4 is what the old arrangement did
to a **stage result** - the OCR stage recorded raw text and a later stage
redacted a copy - and finding 15 is what it did to the **transcript** and to
**related links**, which no stage covered at all.

These tests are written against the sinks a user can actually reach: a **stage
result** in a **staging directory** left by an interrupted run, `transcript.json`
in a published **generation**, and the **manifest** and **render** that carry
related-link labels and destinations.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from test_local_integration import make_short_screencast

from distill import pipeline as distill_session
from distill.artifacts import (
    FrameArtifact,
    Interpretation,
    Provenance,
    RedactionPolicyNotApplied,
    RedactionState,
    Transcript,
    serialize,
)
from distill.bundle_store import BundleRun, BundleStore
from distill.links import RelatedLink, extract_relevant_links
from distill.options import DistillOptions
from distill.progress import DEFAULT_MECHANISM_WEIGHTS
from distill.render import render_markdown
from distill.response import manifest_document, response_related_links
from distill.source import (
    AcquiredSource,
    AcquisitionLease,
    SourceInfo,
    SourceRequest,
    YouTubeMetadata,
    YouTubeSourceProvider,
    _manifest_related_links,
    source_hash,
    youtube_source_info,
)

# One shape of secret used throughout, so that "the secret is gone" means the
# same thing at every sink. `redact_secrets.SECRET_RULES` matches it on the
# OpenAI-style rule.
SECRET = "sk-live-0123456789abcdefghij"

BUNDLE_KEY = "b" * 64


def transcript_saying(text: str) -> Any:
    """A transcriber double whose speech carries `text`.

    The signature is `transcribe_with_imports`'s, because that is the seam the
    pipeline calls and the point finding 15's secret enters the run.
    """

    def fake_transcribe(
        _video_path: Path,
        _work_dir: Path,
        _options: DistillOptions,
        progress: Any,
        duration_sec: float,
    ) -> tuple[dict[str, Any], list[dict[str, str]]]:
        progress.increment()
        return (
            {
                "language": "en",
                "language_probability": 0.99,
                "segments": [
                    {
                        "start": 0.0,
                        "end": 1.0,
                        "text": text,
                        "words": [{"word": text, "start": 0.0, "end": 0.9}],
                    }
                ],
            },
            [],
        )

    return fake_transcribe


def ocr_reading(text: str) -> Any:
    """An image-text reader double that recovers `text` from every **keyframe**."""

    def fake_ocr_frames(
        frames: list[FrameArtifact],
        _language: str,
        _enabled: bool,
        _progress: Any = None,
        _preprocess: bool = True,
    ) -> tuple[list[FrameArtifact], list[dict[str, str]]]:
        # Through the carrier, exactly as the real `ocr_frames` does: a double
        # that set the text some other way would be testing an arrangement in
        # which the redaction policy never ran (R-19).
        return [frame.with_extracted_text(text)[0] for frame in frames], []

    return fake_ocr_frames


def vision_probe(_config: object) -> Any:
    """A vision backend that is available, without one being installed (ADR-0001)."""
    from distill.local_vision import LocalVisionProbe

    return LocalVisionProbe(
        available=True,
        backend="rapid-mlx",
        model="qwen3-vl:8b",
        base_url="http://127.0.0.1:8000/v1",
        code="local_vision_available",
        message="available",
        detail={},
    )


def vision_reading(secret: str) -> Any:
    """A vision model that reports `secret` in every field it has.

    Every field, because the model echoes the screen back everywhere it can:
    a summary that quotes a slide is as durable as the verbatim text is.
    """

    def fake_interpret(
        _config: object,
        _image_path: Path,
        _prompt: str,
        *,
        prompt_profile: str,
    ) -> tuple[Any, None]:
        return (
            Interpretation(
                visual_summary=f"a terminal showing {secret}",
                detected_elements=(f"a shell prompt with {secret}",),
                interpretation=f"the presenter pasted {secret}",
                uncertainty=f"unsure whether {secret} is real",
                backend="rapid-mlx",
                model="qwen3-vl:8b",
                prompt_profile=prompt_profile,
                verbatim_text=f"export API_KEY={secret}",
                text_confidence="high",
            ),
            None,
        )

    return fake_interpret


def files_holding(root: Path, secret: str) -> list[str]:
    """Every file under `root` whose bytes contain `secret`.

    Bytes rather than text, and every file rather than the ones a reader is
    served: the question finding 4 asks is whether the secret is anywhere on
    disk, which includes a **stage result** nobody publishes.
    """
    return [
        str(path.relative_to(root))
        for path in sorted(root.rglob("*"))
        if path.is_file() and secret in path.read_bytes().decode("utf-8", errors="replace")
    ]


def run_interrupted_after_the_stages(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    ocr_text: str,
) -> Path:
    """Run a pipeline that dies before publishing, and return its output root.

    A **stage result** only exists between the stage that wrote it and the
    **publish** that strips it, so an interrupted run is the only way to look at
    one from outside the run. It is also the case finding 4 describes: a crashed
    run leaves a **staging directory** behind for the next run to **resume**.
    """
    video = tmp_path / "fixture.mp4"
    make_short_screencast(video)
    output_dir = tmp_path / "cache"
    monkeypatch.setattr(
        distill_session,
        "transcribe_with_imports",
        # The transcript stage records a stage result too, and finding 15 is
        # that nothing covered it: a run interrupted after it is the only place
        # that file can be looked at, so it says the secret out loud here.
        transcript_saying(f"the key is {SECRET}"),
    )
    monkeypatch.setattr(distill_session, "ocr_frames", ocr_reading(ocr_text))

    def stop_before_publish(*_args: object, **_kwargs: object) -> str:
        raise RuntimeError("stopped once every stage result was recorded")

    monkeypatch.setattr(distill_session, "render_markdown", stop_before_publish)

    with pytest.raises(RuntimeError):
        distill_session.process_local_video(
            {
                "path": str(video),
                "output_dir": str(output_dir),
                "ocr": True,
                "redact_secrets": True,
                "caption_frames": False,
                "max_keyframes": 3,
                "max_static_window_sec": 1,
            }
        )
    return output_dir


def published_generation(response: dict[str, Any]) -> Path:
    return Path(str(response["markdown_path"])).parent


# 1. FAILS FIRST: a secret in OCR text never appears in any **stage result** on
#    disk - finding 4's root cause


def test_a_secret_in_ocr_text_reaches_no_stage_result_on_disk(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Finding 4's root cause: the OCR stage recorded raw text and a later stage redacted it.

    The stage result was already on disk by then, so redaction after the fact
    only ever cleaned up a copy.
    """
    output_dir = run_interrupted_after_the_stages(
        monkeypatch,
        tmp_path,
        ocr_text=f"export API_KEY={SECRET}",
    )

    recorded = {path.name for path in output_dir.rglob("_*.json")}
    assert "_ocr.json" in recorded, "the stage result under test was never written"
    assert "_transcript.json" in recorded
    assert files_holding(output_dir, SECRET) == []


# 2. Redaction applied at carrier construction


def test_a_frame_artifact_is_redacted_the_moment_it_is_constructed() -> None:
    """The carrier never holds unredacted extracted text, not even briefly."""
    frame = FrameArtifact(
        index=1,
        timestamp_sec=0.5,
        path="/tmp/bundle/frames/frame_001.png",
        relative_path="frames/frame_001.png",
        extracted_text=f"export API_KEY={SECRET}",
        interpretation={"verbatim_text": f"the slide showed {SECRET}"},
    )

    assert frame.redaction is RedactionState.APPLIED
    assert SECRET not in frame.extracted_text
    assert frame.interpretation is not None
    assert SECRET not in str(frame.interpretation["verbatim_text"])


def test_construction_applies_the_policy_rather_than_recording_a_claim() -> None:
    """A producer cannot declare `APPLIED` over text the policy never saw.

    M4.1 could only check that a claim was made. Construction is what makes the
    claim true: the state is now set by the constructor that ran the policy.
    """
    claimed = FrameArtifact(
        index=1,
        timestamp_sec=0.5,
        path="/tmp/bundle/frames/frame_001.png",
        relative_path="frames/frame_001.png",
        extracted_text=SECRET,
        redaction=RedactionState.APPLIED,
    )

    assert SECRET not in claimed.extracted_text


def test_an_explicitly_disabled_policy_is_still_the_opt_out() -> None:
    """R-20: `--no-redact-secrets` means the policy does not run, and says so."""
    frame = FrameArtifact(
        index=1,
        timestamp_sec=0.5,
        path="/tmp/bundle/frames/frame_001.png",
        relative_path="frames/frame_001.png",
        extracted_text=SECRET,
        redaction=RedactionState.DISABLED,
    )

    assert frame.redaction is RedactionState.DISABLED
    assert frame.extracted_text == SECRET


# 3. FAILS FIRST: a secret spoken in the transcript is redacted in
#    `transcript.json` (finding 15)


def test_a_secret_spoken_in_the_transcript_is_redacted_in_transcript_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Finding 15: no pipeline stage ever passed the transcript through the policy."""
    video = tmp_path / "fixture.mp4"
    make_short_screencast(video)
    monkeypatch.setattr(
        distill_session,
        "transcribe_with_imports",
        transcript_saying(f"the key is {SECRET} do not share it"),
    )

    response = distill_session.process_local_video(
        {
            "path": str(video),
            "output_dir": str(tmp_path / "cache"),
            "ocr": False,
            "redact_secrets": True,
            "caption_frames": False,
            "max_keyframes": 3,
            "max_static_window_sec": 1,
        }
    )

    transcript = json.loads((published_generation(response) / "transcript.json").read_text())
    spoken = transcript["segments"][0]
    assert SECRET not in spoken["text"]
    assert SECRET not in spoken["words"][0]["word"]


# 4. Transcript redacted on the same terms as keyframe text


def test_a_transcript_is_redacted_on_the_same_terms_as_keyframe_text() -> None:
    """R-21: same policy, every string in the region, same opt-out."""
    spoken: tuple[dict[str, Any], ...] = (
        {"start": 0.0, "end": 1.0, "text": SECRET, "words": [{"word": SECRET}]},
    )

    applied = Transcript(language="en", segments=spoken)
    disabled = Transcript(language="en", segments=spoken, redaction=RedactionState.DISABLED)

    assert applied.redaction is RedactionState.APPLIED
    assert SECRET not in serialize(applied)["segments"][0]["text"]
    assert SECRET not in serialize(applied)["segments"][0]["words"][0]["word"]
    assert serialize(disabled)["segments"][0]["text"] == SECRET


def test_the_transcript_opt_out_survives_to_disk(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The transcript honours `--no-redact-secrets` exactly as keyframe text does."""
    video = tmp_path / "fixture.mp4"
    make_short_screencast(video)
    monkeypatch.setattr(
        distill_session,
        "transcribe_with_imports",
        transcript_saying(f"the key is {SECRET}"),
    )

    response = distill_session.process_local_video(
        {
            "path": str(video),
            "output_dir": str(tmp_path / "cache"),
            "ocr": False,
            "redact_secrets": False,
            "caption_frames": False,
            "max_keyframes": 3,
            "max_static_window_sec": 1,
        }
    )

    transcript = json.loads((published_generation(response) / "transcript.json").read_text())
    spoken = transcript["segments"][0]
    assert SECRET in spoken["text"]
    # The per-word text as well: a policy that reached one and not the other
    # would look opted out at the level a render prints and not at the level
    # the file records.
    assert SECRET in spoken["words"][0]["word"]
    assert transcript["redaction"] == "disabled"


# 5. FAILS FIRST: a secret in a related-link label is redacted
# 6. A secret in a related-link destination is redacted


def test_a_secret_in_a_related_link_label_is_redacted() -> None:
    """R-21: a label is **extracted text** - whoever wrote the description chose it."""
    links = extract_relevant_links(
        f"Setup guide, run export API_KEY={SECRET} first: https://github.com/example/repo",
        source="youtube_description",
    )

    assert len(links) == 1
    assert SECRET not in links[0].label
    assert links[0].redaction is RedactionState.APPLIED


def test_a_secret_in_a_related_link_destination_is_redacted() -> None:
    """The destination is extracted text too, and it is the half easiest to forget."""
    links = extract_relevant_links(
        f"Docs: https://github.com/example/repo?api_key={SECRET}",
        source="youtube_description",
    )

    assert len(links) == 1
    assert SECRET not in links[0].url


def test_a_secret_straddling_the_label_cap_is_redacted_before_it_is_cut() -> None:
    """A cap applied first decides what the policy is allowed to see.

    A label is capped at 160 characters, and a secret crossing that boundary
    used to be sliced into a prefix short enough to match no pattern - after
    which the prefix was durable in full. Redaction runs first for the same
    reason it does before the 256 KiB cap on keyframe text.
    """
    # 140 characters, so the raw label is 169 and the redacted one is 151: a
    # cut applied first removes the last nine characters of the key, leaving a
    # prefix too short for the pattern to match and durable in full.
    filler = "why this matters " * 8 + "here"
    links = extract_relevant_links(
        f"{filler} {SECRET} https://github.com/example/repo",
        source="youtube_description",
    )

    label = links[0].label
    assert len(label) <= 160
    assert label.endswith("[REDACTED:api-key]")
    assert "sk-" not in label


def test_related_links_honour_the_opt_out() -> None:
    """Same terms as keyframe text means the same flag reaches them."""
    links = extract_relevant_links(
        f"Docs: https://github.com/example/repo?api_key={SECRET}",
        source="youtube_description",
        redact=False,
    )

    assert SECRET in links[0].url
    assert links[0].redaction is RedactionState.DISABLED


# 6b. The sinks a related link reaches enforce the policy, as they do for the
#     other carriers


def source_carrying(links: list[RelatedLink]) -> SourceInfo:
    """A **source** whose metadata produced `links` and nothing else of interest."""
    return SourceInfo(
        source_type="youtube",
        resolved_path=Path("/tmp/video.mp4"),
        duration_sec=1.0,
        source_fingerprint="fingerprint",
        source_hash=BUNDLE_KEY,
        warnings=[],
        provenance=Provenance(
            canonical_url="https://www.youtube.com/watch?v=abcdefghijk",
            duration_sec=1.0,
            processed_at="2026-07-29T14:20:00Z",
        ),
        related_links=links,
    )


def test_a_related_link_whose_policy_never_ran_is_refused_by_the_render_and_the_manifest() -> None:
    """The one carrier family whose sinks enforced nothing (R-20, R-21).

    `_require_redaction_policy` covered the frames and the **transcript**, and
    related links arrived at both sinks as plain documents that nothing looked
    at - so a label and a destination could reach a **render** and a
    **manifest** without the check that refuses text no policy ran over. That
    is the hole `BundleRun.write_transcript` was written to close, stated for
    the carrier it had not reached: a caller holding a document has bypassed
    the check already, so it cannot be allowed to hand one in.

    Live callers were correct, so this is the layer being made to hold rather
    than a leak being stopped. The state below is the documented bypass -
    construction cannot produce it - which is what makes the check reachable
    at all.

    The render is given a **transcript** it could render, so the refusal is the
    only reason it can fail. Handed nothing to render it raises `E_NO_CONTENT`
    instead, which passes a bare `DistillError` assertion whether or not the
    link was ever checked.
    """
    link = extract_relevant_links(
        f"Docs: https://github.com/example/repo?api_key={SECRET}",
        source="youtube_description",
    )[0]
    object.__setattr__(link, "redaction", RedactionState.NOT_APPLIED)
    spoken = Transcript(
        language="en",
        segments=({"start": 0.0, "end": 1.0, "text": "they said something"},),
    )
    assert render_markdown("demo.mp4", 1.0, spoken, [], [], [])

    with pytest.raises(RedactionPolicyNotApplied):
        render_markdown("demo.mp4", 1.0, spoken, [], [], [link])
    with pytest.raises(RedactionPolicyNotApplied):
        manifest_document(
            source_carrying([link]),
            DistillOptions(),
            transcript_present=False,
            frames=[],
            warnings=[],
        )


def test_the_documents_a_reader_gets_carry_the_link_and_not_the_carriers_bookkeeping() -> None:
    """A **related link** is bundle content; its policy state and warnings are not.

    `redaction` and `warnings` are how a carrier records what was done to it,
    and they were being written into the **manifest** and handed to the caller
    beside the four fields that describe the link. They are stripped at the
    boundary, on the same terms as a **frame artifact**'s: `response_frames`
    has never carried them either, and the run's policy is already recorded
    once, in the manifest's `options`.

    Which leaves the **warnings** somewhere - they are a **degradation** and
    dropping them would be losing one. They join the source's warnings, so a
    truncated label is counted in `warning_count` like every other warning
    rather than buried in the entry it came from.
    """
    links = extract_relevant_links(
        "Docs: https://github.com/example/repo",
        source="youtube_description",
    )

    documents = response_related_links(links)

    assert documents == [
        {
            "url": "https://github.com/example/repo",
            "label": "Docs",
            "source": "youtube_description",
            "reason": "code_or_reference_domain",
        }
    ]


def test_a_cache_hit_reads_its_related_links_back_as_carriers(tmp_path: Path) -> None:
    """The one route that produces a **related link** without reading a description.

    A cache hit describes a **source** it never fetched, so its links come off
    the **manifest** - and they have to come back as carriers, because the run
    reusing them hands them to the same sinks a fresh run does. Reading them
    back as the mappings the manifest holds is exactly the state finding 5
    closed everywhere else: a document the sinks can no longer refuse.

    The policy is the *reading* run's, on `FrameArtifact.from_document`'s terms,
    which is why `from_document` takes it rather than believing the document.
    Asserted here by the round trip landing on the same four fields the fresh
    run published, with the secret still gone from both halves.
    """
    root = tmp_path / "output"
    root.mkdir()
    options = DistillOptions()
    video_id = "abcdefghijk"
    key = source_hash(hashlib.sha256(video_id.encode()).hexdigest(), options.opts_hash("youtube"))
    fresh = SourceInfo(
        source_type="youtube",
        resolved_path=tmp_path / "video.mp4",
        duration_sec=12.0,
        source_fingerprint="fingerprint",
        source_hash=key,
        warnings=[],
        provenance=Provenance(
            canonical_url=f"https://www.youtube.com/watch?v={video_id}",
            duration_sec=12.0,
            processed_at="2026-07-29T14:20:00Z",
        ),
        related_links=extract_relevant_links(
            f"Skill repo: https://github.com/example/repo?api_key={SECRET}",
            source="youtube_description",
        ),
    )
    run = BundleStore.open(root).begin(key)
    assert isinstance(run, BundleRun)
    run.write_render("# Video\n")
    run.write_self_contained_render("# Video\n")
    run.commit(manifest_document(fresh, options, transcript_present=False, frames=[], warnings=[]))

    served = YouTubeSourceProvider().cached(
        SourceRequest(value=f"https://youtu.be/{video_id}", options=options, output_root=root),
        metadata=YouTubeMetadata(video_id=video_id, description="", warnings=[]),
    )

    assert served is not None
    assert served.related_links is not None
    assert all(isinstance(link, RelatedLink) for link in served.related_links)
    assert all(link.redaction is RedactionState.APPLIED for link in served.related_links)
    assert response_related_links(served.related_links) == response_related_links(
        fresh.related_links
    )
    assert SECRET not in json.dumps(response_related_links(served.related_links))


def test_a_manifest_that_describes_no_links_usably_costs_the_links_and_not_the_cache_hit() -> None:
    """Tolerance on the way in is a claim, so it is exercised rather than stated (D-022).

    A **manifest** is something an older Distill may have written, so what it
    records may not be the shape this one reads - and a cache hit is worth more
    than the links it cannot rebuild. Called directly because no run this
    version can perform publishes such a manifest: `response_related_links`
    serializes carriers, so the shapes below only ever arrive off disk.
    """
    options = DistillOptions()

    assert _manifest_related_links({}, options) is None
    assert _manifest_related_links({"related_links": "https://example.com"}, options) is None
    assert _manifest_related_links(
        {"related_links": ["https://example.com", {"url": "https://github.com/example/repo"}]},
        options,
    ) == [
        RelatedLink(
            url="https://github.com/example/repo",
            label="",
            source="",
            reason="",
            redaction=RedactionState.APPLIED,
        )
    ]


def test_a_warning_a_link_raised_at_construction_reaches_the_runs_warnings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The **redaction** policy's own warning about a link is not lost at the boundary.

    A confusable-obfuscated key in a description is redacted *and* reported -
    the report is what tells a user their bundle met something shaped like a
    secret. Stripping the carrier's bookkeeping from the manifest entry is only
    correct if the warning has somewhere else to be, and the source's warnings
    are where the rest of acquisition's already are.
    """
    obfuscated = "ｓk-live-0123456789abcdefghij"  # fullwidth s
    source = resolve_youtube_source(
        monkeypatch,
        tmp_path,
        DistillOptions(),
        description=f"Skill repo: https://github.com/example/repo?api_key={obfuscated}",
    )

    assert [warning["code"] for warning in source.warnings] == [
        "possible_confusable_secret",
        "possible_confusable_secret",
    ]
    assert source.provenance is not None
    assert "[REDACTED:assigned-secret]" in (source.provenance.description or "")
    assert source.related_links is not None
    assert "[REDACTED:assigned-secret]" in source.related_links[0].url


def resolve_youtube_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    options: DistillOptions,
    description: str | None = None,
) -> Any:
    """Resolve a YouTube **source** with everything outside the process faked.

    Written out rather than asserted at `extract_relevant_links` directly
    because the thing under test is the wiring: the option has to travel from
    the caller's `DistillOptions` to the policy, and a test that calls the
    extractor with an explicit argument cannot see that journey at all.
    """
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video")
    lease = AcquisitionLease.take("abc123", tmp_path / "abc123.lock")
    assert lease is not None

    class FakeDownloader:
        def acquire(self, url: str, lock_key: str, progress: Any = None) -> AcquiredSource:
            _ = (url, lock_key, progress)
            return AcquiredSource(path=video, lease=lease)

    said = (
        description
        if description is not None
        else f"Skill repo: https://github.com/example/repo?api_key={SECRET}"
    )
    monkeypatch.setattr("distill.source.check_disk_floor", lambda _path: None)
    monkeypatch.setattr(
        "distill.source.youtube_metadata",
        lambda _url: YouTubeMetadata(video_id="abc123", description=said, warnings=[]),
    )
    monkeypatch.setattr("distill.source.probe_duration", lambda _path: (12.0, []))
    try:
        return youtube_source_info(
            "https://www.youtube.com/watch?v=abc123",
            options,
            tmp_path,
            FakeDownloader(),
        )
    finally:
        lease.release()


def test_the_link_policy_is_the_users_option_and_not_a_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """R-20 reaches related links through the resolver, not only through the extractor.

    The opt-out is one boolean travelling from `DistillOptions` to the carrier.
    Hard-wiring the policy at the extraction site would satisfy every test that
    calls the extractor directly and would silently ignore what the user asked
    for, so it is asserted from the surface the user's option enters.
    """
    redacted = resolve_youtube_source(monkeypatch, tmp_path, DistillOptions())
    opted_out = resolve_youtube_source(monkeypatch, tmp_path, DistillOptions(redact_secrets=False))

    # Asserted on the documents the sinks produce rather than on the carriers,
    # because that is the form the user's option has to survive all the way to.
    assert SECRET not in json.dumps(response_related_links(redacted.related_links))
    assert SECRET in json.dumps(response_related_links(opted_out.related_links))


# 7. The standalone redaction pipeline stage is DELETED


def test_the_standalone_redaction_stage_is_gone(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """D-019: redaction is not a stage, so it records no **stage result** and no progress.

    The mechanism weight goes with it. A weight for a mechanism nothing reports
    is a permanent share of the progress bar that never fills, which is what
    "keep the user-visible progress coherent" rules out.
    """
    assert not hasattr(distill_session.ProcessingRun, "_produce_redaction")
    assert "redaction" not in {weight.mechanism for weight in DEFAULT_MECHANISM_WEIGHTS}

    output_dir = run_interrupted_after_the_stages(
        monkeypatch,
        tmp_path,
        ocr_text="a slide with no secret on it",
    )

    assert list(output_dir.rglob("_redaction.json")) == []
    assert list(output_dir.rglob("_ocr.json")) != []


# 8. End-to-end: a secret placed in OCR, transcript and a link label is redacted
#    in every file under the published generation


def test_no_file_in_a_published_generation_holds_the_secret(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Every **redaction sink** at once: image text, speech, an interpretation, a link.

    The related links are built the way a run builds them - from the source's
    metadata - because that is where the carrier is constructed and therefore
    where the policy runs.

    Captions are on. The vision model's reading of a **keyframe** is extracted
    text too - it echoes the screen back - and it is the one sink M4.2 does not
    move: `local_vision` still redacts its own fields, and M4.4 is where that
    becomes carrier construction like the rest. Asserted here so the transition
    has something to hold it.
    """
    video = tmp_path / "fixture.mp4"
    make_short_screencast(video)
    monkeypatch.setattr(
        distill_session,
        "transcribe_with_imports",
        transcript_saying(f"the key is {SECRET}"),
    )
    monkeypatch.setattr(distill_session, "ocr_frames", ocr_reading(f"API_KEY={SECRET}"))
    monkeypatch.setattr(distill_session, "probe_local_vision", vision_probe)
    monkeypatch.setattr(distill_session, "try_interpret_image_after_probe", vision_reading(SECRET))
    source = SourceInfo(
        source_type="youtube",
        resolved_path=video,
        duration_sec=1.0,
        source_fingerprint="fingerprint",
        source_hash=BUNDLE_KEY,
        warnings=[],
        provenance=Provenance(
            canonical_url="https://www.youtube.com/watch?v=abcdefghijk",
            duration_sec=1.0,
            processed_at="2026-07-29T14:20:00Z",
        ),
        related_links=extract_relevant_links(
            f"Skill repo ({SECRET}): https://github.com/example/repo?api_key={SECRET}",
            source="youtube_description",
        ),
    )
    output_root = tmp_path / "cache"
    output_root.mkdir()

    response = distill_session.process_resolved_source(
        source,
        DistillOptions(ocr=True, caption_frames=True, max_keyframes=3),
        output_root,
    )

    generation = published_generation(response)
    render = (generation / "video.md").read_text()
    assert (generation / "transcript.json").is_file()
    # Each sink is asserted present in redacted form as well as absent in raw
    # form, so a sink that quietly stopped being written cannot pass this by
    # having nothing in it to find.
    assert "a terminal showing" in render
    assert "github.com/example/repo" in render
    assert "the key is" in render
    assert files_holding(output_root, SECRET) == []


# 9. R-50: a format the pattern set learned reaches every sink without anything
#    else being told about it


def test_a_newly_covered_credential_format_is_redacted_in_a_published_generation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """R-50 with R-19: adding a pattern is the whole change.

    Redaction moved to carrier construction in M4.2, so a format the pattern set
    learns is covered at every **redaction sink** at once. That is a claim about
    the arrangement, not about `redact_text`, so it is asserted where a user
    would find out it was false: in a published **generation**.

    Both credentials here were invisible to the pre-R-50 pattern set - a GitLab
    token matched nothing, and an opaque bearer value is too short for the
    generic base64 rule - so this run publishes them verbatim if either pattern
    is removed.
    """
    # 20 characters after the prefix, which is what GitLab issues. Assembled
    # rather than written as one literal so the file carries no token-shaped
    # string for a secret scanner to block on.
    gitlab_token = "glpat-" + "3xAmPl3T0k3nV4lu3abc"
    bearer_value = "9f8c2b1a4d7e6f0c3b5a8d92e1f4c7b0"
    video = tmp_path / "fixture.mp4"
    make_short_screencast(video)
    monkeypatch.setattr(
        distill_session,
        "transcribe_with_imports",
        transcript_saying(f"the runner token is {gitlab_token}"),
    )
    monkeypatch.setattr(
        distill_session,
        "ocr_frames",
        ocr_reading(f"curl -H 'Authorization: Bearer {bearer_value}' https://api.example.com"),
    )
    source = SourceInfo(
        source_type="youtube",
        resolved_path=video,
        duration_sec=1.0,
        source_fingerprint="fingerprint",
        source_hash=BUNDLE_KEY,
        warnings=[],
        provenance=Provenance(
            canonical_url="https://www.youtube.com/watch?v=abcdefghijk",
            duration_sec=1.0,
            processed_at="2026-07-29T14:20:00Z",
        ),
        related_links=extract_relevant_links(
            f"Runner setup: https://gitlab.com/example/repo?private_token={gitlab_token}",
            source="youtube_description",
        ),
    )
    output_root = tmp_path / "cache"
    output_root.mkdir()

    response = distill_session.process_resolved_source(
        source,
        DistillOptions(ocr=True, caption_frames=False, max_keyframes=3),
        output_root,
    )

    generation = published_generation(response)
    render = (generation / "video.md").read_text()
    # Each sink is asserted present in redacted form as well as absent in raw
    # form, so a sink that quietly stopped being written cannot pass by having
    # nothing in it to find.
    assert "Bearer [REDACTED:oauth-token]" in render
    assert "the runner token is" in render
    assert "gitlab.com/example/repo" in render
    assert files_holding(output_root, gitlab_token) == []
    assert files_holding(output_root, bearer_value) == []
