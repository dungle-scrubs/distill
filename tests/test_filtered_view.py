"""The read-side filtered view of an already-published **bundle**.

The seam is `filtered_view.filtered_view_markdown`: an output root and a
**bundle key** in, the non-authoritative filtered markdown out. Every fixture
here is a bundle a real run could have published - built through
`BundleStore.open` / `begin` / `commit` with `manifest_document` and
`response_frames` - because the whole surface reads a **manifest** another
process wrote, and a hand-written one would let these tests agree with a shape
no run produces.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from distill.artifacts import FrameArtifact, Provenance, RedactionState, Transcript
from distill.bundle_store import MANIFEST_NAME, BundleRun, BundleStore
from distill.errors import DistillError
from distill.filtered_view import filtered_view_markdown
from distill.options import DistillOptions
from distill.render import FILTERED_VIEW_BANNER, render_markdown
from distill.response import manifest_document, response_frames
from distill.source import SourceInfo

BUNDLE_KEY = "hash"


def publish(
    root: Path,
    frames: list[FrameArtifact],
    *,
    transcript: Transcript | None = None,
    render: str = "# Video Bundle\n",
    warnings: list[dict[str, Any]] | None = None,
    options: DistillOptions | None = None,
    bundle_key: str = BUNDLE_KEY,
) -> BundleStore:
    """Publish one **generation** the way a run does, and hand back the store."""
    root.mkdir(parents=True, exist_ok=True)
    store = BundleStore.open(root)
    run = store.begin(bundle_key)
    assert isinstance(run, BundleRun)
    for frame in frames:
        image = Path(frame.path)
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"png")
    run.write_render(render)
    run.write_self_contained_render(render)
    if transcript is not None:
        run.write_transcript(transcript)
    video = root.parent / "video.mp4"
    video.write_bytes(b"video")
    run.commit(
        manifest_document(
            SourceInfo(
                source_type="local",
                resolved_path=video,
                duration_sec=2.0,
                source_fingerprint="fingerprint",
                source_hash=bundle_key,
                warnings=[],
                provenance=Provenance(
                    title=video.name,
                    duration_sec=2.0,
                    processed_at="2026-08-01T00:00:00Z",
                ),
            ),
            options or DistillOptions(),
            transcript_present=transcript is not None,
            frames=response_frames(frames),
            warnings=list(warnings or []),
        )
    )
    return store


def frame(
    root: Path,
    index: int,
    *,
    text: str = "",
    salience: dict[str, Any] | None = None,
    redaction: RedactionState = RedactionState.NOT_APPLIED,
) -> FrameArtifact:
    """A **frame artifact** addressed into the staging directory of `g1`."""
    relative = f"frames/frame_{index:04d}.png"
    return FrameArtifact(
        index=index,
        timestamp_sec=float(index),
        path=str(root / BUNDLE_KEY / ".tmp.g1" / relative),
        relative_path=relative,
        extracted_text=text,
        salience=salience,
        redaction=redaction,
    )


_ABSENT = object()
"""What `rewrite_manifest` removes a key with, since `None` is a value."""


def rewrite_manifest(store: BundleStore, edit: dict[str, Any]) -> None:
    """Change the published **manifest** on disk, the way another process could.

    R-23's premise made concrete: a manifest is a document some other process
    wrote, and the only way to test a view against one that says something a
    run would not have written is to write it after the run published.
    """
    path = store.bundle_root(BUNDLE_KEY) / MANIFEST_NAME
    document = json.loads(path.read_text())
    for key, value in edit.items():
        if value is _ABSENT:
            document.pop(key, None)
        else:
            document[key] = value
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def test_absent_generation_is_a_typed_error(tmp_path: Path) -> None:
    """No **active generation** under the key is a refusal, never an empty view.

    An empty view is a document a reader would keep: it says the bundle held no
    frame worth showing, which is a claim about a generation rather than about
    a key that names none.
    """
    root = tmp_path / "output"
    root.mkdir()

    with pytest.raises(DistillError) as raised:
        filtered_view_markdown(root, BUNDLE_KEY)

    assert raised.value.code == "E_NO_ACTIVE_GENERATION"
    assert raised.value.details["bundle_key"] == BUNDLE_KEY


def test_manifest_without_frames_is_a_typed_error(tmp_path: Path) -> None:
    """A manifest missing `frames` yields a refusal, never a frameless view.

    The refusal is the store's rather than this module's: `frames` is a
    schema-required field, so a manifest without one is not a **bundle marker**
    at all and `load_active` never returns it. Pinned here anyway, because the
    property the view owes a reader is that a manifest it cannot read produces
    an error - not which layer noticed.
    """
    root = tmp_path / "output"
    store = publish(root, [frame(root, 1, text="visible")])
    rewrite_manifest(store, {"frames": _ABSENT})

    with pytest.raises(DistillError) as raised:
        filtered_view_markdown(root, BUNDLE_KEY)

    assert raised.value.code == "E_NO_ACTIVE_GENERATION"


@pytest.mark.parametrize(
    ("malformed", "why"),
    [
        (["frame_0001.png"], "an entry that is not a document at all"),
        ([{"index": "one", "timestamp_sec": 1.0, "relative_path": "frames/f.png"}], "a text index"),
        ([{"index": 1, "timestamp_sec": 1.0}], "no relative_path to address the image by"),
        (
            [{"index": 1, "timestamp_sec": 1.0, "relative_path": "../../escape.png"}],
            "a relative_path leaving the generation",
        ),
        (
            [
                {
                    "index": 1,
                    "timestamp_sec": float("inf"),
                    "relative_path": "frames/f.png",
                    "ocr_text": "visible",
                }
            ],
            "a timestamp no clock could have produced",
        ),
        (
            [
                {
                    "index": 1,
                    "timestamp_sec": float("nan"),
                    "relative_path": "frames/f.png",
                    "ocr_text": "visible",
                }
            ],
            "a timestamp that is not a number",
        ),
    ],
)
def test_malformed_frames_are_a_typed_error(tmp_path: Path, malformed: list[Any], why: str) -> None:
    """A frame document that cannot be thawed ends the view, never trims it.

    `frames` being a list is all the manifest schema promises, so every one of
    these reaches this module as an **active generation** the store accepted.
    Dropping the entry would produce a document that reads as a filtered view -
    frames omitted, banner attached - while the omission was a parse failure
    rather than a judgment of redundancy, which is the one confusion the banner
    cannot survive.
    """
    root = tmp_path / "output"
    store = publish(root, [frame(root, 1, text="visible")])
    rewrite_manifest(store, {"frames": malformed})

    with pytest.raises(DistillError) as raised:
        filtered_view_markdown(root, BUNDLE_KEY)

    assert raised.value.code == "E_BAD_MANIFEST", why


def test_judged_redundant_frames_are_absent_under_the_banner(tmp_path: Path) -> None:
    """The view a reader asked for: the banner, and only the frames left standing.

    Salience survives publication as machine-readable text on the manifest
    frame (D-006), which is the whole reason this view can be produced from a
    published bundle at all rather than only from a live run's carriers.
    """
    root = tmp_path / "output"
    publish(
        root,
        [
            frame(root, 1, text="kept one", salience={"adds_information": True, "reason": "new"}),
            frame(
                root,
                2,
                text="dropped one",
                salience={"adds_information": False, "reason": "restates"},
            ),
        ],
    )

    view = filtered_view_markdown(root, BUNDLE_KEY)

    assert FILTERED_VIEW_BANNER in view
    assert "kept one" in view
    assert "## Frame 1" in view
    assert "dropped one" not in view
    assert "## Frame 2" not in view


def test_a_frame_without_a_readable_salience_is_kept(tmp_path: Path) -> None:
    """Absent salience is the absence of a judgment, not a judgment of redundancy.

    Two ways a frame arrives without one, and they have to answer the same:
    nothing was ever recorded, and something was recorded that does not read
    back as a judgment. `FrameSalience.from_document` refuses the second at the
    same strictness the live parse applies, and the frame it belonged to is
    kept either way - dropping it would let a drifted document remove a frame
    under a banner claiming a model judged it.
    """
    root = tmp_path / "output"
    store = publish(
        root,
        [
            frame(root, 1, text="never judged"),
            frame(root, 2, text="omitted frame", salience={"adds_information": False}),
            frame(root, 3, text="unreadable judgment"),
        ],
    )
    published = json.loads((store.bundle_root(BUNDLE_KEY) / MANIFEST_NAME).read_text())["frames"]
    published[2]["salience"] = {"adds_information": "no"}
    rewrite_manifest(store, {"frames": published})

    view = filtered_view_markdown(root, BUNDLE_KEY)

    assert "never judged" in view
    assert "unreadable judgment" in view
    assert "omitted frame" not in view


def test_a_wholly_redundant_generation_is_a_view_and_not_an_error(tmp_path: Path) -> None:
    """Everything judged redundant is something the banner says, not a failure.

    The stored generation is intact and servable; what is empty is one reading
    of it. A **fatal error** here would tell an operator their bundle is broken
    on the strength of a model's opinion about its frames - and `E_NO_CONTENT`,
    which is what the ordinary render raises with nothing to show, is a
    processing verdict that a read-side view has no standing to reach.
    """
    root = tmp_path / "output"
    publish(
        root,
        [
            frame(root, 1, text="restated once", salience={"adds_information": False}),
            frame(root, 2, text="restated twice", salience={"adds_information": False}),
        ],
    )

    view = filtered_view_markdown(root, BUNDLE_KEY)

    assert view.startswith(FILTERED_VIEW_BANNER)
    assert "Every frame in this generation was judged redundant" in view
    assert "restated once" not in view
    assert "restated twice" not in view


# `AKIA…` with a fullwidth `Ｉ` in it: no secret rule matches the raw run, and
# every rule matches it once `normalize_confusables` has had its turn. It is the
# one shape whose redaction the policy also *reports*, which is what makes the
# divergence below observable rather than merely argued.
CONFUSABLE_SECRET = "AKＩA0123456789ABCDEF"


def test_a_no_redact_bundle_diverges_from_its_stored_render(tmp_path: Path) -> None:
    """A-001's accepted cost, asserted rather than left to be discovered.

    `--no-redact-secrets` is an applied policy (D-020) and the generation it
    published holds the raw text on purpose. This view passes
    `RedactionState.APPLIED` regardless, because the parameter is an
    instruction rather than a label - so over this bundle the view is *not* a
    faithful projection of the stored render: its text is redacted where the
    render's is not, and it carries a **warning** the generation never raised.

    Both halves are the same trade and both are stated here. The alternative -
    reading the policy off the manifest - is the R-23 exception A-001 exists to
    avoid needing, and it would let a forged manifest turn the view's policy
    off.
    """
    root = tmp_path / "output"
    frames = [
        frame(
            root,
            1,
            text=f"config: {CONFUSABLE_SECRET}",
            salience={"adds_information": True},
            redaction=RedactionState.DISABLED,
        )
    ]
    store = publish(
        root,
        frames,
        render=render_markdown("video.mp4", 2.0, None, frames, []),
        options=DistillOptions(redact_secrets=False),
    )
    snapshot = store.load_active(BUNDLE_KEY)
    assert snapshot is not None
    stored = snapshot.markdown.read_text()

    view = filtered_view_markdown(root, BUNDLE_KEY)

    assert CONFUSABLE_SECRET in stored
    assert "possible_confusable_secret" not in stored
    assert CONFUSABLE_SECRET not in view
    assert "[REDACTED:aws-key]" in view
    assert "possible_confusable_secret" in view


def test_a_generations_own_warning_is_not_counted_twice(tmp_path: Path) -> None:
    """The view reports one event once, the way the **manifest** does (R-41).

    A **transcript** is the one input here that records warnings of its own, and
    they are the run's - already folded into the manifest. Rebuilding the
    carrier from the document it was written as, and then folding what the
    carrier holds together with what the manifest holds, would add that record
    to itself and report one truncated or redacted field as two events.
    """
    root = tmp_path / "output"
    spoken = Transcript(
        language="en",
        segments=({"start": 0.0, "end": 2.0, "text": f"the key is {CONFUSABLE_SECRET}"},),
    )
    raised = [dict(record) for record in spoken.warnings]
    assert [record["code"] for record in raised] == ["possible_confusable_secret"]

    publish(
        root,
        [frame(root, 1, text="kept", salience={"adds_information": True})],
        transcript=spoken,
        warnings=raised,
    )

    view = filtered_view_markdown(root, BUNDLE_KEY)

    assert view.count("code: possible_confusable_secret") == 1
    assert "occurrences: 1" in view
    assert "occurrences: 2" not in view


def test_a_transcript_symlink_is_refused_rather_than_followed(tmp_path: Path) -> None:
    """R-16: the one symlink refusal covers every path this view opens.

    The frames are only linked to and never read, so the transcript is the one
    file here whose *contents* reach the output. A link planted at its name is
    how text Distill never recorded gets rendered into a document that says it
    describes this bundle - and the reader is told it is a filtered view of the
    bundle, which is what makes the substitution worth refusing rather than
    following.
    """
    root = tmp_path / "output"
    store = publish(
        root,
        [frame(root, 1, text="kept", salience={"adds_information": True})],
        transcript=Transcript(language="en", segments=({"start": 0.0, "end": 2.0, "text": "hi"},)),
    )
    snapshot = store.load_active(BUNDLE_KEY)
    assert snapshot is not None
    elsewhere = tmp_path / "elsewhere.json"
    elsewhere.write_text(
        json.dumps({"language": "en", "segments": [{"start": 0, "end": 1, "text": "smuggled"}]})
    )
    snapshot.transcript.unlink()
    snapshot.transcript.symlink_to(elsewhere)

    with pytest.raises(DistillError) as raised:
        filtered_view_markdown(root, BUNDLE_KEY)

    assert raised.value.code == "E_BAD_MANIFEST"


@pytest.mark.parametrize(
    ("segment", "why"),
    [
        ({"start": "bad", "end": 1.0, "text": "visible"}, "a start that is not a number"),
        ({"start": 0.0, "end": float("inf"), "text": "visible"}, "an end no clock produced"),
        ({"end": 1.0, "text": "visible"}, "no start to place it by"),
        ("a segment", "an entry that is not a segment at all"),
    ],
)
def test_an_unplaceable_transcript_segment_is_a_typed_error(
    tmp_path: Path, segment: Any, why: str
) -> None:
    """A **transcript** the view cannot place is refused, not rendered around.

    The render interleaves frames with speech by comparing timestamps, so a
    segment that cannot be placed is one the document has no position for.
    Skipping it would drop speech out of a view whose banner says redundant
    frames are its only omission; defaulting it to zero would move somebody's
    words to the start of the recording. Both are worse than saying the
    transcript does not read back, which is what a document another process
    wrote is allowed to turn out to be (R-23).
    """
    root = tmp_path / "output"
    store = publish(
        root,
        [frame(root, 1, text="kept", salience={"adds_information": True})],
        transcript=Transcript(language="en", segments=({"start": 0.0, "end": 2.0, "text": "hi"},)),
    )
    snapshot = store.load_active(BUNDLE_KEY)
    assert snapshot is not None
    snapshot.transcript.write_text(json.dumps({"language": "en", "segments": [segment]}))

    with pytest.raises(DistillError) as raised:
        filtered_view_markdown(root, BUNDLE_KEY)

    assert raised.value.code == "E_BAD_MANIFEST", why


@pytest.mark.parametrize(
    ("duration", "why"),
    [
        (10**400, "a duration no float can hold"),
        (float("inf"), "a duration no clock could have measured"),
        (float("nan"), "a duration that is not a number"),
    ],
)
def test_a_duration_that_is_not_a_measurement_is_a_typed_error(
    tmp_path: Path, duration: Any, why: str
) -> None:
    """The manifest's duration is a number to the schema and a measurement here.

    `validate_manifest_schema` asks only that it be an `int` or a `float`, which
    `Infinity` and a four-hundred-digit integer both are. The render prints it
    as a duration, so the difference shows up either as a document claiming a
    recording ran for `inf` seconds or as an `OverflowError` out of a read-only
    command - a failure with no diagnosis in it.
    """
    root = tmp_path / "output"
    store = publish(root, [frame(root, 1, text="kept", salience={"adds_information": True})])
    rewrite_manifest(store, {"duration_sec": duration})

    with pytest.raises(DistillError) as raised:
        filtered_view_markdown(root, BUNDLE_KEY)

    assert raised.value.code == "E_BAD_MANIFEST", why
