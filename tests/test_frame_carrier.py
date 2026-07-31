"""The **frame artifact** carrier as the only description of a frame's shape.

R-19/R-20, M4.4: `frame_selection`, `ocr`, `local_vision`, `render` and
`bundle_store` used to each restate the frame schema as string keys on a bare
dict, so the **redaction** policy could only ever be applied to a copy of text
that was already durable (finding 4). These tests hold the migration: the five
modules speak `FrameArtifact`, and nothing under `src/distill/` builds a frame
document by hand except the two modules whose job that is.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest

from distill.artifacts import (
    FrameArtifact,
    Interpretation,
    RedactionPolicyNotApplied,
    RedactionState,
)
from distill.bundle_store import BundleRun, BundleStore
from distill.frame_selection import select_keyframes
from distill.render import render_markdown
from distill.response import response_frames

PACKAGE_DIR = Path(__file__).resolve().parents[1] / "src" / "distill"

FRAME_SCHEMA_KEYS = frozenset(
    {
        # the carrier's own field names
        "index",
        "timestamp_sec",
        "path",
        "relative_path",
        "phash",
        "source_candidate_index",
        "extracted_text",
        "interpretation",
        "grounding",
        # the names the bare frame dict used for the same three things, kept in
        # the guarded set so reintroducing the old spelling is caught as loudly
        # as reintroducing the new one
        "ocr_text",
        "visual_interpretation",
        "visual_confidence",
    }
)

FRAME_SCHEMA_QUORUM = 3
"""How many guarded names in one mapping make it a frame document.

Three rather than one, because `path` and `index` are ordinary words that
mean something else in most mappings, and a rule that fires on any single one
of them would be turned off within a week. Three of them together is not a
coincidence: the bare frame dict this milestone deletes carried six.
"""

FRAME_DOCUMENT_AUTHORS = {
    "artifacts.py": (
        "Owns the carrier, so the one mapping it builds from these names is the "
        "carrier's own serialization - the thing every other module now goes "
        "through instead of restating the schema."
    ),
    "response.py": (
        "Owns the **manifest** and response frame document, which is a different "
        "schema that happens to share names with the carrier's: it is what a "
        "caller and a later cache hit read, not what a stage passes along. It "
        "builds that document out of carrier attributes, never out of another "
        "mapping."
    ),
}


def _frame_documents_built_in(source: str) -> list[tuple[int, tuple[str, ...]]]:
    """Every mapping literal in `source` that names a quorum of frame fields.

    Both spellings a contributor reaches for: a `{...}` display, and a
    `dict(index=..., path=...)` call.
    """
    found: list[tuple[int, tuple[str, ...]]] = []
    for node in ast.walk(ast.parse(source)):
        names: set[str] = set()
        if isinstance(node, ast.Dict):
            names = {
                key.value
                for key in node.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "dict"
        ):
            names = {keyword.arg for keyword in node.keywords if keyword.arg is not None}
        matched = names & FRAME_SCHEMA_KEYS
        if len(matched) >= FRAME_SCHEMA_QUORUM:
            found.append((getattr(node, "lineno", 0), tuple(sorted(matched))))
    return found


def test_no_module_constructs_a_bare_frame_dict() -> None:
    """A frame's shape is the carrier's, and only two modules may spell it out.

    This is the structural half of M4.4 and it is stated the way R-22's emitter
    test is (D-022). What it catches: a contributor following the pattern the
    codebase used before this milestone - building a frame as a mapping of
    string keys in whichever module needed one - which is how the same schema
    came to be restated in five places and how the **redaction** policy came to
    run somewhere other than where the text entered.

    What it does not catch, precisely:

    - anything inside `FRAME_DOCUMENT_AUTHORS`, which is a hole by construction:
      the two modules that legitimately name these fields are exactly the two a
      determined contributor could add a third to;
    - a mapping assembled key by key (`frame["ocr_text"] = text`), from a
      variable holding the names, or by `**` from another mapping;
    - a frame document built by a helper in a third-party package;
    - fewer than `FRAME_SCHEMA_QUORUM` of the names at once.

    It is a guard against drift, not a proof that drift is impossible.
    """
    offenders: dict[str, list[tuple[int, tuple[str, ...]]]] = {}
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        name = path.relative_to(PACKAGE_DIR).as_posix()
        if "__pycache__" in path.parts or name in FRAME_DOCUMENT_AUTHORS:
            continue
        built = _frame_documents_built_in(path.read_text())
        if built:
            offenders[name] = built

    assert not offenders, (
        "modules build a frame document by hand instead of using the "
        f"FrameArtifact carrier: {offenders}. Pass the carrier along, or record "
        "the module in FRAME_DOCUMENT_AUTHORS with the reason it owns a frame "
        "document of its own."
    )


def test_the_structural_check_sees_both_spellings_of_a_frame_dict() -> None:
    """The check itself can fail, on a display and on a `dict()` call alike.

    Asserted against source text rather than by editing the package, so the
    negative case cannot leave a module behind if this process is killed.
    """
    display = "frame = {'index': 1, 'timestamp_sec': 2.0, 'ocr_text': text}"
    call = "frame = dict(index=1, relative_path=rel, visual_interpretation=reading)"
    innocent = "record = {'index': 1, 'label': 'x'}"

    assert _frame_documents_built_in(display) == [(1, ("index", "ocr_text", "timestamp_sec"))]
    assert _frame_documents_built_in(call) == [
        (1, ("index", "relative_path", "visual_interpretation"))
    ]
    assert _frame_documents_built_in(innocent) == []


def test_every_frame_document_author_is_a_module_that_exists() -> None:
    """An allowlist naming a module that is gone is an exemption nothing needs."""
    missing = sorted(name for name in FRAME_DOCUMENT_AUTHORS if not (PACKAGE_DIR / name).is_file())
    assert not missing, f"FRAME_DOCUMENT_AUTHORS names modules that do not exist: {missing}"

    unexplained = sorted(name for name, why in FRAME_DOCUMENT_AUTHORS.items() if not why.strip())
    assert not unexplained, f"frame-document authors without a recorded reason: {unexplained}"


@pytest.mark.parametrize("module", sorted(FRAME_DOCUMENT_AUTHORS))
def test_each_allowlisted_author_really_does_build_one(module: str) -> None:
    """An allowlist entry that guards nothing is an exemption that has gone stale.

    If a module stops building a frame document, its entry stops being a
    recorded decision and starts being a hole nobody is watching.
    """
    built = _frame_documents_built_in((PACKAGE_DIR / module).read_text())
    assert built, f"{module} is allowlisted but builds no frame document"


def _extracts_every_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make keyframe extraction succeed without ffmpeg or Pillow being installed."""
    from distill import frame_selection

    def fake_extract(_video: Path, _timestamp: float, output: Path) -> tuple[bool, list[Any]]:
        output.write_bytes(b"png")
        return True, []

    monkeypatch.setattr(frame_selection, "scene_midpoint_candidates", lambda _p, _d: [0.0])
    monkeypatch.setattr(frame_selection, "extract_frame", fake_extract)
    monkeypatch.setattr(frame_selection, "phash", lambda _path: "0f")


def test_frame_selection_produces_carriers_not_dicts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A **keyframe** becomes a `FrameArtifact` the moment it is kept.

    Everything downstream reads a frame through the carrier, so the carrier has
    to be what the first stage produces - a translation layer somewhere later
    is exactly the arrangement M4.2 left behind and M4.4 removes.
    """
    _extracts_every_candidate(monkeypatch)

    frames, warnings = select_keyframes(
        Path("demo.mp4"),
        tmp_path,
        duration_sec=1.0,
        max_keyframes=1,
        min_interval_sec=1.0,
        max_static_window_sec=90.0,
    )

    assert warnings == []
    assert [type(frame) for frame in frames] == [FrameArtifact]
    assert frames[0].index == 1
    assert frames[0].relative_path == "frames/frame_0001.png"


def test_the_runs_redaction_policy_enters_the_frames_once_and_travels_with_them(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`--no-redact-secrets` is recorded at selection and inherited by every stage.

    R-20 keeps the opt-out working and D-020 makes it a state rather than an
    inference. Recording it where a frame begins is what means no later stage
    can be handed the wrong policy: `ocr_frames` and `FrameInterpreter` take no
    redaction argument at all, so there is nothing for a caller to get wrong.
    """
    _extracts_every_candidate(monkeypatch)
    secret = "OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz"

    opted_out, _warnings = select_keyframes(
        Path("demo.mp4"),
        tmp_path,
        duration_sec=1.0,
        max_keyframes=1,
        min_interval_sec=1.0,
        max_static_window_sec=90.0,
        redaction=RedactionState.DISABLED,
    )
    read, _ = opted_out[0].with_extracted_text(secret)
    interpreted, _ = read.with_interpretation(Interpretation(verbatim_text=secret))

    assert interpreted.redaction is RedactionState.DISABLED
    assert interpreted.extracted_text == secret
    reading = interpreted.reading
    assert reading is not None
    assert reading.verbatim_text == secret


def test_a_stage_result_is_written_by_serializing_the_carriers_in_it(
    tmp_path: Path,
) -> None:
    """`bundle_store` turns carriers into documents, and refuses one it may not.

    R-20 is enforced in `serialize`, which is the last point before **extracted
    text** becomes durable. A payload holding carriers therefore has to reach it
    carrier by carrier: flattening them some other way - `json.dumps` with a
    `default=str`, say - would put the text on disk without the check ever
    running, which is finding 4 with an extra step.
    """
    store = BundleStore.open(tmp_path / "output")
    began = store.begin("abc123")
    assert isinstance(began, BundleRun)
    secret = "OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz"
    frame = FrameArtifact(
        index=1,
        timestamp_sec=0.0,
        path=str(began.frames_dir / "frame_0001.png"),
        relative_path="frames/frame_0001.png",
        extracted_text=secret,
    )

    began.write_stage("ocr", {"frames": [frame], "warnings": []})
    recovered = began.read_stage("ocr")
    began.release()

    assert recovered is not None
    document = recovered["frames"][0]
    assert document["extracted_text"] == "OPENAI_API_KEY=[REDACTED:assigned-secret]"
    assert document["redaction"] == RedactionState.APPLIED.value
    assert secret not in (began.paths.generation / "_ocr.json").read_text()


def test_a_carrier_whose_policy_never_ran_is_refused_at_the_write(tmp_path: Path) -> None:
    """The serializer's check reaches a carrier nested in a stage payload.

    Construction cannot produce `NOT_APPLIED`, so what arrives here is a carrier
    assembled around `__post_init__` - the documented bypass. Refusing it is
    what makes the store's serialization the choke point rather than a formality
    the payload could route around by nesting.
    """
    store = BundleStore.open(tmp_path / "output")
    began = store.begin("abc123")
    assert isinstance(began, BundleRun)
    frame = FrameArtifact(index=1, timestamp_sec=0.0, path="p", relative_path="frames/p.png")
    object.__setattr__(frame, "redaction", RedactionState.NOT_APPLIED)

    try:
        with pytest.raises(RedactionPolicyNotApplied):
            began.write_stage("ocr", {"frames": [frame]})
    finally:
        began.release()


def test_a_resumed_run_rebuilds_its_carriers_from_the_stage_result(tmp_path: Path) -> None:
    """A **stage result** is JSON on disk, so a **resume** rebuilds the carriers.

    Under `--no-redact-secrets` the resuming run's policy is `DISABLED`, so the
    scratch it recorded comes back holding what the first run held. `redact_secrets`
    participates in the **options hash**, so the resuming run under this **bundle
    key** is necessarily the one that wrote it.
    """
    secret = "OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz"
    store = BundleStore.open(tmp_path / "output")
    first = store.begin("abc123")
    assert isinstance(first, BundleRun)
    frame = FrameArtifact(
        index=1,
        timestamp_sec=0.0,
        path=str(first.frames_dir / "frame_0001.png"),
        relative_path="frames/frame_0001.png",
        extracted_text=secret,
        redaction=RedactionState.DISABLED,
    )
    first.write_stage("ocr", {"frames": [frame]}, redaction=RedactionState.DISABLED)
    first.release()

    second = store.begin("abc123")
    assert isinstance(second, BundleRun)
    payload = second.read_stage("ocr")
    second.release()
    assert payload is not None

    resumed = FrameArtifact.from_document(
        payload["frames"][0], redaction=RedactionState.DISABLED
    )
    assert resumed.redaction is RedactionState.DISABLED
    assert resumed.extracted_text == secret
    assert resumed.relative_path == "frames/frame_0001.png"


def test_a_stage_result_cannot_talk_a_resume_out_of_the_policy_it_is_under() -> None:
    """The resuming run's policy decides, never the document's claim about itself.

    R-23's premise is that a **stage result** is a document some other process
    wrote, so everything it says is input. A document recording
    `"redaction": "disabled"` beside raw text would otherwise downgrade a run
    that asked for redaction, and that text would be published - the resume
    turned into a way to opt a user out of the policy they chose.
    """
    secret = "OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz"
    forged = {
        "index": 1,
        "timestamp_sec": 0.0,
        "path": "/tmp/frames/frame_0001.png",
        "relative_path": "frames/frame_0001.png",
        "extracted_text": secret,
        "redaction": RedactionState.DISABLED.value,
    }

    resumed = FrameArtifact.from_document(forged, redaction=RedactionState.NOT_APPLIED)

    assert resumed.redaction is RedactionState.APPLIED
    assert resumed.extracted_text == "OPENAI_API_KEY=[REDACTED:assigned-secret]"


def test_the_render_refuses_a_carrier_whose_policy_never_ran() -> None:
    """R-20 names a **render** as a sink beside a **generation**, so it checks too.

    Reached only by the documented bypass, because construction cannot produce
    `NOT_APPLIED`. That is the point: what arrives at a sink is a carrier
    assembled around `__post_init__`, and refusing it there is worth more than
    reading its text out.
    """
    frame = FrameArtifact(
        index=1,
        timestamp_sec=0.0,
        path="/tmp/frames/frame_0001.png",
        relative_path="frames/frame_0001.png",
        extracted_text="slide text",
    )
    object.__setattr__(frame, "redaction", RedactionState.NOT_APPLIED)

    with pytest.raises(RedactionPolicyNotApplied):
        render_markdown("demo.mp4", 1.0, None, [frame], [])


def test_the_manifest_frame_document_is_the_serialized_carrier() -> None:
    """The **manifest** is a sink too, so its frame document comes out of `serialize`.

    Two things follow from that and both matter. A carrier whose policy never
    ran is refused rather than copied into a durable document; and the values
    are plain JSON, so a caller reading a fresh run and a caller reading the
    cache hit that replays the same manifest get the same types.
    """
    reading = Interpretation(visual_summary="a console", detected_elements=("a button",))
    frame, _warnings = FrameArtifact(
        index=1,
        timestamp_sec=0.0,
        path="/tmp/frames/frame_0001.png",
        relative_path="frames/frame_0001.png",
    ).with_interpretation(reading)

    document = response_frames([frame])[0]

    assert document["visual_interpretation"]["detected_elements"] == ["a button"]
    assert json.loads(json.dumps(document)) == document

    object.__setattr__(frame, "redaction", RedactionState.NOT_APPLIED)
    with pytest.raises(RedactionPolicyNotApplied):
        response_frames([frame])


def test_the_policy_does_not_rewrite_the_paths_distill_chose(tmp_path: Path) -> None:
    """A path is Distill's own words, so the **redaction** policy leaves it alone.

    A **bundle key** is 64 hex characters, which is exactly the shape a generic
    secret pattern matches. With path fields treated as **extracted text**, a
    **stage result** recorded `/[REDACTED]/.tmp.g1/frames/frame_0001.png`; the
    **resume** that read it back could not confine that path to the bundle root
    and discarded the whole document, so every stage after the transcript was
    recomputed on every resumed run and nothing said so.

    Asserted on the bytes on disk, because that is where the damage was: a
    document whose paths survive is the whole difference between a resume that
    works and one that silently does not.
    """
    store = BundleStore.open(tmp_path / "output")
    key = "a" * 64
    began = store.begin(key)
    assert isinstance(began, BundleRun)
    image = began.frames_dir / "frame_0001.png"
    frame = FrameArtifact(
        index=1,
        timestamp_sec=0.0,
        path=str(image),
        relative_path="frames/frame_0001.png",
        extracted_text="OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz",
    )

    began.write_stage("frames", {"frames": [frame]})
    recovered = began.read_stage("frames")
    on_disk = (began.paths.generation / "_frames.json").read_text()
    began.release()

    assert recovered is not None, "a stage result whose paths were rewritten cannot resume"
    assert recovered["frames"][0]["path"] == str(image)
    assert key in on_disk
    # The extracted text beside it is still redacted: the exemption is the path
    # field, not the document.
    assert recovered["frames"][0]["extracted_text"] == "OPENAI_API_KEY=[REDACTED:assigned-secret]"
