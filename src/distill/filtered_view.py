"""The read-side filtered view of an already-published **bundle** (D-006).

This module owns one thing: turning a **bundle key** into the non-authoritative
filtered markdown of the **active generation** under it. It reads a **manifest**
and a **generation** back into the carriers `render` speaks - **frame
artifacts**, a **transcript**, **related links** - and hands
`render_filtered_markdown` its inputs.

Read-side rather than a processing option, and that is the decision rather than
a convenience. The filtered view is produced on demand, is never written back,
and does not exist under the **bundle key** (D-006, D-017), so a
`--skip-redundant-frames` flag on a processing subcommand would describe a
stored generation shaped by the flag - and no such generation exists. What the
filter acts on says the same thing from the other side: salience is recorded on
every frame at publication and is informational only (D-018), consulted by
nothing in processing. Dropping a frame is therefore a way of *reading* a
generation, and a reading belongs to whoever is reading rather than to the run
that produced it.

Everything it reads is another process's document and never fact (R-23): the
manifest was written by a run this command did not make and cannot see. Two
consequences, both deliberate. A frame document that will not rebuild ends the
view instead of being skipped, because a skipped frame and a judged-redundant
one are the same absence under a banner that claims the second. And the
**redaction** policy is this command's own - `APPLIED` for every bundle, taken
from nothing the manifest claims about itself (A-001, D-019) - which is what
makes the view safe to produce over a bundle whose policy this command has no
way to verify, at the cost stated on `FrameArtifact.from_document`.

It is read-only, and that is a property of the path rather than a promise made
here: `load_active` opens nothing for writing and takes no lock, so a view can
be produced while another run holds the same key and it reads the generation
that was active when it looked - complete, because a **bundle marker** is not
written until it is.

What it does not own: which frames the banner omits and what the banner says,
which are `render.render_filtered_markdown`'s; the manifest frame document's
schema, which is `response.py`'s in both directions; and what makes a
generation servable, which is `bundle_store.load_active`'s. `cache_doctor` is
the neighbouring read-only surface over the same store.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .artifacts import Carrier, FrameArtifact, RedactionState, Transcript
from .bundle_store import BundleSnapshot, BundleStore, confined_path
from .errors import DistillError, WarningRecord, aggregate_warnings
from .links import RelatedLink
from .render import render_filtered_markdown
from .response import frame_carrier_document
from .transcript import segment_bounds

FILTERED_VIEW_STAGE = "filtered_view"

NO_ACTIVE_GENERATION_CODE = "E_NO_ACTIVE_GENERATION"
"""Nothing servable under the **bundle key** the caller named.

A refusal rather than an empty view, because an empty view is a document a
reader would keep: it says this generation held no frame worth showing, which
is a claim about a generation - and there is not one.
"""

BAD_MANIFEST_CODE = "E_BAD_MANIFEST"
"""A **manifest** or **generation** this view cannot be built from.

The store's own code for the same thing, deliberately: a document that does not
describe the bundle it sits in is one failure whether the schema check or this
module is what noticed, and two codes for it would be two things to handle.
"""


def filtered_view_markdown(output_root: Path, bundle_key: str) -> str:
    """The filtered view of the **active generation** `bundle_key` names.

    The whole read-side surface: an output root and a key in, markdown out,
    nothing written and nothing cached. Stated as a callable rather than left in
    `cli.py` so the behavior is testable without argv, which is what keeps the
    command a boundary.

    The render is asked for the same shape the generation's own **render**
    has - frame links included, no **provenance** block, which is the pair
    `pipeline` renders it under - so a reader comparing the two finds them
    differing by the frames the banner names and by the redaction policy this
    module runs, and by nothing else.
    """
    store = BundleStore.open(output_root)
    snapshot = store.load_active(bundle_key)
    if snapshot is None:
        raise DistillError(
            NO_ACTIVE_GENERATION_CODE,
            FILTERED_VIEW_STAGE,
            "no servable generation exists under this bundle key",
            {"bundle_key": bundle_key, "output_root": str(store.root)},
        )
    frames = _thawed_frames(snapshot)
    transcript = _thawed_transcript(snapshot)
    related_links = _thawed_related_links(snapshot)
    rebuilt: list[Carrier] = [
        *frames,
        *(() if transcript is None else (transcript,)),
        *related_links,
    ]
    return render_filtered_markdown(
        # Schema-required as a `str`, proven by the marker check `load_active`
        # went through, and printed as the **source** label - which is extracted
        # text the render delimits, so nothing here needs it to be anything else.
        str(snapshot.manifest.get("source_resolved_path", "")),
        _measured_duration(snapshot),
        transcript,
        frames,
        _view_warnings(snapshot, rebuilt),
        related_links,
    )


def _measured_duration(snapshot: BundleSnapshot) -> float:
    """The manifest's recorded duration, as a number a render can print.

    The schema asks only for an `int` or a `float`, and `Infinity` and a
    four-hundred-digit integer are both of those without either being a
    duration. Left alone they read as a document claiming the recording ran for
    `inf` seconds, and as an `OverflowError` out of a read-only command - a
    failure with no diagnosis in it. Refused here for the reason the frames are:
    a manifest that cannot describe its own bundle is not a view.
    """
    recorded = snapshot.manifest.get("duration_sec", 0.0)
    try:
        duration = float(recorded)
    except (TypeError, ValueError, OverflowError) as exc:
        raise _bad_manifest(
            "the manifest records a duration that is not a number",
            {"bundle_key": snapshot.bundle_key, "error": str(exc)},
        ) from exc
    if not math.isfinite(duration):
        raise _bad_manifest(
            "the manifest records a duration nothing measured",
            {"bundle_key": snapshot.bundle_key, "duration_sec": repr(duration)},
        )
    return duration


def _thawed_frames(snapshot: BundleSnapshot) -> list[FrameArtifact]:
    """The manifest's frame documents as carriers, or a refusal (R-23).

    Every entry or none. A view that dropped the entries it could not read
    would be a document shaped exactly like a filtered view - frames missing,
    banner attached - whose missing frames were a parse failure rather than the
    judgment of redundancy the banner claims for them.

    `frames` being a list at all is checked here and normally decided earlier:
    it is schema-required, so a manifest without one is not a **bundle marker**
    and `load_active` never hands it over. The check stays because what makes
    that true is a schema table in another module, and the cost of it changing
    unnoticed is a `TypeError` out of a read-only command.
    """
    documents = snapshot.manifest.get("frames")
    if not isinstance(documents, list):
        raise _bad_manifest(
            "the manifest records no list of frames",
            {"bundle_key": snapshot.bundle_key, "frames": type(documents).__name__},
        )
    return [
        _thawed_frame(document, snapshot, position) for position, document in enumerate(documents)
    ]


def _thawed_frame(document: Any, snapshot: BundleSnapshot, position: int) -> FrameArtifact:
    """One manifest frame document as a **frame artifact** under this view's policy.

    The manifest carries the *response* projection of a frame, so the document
    is translated back into the carrier's own names by `response.py` - which
    owns both directions of that projection, and is why this module names one
    frame field rather than six. The one field is `path`: the manifest drops it
    deliberately, because it addresses the producing machine, and only a reader
    holding the **generation** can put it back. It goes back through the same
    confinement a **cache** hit uses, so a `relative_path` reaching out of the
    generation is a manifest this view refuses rather than a path it follows.

    `RedactionState.APPLIED` unconditionally, taking nothing from the manifest
    and no flag from the operator (A-001, D-019). `Carrier.__post_init__` reads
    the value as an instruction rather than a label, so this *runs* the policy:
    idempotent over a bundle published with redaction on, since the text was
    capped and redacted at write time, and strictly more redacting over one
    published under `--no-redact-secrets`. Neither direction can yield a view
    less redacted than the stored render, which is what lets this module take
    nothing from a manifest's claim about its own policy - the R-23 exception
    A-001 exists to avoid needing.
    """
    where = {"bundle_key": snapshot.bundle_key, "frame_position": position}
    if not isinstance(document, Mapping):
        raise _bad_manifest(
            "a manifest frame is not a document",
            {**where, "received": type(document).__name__},
        )
    values = frame_carrier_document(document)
    relative = values["relative_path"]
    if isinstance(relative, str):
        try:
            values["path"] = str(confined_path(snapshot.generation / relative, snapshot.generation))
        except DistillError as exc:
            raise _bad_manifest(
                "a manifest frame addresses an image outside its generation",
                {**where, "relative_path": relative, "refusal": exc.code},
            ) from exc
    try:
        return FrameArtifact.from_document(values, redaction=RedactionState.APPLIED)
    except (TypeError, ValueError) as exc:
        raise _bad_manifest(
            "a manifest frame is not the shape a frame artifact is rebuilt from",
            {**where, "error": str(exc)},
        ) from exc


def _thawed_transcript(snapshot: BundleSnapshot) -> Transcript | None:
    """The generation's **transcript** as a carrier, or `None` if it has none.

    Read on the terms the response reads it - the manifest says one exists and
    the file is there - because a bundle can outlive the transcript beside it
    and a view is not the place to discover that. A file that *is* there and
    does not read back is the other case, and it is refused: the banner names
    redundant frames as this document's only omission, and a view silently
    missing the speech would make that sentence false.
    """
    if not snapshot.manifest.get("transcript_present"):
        return None
    try:
        # R-16, before the file is opened rather than after: this is the one
        # path here whose *contents* reach the output - a frame image is linked
        # to and never read - so a link planted at its name is how text nobody
        # recorded gets rendered into a document that says it describes this
        # bundle. `confined_path` follows nothing, which is why it can answer.
        confined_path(snapshot.transcript, snapshot.root)
    except DistillError as exc:
        raise _bad_manifest(
            "the generation's transcript is not reached from inside the bundle",
            {"bundle_key": snapshot.bundle_key, "refusal": exc.code},
        ) from exc
    if not snapshot.transcript.is_file():
        return None
    try:
        document = json.loads(snapshot.transcript.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise _bad_manifest(
            "the generation's transcript could not be read back",
            {"bundle_key": snapshot.bundle_key, "error": str(exc)},
        ) from exc
    if not isinstance(document, Mapping):
        raise _bad_manifest(
            "the generation's transcript is not a document",
            {"bundle_key": snapshot.bundle_key, "received": type(document).__name__},
        )
    try:
        # The document's own **warnings** are left out of the rebuild rather
        # than inherited. They are the run's, and the manifest already carries
        # them folded (R-41), so a carrier that inherited them would have them
        # folded a second time in `_view_warnings` - reporting one truncated
        # field as two. What this carrier keeps is what constructing it *here*
        # raised, which is the only thing the manifest does not already say.
        transcript = Transcript.from_document(
            {key: value for key, value in document.items() if key != "warnings"},
            redaction=RedactionState.APPLIED,
        )
    except (TypeError, ValueError) as exc:
        raise _bad_manifest(
            "the generation's transcript is not the shape a transcript is rebuilt from",
            {"bundle_key": snapshot.bundle_key, "error": str(exc)},
        ) from exc
    # What a segment *holds* is `transcript.py`'s and the carrier does not check
    # it (D-041), so the one thing this render needs of a segment is asked here:
    # the render places every frame against segment bounds, and a segment it
    # cannot place either ends the command on an arithmetic error or - worse,
    # for a missing bound - is silently read as starting at zero, which moves
    # somebody's words to the beginning of the recording.
    unplaceable = [
        position
        for position, segment in enumerate(transcript.segments)
        if segment_bounds(segment) is None
    ]
    if unplaceable:
        raise _bad_manifest(
            "the generation's transcript holds a segment this view cannot place",
            {"bundle_key": snapshot.bundle_key, "segment_positions": unplaceable[:8]},
        )
    return transcript


def _thawed_related_links(snapshot: BundleSnapshot) -> list[RelatedLink]:
    """The manifest's **related links** as carriers, on `source.py`'s terms.

    An entry that is not a document is dropped rather than refused, which is
    what `_manifest_related_links` decided for a cache hit: a link that cannot
    be rebuilt costs that link. Not the frames' rule, and the difference is the
    banner - a missing link says nothing about redundancy, and a missing frame
    says exactly that.
    """
    recorded = snapshot.manifest.get("related_links")
    if not isinstance(recorded, list):
        return []
    return [
        RelatedLink.from_document(entry, redaction=RedactionState.APPLIED)
        for entry in recorded
        if isinstance(entry, Mapping)
    ]


def _view_warnings(snapshot: BundleSnapshot, rebuilt: list[Carrier]) -> list[WarningRecord]:
    """The **warnings** this view prints: the generation's, and then its own.

    A rebuilt carrier arrives carrying none of its own - the manifest's frame
    projection and its **related links** record none, and a **transcript**'s
    are deliberately not inherited - so everything each one holds after
    construction is what running the **redaction** policy over its text *here*
    had to say. Printing that is the other half of A-001: re-running the policy
    is what makes the view safe over a `--no-redact-secrets` bundle, and it is
    also what makes the view say something its **generation** never said, so a
    reader is told rather than left to compare two documents.

    Folded on (stage, code) like a run's own (R-41), so the two accounts of the
    same bundle read the same way.

    A recorded warning that is not a document is dropped: the render prints a
    warning field by field, so there is nothing to print, and a whole view is
    not worth one unreadable record.
    """
    recorded = snapshot.manifest.get("warnings")
    published = (
        [entry for entry in recorded if isinstance(entry, Mapping)]
        if isinstance(recorded, list)
        else []
    )
    return aggregate_warnings(
        [*published, *(item for carrier in rebuilt for item in carrier.warnings)]
    )


def _bad_manifest(message: str, details: dict[str, Any]) -> DistillError:
    """The one refusal for a document this view cannot be built from."""
    return DistillError(BAD_MANIFEST_CODE, FILTERED_VIEW_STAGE, message, details)
