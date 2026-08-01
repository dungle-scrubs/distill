"""What one run records and what it hands back.

This module owns two documents: the **manifest** content a run writes for a
**bundle**, and the response payload a caller receives. They are stated together
because they are one description of a run seen from two sides - the manifest is
what a later run reads back as a cache hit, and the response is what this run's
caller reads now. Splitting them is how a cache hit came to report a different
frame shape from the run that produced it.

It owns no I/O and no layout. Where the manifest is written, when it becomes the
**bundle marker**, and what makes a **generation** active are `bundle_store`'s;
this module produces the document and never a path to put it at. It owns no
policy either: which frames exist, whether the run was a cache hit, and what
warnings were raised are all decided by the caller and recorded here as given.

It is a signed module (ADR-0003): every field below is written into a manifest,
so editing it changes bundle content at an unchanged **bundle key**.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .artifacts import FrameArtifact, document_carries_a_reading, serialize
from .bundle_store import BundleSnapshot
from .errors import WarningRecord, total_occurrences
from .links import RelatedLink
from .options import DistillOptions
from .release import DISTILL_VERSION
from .source import SourceInfo
from .version import PIPELINE_VERSION


def manifest_document(
    source: SourceInfo,
    options: DistillOptions,
    *,
    transcript_present: bool,
    frames: list[dict[str, Any]],
    warnings: list[WarningRecord],
) -> dict[str, Any]:
    """The **manifest** content for one published **generation**.

    Everything a later reader needs to decide whether this bundle answers its
    question, and nothing about where the bundle lives: `active_generation` is
    added by the publish, because only the publish knows which generation the
    staging directory became.

    `frames` arrives as the documents `response_frames` produced, never as
    carriers. The durable projection removes only `path`, which addresses the
    producing machine; `relative_path` addresses the frame artifact inside its
    generation. A cache hit reads this portable projection from the manifest.

    The **related links** on `source` arrive the other way round - as carriers,
    because nothing before this has serialized them - and are turned into
    documents here, which is the point their **redaction** policy is checked.

    Identity is recorded as `bundle_key` (D-008). The value hashes a **source
    fingerprint** together with an **options hash**, so it names a **bundle**
    rather than a source, and `bundle_store.IDENTITY_FIELDS` already said only
    this name is written from here on - while the writer went on producing the
    legacy `source_hash` for every new manifest (finding 5-opus). Reading still
    accepts both, because a bundle published before the rename records the old
    name and nothing will rewrite it (D-017).

    Machine-local claims are excluded except at two explicit compatibility
    seams. D-006 keeps `source_resolved_path` schema-required for cache reads,
    and warning messages remain unchanged historical diagnostics even when
    they record a path. Invocation controls, frame paths, and progress-detail
    paths are not part of the durable description.
    """
    if source.provenance is None:
        raise AssertionError("current publication requires provenance")
    if source.provenance.duration_sec != source.duration_sec:
        raise AssertionError(
            "source duration and provenance duration must describe the same measurement"
        )
    document = {
        "pipeline_version": PIPELINE_VERSION,
        "distill_version": DISTILL_VERSION,
        "source_type": source.source_type,
        "bundle_key": source.source_hash,
        "source_resolved_path": str(source.resolved_path),
        "related_links": response_related_links(source.related_links),
        "duration_sec": source.duration_sec,
        "options": options.public_dict(source.source_type),
        "frame_count": len(frames),
        "transcript_present": transcript_present,
        # Events, not records. `warnings` arrives folded (R-41), so its length
        # counts distinct (stage, code) pairs - a run that timed out eighty
        # times would publish `warning_count: 2` and read as a clean run.
        "warning_count": total_occurrences(warnings),
        "frames": [
            {key: value for key, value in frame.items() if key != "path"}
            if isinstance(frame, dict)
            else frame
            for frame in frames
        ],
        "warnings": warnings,
    }
    # Provenance is deliberately outside bundle identity. A cache hit therefore
    # carries the title recorded by the generation it serves until the caller
    # asks for --force-reprocess.
    document["provenance"] = serialize(source.provenance)
    return document


def response_related_links(links: list[RelatedLink] | None) -> list[dict[str, Any]]:
    """The **related link** shape both documents carry, produced once like the frames'.

    A carrier in and a document out, and the way out is `serialize`: a
    manifest is durable and a response is what a caller reads, so both are the
    sinks R-20 is stated about. Related links were the one carrier family
    arriving here already flattened, which left nothing for either sink to
    refuse (finding 5).

    Four fields, and deliberately not six. `redaction` and `warnings` are how a
    carrier records what was done to it, not what a link is: a **manifest** is
    bundle content, and a reader asking what this bundle links to is not asking
    about Distill's bookkeeping. The run's policy is already recorded once, in
    the manifest's `options`, and the **warnings** a link raised at construction
    travel with the source's warnings - so nothing is lost by leaving them out,
    and the same four keys reach the caller as reach the manifest (finding 7).
    """
    return [
        {key: document[key] for key in ("url", "label", "source", "reason")}
        for document in (serialize(link) for link in links or ())
    ]


def response_frames(frames: list[FrameArtifact]) -> list[dict[str, Any]]:
    """The frame shape a response carries and a manifest projects from.

    Image text and the vision model's reading of the same frame stay separate
    fields: they are different claims about the frame, and merging them loses
    which one a reader is looking at.

    This is the one adapter from a **frame artifact** to the document a caller
    reads. `manifest_document` projects the same fields minus the machine-local
    `path`; a later cache hit rehydrates that response-only field from the
    portable `relative_path` under its active **generation** before calling
    `run_response`. The names overlap with the carrier's without being the
    carrier's: what a stage passes along and what a reader is served are two
    schemas, and the second one is this module's to change (D-015).
    `frame_carrier_document` is the way back, for a reader that has only the
    manifest and needs carriers again.

    A carrier in and a document out, never a document in, and the way out is
    `serialize`. A **manifest** is durable and a response is what a caller
    reads, so both are sinks R-20 is stated about: reading the fields off the
    carrier directly would produce the same text while skipping the one check
    that refuses a carrier whose **redaction** policy never ran.

    Serializing also settles the shape. The carrier's own values are frozen -
    `detected_elements` is a tuple inside it - and `serialize` thaws them into
    plain JSON, so a fresh run and a cache hit reading the same document back
    hand a caller the same types rather than a tuple one time and a list the
    other.
    """
    response = []
    for frame in frames:
        document = serialize(frame)
        item: dict[str, Any] = {
            "index": document["index"],
            "timestamp_sec": document["timestamp_sec"],
            "path": document["path"],
            "relative_path": document["relative_path"],
            "ocr_text": document["extracted_text"],
        }
        if document["interpretation"] is not None:
            item["visual_interpretation"] = document["interpretation"]
        if document.get("salience") is not None:
            # The judgment must survive publish: a filtered view produced from
            # a manifest has nothing machine-readable without it (D-006).
            item["salience"] = document["salience"]
        response.append(item)
    return response


def frame_carrier_document(document: Mapping[str, Any]) -> dict[str, Any]:
    """The **frame artifact** document one of these frames projects back to.

    The inverse of `response_frames`, and here for the reason the forward
    direction is: two schemas share names without being the same schema, and
    the translation between them is this module's whether it is read left to
    right or right to left. A reader rebuilding carriers out of a **manifest**
    that spelled `ocr_text` itself would be a third module holding an opinion
    about what a frame is called, which is exactly what M4.4 removed.

    A document in and a document out, never a carrier: `FrameArtifact.from_document`
    is what turns the result into one, and it is the only thing that may - it
    is where R-23 puts the **redaction** policy the caller is running under.

    `path` is absent from the result because it is absent from the manifest, on
    purpose: it addresses the producing machine, and what replaces it is a
    reader's own **generation** directory - which this module does not have and
    would be producing a path to have. The caller supplies it.

    **Grounding** is absent for a different reason, and it is a loss rather than
    a decision restated: `response_frames` never projected it, so there is
    nothing here to map it from and a frame rebuilt from a manifest carries
    none. A **render** built from these frames therefore prints a reading
    without the corroboration note that sits above it in the generation's own
    render. Recovering it means adding `grounding` to the projection above,
    which is **manifest** content and owes a **pipeline version** bump;
    `filtered_view` documents the gap for its readers until that is taken.
    """
    return {
        "index": document.get("index"),
        "timestamp_sec": document.get("timestamp_sec"),
        "relative_path": document.get("relative_path"),
        "extracted_text": document.get("ocr_text", ""),
        "interpretation": document.get("visual_interpretation"),
        "salience": document.get("salience"),
    }


def run_response(
    snapshot: BundleSnapshot,
    source: SourceInfo,
    frames: list[dict[str, Any]],
    transcript_present: bool,
    warnings: list[WarningRecord],
    cached: bool,
    progress: dict[str, Any] | None = None,
    job_id: str | None = None,
    artifact_path: str | None = None,
    waited_sec: float = 0.0,
    rekeyed_from: str | None = None,
) -> dict[str, Any]:
    """What a caller receives, whether the run produced the bundle or read it.

    Stated against a `BundleSnapshot` rather than a set of paths, so a cache hit
    and a fresh publish are described by the same code from the same evidence -
    a snapshot exists only where the **active generation** was proven to be on
    disk (R-04). `frames` is that same evidence for the frames: documents, which
    a fresh run got from `response_frames` and a cache hit read out of the
    **manifest** the previous run wrote, then re-addressed to the active
    **generation**.

    Two of the fields below are about how the run *got here* rather than about
    what it produced, and they are here because the alternative is DEBUG logs. A
    run that waited out a contended **bundle key** and one that walked straight
    in hand back the same bundle, and `waited_sec` is the only thing that tells
    them apart. It is the run's whole wait rather than one acquisition's,
    because a run that re-keyed queued twice and the second queue is the shorter
    one - reporting it alone would say a run that waited five minutes waited
    none. `rekeyed_from` is the sharper case: a run whose second chain walk
    moved it publishes under a key its caller never named (D-005), so without it
    the caller is handed a `source_hash` it cannot account for and nothing to
    account for it with.
    """
    # Counted on what a reading says, not on a key being present (R-39): a
    # **manifest** written before an empty response was rejected, or by a server
    # answering `{}`, carries interpretations that interpret nothing, and this
    # sentence is where a caller would be told they exist.
    visual_count = sum(
        1 for frame in frames if document_carries_a_reading(frame.get("visual_interpretation"))
    )
    summary = f"Processed {source.duration_sec:.1f}s video with {len(frames)} keyframes"
    if visual_count:
        summary += f" and visual interpretation for {visual_count} frames"
    response: dict[str, Any] = {
        "markdown_path": str(snapshot.markdown),
        "self_contained_markdown_path": str(snapshot.self_contained_markdown),
        # The deliverable, outside the cache. What an MCP caller quotes back to
        # a user and what a later turn can read without re-running anything;
        # the paths above point into a store a cleaner may reclaim.
        "artifact_path": artifact_path,
        "transcript_path": str(snapshot.transcript)
        if transcript_present and snapshot.transcript.exists()
        else None,
        "manifest_path": str(snapshot.manifest_path),
        "frames": frames,
        "duration_sec": source.duration_sec,
        "frame_count": len(frames),
        "source_hash": source.source_hash,
        # The key this run queued for and gave up, or `None` for the ordinary
        # run that kept the one it resolved to. Always present, because "did
        # this run move" is a question every response answers - answered by
        # absence, it would read as a field this build does not have.
        "rekeyed_from": rekeyed_from,
        "source_resolved_path": str(source.resolved_path),
        "cached": cached,
        "waited_sec": waited_sec,
        "pipeline_version": PIPELINE_VERSION,
        "distill_version": DISTILL_VERSION,
        "job_id": job_id,
        "summary": summary,
        "warnings": warnings,
    }
    related_links = response_related_links(source.related_links)
    if related_links:
        response["related_links"] = related_links
    if progress is not None:
        response["progress"] = progress
    return response
