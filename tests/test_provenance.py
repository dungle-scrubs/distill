"""Provenance capture, redaction, persistence, and cache identity."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest
from fake_tools import FAKE_FFPROBE

from distill.artifacts import (
    Carrier,
    Provenance,
    RedactionPolicyNotApplied,
    RedactionState,
    serialize,
)
from distill.bundle_store import BundleRun, BundleStore, validate_manifest_schema
from distill.errors import DistillError
from distill.options import DistillOptions
from distill.progress import ProgressReporter
from distill.response import manifest_document
from distill.source import (
    AcquiredSource,
    AcquisitionLease,
    LocalSourceProvider,
    SourceInfo,
    SourceRequest,
    SourceResolver,
    YouTubeDownloaderProtocol,
    YouTubeMetadata,
    YouTubeSourceProvider,
    _first_description_paragraph,
    source_hash,
    youtube_metadata,
)

FAKE_YTDLP_WITH_PROVENANCE = """
import json

print(json.dumps({
    "id": "abcdefghijk",
    "title": "Release notes",
    "channel": "Distill Channel",
    "uploader": "Fallback Uploader",
    "description": "First paragraph.\\n\\nSecond paragraph.",
    "upload_date": "20260729",
}))
"""

FAKE_YTDLP_WITH_UPLOADER_ONLY = """
import json

print(json.dumps({
    "id": "abcdefghijk",
    "title": "Release notes",
    "uploader": "Fallback Uploader",
    "description": "First paragraph.",
    "upload_date": "20260729",
}))
"""

FAKE_YTDLP_METADATA_FAILURE = """
raise SystemExit(1)
"""


def test_provenance_is_a_carrier_that_classifies_fields_by_origin() -> None:
    field_names = {field.name for field in dataclasses.fields(Provenance)}

    assert issubclass(Provenance, Carrier)
    assert Provenance.EXTRACTED_TEXT_FIELDS == (
        "title",
        "channel",
        "description",
        "upload_date",
    )
    assert {"canonical_url", "duration_sec", "processed_at"} <= field_names
    assert {"pipeline_version", "distill_version"}.isdisjoint(field_names)


def test_provenance_requires_a_caller_supplied_rfc3339_utc_processing_time() -> None:
    fixed = "2026-07-29T14:20:00Z"

    document = serialize(
        Provenance(title="demo.mp4", duration_sec=12.5, processed_at=fixed)
    )

    assert document["processed_at"] == fixed
    with pytest.raises(ValueError, match="RFC 3339 UTC"):
        Provenance(
            title="demo.mp4",
            duration_sec=12.5,
            processed_at="2026-07-29 14:20:00",
        )


def test_provenance_redacts_a_credential_shaped_title_at_construction() -> None:
    secret = "sk-live-0123456789abcdefghij"

    provenance = Provenance(
        title=f"Deploy with {secret}",
        duration_sec=12.5,
        processed_at="2026-07-29T14:20:00Z",
    )

    assert provenance.title is not None
    assert secret not in provenance.title
    assert provenance.title == "Deploy with [REDACTED:api-key]"


def test_serializing_unredacted_provenance_is_refused() -> None:
    provenance = Provenance(
        title="demo.mp4",
        duration_sec=12.5,
        processed_at="2026-07-29T14:20:00Z",
    )
    object.__setattr__(provenance, "redaction", RedactionState.NOT_APPLIED)

    with pytest.raises(RedactionPolicyNotApplied) as raised:
        serialize(provenance)

    assert raised.value.code == "E_REDACTION_POLICY_NOT_APPLIED"


def test_youtube_metadata_captures_source_chosen_provenance_fields(
    fake_tool: Callable[[str, str], Path],
) -> None:
    fake_tool("yt-dlp", FAKE_YTDLP_WITH_PROVENANCE)

    metadata = youtube_metadata("https://youtu.be/abcdefghijk")

    assert metadata.title == "Release notes"
    assert metadata.channel == "Distill Channel"
    assert metadata.upload_date == "20260729"


def test_youtube_metadata_uses_uploader_when_channel_is_absent(
    fake_tool: Callable[[str, str], Path],
) -> None:
    fake_tool("yt-dlp", FAKE_YTDLP_WITH_UPLOADER_ONLY)

    metadata = youtube_metadata("https://youtu.be/abcdefghijk")

    assert metadata.channel == "Fallback Uploader"


def test_youtube_metadata_uses_uploader_when_channel_is_whitespace(
    fake_tool: Callable[[str, str], Path],
) -> None:
    fake_tool(
        "yt-dlp",
        """
import json

print(json.dumps({
    "id": "abcdefghijk",
    "channel": "   ",
    "uploader": "  Fallback Uploader  ",
}))
""",
    )

    metadata = youtube_metadata("https://youtu.be/abcdefghijk")

    assert metadata.channel == "Fallback Uploader"


def test_youtube_metadata_preserves_the_full_description_for_existing_callers(
    fake_tool: Callable[[str, str], Path],
) -> None:
    fake_tool("yt-dlp", FAKE_YTDLP_WITH_PROVENANCE)

    metadata = youtube_metadata("https://youtu.be/abcdefghijk")

    assert metadata.description == "First paragraph.\n\nSecond paragraph."


@pytest.mark.parametrize("separator", ["\r", "\u2029"])
def test_first_description_paragraph_stops_at_non_lf_paragraph_separators(
    separator: str,
) -> None:
    assert _first_description_paragraph(f"first{separator}second") == "first"


def test_local_source_provenance_uses_only_the_original_basename_and_measured_facts(
    fake_tool: Callable[[str, str], Path],
    tmp_path: Path,
) -> None:
    fake_tool("ffprobe", FAKE_FFPROBE)
    target = tmp_path / "stored-video.mp4"
    target.write_bytes(b"video")
    requested = tmp_path / "original-name.mov"
    requested.symlink_to(target)
    fixed = "2026-07-29T14:20:00Z"

    source = LocalSourceProvider().resolve(
        SourceRequest(
            str(requested),
            DistillOptions(),
            processed_at=fixed,
        )
    )

    assert source.provenance is not None
    assert serialize(source.provenance) == {
        "title": "original-name.mov",
        "duration_sec": 12.5,
        "processed_at": fixed,
    }


def test_youtube_source_combines_metadata_with_measured_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video")
    lease = AcquisitionLease.take("abcdefghijk", tmp_path / "youtube.lock")
    assert lease is not None

    class FakeDownloader:
        def acquire(
            self,
            url: str,
            lock_key: str,
            progress: ProgressReporter | None = None,
        ) -> AcquiredSource:
            _ = (url, lock_key, progress)
            return AcquiredSource(path=video, lease=lease)

    monkeypatch.setattr("distill.source.check_disk_floor", lambda _path: None)
    monkeypatch.setattr("distill.source.probe_duration", lambda _path: (12.5, []))
    fixed = "2026-07-29T14:20:00Z"

    source = YouTubeSourceProvider().resolve(
        SourceRequest(
            "https://youtu.be/abcdefghijk",
            DistillOptions(),
            processed_at=fixed,
            output_root=tmp_path,
        ),
        downloader=FakeDownloader(),
        metadata=YouTubeMetadata(
            video_id="zyxwvutsrqp",
            description="First paragraph.\n\nSecond paragraph.",
            warnings=[],
            title="Release notes",
            channel="Distill Channel",
            upload_date="20260729",
        ),
    )

    try:
        assert source.provenance is not None
        assert serialize(source.provenance) == {
            "title": "Release notes",
            "channel": "Distill Channel",
            "description": "First paragraph.",
            "upload_date": "20260729",
            "canonical_url": "https://www.youtube.com/watch?v=zyxwvutsrqp",
            "duration_sec": 12.5,
            "processed_at": fixed,
        }
    finally:
        lease.release()


def test_a_tool_returned_secret_shaped_id_never_reaches_the_manifest(
    fake_tool: Callable[[str, str], Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    secret = "sk-live-0123456789abcdefghij"
    fake_tool(
        "yt-dlp",
        f"""
import json

print(json.dumps({{
    "id": "{secret}",
    "title": "Release notes",
}}))
""",
    )
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video")
    lease = AcquisitionLease.take("abcdefghijk", tmp_path / "youtube.lock")
    assert lease is not None

    class FakeDownloader:
        def acquire(
            self,
            url: str,
            lock_key: str,
            progress: ProgressReporter | None = None,
        ) -> AcquiredSource:
            _ = (url, lock_key, progress)
            return AcquiredSource(path=video, lease=lease)

    monkeypatch.setattr("distill.source.check_disk_floor", lambda _path: None)
    monkeypatch.setattr("distill.source.probe_duration", lambda _path: (12.5, []))

    source = YouTubeSourceProvider().resolve(
        SourceRequest(
            "https://youtu.be/abcdefghijk",
            DistillOptions(),
            processed_at="2026-07-29T14:20:00Z",
            output_root=tmp_path,
        ),
        downloader=FakeDownloader(),
    )

    try:
        manifest = manifest_document(
            source,
            DistillOptions(),
            transcript_present=False,
            frames=[],
            warnings=source.warnings,
        )
        assert secret not in json.dumps(manifest)
        assert manifest["provenance"]["canonical_url"] == (
            "https://www.youtube.com/watch?v=abcdefghijk"
        )
    finally:
        lease.release()


def test_youtube_metadata_exception_warns_and_keeps_processing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video")
    lease = AcquisitionLease.take("abcdefghijk", tmp_path / "youtube.lock")
    assert lease is not None

    class FakeDownloader:
        def acquire(
            self,
            url: str,
            lock_key: str,
            progress: ProgressReporter | None = None,
        ) -> AcquiredSource:
            _ = (url, lock_key, progress)
            return AcquiredSource(path=video, lease=lease)

    metadata_calls: list[object] = []

    def fail_metadata(*_args: object, **_kwargs: object) -> object:
        metadata_calls.append(_args)
        raise DistillError("E_YTDLP", "youtube", "metadata lookup failed")

    # Patched where the call resolves: `youtube_metadata` invokes `_run_ytdlp`
    # from its own module, so a patch on the `distill.source` re-export would be
    # silently ineffective and the assertion below would pass off a real yt-dlp
    # run as the injected failure.
    monkeypatch.setattr("distill.youtube._run_ytdlp", fail_metadata)
    monkeypatch.setattr("distill.source.check_disk_floor", lambda _path: None)
    monkeypatch.setattr("distill.source.probe_duration", lambda _path: (12.5, []))
    fixed = "2026-07-29T14:20:00Z"

    source = YouTubeSourceProvider().resolve(
        SourceRequest(
            "https://youtu.be/abcdefghijk",
            DistillOptions(),
            processed_at=fixed,
            output_root=tmp_path,
        ),
        downloader=FakeDownloader(),
    )

    try:
        assert metadata_calls, "the injected metadata failure never fired"
        assert source.provenance is not None
        assert serialize(source.provenance) == {
            "canonical_url": "https://www.youtube.com/watch?v=abcdefghijk",
            "duration_sec": 12.5,
            "processed_at": fixed,
        }
        assert [item["code"] for item in source.warnings] == ["metadata_unavailable"]
        manifest = manifest_document(
            source,
            DistillOptions(),
            transcript_present=False,
            frames=[],
            warnings=source.warnings,
        )
        assert [item["code"] for item in manifest["warnings"]] == [
            "metadata_unavailable"
        ]
    finally:
        lease.release()


def test_nonzero_youtube_metadata_lookup_yields_partial_metadata(
    fake_tool: Callable[[str, str], Path],
) -> None:
    fake_tool("yt-dlp", FAKE_YTDLP_METADATA_FAILURE)

    metadata = youtube_metadata("https://youtu.be/abcdefghijk")

    assert metadata.video_id == "abcdefghijk"
    assert metadata.title is None
    assert [item["code"] for item in metadata.warnings] == [
        "metadata_unavailable"
    ]


def test_non_object_youtube_metadata_json_yields_partial_metadata(
    fake_tool: Callable[[str, str], Path],
) -> None:
    fake_tool(
        "yt-dlp",
        """
import json

print(json.dumps("scalar-json"))
""",
    )

    metadata = youtube_metadata("https://youtu.be/abcdefghijk")

    assert metadata.video_id == "abcdefghijk"
    assert metadata.title is None
    assert [item["code"] for item in metadata.warnings] == [
        "metadata_unavailable"
    ]


def test_youtube_resolution_reuses_one_request_for_cache_and_acquisition(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seen: list[SourceRequest] = []

    class RecordingProvider(YouTubeSourceProvider):
        def cached_for_video_id(
            self,
            request: SourceRequest,
            video_id: str,
        ) -> SourceInfo | None:
            _ = video_id
            seen.append(request)
            return None

        def resolve(
            self,
            request: SourceRequest,
            downloader: YouTubeDownloaderProtocol | None = None,
            metadata: YouTubeMetadata | None = None,
        ) -> SourceInfo:
            _ = (downloader, metadata)
            seen.append(request)
            return SourceInfo(
                source_type="youtube",
                resolved_path=tmp_path / "source.mp4",
                duration_sec=12.5,
                source_fingerprint="fingerprint",
                source_hash="bundle-key",
                warnings=[],
            )

    monkeypatch.setattr(
        "distill.source.youtube_metadata",
        lambda _url: YouTubeMetadata("abcdefghijk", "", []),
    )
    resolver = SourceResolver(youtube=RecordingProvider())

    resolver.resolve(
        "youtube",
        "https://youtu.be/abcdefghijk",
        DistillOptions(output_dir=str(tmp_path)),
    )

    assert len(seen) == 2
    assert seen[0] is seen[1]


def test_provenance_survives_when_a_cache_hit_is_removed_before_begin(
    tmp_path: Path,
) -> None:
    options = DistillOptions()
    video_id = "abcdefghijk"
    fingerprint = hashlib.sha256(video_id.encode()).hexdigest()
    bundle_key = source_hash(fingerprint, options.opts_hash("youtube"))
    original = SourceInfo(
        source_type="youtube",
        resolved_path=tmp_path / "source.mp4",
        duration_sec=12.5,
        source_fingerprint=fingerprint,
        source_hash=bundle_key,
        warnings=[],
        provenance=Provenance(
            title="Release notes",
            channel="Distill Channel",
            canonical_url=f"https://www.youtube.com/watch?v={video_id}",
            duration_sec=12.5,
            processed_at="2026-07-29T14:20:00Z",
        ),
    )
    store = BundleStore.open(tmp_path)
    first_run = store.begin(bundle_key)
    assert isinstance(first_run, BundleRun)
    first_run.write_render("# Video\n")
    first_run.write_self_contained_render("# Video\n")
    first = first_run.commit(
        manifest_document(
            original,
            options,
            transcript_present=False,
            frames=[],
            warnings=[],
        )
    )

    cached = YouTubeSourceProvider().cached_for_video_id(
        SourceRequest(
            f"https://youtu.be/{video_id}",
            options,
            output_root=tmp_path,
        ),
        video_id,
    )
    assert cached is not None
    shutil.rmtree(first.root)
    replacement_run = store.begin(bundle_key)
    assert isinstance(replacement_run, BundleRun)
    replacement_run.write_render("# Video\n")
    replacement_run.write_self_contained_render("# Video\n")
    replacement = replacement_run.commit(
        manifest_document(
            cached,
            options,
            transcript_present=False,
            frames=[],
            warnings=[],
        )
    )

    assert replacement.manifest["provenance"] == first.manifest["provenance"]


def test_manifest_serializes_redacted_provenance_without_the_raw_secret(
    tmp_path: Path,
) -> None:
    secret = "sk-live-0123456789abcdefghij"
    source = SourceInfo(
        source_type="youtube",
        resolved_path=Path("/tmp/source.mp4"),
        duration_sec=12.5,
        source_fingerprint="fingerprint",
        source_hash="bundle-key",
        warnings=[],
        provenance=Provenance(
            title=f"Deploy with {secret}",
            channel="Distill Channel",
            description="First paragraph.",
            upload_date="20260729",
            canonical_url="https://www.youtube.com/watch?v=abcdefghijk",
            duration_sec=12.5,
            processed_at="2026-07-29T14:20:00Z",
        ),
    )

    manifest = manifest_document(
        source,
        DistillOptions(),
        transcript_present=False,
        frames=[],
        warnings=[],
    )

    assert manifest["provenance"]["title"] == "Deploy with [REDACTED:api-key]"
    assert secret not in json.dumps(manifest)
    run = BundleStore.open(tmp_path).begin("bundle-key")
    assert isinstance(run, BundleRun)
    run.write_render("# Video\n")
    run.write_self_contained_render("# Video\n")
    snapshot = run.commit(manifest)
    assert snapshot.manifest["provenance"] == manifest["provenance"]
    assert secret not in snapshot.manifest_path.read_text()


def test_manifest_refuses_a_provenance_carrier_whose_policy_never_ran() -> None:
    provenance = Provenance(
        title="demo.mp4",
        duration_sec=12.5,
        processed_at="2026-07-29T14:20:00Z",
    )
    object.__setattr__(provenance, "redaction", RedactionState.NOT_APPLIED)
    source = SourceInfo(
        source_type="local",
        resolved_path=Path("/tmp/demo.mp4"),
        duration_sec=12.5,
        source_fingerprint="fingerprint",
        source_hash="bundle-key",
        warnings=[],
        provenance=provenance,
    )

    with pytest.raises(RedactionPolicyNotApplied):
        manifest_document(
            source,
            DistillOptions(),
            transcript_present=False,
            frames=[],
            warnings=[],
        )


def test_manifest_schema_keeps_provenance_optional_but_validates_it_when_present() -> None:
    source = SourceInfo(
        source_type="local",
        resolved_path=Path("/tmp/demo.mp4"),
        duration_sec=12.5,
        source_fingerprint="fingerprint",
        source_hash="bundle-key",
        warnings=[],
        provenance=Provenance(
            title="demo.mp4",
            duration_sec=12.5,
            processed_at="2026-07-29T14:20:00Z",
        ),
    )
    current = manifest_document(
        source,
        DistillOptions(),
        transcript_present=False,
        frames=[],
        warnings=[],
    )
    legacy = {key: value for key, value in current.items() if key != "provenance"}
    validate_manifest_schema(legacy, require_active_generation=False)

    invalid_provenance_documents = [
        {
            "title": ["not", "text"],
            "duration_sec": "wrong",
            "processed_at": "not-a-time",
        },
        {
            "duration_sec": float("nan"),
            "processed_at": "2026-07-29T14:20:00Z",
        },
        {
            "duration_sec": 12.5,
            "processed_at": "2026-07-29T14:20:00+00:00",
        },
        {
            "channel": 42,
            "duration_sec": 12.5,
            "processed_at": "2026-07-29T14:20:00Z",
        },
        {
            "canonical_url": "https://example.com/watch?v=abcdefghijk",
            "duration_sec": 12.5,
            "processed_at": "2026-07-29T14:20:00Z",
        },
        {
            "duration_sec": 12.5,
            "processed_at": "2026-07-29T14:20:00Z",
            "unknown": "field",
        },
        "attacker-chosen text",
    ]
    for provenance in invalid_provenance_documents:
        with pytest.raises(DistillError) as raised:
            validate_manifest_schema(
                {**legacy, "provenance": provenance},
                require_active_generation=False,
            )

        assert raised.value.code == "E_BAD_MANIFEST"
        assert raised.value.details["field"].startswith("provenance")


def test_manifest_refuses_contradictory_source_and_provenance_durations() -> None:
    source = SourceInfo(
        source_type="local",
        resolved_path=Path("/tmp/demo.mp4"),
        duration_sec=12.5,
        source_fingerprint="fingerprint",
        source_hash="bundle-key",
        warnings=[],
        provenance=Provenance(
            title="demo.mp4",
            duration_sec=99.0,
            processed_at="2026-07-29T14:20:00Z",
        ),
    )

    with pytest.raises(AssertionError, match="provenance duration"):
        manifest_document(
            source,
            DistillOptions(),
            transcript_present=False,
            frames=[],
            warnings=[],
        )


def test_retitle_does_not_change_options_hash_or_bundle_key() -> None:
    first_options = DistillOptions()
    second_options = DistillOptions()
    first = Provenance(
        title="Original title",
        duration_sec=12.5,
        processed_at="2026-07-29T14:20:00Z",
    )
    second = Provenance(
        title="Retitled later",
        duration_sec=12.5,
        processed_at="2026-07-29T14:20:00Z",
    )
    object.__setattr__(first_options, "provenance", serialize(first))
    object.__setattr__(second_options, "provenance", serialize(second))

    first_hash = first_options.opts_hash("youtube")
    second_hash = second_options.opts_hash("youtube")

    assert "provenance" not in first_options.cache_payload("youtube")
    assert first_hash == second_hash
    assert source_hash("source-fingerprint", first_hash) == source_hash(
        "source-fingerprint", second_hash
    )
