"""Serving a **bundle** that is already on disk, without the tool that made it.

R-49, finding 22. yt-dlp exists to acquire a YouTube **source** and ffprobe
exists to read a **source**'s duration. Neither is needed to hand back a
**bundle** whose **active generation** is already published, yet both were
invoked before the cache was ever consulted - so a user who processed a video
yesterday and uninstalled yt-dlp today was told the run could not proceed, for
data already on disk.

Every test here drives the real tool entry points (`process_youtube_video`,
`process_local_video`), with the tool genuinely absent from a `PATH` that holds
nothing but the fakes this test installed. The absence is proved twice: the run
either succeeds or raises `E_MISSING_TOOL`, and the `run_command` boundary event
records every tool Distill actually spawned.

What this file does not cover: which class a tool is in (`test_capabilities.py`),
what a **bundle key** is made of (`test_options_source.py`), or when a bundle is
servable at all (`test_bundle_store.py`). It covers only the *order* - cache
first, capability second - and the two things that order must not change:
identity, and the fatality of a missing **required capability** on a miss.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fake_tools import (
    FAKE_FFMPEG_WRITES_A_REAL_PNG,
    FAKE_FFPROBE,
    FAKE_YTDLP_METADATA_AND_DOWNLOAD,
)
from test_local_integration import fake_transcribe

from distill import frame_selection
from distill import pipeline as distill_session
from distill.errors import DistillError
from distill.options import DistillOptions
from distill.source import local_fingerprint, source_hash

VIDEO_ID = "cachedvideo"
"""Eleven characters, because that is the only shape the fast path reads.

yt-dlp's extractor matches exactly `[0-9A-Za-z_-]{11}`, so an id of any other
length is a value the URL fast path declines - and a fixture that used one would
be exercising the decline in every test rather than the lookup.
"""
URL = f"https://www.youtube.com/watch?v={VIDEO_ID}"


def boundary_details(caplog: pytest.LogCaptureFixture) -> list[dict[str, Any]]:
    return [
        json.loads(record.message)["detail"]
        for record in caplog.records
        if record.name == "distill.run_command"
    ]


def tools_invoked(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Every tool `run_command` actually spawned, in order (Phase 2 boundary)."""
    return [detail["tool"] for detail in boundary_details(caplog)]


def downloads_invoked(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Only the yt-dlp invocations that were asked to *acquire* something.

    yt-dlp answers metadata questions and performs downloads, and the two are
    told apart by the output template: `-o` is on the download command and on no
    other. The distinction matters wherever a run legitimately resolves metadata
    but must not fetch media - `BundleStore.begin` catches a late cache hit and
    reports `cached: True` either way, so "was it served" cannot tell whether
    the source was downloaded first.
    """
    return [
        detail["tool"]
        for detail in boundary_details(caplog)
        if detail["tool"] == "yt-dlp" and "-o" in detail["argv"]
    ]


@pytest.fixture(autouse=True)
def hermetic_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """The parts of a run that are not what these tests are about.

    Scene detection is dictated rather than detected: the fake media is five
    bytes, so a real detector finds nothing anyway, and saying so removes a
    third-party library from the timing of every test here. Transcription is the
    standard double. What is left - source resolution, the cache lookup, the
    publish - is the real thing.
    """
    monkeypatch.setattr(frame_selection, "scene_midpoint_candidates", lambda *_: [])
    monkeypatch.setattr(distill_session, "transcribe_with_imports", fake_transcribe)


def a_producing_path(fake_tool: Callable[[str, str], Path], *, youtube: bool) -> Path:
    """Install the tools a run needs to *produce* a **generation**, and return
    the one whose absence each test is about.

    ffmpeg is installed in every case: it is a **required capability** of
    producing, and these tests are about the tools a *serving* run should not
    need, not about ffmpeg.
    """
    fake_tool("ffmpeg", FAKE_FFMPEG_WRITES_A_REAL_PNG)
    probe = fake_tool("ffprobe", FAKE_FFPROBE)
    if not youtube:
        return probe
    return fake_tool("yt-dlp", FAKE_YTDLP_METADATA_AND_DOWNLOAD)


def youtube_args(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    return {
        "url": URL,
        "output_dir": str(tmp_path / "cache"),
        "ocr": False,
        "redact_secrets": False,
        "caption_frames": False,
        **overrides,
    }


def local_args(video: Path, tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    return {
        "path": str(video),
        "output_dir": str(tmp_path / "cache"),
        "ocr": False,
        "redact_secrets": False,
        "caption_frames": False,
        **overrides,
    }


def a_local_video(tmp_path: Path) -> Path:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    return video


# --- YouTube: the cache before the metadata resolution -----------------------


def test_a_cached_youtube_bundle_is_served_with_yt_dlp_absent(
    fake_tool: Callable[[str, str], Path],
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    """FAILS FIRST (finding 22, R-49): the cache hit demanded the download tool.

    The first run publishes a **bundle** with yt-dlp installed. The user then
    uninstalls it, which is the whole scenario: the second run needs no source
    acquired, because the **active generation** for that **bundle key** is on
    disk. Resolution asked yt-dlp for the video id before it asked the cache
    anything, so the run died at `E_MISSING_TOOL` over a bundle it was about to
    hand back unread.
    """
    ytdlp = a_producing_path(fake_tool, youtube=True)

    first = distill_session.process_youtube_video(youtube_args(tmp_path))
    assert first["cached"] is False

    ytdlp.unlink()
    with caplog.at_level(logging.DEBUG, logger="distill.run_command"):
        second = distill_session.process_youtube_video(youtube_args(tmp_path))

    assert second["cached"] is True
    assert second["source_hash"] == first["source_hash"]
    assert Path(second["markdown_path"]).exists()
    assert "yt-dlp" not in tools_invoked(caplog)


def test_the_cache_is_consulted_before_youtube_metadata_resolution(
    fake_tool: Callable[[str, str], Path],
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    """A served cache hit spawns nothing at all - not yt-dlp, not ffprobe.

    Stronger than "yt-dlp was absent and the run survived": with both tools
    installed and working, a run that consults the cache first has no reason to
    spawn either of them, so the boundary records an empty list. A run that
    resolved metadata first would record `yt-dlp` here even though it went on to
    serve from cache.
    """
    a_producing_path(fake_tool, youtube=True)

    distill_session.process_youtube_video(youtube_args(tmp_path))
    with caplog.at_level(logging.DEBUG, logger="distill.run_command"):
        second = distill_session.process_youtube_video(youtube_args(tmp_path))

    assert second["cached"] is True
    assert tools_invoked(caplog) == []


def test_a_youtube_bundle_keyed_by_the_resolved_video_id_is_still_found(
    fake_tool: Callable[[str, str], Path],  # noqa: ARG001 - installs an empty PATH
    tmp_path: Path,
) -> None:
    """Identity does not drift: the **source fingerprint** is the one it always was.

    The fixture bundle is written at the key the old resolution computed -
    `sha256(video id)` for the fingerprint - and nothing installs yt-dlp, so the
    only way this bundle is found is by deriving that same fingerprint from the
    URL rather than from the tool. A reordering that quietly re-keyed the cache
    would orphan every bundle a user already has, which is a worse outcome than
    the one being fixed.

    What this deliberately does *not* pin is the **options hash**: the fixture
    computes the current one, so a `PIPELINE_VERSION` bump moves the fixture
    with the code. That is D-015's decision and not this reorder's - raising the
    pipeline version is *how* stale output stops being served, and a test that
    froze it would be pinning a mechanism against its own purpose.
    """
    options = DistillOptions.from_args(youtube_args(tmp_path, cache_mode="fingerprint"))
    fingerprint = hashlib.sha256(VIDEO_ID.encode()).hexdigest()
    bundle_key = source_hash(fingerprint, options.opts_hash("youtube"))
    write_published_bundle(tmp_path / "cache", bundle_key, source_type="youtube")

    served = distill_session.process_youtube_video(youtube_args(tmp_path))

    assert served["cached"] is True
    assert served["source_hash"] == bundle_key
    assert served["duration_sec"] == 12.5


def test_a_bundle_keyed_by_a_resolved_id_the_url_does_not_carry_is_still_found(
    fake_tool: Callable[[str, str], Path],
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    """The second lookup is not decoration: it is the old behavior, preserved.

    Here yt-dlp reports an id the URL does not contain, which is the one case
    where the URL's id and the resolved id disagree. The bundle is keyed by the
    id yt-dlp reports, exactly as an earlier Distill would have published it, so
    the first lookup misses and only the resolved-id lookup can find it. Delete
    that fallback and this URL re-downloads a video it already has.
    """
    resolved_id = "resolvedvid"
    a_producing_path(fake_tool, youtube=False)
    fake_tool(
        "yt-dlp",
        FAKE_YTDLP_METADATA_AND_DOWNLOAD.replace(
            'argv[-1].rsplit("=", 1)[-1].rsplit("/", 1)[-1]', repr(resolved_id)
        ),
    )
    options = DistillOptions.from_args(youtube_args(tmp_path, cache_mode="fingerprint"))
    bundle_key = source_hash(
        hashlib.sha256(resolved_id.encode()).hexdigest(), options.opts_hash("youtube")
    )
    write_published_bundle(tmp_path / "cache", bundle_key, source_type="youtube")

    with caplog.at_level(logging.DEBUG, logger="distill.run_command"):
        served = distill_session.process_youtube_video(youtube_args(tmp_path))

    assert served["cached"] is True
    assert served["source_hash"] == bundle_key
    # The resolution found it. Without the fallback the run would still report
    # `cached: True` - `begin` re-asks under the run lock - having downloaded
    # the video first, which is the whole cost R-49 is about.
    assert downloads_invoked(caplog) == []


def test_a_playlist_attached_url_is_not_served_from_the_video_ids_bundle(
    fake_tool: Callable[[str, str], Path],  # noqa: ARG001 - installs an empty PATH
    tmp_path: Path,
) -> None:
    """The boundary of the reorder, stated as a refusal.

    `watch?v=A&list=P` is resolved by yt-dlp as the playlist, so a run of it
    publishes under some entry's id and not under `A`. Reading the **bundle
    key** off the URL would hand back `A`'s bundle for a URL this Distill does
    not produce `A`'s bundle from - a cache hit for something the same command
    cannot make. So the fast path declines, the run resolves the id the way it
    always did, and yt-dlp's absence is fatal for this URL form.
    """
    playlist_url = f"{URL}&list=PLQHpFq3RA7fEJ0z3DABwTPvwre0Vu6OBH"
    args = youtube_args(tmp_path, url=playlist_url, cache_mode="fingerprint")
    options = DistillOptions.from_args(args)
    bundle_key = source_hash(
        hashlib.sha256(VIDEO_ID.encode()).hexdigest(), options.opts_hash("youtube")
    )
    write_published_bundle(tmp_path / "cache", bundle_key, source_type="youtube")

    with pytest.raises(DistillError) as failure:
        distill_session.process_youtube_video(youtube_args(tmp_path, url=playlist_url))

    assert failure.value.code == "E_MISSING_TOOL"
    assert failure.value.details["tool"] == "yt-dlp"


def test_a_youtube_manifest_duration_over_the_cap_is_refused_rather_than_served(
    fake_tool: Callable[[str, str], Path],  # noqa: ARG001 - installs an empty PATH
    tmp_path: Path,
) -> None:
    """The same refusal on the path the cache reorder actually added.

    The local hit re-applies the cap in a branch of its own; the YouTube hit
    read the manifest's number and returned. So a bundle claiming 99999s was
    handed back under `--max-duration-sec 5`, and the same claim on the same
    run's local path was `E_DURATION_CAP`: one operator policy, two answers,
    decided by which kind of **source** the bundle happened to be for.
    """
    args = youtube_args(tmp_path, max_duration_sec=5.0)
    options = DistillOptions.from_args(args)
    bundle_key = source_hash(
        hashlib.sha256(VIDEO_ID.encode()).hexdigest(), options.opts_hash("youtube")
    )
    write_published_bundle(
        tmp_path / "cache", bundle_key, source_type="youtube", duration_sec=99999.0
    )

    with pytest.raises(DistillError) as failure:
        distill_session.process_youtube_video(dict(args))

    assert failure.value.code == "E_DURATION_CAP"
    assert failure.value.stage == "source"
    assert failure.value.details["duration_sec"] == 99999.0


def test_a_local_manifest_duration_over_the_cap_is_refused_rather_than_served(
    fake_tool: Callable[[str, str], Path],  # noqa: ARG001 - installs an empty PATH
    tmp_path: Path,
) -> None:
    """A manifest's duration is input, not a fact, and the cap still applies.

    `max_duration_sec` is in the **options hash**, so a bundle under this
    **bundle key** was published by a run that accepted its duration - but the
    manifest is a document another process wrote, and skipping the probe must
    not become trusting whatever number is in it. The published duration is
    over this run's cap, so the run is refused for the reason it would have been
    refused had the probe reported the same number.
    """
    video = a_local_video(tmp_path)
    args = local_args(video, tmp_path, max_duration_sec=5.0)
    options = DistillOptions.from_args(args)
    bundle_key = source_hash(
        local_fingerprint(video.resolve(), options.cache_mode), options.opts_hash("local")
    )
    write_published_bundle(tmp_path / "cache", bundle_key, source_type="local")

    with pytest.raises(DistillError) as failure:
        distill_session.process_local_video(dict(args))

    assert failure.value.code == "E_DURATION_CAP"
    assert failure.value.details["duration_sec"] == 12.5


def test_a_url_naming_two_video_ids_is_not_served_from_the_first_ones_bundle(
    fake_tool: Callable[[str, str], Path],  # noqa: ARG001 - installs an empty PATH
    tmp_path: Path,
) -> None:
    """The playlist refusal, generalized: the fast path reads an id or declines.

    `watch?v=A&v=B` carries two ids and yt-dlp picks one of them; `parse_qs`
    picks the first, which is a different question with the same shape of
    answer. Reading `A` off it served `A`'s **bundle** for a URL this Distill
    resolves to `B` - a cache hit for a bundle the same command cannot make,
    which is exactly what `youtube_url_names_one_video` exists to prevent for
    the playlist form.

    So the fast path declines, the run resolves the id the way it always did,
    and with yt-dlp uninstalled that is fatal for this URL form rather than
    quietly answered from the wrong bundle.
    """
    two_ids = f"{URL}&v=otherid1234"
    args = youtube_args(tmp_path, url=two_ids)
    options = DistillOptions.from_args(args)
    bundle_key = source_hash(
        hashlib.sha256(VIDEO_ID.encode()).hexdigest(), options.opts_hash("youtube")
    )
    write_published_bundle(tmp_path / "cache", bundle_key, source_type="youtube")

    with pytest.raises(DistillError) as failure:
        distill_session.process_youtube_video(dict(args))

    assert failure.value.code == "E_MISSING_TOOL"
    assert failure.value.details["tool"] == "yt-dlp"


def test_a_video_id_the_url_padded_is_not_a_bundle_key_of_its_own(
    fake_tool: Callable[[str, str], Path],
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    """A twelve-character `v=` value keys nothing, so it must not be looked up.

    yt-dlp matches eleven characters and ignores the rest, so this URL is
    published under `VIDEO_ID` - the bundle written here. The fast path can only
    have keyed the lookup on the twelve-character value, which nothing has ever
    written, so it declines and the resolution finds the bundle under the id
    yt-dlp reports. The **source** is still never downloaded, which is what the
    boundary shows.
    """
    a_producing_path(fake_tool, youtube=False)
    # The real extractor matches eleven characters and ignores the twelfth, so
    # the fake reports what yt-dlp would rather than echoing the URL back.
    fake_tool(
        "yt-dlp",
        FAKE_YTDLP_METADATA_AND_DOWNLOAD.replace(
            'argv[-1].rsplit("=", 1)[-1].rsplit("/", 1)[-1]', repr(VIDEO_ID)
        ),
    )
    padded = f"https://www.youtube.com/watch?v={VIDEO_ID}x"
    args = youtube_args(tmp_path, url=padded)
    options = DistillOptions.from_args(args)
    bundle_key = source_hash(
        hashlib.sha256(VIDEO_ID.encode()).hexdigest(), options.opts_hash("youtube")
    )
    write_published_bundle(tmp_path / "cache", bundle_key, source_type="youtube")

    with caplog.at_level(logging.DEBUG, logger="distill.run_command"):
        served = distill_session.process_youtube_video(dict(args))

    assert served["cached"] is True
    assert served["source_hash"] == bundle_key
    assert downloads_invoked(caplog) == []


def test_a_manifest_recording_a_boolean_duration_is_a_miss_not_a_one_second_hit(
    fake_tool: Callable[[str, str], Path],  # noqa: ARG001 - installs an empty PATH
    tmp_path: Path,
) -> None:
    """What `manifest_duration` refuses, the resolution treats as no bundle at all.

    `True` is an `int` in Python, so a manifest saying `"duration_sec": true`
    would otherwise describe a one-second video - and every window, interval
    and percentage the run derives would be computed from that one second. The
    guard is a line in a function no other test reaches with this shape, so
    this is the end of it a reader can see: with the acquisition tool absent, a
    hit is a served **bundle** and a miss is `E_MISSING_TOOL`.
    """
    args = youtube_args(tmp_path)
    options = DistillOptions.from_args(args)
    bundle_key = source_hash(
        hashlib.sha256(VIDEO_ID.encode()).hexdigest(), options.opts_hash("youtube")
    )
    bundle = write_published_bundle(tmp_path / "cache", bundle_key, source_type="youtube")
    manifest_path = bundle / "_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["duration_sec"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    with pytest.raises(DistillError) as failure:
        distill_session.process_youtube_video(dict(args))

    assert failure.value.code == "E_MISSING_TOOL"
    assert failure.value.details["tool"] == "yt-dlp"


def test_a_youtube_cache_miss_with_yt_dlp_absent_is_still_fatal(
    fake_tool: Callable[[str, str], Path],
    tmp_path: Path,
) -> None:
    """ADR-0002 is not weakened: nothing on disk means the tool is still required.

    The reorder changes when yt-dlp is asked for, never whether its absence is
    survivable. A run with no **bundle** to serve has to acquire a **source**,
    and yt-dlp is a **required capability**, so the absence is a **fatal error**
    naming the tool rather than a **degradation**.
    """
    fake_tool("ffmpeg", FAKE_FFMPEG_WRITES_A_REAL_PNG)
    fake_tool("ffprobe", FAKE_FFPROBE)

    with pytest.raises(DistillError) as failure:
        distill_session.process_youtube_video(youtube_args(tmp_path))

    assert failure.value.code == "E_MISSING_TOOL"
    assert failure.value.stage == "youtube"
    assert failure.value.details["tool"] == "yt-dlp"


def test_force_reprocess_still_acquires_the_youtube_source(
    fake_tool: Callable[[str, str], Path],
    tmp_path: Path,
) -> None:
    """The one option that means "ignore what is on disk" still needs the tool.

    `force_reprocess` is a request to produce the **generation** again, so there
    is no cache hit to serve and yt-dlp is required exactly as it was.
    """
    ytdlp = a_producing_path(fake_tool, youtube=True)
    distill_session.process_youtube_video(youtube_args(tmp_path))
    ytdlp.unlink()

    with pytest.raises(DistillError) as failure:
        distill_session.process_youtube_video(youtube_args(tmp_path, force_reprocess=True))

    assert failure.value.code == "E_MISSING_TOOL"
    assert failure.value.details["tool"] == "yt-dlp"


# --- Local: the fingerprint is computable without probing ---------------------


def test_a_cached_local_bundle_is_served_with_ffprobe_absent(
    fake_tool: Callable[[str, str], Path],
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    """FAILS FIRST (finding 22, R-49): the cache hit demanded the duration probe.

    A local **source fingerprint** is the file's size, mtime and sampled bytes -
    or, in `content` mode, all of them. None of that is anything ffprobe knows.
    The only thing the probe supplied was the duration, and a servable
    **bundle** records the duration its **manifest** was published with, so the
    probe had nothing left to answer. Resolution probed first regardless, so an
    uninstalled ffprobe cost a bundle that was already on disk.
    """
    ffprobe = a_producing_path(fake_tool, youtube=False)
    video = a_local_video(tmp_path)

    first = distill_session.process_local_video(local_args(video, tmp_path))
    assert first["cached"] is False

    ffprobe.unlink()
    with caplog.at_level(logging.DEBUG, logger="distill.run_command"):
        second = distill_session.process_local_video(local_args(video, tmp_path))

    assert second["cached"] is True
    assert second["source_hash"] == first["source_hash"]
    assert second["duration_sec"] == first["duration_sec"]
    assert tools_invoked(caplog) == []


@pytest.mark.parametrize("cache_mode", ["fingerprint", "content"])
def test_neither_local_cache_mode_needs_a_probe_to_find_its_bundle(
    fake_tool: Callable[[str, str], Path],
    tmp_path: Path,
    cache_mode: str,
) -> None:
    """"Where possible" is every option combination: the enumeration is empty.

    `local_fingerprint` reads the file and nothing else in both modes -
    `fingerprint` hashes size, mtime and sampled anchors, `content` hashes every
    byte - and the **options hash** is the options plus the **pipeline
    version**. So no local **bundle key** depends on probed metadata, and there
    is no option combination for which serving a local cache hit requires
    ffprobe. What still requires it is *producing* a **generation**, covered by
    the two tests below.
    """
    ffprobe = a_producing_path(fake_tool, youtube=False)
    video = a_local_video(tmp_path)
    args = local_args(video, tmp_path, cache_mode=cache_mode)

    first = distill_session.process_local_video(dict(args))
    assert first["cached"] is False

    ffprobe.unlink()
    second = distill_session.process_local_video(dict(args))

    assert second["cached"] is True
    assert second["source_hash"] == first["source_hash"]


def test_a_local_cache_miss_with_ffprobe_absent_is_still_fatal(
    fake_tool: Callable[[str, str], Path],
    tmp_path: Path,
) -> None:
    """ADR-0002 is not weakened for the local path either.

    Nothing servable is on disk, so the duration has to be read, and ffprobe is
    a **required capability**: its absence ends the run naming the tool.
    """
    fake_tool("ffmpeg", FAKE_FFMPEG_WRITES_A_REAL_PNG)
    video = a_local_video(tmp_path)

    with pytest.raises(DistillError) as failure:
        distill_session.process_local_video(local_args(video, tmp_path))

    assert failure.value.code == "E_MISSING_TOOL"
    assert failure.value.stage == "source"
    assert failure.value.details["tool"] == "ffprobe"


def test_force_reprocess_still_probes_the_local_source(
    fake_tool: Callable[[str, str], Path],
    tmp_path: Path,
) -> None:
    """A forced run produces a **generation**, and producing one needs the duration."""
    ffprobe = a_producing_path(fake_tool, youtube=False)
    video = a_local_video(tmp_path)
    distill_session.process_local_video(local_args(video, tmp_path))
    ffprobe.unlink()

    with pytest.raises(DistillError) as failure:
        distill_session.process_local_video(
            local_args(video, tmp_path, force_reprocess=True)
        )

    assert failure.value.code == "E_MISSING_TOOL"
    assert failure.value.details["tool"] == "ffprobe"


def test_a_changed_local_file_is_a_miss_and_so_still_needs_the_probe(
    fake_tool: Callable[[str, str], Path],
    tmp_path: Path,
) -> None:
    """Skipping the probe must not weaken what a **cache hit** means.

    The fingerprint is still the identity: rewriting the file gives it a new
    **source fingerprint**, so the published bundle no longer answers for it and
    the run has to produce one - which needs the duration, which needs ffprobe.
    A reorder that served the old bundle for new content would be a far worse
    defect than the one being fixed.
    """
    ffprobe = a_producing_path(fake_tool, youtube=False)
    video = a_local_video(tmp_path)
    first = distill_session.process_local_video(local_args(video, tmp_path))
    video.write_bytes(b"different video entirely")
    ffprobe.unlink()

    with pytest.raises(DistillError) as failure:
        distill_session.process_local_video(local_args(video, tmp_path))

    assert failure.value.code == "E_MISSING_TOOL"
    assert failure.value.details["tool"] == "ffprobe"
    assert first["cached"] is False


def write_published_bundle(
    root: Path,
    bundle_key: str,
    *,
    source_type: str,
    duration_sec: float = 12.5,
) -> Path:
    """A **bundle** on disk as an earlier Distill published it.

    Hand-written rather than produced, because the point of the fixture is the
    **bundle key** a *previous* version computed: a bundle produced by this
    version's code would be found by this version's code whatever the key
    became.
    """
    bundle = root / bundle_key
    (bundle / "g1").mkdir(parents=True)
    (bundle / "g1" / "video.md").write_text("# cached render\n")
    (bundle / "g1" / "transcript.json").write_text(json.dumps({"segments": []}))
    (bundle / "_manifest.json").write_text(
        json.dumps(
            {
                "pipeline_version": 1,
                "distill_version": "0.1.0",
                "source_type": source_type,
                "bundle_key": bundle_key,
                "source_resolved_path": str(root / "source.mp4"),
                "duration_sec": duration_sec,
                "options": {},
                "frame_count": 0,
                "transcript_present": True,
                "warning_count": 0,
                "frames": [],
                "warnings": [],
                "active_generation": "g1",
            },
            indent=2,
        )
        + "\n"
    )
    return bundle
