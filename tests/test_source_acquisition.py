"""Acquisition of a remote source: stage, validate, promote (R-35, R-36, R-37)."""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path

import pytest
from conftest import unused_pid
from fake_tools import (
    FAKE_FFPROBE,
    FAKE_FFPROBE_NO_DURATION,
    FAKE_YTDLP_CLOBBERING,
    FAKE_YTDLP_DOWNLOAD,
    FAKE_YTDLP_LEAVES_FORMAT_FRAGMENTS,
    FAKE_YTDLP_PRODUCES_AUDIO_ONLY,
    FAKE_YTDLP_PRODUCES_EMPTY_FILE,
    FAKE_YTDLP_RECORDS_STAGING_DIR,
    FAKE_YTDLP_TRUNCATES_THEN_FAILS,
)

from distill import source as distill_source
from distill.errors import DistillError
from distill.progress import ProgressReporter
from distill.source import YoutubeDownloader, select_downloaded_media

URL = "https://youtu.be/abc123"
LOCK_KEY = "abc123"


def media_root(output_root: Path) -> Path:
    return output_root / "_youtube_sources" / LOCK_KEY


def staging_root(output_root: Path) -> Path:
    return output_root / "_youtube_staging" / LOCK_KEY


def lock_path(output_root: Path) -> Path:
    return output_root / "_youtube_locks" / f"{LOCK_KEY}.lock"


def advance_monotonic_clock(monkeypatch: pytest.MonkeyPatch, by_sec: float) -> None:
    """Make the module's clock read `by_sec` later, without any test sleeping.

    Only the reading moves; intervals measured inside a single call are
    unchanged, so a bounded wait still ends when it would have. This ages a
    lock by an hour in the eyes of anything that reads a clock to judge it.
    """

    class ShiftedClock:
        def __getattr__(self, name: str) -> object:
            return getattr(time, name)

        def monotonic(self) -> float:
            return time.monotonic() + by_sec

    monkeypatch.setattr(distill_source, "time", ShiftedClock())


def promoted_names(output_root: Path) -> list[str]:
    directory = media_root(output_root)
    return sorted(entry.name for entry in directory.iterdir()) if directory.is_dir() else []


def test_leftover_format_fragment_is_not_selected_over_the_merged_media(
    fake_tool: Callable[[str, str], Path],
    tmp_path: Path,
) -> None:
    """FAILS FIRST (finding 16, R-37): the first glob match is an audio fragment.

    An interrupted run leaves `source.f140.m4a` in the acquisition directory.
    The next run downloads `source.mp4` beside it and takes the first entry of
    `sorted(glob("source.*"))` - and `f` sorts before `m`, so the run proceeds
    against the audio fragment of a previous download instead of the video it
    just fetched.
    """
    fake_tool("yt-dlp", FAKE_YTDLP_DOWNLOAD)
    fake_tool("ffprobe", FAKE_FFPROBE)
    leftovers = media_root(tmp_path)
    leftovers.mkdir(parents=True)
    (leftovers / "source.f140.m4a").write_bytes(b"audio-fragment")

    acquired = YoutubeDownloader(tmp_path).acquire(URL, LOCK_KEY, ProgressReporter())

    assert acquired.path.name == "source.mp4"
    assert acquired.path.read_bytes() == b"video"
    acquired.lease.release()


def test_failed_download_leaves_the_previously_promoted_source_intact(
    fake_tool: Callable[[str, str], Path],
    tmp_path: Path,
) -> None:
    """FAILS FIRST (RV-3): the only good copy is destroyed before a replacement exists.

    yt-dlp opens its destination for writing before the transfer can be known to
    succeed. Pointing that destination at the promoted path means a download
    that dies mid-transfer has already truncated the media the previous run
    proved good.
    """
    fake_tool("yt-dlp", FAKE_YTDLP_DOWNLOAD)
    fake_tool("ffprobe", FAKE_FFPROBE)
    first = YoutubeDownloader(tmp_path).acquire(URL, LOCK_KEY)
    promoted = first.path
    assert promoted.read_bytes() == b"video"
    first.lease.release()

    fake_tool("yt-dlp", FAKE_YTDLP_TRUNCATES_THEN_FAILS)
    with pytest.raises(DistillError) as exc:
        YoutubeDownloader(tmp_path).acquire(URL, LOCK_KEY)

    assert exc.value.code == "E_YTDLP"
    assert promoted.exists()
    assert promoted.read_bytes() == b"video"


def test_second_run_with_different_options_cannot_disturb_media_being_read(
    fake_tool: Callable[[str, str], Path],
    tmp_path: Path,
) -> None:
    """FAILS FIRST (RV-2, R-36): the lease ends before the media is finished with.

    Two runs of the same video under different options share a **lock key** and
    not a **bundle key**. Releasing the lease when the download ends leaves the
    second run free to write over the media file the first run is still reading.
    """
    fake_tool("yt-dlp", FAKE_YTDLP_DOWNLOAD)
    fake_tool("ffprobe", FAKE_FFPROBE)
    first = YoutubeDownloader(tmp_path).acquire(URL, LOCK_KEY)
    assert first.path.read_bytes() == b"video"

    fake_tool("yt-dlp", FAKE_YTDLP_CLOBBERING)
    with pytest.raises(DistillError) as exc:
        YoutubeDownloader(tmp_path).acquire(URL, LOCK_KEY)

    assert exc.value.code == "E_LOCKED"
    assert first.path.read_bytes() == b"video"

    # Once the first run is finished reading, the source is acquirable again.
    first.lease.release()
    second = YoutubeDownloader(tmp_path).acquire(URL, LOCK_KEY)
    assert second.path.read_bytes() == b"clobbered"
    second.lease.release()


def test_a_lease_a_live_run_holds_is_not_stealable_however_old_it_is(
    fake_tool: Callable[[str, str], Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """FAILS FIRST (RV-2, R-36): age decided staleness, and every real run is old.

    The lease covers the media's whole read lifetime, and nothing refreshes the
    lock file, so any run that gets as far as transcription holds a lock older
    than a staleness window worth waiting out. A second run that read age as
    abandonment would take the lease, download, and `os.replace` its result onto
    the path the first run is still reading. Liveness is the only thing that
    says the holder has stopped reading, so the clock is moved forward by an
    hour and the lock file aged on disk to match: neither may matter.
    """
    fake_tool("yt-dlp", FAKE_YTDLP_DOWNLOAD)
    fake_tool("ffprobe", FAKE_FFPROBE)
    first = YoutubeDownloader(tmp_path).acquire(URL, LOCK_KEY)
    lock = lock_path(tmp_path)
    advance_monotonic_clock(monkeypatch, by_sec=3600.0)
    os.utime(lock, (0, 0))

    fake_tool("yt-dlp", FAKE_YTDLP_CLOBBERING)
    with pytest.raises(DistillError) as exc:
        YoutubeDownloader(tmp_path).acquire(URL, LOCK_KEY)

    assert exc.value.code == "E_LOCKED"
    assert first.path.read_bytes() == b"video"
    assert json.loads(lock.read_text())["pid"] == os.getpid()
    first.lease.release()


def test_a_lease_whose_holder_is_gone_is_reclaimed(
    fake_tool: Callable[[str, str], Path],
    tmp_path: Path,
) -> None:
    """FAILS FIRST: a lock outliving its holder blocked everyone for the window.

    A run killed mid-acquisition leaves its lock file behind, freshly written by
    any clock. Waiting it out is waiting for a process that no longer exists, so
    the lock is reclaimed the moment the holder is found to be gone. The lock
    file here is the one production wrote, with only the holder changed, so the
    test cannot pass by naming a field acquisition no longer keeps.
    """
    fake_tool("yt-dlp", FAKE_YTDLP_DOWNLOAD)
    fake_tool("ffprobe", FAKE_FFPROBE)
    abandoned = YoutubeDownloader(tmp_path).acquire(URL, LOCK_KEY)
    lock = lock_path(tmp_path)
    payload = json.loads(lock.read_text())
    lock.write_text(json.dumps({**payload, "pid": unused_pid()}))

    fake_tool("yt-dlp", FAKE_YTDLP_CLOBBERING)
    reclaimed = YoutubeDownloader(tmp_path).acquire(URL, LOCK_KEY)

    assert reclaimed.path.read_bytes() == b"clobbered"
    assert json.loads(lock.read_text())["pid"] == os.getpid()
    reclaimed.lease.release()
    abandoned.lease.release()


def test_releasing_a_lease_this_run_no_longer_owns_leaves_the_holder_alone(
    fake_tool: Callable[[str, str], Path],
    tmp_path: Path,
) -> None:
    """FAILS FIRST: release unlinked the lock file whoever had written it.

    Once a run has lost its lock - reclaimed after a stall, or replaced by hand -
    the file at that path is somebody else's lease. Releasing by unlinking
    whatever is there destroys that lease and lets a third run in behind it, so
    release checks that the lock still names this process and otherwise does
    nothing. The parent process stands in for the second run: a live pid that is
    not this one, with no child to spawn.
    """
    fake_tool("yt-dlp", FAKE_YTDLP_DOWNLOAD)
    fake_tool("ffprobe", FAKE_FFPROBE)
    first = YoutubeDownloader(tmp_path).acquire(URL, LOCK_KEY)
    lock = lock_path(tmp_path)
    other_holder = os.getppid()
    lock.write_text(json.dumps({"pid": other_holder, "created_wall": "2026-01-01T00:00:00Z"}))

    first.lease.release()

    assert lock.exists()
    assert json.loads(lock.read_text())["pid"] == other_holder
    with pytest.raises(DistillError) as exc:
        YoutubeDownloader(tmp_path).acquire(URL, LOCK_KEY)
    assert exc.value.code == "E_LOCKED"


def test_each_run_stages_its_download_in_its_own_directory(
    fake_tool: Callable[[str, str], Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """R-35: the download lands in a staging directory unique to the run.

    A directory no other run can be writing into is what makes the rest of
    acquisition safe to reason about: whatever is in it came from this download,
    so selecting and validating from it says something about this download.
    """
    fake_tool("yt-dlp", FAKE_YTDLP_RECORDS_STAGING_DIR)
    fake_tool("ffprobe", FAKE_FFPROBE)
    log = tmp_path / "staging.log"
    monkeypatch.setenv("FAKE_YTDLP_STAGING_LOG", str(log))
    abandoned = staging_root(tmp_path) / "abandoned-run"
    abandoned.mkdir(parents=True)
    (abandoned / "source.mp4").write_bytes(b"scratch from a run that died")

    first = YoutubeDownloader(tmp_path).acquire(URL, LOCK_KEY)
    first.lease.release()
    second = YoutubeDownloader(tmp_path).acquire(URL, LOCK_KEY)
    second.lease.release()

    staged = [Path(line) for line in log.read_text().splitlines()]
    assert len(set(staged)) == 2
    assert all(directory.parent == staging_root(tmp_path) for directory in staged)
    assert all(directory != media_root(tmp_path) for directory in staged)
    # Staging is scratch, not a **generation**: nothing survives the run that
    # produced it, and the directory an earlier run abandoned goes with it.
    assert not any(directory.exists() for directory in staged)
    assert not abandoned.exists()


def test_a_format_fragment_beside_the_container_is_never_selected(
    fake_tool: Callable[[str, str], Path],
    tmp_path: Path,
) -> None:
    """R-37: selection is a stated rule over the staging directory, not a glob.

    An interrupted merge leaves both format fragments and a `.part` file beside
    the container. Only the container has the stem the download was asked for.
    """
    fake_tool("yt-dlp", FAKE_YTDLP_LEAVES_FORMAT_FRAGMENTS)
    fake_tool("ffprobe", FAKE_FFPROBE)

    acquired = YoutubeDownloader(tmp_path).acquire(URL, LOCK_KEY)

    assert acquired.path.read_bytes() == b"video"
    assert promoted_names(tmp_path) == ["source.mp4"]
    acquired.lease.release()


def test_selection_prefers_a_stated_container_order_over_alphabetical(
    tmp_path: Path,
) -> None:
    """Two complete containers resolve the same way every time they are seen."""
    staging = tmp_path / "staging"
    staging.mkdir()
    for name in ("source.mp4", "source.webm", "source.mkv"):
        (staging / name).write_bytes(b"video")

    assert select_downloaded_media(staging).name == "source.mp4"


def test_selection_reports_what_the_download_left_when_nothing_qualifies(
    tmp_path: Path,
) -> None:
    """A download that produced only fragments is a failure, named as one."""
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "source.f140.m4a").write_bytes(b"audio-fragment")

    with pytest.raises(DistillError) as exc:
        select_downloaded_media(staging)

    assert exc.value.code == "E_YTDLP"
    assert exc.value.details["produced"] == ["source.f140.m4a"]


def test_an_audio_only_download_is_rejected_before_promotion(
    fake_tool: Callable[[str, str], Path],
    tmp_path: Path,
) -> None:
    """R-35: validation gates promotion, and its failure is fatal, not a degradation.

    The media file is the input every later stage reads. Promoting an audio-only
    file leaves no reduced-but-useful **bundle** to produce, so ADR-0002 makes
    this a **required capability** of the run and its absence a **fatal error**.
    """
    fake_tool("yt-dlp", FAKE_YTDLP_PRODUCES_AUDIO_ONLY)
    fake_tool("ffprobe", FAKE_FFPROBE)

    with pytest.raises(DistillError) as exc:
        YoutubeDownloader(tmp_path).acquire(URL, LOCK_KEY)

    assert exc.value.code == "E_BAD_MEDIA"
    assert exc.value.details["codec_types"] == ["audio"]
    assert promoted_names(tmp_path) == []


def test_an_empty_download_is_rejected_before_promotion(
    fake_tool: Callable[[str, str], Path],
    tmp_path: Path,
) -> None:
    """A tool that exits 0 having written nothing has still produced nothing."""
    fake_tool("yt-dlp", FAKE_YTDLP_PRODUCES_EMPTY_FILE)
    fake_tool("ffprobe", FAKE_FFPROBE)

    with pytest.raises(DistillError) as exc:
        YoutubeDownloader(tmp_path).acquire(URL, LOCK_KEY)

    assert exc.value.code == "E_BAD_MEDIA"
    assert promoted_names(tmp_path) == []


def test_promotion_moves_the_validated_file_rather_than_copying_it(
    fake_tool: Callable[[str, str], Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """R-35: promotion is one `os.replace`, so no reader sees a half-written file.

    The file that was validated and the file that is promoted share an inode,
    which is only true of a rename. A copy would produce a second inode and,
    with it, a window in which the promoted path holds part of a file.
    """
    fake_tool("yt-dlp", FAKE_YTDLP_DOWNLOAD)
    fake_tool("ffprobe", FAKE_FFPROBE)
    validated: list[int] = []
    real_validate = distill_source.validate_media_file

    def record_inode(path: Path) -> None:
        real_validate(path)
        validated.append(path.stat().st_ino)

    monkeypatch.setattr(distill_source, "validate_media_file", record_inode)
    previous = media_root(tmp_path)
    previous.mkdir(parents=True)
    (previous / "source.mp4").write_bytes(b"previous")
    previous_inode = (previous / "source.mp4").stat().st_ino

    acquired = YoutubeDownloader(tmp_path).acquire(URL, LOCK_KEY)

    assert acquired.path.stat().st_ino == validated[0]
    assert acquired.path.stat().st_ino != previous_inode
    assert acquired.path.read_bytes() == b"video"
    acquired.lease.release()


def test_a_previously_promoted_source_is_never_cleared_in_place(
    fake_tool: Callable[[str, str], Path],
    tmp_path: Path,
) -> None:
    """R-35: acquisition adds to the promoted directory; it never empties it.

    Reclaiming what an earlier promotion left behind is **prune**'s job. Doing
    it here would mean deleting a good copy at the one moment its replacement is
    least certain to arrive.
    """
    fake_tool("yt-dlp", FAKE_YTDLP_DOWNLOAD)
    fake_tool("ffprobe", FAKE_FFPROBE)
    directory = media_root(tmp_path)
    directory.mkdir(parents=True)
    (directory / "source.mkv").write_bytes(b"promoted by an earlier run")

    acquired = YoutubeDownloader(tmp_path).acquire(URL, LOCK_KEY)

    assert (directory / "source.mkv").read_bytes() == b"promoted by an earlier run"
    assert promoted_names(tmp_path) == ["source.mkv", "source.mp4"]
    acquired.lease.release()


def test_acquisition_emits_lease_validation_and_promotion_events(
    fake_tool: Callable[[str, str], Path],
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    """The four moments an operator needs to reconstruct an acquisition.

    Which run holds the source, what the produced file was judged to be, what
    became authoritative, and when the source was handed back.
    """
    fake_tool("yt-dlp", FAKE_YTDLP_DOWNLOAD)
    fake_tool("ffprobe", FAKE_FFPROBE)

    with caplog.at_level(logging.DEBUG, logger="distill.source"):
        acquired = YoutubeDownloader(tmp_path).acquire(URL, LOCK_KEY)
        with pytest.raises(DistillError):
            YoutubeDownloader(tmp_path).acquire(URL, LOCK_KEY)
        acquired.lease.release()

    events = [
        json.loads(record.message)
        for record in caplog.records
        if record.name == "distill.source"
    ]
    assert [event["event"] for event in events] == [
        "lease_acquired",
        "media_validated",
        "media_promoted",
        "lease_denied",
        "lease_released",
    ]
    assert all(event["type"] == "distill.source" for event in events)
    verdict = next(event for event in events if event["event"] == "media_validated")
    assert verdict["detail"]["verdict"] == "accepted"
    assert verdict["detail"]["codec_types"] == ["video", "audio"]
    promotion = next(event for event in events if event["event"] == "media_promoted")
    assert promotion["detail"]["path"] == str(acquired.path)
    assert promotion["detail"]["source"] != promotion["detail"]["path"]


def test_a_rejected_media_file_is_reported_with_its_verdict(
    fake_tool: Callable[[str, str], Path],
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    """A refusal to promote is as visible as a promotion, and says why."""
    fake_tool("yt-dlp", FAKE_YTDLP_PRODUCES_AUDIO_ONLY)
    fake_tool("ffprobe", FAKE_FFPROBE)

    with caplog.at_level(logging.DEBUG, logger="distill.source"), pytest.raises(DistillError):
        YoutubeDownloader(tmp_path).acquire(URL, LOCK_KEY)

    events = [
        json.loads(record.message)
        for record in caplog.records
        if record.name == "distill.source"
    ]
    assert [event["event"] for event in events] == [
        "lease_acquired",
        "media_validated",
        "lease_released",
    ]
    assert events[1]["detail"]["verdict"] == "rejected"
    assert events[1]["detail"]["reason"] == "no_video_stream"


def test_a_download_with_no_playable_duration_is_rejected_before_promotion(
    fake_tool: Callable[[str, str], Path],
    tmp_path: Path,
) -> None:
    """A container with a video stream is not yet media Distill can read.

    A transfer that stopped before the index was written still opens, and still
    reports a video stream. `probe_duration` would reject it a moment later -
    but by then it would already have replaced media that was fine.
    """
    fake_tool("yt-dlp", FAKE_YTDLP_DOWNLOAD)
    fake_tool("ffprobe", FAKE_FFPROBE_NO_DURATION)

    with pytest.raises(DistillError) as exc:
        YoutubeDownloader(tmp_path).acquire(URL, LOCK_KEY)

    assert exc.value.code == "E_BAD_MEDIA"
    assert exc.value.details["duration_sec"] == 0.0
    assert promoted_names(tmp_path) == []
