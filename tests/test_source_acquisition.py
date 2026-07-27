"""Acquisition of a remote source: stage, validate, promote (R-35, R-36, R-37)."""

from __future__ import annotations

import errno
import json
import logging
import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
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
    fake_ffprobe_flooding_stderr,
)

from distill import source as distill_source
from distill.errors import DistillError
from distill.progress import ProgressReporter
from distill.run_command import OUTPUT_CAP_BYTES, TRUNCATION_WARNING_CODE
from distill.source import AcquisitionLease, YoutubeDownloader, select_downloaded_media

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


CONTENDER = '''
"""One run contending for the acquisition lease of a single **lock key**.

It meets the other contender at the moment the lock file has just been opened
and nobody has yet been established as its holder - the window a scheme that
publishes the lock and then names its owner cannot close - and it holds
whatever it won until the other contender has had its turn.
"""

import json
import os
import sys
import time
from pathlib import Path

from distill import source as distill_source
from distill.errors import DistillError

url, lock_key, output_root, rendezvous_dir, token, parties = sys.argv[1:]
rendezvous = Path(rendezvous_dir)
parties = int(parties)


def meet(name):
    """Block until every contender has reached `name`, or give up on them."""
    (rendezvous / f"{name}-{token}").write_text("")
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if len(list(rendezvous.glob(f"{name}-*"))) >= parties:
            return
        time.sleep(0.005)


class RendezvousOs:
    """`os`, with the first successful open of a lock file made a meeting point."""

    def __init__(self, inner):
        self._inner = inner
        self.met = False

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def open(self, path, flags, *args, **kwargs):
        fd = self._inner.open(path, flags, *args, **kwargs)
        if not self.met and str(path).endswith(".lock"):
            self.met = True
            meet("window")
        return fd


rendezvous_os = RendezvousOs(os)
distill_source.os = rendezvous_os
try:
    acquired = distill_source.YoutubeDownloader(Path(output_root)).acquire(url, lock_key)
    verdict = {"acquired": True, "media": str(acquired.path)}
except DistillError as exc:
    verdict = {"acquired": False, "code": exc.code}
verdict["met_in_window"] = rendezvous_os.met
# Whatever was won is held until the other contender has finished trying: a
# lease given up early would let the loser win a lock that was never free.
meet("done")
print(json.dumps(verdict))
'''


def race_for_the_lease(tmp_path: Path, parties: int = 2) -> list[dict[str, Any]]:
    """Run `parties` real processes into acquisition and collect their verdicts."""
    script = tmp_path / "contender.py"
    script.write_text(CONTENDER)
    rendezvous = tmp_path / "rendezvous"
    rendezvous.mkdir()
    contenders = [
        subprocess.Popen(
            [
                sys.executable,
                str(script),
                URL,
                LOCK_KEY,
                str(tmp_path),
                str(rendezvous),
                str(token),
                str(parties),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for token in range(parties)
    ]
    verdicts: list[dict[str, Any]] = []
    for contender in contenders:
        out, err = contender.communicate(timeout=60)
        assert contender.returncode == 0, err
        verdicts.append(json.loads(out))
    return verdicts


def test_two_processes_racing_for_the_lease_leave_exactly_one_holder(
    fake_tool: Callable[[str, str], Path],
    tmp_path: Path,
) -> None:
    """FAILS FIRST (RV-2, R-07): an unfinished lock file reads as an abandoned one.

    A lock whose *content* names its holder is published empty and named a
    moment later. A contender that reads it inside that window finds no holder,
    concludes nobody will ever release it, unlinks it and creates its own - and
    both runs then believe they hold the lease of a **lock key** that admits
    one. The two contenders meet inside exactly that window, so the interleaving
    is not left to timing, and exactly one of them may come out holding media.
    """
    fake_tool("yt-dlp", FAKE_YTDLP_DOWNLOAD)
    fake_tool("ffprobe", FAKE_FFPROBE)

    verdicts = race_for_the_lease(tmp_path)

    # Both contenders reaching the meeting point is what makes this a race and
    # not two runs that happened to take turns.
    assert [verdict["met_in_window"] for verdict in verdicts] == [True, True]
    assert [verdict["acquired"] for verdict in verdicts].count(True) == 1
    assert [verdict["code"] for verdict in verdicts if not verdict["acquired"]] == ["E_LOCKED"]


def test_a_filesystem_that_cannot_grant_the_lock_stops_the_run(
    fake_tool: Callable[[str, str], Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A lock the kernel refuses to grant is a fatal error, not a weaker lock.

    `ENOLCK` is what a filesystem with no lock manager answers - some network
    mounts, and the reason R-09 exists. Distill cannot tell one run from two
    there, so it says so and stops: continuing would mean going back to a scheme
    where two runs can hold one **lock key**, which is what this replaced.
    """
    fake_tool("yt-dlp", FAKE_YTDLP_DOWNLOAD)
    fake_tool("ffprobe", FAKE_FFPROBE)

    def refuse(_fd: int, _operation: int) -> None:
        raise OSError(errno.ENOLCK, "no locks available")

    monkeypatch.setattr(distill_source.fcntl, "flock", refuse)

    with pytest.raises(DistillError) as exc:
        YoutubeDownloader(tmp_path, lock_wait_sec=5.0).acquire(URL, LOCK_KEY)

    assert exc.value.code == "E_LOCK_UNSUPPORTED"
    assert exc.value.details["errno"] == "ENOLCK"
    assert promoted_names(tmp_path) == []


def test_a_lease_a_live_run_holds_is_not_stealable_however_old_it_is(
    fake_tool: Callable[[str, str], Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """FAILS FIRST (RV-2, R-36): age decided staleness, and every real run is old.

    The lease covers the media's whole read lifetime, so any run that gets as
    far as transcription has held its lock for longer than a staleness window
    worth waiting out. A second run that read age as abandonment would take the
    lease, download, and `os.replace` its result onto the path the first run is
    still reading. The clock is moved forward by an hour and the lock file aged
    on disk to match: a held lease is held, and neither may matter.
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
    first.lease.release()


HOLDER = '''
"""A run that acquires the acquisition lease and then only holds it."""

import sys
import time
from pathlib import Path

from distill import source as distill_source

url, lock_key, output_root = sys.argv[1:]
acquired = distill_source.YoutubeDownloader(Path(output_root)).acquire(url, lock_key)
print(acquired.path, flush=True)
# Nothing here ever releases: giving the lease up is what killing this process
# has to accomplish on its own.
while True:
    time.sleep(3600)
'''


def start_a_holding_run(tmp_path: Path) -> subprocess.Popen[str]:
    """Start a real second run, returning once it holds the lease."""
    script = tmp_path / "holder.py"
    script.write_text(HOLDER)
    holder = subprocess.Popen(
        [sys.executable, str(script), URL, LOCK_KEY, str(tmp_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert holder.stdout is not None
    assert holder.stdout.readline().strip().endswith("source.mp4")
    return holder


def test_a_lease_whose_holder_was_killed_is_reacquirable_at_once(
    fake_tool: Callable[[str, str], Path],
    tmp_path: Path,
) -> None:
    """FAILS FIRST: a pid that still resolves is not a holder that still holds.

    The holder is `SIGKILL`ed and deliberately not reaped, so it is a process
    that has released nothing, cannot release anything, and whose pid the kernel
    still answers for - which is also what a holder whose pid was handed to some
    unrelated process looks like. Asking whether that pid exists says the lease
    is held; asking the kernel for the lock says it is free, because the kernel
    dropped it when the descriptor died with its owner.
    """
    fake_tool("yt-dlp", FAKE_YTDLP_DOWNLOAD)
    fake_tool("ffprobe", FAKE_FFPROBE)
    holder = start_a_holding_run(tmp_path)
    holder.kill()
    # Wait for the death without collecting it: WNOWAIT leaves the pid in the
    # table, so this is a dead holder rather than a dying one, with no sleep.
    os.waitid(os.P_PID, holder.pid, os.WEXITED | os.WNOWAIT)

    fake_tool("yt-dlp", FAKE_YTDLP_CLOBBERING)
    reclaimed = YoutubeDownloader(tmp_path).acquire(URL, LOCK_KEY)

    assert reclaimed.path.read_bytes() == b"clobbered"
    assert reclaimed.warnings == []
    reclaimed.lease.release()
    assert holder.wait() != 0


def test_releasing_a_lease_this_process_does_not_hold_frees_nobody(
    fake_tool: Callable[[str, str], Path],
    tmp_path: Path,
) -> None:
    """FAILS FIRST: release acted on the lock file rather than on what it holds.

    A run that releases can only give up the exclusion it was granted. Here a
    second process holds the lease and this one holds nothing but a descriptor
    on the same path, with the lock file's content hand-written to name this
    process - so any release that consults the path, or the file, or what the
    file claims, frees a holder that never asked to be freed and lets a third
    run in behind it.
    """
    fake_tool("yt-dlp", FAKE_YTDLP_DOWNLOAD)
    fake_tool("ffprobe", FAKE_FFPROBE)
    holder = start_a_holding_run(tmp_path)
    lock = lock_path(tmp_path)
    lock.write_text(json.dumps({"pid": os.getpid()}))
    stray = AcquisitionLease(
        lock_key=LOCK_KEY,
        lock_path=lock,
        fd=os.open(lock, os.O_CREAT | os.O_RDWR, 0o600),
    )

    stray.release()

    fake_tool("yt-dlp", FAKE_YTDLP_CLOBBERING)
    with pytest.raises(DistillError) as exc:
        YoutubeDownloader(tmp_path).acquire(URL, LOCK_KEY)
    assert exc.value.code == "E_LOCKED"
    assert promoted_names(tmp_path) == ["source.mp4"]
    holder.kill()
    holder.wait()


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

    def record_inode(path: Path) -> list[dict[str, str]]:
        probe_warnings = real_validate(path)
        validated.append(path.stat().st_ino)
        return probe_warnings

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


def test_a_truncated_validation_probe_travels_with_the_acquired_source(
    fake_tool: Callable[[str, str], Path],
    tmp_path: Path,
) -> None:
    """R-33: validation accepting the media is not the same as losing nothing.

    The verdict and the loss are separate answers. `validate_media_file` is the
    only place the probe's **warning** exists, so a validator that returned
    nothing would promote the media and drop the record of the truncation.
    """
    fake_tool("yt-dlp", FAKE_YTDLP_DOWNLOAD)
    fake_tool("ffprobe", fake_ffprobe_flooding_stderr(OUTPUT_CAP_BYTES))

    acquired = YoutubeDownloader(tmp_path).acquire(URL, LOCK_KEY)
    try:
        assert acquired.path.exists()
        assert [
            item["code"]
            for item in acquired.warnings
            if item["code"] == TRUNCATION_WARNING_CODE
        ] == [TRUNCATION_WARNING_CODE]
    finally:
        acquired.lease.release()
