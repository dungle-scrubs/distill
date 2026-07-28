"""Acquisition of a remote source: stage, validate, promote (R-35, R-36, R-37)."""

from __future__ import annotations

import errno
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from ast import literal_eval
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fake_tools import (
    FAKE_FFPROBE,
    FAKE_FFPROBE_NAN_DURATION,
    FAKE_FFPROBE_NO_DURATION,
    FAKE_YTDLP_CLOBBERING,
    FAKE_YTDLP_DOWNLOAD,
    FAKE_YTDLP_LEAVES_FORMAT_FRAGMENTS,
    FAKE_YTDLP_PRODUCES_A_SYMLINK,
    FAKE_YTDLP_PRODUCES_AUDIO_ONLY,
    FAKE_YTDLP_PRODUCES_EMPTY_FILE,
    FAKE_YTDLP_RECORDS_STAGING_DIR,
    FAKE_YTDLP_TRUNCATES_THEN_FAILS,
    fake_ffprobe_flooding_stderr,
)

from distill import bundle_store as distill_bundle_store
from distill import source as distill_source
from distill.bundle_store import ExclusiveLock
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
        YoutubeDownloader(tmp_path, lock_wait_sec=0.0).acquire(URL, LOCK_KEY)

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

from distill import bundle_store, source as distill_source
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


# The **acquisition lease** takes its lock through `bundle_store.ExclusiveLock`,
# so that is where a lock file is opened and where the window has to be set.
rendezvous_os = RendezvousOs(os)
bundle_store.os = rendezvous_os
try:
    acquired = distill_source.YoutubeDownloader(
        Path(output_root), lock_wait_sec=0.0
    ).acquire(url, lock_key)
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

    monkeypatch.setattr(distill_bundle_store.fcntl, "flock", refuse)

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
        YoutubeDownloader(tmp_path, lock_wait_sec=0.0).acquire(URL, LOCK_KEY)

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
        lock=ExclusiveLock(
            subject=LOCK_KEY,
            path=lock,
            fd=os.open(lock, os.O_CREAT | os.O_RDWR, 0o600),
        ),
    )

    stray.release()

    fake_tool("yt-dlp", FAKE_YTDLP_CLOBBERING)
    with pytest.raises(DistillError) as exc:
        YoutubeDownloader(tmp_path, lock_wait_sec=0.0).acquire(URL, LOCK_KEY)
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
            YoutubeDownloader(tmp_path, lock_wait_sec=0.0).acquire(URL, LOCK_KEY)
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


# --- Acquisition writes and deletions stay inside the output root (R-16) -----
#
# The **acquisition lease** proves who is running, not what a path is. Every
# directory acquisition touches - staging, media, locks - is derived by joining
# a name onto the output root, and a symlink pre-created at any of those names
# redirects the operation that follows it out of the tree. Staging is the worst
# of the three, because what follows is a recursive delete.


def user_data_outside(root: Path) -> Path:
    """A directory of the user's own files that acquisition has no claim on."""
    victim = root / "user-data" / "photos"
    victim.mkdir(parents=True)
    (victim / "holiday.jpg").write_bytes(b"irreplaceable")
    return victim


def test_staging_cleanup_cannot_delete_through_a_symlinked_staging_directory(
    fake_tool: Callable[[str, str], Path],
    tmp_path: Path,
) -> None:
    """FAILS FIRST (finding 2-codex, R-16): a recursive delete outside the root.

    `_new_staging_dir` created its parent with `mkdir(exist_ok=True)`, which is
    satisfied by a symlink to a directory, then iterated the parent and
    recursively removed every child directory it found. Pointed at a directory
    of the user's files, that walk is a `rmtree` of their subdirectories.
    """
    fake_tool("yt-dlp", FAKE_YTDLP_DOWNLOAD)
    fake_tool("ffprobe", FAKE_FFPROBE)
    output_root = tmp_path / "out"
    output_root.mkdir()
    victim = user_data_outside(tmp_path)
    staging_root(output_root).parent.mkdir(parents=True)
    staging_root(output_root).symlink_to(victim.parent, target_is_directory=True)

    with pytest.raises(DistillError) as exc:
        YoutubeDownloader(output_root).acquire(URL, LOCK_KEY)

    assert exc.value.code == "E_BAD_OUTPUT_DIR"
    # The link names the *parent*, so what the walk would have removed is every
    # directory inside it - `photos` and everything under it.
    container = victim.parent
    survivors = sorted(str(entry.relative_to(container)) for entry in container.rglob("*"))
    assert survivors == ["photos", "photos/holiday.jpg"]


@pytest.mark.parametrize("derived", ["_youtube_staging", "_youtube_sources"])
def test_acquisition_creates_nothing_through_a_symlinked_directory(
    fake_tool: Callable[[str, str], Path],
    tmp_path: Path,
    derived: str,
) -> None:
    """FAILS FIRST (finding 2-codex, R-16): the check has to precede the mkdir.

    Both of these paths are two components deep, and `mkdir(parents=True)`
    builds every missing one before anything reads the path - so a link at the
    first component is already followed by the time a check on the second fires.
    Refusing afterwards still leaves a directory in the user's tree.
    """
    fake_tool("yt-dlp", FAKE_YTDLP_DOWNLOAD)
    fake_tool("ffprobe", FAKE_FFPROBE)
    output_root = tmp_path / "out"
    output_root.mkdir()
    victim = user_data_outside(tmp_path)
    output_root.joinpath(derived).symlink_to(victim, target_is_directory=True)

    with pytest.raises(DistillError) as exc:
        YoutubeDownloader(output_root).acquire(URL, LOCK_KEY)

    assert exc.value.code == "E_BAD_OUTPUT_DIR"
    assert sorted(entry.name for entry in victim.iterdir()) == ["holiday.jpg"]


def test_a_symlink_planted_inside_staging_is_refused_rather_than_walked(
    fake_tool: Callable[[str, str], Path],
    tmp_path: Path,
) -> None:
    """FAILS FIRST (finding 2-codex, R-16): the deletion target itself is a link.

    Validating the staging parent once is not enough. Each stale entry is its
    own deletion target, and a link planted beside the real ones names a
    directory the run never staged into.
    """
    fake_tool("yt-dlp", FAKE_YTDLP_DOWNLOAD)
    fake_tool("ffprobe", FAKE_FFPROBE)
    output_root = tmp_path / "out"
    output_root.mkdir()
    victim = user_data_outside(tmp_path)
    staging_root(output_root).mkdir(parents=True)
    (staging_root(output_root) / "abandoned-run").symlink_to(
        victim, target_is_directory=True
    )

    with pytest.raises(DistillError) as exc:
        YoutubeDownloader(output_root).acquire(URL, LOCK_KEY)

    assert exc.value.code == "E_BAD_OUTPUT_DIR"
    assert sorted(entry.name for entry in victim.iterdir()) == ["holiday.jpg"]


def test_promotion_cannot_write_through_a_symlinked_media_directory(
    fake_tool: Callable[[str, str], Path],
    tmp_path: Path,
) -> None:
    """FAILS FIRST (finding 2-codex, R-16): `os.replace` onto a redirected path.

    `promote_media` created its directory with `mkdir(parents=True,
    exist_ok=True)`, which a symlink to a directory satisfies, and then renamed
    the staged media onto `<media_dir>/source.mp4`. A user file of that name
    under the link's target is replaced outright.
    """
    fake_tool("yt-dlp", FAKE_YTDLP_DOWNLOAD)
    fake_tool("ffprobe", FAKE_FFPROBE)
    output_root = tmp_path / "out"
    output_root.mkdir()
    victim = user_data_outside(tmp_path)
    (victim / "source.mp4").write_bytes(b"the user's only copy")
    media_root(output_root).parent.mkdir(parents=True)
    media_root(output_root).symlink_to(victim, target_is_directory=True)

    with pytest.raises(DistillError) as exc:
        YoutubeDownloader(output_root).acquire(URL, LOCK_KEY)

    assert exc.value.code == "E_BAD_OUTPUT_DIR"
    assert (victim / "source.mp4").read_bytes() == b"the user's only copy"


def test_the_lease_is_not_taken_through_a_symlinked_lock_directory(
    fake_tool: Callable[[str, str], Path],
    tmp_path: Path,
) -> None:
    """FAILS FIRST (finding 2-codex, R-16): the lock file lands outside the root.

    A lease taken at a path outside the output root excludes nothing that
    matters and creates a file where Distill was never asked to write. The lock
    directory is derived the same way staging and media are, so it fails the
    same way.
    """
    fake_tool("yt-dlp", FAKE_YTDLP_DOWNLOAD)
    fake_tool("ffprobe", FAKE_FFPROBE)
    output_root = tmp_path / "out"
    output_root.mkdir()
    victim = user_data_outside(tmp_path)
    output_root.joinpath("_youtube_locks").symlink_to(victim, target_is_directory=True)

    with pytest.raises(DistillError) as exc:
        YoutubeDownloader(output_root).acquire(URL, LOCK_KEY)

    assert exc.value.code == "E_BAD_OUTPUT_DIR"
    assert sorted(entry.name for entry in victim.iterdir()) == ["holiday.jpg"]


def test_a_download_that_produces_a_link_is_never_selected(
    fake_tool: Callable[[str, str], Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """R-37, R-16: what a download leaves behind is as untrusted as its stdout.

    yt-dlp writes into the staging directory, so staging afterwards holds
    whatever a tool Distill does not control put there. A link is not a
    completed container, and selecting one would promote it onto the media path
    for every later read - the probe, the frames, the transcript - to follow
    wherever it points.
    """
    fake_tool("yt-dlp", FAKE_YTDLP_PRODUCES_A_SYMLINK)
    fake_tool("ffprobe", FAKE_FFPROBE)
    output_root = tmp_path / "out"
    output_root.mkdir()
    victim = user_data_outside(tmp_path) / "holiday.jpg"
    monkeypatch.setenv("FAKE_YTDLP_LINK_TARGET", str(victim))

    with pytest.raises(DistillError) as exc:
        YoutubeDownloader(output_root).acquire(URL, LOCK_KEY)

    assert exc.value.code == "E_YTDLP"
    assert exc.value.details["produced"] == ["source.mp4"]
    assert promoted_names(output_root) == []
    assert victim.read_bytes() == b"irreplaceable"


def test_staging_substituted_mid_run_does_not_promote_the_users_file(
    fake_tool: Callable[[str, str], Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """FAILS FIRST (finding 2-codex, R-16): validating once is validating too early.

    A rename has two ends, and staging was checked when it was created rather
    than when it is read from. Substituting it between those two moments makes
    the *source* of the promotion a file of the user's, which `os.replace` then
    moves out of their directory - the file is not copied away, it is gone from
    where they left it. Validation therefore belongs immediately before the
    operation, which is the only placement this substitution cannot get past.
    """
    fake_tool("yt-dlp", FAKE_YTDLP_DOWNLOAD)
    fake_tool("ffprobe", FAKE_FFPROBE)
    output_root = tmp_path / "out"
    output_root.mkdir()
    victim = user_data_outside(tmp_path)
    (victim / "source.mp4").write_bytes(b"the user's only copy")

    def substitute_staging(produced: Path) -> list[dict[str, str]]:
        # Stands in for the window between the check and the use: validation is
        # the last thing to run before promotion, so the swap lands there.
        staging = produced.parent
        shutil.rmtree(staging)
        staging.symlink_to(victim, target_is_directory=True)
        return []

    monkeypatch.setattr(distill_source, "validate_media_file", substitute_staging)

    with pytest.raises(DistillError) as exc:
        YoutubeDownloader(output_root).acquire(URL, LOCK_KEY)

    assert exc.value.code == "E_BAD_OUTPUT_DIR"
    assert sorted(entry.name for entry in victim.iterdir()) == ["holiday.jpg", "source.mp4"]
    assert (victim / "source.mp4").read_bytes() == b"the user's only copy"
    assert promoted_names(output_root) == []


def test_discarding_staging_never_deletes_through_a_substituted_path(
    fake_tool: Callable[[str, str], Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """FAILS FIRST (finding 2-codex, R-16): the cleanup is a deletion too.

    Discarding staging is the last thing a run does with the directory and the
    furthest point from where it was created, so it is where a check made once
    at the start has aged the most. What runs there is a recursive delete, which
    is the operation with the least to gain from being trusting.
    """
    fake_tool("yt-dlp", FAKE_YTDLP_DOWNLOAD)
    fake_tool("ffprobe", FAKE_FFPROBE)
    output_root = tmp_path / "out"
    output_root.mkdir()
    victim = user_data_outside(tmp_path).parent
    real_promote = distill_source.promote_media

    def promote_then_substitute(produced: Path, media_dir: Path, *, root: Path) -> Path:
        # Promotion is the last step before the discard, so the swap lands here.
        # The link goes one level above the staging directory: `rmtree` refuses a
        # link handed to it directly, and follows one in a component every time.
        promoted = real_promote(produced, media_dir, root=root)
        staging = produced.parent
        (victim / staging.name).mkdir()
        (victim / staging.name / "tax-return.pdf").write_bytes(b"irreplaceable")
        shutil.rmtree(staging.parent)
        staging.parent.symlink_to(victim, target_is_directory=True)
        return promoted

    monkeypatch.setattr(distill_source, "promote_media", promote_then_substitute)

    with pytest.raises(DistillError) as exc:
        YoutubeDownloader(output_root).acquire(URL, LOCK_KEY)

    assert exc.value.code == "E_BAD_OUTPUT_DIR"
    survivors = sorted(str(entry.relative_to(victim)) for entry in victim.rglob("*"))
    assert "photos/holiday.jpg" in survivors
    assert [name for name in survivors if name.endswith("tax-return.pdf")] != []


def test_a_link_pre_created_at_the_promoted_media_path_is_refused(
    fake_tool: Callable[[str, str], Path],
    tmp_path: Path,
) -> None:
    """FAILS FIRST (finding 2-codex, R-16): the destination is checked too.

    The media directory being sound when it was created is not the media path
    being sound when the rename runs - that is a check separated from its use.
    Re-walking the destination immediately before `os.replace` is what closes
    the gap, and it is the only thing that sees a link at the leaf.
    """
    fake_tool("yt-dlp", FAKE_YTDLP_DOWNLOAD)
    fake_tool("ffprobe", FAKE_FFPROBE)
    output_root = tmp_path / "out"
    output_root.mkdir()
    victim = user_data_outside(tmp_path) / "holiday.jpg"
    media_root(output_root).mkdir(parents=True)
    (media_root(output_root) / "source.mp4").symlink_to(victim)

    with pytest.raises(DistillError) as exc:
        YoutubeDownloader(output_root).acquire(URL, LOCK_KEY)

    assert exc.value.code == "E_BAD_OUTPUT_DIR"
    assert victim.read_bytes() == b"irreplaceable"


# --- The caller's wait budget reaches the lease (finding 4-opus, D-044) -----
#
# Two runs of *the same video* share a **lock key**, which is the case D-044's
# 300 s budget was written for: the likely holder is a run that is nearly done,
# and waiting for it costs less than failing. The budget is spent here, at the
# **acquisition lease**, because acquisition is what a second run of one video
# reaches first - long before any **bundle key** is contended.


class BudgetClock:
    """A clock that moves only when the waiter polls, so a budget is exact.

    A wait budget is a promise about how long a run waits, not about how long
    the suite takes: `sleep` advances the reading and returns at once, and
    `on_sleep` is where a test lets the world change - a holder giving the lease
    up between two polls, which is the whole reason a run waits at all.
    """

    def __init__(self, on_sleep: Callable[[int], None] | None = None) -> None:
        self.now = 0.0
        self.sleeps = 0
        self._on_sleep = on_sleep

    def __getattr__(self, name: str) -> object:
        return getattr(time, name)

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps += 1
        self.now += seconds
        if self._on_sleep is not None:
            self._on_sleep(self.sleeps)


def youtube_options(root: Path) -> Any:
    from distill.options import DistillOptions

    return DistillOptions.from_args(
        {
            "url": URL,
            "output_dir": str(root),
            "ocr": False,
            "redact_secrets": False,
            "caption_frames": False,
            "cache_mode": "fingerprint",
        }
    )


def hold_the_lease(root: Path, lock_key: str) -> AcquisitionLease:
    """Take the **acquisition lease** the way another run of the video holds it."""
    from distill.source import LOCK_DIR_NAME

    lock = root / LOCK_DIR_NAME / f"{lock_key}.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lease = AcquisitionLease.take(lock_key, lock)
    assert lease is not None
    return lease


def resolving_youtube(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    lock_wait_sec: float,
) -> Any:
    """Resolve a YouTube source the way a tool call does, metadata faked."""
    from distill.source import YouTubeMetadata

    monkeypatch.setattr(
        distill_source,
        "youtube_metadata",
        lambda _url: YouTubeMetadata("abc123", "", []),
    )
    return distill_source.resolve_source_for_processing(
        "youtube",
        URL,
        youtube_options(root),
        lock_wait_sec=lock_wait_sec,
    )


def test_a_single_source_run_waits_for_the_lock_key_another_run_holds(
    fake_tool: Callable[[str, str], Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """FAILS FIRST (finding 4-opus, D-044): the budget never reached the downloader.

    `YoutubeDownloader` defaulted to a wait of zero and production constructed
    it with that default, so a second run of the same video was denied by the
    lease on its first attempt and never reached the wait D-044 describes. The
    300 s budget and the coalescing it exists to deliver were only reachable
    when two runs had *different* lock keys - which is never two runs of one
    video, the exact case the budget was written for.
    """
    from distill.bundle_store import SINGLE_SOURCE_LOCK_WAIT_SEC
    from distill.source import youtube_lock_key

    fake_tool("yt-dlp", FAKE_YTDLP_DOWNLOAD)
    fake_tool("ffprobe", FAKE_FFPROBE)
    root = tmp_path / "out"
    root.mkdir()
    held = hold_the_lease(root, youtube_lock_key("abc123"))
    clock = BudgetClock(on_sleep=lambda count: held.release() if count == 2 else None)
    monkeypatch.setattr(distill_source, "time", clock)

    resolution = resolving_youtube(monkeypatch, root, SINGLE_SOURCE_LOCK_WAIT_SEC)

    assert clock.sleeps == 2, "the run must poll rather than be denied on its first attempt"
    assert clock.now < SINGLE_SOURCE_LOCK_WAIT_SEC
    assert resolution.source.resolved_path.read_bytes() == b"video"
    distill_source.release_acquisition_lease(resolution.source)


def test_a_batch_item_gives_up_on_the_lock_key_after_its_own_budget(
    fake_tool: Callable[[str, str], Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """FAILS FIRST (finding 4-opus, D-044): both budgets were unreachable, not one.

    A playlist item behind another run's 40-minute video must fail fast and let
    the batch move on, which is a different number and the same mechanism. Held
    for the whole wait, the lease is denied - waiting on a lease never makes it
    stealable.
    """
    from distill.bundle_store import BATCH_ITEM_LOCK_WAIT_SEC
    from distill.source import youtube_lock_key

    fake_tool("yt-dlp", FAKE_YTDLP_DOWNLOAD)
    fake_tool("ffprobe", FAKE_FFPROBE)
    root = tmp_path / "out"
    root.mkdir()
    held = hold_the_lease(root, youtube_lock_key("abc123"))
    clock = BudgetClock()
    monkeypatch.setattr(distill_source, "time", clock)

    with pytest.raises(DistillError) as exc:
        resolving_youtube(monkeypatch, root, BATCH_ITEM_LOCK_WAIT_SEC)

    assert exc.value.code == "E_LOCKED"
    assert BATCH_ITEM_LOCK_WAIT_SEC == 5.0
    assert clock.now == pytest.approx(BATCH_ITEM_LOCK_WAIT_SEC)
    assert promoted_names(root) == []
    held.release()


def test_a_source_claiming_an_unusable_duration_is_refused() -> None:
    """FAILS FIRST (finding 18, R-47): a NaN duration passed every guard downstream.

    The number arrives from ffprobe, so it is data rather than operator error,
    and it is refused the way the acquisition path refuses media it cannot use:
    `E_BAD_MEDIA` at the stage that resolved it, not `E_BAD_OPTIONS`. NaN is
    the one that matters most - it clears `duration > max_duration_sec`, clears
    `duration <= 0`, and then poisons every window and interval computed from
    it - but zero, negative and infinite durations are equally unrunnable.

    The rejected value travels as text because the error is published as JSON
    and a bare NaN is not JSON any strict reader will parse.
    """
    from distill.source import ensure_duration_allowed

    for bad in (float("nan"), float("inf"), 0.0, -1.0):
        with pytest.raises(DistillError) as exc:
            ensure_duration_allowed(bad, 7200.0)
        assert exc.value.code == "E_BAD_MEDIA"
        assert exc.value.stage == "source"
        assert exc.value.details["duration_sec"] == repr(bad)
        json.dumps(exc.value.to_dict(), allow_nan=False)

    # A usable duration still passes, and one over the cap is still the cap's
    # error rather than this one.
    ensure_duration_allowed(12.5, 7200.0)
    with pytest.raises(DistillError) as capped:
        ensure_duration_allowed(7200.5, 7200.0)
    assert capped.value.code == "E_DURATION_CAP"


def test_a_container_reporting_a_nan_duration_is_not_promoted(
    fake_tool: Callable[[str, str], Path],
    tmp_path: Path,
) -> None:
    """R-47: 'no playable duration' has to include the duration that is not a number.

    `duration_sec <= 0` is false for NaN, so a container whose header claims one
    passed the promotion check and became the media every later stage read.
    """
    fake_tool("ffprobe", FAKE_FFPROBE_NAN_DURATION)
    media = tmp_path / "source.mp4"
    media.write_bytes(b"video")

    with pytest.raises(DistillError) as exc:
        distill_source.validate_media_file(media)

    assert exc.value.code == "E_BAD_MEDIA"
    assert "duration" in exc.value.message


@pytest.mark.parametrize(
    ("recorded", "expected"),
    [
        pytest.param({}, None, id="no_duration_field"),
        pytest.param({"duration_sec": None}, None, id="null"),
        pytest.param({"duration_sec": "12.5"}, None, id="string"),
        pytest.param({"duration_sec": True}, None, id="true"),
        pytest.param({"duration_sec": 0}, None, id="zero"),
        pytest.param({"duration_sec": -1}, None, id="negative"),
        pytest.param({"duration_sec": float("nan")}, None, id="nan"),
        pytest.param({"duration_sec": float("inf")}, None, id="inf"),
        pytest.param({"duration_sec": 10**400}, None, id="401_digit_integer"),
        pytest.param({"duration_sec": 12.5}, 12.5, id="a_usable_duration"),
    ],
)
def test_a_manifests_recorded_duration_is_read_as_input_rather_than_fact(
    recorded: dict[str, object],
    expected: float | None,
) -> None:
    """Every shape a **manifest**'s `duration_sec` can arrive in, and what it means.

    A manifest is a document another process wrote, so this is the one place
    that decides what counts as a duration a cache hit may reuse - and each
    branch was reachable only through a whole cached run, so deleting the
    `bool` guard or the non-finite test left the suite green.

    The 401-digit integer is the shape the function did not survive: JSON has
    no integer ceiling, `float()` on one raises `OverflowError`, and that
    escaped as a bare stdlib exception from a resolution that should have
    reported a **cache miss**.
    """
    from distill.source import manifest_duration

    assert manifest_duration(recorded) == expected


# A fake yt-dlp that answers a `--print` invocation and records the argv it was
# handed, appending rather than overwriting so a test can see every call a run
# made rather than only the last one.
FAKE_YTDLP_RECORDING_EVERY_ARGV = """
import os, pathlib, sys

argv = sys.argv[1:]
with open(os.environ["FAKE_YTDLP_ARGV_LOG"], "a") as handle:
    handle.write(repr(argv) + "\\n")
sys.stdout.write(argv[-1].rsplit("=", 1)[-1].rsplit("/", 1)[-1] + "\\n")
"""


def recorded_invocations(log: Path) -> list[list[str]]:
    return [literal_eval(line) for line in log.read_text().splitlines() if line]


def test_downloading_a_watch_url_acquires_the_one_video_the_url_names(
    fake_tool: Callable[[str, str], Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """FAILS FIRST (deferred by M8.3): `watch?v=A&list=P` is the playlist to yt-dlp.

    Without `--no-playlist` yt-dlp treats the `list` parameter as the thing being
    asked for, so one output template is pointed at every entry of the playlist
    and the run proceeds against whichever of them landed on `source.mp4`. The
    URL names one video and the operator asked for one video.

    It is also what makes the cache fast path's premise true for this URL shape:
    `youtube_fast_path_video_id` declines a URL carrying a `list`, because the
    id written in it was not the id the run would publish under. With the flag
    it is.
    """
    fake_tool("yt-dlp", FAKE_YTDLP_DOWNLOAD)
    fake_tool("ffprobe", FAKE_FFPROBE)
    argv_file = tmp_path / "argv.txt"
    monkeypatch.setenv("FAKE_YTDLP_ARGV_FILE", str(argv_file))

    acquired = YoutubeDownloader(tmp_path).acquire(
        "https://www.youtube.com/watch?v=abc123&list=PLxyz",
        LOCK_KEY,
        ProgressReporter(),
    )
    acquired.lease.release()

    assert "--no-playlist" in literal_eval(argv_file.read_text())


def test_resolving_a_watch_url_resolves_the_one_video_the_url_names(
    fake_tool: Callable[[str, str], Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """FAILS FIRST: the metadata calls read the playlist too.

    `--dump-json` over a playlist emits one document per entry, which does not
    parse as one, and `--print id` prints one line per entry. Both are the same
    mistake as the download's, and a resolution that disagreed with the download
    about which video this URL is would be worse than either.
    """
    from distill.source import canonical_youtube_id

    fake_tool("yt-dlp", FAKE_YTDLP_RECORDING_EVERY_ARGV)
    argv_log = tmp_path / "argv.log"
    monkeypatch.setenv("FAKE_YTDLP_ARGV_LOG", str(argv_log))

    canonical_youtube_id("https://www.youtube.com/watch?v=abc123&list=PLxyz")

    invocations = recorded_invocations(argv_log)

    assert invocations
    for invocation in invocations:
        assert "--no-playlist" in invocation


def test_listing_a_playlist_is_not_told_to_ignore_the_playlist(
    fake_tool: Callable[[str, str], Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The one yt-dlp call whose subject really is the playlist.

    Named so the flag cannot be added to the shared command builder and left
    there: `--no-playlist` on the listing would make a playlist job enumerate a
    single video.
    """
    from distill.pipeline import youtube_playlist_urls

    fake_tool("yt-dlp", FAKE_YTDLP_RECORDING_EVERY_ARGV)
    argv_log = tmp_path / "argv.log"
    monkeypatch.setenv("FAKE_YTDLP_ARGV_LOG", str(argv_log))

    youtube_playlist_urls("https://www.youtube.com/playlist?list=PLabc", 10)

    for invocation in recorded_invocations(argv_log):
        assert "--no-playlist" not in invocation
