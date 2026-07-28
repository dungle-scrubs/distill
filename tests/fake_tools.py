"""Fake external tools the acquisition tests install on `PATH`.

Every external tool now runs through `run_command`, which spawns a real child in
its own process group, so a fake tool is a real executable rather than a patched
`Popen`. These scripts are shared by the tests that drive source acquisition, so
the shape of "what yt-dlp leaves on disk" is written down once.

This module owns only the scripts' text. It does not install them - the
`fake_tool` fixture in `conftest.py` does that - and it asserts nothing.
"""

from __future__ import annotations

# A fake yt-dlp writing realistic `--newline` download progress to **stdout**,
# which is the stream yt-dlp actually uses (R-32). It produces the media file
# the same way yt-dlp does, from the `-o` template, and never touches a network.
# `FAKE_YTDLP_ARGV_FILE`, when set, records the argv it was invoked with;
# the media itself lands in a staging directory that acquisition removes, so a
# test that wants the argv cannot read it back from beside the media.
FAKE_YTDLP_DOWNLOAD = """
import os, pathlib, sys

argv = sys.argv[1:]
template = argv[argv.index("-o") + 1]
media = pathlib.Path(template.replace("%(ext)s", "mp4"))
media.parent.mkdir(parents=True, exist_ok=True)
media.write_bytes(b"video")
argv_file = os.environ.get("FAKE_YTDLP_ARGV_FILE")
if argv_file:
    pathlib.Path(argv_file).write_text(repr(argv))
if "--progress" not in argv:
    # Standing in for a user config that set --quiet: without --progress the
    # real yt-dlp goes silent, and a silent download starves the idle timer.
    sys.exit(0)
for line in [
    "[download] Destination: source.mp4",
    "[download]  10.0% of   10.00MiB at    1.00MiB/s ETA 00:09",
    "[download]  55.0% of   10.00MiB at    1.00MiB/s ETA 00:04",
    "[download] 100.0% of   10.00MiB in 00:00:10",
]:
    sys.stdout.write(line + "\\n")
    sys.stdout.flush()
"""

# A fake yt-dlp that answers every invocation a YouTube run makes of it: the
# `--dump-json` metadata document, the `--print` id, and the download itself. The
# real tool is one binary doing all three, so a test about "is yt-dlp needed at
# all" needs one fake that cannot be needed for only part of it. The video id it
# reports is the one in the URL, which is what the real tool reports too.
FAKE_YTDLP_METADATA_AND_DOWNLOAD = """
import json, pathlib, sys

argv = sys.argv[1:]
video_id = argv[-1].rsplit("=", 1)[-1].rsplit("/", 1)[-1]
if "--dump-json" in argv:
    sys.stdout.write(json.dumps({"id": video_id, "description": ""}))
elif "-o" not in argv:
    sys.stdout.write(video_id + "\\n")
else:
    template = argv[argv.index("-o") + 1]
    media = pathlib.Path(template.replace("%(ext)s", "mp4"))
    media.parent.mkdir(parents=True, exist_ok=True)
    media.write_bytes(b"video")
"""

FAKE_YTDLP_FAILING = """
import sys

sys.stdout.write("[download] Destination: source.mp4\\n")
sys.stderr.write("ERROR: fatal error\\n")
sys.exit(1)
"""

# yt-dlp opens its destination for writing before it knows the transfer will
# succeed. A download that dies mid-transfer therefore leaves a truncated file
# where the destination was - which is what makes downloading straight into the
# promoted path destructive (RV-3).
FAKE_YTDLP_TRUNCATES_THEN_FAILS = """
import pathlib, sys

argv = sys.argv[1:]
template = argv[argv.index("-o") + 1]
media = pathlib.Path(template.replace("%(ext)s", "mp4"))
media.parent.mkdir(parents=True, exist_ok=True)
media.write_bytes(b"")
sys.stderr.write("ERROR: fatal error\\n")
sys.exit(1)
"""

# A second run's download, distinguishable byte-for-byte from the first run's.
FAKE_YTDLP_CLOBBERING = """
import pathlib, sys

argv = sys.argv[1:]
template = argv[argv.index("-o") + 1]
media = pathlib.Path(template.replace("%(ext)s", "mp4"))
media.parent.mkdir(parents=True, exist_ok=True)
media.write_bytes(b"clobbered")
"""

# An interrupted merge: yt-dlp downloaded the separate audio and video formats
# but never merged them, so the format fragments sit beside the container it
# would have produced.
FAKE_YTDLP_LEAVES_FORMAT_FRAGMENTS = """
import pathlib, sys

argv = sys.argv[1:]
template = argv[argv.index("-o") + 1]
directory = pathlib.Path(template).parent
directory.mkdir(parents=True, exist_ok=True)
(directory / "source.f140.m4a").write_bytes(b"audio-fragment")
(directory / "source.f299.mp4").write_bytes(b"video-fragment")
(directory / "source.mp4.part").write_bytes(b"partial")
(directory / "source.mp4").write_bytes(b"video")
"""

# Records the directory it was told to write into, one line per invocation, so a
# test can see where successive runs staged their downloads.
FAKE_YTDLP_RECORDS_STAGING_DIR = """
import os, pathlib, sys

argv = sys.argv[1:]
template = argv[argv.index("-o") + 1]
media = pathlib.Path(template.replace("%(ext)s", "mp4"))
media.parent.mkdir(parents=True, exist_ok=True)
media.write_bytes(b"video")
with open(os.environ["FAKE_YTDLP_STAGING_LOG"], "a") as handle:
    handle.write(str(media.parent) + "\\n")
"""

# yt-dlp exits 0 having produced nothing but an empty file.
FAKE_YTDLP_PRODUCES_EMPTY_FILE = """
import pathlib, sys

argv = sys.argv[1:]
template = argv[argv.index("-o") + 1]
media = pathlib.Path(template.replace("%(ext)s", "mp4"))
media.parent.mkdir(parents=True, exist_ok=True)
media.write_bytes(b"")
"""

# yt-dlp exits 0 having produced a link rather than a file. The download is the
# one step that runs a tool Distill does not control inside a directory Distill
# writes, so what comes back out of staging is as untrusted as what comes back
# on stdout. The link's target is named by FAKE_YTDLP_LINK_TARGET.
FAKE_YTDLP_PRODUCES_A_SYMLINK = """
import os, pathlib, sys

argv = sys.argv[1:]
template = argv[argv.index("-o") + 1]
media = pathlib.Path(template.replace("%(ext)s", "mp4"))
media.parent.mkdir(parents=True, exist_ok=True)
media.symlink_to(os.environ["FAKE_YTDLP_LINK_TARGET"])
"""

# yt-dlp exits 0 having produced an audio-only file: the format selector fell
# through to an audio format, or the merge step silently dropped the video.
FAKE_YTDLP_PRODUCES_AUDIO_ONLY = """
import pathlib, sys

argv = sys.argv[1:]
template = argv[argv.index("-o") + 1]
media = pathlib.Path(template.replace("%(ext)s", "m4a"))
media.parent.mkdir(parents=True, exist_ok=True)
media.write_bytes(b"audio")
"""

# A fake ffprobe that answers both questions Distill asks of it: which streams a
# file carries, and how long it is. It reads the answer off the file's suffix
# and size, so an audio-only container reports no video stream and an empty file
# reports no duration.
FAKE_FFPROBE = """
import json, pathlib, sys

target = pathlib.Path(sys.argv[-1])
streams = [{"codec_type": "audio"}]
if target.suffix not in {".m4a", ".mp3", ".opus", ".ogg"}:
    streams.insert(0, {"codec_type": "video"})
size = target.stat().st_size if target.exists() else 0
payload = {"streams": streams, "format": {"duration": "12.5" if size else "0"}}
sys.stdout.write(json.dumps(payload))
"""

# A fake ffmpeg that writes a real, if minimal, PNG where it was told to. The
# image has to be openable: a keyframe whose pHash cannot be computed still
# becomes a **frame artifact**, but it records a `phash_failed` **warning**, and
# a test about something else should not be publishing one.
FAKE_FFMPEG_WRITES_A_REAL_PNG = """
import base64, pathlib, sys

pathlib.Path(sys.argv[-1]).write_bytes(
    base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQ"
        "DwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )
)
"""


def fake_ffprobe_flooding_stderr(cap_bytes: int) -> str:
    """A working ffprobe that also floods stderr past run_command's capture cap.

    The invocation both answers and records truncation (R-33). `-v error` keeps
    ffprobe quiet about a file it understands; a container it dislikes makes it
    anything but.
    """
    return f"""
import json, sys

payload = {{"streams": [{{"codec_type": "video"}}], "format": {{"duration": "12.5"}}}}
sys.stdout.write(json.dumps(payload))
sys.stderr.write("x" * ({cap_bytes} + 1024))
"""


# A container whose header claims a duration that is not a number. ffprobe
# reports what the header says, so `nan` reaches Distill as data rather than as
# a probe failure.
FAKE_FFPROBE_NAN_DURATION = """
import json, sys

sys.stdout.write(
    json.dumps({"streams": [{"codec_type": "video"}], "format": {"duration": "nan"}})
)
"""

# A container ffprobe can open and read a video stream from, but whose duration
# it cannot determine - the shape of a transfer that stopped before the index
# was written.
FAKE_FFPROBE_NO_DURATION = """
import json, sys

sys.stdout.write(json.dumps({"streams": [{"codec_type": "video"}], "format": {}}))
"""
