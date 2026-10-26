"""The YouTube metadata invocation: what it asks yt-dlp for, and what it reads.

The invocation used to ask for yt-dlp's whole document (`--dump-json`) and
parse it. That document is every field yt-dlp knows, and for a video with many
auto-translated caption tracks it is 10.2 MB - past the 8 MiB capture cap, so
the captured stdout was truncated, the JSON did not parse, and the bundle lost
its title, channel and upload date.

The invocation now asks for the six fields this code reads. These tests pin
that: what the argv carries, what a six-field document yields, and the bug
itself - a document that would overflow the cap no longer costs the metadata.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from distill.youtube import youtube_metadata

THE_FIELD_TEMPLATE = "%(.{id,title,channel,uploader,upload_date,description})j"
"""The template the metadata invocation asks for, spelled out here.

Written out rather than imported from the module: a test that read the
constant it is testing would pass whatever the constant said, and the question
this file exists to answer is what the invocation asks for.
"""

FAKE_YTDLP_RECORDING_ARGV = """
import json, os, sys

argv = sys.argv[1:]
with open(os.environ["FAKE_YTDLP_ARGV_LOG"], "a") as handle:
    handle.write(json.dumps(argv) + "\\n")
sys.stdout.write(json.dumps({
    "id": "abcdefghijk",
    "title": "Release notes",
    "channel": "Distill Channel",
    "description": "",
}) + "\\n")
"""

FAKE_YTDLP_SIX_FIELDS = """
import json

print(json.dumps({
    "id": "abcdefghijk",
    "title": "Release notes",
    "channel": "Distill Channel",
    "uploader": "Fallback Uploader",
    "upload_date": "20260729",
    "description": "First paragraph.",
}))
"""

FAKE_YTDLP_SIX_FIELDS_WITHOUT_A_CHANNEL = """
import json

print(json.dumps({
    "id": "abcdefghijk",
    "title": "Release notes",
    "uploader": "Fallback Uploader",
    "upload_date": "20260729",
    "description": "First paragraph.",
}))
"""

# A yt-dlp that answers the whole document with the auto-translated caption
# tracks that make it larger than the capture cap, and answers the six fields
# with six fields. The two branches are what the bug was: the first overflows
# the cap, the second does not exist until the invocation asks for it.
FAKE_YTDLP_WHOLE_DOCUMENT_OVERFLOWS_THE_CAP = """
import json, sys

argv = sys.argv[1:]
fields = {
    "id": "abcdefghijk",
    "title": "Release notes",
    "channel": "Distill Channel",
    "uploader": "Fallback Uploader",
    "upload_date": "20260729",
    "description": "First paragraph.",
}
if "--dump-json" in argv:
    document = dict(fields)
    document["automatic_captions"] = {
        "lang-%d" % index: [{"ext": "json3", "url": "x" * 4096}] for index in range(2200)
    }
    sys.stdout.write(json.dumps(document))
else:
    sys.stdout.write(json.dumps(fields))
"""


def recorded_invocations(log: Path) -> list[list[str]]:
    return [json.loads(line) for line in log.read_text().splitlines() if line]


def test_the_metadata_invocation_asks_for_the_fields_it_reads(
    fake_tool: Callable[[str, str], Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The whole document is not the answer, and asking for it is what broke.

    `--dump-json` returns every field yt-dlp knows, which for a video with many
    auto-translated caption tracks is larger than the capture cap. The
    invocation asks for the six fields this code reads instead.
    """
    fake_tool("yt-dlp", FAKE_YTDLP_RECORDING_ARGV)
    argv_log = tmp_path / "argv.log"
    monkeypatch.setenv("FAKE_YTDLP_ARGV_LOG", str(argv_log))

    youtube_metadata("https://youtu.be/abcdefghijk")

    invocations = recorded_invocations(argv_log)
    assert len(invocations) == 1
    invocation = invocations[0]
    assert "--dump-json" not in invocation
    assert "--print" in invocation
    assert THE_FIELD_TEMPLATE in invocation


@pytest.mark.parametrize(
    ("script", "expected_channel"),
    [
        (FAKE_YTDLP_SIX_FIELDS, "Distill Channel"),
        (FAKE_YTDLP_SIX_FIELDS_WITHOUT_A_CHANNEL, "Fallback Uploader"),
    ],
)
def test_a_document_holding_only_the_six_fields_yields_the_provenance(
    fake_tool: Callable[[str, str], Path],
    script: str,
    expected_channel: str,
) -> None:
    """Six fields are enough for the title, the channel and the upload date.

    The channel falls back to the uploader, which is what the field template
    asks for as well: the two are read from the same document, and the
    fallback is unchanged by the invocation asking for fewer fields.
    """
    fake_tool("yt-dlp", script)

    metadata = youtube_metadata("https://youtu.be/abcdefghijk")

    assert metadata.video_id == "abcdefghijk"
    assert metadata.title == "Release notes"
    assert metadata.channel == expected_channel
    assert metadata.upload_date == "20260729"
    assert metadata.warnings == []


def test_a_document_too_large_for_the_capture_cap_still_yields_the_metadata(
    fake_tool: Callable[[str, str], Path],
) -> None:
    """The bug, reproduced and closed.

    The whole document for this video is over 8 MiB - the capture cap. Asked
    for as a document it is truncated, the JSON does not parse, and the bundle
    loses its title, channel and upload date. Asked for as six fields it is
    2.7 KB, and the metadata survives.
    """
    fake_tool("yt-dlp", FAKE_YTDLP_WHOLE_DOCUMENT_OVERFLOWS_THE_CAP)

    metadata = youtube_metadata("https://youtu.be/abcdefghijk")

    assert metadata.title == "Release notes"
    assert metadata.channel == "Distill Channel"
    assert metadata.upload_date == "20260729"
    assert [item["code"] for item in metadata.warnings] == []
