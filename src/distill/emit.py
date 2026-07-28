"""The single point at which Distill's text becomes durable.

This module owns the act of putting text on disk: the two write disciplines
Distill has, and the temporary file the second of them needs. A **render**, a
**transcript**, a **stage result**, a **bundle marker**, a **manifest**, a job
record and the playlist summary all become durable here and nowhere else.

Why one place. **Extracted text** is chosen by whoever produced the **source**,
and a **render** is written to be fed to an LLM agent, so a **redaction sink**
is a place attacker-controlled text stops being in flight and starts being
something a later reader trusts. Coverage of those sinks is only ever as good as
the answer to "how many are there?", and before this module the answer was
"however many modules called `write_text`" - which is how a **stage result** was
published carrying unredacted text (findings 4, 15). One emitter makes that a
question about one function. `tests/test_emit.py` is what keeps the count at
one, and its docstring states exactly what that check can and cannot see.

Two disciplines rather than one, because Distill writes two kinds of target
(D-033). `emit` is an ordinary write for a target held inside a **staging
directory** under this run's lock, which no other process may read: there is
nobody an atomic replace would protect, and a torn write costs the one run that
made it a recomputation. `emit_atomically` is for state another process reads -
the **manifest** above all, where a half-written file is a directory that is
briefly not a **bundle** at all.

Both refuse to be redirected at the moment of use. `O_NOFOLLOW` is R-16 decided
by the kernel rather than by a check that ran first, so a symlink swapped in
after the caller confined the path fails with `ELOOP` instead of being written
through, and `O_NONBLOCK` turns the one file kind that would otherwise wait -
a fifo with no reader - into `ENXIO` rather than a run hanging under its own
lock.

What this module does not own. It does not choose the path: bundle layout is
`bundle_store`'s, and the emitter is handed a target. It does not decide whether
a path may be written at all - that is `bundle_store.confined_path`, which the
caller asks first, and which a deletion asks as sharply as a write does. It does
not know what the text says, whether **redaction** ran over it (that claim is
checked in `artifacts.serialize`, at the point a carrier is frozen), or what
should happen when a write is refused: an `OSError` is raised and the caller
decides whether it ends the run or costs it a **stage result**.

Nor does it see durable content Distill does not write itself - a download from
`yt-dlp`, a **keyframe** from `ffmpeg`. Those are third-party writes, and no
emitter can be their choke point.

The other half of what must be true of text before it is durable is the
**untrusted-data boundary**: **extracted text** was chosen by whoever produced
the **source**, and a **render** is written to be fed to an LLM agent, so text
that stops being quoted starts being read as instruction. Selecting the
delimiter that holds - a fence longer than the longest backtick run in the
content (R-25), an escape that keeps a link label from closing its own
construct (R-27) - lives here, beside the write, because it is the same
question about the same text: what has to be true of it before a later reader
sees it. What it is *not* is markdown document structure: which heading a block
sits under, and what the document says about it, are `render.py`'s.

Delimiting is a mitigation and not a guarantee (D-022). A sufficiently
persuasive payload may still influence a model that reads a render; a delimiter
raises the cost of one and makes its provenance legible, and nothing here
claims more.
"""

from __future__ import annotations

import itertools
import logging
import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar

from .errors import errno_name

LOGGER = logging.getLogger(__name__)

ATOMIC_TEMP_SUFFIX = ".tmp"
"""How the temporary of an atomic write is named.

A trailing suffix on the target's own name, so a temporary left behind by a
writer that failed is legible to whoever finds it: it says which file it was
becoming and, through the pid in the middle, which process left it.
"""

_WRITE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW | os.O_NONBLOCK
"""Open a regular file for writing, or fail: create it if absent, truncate it if
present, refuse a symlink, and refuse to wait on a fifo."""

_FILE_MODE = 0o644
"""One creation mode for everything Distill writes, masked by the umask as usual.

Uniform on purpose. Before the emitter a **stage result** was created `0o644`
while a **render** went through `Path.write_text` and got `0o666` masked, so a
user with a group-writable umask ended up with a bundle whose files disagreed
about who may change them. Group-writable bundle content is the wrong side of
that disagreement to settle on: a **generation** is immutable once published,
and a reader is entitled to it, not a writer (D-015 covers the on-disk change).
"""


UNTRUSTED_TEXT_LABEL = "untrusted-text"
"""The **trust class** a delimited block carries, written as its info string.

One class rather than a set, because only one needs saying: **extracted text**
is the text a reader must not act on, and everything else in a **render** is
Distill's own words, which the document already speaks in. A block tagged this
way says who chose its contents - the person who produced the **source** - and
that is what a reader needs to know before reading it.
"""

MIN_FENCE_BACKTICKS = 3
"""The shortest run of backticks that opens a fenced block at all."""

_BACKTICK_RUN_RE = re.compile(r"`+")

_PLAIN_DESTINATION_RE = re.compile(r"[^\s()<>\\`\[\]\x00-\x1f\x7f]+")
"""A link destination that needs no angle brackets to be unambiguous.

Nothing in it can end the `(...)` that holds it, and nothing in it takes part
in another construct: no parentheses, no whitespace, no line ending, no angle
bracket, no backslash, no backtick and no bracket. The last two are not known
escapes - a destination is parsed from the raw text, so a code span inside one
mangles the link rather than retargeting it - and they are excluded anyway,
because the value of this path is that a reader can see at a glance that
nothing in it participates in anything. Matched in full or not at all: a
destination is not partly safe.
"""

_LABEL_ESCAPED_CHARACTERS = frozenset("\\`[]<>")
"""What a link label must not be allowed to open or close.

`]` is the retarget (RV-5): a label carrying `](` closes itself and opens a
destination the label's author chose. `[` is the same construct opened one
level in, a backtick opens a code span that swallows the `]` that would have
closed the label, `<` opens an autolink, and `\\` is escaped first so that
every other escape below means what it says.

`&` is not here because a backslash is the wrong tool for it: an ampersand
opens nothing, it *rewrites* - see `AMPERSAND_REFERENCE`.
"""

AMPERSAND_REFERENCE = "&amp;"
"""How a literal ampersand is written, in a label and in a destination alike.

CommonMark resolves **entity and numeric character references** in both halves
of a link, so an ampersand is the one character that changes its neighbours
rather than closing a construct. Left alone, `&copy;` is six characters a reader
turns into one: a label reading `©` that the **source** never wrote, and - the
part that matters - a destination `?q=&copy;` that resolves to `?q=%C2%A9`, and
`a&amp;b` that resolves to `a&b`. That is the retarget R-27 exists to stop,
reached through a reference instead of through a bracket, and `&#41;` reaches it
with a `)`.

An entity rather than `\\&`, because a destination is where this has to hold and
the plain form of one may not carry a backslash (`_PLAIN_DESTINATION_RE`).
`&amp;` is what an ampersand is written as in every other markup a reader knows,
and it survives both halves under one rule.
"""

_LABEL_LINE_ENDING_REFERENCES = {"\n": "&#10;", "\r": "&#13;"}
"""A line ending in a label, written so the reader gets the character back.

A label is one line of a document, so the line ending cannot stay a line ending:
a real one lands the rest of the label at the document's own indentation, where
`# PWNED` is a heading the render appears to have written. But a *numeric
character reference* is not a line ending in the source - the parser has already
decided where the line ends by the time it resolves one - and it is a line
ending to the reader. So the label stays one line and the text survives.

This replaces writing a visible `\\n`, which lost the character twice over: a
reader got a backslash and a letter, and a label that carried a literal
backslash-then-`n` was written exactly the same way, so two different sources
arrived as one string nobody could tell apart.
"""

_DESTINATION_ENCODED_RE = re.compile(r"[\x00-\x1f\x7f]")
"""Control characters in a destination, percent-encoded rather than escaped.

A destination in angle brackets may not contain a line ending at all, so a
newline cannot be backslash-escaped into safety - it has to stop being a line
ending. Percent-encoding is what a URL does with one anyway.
"""


class TextEmitter:
    """Every durable write Distill makes, in one of two disciplines.

    A type rather than two loose functions because the choke point is a thing to
    be named, held and eventually asked more of: whatever else must be true of
    text before it is durable belongs on this object, and there is exactly one
    place to add it.

    Stateless except for the sequence that names temporaries, which is shared by
    every instance on purpose. A per-object counter would let two emitters in
    one process pick the same temporary name for the same target, and a second
    writer truncating a temporary while the first replaces it publishes a
    half-written file *onto* the target - worse than having no temporary at all.
    """

    _sequence: ClassVar[Iterator[int]] = itertools.count()

    def emit(self, path: Path, text: str) -> Path:
        """Write `text` at `path`, replacing whatever text was there.

        The ordinary discipline: for a target inside a directory this run holds,
        which no other process may read (D-033). The caller has already decided
        the path is one it may write; this refuses at the open to be redirected
        by what is at the name *now*, because between a check and an open a path
        can be replaced, and the check is the half that goes out of date.

        Raises `OSError` when the target is a symlink, a directory, a fifo, or a
        file this process may not write. What that costs the run is the caller's
        to decide.
        """
        with os.fdopen(os.open(path, _WRITE_FLAGS, _FILE_MODE), "w", encoding="utf-8") as handle:
            handle.write(text)
        return path

    def emit_atomically(self, path: Path, text: str) -> Path:
        """Write `text` at `path` so a concurrent reader sees old bytes or new.

        The discipline for state another process reads (R-14): the **manifest**,
        job records, the playlist summary. Written to a temporary beside the
        target - beside, because a replace has to stay on one filesystem to be
        atomic at all - and renamed over it.

        The temporary name is unique per write, which is what makes the property
        hold under concurrency rather than only in a single process. A writer
        that dies mid-write therefore leaves its own temporary behind, which is
        the cost of not having a shared one to collide on.

        Atomic against another process is not the same as atomic against power,
        and for a **manifest** the difference is severe: a directory entry that
        reaches the disk ahead of the bytes it names leaves a zero-length
        marker, which is an `invalid` bundle - unservable, and a directory a
        walk will not descend into, so the generations under it are
        unreclaimable too. The content is therefore flushed before the replace
        and the directory entry after it: two barriers, because they are two
        different things reaching the disk.
        """
        temporary = self.temporary_for(path)
        # Opened outside the cleanup guard so this only ever removes a file it
        # created. An open that fails - a symlink pre-created at the temporary's
        # name, a full disk - leaves nothing of ours behind, and unlinking on
        # the way out would have this refusal delete whatever is at the name
        # instead of simply declining to write to it.
        descriptor = os.open(temporary, _WRITE_FLAGS, _FILE_MODE)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        _fsync_directory(path.parent)
        return path

    def fence_for(self, text: str) -> str:
        """The shortest backtick fence `text` cannot close (R-25).

        Mechanical, because a fixed length is a guess about content nobody has
        seen yet: three backticks was the guess, and a slide showing a code
        sample - or a payload written to look like one - ended the quotation
        (finding 5). A fence one backtick longer than the longest run in the
        content cannot be closed from inside it, whatever the content is.
        """
        longest = max((len(run.group()) for run in _BACKTICK_RUN_RE.finditer(text)), default=0)
        return "`" * max(MIN_FENCE_BACKTICKS, longest + 1)

    def delimit(self, text: str) -> list[str]:
        """`text` as a fenced block it cannot terminate, tagged as **extracted text**.

        Lines rather than a string because a **render** is assembled as lines,
        and the caller decides what surrounds the block.

        The tag is not a parameter. A caller able to choose it is a caller able
        to label attacker-chosen text as something else, and there is nothing
        for the other value to be: text Distill wrote needs no block, and text
        it did not write is all one trust class.

        Line endings are normalized before the fence is chosen, because the
        fence has to be measured against the lines the block will actually
        hold: a `\\r\\n` left in place makes the content one line here and two
        to a reader (CommonMark ends a line on `\\r`, `\\n` or `\\r\\n`).
        """
        body = text.replace("\r\n", "\n").replace("\r", "\n")
        fence = self.fence_for(body)
        return [f"{fence}{UNTRUSTED_TEXT_LABEL}", *body.split("\n"), fence]

    def link_label(self, text: str) -> str:
        """`text` escaped so it cannot end the link label holding it (R-27).

        Lossless, in the one sense that can be checked: a CommonMark reader
        resolving this label arrives at exactly the characters the **source**
        carried, line endings included. That is a claim about the reader's
        output and not about these bytes - a backslash, an entity and a numeric
        character reference are all longer written than read - and it is the
        claim that matters, because the reader is who the label is for.
        `tests/test_render_delimiting.py` asserts it against a real parser
        rather than against a scanner of the suite's own, which is what let the
        ampersand through in the first place.

        Three rules, one per way a character stops meaning itself. A character
        that would *open or close* a construct is escaped with a backslash. An
        ampersand, which rewrites rather than opens, is written as the reference
        for an ampersand. A line ending, which cannot survive as a character
        without ending the document line the link sits on, is written as the
        reference for that character.
        """
        escaped: list[str] = []
        for character in text:
            if character in _LABEL_ESCAPED_CHARACTERS:
                escaped.append(f"\\{character}")
            elif character == "&":
                escaped.append(AMPERSAND_REFERENCE)
            elif character in _LABEL_LINE_ENDING_REFERENCES:
                escaped.append(_LABEL_LINE_ENDING_REFERENCES[character])
            else:
                escaped.append(character)
        return "".join(escaped)

    def link_destination(self, url: str) -> str:
        """`url` written so it resolves to itself and nothing else (R-27).

        Two different things can go wrong with a destination and they need
        different answers. It can *end* the `(...)` holding it, which angle
        brackets and a backslash escape settle. And it can be *rewritten* by
        what it carries, which only escaping the ampersand settles - brackets
        do not stop a reference resolving, so both halves of the rule run
        whichever form the destination takes.

        Angle brackets when the destination needs them and not otherwise, which
        is a rule rather than a preference: a destination carrying nothing that
        could terminate the construct is already unambiguous, and wrapping it
        would make every ordinary link in every render harder to read for the
        sake of a case it is not in. Everything else is wrapped, its own
        brackets and backslashes escaped, and its control characters
        percent-encoded so no line ending survives inside the brackets.

        Lossless in the same sense the label is, with one step more of the
        reader's own machinery to undo: a parser percent-encodes a destination
        on the way in, so what comes back is the URL after that encoding is
        reversed. A space arrives as `%20` and a line ending as `%0A`, which is
        what a URL does with them anyway.
        """
        destination = url.strip().replace("&", AMPERSAND_REFERENCE)
        if _PLAIN_DESTINATION_RE.fullmatch(destination):
            return destination
        escaped = (
            destination.replace("\\", "\\\\").replace("<", "\\<").replace(">", "\\>")
        )
        encoded = _DESTINATION_ENCODED_RE.sub(
            lambda match: f"%{ord(match.group()):02X}", escaped
        )
        return f"<{encoded}>"

    def temporary_for(self, path: Path) -> Path:
        """Name the temporary an atomic write to `path` would use.

        Named by a method rather than inline so the uniqueness property can be
        asserted without performing a write, and so the pid is in the name: a
        temporary found on disk says which process left it.
        """
        unique = f"{os.getpid()}.{next(TextEmitter._sequence)}"
        return path.with_name(f"{path.name}.{unique}{ATOMIC_TEMP_SUFFIX}")


EMITTER = TextEmitter()
"""The emitter every caller uses.

One instance because the sequence naming temporaries is shared anyway, and
because a module-level name is what makes "go through the emitter" a thing a
reader can grep for.
"""


def _fsync_directory(directory: Path) -> None:
    """Make a rename durable, not just visible.

    The replace publishes a directory entry, and an entry is as much in cache as
    a file's bytes are. Flushed here rather than by the writer above, because
    what has to survive is the directory's state and not the file's.

    A refusal is logged and not raised: the bytes are already visible to every
    reader by this point, so a filesystem that will not flush a directory
    (`EINVAL` on some of them) leaves a durability gap and not a failed write,
    and ending the run over it would cost the caller the write it just made.
    """
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError as exc:
        LOGGER.debug("directory not open-able for fsync: %s", errno_name(exc))
        return
    try:
        os.fsync(fd)
    except OSError as exc:
        LOGGER.debug("directory fsync refused: %s", errno_name(exc))
    finally:
        os.close(fd)

