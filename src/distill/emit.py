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
"""

from __future__ import annotations

import errno
import itertools
import logging
import os
from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar

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
        LOGGER.debug("directory not open-able for fsync: %s", _errno_name(exc))
        return
    try:
        os.fsync(fd)
    except OSError as exc:
        LOGGER.debug("directory fsync refused: %s", _errno_name(exc))
    finally:
        os.close(fd)


def _errno_name(exc: OSError) -> str:
    """The symbolic errno of a refusal, for a reason a user can act on."""
    return (errno.errorcode.get(exc.errno, "") if exc.errno is not None else "") or str(exc.errno)
