"""The durable-emission path and the structural check that keeps it single (R-22).

`TextEmitter` is the one place Distill's text becomes durable. A **render**, a
**transcript**, a **stage result**, a **bundle marker**, a **manifest**, a job
record and the playlist summary all reach the disk through it, so that a
question about durable text - has **redaction** run, is the target confined to
the bundle, did the bytes survive a crash - has one function to be asked of
instead of one per module. That is the whole value of the choke point, and it is
worth only as much as the claim that it *is* one.

The structural check below is what keeps that claim true. It parses every module
under `src/distill/` and fails on any call that makes text durable which
`ALLOWED_WRITERS` does not account for - which module, which function, which
call form, and how many of it.

What it detects is an *enumerated* set of call shapes, matched as they are
written. The set is exactly this, and the list is closed:

- `path.write_text(...)` and `path.write_bytes(...)`, on any receiver
- `open(..., "w")`, `path.open("a")`, `gzip.open(path, "wt")`, `os.fdopen(fd,
  "w")` - any call named `open` or `fdopen` carrying a mode-shaped string that
  asks to write, in any positional slot or as `mode=`, because the slot is not
  stable across those spellings. A mode reaching the call unpacked, as
  `open(*argv)` or `open(path, **options)`, is not written in the call and
  blind spot 1 has it
- `os.open(..., os.O_WRONLY | ...)` and `os.write(...)` - the descriptor form
  the emitter itself is built from, so a copy of it lands somewhere this check
  looks. `os.open` is read from the flag *names* in the call: a call passing
  them in a variable is an aliased writer and blind spot 1 has it, and a call
  passing the number those names add up to is undetected for the plainer
  reason that it carries none of them
- `json.dump(...)`, spelled on the name `json`, which writes through a file
  object it is handed

A mode string is read as one only inside a call named `open` or `fdopen`. That
gate is deliberate: without it every call anywhere carrying a short string drawn
from `rwxabt+` would be a durable write, which is a check reporting `re.sub` and
`str.strip` on their second argument.

What it does not detect - stated first and plainest, because a check whose
limits go unstated is a check that gets trusted past them (spike A-004):

**Any durable-write shape outside that set.** Not by a judgement that it does
not matter; the check simply has no rule that names it. Written literally,
unaliased, in a module under `src/distill/`, every one of these puts something
on disk and leaves this check silent:

    os.pwrite(fd, text.encode(), 0)        os.writev(fd, [text.encode()])
    shutil.copyfile(src, dst)              shutil.copyfileobj(reader, writer)
    tempfile.NamedTemporaryFile(delete=False)
    tempfile.NamedTemporaryFile(mode="w")
    print(text, file=handle)               pickle.dump(document, handle)
    os.symlink(target, name)               handle.writelines(lines)
    os.open(path, 577)

Those eleven are named because they are the ones a reader is likeliest to assume
are covered, not because they complete anything: the complement of an
enumerated set is not enumerable. `mmap`, `zipfile.ZipFile(..., "w")`,
`csv.writer`, a subprocess with a redirect and whatever the standard library
grows next are all outside it too, and the list of matched shapes will always
trail. `test_the_shapes_documented_as_undetected_are_undetected` holds this
paragraph to the code, so widening the check is welcome and fails there until
this paragraph is corrected in the same change - and
`test_the_docstring_lists_exactly_the_shapes_this_file_proves_undetected` reads
the block above back out of `__doc__`, so the sentence cannot be edited into
promising a shape nobody asserted either.

The last three are named for a sharper reason than the rest: each sits close
enough to a matched shape that a reader could reasonably take the detected set
above as covering it. `NamedTemporaryFile(mode="w")` carries a mode string that
asks to write, and the mode rule is gated on the call's name so it never looks.
`writelines` is the file-object write method beside `write`, and neither is
matched - a handle is opened somewhere, and the `open` is the call this check
reads. `os.open(path, 577)` is the matched descriptor form with its flags
spelled as a number: the value is whatever the platform's `O_` constants come
to, which is the point - a number carries none of the names the rule reads, so
it says nothing here however it was arrived at.

Widening it *instead* of saying this was weighed and refused, on the evidence
of the package as it stands. `print(..., file=handle)` cannot be told from
`print(..., file=sys.stderr)` by an AST check, and Distill writes two of the
latter (`cli.py`, `pipeline.py`); adding the shape buys detection of a form
nobody here writes and costs two licence entries for calls that put nothing on
disk. That is how this mechanism dies - the licence list fills with entries
licensing non-writers and stops being a list of durable writers - and R-22
bounds the promise to a contributor following the codebase's patterns, which
these four forms are and `pickle`, `shutil.copyfile` and `os.pwrite` are not.

The three near-misses are refused on the same evidence. `writelines` would flag
every `handle.writelines` in the package for a durability that belongs to the
`open` two lines above it, which this check already reads. A literal `os.open`
flag would have to be *evaluated* rather than read, and following a value is the
line between this check and a type checker - `os` exports the names, and both
call sites in Distill use them. `NamedTemporaryFile` is a real temporary-file
writer, and detecting it means either naming it specifically, which is a rule
for one library call, or ungating the mode rule, which is the false-positive
flood that gate exists to prevent. All three stay out and are stated instead,
which is what this list is for.

Then three further blind spots, which are routes rather than shapes:

1. **Aliased writers.** The check matches the call as it is written. `from json
   import dump` then `dump(document, handle)`, or `writer = path.write_text`
   then `writer(text)`, or `getattr(path, "write_" + "text")` are durable writes
   this check walks straight past. A different module spelling the same
   function - `simplejson.dump`, or a vendored `dump` - is the same miss. So is
   an argument that reaches the call from somewhere else rather than being
   written in it: `os.open(path, flags)`, `open(*argv)`, `open(path, **options)`.
   Every one would need the value followed, and following names is the line
   between this check and a type checker.
2. **Writes through a helper in another module.** A module calling
   `bundle_store.atomic_write_text` writes durably and is not flagged. That is
   correct today - the helper reaches the emitter - but the check cannot tell
   that helper from one that does not. A new helper that writes without the
   emitter is flagged once, where it is defined, and every module that calls it
   stays invisible.
3. **Third-party writes.** `yt-dlp` writing a download, `ffmpeg` writing a
   **keyframe**, Pillow saving an image: durable content Distill causes and does
   not itself write. Nothing here sees them.

One further limit of scope, which is not a blind spot so much as what this check
is about. It sees text *becoming* durable, not durable text being moved:
**publish** renames a **staging directory** into a **generation** and this check
has nothing to say about it.

What is *not* a limit, and was one: an entry in `ALLOWED_WRITERS` licenses an
exact multiset of call forms, counts included, and not the function they sit in.
A licensed function that gains a *second* writer fails this check as loudly as
an unlicensed module does, whether the new call is of a form the licence does
not name or one more of a form it does. Keyed on the function alone - as this
was - a contributor adding a `write_text` inside `ExclusiveLock.take` was
detected by the visitor and then waved through by the licence, which is this
mechanism failing in the ordinary case it exists for rather than in one of its
documented blind spots.
`test_a_second_writer_inside_a_licensed_function_is_reported` and
`test_a_second_writer_of_a_licensed_form_is_reported` are the controls for the
two halves of that, and `test_the_licence_list_names_only_writes_that_exist`
closes the other direction, so the licensed multiset and the written one are
equal rather than one bounding the other.

What a licence still does not identify is *which* calls. It says a function
makes one `os.open`, not which `os.open` that is, so a contributor who replaces
the licensed call with a different one of the same form - opening a different
file, for a purpose the recorded reason never covered - passes this check. That
residue needs blind spot 1 to reach anything, since the licensed call has to
stop being visible for the new one to take its place, and it is the price of
keying on forms instead of on line numbers, which churn on every edit above
them. It is a reason to read a licence's reason when the function under it
changes, not a reason the count is not worth having.

So: this catches a contributor following the codebase's patterns, which is the
way the emitter is most likely to be bypassed and the only way this check
promises to catch. It does not prevent deliberate circumvention - anyone who
means to bypass the emitter can, by any of the routes above, or by one line of
a shape the set does not name - and it is not a security control (D-022).
"""

from __future__ import annotations

import ast
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pytest

from distill import emit
from distill.emit import EMITTER, TextEmitter

PACKAGE_DIR = Path(__file__).resolve().parents[1] / "src" / "distill"

# The characters a mode string uses to ask for a write, and the alphabet a mode
# string is drawn from at all. Both, because "mode" is matched by shape: a
# string argument is only read as a mode when it is short and built from these
# letters, so `Image.open("waterfall.png")` is a path and `path.open("w")` is a
# write.
WRITE_MODE_CHARACTERS = frozenset("wax+")
MODE_ALPHABET = frozenset("rwxabt+")
MAX_MODE_LENGTH = 3

# Calls that return a writable file object from a mode string.
OPEN_FAMILY = frozenset({"open", "fdopen"})

# The shapes the docstring names as putting something on disk while leaving this
# check silent, exactly as it writes them. Read back out of `__doc__` by
# `test_the_docstring_lists_exactly_the_shapes_this_file_proves_undetected`, so
# the paragraph and these cases cannot drift apart in either direction.
UNDETECTED_SHAPES = (
    "os.pwrite(fd, text.encode(), 0)",
    "os.writev(fd, [text.encode()])",
    "shutil.copyfile(src, dst)",
    "shutil.copyfileobj(reader, writer)",
    "tempfile.NamedTemporaryFile(delete=False)",
    'tempfile.NamedTemporaryFile(mode="w")',
    "print(text, file=handle)",
    "pickle.dump(document, handle)",
    "os.symlink(target, name)",
    "handle.writelines(lines)",
    "os.open(path, 577)",
)

_UNDETECTED_BLOCK_OPENS = "on disk and leaves this check silent:"
_UNDETECTED_BLOCK_CLOSES = "\nThose "

_NUMBER_WORDS = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
)

# Flags that make an `os.open` a write. `O_CREAT` counts on its own: a call that
# may create the file is a call that may put something on disk.
OS_WRITE_FLAGS = ("O_WRONLY", "O_RDWR", "O_CREAT", "O_APPEND", "O_TRUNC")

@dataclass(frozen=True)
class Licence:
    """The exact writes one function may make outside `TextEmitter`, and why.

    `forms` is a multiset and not a set: a function licensed for one `os.open`
    is licensed for *one*, and a second appearing beside it is a finding. That
    is the difference between licensing a decision and licensing a location -
    a licence keyed only on the function would let any later writer of any
    named form move in under a reason written about something else.
    """

    forms: tuple[str, ...]
    reason: str


# Durable writers licensed to exist outside `TextEmitter`, keyed by the module
# and the qualified name of the function they sit in. An entry is a recorded
# reason and the writes that reason was recorded about, not a note of where a
# writer happens to be: an entry without a reason is the same defect as an
# unreasoned signature exemption, and an entry without counts is a licence for
# whatever is written in that function next.
ALLOWED_WRITERS: dict[tuple[str, str], Licence] = {
    ("emit.py", "TextEmitter.emit"): Licence(
        forms=("fdopen(mode)",),
        reason=(
            "The emitter itself, ordinary-write discipline. This is the call every "
            "other durable write in Distill resolves to, so it is the one place the "
            "check exists to concentrate writes into rather than to forbid them. One "
            "form and not two because the `os.open` it wraps carries its flags in "
            "`_WRITE_FLAGS`, which blind spot 1 makes invisible here."
        ),
    ),
    ("emit.py", "TextEmitter.emit_atomically"): Licence(
        forms=("fdopen(mode)",),
        reason=(
            "The emitter itself, atomic-replace discipline. Two disciplines and not "
            "one because a target another process may read cannot be written in "
            "place (R-14, D-033); both are the emitter, and both are the point."
        ),
    ),
    ("bundle_store.py", "ExclusiveLock.take"): Licence(
        forms=("os.open",),
        reason=(
            "Writes no text. `O_CREAT | O_RDWR` creates the file `flock` is taken "
            "on and nothing is ever written into it - the lock is the descriptor, "
            "and the file exists only so the kernel has something to hold. Routing "
            "it through the emitter would be incoherent as well as useless: the "
            "emitter closes the descriptor it writes through, and closing this one "
            "is what releasing the lock means."
        ),
    ),
    ("bundle_store.py", "ExclusiveLock.probe"): Licence(
        forms=("os.open",),
        reason=(
            "Writes no text, on the same terms, and deliberately does not create: "
            "`O_RDWR` without `O_CREAT` is what lets a read-only inspection ask "
            "whether a lock is held without leaving a lock file behind (R-57)."
        ),
    ),
}


@dataclass(frozen=True)
class DurableWrite:
    """One call that puts text on disk, and where it was found."""

    module: str
    qualname: str
    form: str
    line: int

    def __str__(self) -> str:
        return f"{self.module}:{self.line} in {self.qualname} ({self.form})"


def _asks_to_write(node: ast.Call) -> bool:
    """Whether any argument of `node` is a mode string that asks for a write.

    Matched by shape rather than by position, because the slot is not stable:
    `open(file, mode)` and `os.fdopen(fd, mode)` take it second, `Path.open(mode)`
    first, and a `mode=` keyword takes it anywhere. Reading whichever argument
    the call shape predicts means a call shape nobody predicted is a miss, and a
    mode is unmistakable enough not to need the position: it is at most three
    characters drawn from `rwxabt+`. A string that is not that is a path, a
    format name or an encoding, and reading it as a mode is how a check like
    this starts reporting `Image.open("waterfall.png")`.

    The shape rule is not exact, and the residue is deliberate: `open("w")`,
    reading a file whose name is one letter, is reported as a write. That way
    round on purpose - a false positive is loud, arrives the moment the line is
    written and is answered by one licence entry, while a false negative is a
    durable writer nobody ever hears about.
    """
    for argument in [*node.args, *(kw.value for kw in node.keywords if kw.arg == "mode")]:
        if not isinstance(argument, ast.Constant) or not isinstance(argument.value, str):
            continue
        mode = argument.value
        if not mode or len(mode) > MAX_MODE_LENGTH or not set(mode) <= MODE_ALPHABET:
            continue
        if set(mode) & WRITE_MODE_CHARACTERS:
            return True
    return False


def _is_module_call(func: ast.Attribute, module: str) -> bool:
    return isinstance(func.value, ast.Name) and func.value.id == module


def _write_flags(node: ast.Call) -> bool:
    """Whether an `os.open` call's flags ask for anything but a read.

    Read from the flag expression as written, positional or `flags=` keyword.
    A call that passes them in a variable says nothing here, which is blind
    spot 1 rather than a hole this could close: the value would have to be
    followed, and following names is what an AST check does not do.
    """
    positional = node.args[1] if len(node.args) > 1 else None
    keyword = next((kw.value for kw in node.keywords if kw.arg == "flags"), None)
    flags = " ".join(ast.unparse(node) for node in (positional, keyword) if node is not None)
    return any(flag in flags for flag in OS_WRITE_FLAGS)


def _write_form(node: ast.Call) -> str | None:
    """Name the durable-write form of `node`, or `None` if it is not one."""
    func = node.func
    if isinstance(func, ast.Attribute):
        name = func.attr
        if name in {"write_text", "write_bytes"}:
            return name
        if name == "dump" and _is_module_call(func, "json"):
            return "json.dump"
        if _is_module_call(func, "os"):
            if name == "write":
                return "os.write"
            if name == "open":
                return "os.open" if _write_flags(node) else None
    elif isinstance(func, ast.Name):
        name = func.id
    else:
        return None
    if name in OPEN_FAMILY:
        return f"{name}(mode)" if _asks_to_write(node) else None
    return None


class _DurableWriteVisitor(ast.NodeVisitor):
    """Collect durable-write calls with the qualified name they sit in."""

    def __init__(self, module: str) -> None:
        self.module = module
        self.found: list[DurableWrite] = []
        self._scope: list[str] = []

    def _in_scope(self, node: ast.AST) -> None:
        self._scope.append(getattr(node, "name", "?"))
        self.generic_visit(node)
        self._scope.pop()

    visit_ClassDef = _in_scope
    visit_FunctionDef = _in_scope
    visit_AsyncFunctionDef = _in_scope

    def visit_Call(self, node: ast.Call) -> None:
        form = _write_form(node)
        if form is not None:
            self.found.append(
                DurableWrite(
                    module=self.module,
                    qualname=".".join(self._scope) or "<module>",
                    form=form,
                    line=node.lineno,
                )
            )
        self.generic_visit(node)


def find_durable_writes(source: str, module: str) -> tuple[DurableWrite, ...]:
    """Every durable write in `source` this check can see, named for `module`.

    Pure over the text, so the negative controls can hand it a module that does
    not exist rather than writing a naive writer into the package under test and
    trusting a later line to take it out again.
    """
    visitor = _DurableWriteVisitor(module)
    visitor.visit(ast.parse(source))
    return tuple(visitor.found)


def _package_durable_writes() -> tuple[DurableWrite, ...]:
    """Every durable write this check can see in the modules actually on disk."""
    found: list[DurableWrite] = []
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        relative = path.relative_to(PACKAGE_DIR)
        if "__pycache__" in relative.parts:
            continue
        found.extend(find_durable_writes(path.read_text(), relative.as_posix()))
    return tuple(found)


def _forms_by_site(writes: Iterable[DurableWrite]) -> dict[tuple[str, str], Counter[str]]:
    """The multiset of write forms each site makes, keyed as a licence is."""
    by_site: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for write in writes:
        by_site[(write.module, write.qualname)][write.form] += 1
    return by_site


def _lines(writes: Iterable[DurableWrite], site: tuple[str, str], form: str) -> str:
    """Where a site's writes of one form are, for a message that can be acted on."""
    found = [write for write in writes if (write.module, write.qualname) == site]
    numbers = sorted(write.line for write in found if write.form == form)
    return f" (line{'s' if len(numbers) > 1 else ''} {', '.join(str(n) for n in numbers)})"


def unlicensed_writes(writes: Iterable[DurableWrite]) -> tuple[str, ...]:
    """Every write in `writes` beyond what `ALLOWED_WRITERS` accounts for.

    Beyond, counted rather than merely named. A function licensed for one write
    that makes two is over its licence by one, and reported: the licence
    describes a decision someone recorded about specific calls, and a call that
    was not among them was not part of the decision.

    Reported per site and form with the count on both sides, because with a
    repeated form there is no such thing as *which* call is the surplus one -
    the site holds more of that form than anyone licensed, and the lines are
    given so a reader can see all of them and decide which is new.
    """
    writes = tuple(writes)
    surplus: list[str] = []
    for site, found in sorted(_forms_by_site(writes).items()):
        licensed = Counter(ALLOWED_WRITERS[site].forms) if site in ALLOWED_WRITERS else Counter()
        for form in sorted(found - licensed):
            module, qualname = site
            surplus.append(
                f"{module} {qualname}: {found[form]} {form} write"
                f"{'s' if found[form] > 1 else ''}, {licensed[form]} licensed"
                f"{_lines(writes, site, form)}"
            )
    return tuple(surplus)


def unwritten_licences(writes: Iterable[DurableWrite]) -> tuple[str, ...]:
    """Every licensed write `writes` no longer contains."""
    by_site = _forms_by_site(writes)
    absent: list[str] = []
    for site, licence in sorted(ALLOWED_WRITERS.items()):
        found = by_site.get(site, Counter())
        licensed = Counter(licence.forms)
        for form in sorted(licensed - found):
            module, qualname = site
            absent.append(
                f"{module} {qualname}: {licensed[form]} {form} write"
                f"{'s' if licensed[form] > 1 else ''} licensed, {found[form]} written"
            )
    return tuple(absent)


def test_every_durable_write_is_the_emitter_or_licensed() -> None:
    """No module writes text to disk in a form this check sees, but the emitter.

    The finding this stands against is not one bad writer; it is that durable
    writes were spread across modules with no single place to ask anything of
    them, so a **redaction sink** could be added without anyone noticing there
    was a set to be complete over (findings 4, 15).
    """
    unlicensed = unlicensed_writes(_package_durable_writes())

    assert not unlicensed, (
        "durable writers outside TextEmitter: "
        + "; ".join(unlicensed)
        + ". Emit through distill.emit.EMITTER, or add the site to "
        "ALLOWED_WRITERS with the reason it is legitimate."
    )


def test_a_second_writer_inside_a_licensed_function_is_reported() -> None:
    """A licence covers the writes it names, not the function they sit in.

    This is the ordinary way the check exists to catch a bypass: a contributor
    adds a durable write of a form the check names, following the codebase's
    patterns, inside a function that already holds one. `ExclusiveLock.take` is
    licensed because it creates a lock file and writes no text into it; a
    `write_text` added beside that `os.open` puts text on disk without reaching
    the emitter, and a licence keyed on the function alone waves it through.
    """
    source = (
        "class ExclusiveLock:\n"
        "    def take(self, path):\n"
        "        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)\n"
        "        path.write_text('held')\n"
        "        return fd\n"
    )

    found = find_durable_writes(source, "bundle_store.py")

    assert ("bundle_store.py", "ExclusiveLock.take") in ALLOWED_WRITERS
    assert [write.form for write in found] == ["os.open", "write_text"]
    assert unlicensed_writes(found) == (
        "bundle_store.py ExclusiveLock.take: 1 write_text write, 0 licensed (line 4)",
    )


def test_a_second_writer_of_a_licensed_form_is_reported() -> None:
    """A licence is a count, so the second write of a licensed form is surplus.

    The sharper half of the same finding, and the one a licence keyed on
    `(site, form)` would still let through. `ExclusiveLock.take` is licensed for
    *one* `os.open`, for a reason recorded about a lock file that holds no text;
    a second `os.open` in the same function is a different file opened for a
    different purpose, and the recorded reason says nothing about it.
    """
    source = (
        "class ExclusiveLock:\n"
        "    def take(self, path, report):\n"
        "        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)\n"
        "        os.close(os.open(report, os.O_WRONLY | os.O_CREAT))\n"
        "        return fd\n"
    )

    found = find_durable_writes(source, "bundle_store.py")

    assert [write.form for write in found] == ["os.open", "os.open"]
    assert unlicensed_writes(found) == (
        "bundle_store.py ExclusiveLock.take: 2 os.open writes, 1 licensed (lines 3, 4)",
    )


def test_the_licence_list_names_only_writes_that_exist() -> None:
    """A licence for a write that is gone is a licence nobody reviewed.

    Stale entries are how an allowlist stops describing the code: the site moves
    or the writer is routed through the emitter, and the entry stays behind
    licensing whatever is later written in a function of that name. Held to the
    form and the count, not just the site, so a licence cannot outlive the call
    it was written about while its function survives.

    Together with `test_every_durable_write_is_the_emitter_or_licensed` this
    makes the two multisets equal, which is the property: the licence list is
    not a floor the code may exceed nor a ceiling it may fall short of, it is a
    statement of what the package does.
    """
    absent = unwritten_licences(_package_durable_writes())

    assert not absent, (
        "ALLOWED_WRITERS licenses writes that are no longer made: "
        + "; ".join(absent)
        + ". Remove the entry, or correct its forms to the calls that remain."
    )


def test_every_licence_records_a_reason() -> None:
    """A licence without a stated reason is not a recorded decision."""
    unexplained = sorted(
        key for key, licence in ALLOWED_WRITERS.items() if not licence.reason.strip()
    )

    assert not unexplained, f"licensed writers without a recorded reason: {unexplained}"


def test_every_licence_names_the_writes_it_covers() -> None:
    """A licence naming no form licenses nothing and reads as though it does."""
    empty = sorted(key for key, licence in ALLOWED_WRITERS.items() if not licence.forms)

    assert not empty, f"licensed writers naming no write form: {empty}"


def test_a_naive_write_text_outside_the_licence_list_is_detected() -> None:
    """The check catches the write a contributor is most likely to reach for.

    `Path.write_text` is what the codebase looked like before the emitter and
    what an editor's completion offers first, so it is the form this check has
    to catch or catch nothing worth catching.
    """
    source = "def write_render(path, markdown):\n    path.write_text(markdown)\n"

    found = find_durable_writes(source, "render.py")

    assert [(write.qualname, write.form) for write in found] == [("write_render", "write_text")]
    assert ("render.py", "write_render") not in ALLOWED_WRITERS


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("def save(path, blob):\n    path.write_bytes(blob)\n", "write_bytes"),
        ("def save(path, text):\n    open(path, 'w').write(text)\n", "open(mode)"),
        ("def save(path, text):\n    with open(path, mode='a') as f:\n        f.write(text)\n", "open(mode)"),
        ("def save(path, text):\n    with path.open('w') as f:\n        f.write(text)\n", "open(mode)"),
        ("def save(fd, text):\n    with os.fdopen(fd, 'w') as f:\n        f.write(text)\n", "fdopen(mode)"),
        ("def save(path):\n    return os.open(path, os.O_WRONLY | os.O_CREAT)\n", "os.open"),
        ("def save(path):\n    return os.open(path, flags=os.O_WRONLY | os.O_CREAT)\n", "os.open"),
        ("def save(fd, text):\n    os.write(fd, text.encode())\n", "os.write"),
        ("def save(path, text):\n    with gzip.open(path, 'wt') as f:\n        f.write(text)\n", "open(mode)"),
        ("def save(document, handle):\n    json.dump(document, handle)\n", "json.dump"),
    ],
)
def test_the_other_write_forms_are_detected_equivalently(source: str, expected: str) -> None:
    """Each form is a way to put text on disk, so each is the same finding.

    Detecting only `write_text` would make the check a style rule about one
    method rather than a statement about durable writes, and the next writer
    would be spelled one of these.
    """
    found = find_durable_writes(source, "somewhere.py")

    assert [write.form for write in found] == [expected]
    assert [write.qualname for write in found] == ["save"]


@pytest.mark.parametrize(
    "source",
    [
        # A read through the same method name.
        "def load(path):\n    return path.open('rb').read()\n",
        # Pillow's `Image.open`, whose first argument is a path and not a mode -
        # including one whose name is spelled from mode letters.
        "def frame(path):\n    return Image.open(path)\n",
        "def frame():\n    return Image.open('waterfall.png')\n",
        # `os.open` for reading, which is how the no-follow reads are written.
        "def load(path):\n    return os.open(path, os.O_RDONLY | os.O_NOFOLLOW)\n",
        # Serializing to a string is not emitting it.
        "def payload(document):\n    return json.dumps(document, sort_keys=True)\n",
        # Reading text is not writing it.
        "def load(path):\n    return path.read_text()\n",
    ],
)
def test_reads_and_lookalikes_are_not_reported_as_writes(source: str) -> None:
    """A check that flags reads gets its findings waved through.

    The cost of a false positive here is not noise, it is the allowlist filling
    with entries that license nothing, at which point the licence list stops
    being a list of durable writers.
    """
    assert find_durable_writes(source, "somewhere.py") == ()


_SHAPE_ARGUMENTS = "fd, text, path, src, dst, reader, writer, handle, lines, document, target, name"
"""Every name the documented shapes use, so one signature parses all of them."""


def _shape_source(shape: str) -> str:
    """One documented shape as the body of a module this check would parse."""
    return f"def save({_SHAPE_ARGUMENTS}):\n    {shape}\n"


def _documented_undetected_shapes() -> tuple[str, ...]:
    """The shapes the module docstring lists as escaping this check, as written.

    Read out of `__doc__` rather than restated here, because a list restated is
    a list that drifts: the paragraph is what a reader trusts, so the paragraph
    is what has to be tested. Two columns per line, separated by a run of
    spaces, which is how the block is laid out to be read.
    """
    doc = __doc__ or ""
    block = doc.split(_UNDETECTED_BLOCK_OPENS)[1].split(_UNDETECTED_BLOCK_CLOSES)[0]
    return tuple(
        entry
        for line in block.splitlines()
        for entry in re.split(r"\s{2,}", line.strip())
        if entry
    )


def test_the_docstring_lists_exactly_the_shapes_this_file_proves_undetected() -> None:
    """The paragraph and the cases below it are one list, not two that agree.

    The failure this stands against is the quiet one: a shape added to the prose
    that nobody ever asserted, or a case removed from the cases while the prose
    still promises it. Either leaves a reader trusting a sentence no test holds
    up, which is the exact defect the statement of limits exists to prevent
    (D-022). The count is checked too, because "those eleven" is a claim.
    """
    assert _documented_undetected_shapes() == UNDETECTED_SHAPES

    doc = __doc__ or ""
    counted = doc.split(_UNDETECTED_BLOCK_CLOSES)[1].split(" are named")[0]
    assert counted == _NUMBER_WORDS[len(UNDETECTED_SHAPES)]


@pytest.mark.parametrize("shape", UNDETECTED_SHAPES)
def test_the_shapes_documented_as_undetected_are_undetected(shape: str) -> None:
    """The module docstring's statement of what escapes this check is true of it.

    Every one of these puts something durable on disk, written plainly and
    unaliased, in a module this check parses - and none is reported, because the
    check recognises an enumerated set of call shapes and these are not in it.
    The docstring says exactly that and names exactly these; this test is what
    stops the two from drifting apart, which is the whole failure the statement
    exists to prevent (D-022, R-22).

    So it is not a bar on widening the check. Adding one of these shapes to
    `_write_form` is welcome and will fail here, at which point the docstring is
    corrected in the same change - the one thing that must not happen is the set
    moving while the paragraph a reader trusts stays where it was.
    """
    assert find_durable_writes(_shape_source(shape), "somewhere.py") == ()


def test_the_qualified_name_locates_a_writer_inside_a_class() -> None:
    """The licence key is module plus qualified name, so the name has to nest.

    Two methods called `write` in two classes are two writers; keying on the
    bare function name would let a licence for one cover the other.
    """
    source = (
        "class Store:\n"
        "    class Inner:\n"
        "        def write(self, path, text):\n"
        "            path.write_text(text)\n"
    )

    found = find_durable_writes(source, "store.py")

    assert [write.qualname for write in found] == ["Store.Inner.write"]


def test_emit_writes_the_text_it_is_given(tmp_path: Path) -> None:
    target = tmp_path / "video.md"

    written = EMITTER.emit(target, "# Render\n")

    assert written == target
    assert target.read_text() == "# Render\n"


def test_emit_overwrites_rather_than_appends(tmp_path: Path) -> None:
    """A re-emitted target holds the new text and nothing of the old.

    A **stage result** is rewritten by the run that recomputes it, and an
    append would leave the second document trailing the first - unparseable,
    and unparseable in a way a **resume** would have to discover.
    """
    target = tmp_path / "_ocr.json"
    EMITTER.emit(target, "a longer first document\n")

    EMITTER.emit(target, "{}\n")

    assert target.read_text() == "{}\n"


def test_emit_refuses_a_symlink_at_the_target(tmp_path: Path) -> None:
    """The kernel decides, at the open, not a check that ran beforehand (R-16).

    Confinement is asked of the path before the emitter is called, and a check
    goes out of date the moment it returns: between it and the open, the name
    can be replaced with a link. `O_NOFOLLOW` is that refusal at the moment of
    use, so the write fails rather than landing wherever the link points.
    """
    outside = tmp_path / "elsewhere.md"
    outside.write_text("untouched\n")
    target = tmp_path / "video.md"
    target.symlink_to(outside)

    with pytest.raises(OSError):
        EMITTER.emit(target, "redirected\n")

    assert outside.read_text() == "untouched\n"


def test_emit_atomically_replaces_the_target_and_leaves_no_temporary(tmp_path: Path) -> None:
    target = tmp_path / "_manifest.json"
    EMITTER.emit_atomically(target, "old\n")

    EMITTER.emit_atomically(target, "new\n")

    assert target.read_text() == "new\n"
    assert [path.name for path in tmp_path.iterdir()] == ["_manifest.json"]


def test_emit_atomically_leaves_the_old_bytes_when_the_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed write is a target that never changed, and no litter.

    The point of the discipline is that a reader sees the old bytes or the new
    ones; a writer that failed holding a half-written temporary must not leave
    it where the next reader or the next **prune** has to reason about it. The
    failure is planted at the flush, which is inside the write and before the
    replace - the one window in which a temporary exists with nothing yet
    depending on it.
    """
    target = tmp_path / "_manifest.json"
    EMITTER.emit_atomically(target, "old\n")

    def refuse(_fd: int) -> None:
        raise OSError("no space left on device")

    monkeypatch.setattr(emit.os, "fsync", refuse)

    with pytest.raises(OSError):
        EMITTER.emit_atomically(target, "new\n")

    assert target.read_text() == "old\n"
    assert [path.name for path in tmp_path.iterdir()] == ["_manifest.json"]


def test_two_emitters_in_one_process_do_not_share_a_temporary_name(tmp_path: Path) -> None:
    """The temporary is unique per write, not per emitter.

    A second emitter constructed in the same process must not be able to
    truncate the first's temporary: that publishes a half-written file onto the
    target, which is worse than having no temporary at all. The sequence is
    therefore shared by every instance rather than held per object.
    """
    first = TextEmitter()
    second = TextEmitter()

    names = {
        first.temporary_for(tmp_path / "_manifest.json"),
        second.temporary_for(tmp_path / "_manifest.json"),
    }

    assert len(names) == 2
