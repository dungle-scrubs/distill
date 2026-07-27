"""Unit tests for the stated optional/required capability table."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from distill.capabilities import (
    EXTERNAL_TOOLS,
    Requirement,
    missing_tool_consequence,
    missing_tool_error,
    missing_tool_warning,
)
from distill.errors import DistillError

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src" / "distill"
README = REPO_ROOT / "README.md"

# The names `run_command` exports for invoking a tool. A call to one of these is
# where a tool name enters the process.
HELPER_NAMES = frozenset({"run", "stream", "run_json"})


class ArgvHeads:
    """Resolves an argv expression to the tool names it can name, in one module.

    Deliberately coarse and module-scoped: a name resolves from every assignment
    to it and every argument bound to a parameter of that name anywhere in the
    module, which over-collects rather than under-collects. That is the right
    error to make here - an extra candidate fails the test loudly and a missed
    one is the unclassified tool D-010 forbids.

    It answers only "which literal tool names can reach this argv's first
    element". It is not a call graph, does not follow imports, and gives no
    answer for a head built at runtime - such a call site resolves to nothing,
    which the test reports as a call site that must name its tool literally.
    """

    def __init__(self, tree: ast.Module) -> None:
        self._assigned: dict[str, list[ast.expr]] = {}
        self._bound: dict[str, list[ast.expr]] = {}
        self._functions: dict[str, ast.FunctionDef] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                self._functions[node.name] = node
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self._assigned.setdefault(target.id, []).append(node.value)
            elif (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.value is not None
            ):
                self._assigned.setdefault(node.target.id, []).append(node.value)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                self._bind_arguments(node)

    def _bind_arguments(self, call: ast.Call) -> None:
        """Record what each parameter of a module-level function is called with."""
        function = self._functions.get(getattr(call.func, "id", ""))
        if function is None:
            return
        parameters = [argument.arg for argument in function.args.args]
        for name, value in zip(parameters, call.args, strict=False):
            self._bound.setdefault(name, []).append(value)
        for keyword in call.keywords:
            if keyword.arg:
                self._bound.setdefault(keyword.arg, []).append(keyword.value)

    def of(self, argv: ast.expr) -> set[str]:
        """The tool names this argv expression's first element can take."""
        return {name for name in self._resolve(argv, set()) if "/" not in name and name}

    def _resolve(self, node: ast.expr, seen: set[int]) -> set[str]:
        if id(node) in seen:
            return set()
        seen.add(id(node))
        if isinstance(node, ast.Constant):
            return {node.value} if isinstance(node.value, str) else set()
        if isinstance(node, ast.List | ast.Tuple):
            return self._resolve(node.elts[0], seen) if node.elts else set()
        if isinstance(node, ast.Name):
            return self._union(
                self._assigned.get(node.id, []) + self._bound.get(node.id, []), seen
            )
        if isinstance(node, ast.BoolOp):
            return self._union(node.values, seen)
        if isinstance(node, ast.IfExp):
            return self._union([node.body, node.orelse], seen)
        if isinstance(node, ast.Call):
            return self._resolve_call(node, seen)
        return set()

    def _resolve_call(self, node: ast.Call, seen: set[int]) -> set[str]:
        # `shutil.which("tesseract")` names the tool as surely as an argv head
        # does; what it returns is the located path, which `of` discards.
        if isinstance(node.func, ast.Attribute) and node.func.attr == "which":
            return self._union(node.args[:1], seen)
        function = self._functions.get(getattr(node.func, "id", ""))
        if function is None:
            return set()
        return self._union(
            [
                statement.value
                for statement in ast.walk(function)
                if isinstance(statement, ast.Return) and statement.value is not None
            ],
            seen,
        )

    def _union(self, nodes: list[ast.expr], seen: set[int]) -> set[str]:
        return set().union(*(self._resolve(node, seen) for node in nodes), set())


def invoked_tools() -> dict[str, set[str]]:
    """Every `run_command` call site in the package, and the tools it can invoke.

    Keyed by `module.py:line` so a call site whose tool cannot be resolved names
    itself in the failure.
    """
    found: dict[str, set[str]] = {}
    for path in sorted(SRC_ROOT.rglob("*.py")):
        if path.name == "run_command.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = {
            alias.asname or alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and (node.module or "").endswith("run_command")
            for alias in node.names
            if alias.name in HELPER_NAMES
        }
        heads = ArgvHeads(tree)
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in imported
            ):
                continue
            argv = next(
                (
                    keyword.value
                    for keyword in node.keywords
                    if keyword.arg == "argv"
                ),
                node.args[0] if node.args else None,
            )
            found[f"{path.name}:{node.lineno}"] = (
                heads.of(argv) if argv is not None else set()
            )
    return found


def test_every_external_tool_distill_invokes_is_classified() -> None:
    assert set(EXTERNAL_TOOLS) == {"ffmpeg", "ffprobe", "tesseract", "yt-dlp"}
    assert EXTERNAL_TOOLS["tesseract"].requirement is Requirement.OPTIONAL
    assert all(
        EXTERNAL_TOOLS[name].requirement is Requirement.REQUIRED
        for name in ("ffmpeg", "ffprobe", "yt-dlp")
    )


def test_every_classification_states_what_its_absence_costs() -> None:
    for name, tool in EXTERNAL_TOOLS.items():
        assert tool.name == name
        assert tool.capability
        assert tool.invoked_when
        assert len(tool.absence_cost.split()) >= 5, name


def test_no_call_site_invokes_a_tool_the_table_omits() -> None:
    """A tool invoked but unclassified is the gap D-010 forbids.

    The set is derived from the argv every call site passes, not from a list of
    tool names the test already knows. A fixed list cannot notice a tool nobody
    thought of - adding `run(["curl", ...])` would leave it unchanged and this
    test green - so the head of each argv reaching `run`, `stream` or `run_json`
    is resolved instead, and a new tool arrives in the set whether or not
    anyone remembers this file.

    Equality, not containment: a classified tool no call site invokes is a
    promise the README repeats and nothing keeps.
    """
    invoked = invoked_tools()
    unresolved = sorted(site for site, tools in invoked.items() if not tools)

    assert unresolved == [], (
        "the tool these call sites invoke could not be read off their argv; "
        "name it with a literal so it can be classified: " + ", ".join(unresolved)
    )
    assert set().union(*invoked.values()) == set(EXTERNAL_TOOLS)


def test_optional_tool_absence_degrades_with_a_warning() -> None:
    result = missing_tool_warning("ocr", "tesseract")

    assert result["stage"] == "ocr"
    assert result["code"] == "tesseract_not_found"
    assert result["message"].startswith("tesseract is not installed or not on PATH; ")


def test_required_tool_absence_is_not_allowed_to_degrade() -> None:
    """ADR-0002 cuts both ways: a required capability must not warn and continue."""
    with pytest.raises(ValueError, match="required capability"):
        missing_tool_warning("source", "ffprobe")


def test_warning_code_is_snake_case_for_a_hyphenated_tool() -> None:
    assert EXTERNAL_TOOLS["yt-dlp"].warning_code == "yt_dlp_not_found"


def test_the_consequence_of_an_absent_optional_tool_is_its_warning() -> None:
    """The one entry point a call site uses, answering for an optional tool."""
    result = missing_tool_consequence("ocr", "tesseract")

    assert result["code"] == "tesseract_not_found"
    assert EXTERNAL_TOOLS["tesseract"].absence_cost in result["message"]


def test_the_consequence_of_an_absent_required_tool_is_a_fatal_error() -> None:
    """ADR-0002 / R-34: a required capability's absence never returns a warning.

    It raises under the missing-tool code, naming the tool and stating what its
    absence costs, so a run that cannot produce a **bundle** stops at the tool
    rather than at the render that finds nothing to write.
    """
    with pytest.raises(DistillError) as failure:
        missing_tool_consequence("frames", "ffmpeg")

    assert failure.value.code == "E_MISSING_TOOL"
    assert failure.value.stage == "frames"
    assert "ffmpeg" in failure.value.message
    assert EXTERNAL_TOOLS["ffmpeg"].absence_cost in failure.value.message
    assert failure.value.details["requirement"] == "required"


def test_the_fatal_error_carries_the_invocation_that_failed() -> None:
    """The failing invocation's payload survives, so the run stays traceable."""
    cause = DistillError(
        "E_MISSING_TOOL",
        "frames",
        "required tool is not installed: ffmpeg",
        {"argv": ["ffmpeg", "-y"], "tool": "ffmpeg"},
    )

    error = missing_tool_error("frames", "ffmpeg", cause=cause)

    assert error.details["argv"] == ["ffmpeg", "-y"]
    assert error.details["tool"] == "ffmpeg"


def test_an_optional_tool_absence_is_not_allowed_to_end_a_run() -> None:
    """ADR-0002 cuts both ways: an optional capability must not raise."""
    with pytest.raises(ValueError, match="optional capability"):
        missing_tool_error("ocr", "tesseract")


def readme_dependency_rows() -> dict[str, tuple[str, str]]:
    """Tool -> (capability, class), read off the README's four-column table."""
    rows: dict[str, tuple[str, str]] = {}
    for line in README.read_text(encoding="utf-8").splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 4 or not cells[0].startswith("`"):
            continue
        rows[cells[0].strip("`")] = (cells[1], cells[2])
    return rows


def test_the_readme_states_the_class_the_table_records() -> None:
    """D-022: the docstring's claim about the README is backed by this test.

    Nothing renders the README's system-dependency table from `EXTERNAL_TOOLS` -
    it is prose, written by hand - so the only thing keeping the promise it
    makes about degradation aligned with the code is this assertion. A tool
    reclassified here without the README changing with it fails the suite.
    """
    rows = readme_dependency_rows()

    for name, tool in EXTERNAL_TOOLS.items():
        assert name in rows, f"{name} is classified but the README does not list it"
        capability, requirement = rows[name]
        assert capability == tool.capability, name
        assert requirement == str(tool.requirement), name
