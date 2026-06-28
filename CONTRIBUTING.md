# Contributing to Distill

Thanks for your interest in improving Distill. This is a small project; keep
changes focused and the working tree green.

## Setup

```bash
git clone https://github.com/dungle-scrubs/distill.git
cd distill
uv sync
uv run pytest          # should pass before you start
```

## Local vision

Distill talks to a local Rapid-MLX server. For day-to-day work you do **not**
need it running — the test suite fakes it by monkeypatching
`distill.local_vision._urlopen_json`, so `uv run pytest` is fully hermetic. The
live smoke test is gated behind `DISTILL_RUN_RAPID_MLX_SMOKE=1`.

## Before opening a PR

1. `uv run pytest` — all tests pass.
2. `uv run ruff check .` — no lint errors.
3. If you changed any output-affecting module (`local_vision.py`, `options.py`,
   `pipeline.py`, or the other signed modules), recompute `PIPELINE_SIGNATURE`
   as described in [AGENTS.md](AGENTS.md) and bump `PIPELINE_VERSION`, so
   `tests/test_pipeline_signature.py` stays green.

## Branching

The project keeps a single branch, `main`. Land changes directly on `main`; do
not open long-lived feature branches.

## License

By contributing you agree your changes will be released under the project's
[MIT license](LICENSE).
