# Project Guidelines

## Python Commands

- Run inline Python snippets with `uv run python -c '...'` so they use the
  project's selected interpreter and environment.
- Use `uv run` without `--no-managed-python` for local commands; that flag is
  intended for CI.

## Validation After Changes

- After each coherent change, run `uvx prek run --files <changed-paths>`,
  including new files. For staged changes, use `uvx prek run`. This runs the
  applicable checks in `prek.toml`: file hygiene, Ruff linting and formatting,
  ty type checking, deptry dependency checking, and rumdl Markdown linting.
- If a hook modifies files, review the edits and rerun prek until it passes.
  Keep the protected-branch check enabled; work on a feature branch.
- For Python or behavior changes, run the relevant tests with
  `uv run pytest -q <test-paths>` during development, then run
  `uv run pytest -q` before finishing. Documentation-only changes do not
  require pytest.
- Fix failures introduced by your changes. Report unrelated existing failures
  or unavailable checks explicitly; do not silently skip them or claim they
  passed.
