# python-template

Default template for python projects. Includes: uv, ruff, basedpyright, pytest, pytest-cov, prek and semantic-release.

## Replace this readme with the project's readme

## New project from this template

```bash
gh repo create my-project --private --template abouganemi/python-template --clone
cd my-project
```

Then rename the package:

1. `pyproject.toml`: set `name` and `description`
2. `git mv src/python_template src/my_project` (underscores, matching `name`)
3. `tests/test_import.py`: update the import
4. `uv lock`

## Notes

- [`prek`](https://github.com/j178/prek) runs the checks in `.pre-commit-config.yaml`:
  - `ruff` formats and lints the code, including import sorting and Google-style docstrings (`D` rules, not applied to `tests/`).
  - `conventional-pre-commit` validates the commit message.
  - `uv-lock` keeps `uv.lock` in sync with `pyproject.toml`.
  - `builtin` fixes whitespace and end of files and checks YAML/TOML, merge conflicts, large files and private keys.
- The hooks run through the global git hooks from `shell-configs` (`core.hooksPath`), so there is no per-repo `prek install` (prek refuses to install while `core.hooksPath` is set).
- `pyproject.toml` configures `ruff`, `basedpyright`, `pytest` and `coverage`; move it next to the code if that is not the repository root.
- `.python-version` pins the Python version uv uses.
- `pytest` measures branch coverage of `src/` and fails below 90%.

## Package management

Packages are managed with [uv](https://docs.astral.sh/uv/): `uv add <package>` (or `uv add --dev <package>`) updates `pyproject.toml` and `uv.lock`.

## CLI and API

- `CLI`s are built with [typer](https://typer.tiangolo.com/)
- `API`s are built with [fastapi](https://fastapi.tiangolo.com/)

## Setup

### 1. Install uv

Linux/macOS:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

macOS also has a Homebrew option:

```bash
brew install uv
```

Keep it up to date with:

```bash
uv self update
```

### 2. Python version

`.python-version` pins the Python version this project uses (currently 3.14). uv downloads that version automatically the first time it is needed, so you don't need it pre-installed. To do it explicitly, or to change the pinned version:

```bash
uv python install
uv python pin <version>
```

### 3. Create the virtual environment

```bash
uv sync --all-groups --locked
```

This creates `.venv/` in the project directory and installs the project plus the `dev` dependency group (`pytest`, `pytest-cov`, `ruff`, `basedpyright`, `prek`). `--all-groups` includes every dependency group, not just the default ones; `--locked` installs exactly what's in `uv.lock` and fails instead of updating it if `pyproject.toml` and the lockfile disagree.

### 4. Using the environment

Prefer `uv run <command>`, which uses the project's `.venv` without activating it:

```bash
uv run pytest
uv run python    # Python shell with the project installed
```

You can also activate the environment directly if you want plain commands without the `uv run` prefix:

```bash
source .venv/bin/activate
deactivate
```

Editors such as Neovim (via basedpyright) and VS Code detect `.venv` automatically once it exists, so no extra configuration is needed.

### 5. Day-to-day dependency changes

```bash
uv add <package>
uv add --dev <package>
uv remove <package>
uv lock --upgrade              # upgrade all dependencies
uv lock --upgrade-package <package>
uv sync
```

Commit both `pyproject.toml` and `uv.lock` after any of these.

### 6. Resetting the environment

If the environment gets into a bad state, delete it and recreate it:

```bash
rm -rf .venv
uv sync --all-groups --locked
```

### Troubleshooting

- `uv sync --locked` fails with a lockfile error: `uv.lock` is out of date with `pyproject.toml`. Run `uv lock` to refresh it (the `uv-lock` pre-commit hook also keeps them in sync automatically).

## Checks

```bash
uv run prek run --all-files
uv run basedpyright
uv run pytest
```

Pull requests run the same checks in `.github/workflows/check_precommit.yml`.

## Releases

Merges to `main` run [semantic-release](https://semantic-release.gitbook.io/) (`.github/workflows/release.yaml`). Based on the conventional commits since the last tag, it creates the tag and GitHub release, updates `CHANGELOG.md` and sets the version in `pyproject.toml`/`uv.lock`, then commits those back to `main` as `chore(release): <version>`.

If `main` is protected, allow GitHub Actions to push to it, or the release commit fails.
