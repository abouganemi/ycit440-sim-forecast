"""Record and verify the exact input data a run used.

``data/`` is gitignored, so the manifest (committed) is the only record of
which bytes produced a result. Usage::

    uv run python -m ycit440_sim_forecast.manifest write data/raw
    uv run python -m ycit440_sim_forecast.manifest verify
"""

import hashlib
import json
from pathlib import Path
from typing import Annotated

import typer

DEFAULT_MANIFEST = Path("data/manifest.json")

# 8 MiB reads keep memory flat on multi-GB files without syscall overhead.
CHUNK_SIZE = 8 * 1024 * 1024

app = typer.Typer(help=__doc__, no_args_is_help=True)


def sha256_file(path: Path) -> str:
    """Hash a file in chunks.

    Args:
        path: File to hash.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(root: Path) -> dict[str, dict[str, int | str]]:
    """Describe every file under ``root``.

    Args:
        root: Directory (or single file) to describe.

    Returns:
        Mapping of POSIX path to ``{"sha256": ..., "bytes": ...}``.
    """
    files = [root] if root.is_file() else sorted(root.rglob("*"))
    return {
        path.as_posix(): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in files
        if path.is_file() and path.name != ".gitkeep"
    }


def diff_manifest(
    expected: dict[str, dict[str, int | str]],
) -> list[str]:
    """Compare recorded entries with the files on disk.

    Args:
        expected: Previously written manifest.

    Returns:
        Human-readable problems; empty when everything matches.
    """
    problems: list[str] = []
    for name, entry in expected.items():
        path = Path(name)
        if not path.is_file():
            problems.append(f"missing: {name}")
        elif path.stat().st_size != entry["bytes"]:
            problems.append(f"size changed: {name}")
        elif sha256_file(path) != entry["sha256"]:
            problems.append(f"content changed: {name}")
    return problems


@app.command()
def write(
    root: Annotated[Path, typer.Argument(help="Directory or file to record.")],
    manifest: Annotated[Path, typer.Option(help="Manifest path.")] = DEFAULT_MANIFEST,
) -> None:
    """Hash the files under ROOT and merge them into the manifest."""
    if not root.exists():
        typer.echo(f"{root} does not exist", err=True)
        raise typer.Exit(code=1)
    current = json.loads(manifest.read_text()) if manifest.is_file() else {}
    current.update(build_manifest(root))
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
    typer.echo(f"recorded {len(current)} file(s) in {manifest}")


@app.command()
def verify(
    manifest: Annotated[Path, typer.Option(help="Manifest path.")] = DEFAULT_MANIFEST,
) -> None:
    """Fail if any recorded file is missing or changed."""
    if not manifest.is_file():
        typer.echo(f"{manifest} not found; run `write` first", err=True)
        raise typer.Exit(code=1)
    problems = diff_manifest(json.loads(manifest.read_text()))
    for problem in problems:
        typer.echo(problem, err=True)
    if problems:
        raise typer.Exit(code=1)
    typer.echo("data matches manifest")


if __name__ == "__main__":
    app()
