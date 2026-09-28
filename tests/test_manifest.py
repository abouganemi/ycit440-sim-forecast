import json
from collections.abc import Callable
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ycit440_sim_forecast import manifest

runner = CliRunner()


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    raw = Path("data/raw")
    raw.mkdir(parents=True)
    (raw / "a.csv").write_text("x,y\n1,2\n")
    (raw / ".gitkeep").write_text("")
    return raw


def test_write_then_verify(data_dir: Path) -> None:
    result = runner.invoke(manifest.app, ["write", str(data_dir)])
    assert result.exit_code == 0, result.output

    recorded = json.loads(manifest.DEFAULT_MANIFEST.read_text())
    assert list(recorded) == ["data/raw/a.csv"]  # .gitkeep is skipped
    assert recorded["data/raw/a.csv"]["bytes"] == len("x,y\n1,2\n")

    assert runner.invoke(manifest.app, ["verify"]).exit_code == 0


def test_write_single_file(data_dir: Path) -> None:
    result = runner.invoke(manifest.app, ["write", str(data_dir / "a.csv")])
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda p: p.unlink(), "missing"),
        (lambda p: p.write_text("x,y\n1,2\n3,4\n"), "size changed"),
        (lambda p: p.write_text("x,y\n9,9\n"), "content changed"),
    ],
)
def test_verify_detects_changes(
    data_dir: Path, change: Callable[[Path], object], message: str
) -> None:
    runner.invoke(manifest.app, ["write", str(data_dir)])
    change(data_dir / "a.csv")

    result = runner.invoke(manifest.app, ["verify"])
    assert result.exit_code == 1
    assert message in result.output


def test_write_missing_root(data_dir: Path) -> None:
    assert runner.invoke(manifest.app, ["write", "nope"]).exit_code == 1


def test_verify_without_manifest(data_dir: Path) -> None:
    assert runner.invoke(manifest.app, ["verify"]).exit_code == 1
