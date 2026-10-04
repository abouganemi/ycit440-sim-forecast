import dataclasses
import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
from typer.testing import CliRunner

from ycit440_sim_forecast import ensemble as en
from ycit440_sim_forecast import horizons as hz
from ycit440_sim_forecast.models import ENSEMBLE_CANDIDATES, REFERENCES, parse_horizon

runner = CliRunner()
CONFIG = en.EnsembleConfig(
    score_start=dt.date(2021, 3, 1), score_end=dt.date(2021, 6, 30), n_boot=50
)


def make_oof(label: str, seed: int = 0) -> pl.DataFrame:
    # Daily issue dates, two divisions; each model is y_true plus its own
    # noise and bias, the windows follow the horizon's spec.
    setting = parse_horizon(label)
    rng = np.random.default_rng(seed)
    issues = pl.date_range(dt.date(2020, 10, 1), dt.date(2021, 6, 20), "1d", eager=True)
    scale = float(setting.window_days)
    base = (
        pl.DataFrame({"issue_date": issues})
        .join(pl.DataFrame({"DIVISION": [1, 2]}), how="cross")
        .with_columns(
            spec=pl.lit(setting.key),
            cutoff=pl.col("issue_date") - pl.duration(days=2),
            target_start=pl.col("issue_date") + pl.duration(days=setting.lead_days),
            target_end=pl.col("issue_date")
            + pl.duration(days=setting.lead_days + setting.window_days - 1),
            y_true=pl.Series(
                scale * rng.integers(20, 80, 2 * len(issues)), dtype=pl.Float64
            ),
            anomaly=pl.Series(rng.random(2 * len(issues)) < 0.05),
        )
    )
    models = (*ENSEMBLE_CANDIDATES, *REFERENCES)
    frames = [
        base.with_columns(
            model=pl.lit(name),
            y_pred=pl.col("y_true")
            + scale * (i - 3) * 0.5
            + pl.Series(scale * rng.normal(0, 3 + i, base.height)),
        )
        for i, name in enumerate(models)
    ]
    return pl.concat(frames).select(en.OUTPUT_SCHEMA.names()).cast(en.OUTPUT_SCHEMA)


def make_report(label: str) -> dict[str, Any]:
    config = dataclasses.replace(CONFIG, spec=parse_horizon(label))
    report, _ = en.ensemble_report(make_oof(label), config, f"sha-{label}")
    return json.loads(json.dumps(report))


@pytest.fixture(scope="module")
def reports() -> dict[str, dict[str, Any]]:
    return {label: make_report(label) for label in ("v2", "weekly", "lead30")}


def write_reports(directory: Path, reports: dict[str, dict[str, Any]]) -> None:
    for label, report in reports.items():
        hz.report_path(directory, label).write_text(json.dumps(report))


def test_report_path() -> None:
    assert hz.report_path(Path("reports"), "v2") == en.DEFAULT_OUT
    assert hz.report_path(Path("r"), "lead30") == Path(
        "r/ensemble_validation_lead30.json"
    )


def test_comparison_layout(reports: dict[str, dict[str, Any]]) -> None:
    got = hz.horizon_comparison(reports)
    assert got["final_model"] == en.FINAL_MODEL
    assert got["labels"] == ["v2", "weekly", "lead30"]
    assert "hindsight" in got["best_single_selection"]
    for label, report in reports.items():
        setting = got["settings"][label]
        assert setting["spec"] == report["config"]["spec"]
        assert setting["config_id"] == report["config_id"]
        assert setting["input_sha256"] == f"sha-{label}"
        assert setting["period"] == {"start": "2021-03-01", "end": "2021-06-30"}
        best = report["bootstrap"]["best_single"]["model"]
        assert setting["best_single"] == best
        models = setting["models"]
        assert list(models) == [en.FINAL_MODEL, *REFERENCES, best]
        assert [m["role"] for m in models.values()] == [
            "final",
            "reference",
            "reference",
            hz.HINDSIGHT,
        ]
        overall = {row["model"]: row for row in report["overall"]}
        assert setting["n"] == overall[en.FINAL_MODEL]["n"]
        for model, row in models.items():
            assert row["mae"] == overall[model]["mae"]
            assert row["bias"] == overall[model]["bias"]
            assert row["wape"] == report["wape"][model]
        assert "final_minus" not in models[en.FINAL_MODEL]
        boot = report["bootstrap"]["references"]["plain_28d"][0]
        assert boot["model"] == en.FINAL_MODEL
        assert models["plain_28d"]["final_minus"] == {
            k: boot[k] for k in ("diff", "ci_low", "ci_high", "n_blocks")
        }


def test_wape_is_scale_free(reports: dict[str, dict[str, Any]]) -> None:
    # Weekly totals are 7 times the daily counts here, so MAE scales by about
    # 7 while WAPE stays comparable.
    got = hz.horizon_comparison(reports)["settings"]
    daily, weekly = (got[k]["models"]["plain_28d"] for k in ("v2", "weekly"))
    assert weekly["mae"] > 5 * daily["mae"]
    assert weekly["wape"] == pytest.approx(daily["wape"], rel=0.2)


def test_comparison_rejects(reports: dict[str, dict[str, Any]]) -> None:
    with pytest.raises(ValueError, match="no ensemble reports"):
        hz.horizon_comparison({})
    old = {k: v for k, v in reports["v2"].items() if k != "wape"}
    with pytest.raises(ValueError, match="ensemble validate --horizon v2"):
        hz.horizon_comparison({"v2": old})
    shifted = {**reports["weekly"], "period": {"start": "x", "end": "y"}}
    with pytest.raises(ValueError, match="different periods"):
        hz.horizon_comparison({"v2": reports["v2"], "weekly": shifted})
    uneven = json.loads(json.dumps(reports["v2"]))
    uneven["overall"][0]["n"] += 1
    with pytest.raises(ValueError, match="different row counts"):
        hz.horizon_comparison({"v2": uneven})
    no_ref = json.loads(json.dumps(reports["v2"]))
    no_ref["overall"] = [r for r in no_ref["overall"] if r["model"] != "plain_28d"]
    with pytest.raises(ValueError, match="plain_28d is missing"):
        hz.horizon_comparison({"v2": no_ref})
    no_wape = json.loads(json.dumps(reports["v2"]))
    del no_wape["wape"]["plain_28d"]
    with pytest.raises(ValueError, match="plain_28d has no WAPE"):
        hz.horizon_comparison({"v2": no_wape})
    swapped = json.loads(json.dumps(reports["v2"]))
    swapped["bootstrap"]["references"]["plain_28d"][0]["reference"] = "other"
    with pytest.raises(ValueError, match="is against other"):
        hz.horizon_comparison({"v2": swapped})


def test_cli_compare(tmp_path: Path, reports: dict[str, dict[str, Any]]) -> None:
    write_reports(tmp_path, reports)
    out = tmp_path / "comparison.json"
    args = ["compare", "v2", "weekly", "lead30", "--reports", str(tmp_path)]
    result = runner.invoke(hz.app, [*args, "--out", str(out)])
    assert result.exit_code == 0, result.output
    saved = json.loads(out.read_text())
    assert saved == hz.horizon_comparison(reports)
    lines = result.output.splitlines()
    assert lines[0].startswith("setting")
    assert sum(line.startswith("weekly ") for line in lines) == 4
    assert "lag2_lead30_win1" in result.output
    assert "*" in result.output
    assert "wrote" in lines[-1]
    again = tmp_path / "again.json"
    runner.invoke(hz.app, [*args, "--out", str(again)])
    assert again.read_bytes() == out.read_bytes()


def test_cli_compare_default_labels_and_missing_report(
    tmp_path: Path, reports: dict[str, dict[str, Any]]
) -> None:
    write_reports(tmp_path, reports)
    result = runner.invoke(hz.app, ["compare", "--reports", str(tmp_path)])
    assert result.exit_code == 1
    assert "ensemble_validation_lead3.json not found" in result.output
    assert "models validate --horizon lead3" in result.output
    assert "ensemble validate --horizon lead3" in result.output


def test_cli_compare_canonical_labels(
    tmp_path: Path, reports: dict[str, dict[str, Any]]
) -> None:
    write_reports(tmp_path, reports)
    out = tmp_path / "c.json"
    result = runner.invoke(
        hz.app, ["compare", "lead1", "--reports", str(tmp_path), "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(out.read_text())["labels"] == ["v2"]
    repeated = runner.invoke(
        hz.app, ["compare", "v2", "lead1", "--reports", str(tmp_path)]
    )
    assert repeated.exit_code == 2
    assert "repeat" in repeated.output
    bad = runner.invoke(hz.app, ["compare", "hourly", "--reports", str(tmp_path)])
    assert bad.exit_code == 2
    assert "unknown horizon" in bad.output
