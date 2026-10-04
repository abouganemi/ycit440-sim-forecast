import dataclasses
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any, Self

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from typer.testing import CliRunner

from ycit440_sim_forecast import ensemble as en
from ycit440_sim_forecast import evaluate as ev
from ycit440_sim_forecast import holdout as ho
from ycit440_sim_forecast.baseline import PlainMean
from ycit440_sim_forecast.models import ENSEMBLE_CANDIDATES, LIBRARIES, REFERENCES
from ycit440_sim_forecast.spec import V2_DAILY, WEEKLY, Forecaster, ForecastSpec

runner = CliRunner()
END = dt.date(2025, 3, 31)
BREAK = dt.date(2025, 2, 15)
CONFIG = ho.HoldoutConfig(end=END, break_date=BREAK, n_boot=50)


class Scaled:
    """Cheap stand-in for a candidate: a plain mean times a factor."""

    def __init__(self, name: str, days: int, factor: float) -> None:
        self._name = name
        self.base = PlainMean(days, math.ceil(0.7 * days))
        self.factor = factor

    @property
    def name(self) -> str:
        return self._name

    def fit(self, history: pl.DataFrame, spec: ForecastSpec) -> Self:
        return self

    def predict(
        self, history: pl.DataFrame, issue_date: dt.date, spec: ForecastSpec
    ) -> pl.DataFrame:
        pred = self.base.predict(history, issue_date, spec)
        return pred.with_columns(pl.col("y_pred") * self.factor)


def stand_ins() -> list[Forecaster]:
    return [
        Scaled(name, days, factor)
        for name, days, factor in zip(
            ENSEMBLE_CANDIDATES,
            (21, 28, 35, 42, 56),
            (0.97, 1.0, 1.02, 0.95, 1.05),
            strict=True,
        )
    ]


@pytest.fixture(autouse=True)
def cheap_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ho, "default_models", lambda seed=0: stand_ins())


def make_history(last: dt.date = END, seed: int = 0) -> pl.DataFrame:
    # Two divisions from mid-2024, with a level drop at BREAK in division 2.
    rng = np.random.default_rng(seed)
    dates = pl.date_range(dt.date(2024, 7, 1), last, "1d", eager=True)
    frame = (
        pl.DataFrame({"date": dates})
        .join(pl.DataFrame({"DIVISION": [1, 2]}), how="cross")
        .sort("date", "DIVISION")
    )
    level = np.where(
        (frame["DIVISION"] == 2).to_numpy() & (frame["date"] >= BREAK).to_numpy(),
        30,
        50,
    )
    n = rng.poisson(level)
    return frame.with_columns(
        n=pl.Series(n, dtype=pl.UInt32),
        n_fr=pl.Series(rng.binomial(n, 0.3), dtype=pl.UInt32),
    )


def make_anomalies(history: pl.DataFrame) -> pl.DataFrame:
    return history.select(
        "date", "DIVISION", anomaly=pl.col("date") == dt.date(2025, 1, 15)
    )


@pytest.fixture(scope="module")
def history() -> pl.DataFrame:
    return make_history()


# Configuration


def test_config_defaults_and_to_dict() -> None:
    config = ho.HoldoutConfig(end=dt.date(2026, 9, 24))
    assert config.to_dict() == {
        "spec": {
            "key": "lag2_lead1_win1",
            "publication_lag_days": 2,
            "lead_days": 1,
            "window_days": 1,
        },
        "final_model": "ens_mean",
        "candidates": list(ENSEMBLE_CANDIDATES),
        "references": list(REFERENCES),
        "best_validation_single": "lgbm_poisson_raw",
        "start": "2025-01-01",
        "end": "2026-09-24",
        "break_date": "2025-12-20",
        "n_boot": 2000,
        "seed": 42,
    }
    assert config.final_model == en.FINAL_MODEL
    assert config.start == ev.TEST_START


def test_config_id_is_stable_and_sensitive() -> None:
    same = ho.HoldoutConfig(end=END, break_date=BREAK, n_boot=50)
    assert same.config_id == CONFIG.config_id
    assert len(CONFIG.config_id) == 12
    for change in (
        {"seed": 1},
        {"end": END - dt.timedelta(days=1)},
        {"break_date": BREAK + dt.timedelta(days=1)},
        {"spec": WEEKLY},
    ):
        assert dataclasses.replace(CONFIG, **change).config_id != CONFIG.config_id


def test_config_is_keyword_only() -> None:
    with pytest.raises(TypeError):
        ho.HoldoutConfig(END)  # pyright: ignore[reportCallIssue]


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"final_model": "ens_median"}, "only 'ens_mean'"),
        ({"candidates": ()}, "non-empty and unique"),
        ({"candidates": ("a", "a")}, "non-empty and unique"),
        ({"references": ()}, "references"),
        ({"references": ("plain_7d",)}, "references"),
        ({"references": ("plain_28d", "plain_28d")}, "references"),
        (
            {"candidates": ("a", "plain_28d"), "best_validation_single": "a"},
            "clash",
        ),
        ({"candidates": ("a", "ens_mean"), "best_validation_single": "a"}, "clash"),
        ({"best_validation_single": "xgb_poisson"}, "not a candidate"),
        ({"start": dt.date(2024, 12, 31)}, "before the test period"),
        ({"break_date": ev.TEST_START}, "start < break_date <= end"),
        ({"break_date": END + dt.timedelta(days=1)}, "start < break_date <= end"),
        ({"end": dt.date(2025, 2, 1)}, "start < break_date <= end"),
        ({"n_boot": 0}, "n_boot"),
    ],
)
def test_config_rejects(changes: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        dataclasses.replace(CONFIG, **changes)


def test_config_allows_break_on_last_day() -> None:
    assert dataclasses.replace(CONFIG, break_date=END).break_date == END


# Final-model rows


def test_mean_rows_equal_ensemble_forecaster(history: pl.DataFrame) -> None:
    # The derived ens_mean rows are what running Ensemble itself gives.
    anomalies = make_anomalies(history)
    results = ho.holdout_results(history, anomalies, CONFIG)
    derived = results.filter(pl.col("model") == en.FINAL_MODEL)
    ensemble = en.Ensemble(en.FINAL_MODEL, stand_ins(), "mean")
    direct = ev.flag_anomalies(
        ev.rolling_origin(
            history, [ensemble], V2_DAILY, CONFIG.start, CONFIG.end, allow_test=True
        ),
        anomalies,
    ).select(en.OUTPUT_SCHEMA.names())
    assert derived.height == len(ev.issue_dates(CONFIG.start, END, V2_DAILY)) * 2
    assert_frame_equal(derived, direct.sort(en.KEYS))


def test_holdout_results_layout(history: pl.DataFrame) -> None:
    results = ho.holdout_results(history, make_anomalies(history), CONFIG)
    assert results.schema == en.OUTPUT_SCHEMA
    assert results.equals(results.sort("model", *en.KEYS))
    assert set(results["model"]) == {
        *ENSEMBLE_CANDIDATES,
        *REFERENCES,
        en.FINAL_MODEL,
    }
    assert results["target_start"].min() == ev.TEST_START
    assert results["target_end"].max() == END
    assert results["y_pred"].is_not_null().all()
    wide = results.pivot("model", index=list(en.KEYS), values="y_pred")
    np.testing.assert_allclose(
        wide[en.FINAL_MODEL].to_numpy(),
        wide.select(ENSEMBLE_CANDIDATES).to_numpy().mean(axis=1),
        rtol=1e-12,
    )


def test_mean_rows_null_when_a_member_is_null(history: pl.DataFrame) -> None:
    results = ho.holdout_results(history, make_anomalies(history), CONFIG)
    key = (pl.col("issue_date") == dt.date(2025, 1, 10)) & (pl.col("DIVISION") == 1)
    member = pl.col("model") == ENSEMBLE_CANDIDATES[2]
    singles = results.filter(pl.col("model") != en.FINAL_MODEL).with_columns(
        y_true=pl.when(key).then(None).otherwise("y_true"),
        y_pred=pl.when(key & member).then(None).otherwise("y_pred"),
    )
    rows = ho.mean_rows(singles, CONFIG)
    assert rows.filter(key)["y_pred"].to_list() == [None]
    assert rows["y_pred"].null_count() == 1


def test_unknown_candidate(
    history: pl.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ho, "default_models", lambda seed=0: stand_ins()[:-1])
    with pytest.raises(ValueError, match=r"unknown candidates \['topdown_ets'\]"):
        ho.holdout_results(history, make_anomalies(history), CONFIG)


# Report


@pytest.fixture(scope="module")
def report(history: pl.DataFrame) -> dict[str, Any]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ho, "default_models", lambda seed=0: stand_ins())
        got, _ = ho.holdout_report(history, make_anomalies(history), CONFIG, "abc")
    return got


def test_report_header(report: dict[str, Any]) -> None:
    assert report["config"] == CONFIG.to_dict()
    assert report["config_id"] == CONFIG.config_id
    assert report["input_sha256"] == "abc"
    assert report["final_model"] == "ens_mean"
    assert report["period"] == {
        "start": "2025-01-01",
        "end": "2025-03-31",
        "break_date": "2025-02-15",
    }
    assert set(report["versions"]) == set(LIBRARIES)


def test_report_tables(report: dict[str, Any]) -> None:
    models = {*ENSEMBLE_CANDIDATES, *REFERENCES, en.FINAL_MODEL}
    columns = {"model", "n", "mae", "bias", "median_error", "under_share"}
    n_all = {row["model"]: row["n"] for row in report["overall"]}
    assert set(n_all) == models
    assert set(n_all.values()) == {90 * 2}
    n_clean = {r["model"]: r["n"] for r in report["overall_without_anomalies"]}
    assert all(n_clean[m] == n_all[m] - 2 for m in models)
    for key, extra in (
        ("overall", set()),
        ("by_division", {"DIVISION"}),
        ("by_year_month", {"year_month"}),
        ("by_period", {"period"}),
        ("by_period_division", {"period", "DIVISION"}),
    ):
        assert set(report[key][0]) == columns | extra
    assert {r["year_month"] for r in report["by_year_month"]} == {
        "2025-01",
        "2025-02",
        "2025-03",
    }
    by_period = {(r["model"], r["period"]): r["n"] for r in report["by_period"]}
    # Targets 2025-01-01 to 2025-02-14 before the break, 2025-02-15 on after.
    assert by_period[en.FINAL_MODEL, "before_break"] == 45 * 2
    assert by_period[en.FINAL_MODEL, "after_break"] == 45 * 2
    assert len(report["by_period_division"]) == len(models) * 2 * 2
    assert all(
        isinstance(v, float) and round(v, 3) == v
        for row in report["overall"]
        for k, v in row.items()
        if k in {"mae", "bias"}
    )


def test_report_bootstraps(report: dict[str, Any]) -> None:
    boot = report["bootstrap"]
    assert set(boot["final"]) == {"test", "before_break", "after_break"}
    for comparisons in boot["final"].values():
        assert list(comparisons) == [*REFERENCES, "lgbm_poisson_raw"]
        for reference, row in comparisons.items():
            assert row["model"] == en.FINAL_MODEL
            assert row["reference"] == reference
            assert row["ci_low"] <= row["diff"] <= row["ci_high"]
    whole = boot["final"]["test"]["plain_28d"]
    before = boot["final"]["before_break"]["plain_28d"]
    after = boot["final"]["after_break"]["plain_28d"]
    assert before["n_blocks"] + after["n_blocks"] in {
        whole["n_blocks"],
        whole["n_blocks"] + 1,  # the break week is split between the periods
    }
    for comparisons in boot["candidates_vs_reference"].values():
        assert [r["model"] for r in comparisons] == list(ENSEMBLE_CANDIDATES)
        assert {r["reference"] for r in comparisons} == {"same_wd_13w"}
    assert boot["best_validation_single"]["model"] == "lgbm_poisson_raw"
    assert "before the test run" in boot["best_validation_single"]["selection"]


def test_report_is_deterministic(history: pl.DataFrame, report: dict[str, Any]) -> None:
    again, _ = ho.holdout_report(history, make_anomalies(history), CONFIG, "abc")
    assert json.dumps(again, sort_keys=True) == json.dumps(report, sort_keys=True)


def test_report_needs_scored_rows_after_break(history: pl.DataFrame) -> None:
    gappy = history.with_columns(
        pl.when(pl.col("date") < BREAK).then(pl.col(c)).alias(c) for c in ("n", "n_fr")
    )
    with pytest.raises(ValueError, match=r"no scored rows in \['after_break'\]"):
        ho.holdout_report(gappy, make_anomalies(gappy), CONFIG, "abc")


def test_format_summary(report: dict[str, Any]) -> None:
    lines = ho.format_summary(report)
    assert lines[0].startswith(f"config {CONFIG.config_id}")
    assert any(
        line.startswith("ens_mean ") and "before break" in line for line in lines
    )
    assert sum("ens_mean vs" in line for line in lines) == 3 * 3


# CLI


@pytest.fixture
def inputs(tmp_path: Path) -> tuple[Path, Path]:
    # Past the default break date, so the CLI's default config is valid.
    history = make_history(dt.date(2026, 1, 10))
    history_path = tmp_path / "division_day.parquet"
    anomalies_path = tmp_path / "anomaly_days.parquet"
    history.write_parquet(history_path)
    make_anomalies(history).write_parquet(anomalies_path)
    return history_path, anomalies_path


def invoke(inputs: tuple[Path, Path], out: Path, *extra: str) -> Any:
    history_path, anomalies_path = inputs
    return runner.invoke(
        ho.app,
        [
            "run",
            "--history",
            str(history_path),
            "--anomalies",
            str(anomalies_path),
            "--out",
            str(out),
            "--predictions",
            str(out.with_suffix(".parquet")),
            *extra,
        ],
    )


def test_cli_run_and_one_shot_guard(tmp_path: Path, inputs: tuple[Path, Path]) -> None:
    out = tmp_path / "reports" / "holdout.json"
    result = invoke(inputs, out)
    assert result.exit_code == 0, result.output
    report = json.loads(out.read_text())
    assert report["period"] == {
        "start": "2025-01-01",
        "end": "2026-01-10",
        "break_date": "2025-12-20",
    }
    assert report["input_sha256"] == ho.sha256_file(inputs[0])
    rows = pl.read_parquet(out.with_suffix(".parquet"))
    assert rows.schema == en.OUTPUT_SCHEMA
    assert rows["target_end"].max() == dt.date(2026, 1, 10)
    assert "ens_mean" in result.output
    assert "after_break" in result.output
    assert "wrote" in result.output
    first = out.read_bytes()

    out.write_text("{}")
    refused = invoke(inputs, out)
    assert refused.exit_code == 1
    assert "already exists" in refused.output
    assert "--force" in refused.output
    assert out.read_text() == "{}"

    forced = invoke(inputs, out, "--force")
    assert forced.exit_code == 0, forced.output
    assert out.read_bytes() == first


def test_cli_missing_input(tmp_path: Path) -> None:
    result = runner.invoke(
        ho.app,
        [
            "run",
            "--history",
            str(tmp_path / "nope.parquet"),
            "--out",
            str(tmp_path / "out.json"),
        ],
    )
    assert result.exit_code == 1
    assert "not found" in result.output
    assert not (tmp_path / "out.json").exists()
