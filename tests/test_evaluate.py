import datetime as dt
import itertools
import json
from pathlib import Path
from typing import Self

import numpy as np
import polars as pl
import pytest
from typer.testing import CliRunner

from ycit440_sim_forecast import evaluate as ev
from ycit440_sim_forecast import spec
from ycit440_sim_forecast.baseline import PlainMean, SameWeekdayMean
from ycit440_sim_forecast.spec import ForecastSpec

runner = CliRunner()
START = dt.date(2024, 1, 1)
END = dt.date(2024, 4, 30)
MISSING = dt.date(2024, 3, 31)
PRESETS = [spec.V2_DAILY, spec.WEEKLY, *spec.daily_horizon(14)]


def make_history(seed: int = 0) -> pl.DataFrame:
    # 2023-2024 for divisions 1 and 2, with one day that has no records at all.
    rng = np.random.default_rng(seed)
    dates = pl.date_range(dt.date(2023, 1, 1), dt.date(2024, 12, 31), "1d", eager=True)
    return (
        pl.DataFrame({"date": dates})
        .join(pl.DataFrame({"DIVISION": [1, 2]}), how="cross")
        .with_columns(
            n=pl.Series(rng.integers(20, 80, size=2 * len(dates)), dtype=pl.UInt32)
        )
        .with_columns(n=pl.when(pl.col("date") == MISSING).then(None).otherwise("n"))
        .sort("date", "DIVISION")
    )


@pytest.fixture
def history() -> pl.DataFrame:
    return make_history()


class Spy:
    """Records the latest day it was shown, to prove the harness cuts history."""

    name = "spy"

    def __init__(self) -> None:
        self.fits: list[dt.date] = []
        self.violations: list[str] = []

    def fit(self, history: pl.DataFrame, spec: ForecastSpec) -> Self:
        self.fits.append(history["date"].max())  # pyright: ignore[reportArgumentType]
        return self

    def predict(
        self, history: pl.DataFrame, issue_date: dt.date, spec: ForecastSpec
    ) -> pl.DataFrame:
        seen = history.select(pl.col("date").max()).item()
        if seen > spec.cutoff(issue_date):
            self.violations.append(f"{issue_date}: saw {seen}")
        if self.fits[-1] > spec.cutoff(issue_date):
            self.violations.append(f"{issue_date}: fitted on {self.fits[-1]}")
        return history.select(pl.col("DIVISION").unique().sort()).with_columns(
            y_pred=pl.lit(1.0)
        )


class Broken:
    name = "broken"

    def __init__(self, frame: pl.DataFrame) -> None:
        self.frame = frame

    def fit(self, history: pl.DataFrame, spec: ForecastSpec) -> Self:
        return self

    def predict(
        self, history: pl.DataFrame, issue_date: dt.date, spec: ForecastSpec
    ) -> pl.DataFrame:
        return self.frame


@pytest.mark.parametrize("s", PRESETS, ids=lambda s: s.key)
def test_no_forecaster_sees_past_the_cutoff(
    history: pl.DataFrame, s: ForecastSpec
) -> None:
    spy = Spy()
    results = ev.rolling_origin(history, [spy], s, START, END)
    assert spy.violations == []
    assert results.height == 2 * len(ev.issue_dates(START, END, s))
    assert (results["cutoff"] < results["target_start"]).all()


def test_refits_once_per_month(history: pl.DataFrame) -> None:
    spy = Spy()
    ev.rolling_origin(history, [spy], spec.V2_DAILY, START, END)
    # Issue dates run from 2023-12-31 to 2024-04-29: Dec, Jan, Feb, Mar, Apr.
    assert len(spy.fits) == 5


def test_results_schema_and_truth(history: pl.DataFrame) -> None:
    results = ev.rolling_origin(history, [PlainMean()], spec.V2_DAILY, START, END)
    assert results.schema == ev.RESULT_SCHEMA
    row = results.filter(
        (pl.col("target_start") == dt.date(2024, 2, 10)) & (pl.col("DIVISION") == 2)
    ).row(0, named=True)
    actual = history.filter(
        (pl.col("date") == dt.date(2024, 2, 10)) & (pl.col("DIVISION") == 2)
    )["n"].item()
    assert row["y_true"] == actual
    assert row["issue_date"] == dt.date(2024, 2, 9)
    assert row["cutoff"] == dt.date(2024, 2, 7)


def test_missing_day_is_not_scored(history: pl.DataFrame) -> None:
    results = ev.rolling_origin(history, [PlainMean()], spec.WEEKLY, START, END)
    touching = results.filter(
        pl.lit(MISSING).is_between(pl.col("target_start"), pl.col("target_end"))
    )
    assert touching.height == 2
    assert touching["y_true"].null_count() == 2
    assert (
        results.filter(~pl.col("issue_date").is_in(touching["issue_date"].implode()))[
            "y_true"
        ].null_count()
        == 0
    )


def test_refuses_test_period(history: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="test period"):
        ev.rolling_origin(history, [PlainMean()], spec.V2_DAILY, START, ev.TEST_START)


def test_allow_test_period() -> None:
    history = make_history().with_columns(pl.col("date") + pl.duration(days=366))
    results = ev.rolling_origin(
        history,
        [PlainMean()],
        spec.V2_DAILY,
        dt.date(2025, 1, 1),
        dt.date(2025, 1, 7),
        allow_test=True,
    )
    assert results.height == 14


def test_duplicate_names(history: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="unique"):
        ev.rolling_origin(
            history, [PlainMean(), PlainMean()], spec.V2_DAILY, START, END
        )


def test_wrong_divisions(history: pl.DataFrame) -> None:
    frame = pl.DataFrame({"DIVISION": [1], "y_pred": [1.0]})
    with pytest.raises(ValueError, match="expected divisions"):
        ev.rolling_origin(history, [Broken(frame)], spec.V2_DAILY, START, END)


def test_null_forecast_on_scored_window(history: pl.DataFrame) -> None:
    frame = pl.DataFrame({"DIVISION": [1, 2], "y_pred": [1.0, None]})
    with pytest.raises(ValueError, match="null forecast"):
        ev.rolling_origin(history, [Broken(frame)], spec.V2_DAILY, START, END)


def test_empty_period(history: pl.DataFrame) -> None:
    results = ev.rolling_origin(history, [PlainMean()], spec.WEEKLY, START, START)
    assert results.is_empty()
    assert results.schema == ev.RESULT_SCHEMA


def test_issue_dates_daily() -> None:
    dates = ev.issue_dates(START, dt.date(2024, 1, 10), spec.V2_DAILY)
    assert dates[0] == dt.date(2023, 12, 31)
    assert len(dates) == 10


def test_issue_dates_weekly_start_monday_without_overlap() -> None:
    dates = ev.issue_dates(dt.date(2024, 1, 3), dt.date(2024, 2, 29), spec.WEEKLY)
    starts = [spec.WEEKLY.target_start(d) for d in dates]
    assert all(s.weekday() == 0 for s in starts)
    assert starts[0] == dt.date(2024, 1, 8)
    assert spec.WEEKLY.target_end(dates[-1]) <= dt.date(2024, 2, 29)
    assert all((b - a).days == 7 for a, b in itertools.pairwise(starts))


def test_issue_dates_rejects_reversed() -> None:
    with pytest.raises(ValueError, match="before start"):
        ev.issue_dates(END, START, spec.V2_DAILY)


def test_validate_history_accepts_fixture(history: pl.DataFrame) -> None:
    ev.validate_history(history)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda h: h.with_columns(pl.col("DIVISION").cast(pl.Int32)), "DIVISION"),
        (lambda h: h.with_columns(pl.col("n").cast(pl.Float64)), "integer"),
        (lambda h: h.drop("n"), "integer"),
        (lambda h: pl.concat([h, h.head(1)]), "repeated"),
        (lambda h: h.filter(pl.col("date") != MISSING), "gaps"),
    ],
)
def test_validate_history_rejects(
    history: pl.DataFrame, change: object, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        ev.validate_history(change(history))  # pyright: ignore[reportCallIssue]


def results_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "model": ["a", "a", "a", "b", "b", "b"],
            "target_start": [
                dt.date(2024, 1, 5),
                dt.date(2024, 1, 6),
                dt.date(2024, 2, 1),
            ]
            * 2,
            "target_end": [
                dt.date(2024, 1, 5),
                dt.date(2024, 1, 6),
                dt.date(2024, 2, 1),
            ]
            * 2,
            "issue_date": [
                dt.date(2024, 1, 4),
                dt.date(2024, 1, 5),
                dt.date(2024, 1, 31),
            ]
            * 2,
            "DIVISION": [1, 1, 1] * 2,
            "y_true": [10.0, 20.0, None, 10.0, 20.0, None],
            "y_pred": [12.0, 16.0, 5.0, 10.0, 21.0, 5.0],
        }
    )


def test_score_mae_and_bias() -> None:
    table = ev.score(results_frame())
    assert table.columns == [
        "model",
        "n",
        "mae",
        "bias",
        "median_error",
        "under_share",
    ]
    a = table.filter(pl.col("model") == "a").row(0, named=True)
    assert a == {
        "model": "a",
        "n": 2,
        "mae": 3.0,
        "bias": -1.0,
        "median_error": -1.0,
        "under_share": 0.5,
    }


def test_score_median_error_and_under_share() -> None:
    # Errors (y_pred - y_true) 3, -1, -2, -6 and 0: median -1; 3 of 5 under.
    # The row without truth is not scored, and a zero error is not under.
    frame = pl.DataFrame(
        {
            "model": ["m"] * 6,
            "target_start": [dt.date(2024, 1, d) for d in range(1, 7)],
            "y_true": [10.0, 10.0, 10.0, 10.0, 10.0, None],
            "y_pred": [13.0, 9.0, 8.0, 4.0, 10.0, 1.0],
        }
    )
    row = ev.score(frame).row(0, named=True)
    assert row["n"] == 5
    assert row["bias"] == pytest.approx(-1.2)
    assert row["median_error"] == -1.0
    assert row["under_share"] == pytest.approx(0.6)


def test_score_by_month() -> None:
    table = ev.score(results_frame(), ["month"])
    assert table["month"].to_list() == [1, 1]  # February row has no truth


def test_flag_anomalies() -> None:
    anomalies = pl.DataFrame(
        {
            "date": [dt.date(2024, 1, 6), dt.date(2024, 1, 5)],
            "DIVISION": [1, 2],
            "anomaly": [True, True],
        }
    )
    flagged = ev.flag_anomalies(results_frame(), anomalies)
    assert flagged.filter(pl.col("anomaly"))["target_start"].unique().to_list() == [
        dt.date(2024, 1, 6)
    ]
    assert flagged.height == results_frame().height


def test_select_window_prefers_shorter_on_tie() -> None:
    frame = results_frame().with_columns(
        model=pl.col("model").replace({"a": "same_wd_26w", "b": "same_wd_13w"}),
        y_pred=pl.col("y_true"),
    )
    assert ev.select_window(frame) == "same_wd_13w"


def test_select_window_without_candidates() -> None:
    with pytest.raises(ValueError, match="no model"):
        ev.select_window(results_frame())


@pytest.fixture
def parquet_inputs(tmp_path: Path, history: pl.DataFrame) -> tuple[Path, Path]:
    history_path = tmp_path / "division_day.parquet"
    anomalies_path = tmp_path / "anomaly_days.parquet"
    history.write_parquet(history_path)
    history.select(
        "date", "DIVISION", anomaly=pl.col("date") == dt.date(2024, 2, 14)
    ).write_parquet(anomalies_path)
    return history_path, anomalies_path


def test_cli_baselines(tmp_path: Path, parquet_inputs: tuple[Path, Path]) -> None:
    history_path, anomalies_path = parquet_inputs
    out = tmp_path / "report.json"
    result = runner.invoke(
        ev.app,
        [
            "baselines",
            "--history",
            str(history_path),
            "--anomalies",
            str(anomalies_path),
            "--out",
            str(out),
            "--horizon",
            "2",
            "--start",
            "2024-02-01",
            "--end",
            "2024-03-15",
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(out.read_text())
    assert list(report["settings"]) == [
        "lag2_lead1_win1",
        "lag2_lead1_win7",
        "lag2_lead2_win1",
    ]
    v2 = report["settings"]["lag2_lead1_win1"]
    assert v2["chosen_same_weekday"] in {f"same_wd_{w}w" for w in ev.SAME_WEEKDAY_WEEKS}
    assert {row["model"] for row in v2["overall"]} == {
        v2["chosen_same_weekday"],
        "plain_28d",
    }
    n_all = {row["model"]: row["n"] for row in v2["overall"]}
    n_clean = {row["model"]: row["n"] for row in v2["overall_without_anomalies"]}
    assert all(n_clean[m] == n_all[m] - 2 for m in n_all)  # one day, two divisions
    assert len(v2["candidates"]) == len(ev.SAME_WEEKDAY_WEEKS) + 1
    assert set(v2["overall"][0]) == {
        "model",
        "n",
        "mae",
        "bias",
        "median_error",
        "under_share",
    }
    assert "wrote" in result.output


def test_cli_missing_input(tmp_path: Path) -> None:
    result = runner.invoke(
        ev.app, ["baselines", "--history", str(tmp_path / "nope.parquet")]
    )
    assert result.exit_code == 1
    assert "not found" in result.output


def test_baseline_names_are_valid_candidates() -> None:
    assert SameWeekdayMean(weeks=13).name.startswith("same_wd_")


def bootstrap_frame(shift: float = 0.0, seed: int = 0) -> pl.DataFrame:
    # Two models over ten weeks and three divisions; "b" errs by 2 or 3 per row,
    # "a" by ``shift`` less on every row.
    rng = np.random.default_rng(seed)
    days = pl.date_range(dt.date(2024, 1, 1), dt.date(2024, 3, 10), "1d", eager=True)
    base = (
        pl.DataFrame({"target_start": days})
        .join(pl.DataFrame({"DIVISION": [1, 2, 3]}), how="cross")
        .with_columns(
            issue_date=pl.col("target_start") - pl.duration(days=1),
            y_true=pl.lit(50.0),
        )
    )
    err = rng.choice([2.0, 3.0], size=base.height)
    return pl.concat(
        [
            base.with_columns(model=pl.lit("b"), y_pred=50.0 + pl.Series(err)),
            base.with_columns(model=pl.lit("a"), y_pred=50.0 + pl.Series(err - shift)),
        ]
    )


def test_paired_bootstrap_identical_models() -> None:
    got = ev.paired_bootstrap(bootstrap_frame(), "a", "b", n_boot=200)
    assert got["diff"] == 0.0
    assert (got["ci_low"], got["ci_high"]) == (0.0, 0.0)
    assert got["n_blocks"] == 10  # ISO weeks 1 to 10 of 2024
    assert got["model"] == "a"
    assert got["reference"] == "b"


def test_paired_bootstrap_uniformly_better() -> None:
    got = ev.paired_bootstrap(bootstrap_frame(shift=1.0), "a", "b", n_boot=200)
    assert got["diff"] == pytest.approx(-1.0)
    assert got["ci_low"] == pytest.approx(-1.0)
    assert got["ci_high"] == pytest.approx(-1.0)
    assert got["mae_reference"] - got["mae_model"] == pytest.approx(1.0)


def test_paired_bootstrap_is_deterministic_and_seeded() -> None:
    # Noisy differences so the interval has width.
    frame = bootstrap_frame()
    noise = pl.Series(np.random.default_rng(1).uniform(0, 5, frame.height))
    frame = frame.with_columns(
        y_pred=pl.when(pl.col("model") == "a").then(50.0 + noise).otherwise("y_pred")
    )
    first = ev.paired_bootstrap(frame, "a", "b", n_boot=500, seed=7)
    assert first == ev.paired_bootstrap(frame, "a", "b", n_boot=500, seed=7)
    assert first["ci_low"] < first["diff"] < first["ci_high"]
    assert first != ev.paired_bootstrap(frame, "a", "b", n_boot=500, seed=8)


def test_paired_bootstrap_skips_unscored_rows() -> None:
    frame = bootstrap_frame(shift=1.0).with_columns(
        y_true=pl.when(pl.col("DIVISION") == 3).then(None).otherwise("y_true")
    )
    got = ev.paired_bootstrap(frame, "a", "b", n_boot=50)
    assert got["diff"] == pytest.approx(-1.0)


def test_paired_bootstrap_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="block"):
        ev.paired_bootstrap(bootstrap_frame(), "a", "b", block="day")  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="no scored rows"):
        ev.paired_bootstrap(bootstrap_frame(), "a", "missing")
