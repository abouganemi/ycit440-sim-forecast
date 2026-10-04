import datetime as dt

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ycit440_sim_forecast import evaluate as ev
from ycit440_sim_forecast import features as ft
from ycit440_sim_forecast import spec
from ycit440_sim_forecast.baseline import PlainMean, SameWeekdayMean
from ycit440_sim_forecast.spec import ForecastSpec

FIRST = dt.date(2023, 1, 1)
MISSING = dt.date(2024, 3, 31)
LEAD30 = spec.ForecastSpec(lead_days=30)
PRESETS = [spec.V2_DAILY, spec.WEEKLY, *spec.daily_horizon(14), LEAD30]
# Early dates (windows still filling), dates around the missing day, the end.
ISSUES = [
    FIRST + dt.timedelta(days=d)
    for d in (2, 5, 9, 20, 40, 95, 200, 450, 452, 455, 457, 460, 470, 729, 731)
]


def make_history(seed: int = 0) -> pl.DataFrame:
    # 2023-2024 for divisions 1 and 2, with one day that has no records at all.
    rng = np.random.default_rng(seed)
    dates = pl.date_range(FIRST, dt.date(2024, 12, 31), "1d", eager=True)
    size = 2 * len(dates)
    n = rng.integers(20, 80, size=size)
    return (
        pl.DataFrame({"date": dates})
        .join(pl.DataFrame({"DIVISION": [1, 2]}), how="cross")
        .with_columns(
            n=pl.Series(n, dtype=pl.UInt32),
            n_fr=pl.Series(rng.integers(0, n + 1), dtype=pl.UInt32),
        )
        .with_columns(
            pl.when(pl.col("date") != MISSING).then(pl.col("n", "n_fr")),
        )
        .sort("date", "DIVISION")
    )


@pytest.fixture(scope="module")
def history() -> pl.DataFrame:
    return make_history()


def row(frame: pl.DataFrame, issue: dt.date, division: int = 1) -> dict[str, object]:
    return frame.filter(
        pl.col("issue_date") == issue, pl.col("DIVISION") == division
    ).row(0, named=True)


@pytest.mark.parametrize("setting", PRESETS, ids=lambda s: s.key)
def test_bulk_equals_reference(history: pl.DataFrame, setting: ForecastSpec) -> None:
    table = ft.feature_table(history, setting).drop("y")
    for issue in ISSUES:
        expected = ft.features_at(history, issue, setting)
        got = table.filter(pl.col("issue_date") == issue)
        assert_frame_equal(got, expected, abs_tol=1e-9)


@pytest.mark.parametrize("setting", PRESETS, ids=lambda s: s.key)
def test_features_ignore_counts_after_cutoff(
    history: pl.DataFrame, setting: ForecastSpec
) -> None:
    features = list(ft.FEATURE_COLUMNS)
    for issue in ISSUES[3::3]:
        cutoff = setting.cutoff(issue)
        poisoned = history.with_columns(
            pl.when(pl.col("date") > cutoff)
            .then(pl.lit(10_000, dtype=pl.UInt32))
            .otherwise(pl.col(c))
            .alias(c)
            for c in ("n", "n_fr")
        )
        for build in (ft.feature_table, lambda h, s, i=issue: ft.features_at(h, i, s)):
            clean = build(history, setting).filter(pl.col("issue_date") == issue)
            dirty = build(poisoned, setting).filter(pl.col("issue_date") == issue)
            assert_frame_equal(clean.select(features), dirty.select(features))


def test_reference_reads_only_up_to_cutoff(history: pl.DataFrame) -> None:
    issue = ISSUES[8]
    cut = history.filter(pl.col("date") <= spec.V2_DAILY.cutoff(issue))
    assert_frame_equal(
        ft.features_at(history, issue, spec.V2_DAILY),
        ft.features_at(cut, issue, spec.V2_DAILY),
    )


@pytest.mark.parametrize("setting", PRESETS, ids=lambda s: s.key)
def test_level_features_match_baselines(
    history: pl.DataFrame, setting: ForecastSpec
) -> None:
    table = ft.feature_table(history, setting)
    for issue in ISSUES[4:]:
        cut = history.filter(pl.col("date") <= setting.cutoff(issue))
        got = table.filter(pl.col("issue_date") == issue).sort("DIVISION")
        same = SameWeekdayMean(13).predict(cut, issue, setting)["y_pred"]
        plain = PlainMean().predict(cut, issue, setting)["y_pred"]
        np.testing.assert_allclose(got["same_wd_13w"] * setting.window_days, same)
        np.testing.assert_allclose(got["mean_28d"] * setting.window_days, plain)


@pytest.mark.parametrize("setting", PRESETS, ids=lambda s: s.key)
def test_target_matches_actuals(history: pl.DataFrame, setting: ForecastSpec) -> None:
    table = ft.feature_table(history, setting)
    dates = table["issue_date"].unique().sort().to_list()
    truth = ev.actuals(history, dates, setting)
    joined = table.join(truth, on=["issue_date", "DIVISION"], how="left")
    assert joined.filter(pl.col("y").ne_missing(pl.col("y_true"))).is_empty()
    assert joined.filter(pl.col("y").is_not_null()).height > 0


def test_target_is_null_when_window_touches_missing_day(history: pl.DataFrame) -> None:
    table = ft.feature_table(history, spec.WEEKLY)
    touching = table.filter(
        pl.col("target_start").is_between(MISSING - dt.timedelta(days=6), MISSING)
    )
    assert touching.height == 14
    assert touching["y"].is_null().all()


def test_hand_values() -> None:
    # Division 1 counts 1, 2, ..., 120 from FIRST; first responders are a third.
    dates = pl.date_range(FIRST, FIRST + dt.timedelta(days=119), "1d", eager=True)
    hist = pl.DataFrame(
        {
            "date": dates,
            "DIVISION": [1] * len(dates),
            "n": pl.Series(range(3, 363, 3), dtype=pl.UInt32),
        }
    ).with_columns(n_fr=pl.col("n") // 3)
    issue = FIRST + dt.timedelta(days=101)  # cutoff = day 99, count 300
    got = row(ft.features_at(hist, issue, spec.V2_DAILY), issue)
    assert got["cutoff"] == FIRST + dt.timedelta(days=99)
    assert got["target_start"] == FIRST + dt.timedelta(days=102)
    assert got["lag_0"] == 300
    assert got["lag_6"] == 282
    assert got["mean_7d"] == pytest.approx(291)
    assert got["mean_91d"] == pytest.approx(165)
    assert got["sd_28d"] == pytest.approx(3 * np.std(np.arange(28), ddof=1))
    assert got["ratio_7_28"] == pytest.approx(291 / 259.5)
    assert got["fr_share_7d"] == pytest.approx(1 / 3)
    # Target day 102 is 3 days after the cutoff, so lag 7 (day 95) is known:
    # mean of the counts on days 95, 88, 81, 74.
    assert got["same_wd_4w"] == pytest.approx(3 * (1 + (95 + 88 + 81 + 74) / 4))


def test_windows_need_min_samples(history: pl.DataFrame) -> None:
    table = ft.feature_table(history, spec.V2_DAILY)
    # Cutoff on day 19 of history: 20 days known, enough for 28 (20 needed) but not 91.
    early = row(table, FIRST + dt.timedelta(days=21))
    assert early["mean_28d"] is not None
    assert early["mean_91d"] is None
    assert early["ratio_28_91"] is None
    too_early = row(table, FIRST + dt.timedelta(days=20))
    assert too_early["mean_28d"] is None
    assert too_early["sd_28d"] is None


def test_lag_over_missing_day_is_null(history: pl.DataFrame) -> None:
    table = ft.feature_table(history, spec.V2_DAILY)
    issue = MISSING + dt.timedelta(days=4)  # cutoff = MISSING + 2
    got = row(table, issue)
    assert got["lag_2"] is None
    assert got["lag_1"] is not None
    assert got["mean_7d"] is not None


def test_last_row_is_the_live_forecast(history: pl.DataFrame) -> None:
    table = ft.feature_table(history, spec.V2_DAILY)
    last = table.filter(pl.col("issue_date") == table["issue_date"].max())
    assert last["cutoff"].to_list() == [dt.date(2024, 12, 31)] * 2
    assert last["y"].is_null().all()
    assert last["mean_28d"].is_not_null().all()


def test_calendar_daily() -> None:
    frame = pl.DataFrame(
        {
            "target_start": [
                dt.date(2024, 6, 24),  # Saint-Jean-Baptiste, a Monday
                dt.date(2024, 6, 23),  # day before it
                dt.date(2024, 6, 20),
                dt.date(2025, 4, 18),  # Good Friday
            ]
        }
    )
    got = ft.add_calendar(frame, spec.V2_DAILY)
    assert got["holidays"].to_list() == [1, 0, 0, 1]
    assert got["holiday_adjacent"].to_list() == [0, 1, 0, 0]
    assert got["target_weekday"].to_list() == [1, 7, 4, 5]
    assert got["target_month"].to_list() == [6, 6, 6, 4]
    radius = got.select(pl.col("target_doy_sin") ** 2 + pl.col("target_doy_cos") ** 2)
    np.testing.assert_allclose(radius.to_series(), 1.0)


def test_calendar_weekly_counts_window_days() -> None:
    # 2024-12-23 to 12-29: Christmas; 24 and 26 are adjacent.
    # 2024-12-30 to 2025-01-05: New Year's Day; Dec 31 and Jan 2 are adjacent.
    frame = pl.DataFrame(
        {"target_start": [dt.date(2024, 12, 23), dt.date(2024, 12, 30)]}
    )
    got = ft.add_calendar(frame, spec.WEEKLY)
    assert got["holidays"].to_list() == [1, 1]
    assert got["holiday_adjacent"].to_list() == [2, 2]


def test_calendar_empty_frame() -> None:
    frame = pl.DataFrame(schema={"target_start": pl.Date})
    got = ft.add_calendar(frame, spec.V2_DAILY)
    assert got.columns == ["target_start", *ft.CALENDAR_COLUMNS]
    assert got.is_empty()


def test_column_order(history: pl.DataFrame) -> None:
    table = ft.feature_table(history, spec.V2_DAILY)
    assert table.columns == [*ft.KEY_COLUMNS, *ft.FEATURE_COLUMNS[1:], "y"]
    reference = ft.features_at(history, ISSUES[5], spec.V2_DAILY)
    assert reference.columns == table.columns[:-1]


def test_min_samples_matches_baselines() -> None:
    assert [ft.min_samples(d) for d in (4, 7, 13, 14, 28, 91)] == [3, 5, 10, 10, 20, 64]
    assert ft.min_samples(13) == SameWeekdayMean(13).min_samples
    assert ft.min_samples(28) == PlainMean().min_samples


def test_check_history_requires_n_fr(history: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="n_fr must be an integer"):
        ft.check_history(history.drop("n_fr"))
    with pytest.raises(ValueError, match="n_fr must be an integer"):
        ft.check_history(history.with_columns(pl.col("n_fr").cast(pl.Float64)))


@pytest.mark.parametrize(
    "broken",
    [
        pl.when(pl.col("date") == FIRST).then(pl.col("n") + 1).otherwise("n_fr"),
        pl.when(pl.col("date") == FIRST).then(None).otherwise("n_fr"),
        pl.when(pl.col("date") == MISSING).then(0).otherwise("n_fr"),
    ],
    ids=["above_n", "null_alone", "set_on_missing_day"],
)
def test_check_history_n_fr_is_part_of_n(
    history: pl.DataFrame, broken: pl.Expr
) -> None:
    bad = history.with_columns(n_fr=broken.cast(pl.UInt32))
    with pytest.raises(ValueError, match="null-aligned part of n"):
        ft.check_history(bad)


def test_check_history_runs_base_validation(history: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="calendar gaps"):
        ft.feature_table(
            history.filter(pl.col("date") != FIRST + dt.timedelta(days=5)),
            spec.V2_DAILY,
        )
