import datetime as dt

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ycit440_sim_forecast import spec
from ycit440_sim_forecast.baseline import PlainMean, SameWeekdayMean

START = dt.date(2024, 1, 1)


def make_history(days: int = 200, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pl.date_range(START, START + dt.timedelta(days=days - 1), "1d", eager=True)
    return (
        pl.DataFrame({"date": dates})
        .join(pl.DataFrame({"DIVISION": [1, 2]}), how="cross")
        .with_columns(n=pl.Series(rng.integers(20, 80, size=2 * days), dtype=pl.UInt32))
        .sort("date", "DIVISION")
    )


def ramp(days: int = 120) -> pl.DataFrame:
    # One division whose count equals the day index, so means are easy to check.
    dates = pl.date_range(START, START + dt.timedelta(days=days - 1), "1d", eager=True)
    return pl.DataFrame(
        {
            "date": dates,
            "DIVISION": [1] * days,
            "n": pl.Series(range(days), dtype=pl.UInt32),
        }
    )


def pred(frame: pl.DataFrame, division: int = 1) -> float | None:
    return frame.filter(pl.col("DIVISION") == division)["y_pred"].item()


def test_plain_mean_hand_value() -> None:
    issue = START + dt.timedelta(days=60)  # cutoff = day 58
    out = PlainMean(days=28).predict(ramp(), issue, spec.V2_DAILY)
    assert pred(out) == pytest.approx(sum(range(31, 59)) / 28)


def test_plain_mean_weekly_is_seven_days() -> None:
    issue = START + dt.timedelta(days=60)
    daily = pred(PlainMean().predict(ramp(), issue, spec.V2_DAILY))
    weekly = pred(PlainMean().predict(ramp(), issue, spec.WEEKLY))
    assert daily is not None
    assert weekly == pytest.approx(7 * daily)


def test_plain_mean_needs_min_samples() -> None:
    issue = START + dt.timedelta(days=20)  # only 19 days up to the cutoff
    assert (
        pred(PlainMean(days=28, min_samples=20).predict(ramp(), issue, spec.V2_DAILY))
        is None
    )
    assert pred(
        PlainMean(days=28, min_samples=19).predict(ramp(), issue, spec.V2_DAILY)
    ) == (pytest.approx(sum(range(19)) / 19))


def test_same_weekday_hand_value() -> None:
    issue = START + dt.timedelta(days=99)  # target day 100, cutoff day 97
    out = SameWeekdayMean(weeks=4).predict(ramp(), issue, spec.V2_DAILY)
    assert pred(out) == pytest.approx((93 + 86 + 79 + 72) / 4)


def test_same_weekday_skips_weeks_after_cutoff() -> None:
    s = spec.ForecastSpec(lead_days=10)  # gap 12 days: lag 7 is past the cutoff
    issue = START + dt.timedelta(days=90)
    model = SameWeekdayMean(weeks=3)
    cutoff = s.cutoff(issue)
    pairs = model.source_dates(issue, s)
    assert [src for _, src in pairs] == [
        s.target_start(issue) - dt.timedelta(weeks=k) for k in (2, 3, 4)
    ]
    assert all(src <= cutoff for _, src in pairs)


def test_same_weekday_weekly_sums_each_day() -> None:
    issue = START + dt.timedelta(days=99)
    model = SameWeekdayMean(weeks=4)
    weekly = pred(model.predict(ramp(), issue, spec.WEEKLY))
    # Cutoff is day 97: days 105 and 106 skip their lag-7 value (days 98, 99).
    expected = sum(
        sum(day - 7 * k for k in range(first, first + 4)) / 4
        for day in range(100, 107)
        for first in [-(-(day - 97) // 7)]
    )
    assert weekly == pytest.approx(expected)


def test_same_weekday_min_samples_with_null_day() -> None:
    history = ramp().with_columns(
        n=pl.when(pl.col("date") == START + dt.timedelta(days=93))
        .then(None)
        .otherwise("n")
    )
    issue = START + dt.timedelta(days=99)
    assert pred(SameWeekdayMean(weeks=4).predict(history, issue, spec.V2_DAILY)) == (
        pytest.approx((86 + 79 + 72) / 3)
    )
    assert (
        pred(
            SameWeekdayMean(weeks=4, min_samples=4).predict(
                history, issue, spec.V2_DAILY
            )
        )
        is None
    )


@pytest.mark.parametrize("model", [PlainMean(), SameWeekdayMean(weeks=8)])
@pytest.mark.parametrize("s", [spec.V2_DAILY, spec.WEEKLY, *spec.daily_horizon(14)])
def test_baselines_ignore_days_after_cutoff(
    model: PlainMean | SameWeekdayMean, s: spec.ForecastSpec
) -> None:
    history = make_history()
    issue = START + dt.timedelta(days=120)
    poisoned = history.with_columns(
        n=pl.when(pl.col("date") > s.cutoff(issue)).then(10_000).otherwise("n")
    )
    assert_frame_equal(
        model.predict(history, issue, s), model.predict(poisoned, issue, s)
    )


def test_matches_eda_preview_formula() -> None:
    # The notebook's vectorised 2024 preview (cell "baseline preview") for v2.
    history = make_history(days=300, seed=1)
    preview = history.sort("DIVISION", "date").with_columns(
        same_wd_8w=pl.mean_horizontal(
            [pl.col("n").shift(7 * i).over("DIVISION") for i in range(1, 9)]
        ),
        plain_28d=pl.col("n")
        .shift(3)
        .rolling_mean(28, min_samples=20)
        .over("DIVISION"),
    )
    for target in (START + dt.timedelta(days=d) for d in (70, 150, 299)):
        issue = spec.V2_DAILY.issue_date_for(target)
        row = preview.filter(pl.col("date") == target).sort("DIVISION")
        for model, col in (
            (SameWeekdayMean(weeks=8), "same_wd_8w"),
            (PlainMean(), "plain_28d"),
        ):
            got = model.predict(history, issue, spec.V2_DAILY)["y_pred"].to_numpy()
            np.testing.assert_allclose(got, row[col].to_numpy())


def test_predicts_every_division_sorted() -> None:
    history = make_history()
    issue = START + dt.timedelta(days=150)
    for model in (PlainMean(), SameWeekdayMean(weeks=13)):
        out = model.predict(history, issue, spec.V2_DAILY)
        assert out.columns == ["DIVISION", "y_pred"]
        assert out["DIVISION"].to_list() == [1, 2]
        assert out["y_pred"].null_count() == 0


def test_names_and_fit() -> None:
    history = make_history()
    assert PlainMean().name == "plain_28d"
    assert SameWeekdayMean(weeks=13).name == "same_wd_13w"
    model = SameWeekdayMean(weeks=13)
    assert model.fit(history, spec.V2_DAILY) is model
    plain = PlainMean()
    assert plain.fit(history, spec.V2_DAILY) is plain
    assert SameWeekdayMean(weeks=10).min_samples == 7


@pytest.mark.parametrize(
    "build",
    [
        lambda: PlainMean(days=0),
        lambda: PlainMean(days=28, min_samples=29),
        lambda: SameWeekdayMean(weeks=0),
        lambda: SameWeekdayMean(weeks=4, min_samples=5),
    ],
)
def test_invalid_windows(build: object) -> None:
    with pytest.raises(ValueError, match="min_samples"):
        build()  # pyright: ignore[reportCallIssue]
