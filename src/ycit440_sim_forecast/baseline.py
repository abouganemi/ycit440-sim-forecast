"""Baseline forecasters the models must beat.

``SameWeekdayMean`` is the v2 baseline: the mean of the most recent
same-weekday counts known at the cutoff. ``PlainMean`` averages the last days
before the cutoff and, in the EDA preview on 2024, beat it (MAE 8.03 against
8.16), so it is reported as a second comparison.

Both forecast each target day and sum over the window, and both read only days
up to the cutoff, whatever ``history`` they are given.
"""

import datetime as dt
import math
from typing import Self

import polars as pl

from ycit440_sim_forecast.spec import ForecastSpec

MIN_SHARE = 0.7
"""Default share of non-null values a mean needs (20 of 28 days, rounded)."""


def _divisions(history: pl.DataFrame) -> pl.DataFrame:
    """Divisions present in ``history``.

    Args:
        history: Frame with a ``DIVISION`` column.

    Returns:
        ``DIVISION``, one row per division, sorted.
    """
    return history.select(pl.col("DIVISION").unique().sort())


class PlainMean:
    """Mean daily count over the ``days`` days ending at the cutoff, times the window.

    Attributes:
        days: Length of the averaging window.
        min_samples: Non-null days required; fewer gives a null forecast.
    """

    def __init__(self, days: int = 28, min_samples: int = 20) -> None:
        """Set the window.

        Args:
            days: Length of the averaging window.
            min_samples: Non-null days required, at most ``days``.

        Raises:
            ValueError: If the window or the minimum is out of range.
        """
        if days < 1 or not 1 <= min_samples <= days:
            msg = f"need 1 <= min_samples <= days, got {min_samples} and {days}"
            raise ValueError(msg)
        self.days = days
        self.min_samples = min_samples

    @property
    def name(self) -> str:
        """Label, e.g. ``plain_28d``."""
        return f"plain_{self.days}d"

    def fit(self, history: pl.DataFrame, spec: ForecastSpec) -> Self:
        """Nothing to learn.

        Args:
            history: Unused.
            spec: Unused.

        Returns:
            This forecaster.
        """
        return self

    def predict(
        self, history: pl.DataFrame, issue_date: dt.date, spec: ForecastSpec
    ) -> pl.DataFrame:
        """Forecast the window of ``issue_date``.

        Args:
            history: ``date, DIVISION, n`` on the full calendar; only the
                ``days`` days ending at the cutoff are read.
            issue_date: Day the forecast is issued.
            spec: Forecast setting.

        Returns:
            ``DIVISION, y_pred`` for every division in ``history``, sorted;
            null when the window has fewer than ``min_samples`` values.
        """
        cutoff = spec.cutoff(issue_date)
        start = cutoff - dt.timedelta(days=self.days - 1)
        recent = (
            history.filter(pl.col("date").is_between(start, cutoff))
            .group_by("DIVISION")
            .agg(mean=pl.col("n").mean(), count=pl.col("n").count())
        )
        return (
            _divisions(history)
            .join(recent, on="DIVISION", how="left")
            .select(
                "DIVISION",
                y_pred=pl.when(pl.col("count") >= self.min_samples).then(
                    pl.col("mean") * spec.window_days
                ),
            )
        )


class SameWeekdayMean:
    """Mean of the ``weeks`` most recent same-weekday counts known at the cutoff.

    For each target day it uses the same weekday 1, 2, ... weeks earlier,
    skipping weeks that fall after the cutoff (only when the gap from the
    cutoff exceeds 7 days), then sums the daily forecasts over the window.

    Attributes:
        weeks: Number of same-weekday values averaged.
        min_samples: Non-null values required per target day.
    """

    def __init__(self, weeks: int, min_samples: int | None = None) -> None:
        """Set the window.

        Args:
            weeks: Number of same-weekday values averaged.
            min_samples: Non-null values required per target day; defaults to
                ``ceil(0.7 * weeks)``.

        Raises:
            ValueError: If the window or the minimum is out of range.
        """
        min_samples = (
            math.ceil(MIN_SHARE * weeks) if min_samples is None else min_samples
        )
        if weeks < 1 or not 1 <= min_samples <= weeks:
            msg = f"need 1 <= min_samples <= weeks, got {min_samples} and {weeks}"
            raise ValueError(msg)
        self.weeks = weeks
        self.min_samples = min_samples

    @property
    def name(self) -> str:
        """Label, e.g. ``same_wd_13w``."""
        return f"same_wd_{self.weeks}w"

    def fit(self, history: pl.DataFrame, spec: ForecastSpec) -> Self:
        """Nothing to learn.

        Args:
            history: Unused.
            spec: Unused.

        Returns:
            This forecaster.
        """
        return self

    def source_dates(
        self, issue_date: dt.date, spec: ForecastSpec
    ) -> list[tuple[dt.date, dt.date]]:
        """Pairs of ``(target_day, source_day)`` the forecast averages.

        Args:
            issue_date: Day the forecast is issued.
            spec: Forecast setting.

        Returns:
            ``weeks`` source days per target day, newest first; every source
            day is on or before the cutoff.
        """
        cutoff = spec.cutoff(issue_date)
        pairs: list[tuple[dt.date, dt.date]] = []
        for day in spec.target_days(issue_date):
            first = math.ceil((day - cutoff).days / 7)
            pairs.extend(
                (day, day - dt.timedelta(weeks=k))
                for k in range(first, first + self.weeks)
            )
        return pairs

    def predict(
        self, history: pl.DataFrame, issue_date: dt.date, spec: ForecastSpec
    ) -> pl.DataFrame:
        """Forecast the window of ``issue_date``.

        Args:
            history: ``date, DIVISION, n`` on the full calendar; only the days
                from ``source_dates`` are read.
            issue_date: Day the forecast is issued.
            spec: Forecast setting.

        Returns:
            ``DIVISION, y_pred`` for every division in ``history``, sorted;
            null when any target day has fewer than ``min_samples`` values.
        """
        target, source = zip(*self.source_dates(issue_date, spec), strict=True)
        lookup = pl.DataFrame(
            {"target_day": target, "date": source},
            schema={"target_day": pl.Date, "date": pl.Date},
        )
        daily = (
            lookup.join(_divisions(history), how="cross")
            .join(
                history.select("date", "DIVISION", "n"),
                on=["date", "DIVISION"],
                how="left",
            )
            .group_by("DIVISION", "target_day")
            .agg(mean=pl.col("n").mean(), count=pl.col("n").count())
            .with_columns(
                day_pred=pl.when(pl.col("count") >= self.min_samples).then(
                    pl.col("mean")
                )
            )
        )
        return (
            daily.group_by("DIVISION")
            .agg(
                y_pred=pl.when(pl.col("day_pred").null_count() == 0).then(
                    pl.col("day_pred").sum()
                )
            )
            .sort("DIVISION")
        )
