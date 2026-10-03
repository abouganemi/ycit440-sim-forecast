"""Model features, computed only from counts known at the cutoff.

A row describes one forecast: an issue date and a division. Every value taken
from the counts ends at the cutoff (``issue_date - publication_lag_days``).
Only calendar facts about the target window (weekday, month, holidays) look
forward, which the v2 framing allows; nothing describing incidents on the
target days is used. There is no linear time trend: it over-predicts after the
late-2025 first-responder break.

``features_at`` is the reference: it builds one issue date's rows from history
cut at the cutoff, reusing the baseline code where it can. ``feature_table``
builds every issue date at once with backward-looking window functions, plus
the target ``y``. The tests check the two agree on every preset and that
changing counts after a cutoff changes no feature for that issue date.

The history is ``date, DIVISION, n, n_fr`` on the full calendar, as written by
``data.division_day``; ``n_fr`` is the first-responder part of ``n``.
"""

import datetime as dt
import math
from functools import cache

import holidays
import polars as pl

from ycit440_sim_forecast.baseline import MIN_SHARE, SameWeekdayMean
from ycit440_sim_forecast.evaluate import validate_history
from ycit440_sim_forecast.spec import ForecastSpec

LAGS = 7
"""Daily counts at the cutoff and the 6 days before it (``lag_0`` to ``lag_6``)."""

MEAN_DAYS: tuple[int, ...] = (7, 14, 28, 91)
SD_DAYS = 28
SAME_WEEKDAY_WEEKS: tuple[int, ...] = (4, 13)
FR_SHARE_DAYS: tuple[int, ...] = (7, 28)

LEVEL_COLUMNS: tuple[str, ...] = (
    *(f"lag_{k}" for k in range(LAGS)),
    *(f"mean_{d}d" for d in MEAN_DAYS),
    f"sd_{SD_DAYS}d",
    "ratio_7_28",
    "ratio_28_91",
    *(f"same_wd_{w}w" for w in SAME_WEEKDAY_WEEKS),
    *(f"fr_share_{d}d" for d in FR_SHARE_DAYS),
)
"""Features from counts up to the cutoff, all per day (not summed over the window)."""

CALENDAR_COLUMNS: tuple[str, ...] = (
    "target_weekday",
    "target_month",
    "target_doy_sin",
    "target_doy_cos",
    "holidays",
    "holiday_adjacent",
)
"""Calendar facts of the target window (weekday, month, day of year: its first day)."""

FEATURE_COLUMNS: tuple[str, ...] = ("DIVISION", *LEVEL_COLUMNS, *CALENDAR_COLUMNS)
"""Model inputs; ``DIVISION`` is meant to be used as a categorical."""

KEY_COLUMNS: tuple[str, ...] = ("issue_date", "cutoff", "target_start", "DIVISION")
"""Columns that identify a forecast row; not model inputs except ``DIVISION``."""


def min_samples(size: int) -> int:
    """Non-null values a window needs, as in the baselines.

    Args:
        size: Number of values in the window.

    Returns:
        ``ceil(MIN_SHARE * size)``, e.g. 20 for 28 days.
    """
    return math.ceil(MIN_SHARE * size)


@cache
def _holiday_dates(first_year: int, last_year: int) -> tuple[dt.date, ...]:
    """Québec statutory holidays between two years.

    Args:
        first_year: First calendar year included.
        last_year: Last calendar year included.

    Returns:
        Holiday dates, sorted.
    """
    calendar = holidays.country_holidays(
        "CA", subdiv="QC", years=range(first_year, last_year + 1)
    )
    return tuple(sorted(calendar))


def check_history(history: pl.DataFrame) -> None:
    """Check the ``date, DIVISION, n, n_fr`` contract.

    Args:
        history: Frame to check, e.g. from ``data.division_day``.

    Raises:
        ValueError: If ``validate_history`` fails, ``n_fr`` is missing or not an
            integer, is null where ``n`` is not (or the reverse), or exceeds ``n``.
    """
    validate_history(history)
    if "n_fr" not in history.columns or not history.schema["n_fr"].is_integer():
        msg = "history.n_fr must be an integer column (see data.division_day)"
        raise ValueError(msg)
    bad = history.filter(
        (pl.col("n").is_null() != pl.col("n_fr").is_null())
        | (pl.col("n_fr") > pl.col("n"))
    )
    if bad.height:
        msg = f"{bad.height} row(s) where n_fr is not a null-aligned part of n"
        raise ValueError(msg)


def add_calendar(frame: pl.DataFrame, spec: ForecastSpec) -> pl.DataFrame:
    """Add ``CALENDAR_COLUMNS`` for windows starting on ``target_start``.

    ``holidays`` counts Québec statutory holidays in the window;
    ``holiday_adjacent`` counts the other window days that fall the day before
    or after one.

    Args:
        frame: Frame with a ``target_start`` date column.
        spec: Forecast setting; its ``window_days`` sets the window length.

    Returns:
        ``frame`` with ``CALENDAR_COLUMNS`` appended.
    """
    if frame.is_empty():
        return frame.with_columns(
            pl.lit(None, dtype=pl.Int8).alias(c) for c in CALENDAR_COLUMNS
        )
    year = pl.col("target_start").dt.year()
    first, last = frame.select(
        (year.min() - 1).alias("first"), (year.max() + 1).alias("last")
    ).row(0)
    holiday = pl.Series(_holiday_dates(first, last), dtype=pl.Date).implode()

    def day(offset: int) -> pl.Expr:
        return pl.col("target_start") + dt.timedelta(days=offset)

    is_holiday = [day(j).is_in(holiday) for j in range(spec.window_days)]
    adjacent = [
        ~day(j).is_in(holiday) & (day(j - 1).is_in(holiday) | day(j + 1).is_in(holiday))
        for j in range(spec.window_days)
    ]
    angle = 2 * math.pi * pl.col("target_start").dt.ordinal_day() / 365.25
    return frame.with_columns(
        target_weekday=pl.col("target_start").dt.weekday(),
        target_month=pl.col("target_start").dt.month(),
        target_doy_sin=angle.sin(),
        target_doy_cos=angle.cos(),
        holidays=pl.sum_horizontal(is_holiday).cast(pl.Int8),
        holiday_adjacent=pl.sum_horizontal(adjacent).cast(pl.Int8),
    )


def _finish(frame: pl.DataFrame, spec: ForecastSpec) -> pl.DataFrame:
    """Add ratios and calendar columns, then order the columns.

    Args:
        frame: ``KEY_COLUMNS`` and the level features except the ratios, plus
            any extra columns (such as ``y``).
        spec: Forecast setting, passed to ``add_calendar``.

    Returns:
        ``KEY_COLUMNS``, the remaining ``FEATURE_COLUMNS``, then the extras.
    """
    frame = frame.with_columns(
        ratio_7_28=pl.col("mean_7d") / pl.col("mean_28d"),
        ratio_28_91=pl.col("mean_28d") / pl.col("mean_91d"),
    )
    extra = [c for c in frame.columns if c not in (*KEY_COLUMNS, *FEATURE_COLUMNS)]
    return add_calendar(frame, spec).select(*KEY_COLUMNS, *FEATURE_COLUMNS[1:], *extra)


def features_at(
    history: pl.DataFrame, issue_date: dt.date, spec: ForecastSpec
) -> pl.DataFrame:
    """Features of the forecast issued on ``issue_date``, one row per division.

    This is the reference implementation: it cuts ``history`` at the cutoff
    first and computes each feature directly. Use ``feature_table`` for bulk.

    Args:
        history: ``date, DIVISION, n, n_fr`` on the full calendar; rows after
            the cutoff are ignored.
        issue_date: Day the forecast is issued.
        spec: Forecast setting.

    Returns:
        ``KEY_COLUMNS`` then the features, sorted by division.

    Raises:
        ValueError: If ``history`` fails ``check_history``.
    """
    check_history(history)
    cutoff = spec.cutoff(issue_date)
    known = history.filter(pl.col("date") <= cutoff).with_columns(
        pl.col("n", "n_fr").cast(pl.Float64)
    )
    rows = history.select(pl.col("DIVISION").unique().sort()).with_columns(
        issue_date=pl.lit(issue_date),
        cutoff=pl.lit(cutoff),
        target_start=pl.lit(spec.target_start(issue_date)),
    )

    def window(days: int) -> pl.DataFrame:
        start = cutoff - dt.timedelta(days=days - 1)
        return known.filter(pl.col("date") >= start)

    def enough(days: int) -> pl.Expr:
        return pl.col("n").count() >= min_samples(days)

    parts = [
        known.filter(pl.col("date") == cutoff - dt.timedelta(days=k)).select(
            "DIVISION", pl.col("n").alias(f"lag_{k}")
        )
        for k in range(LAGS)
    ]
    parts += [
        window(d)
        .group_by("DIVISION")
        .agg(pl.when(enough(d)).then(pl.col("n").mean()).alias(f"mean_{d}d"))
        for d in MEAN_DAYS
    ]
    parts.append(
        window(SD_DAYS)
        .group_by("DIVISION")
        .agg(pl.when(enough(SD_DAYS)).then(pl.col("n").std()).alias(f"sd_{SD_DAYS}d"))
    )
    parts += [
        window(d)
        .group_by("DIVISION")
        .agg(
            pl.when(enough(d))
            .then(pl.col("n_fr").sum() / pl.col("n").sum())
            .alias(f"fr_share_{d}d")
        )
        for d in FR_SHARE_DAYS
    ]
    parts += [
        SameWeekdayMean(weeks=w)
        .predict(known, issue_date, spec)
        .select(
            "DIVISION", (pl.col("y_pred") / spec.window_days).alias(f"same_wd_{w}w")
        )
        for w in SAME_WEEKDAY_WEEKS
    ]
    for part in parts:
        rows = rows.join(part, on="DIVISION", how="left")
    return _finish(rows, spec)


def _same_weekday(weeks: int, spec: ForecastSpec) -> pl.Expr:
    """Bulk ``SameWeekdayMean(weeks)`` per day, for rows indexed by cutoff date.

    Args:
        weeks: Number of same-weekday values averaged per target day.
        spec: Forecast setting; sets the gap and the window.

    Returns:
        Expression for the mean daily forecast over the window, null when any
        target day has too few values. Needs rows sorted by division then date.
    """
    need = min_samples(weeks)
    days: list[pl.Expr] = []
    for j in range(spec.window_days):
        ahead = spec.gap_days + j  # days from the cutoff to this target day
        first = math.ceil(ahead / 7)
        values = [
            pl.col("n").shift(7 * k - ahead).over("DIVISION")
            for k in range(first, first + weeks)
        ]
        count = pl.sum_horizontal(v.is_not_null() for v in values)
        days.append(pl.when(count >= need).then(pl.mean_horizontal(values)))
    return pl.when(pl.all_horizontal(d.is_not_null() for d in days)).then(
        pl.mean_horizontal(days)
    )


def feature_table(history: pl.DataFrame, spec: ForecastSpec) -> pl.DataFrame:
    """Features and target for every issue date ``history`` can describe.

    Each history date is a cutoff; its row is the forecast issued
    ``publication_lag_days`` later, so the last row is the forecast whose
    cutoff is the last day of ``history``. Every feature looks only backward
    from the cutoff.

    Args:
        history: ``date, DIVISION, n, n_fr`` on the full calendar.
        spec: Forecast setting.

    Returns:
        ``KEY_COLUMNS``, the features and ``y`` (the window total, null when
        any target day is missing or after the end of ``history``), sorted by
        issue date then division. Early rows have null features until enough
        history accumulates.

    Raises:
        ValueError: If ``history`` fails ``check_history``.
    """
    check_history(history)
    n, n_fr = pl.col("n"), pl.col("n_fr")

    def rolling_mean(days: int) -> pl.Expr:
        return n.rolling_mean(days, min_samples=min_samples(days)).over("DIVISION")

    def rolling_sum(column: pl.Expr, days: int) -> pl.Expr:
        return column.rolling_sum(days, min_samples=min_samples(days)).over("DIVISION")

    last_target = spec.gap_days + spec.window_days - 1
    frame = (
        history.select("date", "DIVISION", n.cast(pl.Float64), n_fr.cast(pl.Float64))
        .sort("DIVISION", "date")
        .with_columns(
            *(n.shift(k).over("DIVISION").alias(f"lag_{k}") for k in range(LAGS)),
            *(rolling_mean(d).alias(f"mean_{d}d") for d in MEAN_DAYS),
            n.rolling_std(SD_DAYS, min_samples=min_samples(SD_DAYS))
            .over("DIVISION")
            .alias(f"sd_{SD_DAYS}d"),
            *(
                (rolling_sum(n_fr, d) / rolling_sum(n, d)).alias(f"fr_share_{d}d")
                for d in FR_SHARE_DAYS
            ),
            *(
                _same_weekday(w, spec).alias(f"same_wd_{w}w")
                for w in SAME_WEEKDAY_WEEKS
            ),
            y=n.rolling_sum(spec.window_days, min_samples=spec.window_days)
            .shift(-last_target)
            .over("DIVISION"),
            issue_date=pl.col("date") + dt.timedelta(days=spec.publication_lag_days),
            cutoff=pl.col("date"),
            target_start=pl.col("date") + dt.timedelta(days=spec.gap_days),
        )
        .drop("date", "n", "n_fr")
    )
    return _finish(frame, spec).sort("issue_date", "DIVISION")
