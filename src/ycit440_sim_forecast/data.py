"""Load the SIM intervention files and build daily count series.

The two open-data exports share one layout; ``load_interventions`` reads them
with an explicit schema and ``daily_counts`` turns rows into zero-filled daily
series. A calendar date with no records at all (e.g. 2024-12-31) is missing
data, not a quiet day, so its count is null rather than 0.
"""

from collections.abc import Iterable
from pathlib import Path

import polars as pl

RAW_FILES: tuple[str, str] = (
    "donneesouvertes-interventions-sim2020.csv",
    "donneesouvertes-interventions-sim.csv",
)
TARGET_DIVISIONS: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
FIRST_RESPONDER_GROUP = "1-REPOND"

RAW_SCHEMA = pl.Schema(
    {
        "INCIDENT_NBR": pl.Int64,
        "CREATION_DATE_TIME": pl.String,
        "INCIDENT_TYPE_DESC": pl.String,
        "DESCRIPTION_GROUPE": pl.String,
        "CASERNE": pl.Int64,
        "NOM_VILLE": pl.String,
        "NOM_ARROND": pl.String,
        "DIVISION": pl.Int64,
        "NOMBRE_UNITES": pl.Int64,
        "MTM8_X": pl.Float64,
        "MTM8_Y": pl.Float64,
        "LONGITUDE": pl.Float64,
        "LATITUDE": pl.Float64,
    }
)

DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S"
DATE_FORMAT = "%Y-%m-%d"


def _read_raw(path: Path) -> pl.DataFrame:
    """Read one export with the fixed schema and tag rows with the file name.

    Args:
        path: CSV file to read.

    Returns:
        Raw columns plus ``source_file``; empty fields are null.
    """
    return pl.read_csv(path, schema=RAW_SCHEMA).with_columns(
        source_file=pl.lit(path.name, dtype=pl.String)
    )


def load_interventions(raw_dir: Path = Path("data/raw")) -> pl.DataFrame:
    """Read both raw files with an explicit schema and add derived columns.

    ``CREATION_DATE_TIME`` is parsed to ``pl.Datetime``; the few date-only
    values become midnight and are flagged with ``has_time = False``.

    Args:
        raw_dir: Directory holding the files named in ``RAW_FILES``.

    Returns:
        The 13 source columns plus ``has_time``, ``date``, ``year`` and
        ``source_file``.

    Raises:
        FileNotFoundError: If either raw file is absent.
        ValueError: If a timestamp matches neither accepted format.
    """
    paths = [raw_dir / name for name in RAW_FILES]
    for path in paths:
        if not path.is_file():
            msg = f"raw file not found: {path}"
            raise FileNotFoundError(msg)

    raw = pl.concat([_read_raw(path) for path in paths], how="vertical")

    text = pl.col("CREATION_DATE_TIME")
    parsed = pl.coalesce(
        text.str.strptime(pl.Datetime("us"), DATETIME_FORMAT, strict=False),
        text.str.strptime(pl.Date, DATE_FORMAT, strict=False).cast(pl.Datetime("us")),
    )
    df = raw.with_columns(
        has_time=text.str.contains("T", literal=True),
        CREATION_DATE_TIME=parsed,
    )

    bad = df.filter(pl.col("CREATION_DATE_TIME").is_null()).height
    if bad:
        msg = f"{bad} row(s) have an unparseable CREATION_DATE_TIME"
        raise ValueError(msg)

    return df.with_columns(
        date=pl.col("CREATION_DATE_TIME").dt.date(),
        year=pl.col("CREATION_DATE_TIME").dt.year().cast(pl.Int32),
    )


def daily_counts(
    df: pl.DataFrame,
    by: str | None = "DIVISION",
    keep: Iterable[int] | None = TARGET_DIVISIONS,
) -> pl.DataFrame:
    """Count records per day on the full calendar, zero-filled; empty dates are null.

    Args:
        df: Frame with a ``date`` column, e.g. from ``load_interventions``.
        by: Integer column to split series by, or ``None`` for one series.
        keep: Groups to return (other rows are dropped before counting);
            ``None`` keeps every non-null value of ``by`` found in ``df``.

    Returns:
        ``date, n`` when ``by`` is ``None``, else ``date, <by>, n`` with one
        row per date and group, sorted by date then group. ``n`` is 0 when a
        group has no records on a date that has records elsewhere in ``df``,
        and null on dates with no records at all.

    Raises:
        ValueError: If ``df`` is empty.
    """
    if df.is_empty():
        msg = "cannot build a calendar from an empty frame"
        raise ValueError(msg)

    calendar = df.select(
        pl.date_range(pl.col("date").min(), pl.col("date").max(), "1d").alias("date")
    )
    observed = pl.col("date").is_in(df["date"].unique().implode())

    if by is None:
        grid = calendar
        counts = df.group_by("date").agg(n=pl.len())
        keys = ["date"]
    else:
        if keep is None:
            groups = df[by].drop_nulls().unique().sort()
        else:
            groups = pl.Series(by, list(keep), dtype=df.schema[by])
            df = df.filter(pl.col(by).is_in(groups.implode()))
        grid = calendar.join(groups.to_frame(), how="cross")
        counts = df.group_by("date", by).agg(n=pl.len())
        keys = ["date", by]

    return (
        grid.join(counts, on=keys, how="left")
        .with_columns(
            n=pl.when(observed)
            .then(pl.col("n").fill_null(0))
            .otherwise(None)
            .cast(pl.UInt32)
        )
        .sort(keys)
    )


def division_day(
    df: pl.DataFrame, keep: Iterable[int] = TARGET_DIVISIONS
) -> pl.DataFrame:
    """Daily total and first-responder counts per division: the modelling history.

    Args:
        df: Frame from ``load_interventions``.
        keep: Divisions to return.

    Returns:
        ``date, DIVISION, n, n_fr`` from ``daily_counts``; ``n_fr`` counts
        ``DESCRIPTION_GROUPE == FIRST_RESPONDER_GROUP`` (a blank group is not a
        first-responder call) and is null exactly where ``n`` is null.
    """
    first_responder = (
        df.filter(pl.col("DESCRIPTION_GROUPE") == FIRST_RESPONDER_GROUP)
        .group_by("date", "DIVISION")
        .agg(n_fr=pl.len())
    )
    return (
        daily_counts(df, keep=keep)
        .join(first_responder, on=["date", "DIVISION"], how="left")
        .with_columns(
            n_fr=pl.when(pl.col("n").is_not_null())
            .then(pl.col("n_fr").fill_null(0))
            .cast(pl.UInt32)
        )
        .sort("date", "DIVISION")
    )
