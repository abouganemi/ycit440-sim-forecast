import datetime as dt
from pathlib import Path

import polars as pl
import pytest

from ycit440_sim_forecast import data

HEADER = (
    "INCIDENT_NBR,CREATION_DATE_TIME,INCIDENT_TYPE_DESC,DESCRIPTION_GROUPE,"
    "CASERNE,NOM_VILLE,NOM_ARROND,DIVISION,NOMBRE_UNITES,"
    "MTM8_X,MTM8_Y,LONGITUDE,LATITUDE\n"
)


def row(nbr: int, when: str, division: int, group: str = "AUTREFEU") -> str:
    return (
        f"{nbr},{when},Feu,{group},10,Montréal,Ville-Marie,{division},2,"
        "299000.5,5040000.5,-73.56,45.50\n"
    )


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    # 2024-12-31 has no rows at all; 2025-01-02 has division 1 only.
    old = (
        HEADER
        + row(1, "2024-12-29T08:00:00", 1)
        + row(2, "2024-12-29T09:30:00", 2)
        + row(3, "2024-12-29", 1, group="")
        + "4,2024-12-30T23:59:59,Feu,,0,Montréal,,0,,,,,\n"
    )
    new = HEADER + row(1, "2025-01-01T00:00:01", 3) + row(2, "2025-01-02T12:00:00", 1)
    (tmp_path / data.RAW_FILES[0]).write_text(old)
    (tmp_path / data.RAW_FILES[1]).write_text(new)
    return tmp_path


@pytest.fixture
def df(raw_dir: Path) -> pl.DataFrame:
    return data.load_interventions(raw_dir)


def test_load_schema(df: pl.DataFrame) -> None:
    assert df.height == 6
    assert df.schema == pl.Schema(
        {
            **data.RAW_SCHEMA,
            "CREATION_DATE_TIME": pl.Datetime("us"),
            "source_file": pl.String,
            "has_time": pl.Boolean,
            "date": pl.Date,
            "year": pl.Int32,
        }
    )


def test_load_empty_fields_are_null(df: pl.DataFrame) -> None:
    nulls = df.null_count()
    assert nulls["DESCRIPTION_GROUPE"].item() == 2
    assert nulls["NOMBRE_UNITES"].item() == 1
    assert nulls["NOM_ARROND"].item() == 1
    assert nulls["LATITUDE"].item() == 1


def test_load_date_only_rows(df: pl.DataFrame) -> None:
    date_only = df.filter(~pl.col("has_time"))
    assert date_only["INCIDENT_NBR"].to_list() == [3]
    assert date_only["CREATION_DATE_TIME"].item() == dt.datetime(2024, 12, 29)
    assert df["has_time"].sum() == 5


def test_load_derived_columns(df: pl.DataFrame) -> None:
    assert (
        df["source_file"].to_list() == [data.RAW_FILES[0]] * 4 + [data.RAW_FILES[1]] * 2
    )
    assert df["year"].to_list() == [2024] * 4 + [2025] * 2
    assert df["date"][1] == dt.date(2024, 12, 29)
    # (year, INCIDENT_NBR) is the unique key; numbers restart each year.
    assert df.select("year", "INCIDENT_NBR").is_unique().all()


@pytest.mark.parametrize("missing", data.RAW_FILES)
def test_load_missing_file(raw_dir: Path, missing: str) -> None:
    (raw_dir / missing).unlink()
    with pytest.raises(FileNotFoundError, match=missing):
        data.load_interventions(raw_dir)


def test_load_bad_timestamp(raw_dir: Path) -> None:
    path = raw_dir / data.RAW_FILES[1]
    path.write_text(path.read_text() + row(3, "29/12/2025", 1))
    with pytest.raises(ValueError, match="unparseable"):
        data.load_interventions(raw_dir)


def test_daily_counts_total(df: pl.DataFrame) -> None:
    counts = data.daily_counts(df, by=None)
    assert counts.columns == ["date", "n"]
    assert counts.schema["n"] == pl.UInt32
    assert counts.rows() == [
        (dt.date(2024, 12, 29), 3),
        (dt.date(2024, 12, 30), 1),
        (dt.date(2024, 12, 31), None),
        (dt.date(2025, 1, 1), 1),
        (dt.date(2025, 1, 2), 1),
    ]


def test_daily_counts_by_division_keep(df: pl.DataFrame) -> None:
    counts = data.daily_counts(df, keep=(1, 2))
    assert counts.columns == ["date", "DIVISION", "n"]
    assert counts.schema["n"] == pl.UInt32
    assert counts.rows() == [
        (dt.date(2024, 12, 29), 1, 2),
        (dt.date(2024, 12, 29), 2, 1),
        # only division 0 reported; kept divisions are genuine zeros
        (dt.date(2024, 12, 30), 1, 0),
        (dt.date(2024, 12, 30), 2, 0),
        # no records anywhere: missing, not zero
        (dt.date(2024, 12, 31), 1, None),
        (dt.date(2024, 12, 31), 2, None),
        (dt.date(2025, 1, 1), 1, 0),
        (dt.date(2025, 1, 1), 2, 0),
        (dt.date(2025, 1, 2), 1, 1),
        (dt.date(2025, 1, 2), 2, 0),
    ]


def test_daily_counts_default_targets(df: pl.DataFrame) -> None:
    counts = data.daily_counts(df)
    assert counts.height == 5 * len(data.TARGET_DIVISIONS)
    assert set(counts["DIVISION"]) == set(data.TARGET_DIVISIONS)
    assert counts["n"].null_count() == len(data.TARGET_DIVISIONS)
    assert counts["n"].sum() == 5  # the division-0 row is dropped


def test_daily_counts_keep_none(df: pl.DataFrame) -> None:
    counts = data.daily_counts(df, keep=None)
    assert counts["DIVISION"].unique().sort().to_list() == [0, 1, 2, 3]
    assert counts.height == 5 * 4
    assert counts["n"].sum() == 6
    day = counts.filter(pl.col("date") == dt.date(2024, 12, 30))
    assert day["n"].to_list() == [1, 0, 0, 0]


def test_daily_counts_other_column(df: pl.DataFrame) -> None:
    counts = data.daily_counts(df, by="CASERNE", keep=None)
    assert counts.columns == ["date", "CASERNE", "n"]
    assert counts["CASERNE"].unique().sort().to_list() == [0, 10]


def test_daily_counts_empty(df: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="empty"):
        data.daily_counts(df.clear())


def test_division_day_first_responders(df: pl.DataFrame) -> None:
    # One of division 1's two records on 2024-12-29 and its 2025-01-02 record are
    # first-responder calls; the blank-group record on 2024-12-29 is not.
    responder = (pl.col("DIVISION") == 1) & pl.col("has_time")
    marked = df.with_columns(
        DESCRIPTION_GROUPE=pl.when(responder)
        .then(pl.lit(data.FIRST_RESPONDER_GROUP))
        .otherwise("DESCRIPTION_GROUPE")
    )
    counts = data.division_day(marked, keep=(1, 2))
    assert counts.columns == ["date", "DIVISION", "n", "n_fr"]
    assert counts.schema["n_fr"] == pl.UInt32
    assert counts.drop("n_fr").equals(data.daily_counts(marked, keep=(1, 2)))
    assert counts.select("n_fr").to_series().to_list() == [
        1, 0,  # 2024-12-29
        0, 0,  # 2024-12-30: only division 0 reported
        None, None,  # 2024-12-31: missing day stays missing
        0, 0,
        1, 0,
    ]  # fmt: skip
