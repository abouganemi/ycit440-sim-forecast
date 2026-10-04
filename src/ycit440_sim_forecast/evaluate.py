"""Score forecasters by rolling origin, with the cutoff enforced on every forecast.

For each issue date the harness cuts ``history`` at that date's cutoff and only
then calls the forecaster, so nothing it computes can read later counts.
Forecasters are refitted at the first issue date of each month. Target days on
or after ``TEST_START`` are refused unless explicitly allowed, so the test
period stays untouched during model selection. Usage::

    uv run python -m ycit440_sim_forecast.evaluate baselines
"""

import datetime as dt
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import polars as pl
import typer

from ycit440_sim_forecast.baseline import PlainMean, SameWeekdayMean
from ycit440_sim_forecast.seed import DEFAULT_SEED
from ycit440_sim_forecast.spec import (
    V2_DAILY,
    WEEKLY,
    Forecaster,
    ForecastSpec,
    daily_horizon,
)

VALIDATION_START = dt.date(2021, 1, 1)
VALIDATION_END = dt.date(2024, 12, 31)
TEST_START = dt.date(2025, 1, 1)
"""First day of the held-out test period (2025-01-01 to the end of the data)."""

SAME_WEEKDAY_WEEKS: tuple[int, ...] = (4, 8, 13, 26, 52)
DEFAULT_HISTORY = Path("data/processed/division_day.parquet")
DEFAULT_ANOMALIES = Path("data/processed/anomaly_days.parquet")
DEFAULT_OUT = Path("reports/baseline_validation.json")
_CLI_START = dt.datetime.combine(VALIDATION_START, dt.time())
_CLI_END = dt.datetime.combine(VALIDATION_END, dt.time())

RESULT_SCHEMA = pl.Schema(
    {
        "spec": pl.String,
        "model": pl.String,
        "issue_date": pl.Date,
        "cutoff": pl.Date,
        "target_start": pl.Date,
        "target_end": pl.Date,
        "DIVISION": pl.Int64,
        "y_true": pl.Float64,
        "y_pred": pl.Float64,
    }
)

app = typer.Typer(help=__doc__, no_args_is_help=True)


def validate_history(history: pl.DataFrame) -> None:
    """Check the ``date, DIVISION, n`` contract the forecasters rely on.

    Raises:
        ValueError: If columns or dtypes are wrong, a key repeats, or a
            division's calendar has a gap (a missing day must be a null row).
    """
    expected = {"date": pl.Date, "DIVISION": pl.Int64}
    for column, dtype in expected.items():
        if history.schema.get(column) != dtype:
            msg = f"history.{column} must be {dtype}, got {history.schema.get(column)}"
            raise ValueError(msg)
    if "n" not in history.columns or not history.schema["n"].is_integer():
        msg = "history.n must be an integer column"
        raise ValueError(msg)
    if history.select(pl.struct("date", "DIVISION").is_duplicated().any()).item():
        msg = "history has repeated (date, DIVISION) rows"
        raise ValueError(msg)
    span = history.group_by("DIVISION").agg(
        rows=pl.len(),
        days=(pl.col("date").max() - pl.col("date").min()).dt.total_days() + 1,
    )
    if span.filter(pl.col("rows") != pl.col("days")).height:
        msg = "history has calendar gaps; missing days must be rows with a null n"
        raise ValueError(msg)


def issue_dates(start: dt.date, end: dt.date, spec: ForecastSpec) -> list[dt.date]:
    """Issue dates whose whole target window lies in ``[start, end]``.

    Windows do not overlap: issue dates step by ``window_days``. Weekly
    windows start on Mondays, so the forecast is issued the day before the
    week (for lead 1).

    Raises:
        ValueError: If ``end`` is before ``start``.
    """
    if end < start:
        msg = f"end {end} is before start {start}"
        raise ValueError(msg)
    first = start
    if spec.window_days == 7:
        first += dt.timedelta(days=-start.weekday() % 7)
    step = dt.timedelta(days=spec.window_days)
    dates: list[dt.date] = []
    while first + step - dt.timedelta(days=1) <= end:
        dates.append(spec.issue_date_for(first))
        first += step
    return dates


def actuals(
    history: pl.DataFrame, dates: Sequence[dt.date], spec: ForecastSpec
) -> pl.DataFrame:
    """Observed totals over each forecast's window.

    Returns:
        ``issue_date, DIVISION, y_true``; ``y_true`` is null when any day of the
        window is missing (null or outside ``history``).
    """
    windows = pl.DataFrame(
        {
            "issue_date": [d for d in dates for _ in range(spec.window_days)],
            "date": [day for d in dates for day in spec.target_days(d)],
        },
        schema={"issue_date": pl.Date, "date": pl.Date},
    )
    divisions = history.select(pl.col("DIVISION").unique())
    return (
        windows.join(divisions, how="cross")
        .join(
            history.select("date", "DIVISION", "n"), on=["date", "DIVISION"], how="left"
        )
        .group_by("issue_date", "DIVISION")
        .agg(
            y_true=pl.when(pl.col("n").null_count() == 0).then(
                pl.col("n").sum().cast(pl.Float64)
            )
        )
    )


def rolling_origin(
    history: pl.DataFrame,
    forecasters: Iterable[Forecaster],
    spec: ForecastSpec,
    start: dt.date = VALIDATION_START,
    end: dt.date = VALIDATION_END,
    *,
    allow_test: bool = False,
) -> pl.DataFrame:
    """Forecast every window in ``[start, end]`` as it would have been issued.

    Args:
        history: ``date, DIVISION, n`` on the full calendar.
        forecasters: Forecasters to score; names must be unique.
        spec: Forecast setting.
        start: First target day scored.
        end: Last target day scored.
        allow_test: Permit target days on or after ``TEST_START``.

    Returns:
        One row per model, issue date and division, with ``RESULT_SCHEMA``.
        ``y_true`` is null for windows touching a missing day.

    Raises:
        ValueError: If the period reaches the test set without ``allow_test``,
            names repeat, a forecaster returns the wrong divisions, or it gives
            a null forecast for a window that will be scored.
    """
    validate_history(history)
    if end >= TEST_START and not allow_test:
        msg = f"end {end} reaches the test period ({TEST_START}+); set allow_test"
        raise ValueError(msg)
    models = list(forecasters)
    names = [m.name for m in models]
    if len(set(names)) != len(names):
        msg = f"forecaster names must be unique, got {names}"
        raise ValueError(msg)

    history = history.sort("date", "DIVISION")
    dates = issue_dates(start, end, spec)
    truth = actuals(history, dates, spec)
    divisions = history["DIVISION"].unique().sort().to_list()
    frames: list[pl.DataFrame] = []
    fitted_month: tuple[int, int] | None = None

    for issue in dates:
        cutoff = spec.cutoff(issue)
        visible = history.slice(0, history["date"].search_sorted(cutoff, side="right"))
        if (issue.year, issue.month) != fitted_month:
            for model in models:
                model.fit(visible, spec)
            fitted_month = (issue.year, issue.month)
        for model in models:
            pred = model.predict(visible, issue, spec).sort("DIVISION")
            if pred["DIVISION"].to_list() != divisions:
                msg = f"{model.name} on {issue}: expected divisions {divisions}"
                raise ValueError(msg)
            frames.append(
                pred.select(
                    pl.lit(spec.key).alias("spec"),
                    pl.lit(model.name).alias("model"),
                    pl.lit(issue).alias("issue_date"),
                    pl.lit(cutoff).alias("cutoff"),
                    pl.lit(spec.target_start(issue)).alias("target_start"),
                    pl.lit(spec.target_end(issue)).alias("target_end"),
                    "DIVISION",
                    pl.col("y_pred").cast(pl.Float64),
                )
            )

    if not frames:
        return pl.DataFrame(schema=RESULT_SCHEMA)
    results = (
        pl.concat(frames)
        .join(truth, on=["issue_date", "DIVISION"], how="left")
        .select(RESULT_SCHEMA.names())
        .cast(RESULT_SCHEMA)
    )
    unscorable = results.filter(
        pl.col("y_true").is_not_null() & pl.col("y_pred").is_null()
    )
    if unscorable.height:
        first = unscorable.row(0, named=True)
        msg = (
            f"{unscorable.height} null forecast(s) on scored windows, first: "
            f"{first['model']}, division {first['DIVISION']}, {first['issue_date']}"
        )
        raise ValueError(msg)
    return results


def flag_anomalies(results: pl.DataFrame, anomalies: pl.DataFrame) -> pl.DataFrame:
    """Add ``anomaly``: true when any target day of the row is an anomaly day.

    Args:
        results: Output of ``rolling_origin``.
        anomalies: ``date, DIVISION, anomaly`` (the EDA anomaly calendar).
    """
    flagged = anomalies.filter(pl.col("anomaly")).select("date", "DIVISION")
    keys = results.select(
        "issue_date", "target_start", "target_end", "DIVISION"
    ).unique()
    hits = (
        keys.join(flagged, on="DIVISION")
        .filter(pl.col("date").is_between(pl.col("target_start"), pl.col("target_end")))
        .select("issue_date", "DIVISION")
        .unique()
        .with_columns(anomaly=pl.lit(value=True))
    )
    return results.join(hits, on=["issue_date", "DIVISION"], how="left").with_columns(
        pl.col("anomaly").fill_null(value=False)
    )


def score(results: pl.DataFrame, by: Sequence[str] = ()) -> pl.DataFrame:
    """MAE, bias, median error and share under actual per model and ``by`` group.

    Errors are forecast minus actual. MAE is minimised by the median, so on
    right-skewed counts a well-calibrated forecast shows a negative mean bias;
    ``median_error`` and ``under_share`` show whether a model is centred on the
    median. Rows with a null ``y_true`` are not scored. ``by`` may include
    ``month`` (calendar month of the window start) and any column of
    ``results``.

    Args:
        results: Output of ``rolling_origin``.
        by: Grouping columns besides ``model``.

    Returns:
        ``model, *by, n, mae, bias, median_error, under_share`` sorted by model
        then ``by``; ``under_share`` is the share of scored rows with
        ``y_pred < y_true``.
    """
    scored = results.filter(pl.col("y_true").is_not_null()).with_columns(
        month=pl.col("target_start").dt.month()
    )
    error = pl.col("y_pred") - pl.col("y_true")
    return (
        scored.group_by("model", *by)
        .agg(
            n=pl.len(),
            mae=error.abs().mean(),
            bias=error.mean(),
            median_error=error.median(),
            under_share=(pl.col("y_pred") < pl.col("y_true")).mean(),
        )
        .sort("model", *by)
    )


def paired_bootstrap(
    results: pl.DataFrame,
    model: str,
    reference: str,
    n_boot: int = 2000,
    seed: int = DEFAULT_SEED,
    block: Literal["week"] = "week",
) -> dict[str, Any]:
    """Block bootstrap of the MAE difference between two models.

    Rows are paired on ``(issue_date, DIVISION)`` and kept when both models
    have a non-null ``y_true``. Whole ISO weeks of ``target_start`` are
    resampled with replacement, all divisions together, because errors are
    autocorrelated in time and shared across divisions on shock days.

    Args:
        results: Output of ``rolling_origin`` holding both models.
        model: Model whose MAE is compared.
        reference: Model it is compared against.
        n_boot: Number of bootstrap resamples.
        seed: Seed of the NumPy generator.
        block: Resampling unit; only ISO ``week`` is supported.

    Returns:
        ``model``, ``reference``, ``mae_model``, ``mae_reference``, ``diff``
        (MAE of ``model`` minus MAE of ``reference``), ``ci_low`` and
        ``ci_high`` (95% percentile interval of the difference) and
        ``n_blocks``.

    Raises:
        ValueError: If ``block`` is not ``week`` or no rows pair up.
    """
    if block != "week":
        msg = f"block must be 'week', got {block!r}"
        raise ValueError(msg)
    keys = ["issue_date", "DIVISION"]

    def errors(name: str) -> pl.DataFrame:
        return results.filter(
            pl.col("model") == name, pl.col("y_true").is_not_null()
        ).select(*keys, "target_start", (pl.col("y_pred") - pl.col("y_true")).abs())

    paired = (
        errors(model)
        .rename({"y_pred": "err_model"})
        .join(
            errors(reference).drop("target_start").rename({"y_pred": "err_ref"}),
            on=keys,
        )
        .with_columns(
            week=pl.col("target_start").dt.iso_year() * 100
            + pl.col("target_start").dt.week()
        )
    )
    if paired.is_empty():
        msg = f"no scored rows shared by {model} and {reference}"
        raise ValueError(msg)
    # Sorted so a seed always draws the same weeks (group_by order is random).
    blocks = (
        paired.group_by("week")
        .agg(
            model=pl.col("err_model").sum(),
            ref=pl.col("err_ref").sum(),
            count=pl.len(),
        )
        .sort("week")
    )
    sum_model, sum_ref, count = (
        blocks[c].to_numpy().astype(np.float64) for c in ("model", "ref", "count")
    )
    # Each resample draws as many weeks as there are; weeks hold different row
    # counts, so the difference is a ratio of resampled sums.
    draws = np.random.default_rng(seed).integers(
        0, blocks.height, size=(n_boot, blocks.height)
    )
    boot = (sum_model[draws] - sum_ref[draws]).sum(axis=1) / count[draws].sum(axis=1)
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
    mae_model = float(sum_model.sum() / count.sum())
    mae_ref = float(sum_ref.sum() / count.sum())
    return {
        "model": model,
        "reference": reference,
        "mae_model": mae_model,
        "mae_reference": mae_ref,
        "diff": mae_model - mae_ref,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_blocks": blocks.height,
    }


def select_window(results: pl.DataFrame, prefix: str = "same_wd_") -> str:
    """Name of the ``prefix`` model with the lowest overall MAE.

    Ties go to the shorter window, so the choice is deterministic.

    Raises:
        ValueError: If no model name starts with ``prefix``.
    """
    table = score(results.filter(pl.col("model").str.starts_with(prefix))).with_columns(
        length=pl.col("model").str.extract(r"(\d+)").cast(pl.Int64)
    )
    if table.is_empty():
        msg = f"no model named {prefix}*"
        raise ValueError(msg)
    return table.sort("mae", "length").row(0, named=True)["model"]


def baseline_report(
    history: pl.DataFrame,
    anomalies: pl.DataFrame,
    specs: Sequence[ForecastSpec],
    start: dt.date = VALIDATION_START,
    end: dt.date = VALIDATION_END,
) -> dict[str, Any]:
    """Validate both baselines for each setting and choose the same-weekday window.

    Returns:
        Per setting: the chosen window, then overall, per-division and
        per-month MAE, bias, median error and share under actual for the
        chosen same-weekday model and the plain 28-day mean, with and without
        windows touching an anomaly day, plus the overall score of every
        candidate window.
    """
    report: dict[str, Any] = {
        "period": {"start": str(start), "end": str(end)},
        "settings": {},
    }
    for spec in specs:
        candidates = [SameWeekdayMean(weeks=w) for w in SAME_WEEKDAY_WEEKS]
        results = flag_anomalies(
            rolling_origin(history, [*candidates, PlainMean()], spec, start, end),
            anomalies,
        )
        chosen = select_window(results)
        kept = results.filter(pl.col("model").is_in([chosen, PlainMean().name]))
        clean = kept.filter(~pl.col("anomaly"))

        def table(frame: pl.DataFrame, by: Sequence[str] = ()) -> list[dict[str, Any]]:
            return (
                score(frame, by)
                .with_columns(
                    pl.col("mae", "bias", "median_error", "under_share").round(3)
                )
                .to_dicts()
            )

        report["settings"][spec.key] = {
            "spec": {
                "publication_lag_days": spec.publication_lag_days,
                "lead_days": spec.lead_days,
                "window_days": spec.window_days,
            },
            "chosen_same_weekday": chosen,
            "candidates": table(results),
            "overall": table(kept),
            "overall_without_anomalies": table(clean),
            "by_division": table(kept, ["DIVISION"]),
            "by_division_without_anomalies": table(clean, ["DIVISION"]),
            "by_month": table(kept, ["month"]),
        }
    return report


@app.callback()
def main() -> None:
    """Rolling-origin validation of forecasters."""


@app.command()
def baselines(
    history_path: Annotated[
        Path, typer.Option("--history", help="Division-day parquet.")
    ] = DEFAULT_HISTORY,
    anomalies_path: Annotated[
        Path, typer.Option("--anomalies", help="Anomaly calendar parquet.")
    ] = DEFAULT_ANOMALIES,
    out: Annotated[Path, typer.Option(help="JSON report path.")] = DEFAULT_OUT,
    horizon: Annotated[int, typer.Option(help="Daily leads 1..N to add.")] = 7,
    start: Annotated[
        dt.datetime, typer.Option(formats=["%Y-%m-%d"], help="First target day.")
    ] = _CLI_START,
    end: Annotated[
        dt.datetime, typer.Option(formats=["%Y-%m-%d"], help="Last target day.")
    ] = _CLI_END,
) -> None:
    """Validate both baselines (2021-2024 by default) for v2, weekly and leads 2..N."""
    for path in (history_path, anomalies_path):
        if not path.is_file():
            typer.echo(f"{path} not found; run notebooks/01_eda.ipynb first", err=True)
            raise typer.Exit(code=1)
    specs = [V2_DAILY, WEEKLY, *daily_horizon(horizon)[1:]]
    report = baseline_report(
        pl.read_parquet(history_path),
        pl.read_parquet(anomalies_path),
        specs,
        start.date(),
        end.date(),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for key, setting in report["settings"].items():
        overall = ", ".join(
            f"{row['model']} {row['mae']:.2f}" for row in setting["overall"]
        )
        typer.echo(f"{key}: MAE {overall}")
    typer.echo(f"wrote {out}")


if __name__ == "__main__":
    app()
