"""Score the final model once on the held-out test period.

``FINAL_MODEL`` (``ens_mean``, the mean of the ``ENSEMBLE_CANDIDATES``) was
chosen on 2022-2024 validation. Here it meets the test period (``TEST_START``
to the last day of the history) for the first and only time. The candidates
and both baselines run through ``rolling_origin`` with ``allow_test``; the
``ens_mean`` rows are then the mean of the candidates' forecasts per
``(issue_date, DIVISION)``, which equals running ``Ensemble`` itself but fits
each candidate once.

The test is split at ``BREAK_DATE``, the late-2025 first-responder drop, so a
result driven by the break shows up as a difference between ``before_break``
and ``after_break``. ``run`` refuses to overwrite an existing report unless
``--force`` is given. Usage::

    uv run python -m ycit440_sim_forecast.holdout run
"""

import datetime as dt
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Any

import polars as pl
import typer

from ycit440_sim_forecast.baseline import PlainMean, SameWeekdayMean
from ycit440_sim_forecast.ensemble import (
    FINAL_MODEL,
    KEYS,
    OUTPUT_SCHEMA,
    EnsembleConfig,
    combine,
    wide_predictions,
)
from ycit440_sim_forecast.evaluate import (
    DEFAULT_ANOMALIES,
    DEFAULT_HISTORY,
    TEST_START,
    flag_anomalies,
    paired_bootstrap,
    rolling_origin,
    score,
)
from ycit440_sim_forecast.manifest import sha256_file
from ycit440_sim_forecast.models import (
    ENSEMBLE_CANDIDATES,
    LIBRARIES,
    REFERENCES,
    default_models,
    rounded,
)
from ycit440_sim_forecast.seed import DEFAULT_SEED
from ycit440_sim_forecast.spec import V2_DAILY, ForecastSpec

BEST_VALIDATION_SINGLE = "lgbm_poisson_raw"
"""Best single candidate on 2022-2024 validation: lowest overall MAE in
``reports/ensemble_validation.json`` (``bootstrap.best_single``). Fixed before
the test run, so it is not picked in hindsight on the test period."""

BREAK_DATE = dt.date(2025, 12, 20)
"""Late-2025 first-responder drop, as annotated in ``notebooks/01_eda.ipynb``;
target days from this date on are ``after_break``."""

CONTEXT_REFERENCE = "same_wd_13w"
"""Baseline every candidate is also compared against, for context."""

WHOLE = "test"
BEFORE = "before_break"
AFTER = "after_break"
PERIODS: tuple[str, ...] = (BEFORE, AFTER)
"""Sub-periods of the test, split at ``break_date`` on ``target_start``."""

DEFAULT_OUT = Path("reports/holdout_test.json")
DEFAULT_PREDICTIONS = Path("data/processed/holdout_predictions_v2.parquet")

app = typer.Typer(help=__doc__, no_args_is_help=True)


@dataclass(frozen=True, kw_only=True)
class HoldoutConfig:
    """Settings of the one-shot test run.

    Attributes:
        end: Last target day scored, the last day of the history.
        spec: Forecast setting.
        final_model: Model under test; only the mean ensemble is supported.
        candidates: Members of the mean, in column order.
        references: Baselines the final model is compared against.
        best_validation_single: Best single candidate on validation.
        start: First target day scored.
        break_date: First target day of ``after_break``.
        n_boot: Bootstrap resamples per comparison.
        seed: Seed of the models and the bootstrap.
    """

    end: dt.date
    spec: ForecastSpec = V2_DAILY
    final_model: str = FINAL_MODEL
    candidates: tuple[str, ...] = ENSEMBLE_CANDIDATES
    references: tuple[str, ...] = REFERENCES
    best_validation_single: str = BEST_VALIDATION_SINGLE
    start: dt.date = TEST_START
    break_date: dt.date = BREAK_DATE
    n_boot: int = 2000
    seed: int = DEFAULT_SEED

    def __post_init__(self) -> None:
        """Reject settings that are not a test run of the mean ensemble.

        Raises:
            ValueError: If ``final_model`` is not ``FINAL_MODEL``, candidates
                are empty, repeated or clash with a reference, a reference is
                not one of ``REFERENCES``, ``best_validation_single`` is not a
                candidate, ``start`` is before ``TEST_START``, the break is not
                strictly after ``start`` and on or before ``end``, or
                ``n_boot`` is below 1.
        """
        if self.final_model != FINAL_MODEL:
            msg = f"only {FINAL_MODEL!r} is supported, got {self.final_model!r}"
            raise ValueError(msg)
        if not self.candidates or len(set(self.candidates)) != len(self.candidates):
            msg = f"candidates must be non-empty and unique, got {self.candidates}"
            raise ValueError(msg)
        unknown = sorted(set(self.references) - set(REFERENCES))
        if (
            not self.references
            or unknown
            or len(set(self.references)) != len(self.references)
        ):
            msg = (
                f"references must be non-empty, unique, in {REFERENCES}: "
                f"{self.references}"
            )
            raise ValueError(msg)
        clashes = sorted({*self.references, self.final_model} & set(self.candidates))
        if clashes:
            msg = f"candidates clash with the references or the final model: {clashes}"
            raise ValueError(msg)
        if self.best_validation_single not in self.candidates:
            msg = (
                f"best_validation_single {self.best_validation_single!r} "
                "is not a candidate"
            )
            raise ValueError(msg)
        if self.start < TEST_START:
            msg = f"start {self.start} is before the test period ({TEST_START})"
            raise ValueError(msg)
        if not self.start < self.break_date <= self.end:
            msg = (
                f"need start < break_date <= end, got {self.start}, "
                f"{self.break_date} and {self.end}"
            )
            raise ValueError(msg)
        if self.n_boot < 1:
            msg = f"n_boot must be >= 1, got {self.n_boot}"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready settings.

        Returns:
            Every field, dates as ISO strings and the spec as its fields and key.
        """
        return {
            "spec": {
                "key": self.spec.key,
                "publication_lag_days": self.spec.publication_lag_days,
                "lead_days": self.spec.lead_days,
                "window_days": self.spec.window_days,
            },
            "final_model": self.final_model,
            "candidates": list(self.candidates),
            "references": list(self.references),
            "best_validation_single": self.best_validation_single,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "break_date": self.break_date.isoformat(),
            "n_boot": self.n_boot,
            "seed": self.seed,
        }

    @property
    def config_id(self) -> str:
        """First 12 hex digits of the SHA-256 of the canonical ``to_dict`` JSON."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def mean_rows(results: pl.DataFrame, config: HoldoutConfig) -> pl.DataFrame:
    """``final_model`` rows: the mean of the candidates' forecasts per key.

    Args:
        results: Rows with ``OUTPUT_SCHEMA`` holding every candidate.
        config: Run settings; ``spec``, ``candidates`` and ``final_model``
            are used.

    Returns:
        One row per ``(issue_date, DIVISION)`` with ``OUTPUT_SCHEMA``, named
        ``final_model``, sorted by key; ``y_pred`` is null when any
        candidate's is.

    Raises:
        ValueError: From ``ensemble.wide_predictions``.
    """
    members = EnsembleConfig(spec=config.spec, candidates=config.candidates)
    wide = wide_predictions(results, members)
    names = list(config.candidates)
    values = combine(wide.select(names).to_numpy(), names, "mean")
    return (
        wide.with_columns(
            model=pl.lit(config.final_model),
            y_pred=pl.Series(values, dtype=pl.Float64, nan_to_null=True),
        )
        .select(OUTPUT_SCHEMA.names())
        .cast(OUTPUT_SCHEMA)
    )


def holdout_results(
    history: pl.DataFrame, anomalies: pl.DataFrame, config: HoldoutConfig
) -> pl.DataFrame:
    """Forecast the test period with the candidates, both baselines and the mean.

    Args:
        history: ``date, DIVISION, n, n_fr`` on the full calendar.
        anomalies: ``date, DIVISION, anomaly`` (the EDA anomaly calendar).
        config: Run settings.

    Returns:
        Rows with ``OUTPUT_SCHEMA`` for each candidate, ``same_wd_13w``,
        ``plain_28d`` and ``final_model``, sorted by model and key.

    Raises:
        ValueError: If a candidate is not a default model name, or from
            ``rolling_origin`` or ``mean_rows``.
    """
    available = {m.name: m for m in default_models(config.seed)}
    unknown = sorted(set(config.candidates) - set(available))
    if unknown:
        msg = f"unknown candidates {unknown}; default models are {sorted(available)}"
        raise ValueError(msg)
    forecasters = [
        *(available[name] for name in config.candidates),
        SameWeekdayMean(13),
        PlainMean(),
    ]
    singles = flag_anomalies(
        rolling_origin(
            history,
            forecasters,
            config.spec,
            config.start,
            config.end,
            allow_test=True,
        ),
        anomalies,
    ).select(OUTPUT_SCHEMA.names())
    return pl.concat([singles, mean_rows(singles, config)]).sort("model", *KEYS)


def with_period(results: pl.DataFrame, break_date: dt.date) -> pl.DataFrame:
    """Add ``period`` (``before_break`` or ``after_break``) and ``year_month``.

    Args:
        results: Rows with ``target_start``.
        break_date: First target day of ``after_break``.

    Returns:
        ``results`` with ``period`` from ``target_start`` against the break
        and ``year_month`` as ``YYYY-MM`` of ``target_start``.
    """
    return results.with_columns(
        period=pl.when(pl.col("target_start") < break_date)
        .then(pl.lit(BEFORE))
        .otherwise(pl.lit(AFTER)),
        year_month=pl.col("target_start").dt.strftime("%Y-%m"),
    )


def holdout_report(
    history: pl.DataFrame,
    anomalies: pl.DataFrame,
    config: HoldoutConfig,
    input_sha256: str,
) -> tuple[dict[str, Any], pl.DataFrame]:
    """Score the final model, the candidates and the baselines on the test.

    Args:
        history: ``date, DIVISION, n, n_fr`` on the full calendar.
        anomalies: ``date, DIVISION, anomaly`` (the EDA anomaly calendar).
        config: Run settings.
        input_sha256: SHA-256 of the history file.

    Returns:
        The report (config, config_id, input hash, period, score tables
        overall, without anomalies, by division, by year-month, by period and
        by period and division; paired bootstraps of ``final_model`` against
        each reference and ``best_validation_single`` and of each candidate
        against ``same_wd_13w``, for the whole test and each period; library
        versions; floats rounded to 3) and the rows with ``OUTPUT_SCHEMA``.
        Every score table has ``n``, ``mae``, ``bias``, ``median_error`` and
        ``under_share``.

    Raises:
        ValueError: From ``holdout_results``, or if a period has no scored row.
    """
    results = holdout_results(history, anomalies, config)
    labelled = with_period(results, config.break_date)
    scored = labelled.filter(pl.col("y_true").is_not_null())
    present = set(scored["period"].unique().to_list())
    empty = [p for p in PERIODS if p not in present]
    if empty:
        msg = f"no scored rows in {empty} (break_date {config.break_date})"
        raise ValueError(msg)

    def table(frame: pl.DataFrame, by: Sequence[str] = ()) -> list[dict[str, Any]]:
        return score(frame, by).to_dicts()

    def boot(frame: pl.DataFrame, model: str, reference: str) -> dict[str, Any]:
        return paired_bootstrap(frame, model, reference, config.n_boot, config.seed)

    frames = {
        WHOLE: labelled,
        **{p: labelled.filter(pl.col("period") == p) for p in PERIODS},
    }
    against = [*config.references, config.best_validation_single]
    report = {
        "config": config.to_dict(),
        "config_id": config.config_id,
        "input_sha256": input_sha256,
        "final_model": config.final_model,
        "period": {
            "start": config.start.isoformat(),
            "end": config.end.isoformat(),
            "break_date": config.break_date.isoformat(),
        },
        "overall": table(labelled),
        "overall_without_anomalies": table(labelled.filter(~pl.col("anomaly"))),
        "by_division": table(labelled, ["DIVISION"]),
        "by_year_month": table(labelled, ["year_month"]),
        "by_period": table(labelled, ["period"]),
        "by_period_division": table(labelled, ["period", "DIVISION"]),
        "bootstrap": {
            "final": {
                name: {ref: boot(frame, config.final_model, ref) for ref in against}
                for name, frame in frames.items()
            },
            "candidates_vs_reference": {
                name: [boot(frame, c, CONTEXT_REFERENCE) for c in config.candidates]
                for name, frame in frames.items()
            },
            "best_validation_single": {
                "model": config.best_validation_single,
                "selection": (
                    "lowest overall MAE among the candidates on 2022-2024 "
                    "validation, fixed before the test run"
                ),
            },
        },
        "versions": {lib: version(lib) for lib in LIBRARIES},
    }
    return rounded(report), results


def format_summary(report: dict[str, Any]) -> list[str]:
    """Compact text summary of a ``holdout_report`` result.

    Args:
        report: Output of ``holdout_report``.

    Returns:
        Lines: the run, overall MAE and bias per model, MAE before and after
        the break per model, and the bootstraps of the final model.
    """
    period = report["period"]
    lines = [
        f"config {report['config_id']}, test {period['start']} to {period['end']}, "
        f"break {period['break_date']}"
    ]
    by_period = {(r["model"], r["period"]): r for r in report["by_period"]}
    for row in report["overall"]:
        before, after = (by_period[row["model"], p]["mae"] for p in PERIODS)
        lines.append(
            f"{row['model']:<20} MAE {row['mae']:6.3f}  bias {row['bias']:+6.3f}  "
            f"MAE before break {before:6.3f}  after {after:6.3f}"
        )
    for name, comparisons in report["bootstrap"]["final"].items():
        for reference, row in comparisons.items():
            lines.append(
                f"{name:<13} {row['model']} vs {reference:<18} diff "
                f"{row['diff']:+6.3f}  95% CI [{row['ci_low']:+6.3f}, "
                f"{row['ci_high']:+6.3f}]"
            )
    return lines


@app.callback()
def main() -> None:
    """Score the final model once on the held-out test period."""


@app.command()
def run(
    history_path: Annotated[
        Path, typer.Option("--history", help="Division-day parquet.")
    ] = DEFAULT_HISTORY,
    anomalies_path: Annotated[
        Path, typer.Option("--anomalies", help="Anomaly calendar parquet.")
    ] = DEFAULT_ANOMALIES,
    out: Annotated[Path, typer.Option(help="JSON report path.")] = DEFAULT_OUT,
    predictions: Annotated[
        Path, typer.Option(help="Parquet path for the test predictions.")
    ] = DEFAULT_PREDICTIONS,
    force: Annotated[
        bool, typer.Option("--force", help="Overwrite an existing report.")
    ] = False,
) -> None:
    """Score ens_mean once on the test period (TEST_START to the last day).

    Args:
        history_path: Division-day parquet.
        anomalies_path: Anomaly calendar parquet.
        out: JSON report path; must not exist unless ``force``.
        predictions: Parquet path for the test predictions.
        force: Overwrite an existing report.

    Raises:
        typer.Exit: If the report exists without ``--force``, or an input
            file is missing.
    """
    if out.exists() and not force:
        typer.echo(
            f"{out} already exists: the test period is scored once. Refusing to "
            "run again; pass --force only if you mean to replace that result.",
            err=True,
        )
        raise typer.Exit(code=1)
    for path in (history_path, anomalies_path):
        if not path.is_file():
            typer.echo(f"{path} not found; run notebooks/01_eda.ipynb first", err=True)
            raise typer.Exit(code=1)
    history = pl.read_parquet(history_path)
    end: dt.date = history["date"].max()  # pyright: ignore[reportAssignmentType]
    config = HoldoutConfig(end=end)
    report, rows = holdout_report(
        history, pl.read_parquet(anomalies_path), config, sha256_file(history_path)
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    predictions.parent.mkdir(parents=True, exist_ok=True)
    rows.write_parquet(predictions)
    for line in format_summary(report):
        typer.echo(line)
    typer.echo(f"wrote {out} and {predictions}")


if __name__ == "__main__":
    app()
