"""Combine the candidate models into ensembles and validate them out of sample.

Three combiners work on a matrix of member forecasts, one column per candidate
in a fixed order: the ``mean`` and the ``median`` learn nothing; ``weighted``
uses non-negative weights summing to 1 that minimise MAE, found exactly by a
linear program.

``combine_oof`` scores the combiners on the out-of-fold predictions that
``models validate`` wrote. Weights are refitted at the first issue date of each
month, like ``rolling_origin`` refits models, on every earlier row whose target
was known by that date's cutoff, so no weight ever sees the forecast it is
applied to. Scoring starts in 2022 so the weights have at least a year of
history. ``Ensemble`` is the same combination as a ``Forecaster`` for later
live or test use. Usage::

    uv run python -m ycit440_sim_forecast.ensemble validate
"""

import copy
import datetime as dt
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import numpy as np
import polars as pl
import typer
from scipy import sparse
from scipy.optimize import linprog

from ycit440_sim_forecast.evaluate import (
    RESULT_SCHEMA,
    TEST_START,
    VALIDATION_END,
    paired_bootstrap,
    score,
)
from ycit440_sim_forecast.manifest import sha256_file
from ycit440_sim_forecast.models import (
    DEFAULT_PREDICTIONS,
    ENSEMBLE_CANDIDATES,
    REFERENCES,
    default_models,
    rounded,
)
from ycit440_sim_forecast.seed import DEFAULT_SEED
from ycit440_sim_forecast.spec import V2_DAILY, Forecaster, ForecastSpec

type Method = Literal["mean", "median", "weighted"]
METHODS: tuple[Method, ...] = ("mean", "median", "weighted")
"""Known combiners, in report order."""

SCORE_START = dt.date(2022, 1, 1)
"""First scored target day: 2021 only serves as weight history."""

FINAL_MODEL = "ens_mean"
"""Model taken to the test set, chosen on 2022-2024 before any test run: lowest
MAE, and nothing learned; the learned weights did not beat it."""

DEFAULT_OUT = Path("reports/ensemble_validation.json")
DEFAULT_ENSEMBLE_PREDICTIONS = Path("data/processed/ensemble_predictions_v2.parquet")
KEYS: tuple[str, ...] = ("issue_date", "DIVISION")
"""Columns identifying one forecast."""

OUTPUT_SCHEMA = pl.Schema({**RESULT_SCHEMA, "anomaly": pl.Boolean})
"""Schema of the out-of-fold predictions, read and written."""

_META: tuple[str, ...] = tuple(
    c for c in OUTPUT_SCHEMA.names() if c not in {"model", "y_pred"}
)
_RESERVED = frozenset(OUTPUT_SCHEMA.names())
_WEIGHT_FLOOR = 1e-12
"""Solver weights below this are numerical noise and set to 0."""

_CLI_START = dt.datetime.combine(SCORE_START, dt.time())
_CLI_END = dt.datetime.combine(VALIDATION_END, dt.time())

app = typer.Typer(help=__doc__, no_args_is_help=True)


def combine_mean(preds: np.ndarray) -> np.ndarray:
    """Row mean of the member forecasts.

    Args:
        preds: ``(rows, members)`` forecasts; NaN marks a missing forecast.

    Returns:
        One value per row, NaN when any member is NaN.
    """
    return preds.mean(axis=1)


def combine_median(preds: np.ndarray) -> np.ndarray:
    """Row median of the member forecasts.

    Args:
        preds: ``(rows, members)`` forecasts; NaN marks a missing forecast.

    Returns:
        One value per row, NaN when any member is NaN.
    """
    return np.median(preds, axis=1)


def combine_weighted(preds: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted sum of the member forecasts.

    Args:
        preds: ``(rows, members)`` forecasts; NaN marks a missing forecast.
        weights: One weight per member, in column order.

    Returns:
        ``preds @ weights``, NaN when any member is NaN (even at weight 0).
    """
    return preds @ weights


def weight_vector(names: Sequence[str], weights: Mapping[str, float]) -> np.ndarray:
    """Weights in the order of ``names``, checked by name and scaled to sum to 1.

    Args:
        names: Member names in column order.
        weights: One weight per member, keyed by name.

    Returns:
        The weights as an array aligned with ``names``, divided by their sum.

    Raises:
        ValueError: If a name is missing or extra, or a weight is negative,
            non-finite, or all weights are zero.
    """
    missing = sorted(set(names) - set(weights))
    extra = sorted(set(weights) - set(names))
    if missing or extra:
        msg = (
            f"weights must match the members by name; missing {missing}, extra {extra}"
        )
        raise ValueError(msg)
    vector = np.array([float(weights[name]) for name in names], dtype=np.float64)
    if not np.isfinite(vector).all():
        msg = f"weights must be finite, got {dict(weights)}"
        raise ValueError(msg)
    if (vector < 0).any():
        msg = f"weights must be >= 0, got {dict(weights)}"
        raise ValueError(msg)
    total = vector.sum()
    if total == 0:
        msg = "weights are all zero"
        raise ValueError(msg)
    return vector / total


def fit_weights(
    preds: np.ndarray, y_true: np.ndarray, names: Sequence[str]
) -> dict[str, float]:
    """Non-negative weights summing to 1 that minimise the MAE of ``preds @ w``.

    Solved exactly as a linear program with HiGHS: minimise the mean of
    ``u + v`` subject to ``preds @ w + u - v = y_true``, ``sum(w) = 1`` and
    ``w, u, v >= 0``. HiGHS is deterministic, so the same input always gives
    the same weights; when several weight vectors reach the same MAE (for
    example two identical members), the one returned depends on the column
    order, which ``names`` fixes. Weights below 1e-12 (solver noise, including
    tiny negatives) are set to 0 and the rest rescaled to sum to 1.

    Args:
        preds: ``(rows, members)`` forecasts, all finite.
        y_true: Observed values, one per row, all finite.
        names: Member names in column order.

    Returns:
        Weight per member name, in the order of ``names``.

    Raises:
        ValueError: If shapes disagree, there are no rows, a value is not
            finite, or the solver fails.
    """
    rows, members = preds.shape
    if members != len(names) or y_true.shape != (rows,):
        msg = (
            f"preds {preds.shape}, y_true {y_true.shape} and {len(names)} names "
            "do not line up"
        )
        raise ValueError(msg)
    if rows == 0:
        msg = "no rows to fit weights on"
        raise ValueError(msg)
    if not (np.isfinite(preds).all() and np.isfinite(y_true).all()):
        msg = "preds and y_true must be finite to fit weights"
        raise ValueError(msg)
    identity = sparse.identity(rows, format="csr")
    a_eq = sparse.vstack(
        [
            sparse.hstack([sparse.csr_matrix(preds), identity, -identity]),
            sparse.hstack(
                [
                    sparse.csr_matrix(np.ones((1, members))),
                    sparse.csr_matrix((1, 2 * rows)),
                ]
            ),
        ],
        format="csr",
    )
    b_eq = np.append(y_true, 1.0)
    cost = np.concatenate([np.zeros(members), np.full(2 * rows, 1.0 / rows)])
    result = linprog(cost, A_eq=a_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    if not result.success:
        msg = f"weight linear program failed: {result.message}"
        raise ValueError(msg)
    weights = np.asarray(result.x[:members], dtype=np.float64)
    weights[weights < _WEIGHT_FLOOR] = 0.0
    weights /= weights.sum()
    return {name: float(w) for name, w in zip(names, weights, strict=True)}


def combine(
    preds: np.ndarray,
    names: Sequence[str],
    method: Method,
    weights: Mapping[str, float] | None = None,
) -> np.ndarray:
    """Apply one combiner to a ``(rows, members)`` forecast matrix.

    Args:
        preds: Member forecasts, columns in the order of ``names``.
        names: Member names in column order.
        method: Combiner.
        weights: Weights by member name; required iff ``method`` is
            ``weighted``.

    Returns:
        One combined forecast per row; NaN when any member is NaN.

    Raises:
        ValueError: If ``weights`` is given without ``weighted`` or missing
            with it, the weights fail ``weight_vector``, or the method is
            unknown.
    """
    if method == "weighted":
        if weights is None:
            msg = "the 'weighted' method needs weights"
            raise ValueError(msg)
        return combine_weighted(preds, weight_vector(names, weights))
    if weights is not None:
        msg = f"weights are only used by 'weighted', got method {method!r}"
        raise ValueError(msg)
    match method:
        case "mean":
            return combine_mean(preds)
        case "median":
            return combine_median(preds)
        case _:
            msg = f"unknown method {method!r}; expected one of {METHODS}"
            raise ValueError(msg)


@dataclass(frozen=True)
class EnsembleConfig:
    """Settings of one ensemble validation run.

    Attributes:
        spec: Forecast setting the predictions were made for.
        candidates: Member model names, in column order.
        methods: Combiners to score.
        score_start: First target day scored.
        score_end: Last target day scored; must be before ``TEST_START``.
        n_boot: Bootstrap resamples per comparison.
        seed: Seed of the bootstrap.
    """

    spec: ForecastSpec = V2_DAILY
    candidates: tuple[str, ...] = ENSEMBLE_CANDIDATES
    methods: tuple[Method, ...] = METHODS
    score_start: dt.date = SCORE_START
    score_end: dt.date = VALIDATION_END
    n_boot: int = 2000
    seed: int = DEFAULT_SEED

    def __post_init__(self) -> None:
        """Reject settings that cannot be run or would touch the test period.

        Raises:
            ValueError: If candidates are empty, repeated or clash with a
                result column, a method is unknown or repeated, the period is
                reversed or reaches ``TEST_START``, or ``n_boot`` is below 1.
        """
        if not self.candidates or len(set(self.candidates)) != len(self.candidates):
            msg = f"candidates must be non-empty and unique, got {self.candidates}"
            raise ValueError(msg)
        clashes = sorted(_RESERVED.intersection(self.candidates))
        if clashes:
            msg = f"candidate names clash with result columns: {clashes}"
            raise ValueError(msg)
        unknown = [m for m in self.methods if m not in METHODS]
        if not self.methods or unknown or len(set(self.methods)) != len(self.methods):
            msg = f"methods must be non-empty, unique, in {METHODS}: {self.methods}"
            raise ValueError(msg)
        if self.score_start > self.score_end:
            msg = f"score_start {self.score_start} is after score_end {self.score_end}"
            raise ValueError(msg)
        if self.score_end >= TEST_START:
            msg = f"score_end {self.score_end} reaches the test period ({TEST_START}+)"
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
            "candidates": list(self.candidates),
            "methods": list(self.methods),
            "score_start": self.score_start.isoformat(),
            "score_end": self.score_end.isoformat(),
            "n_boot": self.n_boot,
            "seed": self.seed,
        }

    @property
    def config_id(self) -> str:
        """First 12 hex digits of the SHA-256 of the canonical ``to_dict`` JSON."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]


@dataclass(frozen=True)
class WeightFit:
    """Weights learned at one monthly refit.

    Attributes:
        month: Refit month, ``YYYY-MM``.
        issue_date: First issue date of the month, when the refit happens.
        cutoff: Last day known on ``issue_date``.
        n_rows: Training rows (every row with a known outcome by ``cutoff``).
        last_target: Latest ``target_end`` among the training rows.
        weights: Weight per candidate, in candidate order.
    """

    month: str
    issue_date: dt.date
    cutoff: dt.date
    n_rows: int
    last_target: dt.date
    weights: Mapping[str, float]

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready fields.

        Returns:
            Every field but ``month``, dates as ISO strings.
        """
        return {
            "issue_date": self.issue_date.isoformat(),
            "cutoff": self.cutoff.isoformat(),
            "n_rows": self.n_rows,
            "last_target": self.last_target.isoformat(),
            "weights": dict(self.weights),
        }


def wide_predictions(oof: pl.DataFrame, config: EnsembleConfig) -> pl.DataFrame:
    """One row per forecast with a column of predictions per candidate.

    Args:
        oof: Out-of-fold predictions with ``OUTPUT_SCHEMA`` (the parquet
            written by ``models validate``).
        config: Run settings; only ``spec`` and ``candidates`` are used.

    Returns:
        ``spec, issue_date, cutoff, target_start, target_end, DIVISION,
        y_true, anomaly`` plus one column per candidate, sorted by issue date
        and division.

    Raises:
        ValueError: If the spec is not ``config.spec``, a candidate is missing,
            or candidates differ in keys or ``y_true``, repeat a key, or give a
            null forecast where ``y_true`` is known.
    """
    specs = oof["spec"].unique().sort().to_list()
    if specs != [config.spec.key]:
        msg = f"predictions are for spec {specs}, expected [{config.spec.key!r}]"
        raise ValueError(msg)
    missing = sorted(set(config.candidates) - set(oof["model"].unique().to_list()))
    if missing:
        msg = f"candidates missing from the predictions: {missing}"
        raise ValueError(msg)
    frames = {
        name: oof.filter(pl.col("model") == name).sort(KEYS)
        for name in config.candidates
    }
    base = frames[config.candidates[0]]
    if base.select(KEYS).is_duplicated().any():
        msg = f"{config.candidates[0]} repeats an (issue_date, DIVISION) key"
        raise ValueError(msg)
    for name, frame in frames.items():
        if not frame.select(*KEYS, "y_true").equals(base.select(*KEYS, "y_true")):
            msg = (
                f"{name} and {config.candidates[0]} cover different "
                "(issue_date, DIVISION) keys or disagree on y_true"
            )
            raise ValueError(msg)
        if frame.filter(
            pl.col("y_true").is_not_null(), pl.col("y_pred").is_null()
        ).height:
            msg = f"{name} has null forecasts where y_true is known"
            raise ValueError(msg)
    return base.select(_META).with_columns(
        frame["y_pred"].cast(pl.Float64).alias(name) for name, frame in frames.items()
    )


def _month(column: str) -> pl.Expr:
    return pl.col(column).dt.strftime("%Y-%m")


def scored_rows(wide: pl.DataFrame, config: EnsembleConfig) -> pl.DataFrame:
    """Rows whose whole target window lies in the scored period.

    Args:
        wide: Output of ``wide_predictions``.
        config: Run settings.

    Returns:
        The matching rows of ``wide`` with their refit ``month``.

    Raises:
        ValueError: If no row falls in the period.
    """
    scored = wide.filter(
        pl.col("target_start") >= config.score_start,
        pl.col("target_end") <= config.score_end,
    ).with_columns(month=_month("issue_date"))
    if scored.is_empty():
        msg = f"no predictions between {config.score_start} and {config.score_end}"
        raise ValueError(msg)
    return scored


def rolling_weights(wide: pl.DataFrame, config: EnsembleConfig) -> list[WeightFit]:
    """Refit the MAE-optimal weights at the first issue date of each scored month.

    Each refit uses every row of ``wide`` with a known ``y_true`` whose
    ``target_end`` is on or before the cutoff of that first issue date, so it
    only sees outcomes that were published by then (an expanding window).

    Args:
        wide: Output of ``wide_predictions``.
        config: Run settings.

    Returns:
        One fit per month holding a scored row, in month order.

    Raises:
        ValueError: If nothing is scored or a month has no training rows.
    """
    months = scored_rows(wide, config)["month"].unique().sort().to_list()
    first_issue = (
        wide.with_columns(month=_month("issue_date"))
        .filter(pl.col("month").is_in(months))
        .group_by("month")
        .agg(pl.col("issue_date").min())
        .sort("month")
    )
    known = wide.filter(pl.col("y_true").is_not_null())
    names = list(config.candidates)
    fits: list[WeightFit] = []
    for month, issue in first_issue.iter_rows():
        cutoff = config.spec.cutoff(issue)
        train = known.filter(pl.col("target_end") <= cutoff)
        if train.is_empty():
            msg = f"no outcome known by {cutoff} to fit the weights of {month}"
            raise ValueError(msg)
        weights = fit_weights(
            train.select(names).to_numpy(), train["y_true"].to_numpy(), names
        )
        fits.append(
            WeightFit(
                month=month,
                issue_date=issue,
                cutoff=cutoff,
                n_rows=train.height,
                last_target=train.select(pl.col("target_end").max()).item(),
                weights=weights,
            )
        )
    return fits


def combine_oof(
    oof: pl.DataFrame, config: EnsembleConfig
) -> tuple[pl.DataFrame, list[WeightFit]]:
    """Score every configured combiner on the out-of-fold predictions.

    Args:
        oof: Out-of-fold predictions with ``OUTPUT_SCHEMA``.
        config: Run settings.

    Returns:
        Rows with ``OUTPUT_SCHEMA`` for each method, named ``ens_<method>``,
        on every scored ``(issue_date, DIVISION)`` sorted by model, issue date
        and division; and the monthly weight fits (empty unless ``weighted``
        is configured). ``y_pred`` is null when any candidate's is.

    Raises:
        ValueError: From ``wide_predictions``, ``scored_rows`` or
            ``rolling_weights``.
    """
    wide = wide_predictions(oof, config)
    scored = scored_rows(wide, config)
    names = list(config.candidates)
    preds = scored.select(names).to_numpy()
    fits = rolling_weights(wide, config) if "weighted" in config.methods else []
    frames: list[pl.DataFrame] = []
    for method in config.methods:
        if method == "weighted":
            values = np.full(scored.height, np.nan)
            month = scored["month"].to_numpy()
            for fit in fits:
                rows = month == fit.month
                values[rows] = combine(preds[rows], names, method, fit.weights)
        else:
            values = combine(preds, names, method)
        frames.append(
            scored.with_columns(
                model=pl.lit(f"ens_{method}"),
                y_pred=pl.Series(values, nan_to_null=True),
            )
        )
    rows = pl.concat(frames).select(OUTPUT_SCHEMA.names()).cast(OUTPUT_SCHEMA)
    return rows.sort("model", *KEYS), fits


def weights_summary(fits: Sequence[WeightFit]) -> dict[str, dict[str, float]]:
    """Mean, min and max of each candidate's weight across refits.

    Args:
        fits: Output of ``rolling_weights``.

    Returns:
        ``{candidate: {"mean", "min", "max"}}``; empty when there are no fits.
    """
    if not fits:
        return {}
    names = list(fits[0].weights)
    table = np.array([[fit.weights[name] for name in names] for fit in fits])
    return {
        name: {
            "mean": float(table[:, i].mean()),
            "min": float(table[:, i].min()),
            "max": float(table[:, i].max()),
        }
        for i, name in enumerate(names)
    }


def best_single(results: pl.DataFrame, candidates: Sequence[str]) -> str:
    """Candidate with the lowest MAE on ``results``; ties go to the earlier name.

    Args:
        results: Scored rows holding every candidate.
        candidates: Names to choose from.

    Returns:
        The chosen candidate.
    """
    table = score(results.filter(pl.col("model").is_in(candidates)))
    return table.sort("mae", "model").row(0, named=True)["model"]


def ensemble_report(
    oof: pl.DataFrame, config: EnsembleConfig, input_sha256: str
) -> tuple[dict[str, Any], pl.DataFrame]:
    """Compare the ensembles, the candidates and the references on the same rows.

    Args:
        oof: Out-of-fold predictions with ``OUTPUT_SCHEMA``.
        config: Run settings.
        input_sha256: SHA-256 of the file ``oof`` was read from.

    Returns:
        The report (config, config_id, input hash, ``FINAL_MODEL``, period,
        overall, without
        anomalies, by division, by month, by year, monthly weights of
        ``ens_weighted`` and their summary, and paired bootstraps of each
        method against each of ``REFERENCES`` and the best single candidate,
        floats rounded to 3) and the ensemble rows. Every model is scored on
        the scored ``(issue_date, DIVISION)`` keys of the ensembles.

    Raises:
        ValueError: From ``combine_oof``, or if a reference does not cover
            every scored key.
    """
    ensembles, fits = combine_oof(oof, config)
    keys = ensembles.select(KEYS).unique()
    singles = list(dict.fromkeys((*config.candidates, *REFERENCES)))
    others = oof.filter(pl.col("model").is_in(singles)).join(keys, on=KEYS, how="semi")
    counts = dict(others.group_by("model").len().iter_rows())
    short = sorted(name for name in singles if counts.get(name) != keys.height)
    if short:
        msg = f"{short} do not cover the {keys.height} scored keys"
        raise ValueError(msg)
    results = pl.concat([ensembles, others.select(OUTPUT_SCHEMA.names())]).sort(
        "model", *KEYS
    )
    clean = results.filter(~pl.col("anomaly"))
    with_year = results.with_columns(year=pl.col("target_start").dt.year())

    def table(frame: pl.DataFrame, by: Sequence[str] = ()) -> list[dict[str, Any]]:
        return score(frame, by).to_dicts()

    best = best_single(results, config.candidates)
    methods = [f"ens_{m}" for m in config.methods]

    def boot(reference: str) -> list[dict[str, Any]]:
        return [
            paired_bootstrap(results, m, reference, config.n_boot, config.seed)
            for m in methods
        ]

    report = {
        "config": config.to_dict(),
        "config_id": config.config_id,
        "input_sha256": input_sha256,
        "final_model": FINAL_MODEL,
        "period": {
            "start": config.score_start.isoformat(),
            "end": config.score_end.isoformat(),
        },
        "overall": table(results),
        "overall_without_anomalies": table(clean),
        "by_division": table(results, ["DIVISION"]),
        "by_month": table(results, ["month"]),
        "by_year": table(with_year, ["year"]),
        "weights_by_month": {fit.month: fit.to_dict() for fit in fits},
        "weights_summary": weights_summary(fits),
        "bootstrap": {
            "references": {reference: boot(reference) for reference in REFERENCES},
            "best_single": {
                "model": best,
                "selection": (
                    "lowest overall MAE among the candidates on the scored period, "
                    "picked in hindsight, which favours it"
                ),
                "results": boot(best),
            },
        },
    }
    return rounded(report), ensembles


class Ensemble:
    """Forecaster that combines member forecasts per division.

    Attributes:
        members: Deep copies of the members passed in, in order.
        method: Combiner.
        weights: Weight per member name (scaled to sum to 1) for
            ``weighted``, else None.
    """

    def __init__(
        self,
        name: str,
        members: Sequence[Forecaster],
        method: Method,
        weights: Mapping[str, float] | None = None,
    ) -> None:
        """Check and copy the members.

        Args:
            name: Label used in result tables.
            members: Forecasters to combine; deep-copied, so fitting the
                ensemble never changes the caller's instances.
            method: Combiner.
            weights: Weight per member name; required iff ``method`` is
                ``weighted``.

        Raises:
            ValueError: If there are no members, names repeat, the method is
                unknown, or the weights are missing, unexpected or invalid.
        """
        names = [m.name for m in members]
        if not names or len(set(names)) != len(names):
            msg = f"members must be non-empty with unique names, got {names}"
            raise ValueError(msg)
        if method not in METHODS:
            msg = f"unknown method {method!r}; expected one of {METHODS}"
            raise ValueError(msg)
        if (method == "weighted") != (weights is not None):
            msg = f"weights are required for 'weighted' and only for it, got {method!r}"
            raise ValueError(msg)
        self._name = name
        self.members: tuple[Forecaster, ...] = tuple(copy.deepcopy(list(members)))
        self.method: Method = method
        self.weights: dict[str, float] | None = (
            None
            if weights is None
            else dict(zip(names, weight_vector(names, weights).tolist(), strict=True))
        )

    @property
    def name(self) -> str:
        """Label used in result tables."""
        return self._name

    def fit(self, history: pl.DataFrame, spec: ForecastSpec) -> Self:
        """Fit every member on the same history.

        Args:
            history: Passed to each member's ``fit``.
            spec: Passed to each member's ``fit``.

        Returns:
            This ensemble.
        """
        for member in self.members:
            member.fit(history, spec)
        return self

    def predict(
        self, history: pl.DataFrame, issue_date: dt.date, spec: ForecastSpec
    ) -> pl.DataFrame:
        """Combine the members' forecasts for each division.

        Args:
            history: Passed to each member's ``predict``.
            issue_date: Passed to each member's ``predict``.
            spec: Passed to each member's ``predict``.

        Returns:
            ``DIVISION, y_pred`` sorted by division; null when any member's
            forecast is null.

        Raises:
            ValueError: If members return different divisions.
        """
        preds = [
            member.predict(history, issue_date, spec).sort("DIVISION")
            for member in self.members
        ]
        divisions = preds[0]["DIVISION"]
        for member, pred in zip(self.members, preds, strict=True):
            if not pred["DIVISION"].equals(divisions):
                msg = (
                    f"{member.name} on {issue_date} returned divisions "
                    f"{pred['DIVISION'].to_list()}, expected {divisions.to_list()}"
                )
                raise ValueError(msg)
        matrix = np.column_stack(
            [pred["y_pred"].cast(pl.Float64).to_numpy() for pred in preds]
        )
        names = [m.name for m in self.members]
        values = combine(matrix, names, self.method, self.weights)
        return pl.DataFrame(
            {
                "DIVISION": divisions,
                "y_pred": pl.Series(values, dtype=pl.Float64, nan_to_null=True),
            }
        )


def from_candidates(
    config: EnsembleConfig, weights: Mapping[str, float] | None = None
) -> list[Ensemble]:
    """One ``Ensemble`` per configured method, built from ``default_models``.

    Args:
        config: Run settings; ``candidates``, ``methods`` and ``seed`` are used.
        weights: Weight per candidate for ``weighted``; required iff it is a
            configured method.

    Returns:
        Ensembles named ``ens_<method>``, in method order, each with its own
        copies of the candidate models.

    Raises:
        ValueError: If a candidate is not a default model name, or ``weights``
            does not match the configured methods.
    """
    available = {m.name: m for m in default_models(config.seed)}
    unknown = sorted(set(config.candidates) - set(available))
    if unknown:
        msg = f"unknown candidates {unknown}; default models are {sorted(available)}"
        raise ValueError(msg)
    if ("weighted" in config.methods) != (weights is not None):
        msg = "weights are required iff 'weighted' is a configured method"
        raise ValueError(msg)
    members = [available[name] for name in config.candidates]
    return [
        Ensemble(
            f"ens_{method}",
            members,
            method,
            weights if method == "weighted" else None,
        )
        for method in config.methods
    ]


@app.callback()
def main() -> None:
    """Combine the candidate models and validate the ensembles."""


@app.command()
def validate(
    predictions: Annotated[
        Path, typer.Option(help="Out-of-fold predictions parquet.")
    ] = DEFAULT_PREDICTIONS,
    out: Annotated[Path, typer.Option(help="JSON report path.")] = DEFAULT_OUT,
    ensemble_predictions: Annotated[
        Path, typer.Option(help="Parquet path for the ensemble predictions.")
    ] = DEFAULT_ENSEMBLE_PREDICTIONS,
    start: Annotated[
        dt.datetime, typer.Option(formats=["%Y-%m-%d"], help="First scored day.")
    ] = _CLI_START,
    end: Annotated[
        dt.datetime, typer.Option(formats=["%Y-%m-%d"], help="Last scored day.")
    ] = _CLI_END,
) -> None:
    """Validate the mean, median and weighted ensembles (2022-2024 by default).

    Args:
        predictions: Out-of-fold predictions parquet.
        out: JSON report path.
        ensemble_predictions: Parquet path for the ensemble predictions.
        start: First scored day.
        end: Last scored day.

    Raises:
        typer.Exit: If the predictions parquet is missing.
    """
    if not predictions.is_file():
        typer.echo(
            f"{predictions} not found; run "
            "`uv run python -m ycit440_sim_forecast.models validate` first",
            err=True,
        )
        raise typer.Exit(code=1)
    config = EnsembleConfig(score_start=start.date(), score_end=end.date())
    report, rows = ensemble_report(
        pl.read_parquet(predictions), config, sha256_file(predictions)
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    ensemble_predictions.parent.mkdir(parents=True, exist_ok=True)
    rows.write_parquet(ensemble_predictions)
    clean = {row["model"]: row["mae"] for row in report["overall_without_anomalies"]}
    typer.echo(f"config {report['config_id']}, scored {start.date()} to {end.date()}")
    for row in report["overall"]:
        typer.echo(
            f"{row['model']:<26} MAE {row['mae']:6.3f}  bias {row['bias']:+6.3f}  "
            f"MAE without anomalies {clean[row['model']]:6.3f}"
        )
    boot = report["bootstrap"]
    best = boot["best_single"]
    for reference, comparisons in [
        *boot["references"].items(),
        (best["model"], best["results"]),
    ]:
        for row in comparisons:
            typer.echo(
                f"{row['model']:<14} vs {reference:<18} diff {row['diff']:+6.3f}  "
                f"95% CI [{row['ci_low']:+6.3f}, {row['ci_high']:+6.3f}]"
            )
    for name, stats in report["weights_summary"].items():
        typer.echo(
            f"weight {name:<20} mean {stats['mean']:.3f}  "
            f"min {stats['min']:.3f}  max {stats['max']:.3f}"
        )
    typer.echo(f"wrote {out} and {ensemble_predictions}")


if __name__ == "__main__":
    app()
