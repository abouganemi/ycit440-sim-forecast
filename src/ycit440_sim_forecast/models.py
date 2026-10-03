"""Individual forecasting models: gradient boosting and a negative binomial GLM.

Every model predicts a multiplicative adjustment over the recent level. The
offset is ``log(mean_28d * window_days)``, the plain 28-day mean forecast, so
a model that learns nothing reproduces ``PlainMean`` and trees never have to
extrapolate a level they did not see in training (Division 6 keeps rising and
there is a late-2025 level break). Models train on the rows of
``features.feature_table`` with a known target and a positive ``mean_28d``.

Model specs are plain configuration; ``fit`` builds a fresh estimator each
time. ``default_models`` lists the configurations that get validated. Usage::

    uv run python -m ycit440_sim_forecast.models validate
"""

import datetime as dt
import json
import math
import time
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Any, Self, cast

import numpy as np
import pandas as pd
import polars as pl
import typer
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor
from sklearn.base import clone
from statsmodels.discrete.discrete_model import NegativeBinomial
from xgboost import XGBRegressor

from ycit440_sim_forecast import features as ft
from ycit440_sim_forecast.baseline import PlainMean, SameWeekdayMean
from ycit440_sim_forecast.evaluate import (
    DEFAULT_ANOMALIES,
    DEFAULT_HISTORY,
    VALIDATION_END,
    VALIDATION_START,
    flag_anomalies,
    paired_bootstrap,
    rolling_origin,
    score,
)
from ycit440_sim_forecast.seed import DEFAULT_SEED
from ycit440_sim_forecast.spec import V2_DAILY, Forecaster, ForecastSpec

DIVISIONS: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
"""Fixed division levels, so categorical codes match between fit and predict."""

# Days before the cutoff that the longest feature reads. mean_91d spans the
# cutoff and 90 days before it. A same-weekday mean over w weeks for a target
# day ``a`` days after the cutoff starts at week ceil(a / 7), so its oldest
# source day is 7 * ceil(a / 7) - a + 7 * (w - 1) <= 6 + 7 * (w - 1) days back:
# the gap and the window position cancel out, leaving 7 * w - 1 = 90 for 13 weeks.
LOOKBACK = dt.timedelta(
    days=max(
        ft.LAGS - 1,
        *(d - 1 for d in (*ft.MEAN_DAYS, ft.SD_DAYS, *ft.FR_SHARE_DAYS)),
        *(7 * w - 1 for w in ft.SAME_WEEKDAY_WEEKS),
    )
)
"""History before the cutoff needed for every feature of one issue date (90 days)."""

RATIO_FLOOR = 0.05
"""Lower clip on ratios before taking logs in the GLM."""

LOG_RATIOS: tuple[tuple[str, str | None], ...] = (
    ("ratio_7_28", None),
    ("ratio_28_91", None),
    ("same_wd_4w", "mean_28d"),
    ("same_wd_13w", "mean_28d"),
    ("lag_0", "mean_28d"),
)
"""GLM log-ratio terms as ``(numerator, denominator)``; ``None`` means a ready ratio."""

GLM_COLUMNS: tuple[str, ...] = (
    *(f"log_{num}" if den is None else f"log_{num}_rel" for num, den in LOG_RATIOS),
    "fr_share_28d",
    "holidays",
    "holiday_adjacent",
    *(f"weekday_{k}" for k in range(2, 8)),
    *(f"month_{k}" for k in range(2, 13)),
    *(f"division_{k}" for k in DIVISIONS[1:]),
    "const",
)
"""GLM design columns; weekday 1, month 1 and division 1 are the reference levels."""

DEFAULT_OUT = Path("reports/models_validation.json")
DEFAULT_PREDICTIONS = Path("data/processed/model_predictions_v2.parquet")
REFERENCES: tuple[str, ...] = ("same_wd_13w", "plain_28d")
"""Baselines every model is compared against in the report."""

LIBRARIES: tuple[str, ...] = (
    "catboost",
    "lightgbm",
    "numpy",
    "pandas",
    "polars",
    "scikit-learn",
    "statsmodels",
    "xgboost-cpu",
)
_CLI_START = dt.datetime.combine(VALIDATION_START, dt.time())
_CLI_END = dt.datetime.combine(VALIDATION_END, dt.time())

type Booster = LGBMRegressor | XGBRegressor | CatBoostRegressor

app = typer.Typer(help=__doc__, no_args_is_help=True)


def training_rows(history: pl.DataFrame, spec: ForecastSpec) -> pl.DataFrame:
    """Feature rows a model can learn from.

    Args:
        history: ``date, DIVISION, n, n_fr`` on the full calendar.
        spec: Forecast setting.

    Returns:
        Rows of ``feature_table`` with a non-null ``y`` and a positive
        ``mean_28d``. Every target window ends on or before the last day of
        ``history``.
    """
    return ft.feature_table(history, spec).filter(
        pl.col("y").is_not_null(), pl.col("mean_28d") > 0
    )


def issue_rows(
    history: pl.DataFrame, issue_date: dt.date, spec: ForecastSpec
) -> pl.DataFrame:
    """Feature rows of one issue date, built from the tail of ``history``.

    Only days from ``cutoff - LOOKBACK`` to the cutoff are read, which gives
    the same features as the full history at a fraction of the cost.

    Args:
        history: ``date, DIVISION, n, n_fr`` on the full calendar.
        issue_date: Day the forecast is issued.
        spec: Forecast setting.

    Returns:
        ``feature_table`` rows for ``issue_date``; empty when ``history`` does
        not reach the cutoff.
    """
    cutoff = spec.cutoff(issue_date)
    tail = history.filter(pl.col("date").is_between(cutoff - LOOKBACK, cutoff))
    return ft.feature_table(tail, spec).filter(pl.col("issue_date") == issue_date)


def log_offset(rows: pl.DataFrame, spec: ForecastSpec) -> np.ndarray:
    """Log of the plain 28-day mean forecast for each row.

    Args:
        rows: Feature rows with ``mean_28d``.
        spec: Forecast setting; the mean is scaled to ``window_days``.

    Returns:
        ``log(mean_28d * window_days)``; NaN where ``mean_28d`` is null.
    """
    level = rows["mean_28d"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    return np.log(level * spec.window_days)


def _with_divisions(
    history: pl.DataFrame, rows: pl.DataFrame, y_pred: np.ndarray
) -> pl.DataFrame:
    """Attach predictions to every division of ``history``.

    Args:
        history: Frame whose divisions are returned.
        rows: Feature rows the predictions belong to, with ``DIVISION``.
        y_pred: One prediction per row of ``rows``.

    Returns:
        ``DIVISION, y_pred`` sorted by division; null where no prediction.
    """
    predicted = pl.DataFrame(
        {"DIVISION": rows["DIVISION"], "y_pred": pl.Series(y_pred, dtype=pl.Float64)}
    ).with_columns(pl.col("y_pred").fill_nan(None))
    return (
        history.select(pl.col("DIVISION").unique().sort())
        .join(predicted, on="DIVISION", how="left")
        .select("DIVISION", "y_pred")
    )


def _check_divisions(rows: pl.DataFrame) -> None:
    """Reject divisions outside ``DIVISIONS``.

    Args:
        rows: Feature rows with ``DIVISION``.

    Raises:
        ValueError: If a division has no fixed level.
    """
    unknown = set(rows["DIVISION"].unique().to_list()) - set(DIVISIONS)
    if unknown:
        msg = f"divisions {sorted(unknown)} are not in {DIVISIONS}"
        raise ValueError(msg)


class BoostedForecaster:
    """A LightGBM, XGBoost or CatBoost regressor with a log-level offset.

    The estimator must use a log link (Poisson or Tweedie objective). With
    ``offset`` on, the offset enters as LightGBM ``init_score``, XGBoost
    ``base_margin`` or CatBoost ``Pool(baseline=...)``, and the model learns
    the log ratio of the target to the plain 28-day mean. With it off, the
    model fits ``y`` directly. ``DIVISION`` is categorical; other features are
    floats with nulls as NaN.

    Attributes:
        name: Label used in result tables.
        estimator: Unfitted estimator, cloned on every ``fit``.
        offset: Whether predictions are relative to the plain 28-day mean.
    """

    def __init__(self, name: str, estimator: Booster, *, offset: bool = True) -> None:
        """Store the configuration.

        Args:
            name: Label used in result tables.
            estimator: Unfitted ``LGBMRegressor``, ``XGBRegressor`` or
                ``CatBoostRegressor`` with a log-link objective.
            offset: Whether predictions are relative to the plain 28-day mean.

        Raises:
            TypeError: If ``estimator`` is not one of the supported types.
        """
        if not isinstance(estimator, LGBMRegressor | XGBRegressor | CatBoostRegressor):
            msg = f"unsupported estimator {type(estimator).__name__}"
            raise TypeError(msg)
        self.name = name
        self.estimator = estimator
        self.offset = offset
        self._model: Booster | None = None

    @staticmethod
    def _frame(rows: pl.DataFrame, *, categorical: bool) -> pd.DataFrame:
        """Model inputs as pandas.

        Args:
            rows: Feature rows.
            categorical: Encode ``DIVISION`` as a pandas category with fixed
                levels (LightGBM, XGBoost); otherwise keep it as an integer
                (CatBoost).

        Returns:
            ``FEATURE_COLUMNS`` with float features and NaN for nulls.
        """
        frame = rows.select(
            "DIVISION", pl.col(ft.FEATURE_COLUMNS[1:]).cast(pl.Float64)
        ).to_pandas()
        if categorical:
            frame["DIVISION"] = pd.Categorical(frame["DIVISION"], categories=DIVISIONS)
        return frame

    def fit(self, history: pl.DataFrame, spec: ForecastSpec) -> Self:
        """Train a fresh copy of the estimator, replacing any earlier fit.

        Args:
            history: ``date, DIVISION, n, n_fr`` on the full calendar.
            spec: Forecast setting.

        Returns:
            This forecaster.

        Raises:
            ValueError: If there are no training rows, or a division is
                outside ``DIVISIONS``.
        """
        rows = training_rows(history, spec)
        if rows.is_empty():
            msg = f"{self.name}: no training rows in history"
            raise ValueError(msg)
        _check_divisions(rows)
        y = rows["y"].to_numpy()
        margin = log_offset(rows, spec) if self.offset else None
        model = cast(Booster, clone(self.estimator))
        match model:
            case CatBoostRegressor():
                model.fit(
                    Pool(
                        self._frame(rows, categorical=False),
                        y,
                        cat_features=["DIVISION"],
                        baseline=margin,
                    )
                )
            case LGBMRegressor():
                model.fit(self._frame(rows, categorical=True), y, init_score=margin)
            case _:
                model.fit(self._frame(rows, categorical=True), y, base_margin=margin)
        self._model = model
        return self

    def predict(
        self, history: pl.DataFrame, issue_date: dt.date, spec: ForecastSpec
    ) -> pl.DataFrame:
        """Forecast the window of ``issue_date`` for every division.

        Args:
            history: ``date, DIVISION, n, n_fr`` on the full calendar; only
                ``cutoff - LOOKBACK`` to the cutoff is read.
            issue_date: Day the forecast is issued.
            spec: Forecast setting.

        Returns:
            ``DIVISION, y_pred`` for every division in ``history``, sorted;
            null when ``history`` does not reach the cutoff or, with the
            offset on, ``mean_28d`` is missing.

        Raises:
            RuntimeError: If called before ``fit``.
        """
        if self._model is None:
            msg = f"{self.name}: call fit before predict"
            raise RuntimeError(msg)
        rows = issue_rows(history, issue_date, spec)
        if self.offset:
            rows = rows.filter(pl.col("mean_28d") > 0)
        if rows.is_empty():
            return _with_divisions(history, rows, np.empty(0))
        _check_divisions(rows)
        margin = log_offset(rows, spec) if self.offset else None
        match self._model:
            case CatBoostRegressor() as model:
                pool = Pool(
                    self._frame(rows, categorical=False),
                    cat_features=["DIVISION"],
                    baseline=margin,
                )
                # The Pool baseline is added to the raw score; "Exponent" maps
                # it to the count scale.
                y_pred = model.predict(pool, prediction_type="Exponent")
            case LGBMRegressor() as model:
                raw = model.predict(self._frame(rows, categorical=True), raw_score=True)
                y_pred = np.exp(np.asarray(raw) + (0.0 if margin is None else margin))
            case model:
                y_pred = model.predict(
                    self._frame(rows, categorical=True), base_margin=margin
                )
        return _with_divisions(history, rows, np.asarray(y_pred, dtype=np.float64))


class NegBinGLM:
    """Negative binomial (NB2) regression with the log-level offset.

    The design has the log ratios in ``LOG_RATIOS`` (clipped below at
    ``RATIO_FLOOR``), ``fr_share_28d``, the holiday counts, one-hot weekday,
    month and division (first level dropped) and a constant. A null log ratio
    becomes 0, the value of a ratio of 1. A null ``fr_share_28d`` becomes the
    median of the training rows. Columns that are zero on every training row
    (for example a division absent from ``history``) are dropped at fit, so
    such a level gets the reference effect at predict. ``alpha`` is estimated.

    Attributes:
        name: Label used in result tables.
        method: ``statsmodels`` optimiser.
        maxiter: Iteration limit of the optimiser.
    """

    def __init__(
        self, name: str = "nb_glm", method: str = "bfgs", maxiter: int = 500
    ) -> None:
        """Store the configuration.

        Args:
            name: Label used in result tables.
            method: ``statsmodels`` optimiser. ``bfgs`` converges on the real
                data; ``newton`` gives a NaN ``alpha`` there.
            maxiter: Iteration limit of the optimiser.
        """
        self.name = name
        self.method = method
        self.maxiter = maxiter
        self._result: Any = None
        self._keep: np.ndarray | None = None
        self._fr_fill = 0.0

    def design(self, rows: pl.DataFrame) -> np.ndarray:
        """Full design matrix, ``GLM_COLUMNS`` in order.

        Args:
            rows: Feature rows.

        Returns:
            A float matrix with one row per feature row.
        """

        def log_ratio(num: str, den: str | None) -> pl.Expr:
            ratio = pl.col(num) if den is None else pl.col(num) / pl.col(den)
            return (
                ratio.cast(pl.Float64)
                .clip(lower_bound=RATIO_FLOOR)
                .log()
                .fill_nan(None)
                .fill_null(0.0)
            )

        def one_hot(column: str, levels: Sequence[int]) -> list[pl.Expr]:
            return [(pl.col(column) == k).cast(pl.Float64) for k in levels]

        exprs = [
            *(log_ratio(num, den) for num, den in LOG_RATIOS),
            pl.col("fr_share_28d").cast(pl.Float64).fill_null(self._fr_fill),
            pl.col("holidays").cast(pl.Float64),
            pl.col("holiday_adjacent").cast(pl.Float64),
            *one_hot("target_weekday", range(2, 8)),
            *one_hot("target_month", range(2, 13)),
            *one_hot("DIVISION", DIVISIONS[1:]),
            pl.lit(1.0),
        ]
        named = [e.alias(c) for e, c in zip(exprs, GLM_COLUMNS, strict=True)]
        return rows.select(named).to_numpy().astype(np.float64)

    @property
    def columns(self) -> list[str]:
        """Design columns kept by the last ``fit``."""
        if self._keep is None:
            return []
        return [c for c, k in zip(GLM_COLUMNS, self._keep, strict=True) if k]

    def fit(self, history: pl.DataFrame, spec: ForecastSpec) -> Self:
        """Estimate the coefficients and ``alpha``, replacing any earlier fit.

        Args:
            history: ``date, DIVISION, n, n_fr`` on the full calendar.
            spec: Forecast setting.

        Returns:
            This forecaster.

        Raises:
            ValueError: If there are no training rows, or a division is
                outside ``DIVISIONS``.
            RuntimeError: If the optimiser returns a non-finite parameter.
        """
        rows = training_rows(history, spec)
        if rows.is_empty():
            msg = f"{self.name}: no training rows in history"
            raise ValueError(msg)
        _check_divisions(rows)
        median = rows["fr_share_28d"].median()
        self._fr_fill = 0.0 if median is None else float(median)  # pyright: ignore[reportArgumentType]
        full = self.design(rows)
        self._keep = (full != 0).any(axis=0)
        model = NegativeBinomial(
            rows["y"].to_numpy(),
            full[:, self._keep],
            loglike_method="nb2",
            offset=log_offset(rows, spec),
        )
        result = model.fit(method=self.method, maxiter=self.maxiter, disp=0)
        # statsmodels can report convergence with a NaN alpha (seen with newton).
        if not np.isfinite(result.params).all():
            self._result = None
            msg = f"{self.name}: {self.method} gave non-finite parameters"
            raise RuntimeError(msg)
        self._result = result
        return self

    def predict(
        self, history: pl.DataFrame, issue_date: dt.date, spec: ForecastSpec
    ) -> pl.DataFrame:
        """Forecast the window of ``issue_date`` for every division.

        Args:
            history: ``date, DIVISION, n, n_fr`` on the full calendar; only
                ``cutoff - LOOKBACK`` to the cutoff is read.
            issue_date: Day the forecast is issued.
            spec: Forecast setting.

        Returns:
            ``DIVISION, y_pred`` (the predicted mean) for every division in
            ``history``, sorted; null when ``history`` does not reach the
            cutoff or ``mean_28d`` is missing.

        Raises:
            RuntimeError: If called before ``fit``.
        """
        if self._result is None or self._keep is None:
            msg = f"{self.name}: call fit before predict"
            raise RuntimeError(msg)
        rows = issue_rows(history, issue_date, spec).filter(pl.col("mean_28d") > 0)
        if rows.is_empty():
            return _with_divisions(history, rows, np.empty(0))
        _check_divisions(rows)
        exog = self.design(rows)[:, self._keep]
        y_pred = self._result.predict(exog, offset=log_offset(rows, spec))
        return _with_divisions(history, rows, np.asarray(y_pred, dtype=np.float64))


def default_models(seed: int = DEFAULT_SEED) -> list[Forecaster]:
    """The model configurations that get validated.

    Fixed, untuned settings: 400 trees at learning rate 0.05, shallow trees,
    4 threads with the deterministic options of each library, and silent
    logging. XGBoost's ``min_child_weight`` is a hessian sum, not a row count,
    so it keeps its default; CatBoost's symmetric trees take no leaf minimum.

    Args:
        seed: Random seed passed to every library.

    Returns:
        LightGBM Poisson, Tweedie and Poisson without offset; XGBoost Poisson
        and Tweedie; CatBoost Poisson and Tweedie; the NB GLM.
    """
    lgbm = {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_child_samples": 50,
        "random_state": seed,
        "n_jobs": 4,
        "deterministic": True,
        "force_row_wise": True,
        "verbose": -1,
    }
    xgb = {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "max_depth": 4,
        "tree_method": "hist",
        "enable_categorical": True,
        "random_state": seed,
        "n_jobs": 4,
    }
    cat = {
        "iterations": 400,
        "learning_rate": 0.05,
        "depth": 4,
        "random_seed": seed,
        "thread_count": 4,
        "logging_level": "Silent",
        "allow_writing_files": False,
    }
    return [
        BoostedForecaster("lgbm_poisson", LGBMRegressor(objective="poisson", **lgbm)),
        BoostedForecaster(
            "lgbm_tweedie",
            LGBMRegressor(objective="tweedie", tweedie_variance_power=1.5, **lgbm),
        ),
        BoostedForecaster(
            "lgbm_poisson_raw", LGBMRegressor(objective="poisson", **lgbm), offset=False
        ),
        BoostedForecaster(
            "xgb_poisson", XGBRegressor(objective="count:poisson", **xgb)
        ),
        BoostedForecaster(
            "xgb_tweedie",
            XGBRegressor(objective="reg:tweedie", tweedie_variance_power=1.5, **xgb),
        ),
        BoostedForecaster(
            "cat_poisson", CatBoostRegressor(loss_function="Poisson", **cat)
        ),
        BoostedForecaster(
            "cat_tweedie",
            CatBoostRegressor(loss_function="Tweedie:variance_power=1.5", **cat),
        ),
        NegBinGLM(),
    ]


class Timed:
    """Wraps a forecaster and adds up the wall time of its fits and predicts.

    Attributes:
        model: The wrapped forecaster.
        fit_seconds: Total time spent in ``fit``.
        predict_seconds: Total time spent in ``predict``.
    """

    def __init__(self, model: Forecaster) -> None:
        """Wrap ``model`` with zeroed timers.

        Args:
            model: Forecaster to time.
        """
        self.model = model
        self.fit_seconds = 0.0
        self.predict_seconds = 0.0

    @property
    def name(self) -> str:
        """Name of the wrapped forecaster."""
        return self.model.name

    def fit(self, history: pl.DataFrame, spec: ForecastSpec) -> Self:
        """Fit the wrapped forecaster and add the elapsed time.

        Args:
            history: Passed to the wrapped ``fit``.
            spec: Passed to the wrapped ``fit``.

        Returns:
            This wrapper.
        """
        start = time.perf_counter()
        self.model.fit(history, spec)
        self.fit_seconds += time.perf_counter() - start
        return self

    def predict(
        self, history: pl.DataFrame, issue_date: dt.date, spec: ForecastSpec
    ) -> pl.DataFrame:
        """Predict with the wrapped forecaster and add the elapsed time.

        Args:
            history: Passed to the wrapped ``predict``.
            issue_date: Passed to the wrapped ``predict``.
            spec: Passed to the wrapped ``predict``.

        Returns:
            The wrapped forecaster's predictions.
        """
        start = time.perf_counter()
        pred = self.model.predict(history, issue_date, spec)
        self.predict_seconds += time.perf_counter() - start
        return pred


def _rounded(value: Any) -> Any:
    """Round every float in a nested structure to 3 decimals.

    Args:
        value: Dict, list, float or other JSON value.

    Returns:
        The same structure with floats rounded; non-finite floats become None.
    """
    match value:
        case dict():
            return {k: _rounded(v) for k, v in value.items()}
        case list() | tuple():
            return [_rounded(v) for v in value]
        case float():
            return round(value, 3) if math.isfinite(value) else None
        case _:
            return value


def models_report(
    history: pl.DataFrame,
    anomalies: pl.DataFrame,
    forecasters: Sequence[Forecaster],
    start: dt.date = VALIDATION_START,
    end: dt.date = VALIDATION_END,
    spec: ForecastSpec = V2_DAILY,
) -> tuple[dict[str, Any], pl.DataFrame]:
    """Validate the models and both chosen baselines by rolling origin.

    Args:
        history: ``date, DIVISION, n, n_fr`` on the full calendar.
        anomalies: ``date, DIVISION, anomaly`` (the EDA anomaly calendar).
        forecasters: Models to validate, next to ``SameWeekdayMean(13)`` and
            ``PlainMean()``.
        start: First target day scored.
        end: Last target day scored.
        spec: Forecast setting.

    Returns:
        The report (period, setting, overall, without anomalies, by division,
        by month, paired bootstraps against each of ``REFERENCES``, timings and
        library versions, floats rounded to 3) and the raw results with an
        ``anomaly`` column.
    """
    timed = [Timed(m) for m in (*forecasters, SameWeekdayMean(13), PlainMean())]
    results = flag_anomalies(
        rolling_origin(history, timed, spec, start, end), anomalies
    )
    clean = results.filter(~pl.col("anomaly"))

    def table(frame: pl.DataFrame, by: Sequence[str] = ()) -> list[dict[str, Any]]:
        return score(frame, by).to_dicts()

    bootstrap = {
        reference: [
            paired_bootstrap(results, m.name, reference)
            for m in timed
            if m.name != reference
        ]
        for reference in REFERENCES
    }
    report = {
        "period": {"start": str(start), "end": str(end)},
        "spec": spec.key,
        "overall": table(results),
        "overall_without_anomalies": table(clean),
        "by_division": table(results, ["DIVISION"]),
        "by_month": table(results, ["month"]),
        "bootstrap": bootstrap,
        "seconds": {
            m.name: {"fit": m.fit_seconds, "predict": m.predict_seconds} for m in timed
        },
        "versions": {lib: version(lib) for lib in LIBRARIES},
    }
    return _rounded(report), results


@app.callback()
def main() -> None:
    """Fit and validate the individual models."""


@app.command()
def validate(
    history_path: Annotated[
        Path, typer.Option("--history", help="Division-day parquet.")
    ] = DEFAULT_HISTORY,
    anomalies_path: Annotated[
        Path, typer.Option("--anomalies", help="Anomaly calendar parquet.")
    ] = DEFAULT_ANOMALIES,
    out: Annotated[Path, typer.Option(help="JSON report path.")] = DEFAULT_OUT,
    predictions: Annotated[
        Path, typer.Option(help="Parquet path for the raw predictions.")
    ] = DEFAULT_PREDICTIONS,
    start: Annotated[
        dt.datetime, typer.Option(formats=["%Y-%m-%d"], help="First target day.")
    ] = _CLI_START,
    end: Annotated[
        dt.datetime, typer.Option(formats=["%Y-%m-%d"], help="Last target day.")
    ] = _CLI_END,
) -> None:
    """Validate every default model and both baselines on v2 (2021-2024 by default).

    Args:
        history_path: Division-day parquet.
        anomalies_path: Anomaly calendar parquet.
        out: JSON report path.
        predictions: Parquet path for the raw predictions.
        start: First target day.
        end: Last target day.

    Raises:
        typer.Exit: If an input file is missing.
    """
    for path in (history_path, anomalies_path):
        if not path.is_file():
            typer.echo(f"{path} not found; run notebooks/01_eda.ipynb first", err=True)
            raise typer.Exit(code=1)
    report, results = models_report(
        pl.read_parquet(history_path),
        pl.read_parquet(anomalies_path),
        default_models(),
        start.date(),
        end.date(),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    predictions.parent.mkdir(parents=True, exist_ok=True)
    results.write_parquet(predictions)
    clean = {row["model"]: row["mae"] for row in report["overall_without_anomalies"]}
    for row in report["overall"]:
        typer.echo(
            f"{row['model']:<18} MAE {row['mae']:6.3f}  bias {row['bias']:+6.3f}  "
            f"MAE without anomalies {clean[row['model']]:6.3f}"
        )
    typer.echo(f"wrote {out} and {predictions}")


if __name__ == "__main__":
    app()
