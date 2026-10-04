"""Individual forecasting models: boosting, a GLM, state-space models, top-down.

Most feature-based models predict a multiplicative adjustment over the recent
level. The offset is ``log(mean_28d * window_days)``, the plain 28-day mean
forecast, so a model that learns nothing reproduces ``PlainMean`` and trees
never have to extrapolate a level they did not see in training (Division 6
keeps rising and there is a late-2025 level break). Identity-link boosters
reach the same level by learning the ratio to it instead. Feature-based models
train on the rows of ``features.feature_table`` with a known target and a
positive ``mean_28d``.

``ETSWeekly`` and ``ARCalendar`` fit one state-space model per division on
``log(n)``; ``predict`` re-runs the filter with the fitted parameters on every
day up to the cutoff, so the level follows the latest counts between monthly
refits. ``TopDown`` forecasts the citywide total and splits it by recent
division shares.

Model specs are plain configuration; ``fit`` builds a fresh estimator each
time. ``default_models`` lists the configurations that get validated. Usage::

    uv run python -m ycit440_sim_forecast.models validate
"""

import copy
import datetime as dt
import json
import math
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Any, Literal, Self, cast

import numpy as np
import pandas as pd
import polars as pl
import typer
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor
from sklearn.base import clone
from statsmodels.discrete.discrete_model import NegativeBinomial
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.statespace.sarimax import SARIMAX
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

CITYWIDE = 0
"""``DIVISION`` value of the citywide series built by ``citywide``."""

BOOSTED_LEVELS: tuple[int, ...] = (*DIVISIONS, CITYWIDE)
"""Categorical levels of ``DIVISION`` in boosters; the citywide level comes last
so the codes of divisions 1-6 are those of ``DIVISIONS``."""

LOG_LINK_OBJECTIVES: tuple[str, ...] = (
    "poisson",
    "tweedie",
    "gamma",
    "count:poisson",
    "reg:tweedie",
    "reg:gamma",
    "Poisson",
    "Tweedie",
)
"""Objective prefixes with a log link (LightGBM, XGBoost, CatBoost spellings)."""

type LevelMode = Literal["offset", "ratio", "none"]

MIN_FIT_DAYS = 91
"""Non-null days a division needs before a per-division state-space model is fit."""

WEEK = 7
FOURIER_ORDER = 3
YEAR_DAYS = 365.25
AR_COLUMNS: tuple[str, ...] = (
    *(f"weekday_{k}" for k in range(2, 8)),
    *(f"{f}_{k}" for k in range(1, FOURIER_ORDER + 1) for f in ("doy_sin", "doy_cos")),
    "holiday",
    "holiday_adjacent",
    "const",
)
"""``ARCalendar`` regressors of each day; weekday 1 (Monday) is the reference."""

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

ENSEMBLE_CANDIDATES: tuple[str, ...] = (
    "lgbm_poisson_raw",
    "lgbm_poisson",
    "cat_tweedie",
    "ets_weekly",
    "topdown_ets",
)
"""Models to combine: trees with and without the level offset, plus level-tracking
ETS models whose bias is near the baselines', so their errors should differ from
the trees'. XGBoost duplicates LightGBM and ``nb_glm`` is worst, so both are out."""

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


def plain_level(rows: pl.DataFrame, spec: ForecastSpec) -> np.ndarray:
    """The plain 28-day mean forecast for each row.

    Args:
        rows: Feature rows with ``mean_28d``.
        spec: Forecast setting; the mean is scaled to ``window_days``.

    Returns:
        ``mean_28d * window_days``; NaN where ``mean_28d`` is null.
    """
    level = rows["mean_28d"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    return level * spec.window_days


def log_offset(rows: pl.DataFrame, spec: ForecastSpec) -> np.ndarray:
    """Log of the plain 28-day mean forecast for each row.

    Args:
        rows: Feature rows with ``mean_28d``.
        spec: Forecast setting; the mean is scaled to ``window_days``.

    Returns:
        ``log(mean_28d * window_days)``; NaN where ``mean_28d`` is null.
    """
    return np.log(plain_level(rows, spec))


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


def _check_divisions(rows: pl.DataFrame, levels: Sequence[int] = DIVISIONS) -> None:
    """Reject divisions outside ``levels``.

    Args:
        rows: Feature rows with ``DIVISION``.
        levels: Accepted divisions.

    Raises:
        ValueError: If a division has no fixed level.
    """
    unknown = set(rows["DIVISION"].unique().to_list()) - set(levels)
    if unknown:
        msg = f"divisions {sorted(unknown)} are not in {tuple(levels)}"
        raise ValueError(msg)


def _log_link(estimator: Booster) -> bool:
    """Whether the estimator's objective has a log link.

    Args:
        estimator: LightGBM, XGBoost or CatBoost regressor.

    Returns:
        True when the objective starts with one of ``LOG_LINK_OBJECTIVES``.
    """
    match estimator:
        case CatBoostRegressor():
            objective = estimator.get_params().get("loss_function")
        case _:
            objective = estimator.get_params().get("objective")
    return str(objective).startswith(LOG_LINK_OBJECTIVES)


class BoostedForecaster:
    """A LightGBM, XGBoost or CatBoost regressor relative to the recent level.

    ``level`` sets how the level ``L = mean_28d * window_days`` (the plain
    28-day mean forecast) enters:

    - ``"offset"``: ``log(L)`` enters as LightGBM ``init_score``, XGBoost
      ``base_margin`` or CatBoost ``Pool(baseline=...)``, and the model learns
      the log ratio of the target to ``L``. Needs a log-link objective.
    - ``"ratio"``: the model fits ``y / L`` with sample weights ``L`` and the
      forecast is ``L`` times its prediction. Since ``L * |y / L - p|`` equals
      ``|y - L * p|``, an L1 objective then minimises MAE on counts exactly;
      a Poisson objective likewise gives the Poisson deviance of ``y`` at
      ``L * p``.
    - ``"none"``: the model fits ``y`` directly.

    The objective's link (log or identity) is read from the estimator.
    ``DIVISION`` is categorical with the levels ``BOOSTED_LEVELS``; other
    features are floats with nulls as NaN.

    Attributes:
        name: Label used in result tables.
        estimator: Unfitted estimator, cloned on every ``fit``.
        level: How the recent level enters the model.
    """

    def __init__(
        self, name: str, estimator: Booster, *, level: LevelMode = "offset"
    ) -> None:
        """Store the configuration.

        Args:
            name: Label used in result tables.
            estimator: Unfitted ``LGBMRegressor``, ``XGBRegressor`` or
                ``CatBoostRegressor``.
            level: ``"offset"``, ``"ratio"`` or ``"none"`` (see the class).

        Raises:
            TypeError: If ``estimator`` is not one of the supported types.
            ValueError: If ``level`` is unknown, or is ``"offset"`` with an
                objective that has no log link.
        """
        if not isinstance(estimator, LGBMRegressor | XGBRegressor | CatBoostRegressor):
            msg = f"unsupported estimator {type(estimator).__name__}"
            raise TypeError(msg)
        if level not in ("offset", "ratio", "none"):
            msg = f"level must be 'offset', 'ratio' or 'none', got {level!r}"
            raise ValueError(msg)
        if level == "offset" and not _log_link(estimator):
            msg = f"{name}: an offset needs a log-link objective"
            raise ValueError(msg)
        self.name = name
        self.estimator = estimator
        self.level: LevelMode = level
        self._model: Booster | None = None

    @property
    def log_link(self) -> bool:
        """Whether the estimator's objective has a log link."""
        return _log_link(self.estimator)

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
            frame["DIVISION"] = pd.Categorical(
                frame["DIVISION"], categories=BOOSTED_LEVELS
            )
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
                outside ``BOOSTED_LEVELS``.
        """
        rows = training_rows(history, spec)
        if rows.is_empty():
            msg = f"{self.name}: no training rows in history"
            raise ValueError(msg)
        _check_divisions(rows, BOOSTED_LEVELS)
        y = rows["y"].to_numpy()
        margin = log_offset(rows, spec) if self.level == "offset" else None
        weight = None
        if self.level == "ratio":
            weight = plain_level(rows, spec)
            y = y / weight
        model = cast(Booster, clone(self.estimator))
        match model:
            case CatBoostRegressor():
                model.fit(
                    Pool(
                        self._frame(rows, categorical=False),
                        y,
                        cat_features=["DIVISION"],
                        baseline=margin,
                        weight=weight,
                    )
                )
            case LGBMRegressor():
                model.fit(
                    self._frame(rows, categorical=True),
                    y,
                    sample_weight=weight,
                    init_score=margin,
                )
            case _:
                model.fit(
                    self._frame(rows, categorical=True),
                    y,
                    sample_weight=weight,
                    base_margin=margin,
                )
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
            null when ``history`` does not reach the cutoff or, with
            ``level`` ``"offset"`` or ``"ratio"``, ``mean_28d`` is missing.

        Raises:
            RuntimeError: If called before ``fit``.
            ValueError: If a division is outside ``BOOSTED_LEVELS``.
        """
        if self._model is None:
            msg = f"{self.name}: call fit before predict"
            raise RuntimeError(msg)
        rows = issue_rows(history, issue_date, spec)
        if self.level != "none":
            rows = rows.filter(pl.col("mean_28d") > 0)
        if rows.is_empty():
            return _with_divisions(history, rows, np.empty(0))
        _check_divisions(rows, BOOSTED_LEVELS)
        margin = log_offset(rows, spec) if self.level == "offset" else None
        log_link = self.log_link
        match self._model:
            case CatBoostRegressor() as model:
                pool = Pool(
                    self._frame(rows, categorical=False),
                    cat_features=["DIVISION"],
                    baseline=margin,
                )
                # The Pool baseline is added to the raw score; "Exponent" maps
                # it to the count scale.
                kind = "Exponent" if log_link else "RawFormulaVal"
                y_pred = model.predict(pool, prediction_type=kind)
            case LGBMRegressor() as model if log_link:
                raw = model.predict(self._frame(rows, categorical=True), raw_score=True)
                y_pred = np.exp(np.asarray(raw) + (0.0 if margin is None else margin))
            case LGBMRegressor() as model:
                y_pred = model.predict(self._frame(rows, categorical=True))
            case model:
                y_pred = model.predict(
                    self._frame(rows, categorical=True), base_margin=margin
                )
        y_pred = np.asarray(y_pred, dtype=np.float64)
        if self.level == "ratio":
            y_pred = y_pred * plain_level(rows, spec)
        return _with_divisions(history, rows, y_pred)


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


def _log_series(history: pl.DataFrame, division: int) -> tuple[pl.Series, np.ndarray]:
    """Daily ``log(n)`` of one division between its first and last known day.

    A zero count has no log, so it is treated as a missing day.

    Args:
        history: ``date, DIVISION, n, n_fr`` on the full calendar.
        division: Division to extract.

    Returns:
        Dates and ``log(n)``, NaN on missing days. Leading and trailing
        missing days are dropped, so both ends are known; both are empty when
        no day is known.
    """
    frame = (
        history.filter(pl.col("DIVISION") == division)
        .sort("date")
        .select("date", n=pl.when(pl.col("n") > 0).then(pl.col("n")))
    )
    known = frame["n"].is_not_null().arg_true()
    if known.is_empty():
        return frame["date"].clear(), np.empty(0)
    first, last = known[0], known[-1]
    frame = frame.slice(first, last - first + 1)
    y = np.log(frame["n"].cast(pl.Float64).fill_null(np.nan).to_numpy())
    return frame["date"], y


def _interpolate(y: np.ndarray) -> np.ndarray:
    """Fill NaN by linear interpolation over the day index.

    Args:
        y: Series with NaN on missing days.

    Returns:
        ``y`` with interior NaN interpolated and leading or trailing NaN set
        to the nearest known value.
    """
    missing = np.isnan(y)
    if not missing.any():
        return y
    index = np.arange(len(y))
    return np.interp(index, index[~missing], y[~missing])


def calendar_exog(dates: pl.Series) -> np.ndarray:
    """``AR_COLUMNS`` of each day: weekday, yearly Fourier, holidays, constant.

    The holiday columns come from ``features.add_calendar`` with a one-day
    window, so they match the model features day by day. The Fourier terms
    use the day of year over ``YEAR_DAYS``, as ``target_doy_sin`` does.

    Args:
        dates: Days to describe.

    Returns:
        A float matrix with one row per day.
    """
    frame = ft.add_calendar(pl.DataFrame({"target_start": dates}), ForecastSpec())
    doy = pl.col("target_start").dt.ordinal_day()

    def angle(k: int) -> pl.Expr:
        return 2 * math.pi * k * doy / YEAR_DAYS

    exprs = [
        *((pl.col("target_weekday") == k).cast(pl.Float64) for k in range(2, 8)),
        *(
            wave
            for k in range(1, FOURIER_ORDER + 1)
            for wave in (angle(k).sin(), angle(k).cos())
        ),
        pl.col("holidays").cast(pl.Float64),
        pl.col("holiday_adjacent").cast(pl.Float64),
        pl.lit(1.0),
    ]
    named = [e.alias(c) for e, c in zip(exprs, AR_COLUMNS, strict=True)]
    return frame.select(named).to_numpy().astype(np.float64)


class _DivisionStateSpace(ABC):
    """One state-space model on daily ``log(n)`` per division.

    ``fit`` estimates the parameters of every division with at least
    ``min_days`` known days. ``predict`` keeps those parameters, re-runs the
    filter on the division's days up to the cutoff and forecasts
    ``h = gap_days + window_days - 1`` steps past the cutoff, plus one step per
    missing day at the end of the history. The forecast is the sum of
    ``exp`` of the last ``window_days`` log-scale forecasts. With symmetric
    errors on the log scale, ``exp`` of the log forecast is the median count,
    the optimal point forecast under MAE.

    Attributes:
        name: Label used in result tables.
        min_days: Known days a division needs to be fit.
    """

    def __init__(self, name: str, min_days: int = MIN_FIT_DAYS) -> None:
        """Store the configuration.

        Args:
            name: Label used in result tables.
            min_days: Known days a division needs to be fit.
        """
        self.name = name
        self.min_days = min_days
        self._fitted: dict[int, Any] = {}

    @abstractmethod
    def _fit_division(self, dates: pl.Series, y: np.ndarray) -> Any:
        """Estimate one division's parameters.

        Args:
            dates: Consecutive days, first and last known.
            y: ``log(n)`` on those days, NaN where missing.

        Returns:
            Whatever ``_forecast_division`` needs.
        """

    @abstractmethod
    def _forecast_division(
        self, fitted: Any, dates: pl.Series, y: np.ndarray, steps: int
    ) -> np.ndarray:
        """Re-filter ``y`` with fixed parameters and forecast.

        Args:
            fitted: Output of ``_fit_division``.
            dates: Consecutive days, first and last known.
            y: ``log(n)`` on those days, NaN where missing.
            steps: Days to forecast after the last day of ``dates``.

        Returns:
            ``steps`` log-scale forecasts.
        """

    def fit(self, history: pl.DataFrame, spec: ForecastSpec) -> Self:
        """Estimate every division's parameters, replacing any earlier fit.

        Args:
            history: ``date, DIVISION, n, n_fr`` on the full calendar.
            spec: Forecast setting (the parameters do not depend on it).

        Returns:
            This forecaster.

        Raises:
            ValueError: If ``history`` fails ``check_history`` or no division
                has ``min_days`` known days.
        """
        ft.check_history(history)
        self._fitted = {}
        fitted: dict[int, Any] = {}
        for division in history["DIVISION"].unique().sort().to_list():
            dates, y = _log_series(history, division)
            if np.isfinite(y).sum() >= self.min_days:
                fitted[division] = self._fit_division(dates, y)
        if not fitted:
            msg = f"{self.name}: no training rows in history"
            raise ValueError(msg)
        self._fitted = fitted
        return self

    def predict(
        self, history: pl.DataFrame, issue_date: dt.date, spec: ForecastSpec
    ) -> pl.DataFrame:
        """Forecast the window of ``issue_date`` for every division.

        Args:
            history: ``date, DIVISION, n, n_fr`` on the full calendar; only
                days up to the cutoff are read.
            issue_date: Day the forecast is issued.
            spec: Forecast setting.

        Returns:
            ``DIVISION, y_pred`` for every division in ``history``, sorted;
            null when ``history`` does not reach the cutoff or the division
            was not fit.

        Raises:
            RuntimeError: If called before ``fit``.
        """
        if not self._fitted:
            msg = f"{self.name}: call fit before predict"
            raise RuntimeError(msg)
        cutoff = spec.cutoff(issue_date)
        known = history.filter(pl.col("date") <= cutoff)
        forecasts: dict[int, float] = {}
        if not known.is_empty() and known["date"].max() == cutoff:
            for division, fitted in self._fitted.items():
                dates, y = _log_series(known, division)
                if dates.is_empty():
                    continue
                last: dt.date = dates[-1]
                steps = (cutoff - last).days + spec.gap_days + spec.window_days - 1
                path = self._forecast_division(fitted, dates, y, steps)
                total = float(np.exp(path[-spec.window_days :]).sum())
                if math.isfinite(total):
                    forecasts[division] = total
        predicted = pl.DataFrame(
            {"DIVISION": list(forecasts), "y_pred": list(forecasts.values())},
            schema={"DIVISION": pl.Int64, "y_pred": pl.Float64},
        )
        return (
            history.select(pl.col("DIVISION").unique().sort())
            .join(predicted, on="DIVISION", how="left")
            .select("DIVISION", "y_pred")
        )


@dataclass(frozen=True)
class _EtsFit:
    """Fitted ETS parameters of one division.

    Attributes:
        start: First day of the fitted series; the initial states refer to it.
        params: ``ETSModel`` parameters, initial states included.
    """

    start: dt.date
    params: np.ndarray


class ETSWeekly(_DivisionStateSpace):
    """ETS(A,N,A) on ``log(n)``: additive error, no trend, weekly seasonality.

    ``statsmodels`` ``ETSModel`` with estimated initial states. There is no
    trend: a linear trend over-predicts after the late-2025 level break.
    ``ETSModel`` does not accept missing values, so interior missing days are
    filled by linear interpolation in log space, at fit and at predict.
    ``predict`` runs ``ETSModel.smooth`` with the fitted parameters on the new
    series, starting on a day that is a whole number of weeks after the fitted
    start (up to 6 leading days are dropped), so each initial seasonal state
    keeps its weekday.

    Attributes:
        name: Label used in result tables.
        min_days: Known days a division needs to be fit.
        maxiter: Iteration limit of the optimiser.
    """

    def __init__(
        self,
        name: str = "ets_weekly",
        maxiter: int = 1000,
        min_days: int = MIN_FIT_DAYS,
    ) -> None:
        """Store the configuration.

        Args:
            name: Label used in result tables.
            maxiter: Iteration limit of the optimiser (L-BFGS-B).
            min_days: Known days a division needs to be fit.
        """
        super().__init__(name, min_days)
        self.maxiter = maxiter

    @staticmethod
    def _model(y: np.ndarray) -> ETSModel:
        """The ETS(A,N,A) model of a log series.

        Args:
            y: ``log(n)`` without NaN.

        Returns:
            An unfitted ``ETSModel``.
        """
        return ETSModel(
            y, error="add", trend=None, seasonal="add", seasonal_periods=WEEK
        )

    def _fit_division(self, dates: pl.Series, y: np.ndarray) -> _EtsFit:
        """Estimate the smoothing parameters and initial states.

        Args:
            dates: Consecutive days, first and last known.
            y: ``log(n)`` on those days, NaN where missing.

        Returns:
            The first day and the fitted parameters.

        Raises:
            RuntimeError: If the optimiser returns a non-finite parameter.
        """
        result: Any = self._model(_interpolate(y)).fit(disp=False, maxiter=self.maxiter)
        params = np.asarray(result.params, dtype=np.float64)
        if not np.isfinite(params).all():
            msg = f"{self.name}: ETS gave non-finite parameters"
            raise RuntimeError(msg)
        return _EtsFit(start=dates[0], params=params)

    def _forecast_division(
        self, fitted: _EtsFit, dates: pl.Series, y: np.ndarray, steps: int
    ) -> np.ndarray:
        """Smooth the new series with the fitted parameters and forecast.

        Args:
            fitted: Start day and parameters from ``_fit_division``.
            dates: Consecutive days, first and last known.
            y: ``log(n)`` on those days, NaN where missing.
            steps: Days to forecast after the last day of ``dates``.

        Returns:
            ``steps`` log-scale forecasts; NaN when fewer than two weeks of
            days are left after aligning the start (``ETSModel`` needs two
            seasonal cycles to build its heuristic start values).
        """
        first: dt.date = dates[0]
        skip = (fitted.start - first).days % WEEK
        if len(y) - skip < 2 * WEEK:
            return np.full(steps, np.nan)
        result: Any = self._model(_interpolate(y[skip:])).smooth(fitted.params)
        return np.asarray(result.forecast(steps), dtype=np.float64)


class ARCalendar(_DivisionStateSpace):
    """Deseasonalised autoregression on ``log(n)`` (Channouf et al. 2007 style).

    ``statsmodels`` ``SARIMAX`` regression of ``log(n)`` on ``calendar_exog``
    (weekday dummies, yearly Fourier terms of order 3, the Québec holiday and
    holiday-adjacent indicators and a constant) with AR(``order``) errors and
    no differencing. The Kalman filter skips missing days natively. ``predict``
    rebuilds the model on the new series and runs ``filter`` with the fitted
    parameters (what ``results.apply`` does, without the smoother), then
    forecasts with the calendar regressors of the target days.

    Not registered in ``default_models``: on 2021-2024 it scored MAE 8.743 and
    bias -2.38 (commit aab221e), because its stationary constant pulls forecasts
    toward a long-run mean that includes the 2020 dip.

    Attributes:
        name: Label used in result tables.
        min_days: Known days a division needs to be fit.
        order: Autoregressive order.
        method: ``statsmodels`` optimiser.
        maxiter: Iteration limit of the optimiser.
    """

    def __init__(
        self,
        name: str = "ar_calendar",
        order: int = 7,
        method: str = "lbfgs",
        maxiter: int = 200,
        min_days: int = MIN_FIT_DAYS,
    ) -> None:
        """Store the configuration.

        Args:
            name: Label used in result tables.
            order: Autoregressive order.
            method: ``statsmodels`` optimiser.
            maxiter: Iteration limit of the optimiser; the ``statsmodels``
                default of 50 is not always enough for 23 parameters.
            min_days: Known days a division needs to be fit.
        """
        super().__init__(name, min_days)
        self.order = order
        self.method = method
        self.maxiter = maxiter

    def _fit_division(self, dates: pl.Series, y: np.ndarray) -> Any:
        """Estimate the regression, AR and variance parameters.

        Args:
            dates: Consecutive days, first and last known.
            y: ``log(n)`` on those days, NaN where missing.

        Returns:
            The ``SARIMAX`` results.

        Raises:
            RuntimeError: If the optimiser returns a non-finite parameter.
        """
        model = SARIMAX(
            y, exog=calendar_exog(dates), order=(self.order, 0, 0), trend="n"
        )
        result: Any = model.fit(
            disp=False, method=self.method, maxiter=self.maxiter, cov_type="none"
        )
        if not np.isfinite(result.params).all():
            msg = f"{self.name}: SARIMAX gave non-finite parameters"
            raise RuntimeError(msg)
        return result

    def _forecast_division(
        self, fitted: Any, dates: pl.Series, y: np.ndarray, steps: int
    ) -> np.ndarray:
        """Filter the new series with the fitted parameters and forecast.

        Args:
            fitted: ``SARIMAX`` results from ``_fit_division``.
            dates: Consecutive days, first and last known.
            y: ``log(n)`` on those days, NaN where missing.
            steps: Days to forecast after the last day of ``dates``.

        Returns:
            ``steps`` log-scale forecasts (the conditional means).
        """
        last: dt.date = dates[-1]
        future = pl.date_range(
            last + dt.timedelta(days=1), last + dt.timedelta(days=steps), eager=True
        )
        filtered = fitted.model.clone(y, exog=calendar_exog(dates)).filter(
            fitted.params, cov_type="none"
        )
        forecast = filtered.forecast(steps, exog=calendar_exog(future))
        return np.asarray(forecast, dtype=np.float64)


def citywide(history: pl.DataFrame) -> pl.DataFrame:
    """Sum every division per day into one series with ``DIVISION`` ``CITYWIDE``.

    Args:
        history: ``date, DIVISION, n, n_fr`` on the full calendar.

    Returns:
        ``date, DIVISION, n, n_fr`` with the input dtypes, sorted by date;
        ``n`` and ``n_fr`` are null on a day where any division is null or
        has no row.

    Raises:
        ValueError: If ``history`` fails ``check_history``.
    """
    ft.check_history(history)
    complete = pl.col("n").count() == history["DIVISION"].n_unique()
    return (
        history.group_by("date")
        .agg(
            n=pl.when(complete).then(pl.col("n").sum()),
            n_fr=pl.when(complete).then(pl.col("n_fr").sum()),
        )
        .select(
            "date",
            pl.lit(CITYWIDE, dtype=pl.Int64).alias("DIVISION"),
            pl.col("n").cast(history.schema["n"]),
            pl.col("n_fr").cast(history.schema["n_fr"]),
        )
        .sort("date")
    )


def division_shares(
    history: pl.DataFrame, cutoff: dt.date, days: int = 28
) -> pl.DataFrame:
    """Each division's share of the citywide count over the days ending at ``cutoff``.

    Only days in the window where every division is known count, for the
    division sums and the citywide sum alike, so the shares add up to 1.

    Args:
        history: ``date, DIVISION, n, n_fr`` on the full calendar.
        cutoff: Last day of the window.
        days: Window length.

    Returns:
        ``DIVISION, share`` for every division in ``history``, sorted; null
        when fewer than ``features.min_samples(days)`` days are complete.
    """
    divisions = history.select(pl.col("DIVISION").unique().sort())
    window = history.filter(
        pl.col("date").is_between(cutoff - dt.timedelta(days=days - 1), cutoff)
    )
    complete = (
        window.group_by("date")
        .agg(known=pl.col("n").count())
        .filter(pl.col("known") == divisions.height)
    )
    totals = (
        window.join(complete, on="date", how="semi")
        .group_by("DIVISION")
        .agg(n=pl.col("n").cast(pl.Float64).sum())
    )
    enough = complete.height >= ft.min_samples(days)
    return divisions.join(totals, on="DIVISION", how="left").select(
        "DIVISION",
        share=pl.when(pl.lit(enough)).then(pl.col("n") / pl.col("n").sum()),
    )


class TopDown:
    """Forecast the citywide total with ``base`` and split it by division share.

    ``fit`` fits a deep copy of ``base`` on ``citywide(history)``, so ``base``
    itself stays unfitted. ``predict`` multiplies the citywide forecast by
    ``division_shares`` over the ``share_days`` days ending at the cutoff.

    Attributes:
        name: Label used in result tables.
        base: Unfitted forecaster of the citywide series, copied on every fit.
        share_days: Days the division shares are measured over.
    """

    def __init__(self, name: str, base: Forecaster, share_days: int = 28) -> None:
        """Store the configuration.

        Args:
            name: Label used in result tables.
            base: Unfitted forecaster of the citywide series.
            share_days: Days the division shares are measured over.

        Raises:
            ValueError: If ``share_days`` is below 1.
        """
        if share_days < 1:
            msg = f"share_days must be >= 1, got {share_days}"
            raise ValueError(msg)
        self.name = name
        self.base = base
        self.share_days = share_days
        self._model: Forecaster | None = None

    def fit(self, history: pl.DataFrame, spec: ForecastSpec) -> Self:
        """Fit a fresh copy of ``base`` on the citywide series.

        Args:
            history: ``date, DIVISION, n, n_fr`` on the full calendar.
            spec: Forecast setting.

        Returns:
            This forecaster.

        Raises:
            ValueError: If ``history`` fails ``check_history``, or as raised
                by the base ``fit``.
        """
        self._model = None
        self._model = copy.deepcopy(self.base).fit(citywide(history), spec)
        return self

    def predict(
        self, history: pl.DataFrame, issue_date: dt.date, spec: ForecastSpec
    ) -> pl.DataFrame:
        """Forecast the window of ``issue_date`` for every division.

        Args:
            history: ``date, DIVISION, n, n_fr`` on the full calendar; only
                days up to the cutoff are read.
            issue_date: Day the forecast is issued.
            spec: Forecast setting.

        Returns:
            ``DIVISION, y_pred`` for every division in ``history``, sorted;
            null when the citywide forecast or the share is null.

        Raises:
            RuntimeError: If called before ``fit``.
        """
        if self._model is None:
            msg = f"{self.name}: call fit before predict"
            raise RuntimeError(msg)
        cutoff = spec.cutoff(issue_date)
        known = history.filter(pl.col("date") <= cutoff)
        city = self._model.predict(citywide(known), issue_date, spec)["y_pred"]
        total = city[0] if city.len() else None
        shares = division_shares(known, cutoff, self.share_days)
        return (
            history.select(pl.col("DIVISION").unique().sort())
            .join(shares, on="DIVISION", how="left")
            .select(
                "DIVISION",
                y_pred=pl.col("share") * pl.lit(total, dtype=pl.Float64),
            )
        )


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
        and Tweedie; CatBoost Poisson and Tweedie; the NB GLM; LightGBM L1 on
        raw counts and on the ratio to the level; ETS(A,N,A) per division;
        top-down LightGBM Poisson (raw) and top-down ETS.
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

    def lgbm_poisson_raw() -> BoostedForecaster:
        return BoostedForecaster(
            "lgbm_poisson_raw", LGBMRegressor(objective="poisson", **lgbm), level="none"
        )

    return [
        BoostedForecaster("lgbm_poisson", LGBMRegressor(objective="poisson", **lgbm)),
        BoostedForecaster(
            "lgbm_tweedie",
            LGBMRegressor(objective="tweedie", tweedie_variance_power=1.5, **lgbm),
        ),
        lgbm_poisson_raw(),
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
        BoostedForecaster(
            "lgbm_l1_raw", LGBMRegressor(objective="l1", **lgbm), level="none"
        ),
        BoostedForecaster(
            "lgbm_l1", LGBMRegressor(objective="l1", **lgbm), level="ratio"
        ),
        ETSWeekly(),
        TopDown("topdown_lgbm_poisson_raw", lgbm_poisson_raw()),
        TopDown("topdown_ets", ETSWeekly()),
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


def rounded(value: Any) -> Any:
    """Round every float in a nested structure to 3 decimals.

    Args:
        value: Dict, list, float or other JSON value.

    Returns:
        The same structure with floats rounded; non-finite floats become None.
    """
    match value:
        case dict():
            return {k: rounded(v) for k, v in value.items()}
        case list() | tuple():
            return [rounded(v) for v in value]
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
        by month, by year, bias and median error by year and division, paired
        bootstraps against each of ``REFERENCES``, timings and library
        versions, floats rounded to 3) and the raw results with an ``anomaly``
        column. The year is the calendar year of ``target_start``. Every score
        table has ``n``, ``mae``, ``bias``, ``median_error`` and
        ``under_share``.
    """
    timed = [Timed(m) for m in (*forecasters, SameWeekdayMean(13), PlainMean())]
    results = flag_anomalies(
        rolling_origin(history, timed, spec, start, end), anomalies
    )
    clean = results.filter(~pl.col("anomaly"))
    with_year = results.with_columns(year=pl.col("target_start").dt.year())

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
        "by_year": table(with_year, ["year"]),
        "by_year_division": score(with_year, ["year", "DIVISION"])
        .select("model", "year", "DIVISION", "bias", "median_error")
        .to_dicts(),
        "bootstrap": bootstrap,
        "seconds": {
            m.name: {"fit": m.fit_seconds, "predict": m.predict_seconds} for m in timed
        },
        "versions": {lib: version(lib) for lib in LIBRARIES},
    }
    return rounded(report), results


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
            f"{row['model']:<26} MAE {row['mae']:6.3f}  bias {row['bias']:+6.3f}  "
            f"MAE without anomalies {clean[row['model']]:6.3f}"
        )
    typer.echo(f"wrote {out} and {predictions}")


if __name__ == "__main__":
    app()
