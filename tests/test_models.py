import copy
import datetime as dt
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import polars as pl
import pytest
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor
from polars.testing import assert_frame_equal
from sklearn.base import clone
from sklearn.linear_model import PoissonRegressor
from typer.testing import CliRunner
from xgboost import XGBRegressor

from ycit440_sim_forecast import evaluate as ev
from ycit440_sim_forecast import features as ft
from ycit440_sim_forecast import models as md
from ycit440_sim_forecast import spec
from ycit440_sim_forecast.spec import Forecaster, ForecastSpec

runner = CliRunner()
FIRST = dt.date(2023, 1, 1)
MISSING = dt.date(2024, 3, 31)
START = dt.date(2024, 2, 1)
END = dt.date(2024, 3, 10)
PRESETS = [spec.V2_DAILY, spec.WEEKLY, *spec.daily_horizon(14)]
REGISTERED_NAMES = [
    "lgbm_poisson",
    "lgbm_tweedie",
    "lgbm_poisson_raw",
    "xgb_poisson",
    "xgb_tweedie",
    "cat_poisson",
    "cat_tweedie",
    "nb_glm",
    "lgbm_l1_raw",
    "lgbm_l1",
    "ets_weekly",
    "topdown_lgbm_poisson_raw",
    "topdown_ets",
]
"""Names returned by ``default_models``, in order."""
NAMES = [*REGISTERED_NAMES, "ar_calendar"]
"""Every model under test; ``ar_calendar`` is tested but not registered."""
BOOSTED_NAMES = [
    "lgbm_poisson",
    "lgbm_tweedie",
    "lgbm_poisson_raw",
    "xgb_poisson",
    "xgb_tweedie",
    "cat_poisson",
    "cat_tweedie",
]
"""Models that existed before the level modes; their predictions are pinned."""
RELATIVE_NAMES = [
    "lgbm_poisson",
    "lgbm_tweedie",
    "xgb_poisson",
    "xgb_tweedie",
    "cat_poisson",
    "cat_tweedie",
    "lgbm_l1",
]
"""Boosters whose forecast is relative to the plain 28-day mean."""
STATE_SPACE_NAMES = ["ets_weekly", "ar_calendar"]
WEEKLY_EFFECT = np.array([1.1, 1.05, 1.0, 1.0, 1.05, 0.9, 0.9])


def make_history(seed: int = 0) -> pl.DataFrame:
    # 2023-2024 for divisions 1 and 2, with one day that has no records at all.
    rng = np.random.default_rng(seed)
    dates = pl.date_range(FIRST, dt.date(2024, 12, 31), "1d", eager=True)
    n = rng.integers(20, 80, size=2 * len(dates))
    return (
        pl.DataFrame({"date": dates})
        .join(pl.DataFrame({"DIVISION": [1, 2]}), how="cross")
        .with_columns(
            n=pl.Series(n, dtype=pl.UInt32),
            n_fr=pl.Series(rng.integers(0, n + 1), dtype=pl.UInt32),
        )
        .with_columns(pl.when(pl.col("date") != MISSING).then(pl.col("n", "n_fr")))
        .sort("date", "DIVISION")
    )


def constant_history(levels: dict[int, int], last: dt.date) -> pl.DataFrame:
    # Every day of each division has the same count, a third first responders.
    dates = pl.date_range(FIRST, last, "1d", eager=True)
    return (
        pl.DataFrame({"date": dates})
        .join(pl.DataFrame({"DIVISION": list(levels)}), how="cross")
        .with_columns(
            n=pl.col("DIVISION").replace_strict(levels, return_dtype=pl.UInt32)
        )
        .with_columns(n_fr=pl.col("n") // 3)
        .sort("date", "DIVISION")
    )


def pattern_history(
    levels: list[tuple[dt.date, int]], last: dt.date, seed: int = 0
) -> pl.DataFrame:
    # Division 1 only: a piecewise-constant level from each date in ``levels``,
    # times WEEKLY_EFFECT, with 5% log-normal noise.
    rng = np.random.default_rng(seed)
    dates = pl.date_range(FIRST, last, "1d", eager=True)
    starts = np.array([np.datetime64(d) for d, _ in levels])
    index = np.searchsorted(starts, dates.to_numpy(), side="right") - 1
    level = np.array([v for _, v in levels])[index]
    weekday = dates.dt.weekday().to_numpy() - 1
    noise = np.exp(rng.normal(0, 0.05, len(dates)))
    n = np.round(level * WEEKLY_EFFECT[weekday] * noise)
    return pl.DataFrame(
        {
            "date": dates,
            "DIVISION": pl.Series([1] * len(dates), dtype=pl.Int64),
            "n": pl.Series(n, dtype=pl.UInt32),
        }
    ).with_columns(n_fr=pl.col("n") // 3)


def light(model: Forecaster) -> Forecaster:
    # Same configuration with 20 trees, so the suite stays fast.
    if isinstance(model, md.TopDown):
        return md.TopDown(model.name, light(model.base), model.share_days)
    if isinstance(model, md.BoostedForecaster):
        trees = (
            {"iterations": 20}
            if isinstance(model.estimator, CatBoostRegressor)
            else {"n_estimators": 20}
        )
        estimator = cast(md.Booster, clone(model.estimator))
        estimator.set_params(**trees)
        return md.BoostedForecaster(model.name, estimator, level=model.level)
    return model


def light_models() -> list[Forecaster]:
    return [*(light(m) for m in md.default_models()), md.ARCalendar()]


def by_name(name: str) -> Forecaster:
    return next(m for m in light_models() if m.name == name)


@pytest.fixture(scope="module")
def history() -> pl.DataFrame:
    return make_history()


def cut(frame: pl.DataFrame, last: dt.date) -> pl.DataFrame:
    return frame.filter(pl.col("date") <= last)


def test_default_models_names_and_types() -> None:
    models = md.default_models()
    assert [m.name for m in models] == REGISTERED_NAMES
    for m in models:
        assert isinstance(
            m, md.BoostedForecaster | md.NegBinGLM | md.ETSWeekly | md.TopDown
        )
    boosted = {m.name: m for m in models if isinstance(m, md.BoostedForecaster)}
    levels = {name: m.level for name, m in boosted.items()}
    assert [n for n, lv in levels.items() if lv == "none"] == [
        "lgbm_poisson_raw",
        "lgbm_l1_raw",
    ]
    assert [n for n, lv in levels.items() if lv == "ratio"] == ["lgbm_l1"]
    assert [n for n, m in boosted.items() if not m.log_link] == [
        "lgbm_l1_raw",
        "lgbm_l1",
    ]
    tops = {m.name: m for m in models if isinstance(m, md.TopDown)}
    base = tops["topdown_lgbm_poisson_raw"].base
    assert isinstance(base, md.BoostedForecaster)
    assert base.level == "none"
    assert (
        base.estimator.get_params()
        == boosted["lgbm_poisson_raw"].estimator.get_params()
    )
    assert isinstance(tops["topdown_ets"].base, md.ETSWeekly)


def test_ensemble_candidates_are_registered() -> None:
    names = {m.name for m in md.default_models()}
    assert set(md.ENSEMBLE_CANDIDATES) <= names
    assert len(set(md.ENSEMBLE_CANDIDATES)) == len(md.ENSEMBLE_CANDIDATES)


def test_default_models_are_unfitted_and_independent() -> None:
    first, second = md.default_models(), md.default_models()
    for a, b in zip(first, second, strict=True):
        assert a is not b
        if isinstance(a, md.BoostedForecaster):
            assert a.estimator is not b.estimator  # pyright: ignore[reportAttributeAccessIssue]
            assert a._model is None
        if isinstance(a, md.TopDown):
            assert a.base is not b.base  # pyright: ignore[reportAttributeAccessIssue]
            assert a._model is None


def test_every_model_runs_through_rolling_origin(history: pl.DataFrame) -> None:
    models = light_models()
    results = ev.rolling_origin(history, models, spec.V2_DAILY, START, END)
    days = len(ev.issue_dates(START, END, spec.V2_DAILY))
    assert results.height == len(models) * days * 2
    scored = results.filter(pl.col("y_true").is_not_null())
    assert scored["y_pred"].is_not_null().all()
    assert (scored["y_pred"] > 0).all()
    table = ev.score(results)
    assert set(table["model"]) == set(NAMES)
    assert (table["mae"] < 30).all()  # counts are uniform on 20-79


def test_real_default_configs_smoke(history: pl.DataFrame) -> None:
    end = START + dt.timedelta(days=3)
    results = ev.rolling_origin(history, md.default_models(), spec.V2_DAILY, START, end)
    assert results["y_pred"].is_not_null().all()
    assert (ev.score(results)["mae"] < 30).all()


@pytest.mark.parametrize("name", NAMES)
def test_harness_ignores_counts_after_cutoff(history: pl.DataFrame, name: str) -> None:
    """Poisoning every count after an issue date's cutoff leaves its forecast unchanged.

    ``rolling_origin`` slices history at each cutoff before ``fit`` and
    ``predict``, which is the contract every model relies on.
    """
    issue = dt.date(2024, 2, 20)
    cutoff = spec.V2_DAILY.cutoff(issue)
    poisoned = history.with_columns(
        pl.when(pl.col("date") > cutoff)
        .then(pl.lit(10_000, dtype=pl.UInt32))
        .otherwise(pl.col(c))
        .alias(c)
        for c in ("n", "n_fr")
    )
    runs = [
        ev.rolling_origin(h, [by_name(name)], spec.V2_DAILY, START, END)
        .filter(pl.col("issue_date") <= issue)
        .select("issue_date", "DIVISION", "y_pred")
        for h in (history, poisoned)
    ]
    assert runs[0].height > 0
    assert_frame_equal(runs[0], runs[1])


@pytest.mark.parametrize("name", NAMES)
def test_predict_reads_only_up_to_cutoff(history: pl.DataFrame, name: str) -> None:
    """``predict`` uses only the feature row of ``issue_date``, cut at its cutoff.

    ``fit`` learns from every row whose target window ends by the last day of
    the history it is given, so a model fitted on unsliced history does see
    later targets; preventing that is the harness's job (tested above). Given
    a fitted model, ``predict`` on unsliced, poisoned history matches
    ``predict`` on history ending at the cutoff.
    """
    issue = dt.date(2024, 5, 10)
    cutoff = spec.V2_DAILY.cutoff(issue)
    model = by_name(name).fit(cut(history, cutoff), spec.V2_DAILY)
    poisoned = history.with_columns(
        pl.when(pl.col("date") > cutoff)
        .then(pl.lit(10_000, dtype=pl.UInt32))
        .otherwise(pl.col(c))
        .alias(c)
        for c in ("n", "n_fr")
    )
    clean = model.predict(cut(history, cutoff), issue, spec.V2_DAILY)
    assert clean["y_pred"].is_not_null().all()
    assert_frame_equal(clean, model.predict(poisoned, issue, spec.V2_DAILY))


@pytest.mark.parametrize("name", NAMES)
def test_refit_resets_state(history: pl.DataFrame, name: str) -> None:
    issue = dt.date(2024, 4, 5)
    a = cut(history, dt.date(2024, 9, 30))
    b = cut(history, spec.V2_DAILY.cutoff(issue))
    refitted = by_name(name).fit(a, spec.V2_DAILY).fit(b, spec.V2_DAILY)
    fresh = by_name(name).fit(b, spec.V2_DAILY)
    assert_frame_equal(
        refitted.predict(b, issue, spec.V2_DAILY),
        fresh.predict(b, issue, spec.V2_DAILY),
    )


@pytest.mark.parametrize("name", NAMES)
def test_deterministic(history: pl.DataFrame, name: str) -> None:
    runs = [
        ev.rolling_origin(history, [by_name(name)], spec.V2_DAILY, START, START)
        for _ in range(2)
    ]
    assert_frame_equal(runs[0], runs[1])


@pytest.mark.parametrize("name", RELATIVE_NAMES)
def test_offset_carries_an_unseen_level(name: str) -> None:
    # Trained where every count equals its level, the model learns no
    # adjustment, so it must return the new level even far outside training.
    train = constant_history({1: 30, 2: 60}, dt.date(2024, 6, 30))
    model = by_name(name).fit(train, spec.V2_DAILY)
    shifted = constant_history({1: 300, 2: 60}, dt.date(2024, 6, 30))
    issue = dt.date(2024, 7, 2)
    got = model.predict(shifted, issue, spec.V2_DAILY)["y_pred"].to_numpy()
    np.testing.assert_allclose(got, [300.0, 60.0], rtol=1e-3)
    weekly = model.fit(train, spec.WEEKLY).predict(shifted, issue, spec.WEEKLY)
    np.testing.assert_allclose(weekly["y_pred"].to_numpy(), [2100.0, 420.0], rtol=1e-3)


def test_raw_model_cannot_follow_an_unseen_level() -> None:
    train = constant_history({1: 30, 2: 60}, dt.date(2024, 6, 30))
    model = by_name("lgbm_poisson_raw").fit(train, spec.V2_DAILY)
    shifted = constant_history({1: 300, 2: 60}, dt.date(2024, 6, 30))
    got = model.predict(shifted, dt.date(2024, 7, 2), spec.V2_DAILY)["y_pred"]
    assert got[0] < 61  # stays within the training range


def test_raw_model_predicts_without_mean_28d(history: pl.DataFrame) -> None:
    # 10 days of history: no mean_28d, so offset models give nulls.
    short = cut(history, FIRST + dt.timedelta(days=9))
    issue = FIRST + dt.timedelta(days=11)
    raw = by_name("lgbm_poisson_raw").fit(history, spec.V2_DAILY)
    with_offset = by_name("lgbm_poisson").fit(history, spec.V2_DAILY)
    assert raw.predict(short, issue, spec.V2_DAILY)["y_pred"].is_not_null().all()
    assert with_offset.predict(short, issue, spec.V2_DAILY)["y_pred"].is_null().all()


@pytest.mark.parametrize("loss", ["Poisson", "Tweedie:variance_power=1.5"])
def test_catboost_pool_baseline_is_added_at_predict(loss: str) -> None:
    # Pins the behaviour BoostedForecaster relies on: a Pool baseline enters
    # the raw score at predict, for both prediction types, and "Exponent" is
    # the count scale. The target is exactly twice the baseline level.
    rng = np.random.default_rng(0)
    x = pd.DataFrame({"a": rng.normal(size=500)})
    level = rng.uniform(20, 100, 500)
    model = CatBoostRegressor(
        loss_function=loss,
        iterations=300,
        random_seed=0,
        logging_level="Silent",
        allow_writing_files=False,
    ).fit(Pool(x, 2 * level, baseline=np.log(level)))
    with_base = Pool(x, baseline=np.log(level))
    raw = model.predict(with_base, prediction_type="RawFormulaVal")
    np.testing.assert_allclose(raw, np.log(2 * level), atol=5e-3)
    count = model.predict(with_base, prediction_type="Exponent")
    np.testing.assert_allclose(count, 2 * level, rtol=5e-3)
    np.testing.assert_allclose(model.predict(with_base), count)
    without = model.predict(Pool(x), prediction_type="Exponent")
    np.testing.assert_allclose(without, 2.0, rtol=5e-3)


@pytest.mark.parametrize(
    "estimator",
    [
        LGBMRegressor(objective="poisson", n_estimators=300, verbose=-1),
        XGBRegressor(objective="count:poisson", n_estimators=300),
    ],
    ids=["lightgbm", "xgboost"],
)
def test_lightgbm_xgboost_offset_paths(estimator: LGBMRegressor | XGBRegressor) -> None:
    # init_score is left out of raw_score, base_margin must be passed again;
    # the multiplier depends on a binary feature so LightGBM finds a split and
    # no histogram bin mixes the two groups.
    rng = np.random.default_rng(0)
    x = pd.DataFrame({"a": rng.integers(0, 2, size=1000).astype(float)})
    level = rng.uniform(20, 100, 1000)
    y = np.where(x["a"] == 1, 2.0, 0.5) * level
    if isinstance(estimator, LGBMRegressor):
        estimator.fit(x, y, init_score=np.log(level))
        got = np.exp(np.asarray(estimator.predict(x, raw_score=True)) + np.log(level))
    else:
        estimator.fit(x, y, base_margin=np.log(level))
        got = estimator.predict(x, base_margin=np.log(level))
    np.testing.assert_allclose(got, y, rtol=1e-2)


def test_lookback_value() -> None:
    assert md.LOOKBACK.days == 90


@pytest.mark.parametrize("setting", PRESETS, ids=lambda s: s.key)
def test_tail_features_equal_full_history(
    history: pl.DataFrame, setting: ForecastSpec
) -> None:
    full = ft.feature_table(history, setting).drop("y")
    issues = [FIRST + dt.timedelta(days=d) for d in (40, 95, 200, 452, 455, 470, 731)]
    for issue in issues:
        expected = full.filter(pl.col("issue_date") == issue)
        got = md.issue_rows(history, issue, setting).drop("y")
        assert got.height == 2
        assert_frame_equal(got, expected, abs_tol=1e-9)


def test_tail_one_day_shorter_changes_features(history: pl.DataFrame) -> None:
    # LOOKBACK is tight: dropping its first day changes mean_91d.
    issue = FIRST + dt.timedelta(days=300)
    cutoff = spec.V2_DAILY.cutoff(issue)
    shorter = history.filter(pl.col("date") > cutoff - md.LOOKBACK)
    full = md.issue_rows(history, issue, spec.V2_DAILY)
    short = md.issue_rows(shorter, issue, spec.V2_DAILY)
    assert not full["mean_91d"].equals(short["mean_91d"])


def test_predict_without_cutoff_row_is_null(history: pl.DataFrame) -> None:
    issue = dt.date(2024, 5, 10)
    stale = cut(history, spec.V2_DAILY.cutoff(issue) - dt.timedelta(days=1))
    for name in (
        "lgbm_poisson",
        "lgbm_poisson_raw",
        "nb_glm",
        "lgbm_l1",
        "ets_weekly",
        "ar_calendar",
        "topdown_ets",
    ):
        model = by_name(name).fit(stale, spec.V2_DAILY)
        got = model.predict(stale, issue, spec.V2_DAILY)
        assert got["DIVISION"].to_list() == [1, 2]
        assert got["y_pred"].is_null().all()


def test_glm_null_features_give_finite_prediction(history: pl.DataFrame) -> None:
    # Cutoff and the 3 days before it are missing: lag_0 and mean_7d are null.
    issue = dt.date(2024, 5, 10)
    cutoff = spec.V2_DAILY.cutoff(issue)
    gappy = cut(history, cutoff).with_columns(
        pl.when(pl.col("date") <= cutoff - dt.timedelta(days=4)).then(
            pl.col("n", "n_fr")
        )
    )
    rows = md.issue_rows(gappy, issue, spec.V2_DAILY)
    assert rows["lag_0"].is_null().all()
    assert rows["ratio_7_28"].is_null().all()
    model = md.NegBinGLM().fit(cut(history, cutoff), spec.V2_DAILY)
    design = model.design(rows)
    assert np.isfinite(design).all()
    assert (design[:, md.GLM_COLUMNS.index("log_ratio_7_28")] == 0).all()
    assert (design[:, md.GLM_COLUMNS.index("log_lag_0_rel")] == 0).all()
    got = model.predict(gappy, issue, spec.V2_DAILY)["y_pred"].to_numpy()
    assert np.isfinite(got).all()
    assert (got > 0).all()


def test_glm_fills_fr_share_with_training_median(history: pl.DataFrame) -> None:
    train = cut(history, dt.date(2024, 5, 1))
    model = md.NegBinGLM().fit(train, spec.V2_DAILY)
    median = md.training_rows(train, spec.V2_DAILY)["fr_share_28d"].to_numpy()
    rows = md.issue_rows(history, dt.date(2024, 5, 3), spec.V2_DAILY).with_columns(
        fr_share_28d=None
    )
    column = model.design(rows)[:, md.GLM_COLUMNS.index("fr_share_28d")]
    np.testing.assert_allclose(column, np.median(median))


def test_glm_drops_absent_levels(history: pl.DataFrame) -> None:
    model = md.NegBinGLM().fit(history, spec.V2_DAILY)
    assert "division_2" in model.columns
    assert "division_3" not in model.columns
    assert model.columns[-1] == "const"
    assert md.NegBinGLM().columns == []
    # The constant plus alpha: one parameter per kept column, then alpha.
    assert len(model._result.params) == len(model.columns) + 1
    assert model._result.params[-1] > 0


def test_glm_rejects_non_finite_fit(
    history: pl.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Result:
        params = np.array([np.nan, 1.0])

    class FakeNegativeBinomial:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def fit(self, **kwargs: object) -> Result:
            return Result()

    monkeypatch.setattr(md, "NegativeBinomial", FakeNegativeBinomial)
    model = md.NegBinGLM()
    with pytest.raises(RuntimeError, match="non-finite"):
        model.fit(history, spec.V2_DAILY)
    with pytest.raises(RuntimeError, match="call fit"):
        model.predict(history, dt.date(2024, 5, 1), spec.V2_DAILY)


@pytest.mark.parametrize("name", ["lgbm_poisson", "nb_glm", "lgbm_l1"])
def test_errors(history: pl.DataFrame, name: str) -> None:
    with pytest.raises(RuntimeError, match="call fit"):
        by_name(name).predict(history, dt.date(2024, 5, 1), spec.V2_DAILY)
    with pytest.raises(ValueError, match="no training rows"):
        by_name(name).fit(cut(history, FIRST + dt.timedelta(days=10)), spec.V2_DAILY)
    seven = history.with_columns(
        DIVISION=pl.when(pl.col("DIVISION") == 2).then(7).otherwise("DIVISION")
    )
    with pytest.raises(ValueError, match=r"divisions \[7\]"):
        by_name(name).fit(seven, spec.V2_DAILY)
    fitted = by_name(name).fit(history, spec.V2_DAILY)
    with pytest.raises(ValueError, match=r"divisions \[7\]"):
        fitted.predict(seven, dt.date(2024, 5, 1), spec.V2_DAILY)


def test_unsupported_estimator() -> None:
    with pytest.raises(TypeError, match="PoissonRegressor"):
        md.BoostedForecaster("x", PoissonRegressor())  # pyright: ignore[reportArgumentType]


def test_level_mode_errors() -> None:
    with pytest.raises(ValueError, match="log-link"):
        md.BoostedForecaster("x", LGBMRegressor(objective="l1"))
    with pytest.raises(ValueError, match="level must be"):
        md.BoostedForecaster("x", LGBMRegressor(), level="log")  # pyright: ignore[reportArgumentType]


END_SHIFT = dt.date(2024, 8, 31)


def legacy_predict(
    estimator: md.Booster,
    train: pl.DataFrame,
    issue: dt.date,
    setting: ForecastSpec,
    *,
    offset: bool,
) -> np.ndarray:
    # BoostedForecaster.fit and predict as they were before the level modes
    # (commit 43404a4), with DIVISION levels 1-6 only.
    def frame(rows: pl.DataFrame, *, categorical: bool) -> pd.DataFrame:
        out = rows.select(
            "DIVISION", pl.col(ft.FEATURE_COLUMNS[1:]).cast(pl.Float64)
        ).to_pandas()
        if categorical:
            out["DIVISION"] = pd.Categorical(out["DIVISION"], categories=md.DIVISIONS)
        return out

    rows = md.training_rows(train, setting)
    y = rows["y"].to_numpy()
    margin = md.log_offset(rows, setting) if offset else None
    model = cast(md.Booster, clone(estimator))
    match model:
        case CatBoostRegressor():
            model.fit(
                Pool(
                    frame(rows, categorical=False),
                    y,
                    cat_features=["DIVISION"],
                    baseline=margin,
                )
            )
        case LGBMRegressor():
            model.fit(frame(rows, categorical=True), y, init_score=margin)
        case _:
            model.fit(frame(rows, categorical=True), y, base_margin=margin)
    rows = md.issue_rows(train, issue, setting)
    if offset:
        rows = rows.filter(pl.col("mean_28d") > 0)
    margin = md.log_offset(rows, setting) if offset else None
    match model:
        case CatBoostRegressor():
            pool = Pool(
                frame(rows, categorical=False),
                cat_features=["DIVISION"],
                baseline=margin,
            )
            return np.asarray(model.predict(pool, prediction_type="Exponent"))
        case LGBMRegressor():
            raw = model.predict(frame(rows, categorical=True), raw_score=True)
            return np.exp(np.asarray(raw) + (0.0 if margin is None else margin))
        case _:
            return np.asarray(
                model.predict(frame(rows, categorical=True), base_margin=margin)
            )


@pytest.mark.parametrize("name", BOOSTED_NAMES)
@pytest.mark.parametrize("setting", [spec.V2_DAILY, spec.WEEKLY], ids=lambda s: s.key)
def test_existing_boosters_bit_identical(
    history: pl.DataFrame, name: str, setting: ForecastSpec
) -> None:
    # The full default configuration, against the pre-change implementation.
    model = next(m for m in md.default_models() if m.name == name)
    assert isinstance(model, md.BoostedForecaster)
    issue = dt.date(2024, 5, 10)
    train = cut(history, setting.cutoff(issue))
    got = model.fit(train, setting).predict(train, issue, setting)["y_pred"]
    expected = legacy_predict(
        model.estimator, train, issue, setting, offset=model.level == "offset"
    )
    np.testing.assert_array_equal(got.to_numpy(), expected)


def test_ratio_weights_give_mae_on_counts() -> None:
    # L * |y / L - p| == |y - L * p| for L > 0, so a weighted L1 fit of the
    # ratio is an L1 fit of the counts.
    rng = np.random.default_rng(0)
    y = rng.integers(0, 200, 1000).astype(float)
    level = rng.uniform(5, 500, 1000)
    p = rng.uniform(0.2, 3.0, 1000)
    np.testing.assert_allclose(level * np.abs(y / level - p), np.abs(y - level * p))


def test_ratio_mode_fits_ratio_with_level_weights(history: pl.DataFrame) -> None:
    model = by_name("lgbm_l1")
    assert isinstance(model, md.BoostedForecaster)
    issue = dt.date(2024, 5, 10)
    train = cut(history, spec.V2_DAILY.cutoff(issue))
    got = model.fit(train, spec.V2_DAILY).predict(train, issue, spec.V2_DAILY)
    rows = md.training_rows(train, spec.V2_DAILY)
    level = md.plain_level(rows, spec.V2_DAILY)
    reference = cast(LGBMRegressor, clone(model.estimator)).fit(
        md.BoostedForecaster._frame(rows, categorical=True),
        rows["y"].to_numpy() / level,
        sample_weight=level,
    )
    issue_row = md.issue_rows(train, issue, spec.V2_DAILY)
    ratio = np.asarray(
        reference.predict(md.BoostedForecaster._frame(issue_row, categorical=True))
    )
    expected = ratio * md.plain_level(issue_row, spec.V2_DAILY)
    np.testing.assert_array_equal(got["y_pred"].to_numpy(), expected)


def test_l1_raw_predicts_on_count_scale(history: pl.DataFrame) -> None:
    # Identity link: predictions are the model output, not its exponential.
    model = by_name("lgbm_l1_raw").fit(history, spec.V2_DAILY)
    got = model.predict(history, dt.date(2024, 5, 10), spec.V2_DAILY)["y_pred"]
    assert got.is_between(20, 80).all()


@pytest.mark.parametrize(
    ("estimator", "expected"),
    [
        (LGBMRegressor(objective="tweedie"), True),
        (LGBMRegressor(objective="l1"), False),
        (LGBMRegressor(), False),
        (XGBRegressor(objective="count:poisson"), True),
        (XGBRegressor(objective="reg:absoluteerror"), False),
        (CatBoostRegressor(loss_function="Tweedie:variance_power=1.5"), True),
        (CatBoostRegressor(loss_function="MAE"), False),
    ],
)
def test_log_link(estimator: md.Booster, *, expected: bool) -> None:
    assert md._log_link(estimator) is expected


@pytest.mark.parametrize(
    "estimator",
    [
        XGBRegressor(
            objective="reg:absoluteerror", n_estimators=20, enable_categorical=True
        ),
        CatBoostRegressor(
            loss_function="MAE",
            iterations=20,
            logging_level="Silent",
            allow_writing_files=False,
        ),
    ],
    ids=["xgboost", "catboost"],
)
def test_ratio_mode_other_libraries(
    history: pl.DataFrame, estimator: md.Booster
) -> None:
    # Identity link: the ratio prediction (near 1) times the level is on the
    # count scale (uniform 20-79); an exponent would put it near e^50.
    model = md.BoostedForecaster("x", estimator, level="ratio").fit(
        history, spec.V2_DAILY
    )
    got = model.predict(history, dt.date(2024, 5, 10), spec.V2_DAILY)["y_pred"]
    assert got.is_between(20, 80).all()


@pytest.mark.parametrize("name", STATE_SPACE_NAMES)
@pytest.mark.parametrize(
    ("setting", "rtol"),
    [(spec.V2_DAILY, 0.1), (spec.WEEKLY, 0.15)],
    ids=["v2", "weekly"],
)
def test_state_space_follows_new_level_without_refit(
    name: str, setting: ForecastSpec, rtol: float
) -> None:
    # Fitted on a level that moved twice, then given two months at a new level
    # it never saw: re-running the filter carries the forecast to it. The
    # stationary AR reverts part of the way to its mean, hence the tolerance.
    levels = [(FIRST, 40), (dt.date(2023, 7, 1), 60), (dt.date(2024, 1, 1), 45)]
    train = pattern_history(levels, dt.date(2024, 6, 30))
    shifted = pattern_history([*levels, (dt.date(2024, 7, 1), 90)], END_SHIFT)
    model = by_name(name).fit(train, setting)
    issue = END_SHIFT + dt.timedelta(days=setting.publication_lag_days)
    expected = sum(90 * WEEKLY_EFFECT[d.weekday()] for d in setting.target_days(issue))
    before = model.predict(train, dt.date(2024, 7, 2), setting)["y_pred"][0]
    got = model.predict(shifted, issue, setting)["y_pred"][0]
    assert before < 0.6 * expected
    np.testing.assert_allclose(got, expected, rtol=rtol)


class Probe(md._DivisionStateSpace):
    # Forecast of step k (1-based) is log(k), so a window total names its steps.
    def __init__(self) -> None:
        super().__init__("probe", min_days=1)
        self.steps: list[int] = []

    def _fit_division(self, dates: pl.Series, y: np.ndarray) -> None:
        return None

    def _forecast_division(
        self, fitted: object, dates: pl.Series, y: np.ndarray, steps: int
    ) -> np.ndarray:
        self.steps.append(steps)
        return np.log(np.arange(1, steps + 1, dtype=float))


@pytest.mark.parametrize(
    ("setting", "steps"),
    [(spec.V2_DAILY, 3), (spec.WEEKLY, 9), (spec.ForecastSpec(lead_days=5), 7)],
    ids=lambda v: v.key if isinstance(v, ForecastSpec) else str(v),
)
def test_state_space_horizon(
    history: pl.DataFrame, setting: ForecastSpec, steps: int
) -> None:
    # h = gap_days + window_days - 1 past the cutoff; the window sums the last
    # window_days steps. A missing cutoff day adds one step.
    issue = dt.date(2024, 5, 10)
    cutoff = setting.cutoff(issue)
    probe = Probe().fit(history, setting)
    got = probe.predict(history, issue, setting)["y_pred"].to_numpy()
    window = range(steps - setting.window_days + 1, steps + 1)
    np.testing.assert_allclose(got, [sum(window)] * 2)
    assert probe.steps == [steps, steps]
    gappy = cut(history, cutoff).with_columns(
        pl.when(pl.col("date") < cutoff).then(pl.col("n", "n_fr"))
    )
    probe.steps.clear()
    got = probe.predict(gappy, issue, setting)["y_pred"].to_numpy()
    np.testing.assert_allclose(got, [sum(w + 1 for w in window)] * 2)
    assert probe.steps == [steps + 1, steps + 1]


@pytest.mark.parametrize("name", STATE_SPACE_NAMES)
def test_state_space_missing_days(history: pl.DataFrame, name: str) -> None:
    # history has MISSING (2024-03-31) null in both divisions; add a null at
    # the cutoff and one a week before it.
    issue = dt.date(2024, 4, 9)
    cutoff = spec.V2_DAILY.cutoff(issue)
    gappy = cut(history, cutoff).with_columns(
        pl.when(~pl.col("date").is_in([cutoff, cutoff - dt.timedelta(days=7)])).then(
            pl.col("n", "n_fr")
        )
    )
    for setting in (spec.V2_DAILY, spec.WEEKLY):
        model = by_name(name).fit(gappy, setting)
        got = model.predict(gappy, issue, setting)["y_pred"].to_numpy()
        assert np.isfinite(got).all()
        assert (got > 0).all()


def test_ets_refilter_matches_fitted_forecast(history: pl.DataFrame) -> None:
    # On the fit history itself, smoothing with the stored parameters gives
    # the forecast of the fitted results; a start 3 days later (dropping the
    # 4 days that break the weekday alignment) changes it only slightly.
    issue = dt.date(2024, 9, 10)
    train = cut(history, spec.V2_DAILY.cutoff(issue)).filter(pl.col("date") > MISSING)
    model = md.ETSWeekly().fit(train, spec.V2_DAILY)
    got = model.predict(train, issue, spec.V2_DAILY)["y_pred"].to_numpy()
    expected = []
    for division in (1, 2):
        _, y = md._log_series(train, division)
        result: Any = md.ETSWeekly._model(y).fit(disp=False, maxiter=model.maxiter)
        expected.append(np.exp(np.asarray(result.forecast(3))[-1]))
    np.testing.assert_allclose(got, expected, rtol=1e-10)
    first: dt.date = train["date"].min()  # pyright: ignore[reportAssignmentType]
    later = train.filter(pl.col("date") > first + dt.timedelta(days=2))
    shifted = model.predict(later, issue, spec.V2_DAILY)["y_pred"].to_numpy()
    np.testing.assert_allclose(shifted, got, rtol=1e-3)


def test_ar_refilter_matches_fitted_forecast(history: pl.DataFrame) -> None:
    issue = dt.date(2024, 5, 10)
    train = cut(history, spec.V2_DAILY.cutoff(issue))
    model = md.ARCalendar().fit(train, spec.V2_DAILY)
    got = model.predict(train, issue, spec.V2_DAILY)["y_pred"].to_numpy()
    expected = []
    for division in (1, 2):
        dates, _ = md._log_series(train, division)
        result = model._fitted[division]
        future = pl.date_range(
            dates[-1] + dt.timedelta(days=1),
            dates[-1] + dt.timedelta(days=3),
            eager=True,
        )
        forecast = result.forecast(3, exog=md.calendar_exog(future))
        expected.append(np.exp(np.asarray(forecast)[-1]))
    np.testing.assert_allclose(got, expected, rtol=1e-10)


def test_calendar_exog_matches_features() -> None:
    dates = pl.date_range(dt.date(2024, 1, 1), dt.date(2024, 12, 31), "1d", eager=True)
    exog = md.calendar_exog(dates)
    assert exog.shape == (366, len(md.AR_COLUMNS))
    column = {c: exog[:, i] for i, c in enumerate(md.AR_COLUMNS)}
    rows = ft.add_calendar(pl.DataFrame({"target_start": dates}), spec.V2_DAILY)
    np.testing.assert_array_equal(column["holiday"], rows["holidays"].to_numpy())
    np.testing.assert_array_equal(
        column["holiday_adjacent"], rows["holiday_adjacent"].to_numpy()
    )
    np.testing.assert_allclose(column["doy_sin_1"], rows["target_doy_sin"].to_numpy())
    assert column["holiday"].sum() == len(ft._holiday_dates(2024, 2024))
    weekdays = exog[:, :6]
    assert (weekdays.sum(axis=1) == (dates.dt.weekday() > 1).to_numpy()).all()
    assert (column["const"] == 1).all()


def test_log_series_trims_and_skips_zeros() -> None:
    frame = pl.DataFrame(
        {
            "date": pl.date_range(
                FIRST, FIRST + dt.timedelta(days=5), "1d", eager=True
            ),
            "DIVISION": pl.Series([1] * 6, dtype=pl.Int64),
            "n": pl.Series([None, 4, 0, None, 8, None], dtype=pl.UInt32),
        }
    )
    dates, y = md._log_series(frame, 1)
    assert dates.to_list() == [FIRST + dt.timedelta(days=d) for d in (1, 2, 3, 4)]
    np.testing.assert_allclose(y, [np.log(4), np.nan, np.nan, np.log(8)])
    np.testing.assert_allclose(md._interpolate(y), np.linspace(np.log(4), np.log(8), 4))
    empty_dates, empty_y = md._log_series(frame.with_columns(n=None), 1)
    assert empty_dates.is_empty()
    assert empty_y.size == 0


@pytest.mark.parametrize("name", STATE_SPACE_NAMES)
def test_state_space_errors(history: pl.DataFrame, name: str) -> None:
    with pytest.raises(RuntimeError, match="call fit"):
        by_name(name).predict(history, dt.date(2024, 5, 1), spec.V2_DAILY)
    with pytest.raises(ValueError, match="no training rows"):
        by_name(name).fit(cut(history, FIRST + dt.timedelta(days=60)), spec.V2_DAILY)
    # A division unseen at fit, or with too little history, gets a null.
    fitted = by_name(name).fit(history.filter(pl.col("DIVISION") == 1), spec.V2_DAILY)
    got = fitted.predict(history, dt.date(2024, 5, 1), spec.V2_DAILY)
    assert got["DIVISION"].to_list() == [1, 2]
    assert got["y_pred"].is_null().to_list() == [False, True]


@pytest.mark.parametrize(("offset", "known"), [(0, True), (1, False)])
def test_ets_short_series_after_alignment(
    history: pl.DataFrame, offset: int, *, known: bool
) -> None:
    # 16 days of history: starting on the fitted start's weekday keeps all 16;
    # starting a day later drops 6, leaving fewer than the 14 ETSModel needs,
    # so the forecast is null.
    model = md.ETSWeekly().fit(history, spec.V2_DAILY)
    start = FIRST + dt.timedelta(days=7 * 70 + offset)
    cutoff = start + dt.timedelta(days=15)
    short = history.filter(pl.col("date").is_between(start, cutoff))
    issue = cutoff + dt.timedelta(days=spec.V2_DAILY.publication_lag_days)
    got = model.predict(short, issue, spec.V2_DAILY)["y_pred"]
    assert got.is_not_null().all() is known


@pytest.mark.parametrize(
    ("name", "target"), [("ets_weekly", "ETSModel"), ("ar_calendar", "SARIMAX")]
)
def test_state_space_rejects_non_finite_fit(
    history: pl.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    target: str,
) -> None:
    class Result:
        params = np.array([np.nan, 1.0])

    class FakeModel:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def fit(self, **kwargs: object) -> Result:
            return Result()

    monkeypatch.setattr(md, target, FakeModel)
    model = by_name(name)
    with pytest.raises(RuntimeError, match="non-finite"):
        model.fit(history, spec.V2_DAILY)
    with pytest.raises(RuntimeError, match="call fit"):
        model.predict(history, dt.date(2024, 5, 1), spec.V2_DAILY)


def test_citywide(history: pl.DataFrame) -> None:
    # Division 2 is also null on 2024-06-01: the citywide day is null there.
    one_null = dt.date(2024, 6, 1)
    gappy = history.with_columns(
        pl.when((pl.col("date") != one_null) | (pl.col("DIVISION") == 1)).then(
            pl.col("n", "n_fr")
        )
    )
    city = md.citywide(gappy)
    ft.check_history(city)
    assert city.schema == gappy.schema
    assert city["DIVISION"].unique().to_list() == [md.CITYWIDE]
    assert city["date"].to_list() == gappy["date"].unique().sort().to_list()
    assert city.filter(pl.col("n").is_null())["date"].to_list() == [MISSING, one_null]
    sums = gappy.group_by("date").agg(pl.col("n", "n_fr").sum()).sort("date")
    known = city.filter(pl.col("n").is_not_null())
    expected = sums.join(known.select("date"), on="date", how="semi")
    assert known["n"].to_list() == expected["n"].to_list()
    assert known["n_fr"].to_list() == expected["n_fr"].to_list()


def test_division_shares(history: pl.DataFrame) -> None:
    cutoff = dt.date(2024, 4, 10)  # the window holds MISSING, a null citywide day
    shares = md.division_shares(history, cutoff, 28)
    assert shares["DIVISION"].to_list() == [1, 2]
    assert shares["share"].sum() == pytest.approx(1.0)
    window = history.filter(
        pl.col("date").is_between(cutoff - dt.timedelta(days=27), cutoff)
    )
    totals = window.group_by("DIVISION").agg(pl.col("n").sum()).sort("DIVISION")
    np.testing.assert_allclose(
        shares["share"].to_numpy(), totals["n"].to_numpy() / totals["n"].sum()
    )
    poisoned = history.with_columns(
        pl.when(pl.col("date") > cutoff)
        .then(pl.lit(10_000, dtype=pl.UInt32))
        .otherwise(pl.col("n"))
        .alias("n")
    )
    assert_frame_equal(md.division_shares(poisoned, cutoff, 28), shares)
    # 9 more missing days in division 1, plus MISSING, leave 18 complete days.
    sparse = history.with_columns(
        pl.when(
            (pl.col("DIVISION") == 1)
            & pl.col("date").is_between(cutoff - dt.timedelta(days=8), cutoff)
        )
        .then(None)
        .otherwise(pl.col("n"))
        .alias("n")
    )
    assert md.division_shares(sparse, cutoff, 28)["share"].is_null().all()


@pytest.mark.parametrize("name", ["topdown_lgbm_poisson_raw", "topdown_ets"])
def test_topdown_splits_citywide_forecast(history: pl.DataFrame, name: str) -> None:
    issue = dt.date(2024, 5, 10)
    train = cut(history, spec.V2_DAILY.cutoff(issue))
    model = by_name(name)
    assert isinstance(model, md.TopDown)
    model.fit(train, spec.V2_DAILY)
    got = model.predict(train, issue, spec.V2_DAILY)
    city = (
        copy.deepcopy(model.base)
        .fit(md.citywide(train), spec.V2_DAILY)
        .predict(md.citywide(train), issue, spec.V2_DAILY)
    )
    assert city["DIVISION"].to_list() == [md.CITYWIDE]
    assert got["DIVISION"].to_list() == [1, 2]
    assert got["y_pred"].sum() == pytest.approx(city["y_pred"][0], rel=1e-12)
    shares = md.division_shares(train, spec.V2_DAILY.cutoff(issue))["share"]
    np.testing.assert_allclose(
        got["y_pred"].to_numpy(), shares.to_numpy() * city["y_pred"][0]
    )
    # The template stays unfitted.
    base = model.base
    if isinstance(base, md.BoostedForecaster):
        assert base._model is None
    else:
        assert isinstance(base, md.ETSWeekly)
        assert base._fitted == {}


def test_topdown_errors_and_null_share(history: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="share_days"):
        md.TopDown("x", md.ETSWeekly(), share_days=0)
    model = md.TopDown("x", md.ETSWeekly())
    with pytest.raises(RuntimeError, match="call fit"):
        model.predict(history, dt.date(2024, 5, 1), spec.V2_DAILY)
    issue = dt.date(2024, 5, 10)
    cutoff = spec.V2_DAILY.cutoff(issue)
    model.fit(cut(history, cutoff), spec.V2_DAILY)
    sparse = cut(history, cutoff).with_columns(
        pl.when(
            (pl.col("DIVISION") == 1)
            & pl.col("date").is_between(
                cutoff - dt.timedelta(days=10), cutoff - dt.timedelta(days=1)
            )
        )
        .then(None)
        .otherwise(pl.col(c))
        .alias(c)
        for c in ("n", "n_fr")
    )
    got = model.predict(sparse, issue, spec.V2_DAILY)
    assert got["DIVISION"].to_list() == [1, 2]
    assert got["y_pred"].is_null().all()


def test_timed_adds_up(history: pl.DataFrame) -> None:
    timed = md.Timed(by_name("lgbm_poisson"))
    assert timed.name == "lgbm_poisson"
    timed.fit(history, spec.V2_DAILY).predict(history, START, spec.V2_DAILY)
    assert timed.fit_seconds > 0
    assert timed.predict_seconds > 0


def test_rounded() -> None:
    got = md.rounded({"a": [1.23456, (2.0, float("nan"))], "b": "x", "c": 3})
    assert got == {"a": [1.235, [2.0, None]], "b": "x", "c": 3}


@pytest.fixture
def parquet_inputs(tmp_path: Path, history: pl.DataFrame) -> tuple[Path, Path]:
    history_path = tmp_path / "division_day.parquet"
    anomalies_path = tmp_path / "anomaly_days.parquet"
    history.write_parquet(history_path)
    history.select(
        "date", "DIVISION", anomaly=pl.col("date") == dt.date(2024, 2, 14)
    ).write_parquet(anomalies_path)
    return history_path, anomalies_path


def test_cli_validate(
    tmp_path: Path,
    parquet_inputs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models = [by_name("lgbm_poisson"), by_name("nb_glm"), by_name("topdown_ets")]
    monkeypatch.setattr(md, "default_models", lambda: models)
    history_path, anomalies_path = parquet_inputs
    out = tmp_path / "report.json"
    predictions = tmp_path / "predictions.parquet"
    result = runner.invoke(
        md.app,
        [
            "validate",
            "--history",
            str(history_path),
            "--anomalies",
            str(anomalies_path),
            "--out",
            str(out),
            "--predictions",
            str(predictions),
            "--start",
            "2024-02-01",
            "--end",
            "2024-02-29",
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(out.read_text())
    names = {"lgbm_poisson", "nb_glm", "topdown_ets", "same_wd_13w", "plain_28d"}
    assert report["period"] == {"start": "2024-02-01", "end": "2024-02-29"}
    assert report["spec"] == "lag2_lead1_win1"
    assert {row["model"] for row in report["overall"]} == names
    n_all = {row["model"]: row["n"] for row in report["overall"]}
    n_clean = {row["model"]: row["n"] for row in report["overall_without_anomalies"]}
    assert all(n_clean[m] == n_all[m] - 2 for m in names)
    assert {row["DIVISION"] for row in report["by_division"]} == {1, 2}
    assert {row["month"] for row in report["by_month"]} == {2}
    assert {(row["model"], row["year"]) for row in report["by_year"]} == {
        (m, 2024) for m in names
    }
    assert {"mae", "bias", "n"} <= set(report["by_year"][0])
    by_year_division = report["by_year_division"]
    assert len(by_year_division) == len(names) * 2
    assert set(by_year_division[0]) == {
        "model",
        "year",
        "DIVISION",
        "bias",
        "median_error",
    }
    columns = {"model", "n", "mae", "bias", "median_error", "under_share"}
    for key in ("overall", "overall_without_anomalies"):
        assert set(report[key][0]) == columns
    for key, by in (("by_division", "DIVISION"), ("by_month", "month")):
        assert set(report[key][0]) == columns | {by}
    assert set(report["by_year"][0]) == columns | {"year"}
    shares = [row["under_share"] for row in report["overall"]]
    assert all(0 <= share <= 1 for share in shares)
    assert set(report["bootstrap"]) == {"same_wd_13w", "plain_28d"}
    against = {b["model"] for b in report["bootstrap"]["same_wd_13w"]}
    assert against == names - {"same_wd_13w"}
    first = report["bootstrap"]["plain_28d"][0]
    assert first["ci_low"] <= first["diff"] <= first["ci_high"]
    assert set(report["seconds"]) == names
    assert set(report["versions"]) == set(md.LIBRARIES)
    saved = pl.read_parquet(predictions)
    assert saved.height == len(names) * 29 * 2
    assert "anomaly" in saved.columns
    assert "lgbm_poisson" in result.output
    assert "wrote" in result.output


def test_cli_missing_input(tmp_path: Path) -> None:
    result = runner.invoke(
        md.app, ["validate", "--history", str(tmp_path / "nope.parquet")]
    )
    assert result.exit_code == 1
    assert "not found" in result.output
