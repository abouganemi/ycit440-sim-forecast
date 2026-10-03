import datetime as dt
import json
from pathlib import Path
from typing import cast

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
NAMES = [
    "lgbm_poisson",
    "lgbm_tweedie",
    "lgbm_poisson_raw",
    "xgb_poisson",
    "xgb_tweedie",
    "cat_poisson",
    "cat_tweedie",
    "nb_glm",
]


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


def light(model: Forecaster) -> Forecaster:
    # Same configuration with 20 trees, so the suite stays fast.
    if isinstance(model, md.BoostedForecaster):
        trees = (
            {"iterations": 20}
            if isinstance(model.estimator, CatBoostRegressor)
            else {"n_estimators": 20}
        )
        estimator = cast(md.Booster, clone(model.estimator))
        estimator.set_params(**trees)
        return md.BoostedForecaster(model.name, estimator, offset=model.offset)
    return model


def light_models() -> list[Forecaster]:
    return [light(m) for m in md.default_models()]


def by_name(name: str) -> Forecaster:
    return next(m for m in light_models() if m.name == name)


@pytest.fixture(scope="module")
def history() -> pl.DataFrame:
    return make_history()


def cut(frame: pl.DataFrame, last: dt.date) -> pl.DataFrame:
    return frame.filter(pl.col("date") <= last)


def test_default_models_names_and_types() -> None:
    models = md.default_models()
    assert [m.name for m in models] == NAMES
    for m in models:
        assert isinstance(m, md.BoostedForecaster | md.NegBinGLM)
    raw = [m for m in models if isinstance(m, md.BoostedForecaster) and not m.offset]
    assert [m.name for m in raw] == ["lgbm_poisson_raw"]


def test_default_models_are_unfitted_and_independent() -> None:
    first, second = md.default_models(), md.default_models()
    for a, b in zip(first, second, strict=True):
        assert a is not b
        if isinstance(a, md.BoostedForecaster):
            assert a.estimator is not b.estimator  # pyright: ignore[reportAttributeAccessIssue]
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


@pytest.mark.parametrize("name", [n for n in NAMES if n != "lgbm_poisson_raw"][:-1])
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
    for name in ("lgbm_poisson", "lgbm_poisson_raw", "nb_glm"):
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


@pytest.mark.parametrize("name", ["lgbm_poisson", "nb_glm"])
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


def test_timed_adds_up(history: pl.DataFrame) -> None:
    timed = md.Timed(by_name("lgbm_poisson"))
    assert timed.name == "lgbm_poisson"
    timed.fit(history, spec.V2_DAILY).predict(history, START, spec.V2_DAILY)
    assert timed.fit_seconds > 0
    assert timed.predict_seconds > 0


def test_rounded() -> None:
    got = md._rounded({"a": [1.23456, (2.0, float("nan"))], "b": "x", "c": 3})
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
    models = [by_name("lgbm_poisson"), by_name("nb_glm")]
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
    names = {"lgbm_poisson", "nb_glm", "same_wd_13w", "plain_28d"}
    assert report["period"] == {"start": "2024-02-01", "end": "2024-02-29"}
    assert report["spec"] == "lag2_lead1_win1"
    assert {row["model"] for row in report["overall"]} == names
    n_all = {row["model"]: row["n"] for row in report["overall"]}
    n_clean = {row["model"]: row["n"] for row in report["overall_without_anomalies"]}
    assert all(n_clean[m] == n_all[m] - 2 for m in names)
    assert {row["DIVISION"] for row in report["by_division"]} == {1, 2}
    assert {row["month"] for row in report["by_month"]} == {2}
    assert set(report["bootstrap"]) == {"same_wd_13w", "plain_28d"}
    against = {b["model"] for b in report["bootstrap"]["same_wd_13w"]}
    assert against == names - {"same_wd_13w"}
    first = report["bootstrap"]["plain_28d"][0]
    assert first["ci_low"] <= first["diff"] <= first["ci_high"]
    assert set(report["seconds"]) == names
    assert set(report["versions"]) == set(md.LIBRARIES)
    saved = pl.read_parquet(predictions)
    assert saved.height == 4 * 29 * 2
    assert "anomaly" in saved.columns
    assert "lgbm_poisson" in result.output
    assert "wrote" in result.output


def test_cli_missing_input(tmp_path: Path) -> None:
    result = runner.invoke(
        md.app, ["validate", "--history", str(tmp_path / "nope.parquet")]
    )
    assert result.exit_code == 1
    assert "not found" in result.output
