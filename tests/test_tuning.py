import dataclasses
import datetime as dt
import functools
import json
from pathlib import Path
from typing import Any, Self

import numpy as np
import optuna
import polars as pl
import pytest
from typer.testing import CliRunner

from ycit440_sim_forecast import ensemble as en
from ycit440_sim_forecast import evaluate as ev
from ycit440_sim_forecast import tuning as tu
from ycit440_sim_forecast.baseline import PlainMean
from ycit440_sim_forecast.manifest import sha256_file
from ycit440_sim_forecast.models import ENSEMBLE_CANDIDATES, REFERENCES
from ycit440_sim_forecast.spec import V2_DAILY, ForecastSpec

runner = CliRunner()
FIRST = dt.date(2023, 1, 1)
LAST = dt.date(2024, 6, 30)
# Small periods on synthetic data: tune on Jan-Feb 2024, score Mar-Apr 2024.
CONFIG = tu.TuningConfig(
    candidates=("lgbm_poisson_raw",),
    tune_start=dt.date(2024, 1, 1),
    tune_end=dt.date(2024, 2, 29),
    refit_start=dt.date(2024, 1, 1),
    score_start=dt.date(2024, 3, 1),
    score_end=dt.date(2024, 4, 30),
    n_trials=2,
    n_boot=50,
)


def make_history(seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pl.date_range(FIRST, LAST, "1d", eager=True)
    n = rng.integers(20, 80, size=2 * len(dates))
    return (
        pl.DataFrame({"date": dates})
        .join(pl.DataFrame({"DIVISION": [1, 2]}), how="cross")
        .with_columns(
            n=pl.Series(n, dtype=pl.UInt32),
            n_fr=pl.Series(rng.integers(0, n + 1), dtype=pl.UInt32),
        )
        .sort("date", "DIVISION")
    )


def make_anomalies(history: pl.DataFrame) -> pl.DataFrame:
    return history.select(
        "date", "DIVISION", anomaly=pl.col("date") == dt.date(2024, 3, 15)
    )


@pytest.fixture(scope="module")
def history() -> pl.DataFrame:
    return make_history()


class Scaled:
    """Cheap stand-in for an untuned model: a plain mean times a factor."""

    def __init__(self, name: str, factor: float) -> None:
        self._name = name
        self.base = PlainMean()
        self.factor = factor

    @property
    def name(self) -> str:
        return self._name

    def fit(self, history: pl.DataFrame, spec: ForecastSpec) -> Self:
        self.base.fit(history, spec)
        return self

    def predict(
        self, history: pl.DataFrame, issue_date: dt.date, spec: ForecastSpec
    ) -> pl.DataFrame:
        pred = self.base.predict(history, issue_date, spec)
        return pred.with_columns(pl.col("y_pred") * self.factor)


def make_oof(history: pl.DataFrame, config: tu.TuningConfig) -> pl.DataFrame:
    names = list(dict.fromkeys((*ENSEMBLE_CANDIDATES, *REFERENCES)))
    models = [Scaled(name, 0.9 + 0.05 * i) for i, name in enumerate(names)]
    results = ev.rolling_origin(
        history, models, config.spec, config.refit_start, config.score_end
    )
    return (
        ev.flag_anomalies(results, make_anomalies(history))
        .select(en.OUTPUT_SCHEMA.names())
        .cast(en.OUTPUT_SCHEMA)
    )


# Config


def test_default_config_periods() -> None:
    config = tu.TuningConfig()
    assert config.candidates == ("lgbm_poisson_raw", "lgbm_poisson", "cat_tweedie")
    assert config.tune_end < config.score_start == en.SCORE_START
    assert config.score_end == ev.VALIDATION_END < ev.TEST_START
    assert config.n_trials == 40


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"tune_end": dt.date(2022, 1, 1)}, "leaks into the comparison"),
        ({"tune_end": dt.date(2022, 6, 1)}, "leaks into the comparison"),
        ({"score_end": ev.TEST_START}, "test period"),
        ({"score_end": dt.date(2025, 6, 30)}, "test period"),
        ({"candidates": ()}, "non-empty"),
        ({"candidates": ("cat_tweedie", "cat_tweedie")}, "unique"),
        ({"candidates": ("ets_weekly",)}, "not tunable"),
        ({"tune_start": dt.date(2022, 1, 1)}, "tune_start"),
        ({"score_start": dt.date(2025, 1, 1)}, "score_start"),
        ({"refit_start": dt.date(2021, 2, 1)}, "refit_start"),
        ({"n_trials": 0}, "n_trials"),
        ({"n_boot": 0}, "n_boot"),
    ],
)
def test_config_rejects(changes: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        dataclasses.replace(tu.TuningConfig(), **changes)


def test_config_is_frozen_and_keyword_only() -> None:
    config = tu.TuningConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.n_trials = 3  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(TypeError):
        tu.TuningConfig(V2_DAILY)  # pyright: ignore[reportCallIssue]


def test_config_id_and_round_trip() -> None:
    config = tu.TuningConfig()
    assert len(config.config_id) == 12
    assert config.config_id == tu.TuningConfig().config_id
    assert dataclasses.replace(config, n_trials=41).config_id != config.config_id
    assert tu.config_from_dict(config.to_dict()) == config
    assert tu.config_from_dict(CONFIG.to_dict()) == CONFIG


# Search spaces and the defaults


LGBM_FIXED = {
    "n_estimators": 250,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 20,
    "reg_lambda": 0.5,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
}
CAT_FIXED = {
    "iterations": 300,
    "learning_rate": 0.08,
    "depth": 6,
    "l2_leaf_reg": 2.0,
    "rsm": 0.9,
    "random_strength": 0.5,
}


@pytest.mark.parametrize(
    ("space", "params"), [(tu.lgbm_space, LGBM_FIXED), (tu.cat_space, CAT_FIXED)]
)
def test_space_with_fixed_trial(space: tu.SearchSpace, params: dict[str, Any]) -> None:
    trial = optuna.trial.FixedTrial(params)
    assert space(trial) == params
    assert space(optuna.trial.FixedTrial(params)) == space(trial)


@pytest.mark.parametrize("space", [tu.lgbm_space, tu.cat_space])
def test_space_draws_are_seeded(space: tu.SearchSpace) -> None:
    def draws(seed: int) -> list[dict[str, Any]]:
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=seed))
        return [space(study.ask()) for _ in range(3)]

    assert draws(42) == draws(42)
    assert draws(42) != draws(7)


def test_build_forecaster_settings() -> None:
    raw = tu.build_forecaster("lgbm_poisson_raw", LGBM_FIXED, seed=3, label="x")
    assert raw.name == "x"
    assert raw.level == "none"
    params = raw.estimator.get_params()
    assert params["objective"] == "poisson"
    assert params["subsample_freq"] == 1
    assert params["random_state"] == 3
    assert params["deterministic"]
    assert {k: params[k] for k in LGBM_FIXED} == LGBM_FIXED
    offset = tu.build_forecaster("lgbm_poisson", LGBM_FIXED)
    assert offset.level == "offset"
    assert offset.name == "lgbm_poisson"
    cat = tu.build_forecaster("cat_tweedie", CAT_FIXED)
    assert cat.level == "offset"
    cat_params = cat.estimator.get_params()
    assert cat_params["loss_function"] == "Tweedie:variance_power=1.5"
    assert not cat_params["allow_writing_files"]
    assert {k: cat_params[k] for k in CAT_FIXED} == CAT_FIXED


def test_build_forecaster_errors() -> None:
    with pytest.raises(ValueError, match="not tunable"):
        tu.build_forecaster("ets_weekly", {})
    with pytest.raises(ValueError, match="outside the search space"):
        tu.build_forecaster("cat_tweedie", {"max_depth": 3})
    with pytest.raises(ValueError, match="not tunable"):
        tu.default_trial_params("xgb_poisson")
    with pytest.raises(ValueError, match="not tunable"):
        tu.default_booster("xgb_poisson")


@pytest.mark.parametrize("name", tu.TUNABLE)
def test_trial0_reproduces_default_models(name: str) -> None:
    trial0 = tu.default_trial_params(name)
    # The enqueued values are valid draws of the search space.
    assert tu.SEARCH_SPACES[name](optuna.trial.FixedTrial(trial0)) == trial0
    built = tu.build_forecaster(name, trial0)
    default = tu.default_booster(name)
    assert built.level == default.level
    assert type(built.estimator) is type(default.estimator)
    ours = built.estimator.get_params()
    theirs = default.estimator.get_params()
    differ = {
        k: (ours.get(k), theirs.get(k))
        for k in ours.keys() | theirs.keys()
        if ours.get(k) != theirs.get(k)
    }
    if name == "cat_tweedie":
        # Set explicitly at CatBoost's own defaults; unset in default_models.
        assert differ == {
            "l2_leaf_reg": (3.0, None),
            "rsm": (1.0, None),
            "random_strength": (1.0, None),
        }
    else:
        # reg_lambda 0 is off the log scale; bagging stays off at subsample 1.
        assert differ == {
            "reg_lambda": (tu.LGBM_REG_LAMBDA_FLOOR, 0.0),
            "subsample_freq": (1, 0),
        }


# Objective


def test_month_folds_match_one_rolling_origin(history: pl.DataFrame) -> None:
    folds = tu.issue_month_folds(CONFIG.tune_start, CONFIG.tune_end, V2_DAILY)
    # The 2023-12-31 issue date forecasts 2024-01-01 and is its own month.
    assert folds == [
        (dt.date(2024, 1, 1), dt.date(2024, 1, 1)),
        (dt.date(2024, 1, 2), dt.date(2024, 2, 1)),
        (dt.date(2024, 2, 2), dt.date(2024, 2, 29)),
    ]
    model = tu.build_forecaster("lgbm_poisson_raw", {"n_estimators": 100})
    whole = ev.rolling_origin(
        history, [model], V2_DAILY, CONFIG.tune_start, CONFIG.tune_end
    )
    parts = pl.concat(
        [ev.rolling_origin(history, [model], V2_DAILY, s, e) for s, e in folds]
    )
    assert parts.equals(whole)
    expected = ev.score(whole)["mae"].item()
    assert tu.tuning_mae(history, model, CONFIG) == pytest.approx(expected)


class StubTrial:
    """Records reports and prunes after ``prune_at`` steps."""

    def __init__(self, prune_at: int) -> None:
        self.prune_at = prune_at
        self.reports: list[tuple[float, int]] = []

    def report(self, value: float, step: int) -> None:
        self.reports.append((value, step))

    def should_prune(self) -> bool:
        return len(self.reports) >= self.prune_at


def test_tuning_mae_reports_running_mae_and_prunes(history: pl.DataFrame) -> None:
    model = tu.build_forecaster("lgbm_poisson_raw", {"n_estimators": 100})
    full = StubTrial(prune_at=99)
    mae = tu.tuning_mae(history, model, CONFIG, full)  # pyright: ignore[reportArgumentType]
    assert [step for _, step in full.reports] == [0, 1, 2]
    assert full.reports[-1][0] == pytest.approx(mae)
    with pytest.raises(optuna.TrialPruned):
        tu.tuning_mae(history, model, CONFIG, StubTrial(prune_at=2))  # pyright: ignore[reportArgumentType]


def test_tuning_mae_needs_a_scored_window(history: pl.DataFrame) -> None:
    day = CONFIG.tune_start
    gap = history.with_columns(
        n=pl.when(pl.col("date") == day).then(None).otherwise(pl.col("n"))
    )
    model = tu.build_forecaster("lgbm_poisson_raw", {"n_estimators": 100})
    config = dataclasses.replace(CONFIG, tune_end=day)
    with pytest.raises(ValueError, match="no scored window"):
        tu.tuning_mae(gap, model, config)


def test_tune_two_trials(history: pl.DataFrame) -> None:
    result = tu.tune(history, "lgbm_poisson_raw", CONFIG)
    assert result.trials.schema == tu.TRIALS_SCHEMA
    assert result.trials["number"].to_list() == [0, 1]
    assert json.loads(result.trials["params"][0]) == tu.default_trial_params(
        "lgbm_poisson_raw"
    )
    assert result.n_complete + result.n_pruned == 2
    assert result.best_mae <= result.trial0_mae
    assert result.best_mae == pytest.approx(
        result.trials.filter(pl.col("state") == "COMPLETE")["value"].min()
    )
    assert set(result.best_params) == set(LGBM_FIXED)
    # reg_lambda 1e-3 vs 0 barely moves the default.
    assert result.trial0_mae == pytest.approx(result.default_mae, rel=0.02)
    again = tu.tune(history, "lgbm_poisson_raw", CONFIG)
    assert again.best_params == result.best_params
    assert again.best_mae == result.best_mae
    summary = result.to_dict()
    assert summary["mean_seconds_complete"] > 0
    with pytest.raises(ValueError, match="not tunable"):
        tu.tune(history, "ets_weekly", CONFIG)


# Comparison


def test_tuned_members_order() -> None:
    assert tu.tuned_members(tu.TuningConfig()) == (
        "lgbm_poisson_raw_tuned",
        "lgbm_poisson_tuned",
        "cat_tweedie_tuned",
        "ets_weekly",
        "topdown_ets",
    )
    assert tu.tuned_members(CONFIG)[:2] == ("lgbm_poisson_raw_tuned", "lgbm_poisson")


def test_ens_mean_tuned_is_mean_of_members(history: pl.DataFrame) -> None:
    oof = make_oof(history, CONFIG)
    tuned = tu.tuned_predictions(
        history,
        make_anomalies(history),
        {"lgbm_poisson_raw": {"n_estimators": 100}},
        CONFIG,
    )
    assert tuned.schema == en.OUTPUT_SCHEMA
    assert tuned["model"].unique().to_list() == ["lgbm_poisson_raw_tuned"]
    assert tuned["target_start"].min() == CONFIG.refit_start
    assert tuned["target_end"].max() == CONFIG.score_end
    report, results = tu.comparison(oof, tuned, CONFIG)
    members = tu.tuned_members(CONFIG)
    wide = results.pivot("model", index=list(en.KEYS), values="y_pred")
    expected = wide.select(pl.mean_horizontal(members)).to_series()
    np.testing.assert_allclose(wide[tu.TUNED_ENSEMBLE], expected)
    expected_current = wide.select(pl.mean_horizontal(ENSEMBLE_CANDIDATES)).to_series()
    np.testing.assert_allclose(wide[en.FINAL_MODEL], expected_current)
    outside = results.filter(
        (pl.col("target_start") < CONFIG.score_start)
        | (pl.col("target_end") > CONFIG.score_end)
    )
    assert outside.is_empty()
    models = {row["model"] for row in report["overall"]}
    assert {tu.TUNED_ENSEMBLE, en.FINAL_MODEL, *REFERENCES} <= models
    assert {"mae", "bias", "median_error", "under_share"} <= set(report["overall"][0])
    assert {row["year"] for row in report["by_year"]} == {2024}
    boots = report["bootstrap"]
    assert [b["reference"] for b in boots["ensemble"]] == [en.FINAL_MODEL, *REFERENCES]
    assert [(b["model"], b["reference"]) for b in boots["boosters"]] == [
        ("lgbm_poisson_raw_tuned", "lgbm_poisson_raw")
    ]
    with pytest.raises(ValueError, match="no tuned parameters"):
        tu.tuned_predictions(history, make_anomalies(history), {}, CONFIG)
    short = oof.filter(
        (pl.col("model") != "plain_28d") | (pl.col("issue_date") != dt.date(2024, 3, 9))
    )
    with pytest.raises(ValueError, match="do not cover"):
        tu.comparison(short, tuned, CONFIG)


# CLI


@pytest.fixture
def inputs(tmp_path: Path, history: pl.DataFrame) -> dict[str, Path]:
    paths = {
        "history": tmp_path / "history.parquet",
        "anomalies": tmp_path / "anomalies.parquet",
        "predictions": tmp_path / "oof.parquet",
    }
    history.write_parquet(paths["history"])
    make_anomalies(history).write_parquet(paths["anomalies"])
    make_oof(history, CONFIG).write_parquet(paths["predictions"])
    return paths


def outputs(tmp_path: Path, tag: str) -> dict[str, Path]:
    return {
        "out": tmp_path / tag / "tuning.json",
        "tuned": tmp_path / tag / "tuned.parquet",
        "trials": tmp_path / tag / "trials.parquet",
    }


def invoke_run(inputs: dict[str, Path], out: dict[str, Path], *extra: str) -> Any:
    return runner.invoke(
        tu.app,
        [
            "run",
            "--history",
            str(inputs["history"]),
            "--anomalies",
            str(inputs["anomalies"]),
            "--predictions",
            str(inputs["predictions"]),
            "--ensemble-report",
            str(inputs.get("ensemble", inputs["history"].with_name("none.json"))),
            "--out",
            str(out["out"]),
            "--tuned-predictions",
            str(out["tuned"]),
            "--trials-out",
            str(out["trials"]),
            "--trials",
            "2",
            *extra,
        ],
    )


def test_cli_run_compare_and_guards(
    tmp_path: Path, inputs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        tu,
        "TuningConfig",
        functools.partial(
            tu.TuningConfig,
            **{
                f.name: getattr(CONFIG, f.name)
                for f in dataclasses.fields(CONFIG)
                if f.name != "n_trials"
            },
        ),
    )
    first = outputs(tmp_path, "first")
    result = invoke_run(inputs, first)
    assert result.exit_code == 0, result.output
    assert "not cross-checked" in result.output
    report = json.loads(first["out"].read_text())
    assert report["config"] == CONFIG.to_dict()
    assert report["config_id"] == CONFIG.config_id
    assert report["input_sha256"]["predictions"] == sha256_file(inputs["predictions"])
    assert set(report["input_sha256"]) == {"history", "anomalies", "predictions"}
    tuned = report["tuning"]["lgbm_poisson_raw"]
    assert set(tuned["best_params"]) == set(LGBM_FIXED)
    assert tuned["n_complete"] + tuned["n_pruned"] == 2
    assert tuned["best_mae"] <= tuned["trial0_mae"]
    assert pl.read_parquet(first["trials"]).height == 2
    assert pl.read_parquet(first["tuned"]).schema == en.OUTPUT_SCHEMA
    assert tu.TUNED_ENSEMBLE in result.output
    ens_mae = {r["model"]: r["mae"] for r in report["comparison"]["overall"]}

    # Every output is guarded, and nothing is rewritten on refusal.
    before = first["out"].read_bytes()
    refused = invoke_run(inputs, first)
    assert refused.exit_code == 1
    assert "already exists" in refused.output
    assert first["out"].read_bytes() == before
    clash = {**outputs(tmp_path, "clash"), "tuned": inputs["predictions"]}
    refused = invoke_run(inputs, clash, "--force")
    assert refused.exit_code == 1
    assert "also an input" in refused.output
    same = outputs(tmp_path, "same")
    refused = invoke_run(inputs, {**same, "trials": same["tuned"]})
    assert refused.exit_code == 1
    assert "must differ" in refused.output

    # The ens_mean cross-check passes on a match and stops on a mismatch.
    ensemble = tmp_path / "ensemble.json"
    ensemble.write_text(json.dumps({"overall": [{"model": "ens_mean", "mae": -1.0}]}))
    mismatch = invoke_run({**inputs, "ensemble": ensemble}, outputs(tmp_path, "bad"))
    assert mismatch.exit_code == 1
    assert "differs" in mismatch.output
    assert not outputs(tmp_path, "bad")["out"].exists()
    ensemble.write_text(
        json.dumps({"overall": [{"model": "ens_mean", "mae": ens_mae["ens_mean"]}]})
    )
    forced = invoke_run({**inputs, "ensemble": ensemble}, first, "--force")
    assert forced.exit_code == 0, forced.output
    assert "matches" in forced.output
    assert json.loads(first["out"].read_text())["comparison"] == report["comparison"]

    # compare reuses the saved parameters and gives the same comparison.
    again = outputs(tmp_path, "again")
    compared = runner.invoke(
        tu.app,
        [
            "compare",
            "--tuning",
            str(first["out"]),
            "--history",
            str(inputs["history"]),
            "--anomalies",
            str(inputs["anomalies"]),
            "--predictions",
            str(inputs["predictions"]),
            "--ensemble-report",
            str(ensemble),
            "--out",
            str(again["out"]),
            "--tuned-predictions",
            str(again["tuned"]),
        ],
    )
    assert compared.exit_code == 0, compared.output
    redo = json.loads(again["out"].read_text())
    assert redo["comparison"] == report["comparison"]
    assert redo["tuning"] == json.loads(first["out"].read_text())["tuning"]
    assert redo["input_sha256"]["tuning"] == sha256_file(first["out"])


def test_cli_missing_input(tmp_path: Path) -> None:
    out = outputs(tmp_path, "x")
    result = invoke_run(
        {
            "history": tmp_path / "nope.parquet",
            "anomalies": tmp_path / "nope2.parquet",
            "predictions": tmp_path / "nope3.parquet",
        },
        out,
    )
    assert result.exit_code == 1
    assert "not found" in result.output
    assert not out["out"].exists()
