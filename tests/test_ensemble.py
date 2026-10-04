import dataclasses
import datetime as dt
import json
import math
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Self, cast

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from typer.testing import CliRunner

from ycit440_sim_forecast import ensemble as en
from ycit440_sim_forecast import evaluate as ev
from ycit440_sim_forecast.baseline import PlainMean, SameWeekdayMean
from ycit440_sim_forecast.manifest import sha256_file
from ycit440_sim_forecast.models import ENSEMBLE_CANDIDATES, REFERENCES
from ycit440_sim_forecast.spec import V2_DAILY, WEEKLY, Forecaster, ForecastSpec

runner = CliRunner()
NAMES = ("a", "b", "c")
FIRST_ISSUE = dt.date(2020, 12, 31)
LAST_ISSUE = dt.date(2021, 6, 29)
SCORE_START = dt.date(2021, 3, 1)
SCORE_END = dt.date(2021, 6, 30)
CONFIG = en.EnsembleConfig(
    candidates=NAMES, score_start=SCORE_START, score_end=SCORE_END, n_boot=50
)


def make_oof(models: Sequence[str] = NAMES, seed: int = 0) -> pl.DataFrame:
    # Two divisions; each model is y_true plus its own noise and bias.
    rng = np.random.default_rng(seed)
    issues = pl.date_range(FIRST_ISSUE, LAST_ISSUE, "1d", eager=True)
    base = (
        pl.DataFrame({"issue_date": issues})
        .join(pl.DataFrame({"DIVISION": [1, 2]}), how="cross")
        .with_columns(
            spec=pl.lit(V2_DAILY.key),
            cutoff=pl.col("issue_date") - pl.duration(days=2),
            target_start=pl.col("issue_date") + pl.duration(days=1),
            target_end=pl.col("issue_date") + pl.duration(days=1),
            y_true=pl.Series(rng.integers(20, 80, 2 * len(issues)), dtype=pl.Float64),
            anomaly=pl.Series(rng.random(2 * len(issues)) < 0.05),
        )
    )
    frames = [
        base.with_columns(
            model=pl.lit(name),
            y_pred=pl.col("y_true")
            + (i - 1) * 2.0
            + pl.Series(rng.normal(0, 3 + i, base.height)),
        )
        for i, name in enumerate(models)
    ]
    return pl.concat(frames).select(en.OUTPUT_SCHEMA.names()).cast(en.OUTPUT_SCHEMA)


@pytest.fixture
def oof() -> pl.DataFrame:
    return make_oof()


# Combiners and the weight LP


def test_mean_and_median_hand_values() -> None:
    preds = np.array([[1.0, 2.0, 9.0], [4.0, 4.0, 1.0]])
    np.testing.assert_allclose(en.combine(preds, NAMES, "mean"), [4.0, 3.0])
    np.testing.assert_allclose(en.combine(preds, NAMES, "median"), [2.0, 4.0])
    weighted = en.combine(preds, NAMES, "weighted", {"a": 1.0, "b": 1.0, "c": 2.0})
    np.testing.assert_allclose(weighted, [5.25, 2.5])


def test_missing_member_forecast_propagates() -> None:
    preds = np.array([[1.0, np.nan, 3.0]])
    for method in ("mean", "median"):
        assert np.isnan(en.combine(preds, NAMES, method)).all()
    weights = {"a": 1.0, "b": 0.0, "c": 0.0}
    assert np.isnan(en.combine(preds, NAMES, "weighted", weights)).all()


def test_lp_recovers_convex_weights() -> None:
    rng = np.random.default_rng(1)
    preds = rng.uniform(10, 50, size=(300, 3))
    truth = np.array([0.5, 0.3, 0.2])
    weights = en.fit_weights(preds, preds @ truth, NAMES)
    assert list(weights) == list(NAMES)
    np.testing.assert_allclose(list(weights.values()), truth, atol=1e-7)
    vector = np.array(list(weights.values()))
    assert (vector >= 0).all()
    assert math.isclose(vector.sum(), 1.0)
    assert np.abs(preds @ vector - preds @ truth).mean() < 1e-6
    assert en.fit_weights(preds, preds @ truth, NAMES) == weights


def test_lp_is_deterministic_with_tied_members() -> None:
    rng = np.random.default_rng(2)
    column = rng.uniform(10, 50, size=(200, 1))
    preds = np.hstack([column, column, rng.uniform(10, 50, size=(200, 1))])
    y = column[:, 0] + rng.normal(0, 1, 200)
    first = en.fit_weights(preds, y, NAMES)
    assert all(en.fit_weights(preds, y, NAMES) == first for _ in range(3))
    assert math.isclose(sum(first.values()), 1.0)


def test_lp_beats_mean_and_every_member(oof: pl.DataFrame) -> None:
    wide = en.wide_predictions(oof, CONFIG).filter(pl.col("y_true").is_not_null())
    preds = wide.select(NAMES).to_numpy()
    y = wide["y_true"].to_numpy()
    weights = en.fit_weights(preds, y, NAMES)

    def mae(pred: np.ndarray) -> float:
        return float(np.abs(pred - y).mean())

    best = mae(en.combine(preds, NAMES, "weighted", weights))
    assert best <= mae(en.combine(preds, NAMES, "mean")) + 1e-9
    for i in range(len(NAMES)):
        assert best <= mae(preds[:, i]) + 1e-9


def test_lp_clips_solver_noise(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = SimpleNamespace(success=True, x=np.array([0.6, -1e-14, 0.4, 0.0, 0.0]))
    monkeypatch.setattr(en, "linprog", lambda *_a, **_k: fake)
    weights = en.fit_weights(np.ones((1, 3)), np.ones(1), NAMES)
    assert weights == {"a": 0.6, "b": 0.0, "c": 0.4}


def test_lp_failure_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = SimpleNamespace(success=False, message="infeasible")
    monkeypatch.setattr(en, "linprog", lambda *_a, **_k: fake)
    with pytest.raises(ValueError, match="linear program failed: infeasible"):
        en.fit_weights(np.ones((2, 3)), np.ones(2), NAMES)


@pytest.mark.parametrize(
    ("preds", "y", "match"),
    [
        (np.ones((2, 2)), np.ones(2), "do not line up"),
        (np.ones((2, 3)), np.ones(3), "do not line up"),
        (np.ones((0, 3)), np.ones(0), "no rows"),
        (np.array([[1.0, np.nan, 1.0]]), np.ones(1), "finite"),
        (np.ones((1, 3)), np.array([np.inf]), "finite"),
    ],
)
def test_fit_weights_rejects(preds: np.ndarray, y: np.ndarray, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        en.fit_weights(preds, y, NAMES)


@pytest.mark.parametrize(
    ("weights", "match"),
    [
        ({"a": 1.0, "b": 1.0}, r"missing \['c'\]"),
        ({"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0}, r"extra \['d'\]"),
        ({"a": 1.0, "b": -0.1, "c": 1.0}, ">= 0"),
        ({"a": 1.0, "b": math.nan, "c": 1.0}, "finite"),
        ({"a": 1.0, "b": math.inf, "c": 1.0}, "finite"),
        ({"a": 0.0, "b": 0.0, "c": 0.0}, "all zero"),
    ],
)
def test_weight_vector_rejects(weights: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        en.weight_vector(NAMES, weights)


def test_weight_vector_matches_by_name_not_position() -> None:
    vector = en.weight_vector(NAMES, {"c": 2.0, "a": 1.0, "b": 1.0})
    np.testing.assert_allclose(vector, [0.25, 0.25, 0.5])


def test_combine_method_weight_mismatch() -> None:
    preds = np.ones((1, 3))
    with pytest.raises(ValueError, match="needs weights"):
        en.combine(preds, NAMES, "weighted")
    with pytest.raises(ValueError, match="only used by 'weighted'"):
        en.combine(preds, NAMES, "mean", {"a": 1.0, "b": 1.0, "c": 1.0})
    with pytest.raises(ValueError, match="unknown method"):
        en.combine(preds, NAMES, cast(Any, "mode"))


# Configuration


def test_config_id_is_stable_and_sensitive() -> None:
    same = en.EnsembleConfig(
        candidates=NAMES, score_start=SCORE_START, score_end=SCORE_END, n_boot=50
    )
    assert same.config_id == CONFIG.config_id
    assert len(CONFIG.config_id) == 12
    assert en.EnsembleConfig().config_id != CONFIG.config_id
    assert dataclasses.replace(CONFIG, seed=1).config_id != CONFIG.config_id
    assert dataclasses.replace(CONFIG, spec=WEEKLY).config_id != CONFIG.config_id


def test_config_to_dict() -> None:
    assert en.EnsembleConfig().to_dict() == {
        "spec": {
            "key": "lag2_lead1_win1",
            "publication_lag_days": 2,
            "lead_days": 1,
            "window_days": 1,
        },
        "candidates": list(ENSEMBLE_CANDIDATES),
        "methods": ["mean", "median", "weighted"],
        "score_start": "2022-01-01",
        "score_end": "2024-12-31",
        "n_boot": 2000,
        "seed": 42,
    }


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"candidates": ()}, "non-empty and unique"),
        ({"candidates": ("a", "a")}, "non-empty and unique"),
        ({"candidates": ("a", "y_true")}, "clash"),
        ({"methods": ()}, "methods"),
        ({"methods": ("mean", "mean")}, "methods"),
        ({"methods": ("mean", "mode")}, "methods"),
        (
            {"score_start": dt.date(2022, 1, 2), "score_end": dt.date(2022, 1, 1)},
            "after",
        ),
        ({"score_end": ev.TEST_START}, "test period"),
        ({"n_boot": 0}, "n_boot"),
    ],
)
def test_config_rejects(changes: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        en.EnsembleConfig(**changes)


# Out-of-fold combination


def test_wide_predictions_layout(oof: pl.DataFrame) -> None:
    wide = en.wide_predictions(oof, CONFIG)
    assert wide.columns == [*en._META, *NAMES]
    assert wide.height == oof.height // len(NAMES)
    assert wide.select(en.KEYS).equals(wide.select(en.KEYS).sort(en.KEYS))


def test_wide_rejects_spec_mismatch(oof: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="spec"):
        en.wide_predictions(oof, dataclasses.replace(CONFIG, spec=WEEKLY))


def test_wide_rejects_missing_candidate(oof: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match=r"missing from the predictions: \['c'\]"):
        en.wide_predictions(oof.filter(pl.col("model") != "c"), CONFIG)


def test_wide_rejects_misaligned_keys(oof: pl.DataFrame) -> None:
    dropped = oof.filter(
        ~((pl.col("model") == "b") & (pl.col("issue_date") == FIRST_ISSUE))
    )
    with pytest.raises(ValueError, match="different"):
        en.wide_predictions(dropped, CONFIG)


def test_wide_rejects_disagreeing_truth(oof: pl.DataFrame) -> None:
    changed = oof.with_columns(
        y_true=pl.when(pl.col("model") == "c")
        .then(pl.col("y_true") + 1)
        .otherwise("y_true")
    )
    with pytest.raises(ValueError, match="disagree on y_true"):
        en.wide_predictions(changed, CONFIG)


def test_wide_rejects_repeated_key(oof: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="repeats"):
        en.wide_predictions(pl.concat([oof, oof]), CONFIG)


def test_wide_rejects_null_forecast_on_known_truth(oof: pl.DataFrame) -> None:
    nulled = oof.with_columns(
        y_pred=pl.when((pl.col("model") == "b") & (pl.col("issue_date") == FIRST_ISSUE))
        .then(None)
        .otherwise("y_pred")
    )
    with pytest.raises(ValueError, match="b has null forecasts"):
        en.wide_predictions(nulled, CONFIG)


def test_rolling_weights_refit_monthly_on_known_rows(oof: pl.DataFrame) -> None:
    wide = en.wide_predictions(oof, CONFIG)
    fits = en.rolling_weights(wide, CONFIG)
    assert [f.month for f in fits] == [
        "2021-02",
        "2021-03",
        "2021-04",
        "2021-05",
        "2021-06",
    ]
    known = wide.filter(pl.col("y_true").is_not_null())
    for fit in fits:
        assert fit.issue_date.day == 1
        assert fit.cutoff == V2_DAILY.cutoff(fit.issue_date)
        assert fit.last_target <= fit.cutoff
        assert fit.n_rows == known.filter(pl.col("target_end") <= fit.cutoff).height
        assert math.isclose(sum(fit.weights.values()), 1.0)
    # The first scored month (target 2021-03-01, issued 2021-02-28) is refit on
    # 2021-02-01 with data through 2021-01-30.
    assert fits[0].issue_date == dt.date(2021, 2, 1)
    assert fits[0].last_target == dt.date(2021, 1, 30)
    assert fits[0].n_rows == 2 * 30


def test_rolling_weights_ignore_unknown_outcomes(oof: pl.DataFrame) -> None:
    fits = en.rolling_weights(en.wide_predictions(oof, CONFIG), CONFIG)
    rng = np.random.default_rng(9)
    for fit in fits:
        later = pl.col("target_end") > fit.cutoff
        poisoned = oof.with_columns(
            y_true=pl.when(later).then(pl.lit(1e6)).otherwise("y_true"),
            y_pred=pl.when(later)
            .then(pl.Series(rng.uniform(-1e6, 1e6, oof.height)))
            .otherwise("y_pred"),
        )
        again = en.rolling_weights(en.wide_predictions(poisoned, CONFIG), CONFIG)
        match = next(f for f in again if f.month == fit.month)
        assert match == fit


def test_rolling_weights_need_history(oof: pl.DataFrame) -> None:
    config = dataclasses.replace(CONFIG, score_start=dt.date(2021, 1, 1))
    with pytest.raises(ValueError, match="no outcome known by 2020-12-29"):
        en.rolling_weights(en.wide_predictions(oof, config), config)


def test_no_scored_rows(oof: pl.DataFrame) -> None:
    config = dataclasses.replace(
        CONFIG, score_start=dt.date(2022, 1, 1), score_end=dt.date(2022, 2, 1)
    )
    with pytest.raises(ValueError, match="no predictions between"):
        en.combine_oof(oof, config)


def test_combine_oof_rows(oof: pl.DataFrame) -> None:
    rows, fits = en.combine_oof(oof, CONFIG)
    assert rows.schema == en.OUTPUT_SCHEMA
    assert rows["model"].unique().sort().to_list() == [
        "ens_mean",
        "ens_median",
        "ens_weighted",
    ]
    assert rows["target_start"].min() == SCORE_START
    assert rows["target_end"].max() == SCORE_END
    assert rows.equals(rows.sort("model", *en.KEYS))
    wide = en.scored_rows(en.wide_predictions(oof, CONFIG), CONFIG)
    by_model = rows.partition_by("model", as_dict=True)
    preds = wide.select(NAMES).to_numpy()
    np.testing.assert_allclose(by_model[("ens_mean",)]["y_pred"], preds.mean(axis=1))
    np.testing.assert_allclose(
        by_model[("ens_median",)]["y_pred"], np.median(preds, axis=1)
    )
    weights = {f.month: np.array(list(f.weights.values())) for f in fits}
    expected = [
        float(p @ weights[m])
        for p, m in zip(preds, wide["month"].to_list(), strict=True)
    ]
    np.testing.assert_allclose(by_model[("ens_weighted",)]["y_pred"], expected)
    # The ensemble rows join back onto the parquet and feed score unchanged.
    table = ev.score(pl.concat([rows, oof]))
    assert set(table["model"]) == {*NAMES, "ens_mean", "ens_median", "ens_weighted"}


def test_combine_oof_without_weighted(oof: pl.DataFrame) -> None:
    rows, fits = en.combine_oof(oof, dataclasses.replace(CONFIG, methods=("median",)))
    assert fits == []
    assert rows["model"].unique().to_list() == ["ens_median"]
    assert en.weights_summary(fits) == {}


def test_combine_oof_keeps_unscored_truth_null() -> None:
    oof = make_oof().with_columns(
        y_true=pl.when(pl.col("issue_date") == dt.date(2021, 4, 1))
        .then(None)
        .otherwise("y_true")
    )
    rows, _ = en.combine_oof(oof, CONFIG)
    assert rows.filter(pl.col("y_true").is_null()).height == 3 * 2


# Report


def test_report_compares_on_same_rows(oof: pl.DataFrame) -> None:
    full = make_oof((*NAMES, *REFERENCES))
    report, rows = en.ensemble_report(full, CONFIG, "abc")
    assert report["config_id"] == CONFIG.config_id
    assert report["input_sha256"] == "abc"
    assert report["final_model"] == en.FINAL_MODEL == "ens_mean"
    assert report["period"] == {"start": "2021-03-01", "end": "2021-06-30"}
    counts = {row["model"]: row["n"] for row in report["overall"]}
    assert set(counts) == {
        *NAMES,
        *REFERENCES,
        "ens_mean",
        "ens_median",
        "ens_weighted",
    }
    assert len(set(counts.values())) == 1
    assert rows.height == 3 * next(iter(counts.values()))
    assert set(report["weights_by_month"]) == {f"2021-0{m}" for m in range(2, 7)}
    assert set(report["weights_summary"]) == set(NAMES)
    for stats in report["weights_summary"].values():
        assert stats["min"] <= stats["mean"] <= stats["max"]
    best = report["bootstrap"]["best_single"]
    assert best["model"] == en.best_single(full, NAMES)
    assert "hindsight" in best["selection"]
    assert [r["reference"] for r in best["results"]] == [best["model"]] * 3
    for reference in REFERENCES:
        comparisons = report["bootstrap"]["references"][reference]
        assert [r["model"] for r in comparisons] == [
            "ens_mean",
            "ens_median",
            "ens_weighted",
        ]
    assert {r["year"] for r in report["by_year"]} == {2021}


def test_report_rejects_missing_reference(oof: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="do not cover"):
        en.ensemble_report(oof, CONFIG, "abc")


def test_best_single_tie_goes_to_earlier_name(oof: pl.DataFrame) -> None:
    tied = pl.concat(
        [oof, oof.filter(pl.col("model") == "b").with_columns(model=pl.lit("a0"))]
    )
    assert (
        en.best_single(tied.filter(pl.col("model").is_in(["b", "a0"])), ["b", "a0"])
        == "a0"
    )


# Ensemble forecaster


class Counter:
    """Counts fits so a test can see whether an instance was touched."""

    def __init__(self, name: str, value: float) -> None:
        self._name = name
        self.value = value
        self.fits = 0

    @property
    def name(self) -> str:
        return self._name

    def fit(self, history: pl.DataFrame, spec: ForecastSpec) -> Self:
        self.fits += 1
        return self

    def predict(
        self, history: pl.DataFrame, issue_date: dt.date, spec: ForecastSpec
    ) -> pl.DataFrame:
        divisions = history["DIVISION"].unique().sort()
        return pl.DataFrame(
            {"DIVISION": divisions, "y_pred": [self.value] * len(divisions)}
        )


def make_history(seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pl.date_range(dt.date(2023, 1, 1), dt.date(2024, 6, 30), "1d", eager=True)
    return (
        pl.DataFrame({"date": dates})
        .join(pl.DataFrame({"DIVISION": [1, 2]}), how="cross")
        .with_columns(
            n=pl.Series(rng.integers(20, 80, size=2 * len(dates)), dtype=pl.Int64)
        )
        .sort("date", "DIVISION")
    )


def test_ensemble_does_not_mutate_members() -> None:
    members = [Counter("a", 1.0), Counter("b", 3.0)]
    ens = en.Ensemble("ens", members, "mean")
    ens.fit(make_history(), V2_DAILY)
    assert [m.fits for m in members] == [0, 0]
    assert [cast(Counter, m).fits for m in ens.members] == [1, 1]
    assert all(c is not o for c, o in zip(ens.members, members, strict=True))


def test_ensemble_predict_and_weights_by_name() -> None:
    members = [Counter("a", 1.0), Counter("b", 3.0)]
    ens = en.Ensemble("ens", members, "weighted", {"b": 3.0, "a": 1.0})
    assert ens.weights == {"a": 0.25, "b": 0.75}
    assert ens.name == "ens"
    pred = ens.predict(make_history(), dt.date(2024, 1, 1), V2_DAILY)
    assert pred.to_dict(as_series=False) == {"DIVISION": [1, 2], "y_pred": [2.5, 2.5]}


def test_ensemble_null_member_gives_null() -> None:
    ens = en.Ensemble("ens", [Counter("a", 1.0), Counter("b", math.nan)], "median")
    pred = ens.predict(make_history(), dt.date(2024, 1, 1), V2_DAILY)
    assert pred["y_pred"].null_count() == 2


class OneDivision(Counter):
    def predict(
        self, history: pl.DataFrame, issue_date: dt.date, spec: ForecastSpec
    ) -> pl.DataFrame:
        return pl.DataFrame({"DIVISION": [1], "y_pred": [self.value]})


def test_ensemble_rejects_different_divisions() -> None:
    ens = en.Ensemble("ens", [Counter("a", 1.0), OneDivision("b", 2.0)], "mean")
    with pytest.raises(ValueError, match="b on 2024-01-01 returned divisions"):
        ens.predict(make_history(), dt.date(2024, 1, 1), V2_DAILY)


@pytest.mark.parametrize(
    ("members", "method", "weights", "match"),
    [
        ([], "mean", None, "non-empty"),
        ([Counter("a", 1), Counter("a", 2)], "mean", None, "unique"),
        ([Counter("a", 1)], "mode", None, "unknown method"),
        ([Counter("a", 1)], "weighted", None, "required"),
        ([Counter("a", 1)], "mean", {"a": 1.0}, "required"),
        ([Counter("a", 1)], "weighted", {"b": 1.0}, "match the members by name"),
        ([Counter("a", 1)], "weighted", {"a": -1.0}, ">= 0"),
    ],
)
def test_ensemble_rejects(
    members: list[Forecaster],
    method: str,
    weights: dict[str, float] | None,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        en.Ensemble("ens", members, cast(Any, method), weights)


def test_ensemble_matches_combined_rolling_origin() -> None:
    history = make_history()
    members: list[Forecaster] = [SameWeekdayMean(4), SameWeekdayMean(8), PlainMean()]
    names = [m.name for m in members]
    weights = {names[0]: 0.5, names[1]: 0.2, names[2]: 0.3}
    ensembles = [
        en.Ensemble("ens_mean", members, "mean"),
        en.Ensemble("ens_median", members, "median"),
        en.Ensemble("ens_weighted", members, "weighted", weights),
    ]
    start, end = dt.date(2024, 2, 1), dt.date(2024, 4, 10)
    results = ev.rolling_origin(history, [*ensembles, *members], V2_DAILY, start, end)
    wide = (
        results.filter(pl.col("model").is_in(names))
        .pivot("model", index=list(en.KEYS), values="y_pred")
        .sort(en.KEYS)
    )
    preds = wide.select(names).to_numpy()
    for ens in ensembles:
        got = results.filter(pl.col("model") == ens.name).sort(en.KEYS)
        expected = en.combine(preds, names, ens.method, ens.weights)
        assert got.select(en.KEYS).equals(wide.select(en.KEYS))
        np.testing.assert_allclose(got["y_pred"].to_numpy(), expected, rtol=1e-12)


def test_from_candidates() -> None:
    config = en.EnsembleConfig()
    weights = dict.fromkeys(ENSEMBLE_CANDIDATES, 1.0)
    built = en.from_candidates(config, weights)
    assert [e.name for e in built] == ["ens_mean", "ens_median", "ens_weighted"]
    assert [m.name for m in built[0].members] == list(ENSEMBLE_CANDIDATES)
    assert built[0].members[0] is not built[1].members[0]
    assert built[2].weights == dict.fromkeys(ENSEMBLE_CANDIDATES, 0.2)
    assert built[0].weights is None
    with pytest.raises(ValueError, match="weights are required iff"):
        en.from_candidates(config)
    with pytest.raises(ValueError, match="weights are required iff"):
        en.from_candidates(dataclasses.replace(config, methods=("mean",)), weights)
    with pytest.raises(ValueError, match=r"unknown candidates \['nope'\]"):
        en.from_candidates(dataclasses.replace(config, candidates=("nope",)))


# CLI


def make_cli_parquet(path: Path) -> Path:
    oof = make_oof((*ENSEMBLE_CANDIDATES, *REFERENCES))
    # Shift the synthetic months so the default 2022+ period has rows: keep
    # one year of history before 2022.
    shift = pl.duration(days=(dt.date(2022, 1, 1) - SCORE_START).days)
    oof = oof.with_columns(
        pl.col("issue_date", "cutoff", "target_start", "target_end") + shift
    )
    oof.write_parquet(path)
    return path


def test_cli_validate(tmp_path: Path) -> None:
    predictions = make_cli_parquet(tmp_path / "oof.parquet")
    outputs: list[bytes] = []
    for run in range(2):
        out = tmp_path / f"report{run}.json"
        combined = tmp_path / f"ens{run}.parquet"
        result = runner.invoke(
            en.app,
            [
                "validate",
                "--predictions",
                str(predictions),
                "--out",
                str(out),
                "--ensemble-predictions",
                str(combined),
                "--end",
                "2022-04-30",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "ens_weighted" in result.output
        assert "vs same_wd_13w" in result.output
        assert "weight lgbm_poisson_raw" in result.output
        outputs.append(out.read_bytes())
        rows = pl.read_parquet(combined)
        assert rows.schema == en.OUTPUT_SCHEMA
        assert rows["target_end"].max() == dt.date(2022, 4, 30)
    assert outputs[0] == outputs[1]


def test_cli_missing_parquet(tmp_path: Path) -> None:
    result = runner.invoke(
        en.app, ["validate", "--predictions", str(tmp_path / "missing.parquet")]
    )
    assert result.exit_code == 1
    assert "ycit440_sim_forecast.models validate" in result.output


def test_rows_feed_paired_bootstrap(oof: pl.DataFrame) -> None:
    rows, _ = en.combine_oof(oof, CONFIG)
    both = pl.concat([rows, oof.select(en.OUTPUT_SCHEMA.names())])
    result = ev.paired_bootstrap(both, "ens_weighted", "a", n_boot=20)
    assert result["n_blocks"] > 0
    assert_frame_equal(rows, en.combine_oof(oof, CONFIG)[0])


# Horizons and WAPE


def test_wape_hand_values() -> None:
    rows = pl.DataFrame(
        {
            "model": ["a", "a", "a", "b", "b", "z"],
            "y_true": [10.0, 30.0, None, 10.0, 30.0, 0.0],
            "y_pred": [12.0, 27.0, 99.0, 10.0, 30.0, 1.0],
        }
    )
    got = en.wape(rows)
    assert list(got) == ["a", "b", "z"]
    assert got["a"] == pytest.approx(5 / 40)
    assert got["b"] == 0.0
    assert math.isnan(got["z"])


def test_report_wape_matches_mae_over_mean_actual() -> None:
    full = make_oof((*NAMES, *REFERENCES))
    report, rows = en.ensemble_report(full, CONFIG, "abc")
    mean_actual = rows.filter(pl.col("model") == "ens_mean")["y_true"].mean()
    assert isinstance(mean_actual, float)
    assert set(report["wape"]) == {row["model"] for row in report["overall"]}
    for row in report["overall"]:
        assert report["wape"][row["model"]] == pytest.approx(
            row["mae"] / mean_actual, abs=2e-3
        )


def test_cli_validate_rejects_other_horizon(tmp_path: Path) -> None:
    predictions = make_cli_parquet(tmp_path / "oof.parquet")
    result = runner.invoke(
        en.app,
        [
            "validate",
            "--predictions",
            str(predictions),
            "--out",
            str(tmp_path / "report.json"),
            "--horizon",
            "weekly",
        ],
    )
    assert result.exit_code == 1
    assert "holds predictions for spec ['lag2_lead1_win1']" in result.output
    assert "--horizon weekly is lag2_lead1_win7" in result.output
    assert not (tmp_path / "report.json").exists()


def test_cli_validate_rejects_bad_horizon(tmp_path: Path) -> None:
    result = runner.invoke(en.app, ["validate", "--horizon", "lead0"])
    assert result.exit_code == 2
    assert "unknown horizon" in result.output


def test_cli_validate_horizon_default_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Weekly windows from the daily synthetic rows: only the spec key and the
    # window end matter to the ensemble. The parquet holds only the candidates
    # and the references, as `models validate --models candidates` writes.
    monkeypatch.chdir(tmp_path)
    source = Path("data/processed/model_predictions_weekly.parquet")
    source.parent.mkdir(parents=True)
    weekly = pl.read_parquet(make_cli_parquet(tmp_path / "oof.parquet")).with_columns(
        spec=pl.lit(WEEKLY.key), target_end=pl.col("target_start") + pl.duration(days=6)
    )
    weekly.write_parquet(source)
    result = runner.invoke(
        en.app, ["validate", "--horizon", "weekly", "--end", "2022-04-30"]
    )
    assert result.exit_code == 0, result.output
    assert "horizon weekly (lag2_lead1_win7)" in result.output
    report = json.loads(Path("reports/ensemble_validation_weekly.json").read_text())
    assert report["config"]["spec"]["key"] == WEEKLY.key
    assert report["input_sha256"] == sha256_file(source)
    assert set(report["wape"]) == {
        *ENSEMBLE_CANDIDATES,
        *REFERENCES,
        "ens_mean",
        "ens_median",
        "ens_weighted",
    }
    rows = pl.read_parquet("data/processed/ensemble_predictions_weekly.parquet")
    assert rows["spec"].unique().to_list() == [WEEKLY.key]
    assert not Path("reports/ensemble_validation.json").exists()


def test_cli_missing_parquet_names_horizon(tmp_path: Path) -> None:
    result = runner.invoke(
        en.app,
        [
            "validate",
            "--horizon",
            "lead7",
            "--predictions",
            str(tmp_path / "missing.parquet"),
        ],
    )
    assert result.exit_code == 1
    assert "models validate --horizon lead7" in result.output
