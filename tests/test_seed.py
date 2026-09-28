import random
import sys
import types

import numpy as np
import pytest

from ycit440_sim_forecast import seed


def test_seed_everything_repeats_python_and_numpy() -> None:
    # S311 flags non-crypto randomness; here it is the thing under test.
    seed.seed_everything(123)
    first = (random.random(), np.random.rand())  # noqa: S311
    seed.seed_everything(123)
    assert (random.random(), np.random.rand()) == first  # noqa: S311


def test_rng_is_reproducible() -> None:
    assert seed.rng(7).integers(0, 1_000) == seed.rng(7).integers(0, 1_000)


def test_seed_everything_seeds_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, object]] = []
    fake_torch = types.SimpleNamespace(
        manual_seed=lambda value: calls.append(("manual_seed", value)),
        use_deterministic_algorithms=lambda *, mode: calls.append(("det", mode)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert seed.seed_everything(5, deterministic=True) == 5
    assert calls == [("manual_seed", 5), ("det", True)]


def test_seed_everything_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)  # import raises ImportError
    assert seed.seed_everything(9) == 9
