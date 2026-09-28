"""Seed every random number generator a training run touches."""

import os
import random

import numpy as np

DEFAULT_SEED = 42


def seed_everything(seed: int = DEFAULT_SEED, *, deterministic: bool = False) -> int:
    """Seed Python, NumPy and (if installed) PyTorch.

    Libraries that take their own seed (scikit-learn, XGBoost, LightGBM,
    CatBoost) still need ``random_state``/``seed`` passed explicitly; pass
    the value this function returns.

    Args:
        seed: Seed applied to every generator.
        deterministic: Also request deterministic PyTorch kernels. Slower,
            and some GPU ops raise instead of running nondeterministically.

    Returns:
        The seed that was applied.
    """
    # Only affects subprocesses started after this point (e.g. DataLoader
    # workers); the current interpreter's hash seed is fixed at startup.
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # legacy global state still read by older libs

    try:
        import torch  # pyright: ignore[reportMissingImports] - gpu group, absent in CI
    except ImportError:
        return seed

    torch.manual_seed(seed)  # also seeds every GPU device
    if deterministic:
        torch.use_deterministic_algorithms(mode=True)
    return seed


def rng(seed: int = DEFAULT_SEED) -> np.random.Generator:
    """Return a NumPy ``Generator``; prefer it over the global NumPy state.

    Args:
        seed: Seed for the generator.

    Returns:
        A new, independently seeded generator.
    """
    return np.random.default_rng(seed)
