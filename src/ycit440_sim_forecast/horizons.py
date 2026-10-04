"""Compare the final ensemble across forecast settings, from the validation reports.

Reads the ensemble report that ``ensemble validate --horizon <h>`` wrote for
each horizon label and puts, side by side, ``FINAL_MODEL``, both references
and the best single candidate (picked in hindsight per setting, as in the
ensemble report): scored rows, MAE, bias, WAPE (scale-free, so daily and
weekly settings compare directly) and the paired-bootstrap MAE difference of
``FINAL_MODEL`` against each. Usage::

    uv run python -m ycit440_sim_forecast.horizons compare
    uv run python -m ycit440_sim_forecast.horizons compare v2 weekly lead30
"""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any

import typer

from ycit440_sim_forecast.ensemble import DEFAULT_OUT as ENSEMBLE_OUT
from ycit440_sim_forecast.ensemble import FINAL_MODEL
from ycit440_sim_forecast.models import (
    REFERENCES,
    canonical_horizon,
    horizon_path,
    rounded,
)

DEFAULT_LABELS: tuple[str, ...] = ("v2", "weekly", "lead3", "lead7", "lead14", "lead30")
"""Settings compared by default, in table order."""

DEFAULT_REPORTS = ENSEMBLE_OUT.parent
DEFAULT_OUT = Path("reports/horizon_comparison.json")

HINDSIGHT = "best_single_hindsight"
"""Role of the best single candidate, picked on the scored period itself."""

app = typer.Typer(help=__doc__, no_args_is_help=True)


def report_path(reports: Path, label: str) -> Path:
    """Ensemble report written by ``ensemble validate --horizon <label>``.

    Args:
        reports: Directory holding the ensemble reports.
        label: Canonical horizon label.

    Returns:
        ``reports / ensemble_validation[_<label>].json``.
    """
    return reports / horizon_path(ENSEMBLE_OUT, label).name


def _row(table: Sequence[Mapping[str, Any]], model: str, label: str) -> dict[str, Any]:
    """The row of ``model`` in a score table.

    Args:
        table: Score rows with a ``model`` key.
        model: Model to find.
        label: Horizon label, for the error message.

    Returns:
        The matching row.

    Raises:
        ValueError: If ``model`` has no row.
    """
    found = [dict(row) for row in table if row["model"] == model]
    if not found:
        msg = f"{label}: {model} is missing from the ensemble report"
        raise ValueError(msg)
    return found[0]


def _final_vs(
    comparisons: Sequence[Mapping[str, Any]], reference: str, label: str
) -> dict[str, Any]:
    """Paired bootstrap of ``FINAL_MODEL`` against ``reference``.

    Args:
        comparisons: Bootstrap results against ``reference``, one per method.
        reference: Model compared against.
        label: Horizon label, for the error message.

    Returns:
        ``diff``, ``ci_low``, ``ci_high`` and ``n_blocks``.

    Raises:
        ValueError: If the report has no ``FINAL_MODEL`` comparison.
    """
    row = _row(comparisons, FINAL_MODEL, label)
    if row["reference"] != reference:
        msg = f"{label}: bootstrap of {FINAL_MODEL} is against {row['reference']}"
        raise ValueError(msg)
    return {k: row[k] for k in ("diff", "ci_low", "ci_high", "n_blocks")}


def setting_summary(report: Mapping[str, Any], label: str) -> dict[str, Any]:
    """One setting's row of the comparison, from its ensemble report.

    Args:
        report: Parsed ``ensemble_validation`` JSON of the setting.
        label: Canonical horizon label of the setting.

    Returns:
        ``spec``, ``config_id``, ``input_sha256``, ``period``, ``n`` (scored
        rows, the same for every model), ``best_single`` and ``models``: for
        ``FINAL_MODEL``, each of ``REFERENCES`` and the best single candidate,
        its ``role``, ``mae``, ``bias``, ``wape`` and, except for
        ``FINAL_MODEL``, ``final_minus`` (the paired-bootstrap MAE difference
        of ``FINAL_MODEL`` minus that model, with its 95% interval).

    Raises:
        ValueError: If the report has no ``wape`` (it predates WAPE; rerun
            ``ensemble validate``), or a model, bootstrap or count is missing
            or inconsistent.
    """
    if "wape" not in report:
        msg = (
            f"{label}: the ensemble report has no WAPE; rerun "
            f"`uv run python -m ycit440_sim_forecast.ensemble validate "
            f"--horizon {label}`"
        )
        raise ValueError(msg)
    overall = report["overall"]
    counts = {row["n"] for row in overall}
    if len(counts) != 1:
        msg = f"{label}: models are scored on different row counts {sorted(counts)}"
        raise ValueError(msg)
    boot = report["bootstrap"]
    best = boot["best_single"]["model"]
    roles = {
        FINAL_MODEL: "final",
        **dict.fromkeys(REFERENCES, "reference"),
        best: HINDSIGHT,
    }
    comparisons = {
        **{ref: boot["references"][ref] for ref in REFERENCES},
        best: boot["best_single"]["results"],
    }
    models: dict[str, dict[str, Any]] = {}
    for model, role in roles.items():
        row = _row(overall, model, label)
        if model not in report["wape"]:
            msg = f"{label}: {model} has no WAPE in the ensemble report"
            raise ValueError(msg)
        entry: dict[str, Any] = {
            "role": role,
            "mae": row["mae"],
            "bias": row["bias"],
            "wape": report["wape"][model],
        }
        if model != FINAL_MODEL:
            entry["final_minus"] = _final_vs(comparisons[model], model, label)
        models[model] = entry
    return {
        "spec": report["config"]["spec"],
        "config_id": report["config_id"],
        "input_sha256": report["input_sha256"],
        "period": report["period"],
        "n": counts.pop(),
        "best_single": best,
        "models": models,
    }


def horizon_comparison(reports: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Side-by-side summary of ``FINAL_MODEL`` across settings.

    Args:
        reports: Parsed ensemble report per canonical horizon label, in
            table order.

    Returns:
        ``final_model``, ``labels`` (in order), ``best_single_selection``
        (how the hindsight model is picked) and ``settings`` (per label, see
        ``setting_summary``), floats rounded to 3.

    Raises:
        ValueError: If ``reports`` is empty, the settings were scored on
            different periods, or ``setting_summary`` fails.
    """
    if not reports:
        msg = "no ensemble reports to compare"
        raise ValueError(msg)
    settings = {label: setting_summary(r, label) for label, r in reports.items()}
    periods = {json.dumps(s["period"], sort_keys=True) for s in settings.values()}
    if len(periods) != 1:
        msg = f"settings were scored on different periods: {sorted(periods)}"
        raise ValueError(msg)
    selection = next(iter(reports.values()))["bootstrap"]["best_single"]["selection"]
    return rounded(
        {
            "final_model": FINAL_MODEL,
            "labels": list(settings),
            "best_single_selection": selection,
            "settings": settings,
        }
    )


def format_table(comparison: Mapping[str, Any]) -> list[str]:
    """Plain-text table of a ``horizon_comparison`` result.

    Args:
        comparison: Output of ``horizon_comparison``.

    Returns:
        A header and one line per setting and model; ``*`` marks the best
        single candidate picked in hindsight.
    """
    lines = [
        f"{'setting':<8} {'spec':<16} {'n':>6}  {'model':<20} "
        f"{'MAE':>8} {'bias':>8} {'WAPE':>6}  {FINAL_MODEL} minus model [95% CI]"
    ]
    for label in comparison["labels"]:
        setting = comparison["settings"][label]
        for model, row in setting["models"].items():
            mark = "*" if row["role"] == HINDSIGHT else ""
            diff = row.get("final_minus")
            versus = (
                ""
                if diff is None
                else f"{diff['diff']:+7.3f} [{diff['ci_low']:+7.3f}, "
                f"{diff['ci_high']:+7.3f}]"
            )
            lines.append(
                f"{label:<8} {setting['spec']['key']:<16} {setting['n']:>6}  "
                f"{model + mark:<20} {row['mae']:8.3f} {row['bias']:+8.3f} "
                f"{row['wape']:6.3f}  {versus}"
            )
    lines.append("* best single candidate, picked in hindsight on the scored period")
    return lines


@app.callback()
def main() -> None:
    """Compare the final ensemble across forecast settings."""


@app.command()
def compare(
    labels: Annotated[
        list[str] | None,
        typer.Argument(
            help="Horizon labels (v2, weekly, lead<N>) [default: "
            + " ".join(DEFAULT_LABELS)
            + "]."
        ),
    ] = None,
    reports: Annotated[
        Path, typer.Option(help="Directory of the ensemble reports.")
    ] = DEFAULT_REPORTS,
    out: Annotated[Path, typer.Option(help="JSON comparison path.")] = DEFAULT_OUT,
) -> None:
    """Summarise the ensemble reports of several horizons in one table.

    Args:
        labels: Horizon labels; ``DEFAULT_LABELS`` when none are given.
        reports: Directory of the ensemble reports.
        out: JSON comparison path.

    Raises:
        typer.BadParameter: If a label is not a horizon label or repeats.
        typer.Exit: If a setting's ensemble report is missing.
    """
    canonical = [canonical_horizon(label) for label in labels or DEFAULT_LABELS]
    if len(set(canonical)) != len(canonical):
        msg = f"horizon labels repeat: {canonical}"
        raise typer.BadParameter(msg)
    loaded: dict[str, Any] = {}
    for label in canonical:
        path = report_path(reports, label)
        if not path.is_file():
            typer.echo(
                f"{path} not found; run `uv run python -m "
                f"ycit440_sim_forecast.models validate --horizon {label}` then "
                f"`uv run python -m ycit440_sim_forecast.ensemble validate "
                f"--horizon {label}` first",
                err=True,
            )
            raise typer.Exit(code=1)
        loaded[label] = json.loads(path.read_text())
    comparison = horizon_comparison(loaded)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    for line in format_table(comparison):
        typer.echo(line)
    typer.echo(f"wrote {out}")


if __name__ == "__main__":
    app()
