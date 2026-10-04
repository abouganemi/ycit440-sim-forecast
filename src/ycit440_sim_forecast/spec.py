"""Define what a forecast is: when it is issued, what it may see, what it targets.

A forecast is issued on ``issue_date``. The portal publishes about
``publication_lag_days`` late, so the last complete day it can use is the
``cutoff``. It targets the total count over ``window_days`` consecutive days
starting ``lead_days`` after the issue date. The v2 framing is the default:
issued on D-1, data through D-3, target day D.

Every forecaster, baseline or model, follows the ``Forecaster`` protocol so the
validation harness can score them the same way.
"""

import datetime as dt
from dataclasses import dataclass
from typing import Protocol, Self

import polars as pl


@dataclass(frozen=True)
class ForecastSpec:
    """Timing of one forecast setting.

    Attributes:
        publication_lag_days: Days between the cutoff and the issue date.
        lead_days: Days from the issue date to the first target day.
        window_days: Number of consecutive target days summed into one value.
    """

    publication_lag_days: int = 2
    lead_days: int = 1
    window_days: int = 1

    def __post_init__(self) -> None:
        """Reject settings that would let a forecast see its own target.

        Raises:
            ValueError: If a field is out of range.
        """
        if self.publication_lag_days < 0:
            msg = f"publication_lag_days must be >= 0, got {self.publication_lag_days}"
            raise ValueError(msg)
        if self.lead_days < 1:
            msg = f"lead_days must be >= 1, got {self.lead_days}"
            raise ValueError(msg)
        if self.window_days < 1:
            msg = f"window_days must be >= 1, got {self.window_days}"
            raise ValueError(msg)

    @property
    def key(self) -> str:
        """Short label for reports, e.g. ``lag2_lead1_win1``."""
        return (
            f"lag{self.publication_lag_days}_lead{self.lead_days}_win{self.window_days}"
        )

    @property
    def gap_days(self) -> int:
        """Days from the cutoff to the first target day (3 for v2: D-3 to D)."""
        return self.publication_lag_days + self.lead_days

    def cutoff(self, issue_date: dt.date) -> dt.date:
        """Last day whose count is known on ``issue_date``.

        Args:
            issue_date: Day the forecast is issued.

        Returns:
            ``issue_date - publication_lag_days``.
        """
        return issue_date - dt.timedelta(days=self.publication_lag_days)

    def target_start(self, issue_date: dt.date) -> dt.date:
        """First target day of the forecast issued on ``issue_date``.

        Args:
            issue_date: Day the forecast is issued.

        Returns:
            ``issue_date + lead_days``.
        """
        return issue_date + dt.timedelta(days=self.lead_days)

    def target_end(self, issue_date: dt.date) -> dt.date:
        """Last target day (inclusive) of the forecast issued on ``issue_date``.

        Args:
            issue_date: Day the forecast is issued.

        Returns:
            ``issue_date + lead_days + window_days - 1``.
        """
        return issue_date + dt.timedelta(days=self.lead_days + self.window_days - 1)

    def target_days(self, issue_date: dt.date) -> list[dt.date]:
        """Every target day of the forecast issued on ``issue_date``, in order.

        Args:
            issue_date: Day the forecast is issued.

        Returns:
            ``window_days`` consecutive days from ``target_start`` to
            ``target_end``.
        """
        start = self.target_start(issue_date)
        return [start + dt.timedelta(days=i) for i in range(self.window_days)]

    def issue_date_for(self, target_start: dt.date) -> dt.date:
        """Issue date whose window starts on ``target_start``.

        Args:
            target_start: First target day.

        Returns:
            ``target_start - lead_days``.
        """
        return target_start - dt.timedelta(days=self.lead_days)


V2_DAILY = ForecastSpec()
"""The official v2 target: tomorrow's count, data through D-3."""

WEEKLY = ForecastSpec(window_days=7)
"""Weekly total, issued the day before the week starts (planned comparison)."""


def daily_horizon(n: int) -> tuple[ForecastSpec, ...]:
    """Daily forecasts for each of the next ``n`` days (``DAILY_H<n>``).

    Args:
        n: Number of leads, 1 to ``n``.

    Returns:
        One spec per lead, each forecasting a single day.

    Raises:
        ValueError: If ``n`` is below 1.
    """
    if n < 1:
        msg = f"n must be >= 1, got {n}"
        raise ValueError(msg)
    return tuple(ForecastSpec(lead_days=lead) for lead in range(1, n + 1))


class Forecaster(Protocol):
    """Anything the validation harness can fit and score.

    ``history`` is a ``date, DIVISION, n`` frame on the full calendar (a day
    with no records has a null ``n``). The harness only ever passes rows up to
    the relevant cutoff, so a forecaster cannot read past it.
    """

    @property
    def name(self) -> str:
        """Label used in result tables."""
        ...

    def fit(self, history: pl.DataFrame, spec: ForecastSpec) -> Self:
        """Learn from ``history``; must reset any state from an earlier call.

        Args:
            history: ``date, DIVISION, n`` up to the cutoff of the issue date
                the fit happens on.
            spec: Forecast setting.

        Returns:
            This forecaster.
        """
        ...

    def predict(
        self, history: pl.DataFrame, issue_date: dt.date, spec: ForecastSpec
    ) -> pl.DataFrame:
        """Forecast the window of ``issue_date`` for every division.

        Args:
            history: ``date, DIVISION, n`` up to the cutoff of ``issue_date``.
            issue_date: Day the forecast is issued.
            spec: Forecast setting.

        Returns:
            ``DIVISION, y_pred`` with one row per division in ``history``;
            ``y_pred`` is null when there is too little data.
        """
        ...
