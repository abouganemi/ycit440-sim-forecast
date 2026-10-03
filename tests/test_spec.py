import dataclasses
import datetime as dt

import pytest

from ycit440_sim_forecast import spec
from ycit440_sim_forecast.spec import ForecastSpec

ISSUE = dt.date(2024, 3, 9)


def test_v2_daily_matches_framing() -> None:
    # Issued on D-1 with data through D-3, target day D.
    s = spec.V2_DAILY
    assert s.cutoff(ISSUE) == dt.date(2024, 3, 7)
    assert s.target_days(ISSUE) == [dt.date(2024, 3, 10)]
    assert s.gap_days == 3
    assert s.key == "lag2_lead1_win1"


def test_weekly_window() -> None:
    s = spec.WEEKLY
    assert s.target_start(ISSUE) == dt.date(2024, 3, 10)
    assert s.target_end(ISSUE) == dt.date(2024, 3, 16)
    assert len(s.target_days(ISSUE)) == 7
    assert s.cutoff(ISSUE) == spec.V2_DAILY.cutoff(ISSUE)


def test_issue_date_for_inverts_target_start() -> None:
    for s in (spec.V2_DAILY, spec.WEEKLY, *spec.daily_horizon(5)):
        start = dt.date(2024, 1, 1)
        assert s.target_start(s.issue_date_for(start)) == start


def test_daily_horizon() -> None:
    specs = spec.daily_horizon(3)
    assert [s.lead_days for s in specs] == [1, 2, 3]
    assert all(s.window_days == 1 and s.publication_lag_days == 2 for s in specs)
    assert [s.gap_days for s in specs] == [3, 4, 5]


def test_daily_horizon_rejects_zero() -> None:
    with pytest.raises(ValueError, match="n must be"):
        spec.daily_horizon(0)


@pytest.mark.parametrize(
    ("field", "value"),
    [("publication_lag_days", -1), ("lead_days", 0), ("window_days", 0)],
)
def test_invalid_spec(field: str, value: int) -> None:
    with pytest.raises(ValueError, match=field):
        ForecastSpec(**{field: value})


def test_spec_is_frozen_and_hashable() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.V2_DAILY.lead_days = 2  # pyright: ignore[reportAttributeAccessIssue]
    assert {spec.V2_DAILY: 1}[ForecastSpec()] == 1
