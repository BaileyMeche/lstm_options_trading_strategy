from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.event_panels import (
    aggregate_event_time_intensity,
    build_event_time_abs_return_panel,
    build_event_time_metric_panel,
    build_global_trading_calendar,
    extract_fundamental_events,
)


def test_build_global_trading_calendar_prefers_spy_when_available() -> None:
    prices = pd.DataFrame(
        {
            "ticker": ["A", "A", "SPY", "SPY"],
            "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-02", "2020-01-03"]),
            "adj_close": [10, 11, 100, 101],
        }
    )
    cal = build_global_trading_calendar(prices)
    assert list(cal) == [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")]


def test_extract_fundamental_events_changed_only_handles_nan_stability() -> None:
    fundamentals = pd.DataFrame(
        {
            "ticker_price": ["A", "A", "A", "B", "B", "B"],
            "feature_available_date": pd.to_datetime(
                ["2020-01-01", "2020-02-01", "2020-03-01", "2020-01-05", "2020-02-05", "2020-03-05"]
            ),
            "ret_equity": [1.0, 1.0, 1.0, np.nan, np.nan, 2.0],
            "profit_margin": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        }
    )

    out = extract_fundamental_events(fundamentals, ticker_col="ticker_price", changed_only=True)

    a_dates = out[out["ticker"] == "A"]["feature_available_date"].tolist()
    b_dates = out[out["ticker"] == "B"]["feature_available_date"].tolist()

    assert a_dates == [pd.Timestamp("2020-01-01")]
    assert b_dates == [pd.Timestamp("2020-01-05"), pd.Timestamp("2020-03-05")]


def test_build_event_time_panels_and_aggregate() -> None:
    dates = pd.bdate_range("2020-01-01", periods=6)
    prices = pd.DataFrame(
        {
            "ticker": ["A"] * len(dates) + ["SPY"] * len(dates),
            "date": list(dates) + list(dates),
            "adj_close": [10, 11, 10, 12, 11, 12, 100, 101, 102, 101, 102, 103],
        }
    )
    events = pd.DataFrame(
        {
            "ticker": ["A"],
            # Weekend anchor should snap to next trading date.
            "feature_available_date": [pd.Timestamp("2020-01-04")],
        }
    )
    global_dates = build_global_trading_calendar(prices)

    raw_panel = build_event_time_abs_return_panel(prices_df=prices[prices["ticker"] == "A"], events_df=events, global_dates=global_dates, window=1)
    assert not raw_panel.empty
    assert (raw_panel["anchor_date"] >= raw_panel["feature_available_date"]).all()

    metric_df = pd.DataFrame(
        {
            "ticker": ["A"] * len(dates),
            "date": dates,
            "idio": [0.1, 0.2, 0.0, 0.3, 0.1, 0.4],
        }
    )
    metric_panel = build_event_time_metric_panel(metric_df, events, global_dates, value_col="idio", window=1)
    heat, baseline = aggregate_event_time_intensity(metric_panel, ticker_order=["A"], window=1, agg="mean", value_col="idio")

    assert list(heat.columns) == [-1, 0, 1]
    assert baseline.index.tolist() == [-1, 0, 1]


def test_aggregate_event_time_intensity_rejects_invalid_agg() -> None:
    panel = pd.DataFrame(
        {
            "ticker": ["A"],
            "event_day": [0],
            "abs_log_ret": [0.1],
        }
    )
    with pytest.raises(ValueError, match="agg"):
        aggregate_event_time_intensity(panel, ticker_order=["A"], window=0, agg="sum")


def test_aggregate_event_time_intensity_empty_panel_shape() -> None:
    empty = pd.DataFrame(columns=["ticker", "event_day", "abs_log_ret"])
    heat, baseline = aggregate_event_time_intensity(empty, ticker_order=["A", "B"], window=2)

    assert heat.shape == (2, 5)
    assert baseline.shape[0] == 5
