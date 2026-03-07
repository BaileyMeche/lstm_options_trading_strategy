from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    assign_time_split,
    compute_event_intensity_diagnostics,
    compute_rolling_beta_vs_spy,
)


def test_compute_rolling_beta_vs_spy_recovers_linear_exposure() -> None:
    dates = pd.bdate_range("2020-01-01", periods=8)
    spy_price = np.array([100, 101, 102, 103, 104, 105, 106, 107], dtype=float)

    # Build A with approximately 2x daily log return exposure to SPY.
    spy_lr = np.r_[np.nan, np.diff(np.log(spy_price))]
    a_price = [50.0]
    for idx in range(1, len(dates)):
        a_price.append(a_price[-1] * float(np.exp(2.0 * spy_lr[idx])))

    prices = pd.DataFrame(
        {
            "ticker": ["SPY"] * len(dates) + ["A"] * len(dates),
            "date": list(dates) + list(dates),
            "adj_close": list(spy_price) + a_price,
        }
    )

    beta = compute_rolling_beta_vs_spy(prices, window=4, min_obs=4)
    last = beta[beta["ticker"] == "A"].dropna(subset=["beta_252d"]).iloc[-1]
    assert np.isclose(last["beta_252d"], 2.0, atol=0.15)


def test_compute_event_intensity_diagnostics_raw_and_beta_hedged() -> None:
    dates = pd.bdate_range("2020-01-01", periods=12)
    prices = pd.DataFrame(
        {
            "ticker": ["SPY"] * len(dates) + ["A"] * len(dates),
            "date": list(dates) + list(dates),
            "adj_close": list(np.linspace(100, 112, len(dates))) + list([50, 51, 52, 53, 54, 56, 55, 57, 58, 59, 60, 61]),
        }
    )

    fundamentals = pd.DataFrame(
        {
            "ticker_price": ["A", "A"],
            "feature_available_date": [dates[3], dates[7]],
            "tot_debt_tot_equity": [1.0, 1.1],
            "ret_equity": [0.1, 0.12],
            "profit_margin": [0.2, 0.25],
            "book_val_per_share": [10.0, 10.5],
            "diluted_net_eps": [2.0, 2.2],
        }
    )

    raw = compute_event_intensity_diagnostics(mode="raw", prices_df=prices, fundamentals_df=fundamentals, window=2)
    assert raw["heatmap_df"].shape[1] == 5
    assert not raw["event_panel_df"].empty

    beta_df = compute_rolling_beta_vs_spy(prices, window=4, min_obs=4)
    hedged = compute_event_intensity_diagnostics(
        mode="beta_hedged",
        prices_df=prices,
        fundamentals_df=fundamentals,
        beta_df=beta_df,
        window=2,
    )
    assert hedged["heatmap_df"].shape[1] == 5

    with pytest.raises(ValueError, match="beta_df is required"):
        compute_event_intensity_diagnostics(mode="beta_hedged", prices_df=prices, fundamentals_df=fundamentals, window=2)


def test_assign_time_split_labels_expected_ranges() -> None:
    panel = pd.DataFrame({"date": pd.to_datetime(["2019-12-31", "2020-01-02", "2021-01-04", "2022-01-03"])})
    out = assign_time_split(
        panel_df=panel,
        train_start=pd.Timestamp("2020-01-01"),
        train_end=pd.Timestamp("2020-12-31"),
        dev_start=pd.Timestamp("2021-01-01"),
        dev_end=pd.Timestamp("2021-12-31"),
        test_start=pd.Timestamp("2022-01-01"),
        test_end=pd.Timestamp("2022-12-31"),
    )

    assert pd.isna(out.loc[0, "split"])
    assert out.loc[1, "split"] == "train"
    assert out.loc[2, "split"] == "dev"
    assert out.loc[3, "split"] == "test"
