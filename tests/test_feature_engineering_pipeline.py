from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    add_fundamental_change_features,
    add_price_liquidity_features,
    add_split_adjusted_intraday_prices,
    add_staged_features,
    build_lstm_tensors,
    get_stage_feature_columns,
    winsorize_cross_sectional,
)


def test_add_split_adjusted_intraday_prices_scales_open_and_close() -> None:
    df = pd.DataFrame({"open": [10.0], "close": [20.0], "adj_close": [30.0], "volume": [100]})
    out = add_split_adjusted_intraday_prices(df)

    assert np.isclose(out.loc[0, "adj_factor"], 1.5)
    assert np.isclose(out.loc[0, "adj_open"], 15.0)
    assert np.isclose(out.loc[0, "adj_close_intraday"], 30.0)


def test_add_fundamental_change_features_builds_diff_and_growth() -> None:
    df = pd.DataFrame(
        {
            "ticker": ["A", "A", "A"],
            "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "tot_debt_tot_equity": [1.0, 1.5, 1.2],
            "ret_equity": [0.1, 0.15, 0.20],
            "profit_margin": [0.2, 0.25, 0.20],
            "diluted_net_eps": [2.0, 2.5, 2.5],
            "book_val_per_share": [10.0, 11.0, 11.0],
        }
    )
    out = add_fundamental_change_features(df)

    assert np.isclose(out.loc[1, "leverage_change"], 0.5)
    assert np.isclose(out.loc[2, "roe_change"], 0.05)
    assert np.isclose(out.loc[1, "eps_growth"], 0.25)
    assert np.isclose(out.loc[2, "book_value_growth"], 0.0)


def test_add_price_liquidity_features_adds_log_return_and_volume_ratio() -> None:
    df = pd.DataFrame(
        {
            "ticker": ["A", "A", "A", "A"],
            "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06"]),
            "adj_close": [10.0, 11.0, 12.1, 12.1],
            "volume": [100.0, 100.0, 200.0, 300.0],
        }
    )
    out = add_price_liquidity_features(df, volume_window=2, min_volume_obs=2)

    assert np.isnan(out.loc[0, "log_return"])
    assert np.isclose(out.loc[1, "log_return"], np.log(11.0 / 10.0))
    assert np.isnan(out.loc[0, "volume_ratio"])
    assert np.isclose(out.loc[2, "volume_ratio"], 200.0 / 150.0)


def test_add_staged_features_handles_pre_report_rows_and_computes_late_stages() -> None:
    dates = pd.bdate_range("2020-01-01", periods=10)
    feature_available = [pd.NaT, pd.NaT, dates[2], dates[2], dates[5], dates[5], dates[5], dates[7], dates[7], dates[7]]

    panel = pd.DataFrame(
        {
            "ticker": ["A"] * len(dates),
            "date": dates,
            "feature_available_date": feature_available,
            "adj_close": [100, 101, 102, 103, 102, 104, 106, 107, 109, 108],
            "roe_change": [np.nan, np.nan, 0.10, 0.10, 0.10, 0.25, 0.25, 0.40, 0.40, 0.40],
            "margin_change": [np.nan, np.nan, 0.02, 0.02, 0.02, 0.03, 0.03, 0.01, 0.01, 0.01],
            "eps_growth": [np.nan, np.nan, 0.05, 0.05, 0.05, 0.10, 0.10, 0.12, 0.12, 0.12],
        }
    )

    out = add_staged_features(
        panel_df=panel,
        max_stage=5,
        surprise_window=2,
        surprise_min_obs=1,
        vol_short_window=3,
        vol_short_min_obs=2,
        vol_long_window=5,
        vol_long_min_obs=3,
    )

    # Stage 2: acceleration updates when report regime changes.
    row_report_2 = out[out["date"] == dates[5]].iloc[0]
    assert np.isclose(row_report_2["roe_change_accel"], 0.15)

    # Stage 4 guard: no report available means no reaction-speed metric.
    pre_report = out[out["date"].isin([dates[0], dates[1]])]
    assert pre_report["reaction_speed"].isna().all()

    # Stage 5 should eventually be available with custom short windows.
    assert out["vol_regime_ratio_20_60"].notna().sum() > 0


def test_winsorize_cross_sectional_clips_by_date() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"] * 5 + ["2020-01-02"] * 5),
            "x": [1, 2, 3, 4, 100, 1, 2, 3, 4, 100],
        }
    )
    out = winsorize_cross_sectional(df, feature_cols=["x"], lower_q=0.2, upper_q=0.8)

    d1 = out[out["date"] == pd.Timestamp("2020-01-01")]["x"]
    assert d1.max() < 100


def test_get_stage_feature_columns_validates_range() -> None:
    cols = get_stage_feature_columns(max_stage=3)
    assert "eps_growth_surprise" in cols

    with pytest.raises(ValueError, match="between 1 and 5"):
        get_stage_feature_columns(max_stage=0)


def test_build_lstm_tensors_shapes_and_invalid_lookback() -> None:
    panel = pd.DataFrame(
        {
            "ticker": ["A", "A", "A", "B", "B", "B"],
            "date": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-01", "2020-01-02", "2020-01-03"]
            ),
            "split": ["train", "train", "train", "dev", "dev", "test"],
            "f1": [1, 2, 3, 1, 2, 3],
            "f2": [2, 3, 4, 2, 3, 4],
            "target": [0.1, 0.2, 0.3, 0.1, 0.2, 0.3],
        }
    )

    tensors = build_lstm_tensors(panel, feature_cols=["f1", "f2"], target_col="target", lookback=2)
    x_train, y_train = tensors["train"]
    assert x_train.shape == (2, 2, 2)
    assert y_train.shape == (2,)

    with pytest.raises(ValueError, match="lookback"):
        build_lstm_tensors(panel, feature_cols=["f1", "f2"], target_col="target", lookback=0)
