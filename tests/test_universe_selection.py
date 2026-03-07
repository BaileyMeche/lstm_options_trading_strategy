import numpy as np
import pandas as pd

from src.universe_selection import (
    attach_universe_flags,
    build_annual_candidate_table,
    build_rebalance_calendar,
    expand_annual_membership_to_daily,
    finalize_annual_universe_with_options,
)


FUND_COLS = [
    "tot_debt_tot_equity",
    "ret_equity",
    "profit_margin",
    "book_val_per_share",
    "diluted_net_eps",
]


def _make_prices() -> pd.DataFrame:
    dates_a = pd.bdate_range("2005-01-03", "2007-12-31")
    dates_c = pd.bdate_range("2006-01-03", "2007-12-31")

    def _frame(ticker: str, dates: pd.DatetimeIndex, price: float, volume: float) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ticker": ticker,
                "date": dates,
                "adj_close": price,
                "volume": volume,
            }
        )

    return pd.concat(
        [
            _frame("A", dates_a, 100.0, 300_000.0),
            _frame("B", dates_a, 80.0, 350_000.0),
            _frame("C", dates_c, 50.0, 500_000.0),
        ],
        ignore_index=True,
    ).sort_values(["ticker", "date"])


def _make_mktv() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["A", "A", "B", "B", "C", "C"],
            "per_end_date": pd.to_datetime(
                ["2006-03-31", "2007-03-31", "2006-03-31", "2007-03-31", "2006-03-31", "2007-03-31"]
            ),
            "per_type": ["Q"] * 6,
            "mkt_val": [100.0, 120.0, 90.0, 80.0, 70.0, 130.0],
        }
    )


def _make_fundamentals() -> pd.DataFrame:
    def _rows_for_ticker(ticker: str, per_dates: list[str], margin_missing_at: set[int] | None = None) -> list[dict]:
        out: list[dict] = []
        margin_missing_at = margin_missing_at or set()
        for idx, dt_str in enumerate(per_dates):
            per_end = pd.Timestamp(dt_str)
            out.append(
                {
                    "ticker_price": ticker,
                    "per_end_date": per_end,
                    "feature_available_date": per_end + pd.Timedelta(days=45),
                    "tot_debt_tot_equity": 0.5 + 0.01 * idx,
                    "ret_equity": 0.10 + 0.005 * idx,
                    "profit_margin": np.nan if idx in margin_missing_at else 0.20 + 0.01 * idx,
                    "book_val_per_share": 10.0 + idx,
                    "diluted_net_eps": 2.0 + 0.1 * idx,
                }
            )
        return out

    rows: list[dict] = []
    rows.extend(_rows_for_ticker("A", ["2005-12-31", "2006-03-31", "2006-06-30", "2006-09-30", "2006-12-31"]))
    # B has 60% missing profit margin by 2007 rebalance (>40% threshold).
    rows.extend(
        _rows_for_ticker(
            "B",
            ["2005-12-31", "2006-03-31", "2006-06-30", "2006-09-30", "2006-12-31"],
            margin_missing_at={0, 1, 2},
        )
    )
    # C has only 3 quarters by 2007 rebalance (fails min-quarter rule outside bootstrap).
    rows.extend(_rows_for_ticker("C", ["2006-03-31", "2006-06-30", "2006-09-30"]))
    return pd.DataFrame(rows)


def test_build_rebalance_calendar_uses_first_trading_day_after_may_15() -> None:
    prices = pd.DataFrame({"date": pd.bdate_range("2006-05-10", "2007-05-25")})
    cal = build_rebalance_calendar(prices_df=prices, start_year=2006, end_year=2007, anchor_month=5, anchor_day=15)

    assert cal["year"].tolist() == [2006, 2007]
    assert cal.loc[0, "rebalance_date"] == pd.Timestamp("2006-05-16")
    assert cal.loc[1, "rebalance_date"] == pd.Timestamp("2007-05-16")


def test_build_annual_candidate_table_snapshot_bootstrap_and_missingness() -> None:
    prices = _make_prices()
    mktv = _make_mktv()
    fundamentals = _make_fundamentals()
    rebalance = build_rebalance_calendar(prices_df=prices, start_year=2006, end_year=2007)

    candidates = build_annual_candidate_table(
        mktv_df=mktv,
        fundamentals_df=fundamentals,
        prices_df=prices,
        rebalance_df=rebalance,
        target_size=30,
        buffer_size=15,
        missingness_max=0.40,
        addv_min=20_000_000,
        min_price=5.0,
    )

    assert not candidates.empty
    assert (candidates["mktv_per_end_date"] <= candidates["rebalance_date"]).all()

    c_2006 = candidates[(candidates["ticker"] == "C") & (candidates["year"] == 2006)].iloc[0]
    c_2007 = candidates[(candidates["ticker"] == "C") & (candidates["year"] == 2007)].iloc[0]
    b_2007 = candidates[(candidates["ticker"] == "B") & (candidates["year"] == 2007)].iloc[0]

    # 2006 bootstrap waiver allows quarter/history checks to pass.
    assert bool(c_2006["is_bootstrap_2006"])
    assert bool(c_2006["pass_min_quarters"])
    assert bool(c_2006["pass_history_252d"])
    # Outside bootstrap, C fails min-quarter rule.
    assert not bool(c_2007["is_bootstrap_2006"])
    assert not bool(c_2007["pass_min_quarters"])
    # B fails missingness due to 60% missing profit margin.
    assert not bool(b_2007["pass_missingness_profit_margin"])
    assert not bool(b_2007["pass_missingness_all"])


def test_finalize_annual_universe_with_options_uses_plus_minus_window() -> None:
    rebalance_date = pd.Timestamp("2007-05-16")
    candidates = pd.DataFrame(
        {
            "year": [2007, 2007, 2007, 2007],
            "rebalance_date": [rebalance_date] * 4,
            "ticker": ["A", "B", "C", "D"],
            "mkt_val": [100.0, 90.0, 80.0, 70.0],
            "pre_options_pass": [True, True, True, True],
            "in_pre_options_buffer": [True, True, False, False],
            "pre_options_rank": [1, 2, 3, 4],
        }
    )
    trading_calendar = pd.DataFrame({"date": pd.bdate_range("2007-05-01", "2007-05-31")})
    options = pd.DataFrame(
        {
            "ticker": ["A", "C"],
            "date": [pd.Timestamp("2007-05-22"), pd.Timestamp("2007-05-11")],  # +4 and -5 trading days
        }
    )

    final = finalize_annual_universe_with_options(
        candidates_df=candidates,
        options_df=options,
        trading_calendar_df=trading_calendar,
        target_size=2,
        window_days=5,
    )

    assert final["ticker"].tolist() == ["A", "C"]
    assert final["selection_rank"].tolist() == [1, 2]
    assert final["has_options_rebalance_window"].all()


def test_expand_annual_membership_to_daily_and_attach_flags() -> None:
    annual = pd.DataFrame(
        {
            "selection_year": [2006, 2007, 2007],
            "rebalance_date": pd.to_datetime(["2006-05-16", "2007-05-16", "2007-05-16"]),
            "ticker": ["A", "B", "C"],
            "selection_rank": [1, 1, 2],
        }
    )
    calendar = pd.DataFrame({"date": pd.bdate_range("2006-05-15", "2007-05-21")})
    daily = expand_annual_membership_to_daily(annual_universe_df=annual, trading_calendar_df=calendar)

    # A should be in universe before the 2007 rebalance switch.
    assert (
        daily[(daily["ticker"] == "A") & (daily["date"] == pd.Timestamp("2006-05-16"))]["in_universe"].iloc[0]
    )
    # After 2007 rebalance, membership should switch to B/C.
    assert daily[(daily["date"] == pd.Timestamp("2007-05-16")) & (daily["ticker"] == "A")].empty
    assert (
        daily[(daily["ticker"] == "B") & (daily["date"] == pd.Timestamp("2007-05-16"))]["in_universe"].iloc[0]
    )

    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2007-05-16", "2007-05-16", "2007-05-17"]),
            "ticker": ["B", "C", "A"],
        }
    )
    options = pd.DataFrame(
        {
            "date": pd.to_datetime(["2007-05-16"]),
            "ticker": ["B"],
        }
    )
    flagged = attach_universe_flags(panel_df=panel, daily_membership_df=daily, options_df=options)

    b_row = flagged[(flagged["ticker"] == "B") & (flagged["date"] == pd.Timestamp("2007-05-16"))].iloc[0]
    c_row = flagged[(flagged["ticker"] == "C") & (flagged["date"] == pd.Timestamp("2007-05-16"))].iloc[0]
    a_row = flagged[(flagged["ticker"] == "A") & (flagged["date"] == pd.Timestamp("2007-05-17"))].iloc[0]

    assert bool(b_row["in_universe"])
    assert bool(b_row["has_option_today"])
    assert bool(b_row["tradable_today"])

    assert bool(c_row["in_universe"])
    assert not bool(c_row["has_option_today"])
    assert not bool(c_row["tradable_today"])

    assert not bool(a_row["in_universe"])
    assert not bool(a_row["tradable_today"])
