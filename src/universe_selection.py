from __future__ import annotations

import numpy as np
import pandas as pd

from .data_utils import normalize_ticker_for_prices


FUNDAMENTAL_FIELDS = [
    "tot_debt_tot_equity",
    "ret_equity",
    "profit_margin",
    "book_val_per_share",
    "diluted_net_eps",
]


def _normalize_ticker(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.upper()
        .str.strip()
        .replace({"": np.nan, "NAN": np.nan})
        .map(normalize_ticker_for_prices)
    )


def _require_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"{label} missing required columns: {missing}")


def build_rebalance_calendar(
    prices_df: pd.DataFrame,
    start_year: int,
    end_year: int,
    anchor_month: int = 5,
    anchor_day: int = 15,
) -> pd.DataFrame:
    """Build annual rebalance dates: first trading day strictly after anchor date."""
    _require_columns(prices_df, ["date"], "prices_df")
    if start_year > end_year:
        raise ValueError(f"start_year must be <= end_year, got {start_year} > {end_year}")

    dates = pd.to_datetime(prices_df["date"], errors="coerce").dropna()
    trading_dates = pd.DatetimeIndex(sorted(pd.Series(dates.dt.normalize()).drop_duplicates().tolist()))
    if trading_dates.empty:
        raise ValueError("prices_df does not contain valid trading dates")

    rows: list[dict[str, object]] = []
    for year in range(start_year, end_year + 1):
        anchor_date = pd.Timestamp(year=year, month=anchor_month, day=anchor_day)
        pos = int(trading_dates.searchsorted(anchor_date, side="right"))
        if pos >= len(trading_dates):
            continue
        rebalance_date = pd.Timestamp(trading_dates[pos])
        rows.append(
            {
                "year": year,
                "anchor_date": anchor_date,
                "rebalance_date": rebalance_date,
            }
        )

    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def _prepare_mktv(mktv_df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(mktv_df, ["ticker", "per_end_date", "mkt_val"], "mktv_df")
    out = mktv_df.copy()
    out["ticker"] = _normalize_ticker(out["ticker"])
    out["per_end_date"] = pd.to_datetime(out["per_end_date"], errors="coerce")
    out["mkt_val"] = pd.to_numeric(out["mkt_val"], errors="coerce")
    if "per_type" in out.columns:
        out = out[out["per_type"].astype(str).str.upper() == "Q"].copy()
    out = out.dropna(subset=["ticker", "per_end_date", "mkt_val"])
    return out.sort_values(["ticker", "per_end_date"]).reset_index(drop=True)


def _prepare_fundamentals(fundamentals_df: pd.DataFrame) -> pd.DataFrame:
    if "ticker_price" in fundamentals_df.columns:
        ticker_series = fundamentals_df["ticker_price"]
    elif "ticker" in fundamentals_df.columns:
        ticker_series = fundamentals_df["ticker"]
    else:
        raise KeyError("fundamentals_df must contain either 'ticker_price' or 'ticker'")

    required = ["feature_available_date", *FUNDAMENTAL_FIELDS]
    _require_columns(fundamentals_df, required, "fundamentals_df")

    out = fundamentals_df.copy()
    out["ticker"] = _normalize_ticker(ticker_series)
    out["feature_available_date"] = pd.to_datetime(out["feature_available_date"], errors="coerce")
    if "per_end_date" in out.columns:
        out["per_end_date"] = pd.to_datetime(out["per_end_date"], errors="coerce")
    else:
        out["per_end_date"] = out["feature_available_date"]

    for col in FUNDAMENTAL_FIELDS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["ticker", "feature_available_date"])
    out = out.sort_values(["ticker", "feature_available_date", "per_end_date"]).reset_index(drop=True)
    return out


def _prepare_prices(prices_df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(prices_df, ["ticker", "date", "adj_close", "volume"], "prices_df")
    out = prices_df.copy()
    out["ticker"] = _normalize_ticker(out["ticker"])
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["adj_close"] = pd.to_numeric(out["adj_close"], errors="coerce")
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
    out = out.dropna(subset=["ticker", "date", "adj_close", "volume"])
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    out["addv"] = out["adj_close"] * out["volume"]
    return out


def _trailing_addv_mean(series: pd.Series, window: int = 252, min_obs: int = 60) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if len(vals) < min_obs:
        return np.nan
    return float(vals.tail(window).mean())


def _compute_price_stats_asof(prices: pd.DataFrame, rebalance_date: pd.Timestamp) -> pd.DataFrame:
    sub = prices[prices["date"] <= rebalance_date].copy()
    if sub.empty:
        return pd.DataFrame(
            columns=["ticker", "price_obs_count", "latest_price_date", "latest_adj_close", "addv_lookback_mean"]
        )

    sub = sub.sort_values(["ticker", "date"])
    obs = sub.groupby("ticker")["date"].size().rename("price_obs_count")
    latest = sub.groupby("ticker", as_index=False).tail(1)[["ticker", "date", "adj_close"]].rename(
        columns={"date": "latest_price_date", "adj_close": "latest_adj_close"}
    )
    addv_mean = sub.groupby("ticker")["addv"].apply(_trailing_addv_mean).rename("addv_lookback_mean")

    out = obs.to_frame().merge(latest, on="ticker", how="outer")
    out = out.merge(addv_mean.to_frame(), on="ticker", how="outer")
    return out.reset_index(drop=True)


def _compute_fund_stats_asof(fundamentals: pd.DataFrame, rebalance_date: pd.Timestamp) -> pd.DataFrame:
    sub = fundamentals[fundamentals["feature_available_date"] <= rebalance_date].copy()
    if sub.empty:
        cols = ["ticker", "fund_obs_count", *[f"{col}_missingness" for col in FUNDAMENTAL_FIELDS]]
        return pd.DataFrame(columns=cols)

    # Treat each quarter once per ticker at availability time.
    sub = sub.sort_values(["ticker", "per_end_date", "feature_available_date"])
    sub = sub.drop_duplicates(["ticker", "per_end_date"], keep="last")
    grouped = sub.groupby("ticker")

    out = grouped["per_end_date"].nunique().rename("fund_obs_count").to_frame().reset_index()
    for col in FUNDAMENTAL_FIELDS:
        out[f"{col}_missingness"] = grouped[col].agg(lambda s: float(s.isna().mean())).to_numpy()
    return out


def build_annual_candidate_table(
    mktv_df: pd.DataFrame,
    fundamentals_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    rebalance_df: pd.DataFrame,
    target_size: int = 30,
    buffer_size: int = 15,
    missingness_max: float = 0.40,
    addv_min: float = 20_000_000,
    min_price: float = 5.0,
) -> pd.DataFrame:
    """Build annual pre-options universe candidates with PIT filters and rankings."""
    _require_columns(rebalance_df, ["year", "rebalance_date"], "rebalance_df")
    if target_size <= 0 or buffer_size < 0:
        raise ValueError("target_size must be > 0 and buffer_size must be >= 0")

    mktv = _prepare_mktv(mktv_df)
    fundamentals = _prepare_fundamentals(fundamentals_df)
    prices = _prepare_prices(prices_df)

    rebalance = rebalance_df.copy()
    rebalance["rebalance_date"] = pd.to_datetime(rebalance["rebalance_date"], errors="coerce")
    rebalance = rebalance.dropna(subset=["rebalance_date"]).sort_values("rebalance_date")

    rows: list[pd.DataFrame] = []
    buffer_cutoff = int(target_size + buffer_size)

    for _, reb_row in rebalance.iterrows():
        year = int(reb_row["year"])
        rebalance_date = pd.Timestamp(reb_row["rebalance_date"])

        snap = mktv[mktv["per_end_date"] <= rebalance_date].copy()
        if snap.empty:
            continue
        snap = snap.sort_values(["ticker", "per_end_date"]).drop_duplicates(["ticker"], keep="last")
        snap = snap.rename(columns={"per_end_date": "mktv_per_end_date"})

        price_stats = _compute_price_stats_asof(prices, rebalance_date)
        fund_stats = _compute_fund_stats_asof(fundamentals, rebalance_date)

        cand = snap.merge(price_stats, on="ticker", how="left").merge(fund_stats, on="ticker", how="left")
        cand["year"] = year
        cand["rebalance_date"] = rebalance_date

        for col in [
            "price_obs_count",
            "fund_obs_count",
            "latest_adj_close",
            "addv_lookback_mean",
            *[f"{name}_missingness" for name in FUNDAMENTAL_FIELDS],
        ]:
            if col in cand.columns:
                cand[col] = pd.to_numeric(cand[col], errors="coerce")

        cand["has_price_up_to_rebalance"] = cand["price_obs_count"].fillna(0).ge(1)
        cand["has_fund_up_to_rebalance"] = cand["fund_obs_count"].fillna(0).ge(1)
        cand["is_bootstrap_2006"] = year == 2006

        cand["pass_min_quarters"] = cand["fund_obs_count"].fillna(0).ge(4) | cand["is_bootstrap_2006"]
        cand["pass_history_252d"] = cand["price_obs_count"].fillna(0).ge(252) | cand["is_bootstrap_2006"]
        cand["pass_price_floor"] = cand["latest_adj_close"].ge(min_price)
        cand["pass_liquidity"] = cand["addv_lookback_mean"].ge(addv_min)

        missing_pass_cols: list[str] = []
        for name in FUNDAMENTAL_FIELDS:
            miss_col = f"{name}_missingness"
            pass_col = f"pass_missingness_{name}"
            cand[pass_col] = cand[miss_col].le(missingness_max)
            missing_pass_cols.append(pass_col)
        cand["pass_missingness_all"] = cand[missing_pass_cols].all(axis=1)

        cand["pre_options_pass"] = (
            cand["has_price_up_to_rebalance"]
            & cand["has_fund_up_to_rebalance"]
            & cand["pass_min_quarters"]
            & cand["pass_history_252d"]
            & cand["pass_price_floor"]
            & cand["pass_liquidity"]
            & cand["pass_missingness_all"]
        )

        survivors = cand[cand["pre_options_pass"]].copy()
        survivors = survivors.sort_values(["mkt_val", "ticker"], ascending=[False, True]).reset_index(drop=True)
        survivors["pre_options_rank"] = np.arange(1, len(survivors) + 1)

        cand = cand.merge(survivors[["ticker", "pre_options_rank"]], on="ticker", how="left")
        cand["pre_options_rank"] = cand["pre_options_rank"].astype("Int64")
        cand["in_pre_options_buffer"] = cand["pre_options_pass"] & cand["pre_options_rank"].le(buffer_cutoff)

        rows.append(cand)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(["year", "pre_options_pass", "mkt_val", "ticker"], ascending=[True, False, False, True])
    return out.reset_index(drop=True)


def finalize_annual_universe_with_options(
    candidates_df: pd.DataFrame,
    options_df: pd.DataFrame,
    trading_calendar_df: pd.DataFrame,
    target_size: int = 30,
    window_days: int = 5,
) -> pd.DataFrame:
    """Finalize annual universe using options availability around rebalance dates."""
    required = ["year", "rebalance_date", "ticker", "mkt_val", "pre_options_pass", "in_pre_options_buffer"]
    _require_columns(candidates_df, required, "candidates_df")
    _require_columns(trading_calendar_df, ["date"], "trading_calendar_df")
    _require_columns(options_df, ["ticker", "date"], "options_df")

    cand = candidates_df.copy()
    cand["rebalance_date"] = pd.to_datetime(cand["rebalance_date"], errors="coerce")
    cand["ticker"] = _normalize_ticker(cand["ticker"])
    cand = cand.dropna(subset=["year", "rebalance_date", "ticker"])

    if "pre_options_rank" not in cand.columns:
        survivors = cand[cand["pre_options_pass"]].copy()
        survivors = survivors.sort_values(["year", "mkt_val", "ticker"], ascending=[True, False, True])
        survivors["pre_options_rank"] = survivors.groupby("year").cumcount() + 1
        cand = cand.merge(survivors[["year", "ticker", "pre_options_rank"]], on=["year", "ticker"], how="left")

    calendar = pd.to_datetime(trading_calendar_df["date"], errors="coerce").dropna().dt.normalize()
    trading_dates = pd.DatetimeIndex(sorted(pd.Series(calendar).drop_duplicates().tolist()))
    if trading_dates.empty:
        raise ValueError("trading_calendar_df does not contain valid dates")

    options = options_df.copy()
    options["ticker"] = _normalize_ticker(options["ticker"])
    options["date"] = pd.to_datetime(options["date"], errors="coerce").dt.normalize()
    options = options.dropna(subset=["ticker", "date"]).drop_duplicates(["ticker", "date"])

    selected_rows: list[pd.DataFrame] = []
    for year, group in cand.groupby("year", sort=True):
        survivors = group[group["pre_options_pass"]].copy()
        if survivors.empty:
            continue
        survivors = survivors.sort_values(["pre_options_rank", "mkt_val", "ticker"], ascending=[True, False, True])
        rebalance_date = pd.Timestamp(survivors["rebalance_date"].iloc[0]).normalize()

        pos = int(trading_dates.searchsorted(rebalance_date, side="left"))
        if pos >= len(trading_dates):
            pos = len(trading_dates) - 1
        if pos > 0 and trading_dates[pos] > rebalance_date and rebalance_date not in trading_dates:
            pos -= 1

        lo = max(0, pos - int(window_days))
        hi = min(len(trading_dates), pos + int(window_days) + 1)
        window = set(pd.DatetimeIndex(trading_dates[lo:hi]))

        has_option_window = (
            options[options["date"].isin(window)]["ticker"].drop_duplicates().astype(str).tolist()
        )
        has_option_set = set(has_option_window)
        survivors["has_options_rebalance_window"] = survivors["ticker"].isin(has_option_set)

        primary = survivors[survivors["in_pre_options_buffer"] & survivors["has_options_rebalance_window"]].copy()
        primary = primary.head(target_size)

        if len(primary) < target_size:
            needed = target_size - len(primary)
            fallback = survivors[
                (~survivors["in_pre_options_buffer"]) & survivors["has_options_rebalance_window"]
            ].copy()
            fallback = fallback.head(needed)
            selected = pd.concat([primary, fallback], ignore_index=True)
        else:
            selected = primary

        if selected.empty:
            continue

        selected = selected.sort_values(["pre_options_rank", "mkt_val", "ticker"], ascending=[True, False, True])
        selected["selection_rank"] = np.arange(1, len(selected) + 1)
        selected["selection_year"] = int(year)
        selected_rows.append(selected)

    if not selected_rows:
        return pd.DataFrame(
            columns=[
                "selection_year",
                "rebalance_date",
                "ticker",
                "mkt_val",
                "pre_options_rank",
                "in_pre_options_buffer",
                "has_options_rebalance_window",
                "selection_rank",
            ]
        )

    out = pd.concat(selected_rows, ignore_index=True)
    out = out[
        [
            "selection_year",
            "rebalance_date",
            "ticker",
            "mkt_val",
            "pre_options_rank",
            "in_pre_options_buffer",
            "has_options_rebalance_window",
            "selection_rank",
        ]
    ]
    out = out.sort_values(["selection_year", "selection_rank", "ticker"]).reset_index(drop=True)
    return out


def expand_annual_membership_to_daily(
    annual_universe_df: pd.DataFrame,
    trading_calendar_df: pd.DataFrame,
) -> pd.DataFrame:
    """Expand annual universe membership into daily rows between rebalances."""
    _require_columns(annual_universe_df, ["rebalance_date", "ticker"], "annual_universe_df")
    _require_columns(trading_calendar_df, ["date"], "trading_calendar_df")

    annual = annual_universe_df.copy()
    annual["rebalance_date"] = pd.to_datetime(annual["rebalance_date"], errors="coerce").dt.normalize()
    annual["ticker"] = _normalize_ticker(annual["ticker"])
    annual = annual.dropna(subset=["rebalance_date", "ticker"]).drop_duplicates(["rebalance_date", "ticker"])

    calendar = pd.to_datetime(trading_calendar_df["date"], errors="coerce").dropna().dt.normalize()
    trading_dates = pd.DatetimeIndex(sorted(pd.Series(calendar).drop_duplicates().tolist()))
    if trading_dates.empty:
        raise ValueError("trading_calendar_df does not contain valid dates")

    rebalance_dates = sorted(annual["rebalance_date"].dropna().unique().tolist())
    frames: list[pd.DataFrame] = []

    for idx, start in enumerate(rebalance_dates):
        end = rebalance_dates[idx + 1] if idx + 1 < len(rebalance_dates) else None
        members = np.sort(annual.loc[annual["rebalance_date"] == start, "ticker"].dropna().unique())
        if len(members) == 0:
            continue

        if end is None:
            period_dates = trading_dates[trading_dates >= start]
        else:
            period_dates = trading_dates[(trading_dates >= start) & (trading_dates < end)]

        if len(period_dates) == 0:
            continue

        date_arr = np.repeat(period_dates.to_numpy(), len(members))
        ticker_arr = np.tile(members, len(period_dates))
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(date_arr),
                "ticker": ticker_arr,
                "rebalance_date": pd.Timestamp(start),
                "in_universe": True,
            }
        )
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "rebalance_date", "in_universe"])

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["date", "ticker"]).reset_index(drop=True)
    return out


def attach_universe_flags(
    panel_df: pd.DataFrame,
    daily_membership_df: pd.DataFrame,
    options_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach in-universe and options-availability flags to a panel."""
    _require_columns(panel_df, ["ticker", "date"], "panel_df")
    _require_columns(daily_membership_df, ["ticker", "date"], "daily_membership_df")
    _require_columns(options_df, ["ticker", "date"], "options_df")

    out = panel_df.copy()
    out["ticker"] = _normalize_ticker(out["ticker"])
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()

    membership = daily_membership_df.copy()
    membership["ticker"] = _normalize_ticker(membership["ticker"])
    membership["date"] = pd.to_datetime(membership["date"], errors="coerce").dt.normalize()
    membership = membership.dropna(subset=["ticker", "date"]).drop_duplicates(["ticker", "date"])
    membership["in_universe"] = True

    opt = options_df.copy()
    opt["ticker"] = _normalize_ticker(opt["ticker"])
    opt["date"] = pd.to_datetime(opt["date"], errors="coerce").dt.normalize()
    opt = opt.dropna(subset=["ticker", "date"]).drop_duplicates(["ticker", "date"])
    opt["has_option_today"] = True

    out = out.merge(membership[["ticker", "date", "in_universe"]], on=["ticker", "date"], how="left")
    out = out.merge(opt[["ticker", "date", "has_option_today"]], on=["ticker", "date"], how="left")
    out["in_universe"] = out["in_universe"].eq(True)
    out["has_option_today"] = out["has_option_today"].eq(True)
    out["tradable_today"] = out["in_universe"] & out["has_option_today"]
    return out
