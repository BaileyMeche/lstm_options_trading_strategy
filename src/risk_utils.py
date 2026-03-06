
# Stage 10 — Risk Management Framework
# Input: realized backtest outputs (trade_log, position_log, daily_pnl)
# Design: all metrics derived from executed positions, never raw signals.
# Contract size: 100 shares per option contract.
# Timing: consistent with Option A convention (signal after close t,
#          execution at open t+1, PnL realized open(t+1)->close(t+1)).

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

CONTRACT_SIZE: int = 100
RISK_FREE_ANNUAL: float = 0.04
PERIODS_PER_YEAR: int = 252


def _ensure_datetime(df: pd.DataFrame | None, col: str = "date") -> pd.DataFrame | None:
    if df is None:
        return None
    out = df.copy()
    if col in out.columns:
        out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def _first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _safe_divide(a: pd.Series, b: pd.Series | float) -> pd.Series:
    out = a.astype(float) / b
    return out.replace([np.inf, -np.inf], np.nan)


def _infer_contracts_series(df: pd.DataFrame) -> pd.Series:
    for col in ["contracts", "num_contracts", "n_contracts", "quantity", "qty"]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").abs()
            return s.fillna(1.0).clip(lower=1.0)

    if {"stock_position", "delta"}.issubset(df.columns):
        stock_pos = pd.to_numeric(df["stock_position"], errors="coerce").abs()
        delta = pd.to_numeric(df["delta"], errors="coerce").abs()
        denom = (delta * CONTRACT_SIZE).replace(0, np.nan)
        inferred = _safe_divide(stock_pos, denom).round()
        if inferred.notna().any():
            return inferred.fillna(1.0).clip(lower=1.0)

    return pd.Series(1.0, index=df.index, dtype=float)


def _coerce_beta_lookup(beta_lookup: dict[str, float] | None) -> dict[str, float]:
    if beta_lookup is None:
        return {}
    return {str(k): float(v) for k, v in beta_lookup.items() if pd.notna(v)}


# ---------------------------------------------------------------------------
# Equity curve
# ---------------------------------------------------------------------------


def build_equity_curve(
    daily_pnl_df: pd.DataFrame | None,
    trade_log: pd.DataFrame,
    initial_capital: float = 100_000.0,
) -> pd.Series:
    """Build cumulative equity curve from daily PnL DataFrame or trade_log exits."""
    if (
        daily_pnl_df is not None
        and "date" in daily_pnl_df.columns
        and "daily_pnl" in daily_pnl_df.columns
    ):
        df = daily_pnl_df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date").sort_index()
        equity = (initial_capital + df["daily_pnl"].fillna(0.0).cumsum()).rename("equity")
        source = "daily_pnl_df"
        n = len(equity)
    else:
        exits = trade_log[trade_log["action"] == "exit"].copy()
        exits["date"] = pd.to_datetime(exits["date"], errors="coerce")
        exits = exits.dropna(subset=["date"])

        if exits.empty and "date" in trade_log.columns:
            all_dates = pd.to_datetime(trade_log["date"], errors="coerce").dropna()
            if all_dates.empty:
                return pd.Series([initial_capital], index=[pd.Timestamp.today().normalize()], name="equity")
            start = all_dates.min()
            end = all_dates.max()
            cal = pd.bdate_range(start=start, end=end)
            equity = pd.Series(initial_capital, index=cal, name="equity")
            source = "trade_log (no exits available)"
            n = len(equity)
        else:
            all_dates = pd.to_datetime(trade_log["date"], errors="coerce").dropna()
            start = all_dates.min()
            end = all_dates.max()
            cal = pd.bdate_range(start=start, end=end)

            pnl_col = "realized_pnl" if "realized_pnl" in exits.columns else exits.columns[-1]
            daily_pnl = exits.groupby("date")[pnl_col].sum()
            daily_pnl = daily_pnl.reindex(cal, fill_value=0.0)
            equity = (initial_capital + daily_pnl.cumsum()).rename("equity")
            source = "trade_log (exit realized_pnl)"
            n = len(equity)

    print(
        f"[Stage 10] Equity curve built from {source}: "
        f"{n} days, initial=${initial_capital:,.0f}"
    )
    return equity


# ---------------------------------------------------------------------------
# Drawdown statistics
# ---------------------------------------------------------------------------


def compute_drawdown_stats(equity: pd.Series) -> dict[str, Any]:
    """Compute comprehensive drawdown statistics from an equity curve."""
    cum_max = equity.cummax()
    drawdown = ((equity - cum_max) / cum_max).rename("drawdown")

    max_drawdown = float(drawdown.min())
    max_drawdown_date = pd.Timestamp(drawdown.idxmin())

    peak_value = float(cum_max.loc[max_drawdown_date])
    before_trough = equity.loc[:max_drawdown_date]
    cum_max_before = cum_max.loc[:max_drawdown_date]
    at_peak = before_trough == cum_max_before
    drawdown_start_date = (
        pd.Timestamp(at_peak[at_peak].index[-1])
        if at_peak.any()
        else pd.Timestamp(equity.index[0])
    )

    trough_loc = equity.index.get_loc(max_drawdown_date)
    after_trough = equity.iloc[trough_loc + 1 :]
    recovered = after_trough[after_trough >= peak_value * (1 - 1e-9)]
    if not recovered.empty:
        recovery_date: pd.Timestamp | None = pd.Timestamp(recovered.index[0])
        recovery_days: int | None = int((recovery_date - max_drawdown_date).days)
    else:
        recovery_date = None
        recovery_days = None

    in_dd = (drawdown < -0.01).astype(int)
    transitions = in_dd.diff()
    n_drawdown_periods = int((transitions == 1).sum())
    if drawdown.iloc[0] < -0.01:
        n_drawdown_periods += 1

    return {
        "drawdown_series": drawdown,
        "max_drawdown": max_drawdown,
        "max_drawdown_date": max_drawdown_date,
        "drawdown_start_date": drawdown_start_date,
        "recovery_date": recovery_date,
        "recovery_days": recovery_days,
        "n_drawdown_periods": n_drawdown_periods,
    }


# ---------------------------------------------------------------------------
# Rolling risk metrics
# ---------------------------------------------------------------------------


def compute_rolling_risk(
    equity: pd.Series,
    window: int = 21,
    min_periods: int = 5,
) -> pd.DataFrame:
    """Compute rolling Sharpe, Sortino, and annualized volatility."""
    rf_daily = float((1 + RISK_FREE_ANNUAL) ** (1 / PERIODS_PER_YEAR) - 1)
    daily_ret = equity.pct_change()

    rolling_vol = (
        daily_ret.rolling(window=window, min_periods=min_periods).std()
        * np.sqrt(PERIODS_PER_YEAR)
    )

    excess = daily_ret - rf_daily
    rolling_mean_excess = excess.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = daily_ret.rolling(window=window, min_periods=min_periods).std()
    rolling_sharpe = (rolling_mean_excess / rolling_std) * np.sqrt(PERIODS_PER_YEAR)

    def _sortino_func(arr: np.ndarray) -> float:
        excess_arr = arr - rf_daily
        mean_exc = float(excess_arr.mean())
        downside = excess_arr[excess_arr < 0]
        if len(downside) < 2:
            return np.nan
        ds = float(downside.std())
        if ds == 0.0 or np.isnan(ds):
            return np.nan
        return mean_exc / ds * np.sqrt(PERIODS_PER_YEAR)

    rolling_sortino = daily_ret.rolling(window=window, min_periods=min_periods).apply(
        _sortino_func, raw=True
    )

    return (
        pd.DataFrame(
            {
                "rolling_sharpe_21d": rolling_sharpe,
                "rolling_sortino_21d": rolling_sortino,
                "rolling_vol_21d": rolling_vol,
            },
            index=equity.index,
        )
        .replace([np.inf, -np.inf], np.nan)
    )


# ---------------------------------------------------------------------------
# VaR and CVaR
# ---------------------------------------------------------------------------


def compute_var_cvar(
    equity: pd.Series,
    confidence_levels: list[float] | None = None,
) -> pd.DataFrame:
    """Compute historical simulation VaR and CVaR (Expected Shortfall)."""
    if confidence_levels is None:
        confidence_levels = [0.001, 0.01, 0.05, 0.10]

    returns = equity.pct_change().dropna().values
    initial_equity = float(equity.iloc[0])

    if len(returns) < 50:
        warnings.warn(
            f"[Stage 10] Only {len(returns)} return observations available. "
            "Historical VaR is unreliable on short samples (recommend >= 50).",
            stacklevel=2,
        )

    rows: list[dict[str, float]] = []
    for alpha in confidence_levels:
        var_pct = float(np.quantile(returns, alpha))
        tail = returns[returns <= var_pct]
        cvar_pct = float(tail.mean()) if len(tail) > 0 else var_pct
        cvar_pct = min(cvar_pct, var_pct)
        rows.append(
            {
                "confidence_level": alpha,
                "var_pct": var_pct,
                "var_dollar": var_pct * initial_equity,
                "cvar_pct": cvar_pct,
                "cvar_dollar": cvar_pct * initial_equity,
            }
        )

    df = pd.DataFrame(rows)
    assert (df["cvar_pct"] <= df["var_pct"] + 1e-9).all(), (
        "CVaR invariant violated: CVaR must be <= VaR (more extreme tail)."
    )
    return df


def compute_stress_scenarios(
    position_log: pd.DataFrame | None,
    equity: pd.Series,
    spot_shocks: list[float] | None = None,
    vol_shocks: list[float] | None = None,
) -> pd.DataFrame:
    """Compute simple option-aware stress scenarios from active marked positions.

    This is a first-order approximation:
      stressed_pnl ≈ dollar_delta * spot_shock + dollar_vega * vol_shock
    using latest active marked positions if available.
    """
    if spot_shocks is None:
        spot_shocks = [-0.03, -0.05, 0.03]
    if vol_shocks is None:
        vol_shocks = [0.0, 0.10]

    if position_log is None or position_log.empty or "date" not in position_log.columns:
        return pd.DataFrame(
            [{"scenario": "portfolio_equity_only", "stress_pnl": np.nan, "stress_pct_equity": np.nan}]
        )

    pos = _ensure_datetime(position_log, "date")
    latest_date = pos["date"].max()
    active = pos[pos["date"] == latest_date].copy()

    if active.empty:
        return pd.DataFrame(
            [{"scenario": "no_active_positions", "stress_pnl": 0.0, "stress_pct_equity": 0.0}]
        )

    option_price_col = _first_existing(active, ["option_price", "option_mid", "option_mark", "option_market_price"])
    stock_price_col = _first_existing(active, ["stock_price", "underlying_price", "adj_close", "close"])
    delta_col = _first_existing(active, ["delta"])
    vega_col = _first_existing(active, ["vega", "option_vega"])
    contracts = _infer_contracts_series(active)

    if option_price_col is None and stock_price_col is None:
        return pd.DataFrame(
            [{"scenario": "insufficient_position_mark_data", "stress_pnl": np.nan, "stress_pct_equity": np.nan}]
        )

    active["stock_price_eff"] = pd.to_numeric(active.get(stock_price_col, np.nan), errors="coerce")
    active["delta_eff"] = pd.to_numeric(active.get(delta_col, np.nan), errors="coerce").fillna(0.0)
    active["stock_position_eff"] = pd.to_numeric(active.get("stock_position", 0.0), errors="coerce").fillna(0.0)
    active["dollar_delta"] = (
        active["delta_eff"] * contracts * CONTRACT_SIZE * active["stock_price_eff"]
        + active["stock_position_eff"] * active["stock_price_eff"]
    ).fillna(0.0)

    if vega_col is not None:
        active["dollar_vega"] = (
            pd.to_numeric(active[vega_col], errors="coerce").fillna(0.0)
            * contracts * CONTRACT_SIZE
        )
    else:
        active["dollar_vega"] = 0.0

    latest_equity = float(equity.iloc[-1])
    rows = []
    for s in spot_shocks:
        for v in vol_shocks:
            pnl = float((active["dollar_delta"] * s + active["dollar_vega"] * v).sum())
            rows.append(
                {
                    "scenario": f"spot_{s:+.0%}_vol_{v:+.0%}",
                    "spot_shock": s,
                    "vol_shock": v,
                    "stress_pnl": pnl,
                    "stress_pct_equity": pnl / latest_equity if latest_equity else np.nan,
                    "as_of_date": latest_date,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Trade-level risk statistics
# ---------------------------------------------------------------------------


def compute_trade_risk_stats(trade_log: pd.DataFrame) -> dict[str, Any]:
    """Compute trade-level risk statistics from the trade log."""
    entries = trade_log[trade_log["action"] == "enter"].copy()
    exits = trade_log[trade_log["action"] == "exit"].copy()

    n_entries = len(entries)
    n_exits = len(exits)

    reason_series = exits["exit_reason"].dropna() if "exit_reason" in exits.columns else pd.Series(dtype=object)
    reason_counts = reason_series.value_counts()
    exit_reason_counts = reason_counts.to_dict()
    exit_reason_pct = {k: float(v / n_exits) for k, v in exit_reason_counts.items()} if n_exits > 0 else {}

    pnl = exits["realized_pnl"].dropna() if "realized_pnl" in exits.columns else pd.Series(dtype=float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]

    win_rate = float(len(wins) / n_exits) if n_exits > 0 else float("nan")
    avg_win_pnl = float(wins.mean()) if len(wins) > 0 else float("nan")
    avg_loss_pnl = float(losses.mean()) if len(losses) > 0 else float("nan")
    reward_risk_ratio = (
        float(abs(avg_win_pnl) / abs(avg_loss_pnl))
        if not np.isnan(avg_win_pnl) and not np.isnan(avg_loss_pnl) and avg_loss_pnl != 0
        else float("nan")
    )

    avg_entry_cost = (
        float(entries["entry_cost"].dropna().mean())
        if "entry_cost" in entries.columns
        else float("nan")
    )

    dh = exits["days_held"].dropna() if "days_held" in exits.columns else pd.Series(dtype=float)
    avg_days_held = float(dh.mean()) if len(dh) > 0 else float("nan")
    p25_days_held = float(dh.quantile(0.25)) if len(dh) > 0 else float("nan")
    p50_days_held = float(dh.quantile(0.50)) if len(dh) > 0 else float("nan")
    p75_days_held = float(dh.quantile(0.75)) if len(dh) > 0 else float("nan")
    p95_days_held = float(dh.quantile(0.95)) if len(dh) > 0 else float("nan")

    sl_exits = exits[exits["exit_reason"] == "stop_loss"] if "exit_reason" in exits.columns else pd.DataFrame()
    n_stop_loss = len(sl_exits)
    stop_loss_rate = float(n_stop_loss / n_exits) if n_exits > 0 else float("nan")
    avg_stop_loss_pnl = float(sl_exits["realized_pnl"].mean()) if len(sl_exits) > 0 and "realized_pnl" in sl_exits.columns else float("nan")

    total_realized_pnl = float(pnl.sum()) if len(pnl) > 0 else 0.0
    mean_realized_pnl = float(pnl.mean()) if len(pnl) > 0 else float("nan")

    return {
        "n_entries": n_entries,
        "n_exits": n_exits,
        "n_trades": n_exits,
        "exit_reason_counts": exit_reason_counts,
        "exit_reason_pct": exit_reason_pct,
        "win_rate": win_rate,
        "avg_win_pnl": avg_win_pnl,
        "avg_loss_pnl": avg_loss_pnl,
        "reward_risk_ratio": reward_risk_ratio,
        "avg_entry_cost": avg_entry_cost,
        "avg_days_held": avg_days_held,
        "p25_days_held": p25_days_held,
        "p50_days_held": p50_days_held,
        "p75_days_held": p75_days_held,
        "p95_days_held": p95_days_held,
        "n_stop_loss": n_stop_loss,
        "stop_loss_rate": stop_loss_rate,
        "avg_stop_loss_pnl": avg_stop_loss_pnl,
        "total_realized_pnl": total_realized_pnl,
        "mean_realized_pnl": mean_realized_pnl,
    }


# ---------------------------------------------------------------------------
# Exposure helpers
# ---------------------------------------------------------------------------


def _expand_positions_to_daily(trade_log: pd.DataFrame) -> pd.DataFrame:
    """Expand each enter-exit pair into a per-(date, position) panel."""
    enters = trade_log[trade_log["action"] == "enter"].copy()
    exits = trade_log[trade_log["action"] == "exit"].copy()
    enters["date"] = pd.to_datetime(enters["date"], errors="coerce")
    exits["date"] = pd.to_datetime(exits["date"], errors="coerce")

    enters = enters.dropna(subset=["date"]).sort_values(["ticker", "date"]).reset_index(drop=True)
    exits = exits.dropna(subset=["date"]).sort_values(["ticker", "date"]).reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    for ticker in enters["ticker"].dropna().unique():
        t_enters = enters[enters["ticker"] == ticker].reset_index(drop=True)
        t_exits = exits[exits["ticker"] == ticker].reset_index(drop=True)
        n = min(len(t_enters), len(t_exits))
        for i in range(n):
            en = t_enters.iloc[i]
            ex = t_exits.iloc[i]
            entry_date = pd.Timestamp(en["date"])
            exit_date = pd.Timestamp(ex["date"])

            opt_price = float(pd.to_numeric(en.get("option_price", np.nan), errors="coerce"))
            stk_price = float(pd.to_numeric(en.get("stock_price", np.nan), errors="coerce"))
            delta = float(pd.to_numeric(en.get("delta", 0.0), errors="coerce"))
            stock_pos = float(pd.to_numeric(en.get("stock_position", 0.0), errors="coerce"))
            contracts = float(pd.to_numeric(en.get("contracts", en.get("num_contracts", 1.0)), errors="coerce"))
            if pd.isna(contracts) or contracts <= 0:
                contracts = 1.0

            long_opt_val = opt_price * CONTRACT_SIZE * contracts
            short_stk_val = abs(stock_pos) * stk_price
            gross_exp = long_opt_val + short_stk_val
            net_exp = long_opt_val - short_stk_val
            dollar_delta = delta * CONTRACT_SIZE * contracts * stk_price + stock_pos * stk_price
            net_delta_frac = delta * contracts + stock_pos / CONTRACT_SIZE
            gross_delta_frac = abs(delta * contracts) + abs(stock_pos) / CONTRACT_SIZE

            for d in pd.bdate_range(start=entry_date, end=exit_date):
                rows.append(
                    {
                        "date": d,
                        "ticker": ticker,
                        "long_option_value": long_opt_val,
                        "short_stock_value": short_stk_val,
                        "gross_exposure": gross_exp,
                        "net_exposure": net_exp,
                        "dollar_delta": dollar_delta,
                        "net_delta_frac": net_delta_frac,
                        "gross_delta_frac": gross_delta_frac,
                    }
                )

    cols = [
        "date", "ticker", "long_option_value", "short_stock_value",
        "gross_exposure", "net_exposure", "dollar_delta",
        "net_delta_frac", "gross_delta_frac",
    ]
    return pd.DataFrame(rows, columns=cols)


def _prepare_position_log_panel(position_log: pd.DataFrame) -> pd.DataFrame:
    pos = _ensure_datetime(position_log, "date").copy()

    option_price_col = _first_existing(pos, ["option_price", "option_mid", "option_mark", "option_market_price"])
    stock_price_col = _first_existing(pos, ["stock_price", "underlying_price", "adj_close", "close"])
    delta_col = _first_existing(pos, ["delta"])
    gamma_col = _first_existing(pos, ["gamma", "option_gamma"])
    vega_col = _first_existing(pos, ["vega", "option_vega"])
    theta_col = _first_existing(pos, ["theta", "option_theta"])
    position_id_col = _first_existing(pos, ["position_id", "trade_id", "option_id"])

    def _series_or_default(df: pd.DataFrame, col: str | None, default: float = 0.0) -> pd.Series:
        if col is not None and col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
        return pd.Series(default, index=df.index, dtype=float)

    pos["contracts_eff"] = _infer_contracts_series(pos)
    pos["stock_price_eff"] = _series_or_default(pos, stock_price_col, np.nan)
    pos["option_price_eff"] = _series_or_default(pos, option_price_col, np.nan)
    pos["delta_eff"] = _series_or_default(pos, delta_col, 0.0).fillna(0.0)
    pos["gamma_eff"] = _series_or_default(pos, gamma_col, 0.0).fillna(0.0)
    pos["vega_eff"] = _series_or_default(pos, vega_col, 0.0).fillna(0.0)
    pos["theta_eff"] = _series_or_default(pos, theta_col, 0.0).fillna(0.0)
    pos["stock_position_eff"] = _series_or_default(pos, "stock_position", 0.0).fillna(0.0)

    if "option_position" in pos.columns:
        option_pos = pd.to_numeric(pos["option_position"], errors="coerce").fillna(1.0)
    else:
        option_pos = pd.Series(1.0, index=pos.index, dtype=float)

    # If marked values already exist, respect them.
    opt_value_col = _first_existing(pos, ["option_market_value", "option_value", "long_option_value"])
    stock_value_col = _first_existing(pos, ["stock_market_value", "short_stock_value"])

    if opt_value_col is not None:
        pos["long_option_value"] = pd.to_numeric(pos[opt_value_col], errors="coerce").abs().fillna(0.0)
    else:
        pos["long_option_value"] = (
            pos["option_price_eff"].fillna(0.0)
            * CONTRACT_SIZE
            * pos["contracts_eff"]
            * option_pos.abs()
        )

    if stock_value_col is not None:
        pos["short_stock_value"] = pd.to_numeric(pos[stock_value_col], errors="coerce").abs().fillna(0.0)
    else:
        pos["short_stock_value"] = (pos["stock_position_eff"].abs() * pos["stock_price_eff"].fillna(0.0))

    pos["gross_exposure"] = pos["long_option_value"] + pos["short_stock_value"]
    pos["net_exposure"] = pos["long_option_value"] - pos["short_stock_value"]

    pos["option_dollar_delta"] = (
        pos["delta_eff"] * CONTRACT_SIZE * pos["contracts_eff"] * option_pos * pos["stock_price_eff"].fillna(0.0)
    )
    pos["stock_dollar_delta"] = pos["stock_position_eff"] * pos["stock_price_eff"].fillna(0.0)
    pos["dollar_delta"] = pos["option_dollar_delta"] + pos["stock_dollar_delta"]

    pos["option_dollar_gamma"] = (
        pos["gamma_eff"] * CONTRACT_SIZE * pos["contracts_eff"] * option_pos * (pos["stock_price_eff"].fillna(0.0) ** 2)
    )
    pos["option_dollar_vega"] = pos["vega_eff"] * CONTRACT_SIZE * pos["contracts_eff"] * option_pos
    pos["option_dollar_theta"] = pos["theta_eff"] * CONTRACT_SIZE * pos["contracts_eff"] * option_pos

    if position_id_col is not None:
        pos["position_key"] = pos[position_id_col].astype(str)
    else:
        pos["position_key"] = (
            pos.get("ticker", pd.Series("UNK", index=pos.index)).astype(str)
            + "|"
            + pos["date"].astype(str)
            + "|"
            + pd.Series(np.arange(len(pos)), index=pos.index).astype(str)
        )

    if "ticker" not in pos.columns:
        pos["ticker"] = "UNKNOWN"

    return pos


# ---------------------------------------------------------------------------
# Daily exposure
# ---------------------------------------------------------------------------


def compute_exposure_from_trade_log(trade_log: pd.DataFrame) -> pd.DataFrame:
    """Derive daily portfolio dollar exposure from trade_log."""
    panel = _expand_positions_to_daily(trade_log)
    if panel.empty:
        return pd.DataFrame(
            columns=[
                "date", "n_positions", "gross_exposure", "net_exposure",
                "net_delta_exposure", "long_option_value", "short_stock_value",
            ]
        )

    return (
        panel.groupby("date")
        .agg(
            n_positions=("ticker", "count"),
            gross_exposure=("gross_exposure", "sum"),
            net_exposure=("net_exposure", "sum"),
            net_delta_exposure=("dollar_delta", "sum"),
            long_option_value=("long_option_value", "sum"),
            short_stock_value=("short_stock_value", "sum"),
        )
        .reset_index()
        .sort_values("date")
        .reset_index(drop=True)
    )


def compute_exposure_from_position_log(position_log: pd.DataFrame) -> pd.DataFrame:
    """Compute marked daily exposures from position_log."""
    pos = _prepare_position_log_panel(position_log)
    if pos.empty:
        return pd.DataFrame(
            columns=[
                "date", "n_positions", "gross_exposure", "net_exposure",
                "net_delta_exposure", "long_option_value", "short_stock_value",
            ]
        )

    return (
        pos.groupby("date")
        .agg(
            n_positions=("position_key", "nunique"),
            gross_exposure=("gross_exposure", "sum"),
            net_exposure=("net_exposure", "sum"),
            net_delta_exposure=("dollar_delta", "sum"),
            long_option_value=("long_option_value", "sum"),
            short_stock_value=("short_stock_value", "sum"),
        )
        .reset_index()
        .sort_values("date")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Concentration
# ---------------------------------------------------------------------------


def compute_concentration_from_trade_log(trade_log: pd.DataFrame) -> pd.DataFrame:
    panel = _expand_positions_to_daily(trade_log)
    if panel.empty:
        return pd.DataFrame(columns=["date", "hhi", "max_weight_pct", "max_weight_ticker"])

    panel["weight"] = panel.groupby("date")["long_option_value"].transform(
        lambda s: s / s.sum() if s.sum() > 0 else np.nan
    )

    out = (
        panel.groupby("date")
        .apply(
            lambda g: pd.Series(
                {
                    "hhi": float((g["weight"].fillna(0.0) ** 2).sum()),
                    "max_weight_pct": float(g["weight"].max()),
                    "max_weight_ticker": g.loc[g["weight"].idxmax(), "ticker"] if g["weight"].notna().any() else np.nan,
                }
            )
        )
        .reset_index()
        .sort_values("date")
        .reset_index(drop=True)
    )
    return out


def compute_concentration_from_position_log(
    position_log: pd.DataFrame,
    sector_lookup: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute ticker and sector concentration from marked positions."""
    pos = _prepare_position_log_panel(position_log)
    if pos.empty:
        return (
            pd.DataFrame(columns=["date", "hhi", "max_weight_pct", "max_weight_ticker"]),
            pd.DataFrame(columns=["date", "sector_hhi", "max_sector_weight_pct", "max_weight_sector"]),
        )

    pos["weight"] = pos.groupby("date")["long_option_value"].transform(
        lambda s: s / s.sum() if s.sum() > 0 else np.nan
    )

    ticker_conc = (
        pos.groupby("date")
        .apply(
            lambda g: pd.Series(
                {
                    "hhi": float((g["weight"].fillna(0.0) ** 2).sum()),
                    "max_weight_pct": float(g["weight"].max()),
                    "max_weight_ticker": g.loc[g["weight"].idxmax(), "ticker"] if g["weight"].notna().any() else np.nan,
                }
            )
        )
        .reset_index()
        .sort_values("date")
        .reset_index(drop=True)
    )

    if sector_lookup is None:
        return (
            ticker_conc,
            pd.DataFrame(columns=["date", "sector_hhi", "max_sector_weight_pct", "max_weight_sector"]),
        )

    pos["sector"] = pos["ticker"].map(lambda t: sector_lookup.get(str(t), "UNKNOWN"))
    sector_daily = (
        pos.groupby(["date", "sector"], dropna=False)["long_option_value"]
        .sum()
        .reset_index()
    )
    sector_daily["sector_weight"] = sector_daily.groupby("date")["long_option_value"].transform(
        lambda s: s / s.sum() if s.sum() > 0 else np.nan
    )
    sector_conc = (
        sector_daily.groupby("date")
        .apply(
            lambda g: pd.Series(
                {
                    "sector_hhi": float((g["sector_weight"].fillna(0.0) ** 2).sum()),
                    "max_sector_weight_pct": float(g["sector_weight"].max()),
                    "max_weight_sector": g.loc[g["sector_weight"].idxmax(), "sector"] if g["sector_weight"].notna().any() else np.nan,
                }
            )
        )
        .reset_index()
        .sort_values("date")
        .reset_index(drop=True)
    )
    return ticker_conc, sector_conc


# ---------------------------------------------------------------------------
# Greeks and beta exposure
# ---------------------------------------------------------------------------


def compute_beta_exposure_from_trade_log(
    trade_log: pd.DataFrame,
    beta_lookup: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Derive portfolio beta exposure from trade_log."""
    lookup = _coerce_beta_lookup(beta_lookup)
    if beta_lookup is None:
        warnings.warn(
            "[Stage 10] beta_lookup is None — assuming beta=1.0 for all tickers. "
            "Pass a {ticker: beta} dict for accurate market beta exposure estimates.",
            stacklevel=2,
        )

    panel = _expand_positions_to_daily(trade_log)
    if panel.empty:
        return pd.DataFrame(columns=["date", "net_beta_exposure", "gross_beta_exposure", "n_positions"])

    panel["ticker_beta"] = panel["ticker"].map(lambda t: lookup.get(str(t), 1.0))
    panel["net_beta_contrib"] = panel["net_delta_frac"] * panel["ticker_beta"]
    panel["gross_beta_contrib"] = panel["gross_delta_frac"] * panel["ticker_beta"]

    return (
        panel.groupby("date")
        .agg(
            net_beta_exposure=("net_beta_contrib", "sum"),
            gross_beta_exposure=("gross_beta_contrib", "sum"),
            n_positions=("ticker", "count"),
        )
        .reset_index()
        .sort_values("date")
        .reset_index(drop=True)
    )


def compute_beta_exposure_from_position_log(
    position_log: pd.DataFrame,
    beta_lookup: dict[str, float] | None = None,
) -> pd.DataFrame:
    lookup = _coerce_beta_lookup(beta_lookup)
    if beta_lookup is None:
        warnings.warn(
            "[Stage 10] beta_lookup is None — assuming beta=1.0 for all tickers. "
            "Pass a {ticker: beta} dict for accurate market beta exposure estimates.",
            stacklevel=2,
        )

    pos = _prepare_position_log_panel(position_log)
    if pos.empty:
        return pd.DataFrame(columns=["date", "net_beta_exposure", "gross_beta_exposure", "n_positions"])

    pos["ticker_beta"] = pos["ticker"].map(lambda t: lookup.get(str(t), 1.0))
    pos["net_beta_contrib"] = pos["dollar_delta"] * pos["ticker_beta"]
    pos["gross_beta_contrib"] = pos["gross_exposure"] * pos["ticker_beta"]

    return (
        pos.groupby("date")
        .agg(
            net_beta_exposure=("net_beta_contrib", "sum"),
            gross_beta_exposure=("gross_beta_contrib", "sum"),
            n_positions=("position_key", "nunique"),
        )
        .reset_index()
        .sort_values("date")
        .reset_index(drop=True)
    )


def compute_greeks_exposure_from_position_log(position_log: pd.DataFrame) -> pd.DataFrame:
    pos = _prepare_position_log_panel(position_log)
    if pos.empty:
        return pd.DataFrame(columns=["date", "net_dollar_delta", "net_dollar_gamma", "net_dollar_vega", "net_dollar_theta"])

    return (
        pos.groupby("date")
        .agg(
            net_dollar_delta=("dollar_delta", "sum"),
            net_dollar_gamma=("option_dollar_gamma", "sum"),
            net_dollar_vega=("option_dollar_vega", "sum"),
            net_dollar_theta=("option_dollar_theta", "sum"),
        )
        .reset_index()
        .sort_values("date")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Limits and risk events
# ---------------------------------------------------------------------------


def build_limit_flags(
    exposure_df: pd.DataFrame,
    concentration_df: pd.DataFrame | None,
    beta_exposure_df: pd.DataFrame | None,
    equity: pd.Series,
    sector_concentration_df: pd.DataFrame | None = None,
    max_gross_exposure_pct: float = 0.10,
    max_net_exposure_pct: float = 0.05,
    max_abs_beta_exposure: float = 0.25,
    max_single_name_weight: float = 0.50,
    max_sector_weight: float = 0.60,
) -> pd.DataFrame:
    """Build normalized exposures and daily risk limit flags."""
    spine = equity.rename_axis("date").reset_index()
    spine.columns = ["date", "equity"]
    out = spine.copy()

    if exposure_df is not None and not exposure_df.empty:
        tmp = exposure_df.copy()
        tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
        out = out.merge(tmp, on="date", how="left")
    if concentration_df is not None and not concentration_df.empty:
        tmp = concentration_df.copy()
        tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
        out = out.merge(tmp, on="date", how="left")
    if beta_exposure_df is not None and not beta_exposure_df.empty:
        tmp = beta_exposure_df.copy()
        tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
        out = out.merge(tmp, on="date", how="left")
    if sector_concentration_df is not None and not sector_concentration_df.empty:
        tmp = sector_concentration_df.copy()
        tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
        out = out.merge(tmp, on="date", how="left")

    for col in ["gross_exposure", "net_exposure", "long_option_value", "short_stock_value", "net_beta_exposure", "gross_beta_exposure"]:
        if col not in out.columns:
            out[col] = np.nan

    out["gross_exposure_pct_equity"] = _safe_divide(out["gross_exposure"].abs(), out["equity"])
    out["net_exposure_pct_equity"] = _safe_divide(out["net_exposure"].abs(), out["equity"])
    out["long_option_value_pct_equity"] = _safe_divide(out["long_option_value"].abs(), out["equity"])
    out["short_stock_value_pct_equity"] = _safe_divide(out["short_stock_value"].abs(), out["equity"])
    out["net_beta_exposure_pct_equity"] = _safe_divide(out["net_beta_exposure"].abs(), out["equity"])
    out["gross_beta_exposure_pct_equity"] = _safe_divide(out["gross_beta_exposure"].abs(), out["equity"])

    out["gross_exposure_limit_breach"] = out["gross_exposure_pct_equity"] > max_gross_exposure_pct
    out["net_exposure_limit_breach"] = out["net_exposure_pct_equity"] > max_net_exposure_pct
    out["beta_limit_breach"] = out["net_beta_exposure_pct_equity"] > max_abs_beta_exposure

    if "max_weight_pct" in out.columns:
        out["single_name_limit_breach"] = out["max_weight_pct"] > max_single_name_weight
    else:
        out["single_name_limit_breach"] = False

    if "max_sector_weight_pct" in out.columns:
        out["sector_limit_breach"] = out["max_sector_weight_pct"] > max_sector_weight
    else:
        out["sector_limit_breach"] = False

    return out.sort_values("date").reset_index(drop=True)


def identify_risk_events(
    equity: pd.Series,
    drawdown_threshold: float = -0.10,
    vol_window: int = 21,
    vol_spike_multiplier: float = 2.0,
    limits_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Identify drawdown, volatility, and limit-breach events."""
    daily_ret = equity.pct_change()
    cum_max = equity.cummax()
    drawdown = (equity - cum_max) / cum_max
    rolling_vol = (
        daily_ret.rolling(window=vol_window, min_periods=5).std()
        * np.sqrt(PERIODS_PER_YEAR)
    ).replace([np.inf, -np.inf], np.nan)
    vol_baseline = rolling_vol.expanding(min_periods=vol_window).median()

    rows: list[dict[str, Any]] = []

    in_breach = drawdown < drawdown_threshold
    prev_breach = in_breach.shift(1, fill_value=False)
    new_breach = in_breach & (~prev_breach)
    for date in drawdown.index[new_breach]:
        dd_val = float(drawdown.loc[date])
        rows.append(
            {
                "date": date,
                "trigger_type": "drawdown_breach",
                "value": dd_val,
                "description": (
                    f"Drawdown {dd_val:.2%} breached {drawdown_threshold:.0%}. "
                    "Suggested action: halve gross exposure and freeze new entries until recovery."
                ),
                "suggested_action": "de-risk_50pct_and_freeze_new_entries",
            }
        )

    vol_spike = (
        rolling_vol.notna()
        & vol_baseline.notna()
        & (rolling_vol > vol_spike_multiplier * vol_baseline)
    )
    prev_spike = vol_spike.shift(1, fill_value=False)
    new_spike = vol_spike & (~prev_spike)
    for date in rolling_vol.index[new_spike]:
        vol_val = float(rolling_vol.loc[date])
        rows.append(
            {
                "date": date,
                "trigger_type": "volatility_spike",
                "value": vol_val,
                "description": (
                    f"Vol {vol_val:.2%} exceeded {vol_spike_multiplier:.1f}× trailing median. "
                    "Suggested action: tighten stop-losses and reduce position size."
                ),
                "suggested_action": "tighten_stops_reduce_size",
            }
        )

    if limits_df is not None and not limits_df.empty:
        flag_map = {
            "gross_exposure_limit_breach": "gross_exposure_breach",
            "net_exposure_limit_breach": "net_exposure_breach",
            "beta_limit_breach": "beta_limit_breach",
            "single_name_limit_breach": "single_name_concentration_breach",
            "sector_limit_breach": "sector_concentration_breach",
        }
        tmp = limits_df.copy()
        tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
        tmp = tmp.sort_values("date")
        for flag_col, trig in flag_map.items():
            if flag_col not in tmp.columns:
                continue
            flag = tmp[flag_col].fillna(False).astype(bool)
            new = flag & (~flag.shift(1, fill_value=False))
            for _, row in tmp.loc[new].iterrows():
                value_col = None
                if trig == "gross_exposure_breach":
                    value_col = "gross_exposure_pct_equity"
                elif trig == "net_exposure_breach":
                    value_col = "net_exposure_pct_equity"
                elif trig == "beta_limit_breach":
                    value_col = "net_beta_exposure_pct_equity"
                elif trig == "single_name_concentration_breach":
                    value_col = "max_weight_pct"
                elif trig == "sector_concentration_breach":
                    value_col = "max_sector_weight_pct"
                val = float(row[value_col]) if value_col in row and pd.notna(row[value_col]) else np.nan
                rows.append(
                    {
                        "date": row["date"],
                        "trigger_type": trig,
                        "value": val,
                        "description": f"{trig} detected from daily marked exposures.",
                        "suggested_action": "enforce_limit_reduce_positions",
                    }
                )

    if not rows:
        return pd.DataFrame(columns=["date", "trigger_type", "value", "description", "suggested_action"])

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def compute_all_risk_metrics(
    trade_log: pd.DataFrame,
    position_log: pd.DataFrame | None = None,
    daily_pnl_df: pd.DataFrame | None = None,
    beta_lookup: dict[str, float] | None = None,
    sector_lookup: dict[str, str] | None = None,
    initial_capital: float = 100_000.0,
    drawdown_threshold: float = -0.10,
    var_confidence_levels: list[float] | None = None,
    max_gross_exposure_pct: float = 0.10,
    max_net_exposure_pct: float = 0.05,
    max_abs_beta_exposure: float = 0.25,
    max_single_name_weight: float = 0.50,
    max_sector_weight: float = 0.60,
) -> dict[str, Any]:
    """Orchestrate all Stage 10 risk computations and print a summary."""
    trade_log = _ensure_datetime(trade_log, "date")
    position_log = _ensure_datetime(position_log, "date")
    daily_pnl_df = _ensure_datetime(daily_pnl_df, "date")

    equity = build_equity_curve(daily_pnl_df, trade_log, initial_capital)
    drawdown_stats = compute_drawdown_stats(equity)
    rolling_risk = compute_rolling_risk(equity)
    var_cvar_table = compute_var_cvar(equity, var_confidence_levels)
    trade_risk_stats = compute_trade_risk_stats(trade_log)

    if position_log is not None and not position_log.empty and {"date", "ticker"}.issubset(position_log.columns):
        daily_exposure = compute_exposure_from_position_log(position_log)
        concentration, sector_concentration = compute_concentration_from_position_log(position_log, sector_lookup)
        beta_exposure = compute_beta_exposure_from_position_log(position_log, beta_lookup)
        greeks_exposure = compute_greeks_exposure_from_position_log(position_log)
        exposure_source = "position_log"
    else:
        daily_exposure = compute_exposure_from_trade_log(trade_log)
        concentration = compute_concentration_from_trade_log(trade_log)
        sector_concentration = pd.DataFrame(columns=["date", "sector_hhi", "max_sector_weight_pct", "max_weight_sector"])
        beta_exposure = compute_beta_exposure_from_trade_log(trade_log, beta_lookup)
        greeks_exposure = pd.DataFrame(columns=["date", "net_dollar_delta", "net_dollar_gamma", "net_dollar_vega", "net_dollar_theta"])
        exposure_source = "trade_log_fallback"

    limits_df = build_limit_flags(
        exposure_df=daily_exposure,
        concentration_df=concentration,
        beta_exposure_df=beta_exposure,
        equity=equity,
        sector_concentration_df=sector_concentration,
        max_gross_exposure_pct=max_gross_exposure_pct,
        max_net_exposure_pct=max_net_exposure_pct,
        max_abs_beta_exposure=max_abs_beta_exposure,
        max_single_name_weight=max_single_name_weight,
        max_sector_weight=max_sector_weight,
    )
    risk_events = identify_risk_events(
        equity=equity,
        drawdown_threshold=drawdown_threshold,
        limits_df=limits_df,
    )
    stress_table = compute_stress_scenarios(position_log, equity)

    var_row = var_cvar_table[var_cvar_table["confidence_level"] == 0.01]
    var_1pct = float(var_row["var_pct"].iloc[0]) if len(var_row) > 0 else float("nan")
    cvar_1pct = float(var_row["cvar_pct"].iloc[0]) if len(var_row) > 0 else float("nan")
    median_sharpe = float(rolling_risk["rolling_sharpe_21d"].median())
    median_vol = float(rolling_risk["rolling_vol_21d"].median())
    peak_gross = float(daily_exposure["gross_exposure"].max()) if not daily_exposure.empty else float("nan")

    rec_date_str = str(drawdown_stats["recovery_date"]) if drawdown_stats["recovery_date"] is not None else "Not recovered in sample"
    rec_days_str = str(drawdown_stats["recovery_days"]) if drawdown_stats["recovery_days"] is not None else "N/A"
    rr = trade_risk_stats["reward_risk_ratio"]
    rr_str = f"{rr:.3f}" if not np.isnan(rr) else "N/A"

    print("\n[Stage 10] === Risk Summary ===")
    print(f"  Exposure source:        {exposure_source}")
    print(f"  Max drawdown:           {drawdown_stats['max_drawdown']:.2%}")
    print(f"  Max drawdown date:      {drawdown_stats['max_drawdown_date']}")
    print(f"  Recovery date:          {rec_date_str}")
    print(f"  Recovery days:          {rec_days_str}")
    print(f"  N drawdown periods:     {drawdown_stats['n_drawdown_periods']}")
    print(f"  Median rolling Sharpe:  {median_sharpe:.3f}")
    print(f"  Median rolling vol:     {median_vol:.3f}")
    print(f"  VaR 1% (daily):         {var_1pct:.4%}")
    print(f"  CVaR 1% (daily):        {cvar_1pct:.4%}")
    print(f"  Win rate:               {trade_risk_stats['win_rate']:.2%}")
    print(f"  Reward/risk ratio:      {rr_str}")
    print(f"  Stop-loss rate:         {trade_risk_stats['stop_loss_rate']:.2%}")
    print(f"  Total realized PnL:     ${trade_risk_stats['total_realized_pnl']:,.2f}")
    print(f"  Peak gross exposure:    ${peak_gross:,.0f}")
    print(f"  N risk events flagged:  {len(risk_events)}")

    return {
        "equity": equity,
        "drawdown_stats": drawdown_stats,
        "rolling_risk": rolling_risk,
        "var_cvar_table": var_cvar_table,
        "trade_risk_stats": trade_risk_stats,
        "daily_exposure": daily_exposure,
        "concentration": concentration,
        "sector_concentration": sector_concentration,
        "beta_exposure": beta_exposure,
        "greeks_exposure": greeks_exposure,
        "limits_df": limits_df,
        "risk_events": risk_events,
        "stress_table": stress_table,
        "exposure_source": exposure_source,
    }
