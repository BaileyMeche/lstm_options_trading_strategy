# Stage 13 — Performance Evaluation
# Converts daily P&L into an equity curve and computes standard quant metrics.
# Benchmarks: SPY buy-and-hold and equal-weight universe buy-and-hold.
# All benchmark returns use the same daily return definition (close-to-close)
# and identical evaluation window as the strategy.

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Equity curve
# ---------------------------------------------------------------------------


def equity_curve(
    daily_pnl: pd.Series,
    initial_capital: float = 100_000.0,
) -> pd.Series:
    """Build cumulative equity series from daily P&L.

    Parameters
    ----------
    daily_pnl : Series indexed by date, values in dollars.
    initial_capital : Starting portfolio value.

    Returns
    -------
    pd.Series indexed by date with cumulative equity values.
    """
    return (initial_capital + daily_pnl.cumsum()).rename("equity")


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    eq: pd.Series,
    risk_free_annual: float = 0.04,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Compute standard performance metrics from an equity curve.

    Parameters
    ----------
    eq : pd.Series of equity values indexed by date.
    risk_free_annual : Annualized risk-free rate (default 4%).
    periods_per_year : Trading days per year.

    Returns
    -------
    dict with keys:
        total_return, cagr, ann_volatility, sharpe, sortino,
        calmar, max_drawdown, hit_rate, n_days.
    """
    returns = eq.pct_change().dropna()
    if returns.empty:
        return {k: np.nan for k in [
            "total_return", "cagr", "ann_volatility", "sharpe",
            "sortino", "calmar", "max_drawdown", "hit_rate", "n_days",
        ]}

    rf_daily = (1 + risk_free_annual) ** (1 / periods_per_year) - 1
    excess = returns - rf_daily
    n_days = len(returns)
    n_years = n_days / periods_per_year

    total_return = float(eq.iloc[-1] / eq.iloc[0] - 1)
    cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1 / n_years) - 1) if n_years > 0 else np.nan

    ann_vol = float(returns.std() * np.sqrt(periods_per_year))
    sharpe = (
        float(excess.mean() / returns.std() * np.sqrt(periods_per_year))
        if returns.std() > 0 else np.nan
    )

    downside = returns[returns < rf_daily]
    sortino = (
        float(excess.mean() / downside.std() * np.sqrt(periods_per_year))
        if len(downside) > 1 and downside.std() > 0 else np.nan
    )

    cum_max = eq.cummax()
    drawdown_series = (eq - cum_max) / cum_max
    max_dd = float(drawdown_series.min())
    calmar = float(cagr / abs(max_dd)) if max_dd != 0 else np.nan

    hit_rate = float((returns > 0).sum() / n_days)

    return {
        "total_return": total_return,
        "cagr": cagr,
        "ann_volatility": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": max_dd,
        "hit_rate": hit_rate,
        "n_days": n_days,
    }


def drawdown_series(eq: pd.Series) -> pd.Series:
    """Compute rolling drawdown from peak for an equity curve."""
    cum_max = eq.cummax()
    return ((eq - cum_max) / cum_max).rename("drawdown")


# ---------------------------------------------------------------------------
# Benchmark construction
# ---------------------------------------------------------------------------


def benchmark_equity_curve(
    prices_df: pd.DataFrame,
    tickers: list[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    initial_capital: float = 100_000.0,
    date_col: str = "date",
    price_col: str = "adj_close",
    ticker_col: str = "ticker",
    weight_scheme: str = "equal",
) -> pd.Series:
    """Build a benchmark equity curve for a basket of tickers.

    Parameters
    ----------
    prices_df : Daily price panel with columns [date, ticker, adj_close].
    tickers : List of tickers to include (e.g. universe tickers or ['SPY']).
    start_date, end_date : Evaluation window (inclusive).
    initial_capital : Starting value.
    weight_scheme : 'equal' (equal-weight rebalanced daily) or 'spy' (single ticker).

    Returns
    -------
    pd.Series of daily equity values indexed by date.
    """
    df = prices_df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df[
        (df[ticker_col].isin(tickers))
        & (df[date_col] >= start_date)
        & (df[date_col] <= end_date)
    ].copy()

    if df.empty:
        return pd.Series(dtype=float, name="benchmark_equity")

    # Pivot to wide format: date × ticker
    pivot = df.pivot_table(index=date_col, columns=ticker_col, values=price_col)
    pivot = pivot.sort_index()

    # Daily returns per ticker
    rets = pivot.pct_change()

    # Equal-weight daily portfolio return
    port_ret = rets.mean(axis=1)

    eq = initial_capital * (1 + port_ret).cumprod()
    eq.name = "benchmark_equity"
    return eq


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def build_performance_table(
    strategy_equity: pd.Series,
    benchmark_equities: dict[str, pd.Series],
    risk_free_annual: float = 0.04,
) -> pd.DataFrame:
    """Build a comparison table of strategy vs benchmarks.

    Parameters
    ----------
    strategy_equity : Strategy equity curve.
    benchmark_equities : Dict of {label: equity_curve}.
    risk_free_annual : For Sharpe/Sortino computation.

    Returns
    -------
    DataFrame indexed by strategy/benchmark name, columns = metric names.
    """
    rows: dict[str, dict] = {
        "Strategy": compute_metrics(strategy_equity, risk_free_annual),
    }
    for name, eq in benchmark_equities.items():
        rows[name] = compute_metrics(eq, risk_free_annual)

    df = pd.DataFrame(rows).T
    pct_cols = ["total_return", "cagr", "ann_volatility", "max_drawdown", "hit_rate"]
    for col in pct_cols:
        if col in df.columns:
            df[col] = df[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
    ratio_cols = ["sharpe", "sortino", "calmar"]
    for col in ratio_cols:
        if col in df.columns:
            df[col] = df[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
    return df
