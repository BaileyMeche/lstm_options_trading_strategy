# metrics.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def pnl_from_trades(trades):
    # trades: list of dicts with qty (+ for buy), price; we'll compute realized pnl naive
    # For simplicity, assume buys reduce cash and increase position; selling realizes P&L.
    return None

def equity_series(equity_curve):
    # equity_curve: list of (timestamp, equity)
    df = pd.DataFrame(equity_curve, columns=["timestamp","equity"]).set_index("timestamp")
    return df

def compute_performance(equity_df):
    eq = equity_df["equity"].astype(float)
    returns = eq.pct_change().dropna()
    total_return = eq.iloc[-1] / eq.iloc[0] - 1
    annual_factor = 252*6.5*60  # rough if intraday 1-min obs -> not exact; for demo keep simple
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else np.nan
    # drawdown
    cum_max = eq.cummax()
    drawdown = (eq - cum_max) / cum_max
    max_dd = drawdown.min()
    return {"total_return": total_return, "sharpe": sharpe, "max_drawdown": max_dd, "returns": returns}

def plot_equity(equity_df):
    plt.figure(figsize=(10,5))
    plt.plot(equity_df.index, equity_df["equity"])
    plt.title("Equity Curve")
    plt.xlabel("Time")
    plt.ylabel("Equity")
    plt.grid(True)
    plt.show()
