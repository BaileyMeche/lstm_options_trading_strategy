from __future__ import annotations

import pandas as pd

try:
    from .universe_selection import UNIVERSE_30
except ImportError:
    from universe_selection import UNIVERSE_30  # type: ignore[no-redef]


#
# Quarterly dynamic reconstitution of the 30-stock panel.
# Applies the same C1-C5 filters as universe_selection.py, point-in-time,
# with a turnover buffer (rank_buffer) that prevents excessive churn.
# Full selection rationale: docs/universe_selection_rationale.md
#v

# ── Constants ────────────────────────────────────────────────────────────────

SECTOR_CAP: int = 4  # C5: max stocks per GICS sector
ADDV_FLOOR: float = 50.0  # C3: min average daily dollar volume ($M)
MISSINGNESS_CEILING: float = 0.20  # C1: max allowed fundamental missingness
BETA_SIGMA_CEILING: float = 0.40  # C4: max σ(β_252d) in trailing window

FUNDAMENTAL_COLS: list[str] = [
    "tot_debt_tot_equity",
    "ret_equity",
    "profit_margin",
    "book_val_per_share",
    "diluted_net_eps",
]

# GICS sectors excluded due to structurally incomparable D/E and EPS
# (also excluded in universe_selection.py static universe).
EXCLUDED_SECTORS: frozenset[str] = frozenset({"Financials", "Real Estate"})

# Populated by build_dynamic_universe(); one entry per rebalance date.
REBALANCE_LOG: list[dict] = []


# ── Core reconstitution ───────────────────────────────────────────────────────


def _check_fundamental_coverage(
    fundamentals_df: pd.DataFrame,
    ticker: str,
    as_of: pd.Timestamp,
    lookback_days: int = 504,
) -> bool:
    """Return True if ticker passes C1 (<20% missingness on key fields) as of as_of."""
    subset = fundamentals_df[
        (fundamentals_df["ticker"] == ticker)
        & (fundamentals_df["date"] <= as_of)
        & (fundamentals_df["date"] >= as_of - pd.Timedelta(days=lookback_days))
    ]
    if subset.empty:
        return False
    for col in FUNDAMENTAL_COLS:
        if col not in subset.columns:
            return False
        miss = subset[col].isna().mean()
        if miss > MISSINGNESS_CEILING:
            return False
    return True


def _check_price_history(
    prices_df: pd.DataFrame,
    ticker: str,
    required_start: pd.Timestamp,
) -> bool:
    """Return True if ticker has continuous prices from required_start (C2)."""
    px = prices_df[(prices_df["ticker"] == ticker) & (prices_df["date"] >= required_start)]
    return not px.empty and px["date"].min() <= required_start + pd.Timedelta(days=5)


def _check_liquidity(
    prices_df: pd.DataFrame,
    ticker: str,
    as_of: pd.Timestamp,
    lookback_days: int = 252,
) -> bool:
    """Return True if ticker's trailing ADDV >= ADDV_FLOOR (C3)."""
    subset = prices_df[
        (prices_df["ticker"] == ticker)
        & (prices_df["date"] <= as_of)
        & (prices_df["date"] >= as_of - pd.Timedelta(days=lookback_days))
    ].copy()
    if subset.empty:
        return False
    subset["dv"] = pd.to_numeric(subset["close"], errors="coerce") * pd.to_numeric(
        subset["volume"], errors="coerce"
    )
    addv_m = subset["dv"].mean() / 1e6
    return bool(addv_m >= ADDV_FLOOR)


def _check_beta_stability(
    beta_df: pd.DataFrame,
    ticker: str,
    as_of: pd.Timestamp,
    lookback_days: int = 504,
) -> bool:
    """Return True if σ(β_252d) <= BETA_SIGMA_CEILING in trailing window (C4)."""
    subset = beta_df[
        (beta_df["ticker"] == ticker)
        & (beta_df["date"] <= as_of)
        & (beta_df["date"] >= as_of - pd.Timedelta(days=lookback_days))
    ]
    if len(subset) < 30:
        return False
    sigma = subset["beta_252d"].dropna().std()
    return bool(sigma <= BETA_SIGMA_CEILING)


def build_dynamic_universe(
    fundamentals_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    beta_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    rebalance_dates: list[pd.Timestamp],
    rank_buffer: int = 10,
    price_history_start: str = "2006-01-01",
) -> pd.DataFrame:
    """Reconstitute the universe quarterly, applying C1-C5 point-in-time.

    candidates_df must have columns: ticker, sector, mkt_val (at each rebalance date).
    rank_buffer prevents churn: an incumbent only exits if a challenger ranks
    rank_buffer positions higher by market cap. Returns a long DataFrame with
    columns [rebalance_date, ticker, sector, mkt_val, rank].
    """
    global REBALANCE_LOG
    REBALANCE_LOG.clear()

    start_ts = pd.Timestamp(price_history_start)
    prev_universe: set[str] = set()
    records: list[dict] = []

    for rd in sorted(rebalance_dates):
        # Filter candidates to this rebalance date when a date column is present.
        if "date" in candidates_df.columns:
            rd_cands = candidates_df[candidates_df["date"] == rd].copy()
        elif "rank_date" in candidates_df.columns:
            rd_cands = candidates_df[candidates_df["rank_date"] == rd].copy()
        else:
            rd_cands = candidates_df.copy()

        # C5 pre-filter: remove excluded sectors.
        if "sector" in rd_cands.columns:
            rd_cands = rd_cands[~rd_cands["sector"].isin(EXCLUDED_SECTORS)]

        # Apply C1-C4 per ticker.
        eligible: list[dict] = []
        for _, row in rd_cands.iterrows():
            ticker = row["ticker"]
            sector = row.get("sector", "Unknown")
            mkt_val = row.get("mkt_val", 0.0)

            if not _check_fundamental_coverage(fundamentals_df, ticker, rd):
                continue
            if not _check_price_history(prices_df, ticker, start_ts):
                continue
            if not _check_liquidity(prices_df, ticker, rd):
                continue
            if not _check_beta_stability(beta_df, ticker, rd):
                continue

            eligible.append({"ticker": ticker, "sector": sector, "mkt_val": mkt_val})

        eligible_df = (
            pd.DataFrame(eligible)
            .sort_values("mkt_val", ascending=False)
            .reset_index(drop=True)
        )
        eligible_df["raw_rank"] = eligible_df.index + 1

        # C5: sector cap = SECTOR_CAP.
        chosen: list[dict] = []
        sector_counts: dict[str, int] = {}
        for _, row in eligible_df.iterrows():
            s = row["sector"]
            if sector_counts.get(s, 0) < SECTOR_CAP and len(chosen) < 30:
                chosen.append(row.to_dict())
                sector_counts[s] = sector_counts.get(s, 0) + 1

        chosen_df = pd.DataFrame(chosen).reset_index(drop=True)
        chosen_df["rank"] = chosen_df.index + 1
        chosen_set = set(chosen_df["ticker"])

        # Turnover buffer: retain incumbents unless challenger is rank_buffer better.
        if prev_universe:
            retained: set[str] = set()
            challenger_ranks = {
                row["ticker"]: row["rank"] for _, row in chosen_df.iterrows()
            }
            for incumbent in prev_universe:
                if incumbent not in challenger_ranks:
                    # No longer eligible — force exit.
                    continue
                inc_rank = challenger_ranks[incumbent]
                # Keep if within buffer of top-30 threshold.
                if inc_rank <= 30 + rank_buffer:
                    retained.add(incumbent)

            # Build buffered universe: retained incumbents + top new entries.
            buffered = retained & chosen_set
            new_entries = [
                t for t in chosen_df["ticker"] if t not in prev_universe
            ]
            slots = 30 - len(buffered)
            buffered |= set(new_entries[:slots])

            final_df = chosen_df[chosen_df["ticker"].isin(buffered)].copy()
        else:
            final_df = chosen_df.copy()

        final_df = final_df.reset_index(drop=True)
        final_df["rank"] = final_df.index + 1
        final_df["rebalance_date"] = rd

        added = set(final_df["ticker"]) - prev_universe
        removed = prev_universe - set(final_df["ticker"])
        n = len(final_df)
        turnover = len(added) / n if n > 0 else 0.0

        REBALANCE_LOG.append(
            {
                "rebalance_date": rd,
                "n_stocks": n,
                "added": sorted(added),
                "removed": sorted(removed),
                "turnover_rate": round(turnover, 4),
                "universe": sorted(final_df["ticker"].tolist()),
            }
        )

        records.append(final_df)
        prev_universe = set(final_df["ticker"])

    if not records:
        return pd.DataFrame(columns=["rebalance_date", "ticker", "sector", "mkt_val", "rank"])

    return pd.concat(records, ignore_index=True)


# ── Panel helpers ─────────────────────────────────────────────────────────────


def expand_to_daily_universe(
    quarterly_universe_df: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Forward-fill quarterly universe membership to each trading date.

    Returns a long DataFrame [date, ticker, sector, rebalance_date] covering
    all trading_dates between the first and last rebalance date.
    """
    if quarterly_universe_df.empty:
        return pd.DataFrame(columns=["date", "ticker", "sector", "rebalance_date"])

    rebalance_dates = sorted(quarterly_universe_df["rebalance_date"].unique())
    parts: list[pd.DataFrame] = []

    for i, rd in enumerate(rebalance_dates):
        next_rd = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else trading_dates[-1]
        active = quarterly_universe_df[quarterly_universe_df["rebalance_date"] == rd].copy()
        period_dates = trading_dates[(trading_dates >= rd) & (trading_dates < next_rd)]

        for td in period_dates:
            day_df = active[["ticker", "sector"]].copy()
            day_df["date"] = td
            day_df["rebalance_date"] = rd
            parts.append(day_df)

    if not parts:
        return pd.DataFrame(columns=["date", "ticker", "sector", "rebalance_date"])

    return pd.concat(parts, ignore_index=True)[["date", "ticker", "sector", "rebalance_date"]]


def filter_panel_to_active_universe(
    panel_df: pd.DataFrame,
    daily_universe_df: pd.DataFrame,
) -> pd.DataFrame:
    """Keep only rows in panel_df where (date, ticker) is in daily_universe_df."""
    key = daily_universe_df[["date", "ticker"]].drop_duplicates()
    key["_in_universe"] = True
    out = panel_df.merge(key, on=["date", "ticker"], how="left")
    out = out[out["_in_universe"].eq(True)].drop(columns=["_in_universe"]).copy()
    return out.reset_index(drop=True)


# ── Diagnostics ───────────────────────────────────────────────────────────────


def compute_turnover_stats(rebalance_log: list[dict] | None = None) -> pd.DataFrame:
    """Summarise per-period turnover from REBALANCE_LOG (or a provided list).

    Returns a DataFrame with columns [rebalance_date, n_stocks, n_added,
    n_removed, turnover_rate].
    """
    log = rebalance_log if rebalance_log is not None else REBALANCE_LOG
    rows = [
        {
            "rebalance_date": entry["rebalance_date"],
            "n_stocks": entry["n_stocks"],
            "n_added": len(entry["added"]),
            "n_removed": len(entry["removed"]),
            "turnover_rate": entry["turnover_rate"],
        }
        for entry in log
    ]
    return pd.DataFrame(rows)


def compare_static_vs_dynamic(
    dynamic_log: list[dict] | None = None,
) -> pd.DataFrame:
    """Compare static UNIVERSE_30 membership against each dynamic rebalance.

    Returns a DataFrame with columns [rebalance_date, in_static, in_dynamic,
    static_only, dynamic_only, overlap].
    """
    static_tickers = set(UNIVERSE_30["ticker"].tolist())
    log = dynamic_log if dynamic_log is not None else REBALANCE_LOG

    rows = []
    for entry in log:
        dynamic_tickers = set(entry["universe"])
        rows.append(
            {
                "rebalance_date": entry["rebalance_date"],
                "in_static": len(static_tickers),
                "in_dynamic": len(dynamic_tickers),
                "static_only": sorted(static_tickers - dynamic_tickers),
                "dynamic_only": sorted(dynamic_tickers - static_tickers),
                "overlap": len(static_tickers & dynamic_tickers),
            }
        )
    return pd.DataFrame(rows)


def print_rebalance_log_summary(rebalance_log: list[dict] | None = None) -> None:
    """Print a human-readable summary of each rebalance period."""
    log = rebalance_log if rebalance_log is not None else REBALANCE_LOG
    if not log:
        print("REBALANCE_LOG is empty; run build_dynamic_universe() first.")
        return

    stats = compute_turnover_stats(log)
    print("=" * 64)
    print("Dynamic Universe Rebalance Summary")
    print("=" * 64)
    print(f"{'Date':<14} {'N':>4} {'Added':>6} {'Removed':>8} {'Turnover':>10}")
    print("-" * 64)
    for _, row in stats.iterrows():
        print(
            f"{str(row['rebalance_date'].date()):<14}"
            f" {row['n_stocks']:>4}"
            f" {row['n_added']:>6}"
            f" {row['n_removed']:>8}"
            f" {row['turnover_rate']:>9.1%}"
        )
    print("-" * 64)
    print(f"  Mean turnover per period : {stats['turnover_rate'].mean():.1%}")
    print(f"  Total rebalance periods  : {len(stats)}")
    print("=" * 64)


if __name__ == "__main__":
    print("dynamic_universe.py — use build_dynamic_universe() to reconstitute.")
    print("Static universe baseline:")
    from universe_selection import _print_universe_summary  # type: ignore[attr-defined]
    _print_universe_summary()
