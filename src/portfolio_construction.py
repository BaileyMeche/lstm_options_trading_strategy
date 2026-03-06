from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

REBALANCE_LOG: list[dict] = []
"""Populated each call to build_dynamic_universe; cleared at the start of each run."""

EXCLUDED_SECTORS: frozenset[str] = frozenset({
    "Financials",  # leverage-distorted accounting ratios
    "Utilities",   # rate-sensitive, fundamentally different valuation model
})

SECTOR_CAP: int = 8
"""Maximum stocks from any single sector allowed in top-30."""

_MIN_AVG_DOLLAR_VOL: float = 1_000_000.0
_BETA_MIN: float = 0.0
_BETA_MAX: float = 3.0


# ---------------------------------------------------------------------------
# Eligibility helpers (C1–C4)
# ---------------------------------------------------------------------------


def _check_fundamental_coverage(
    fundamentals_df: pd.DataFrame,
    ticker: str,
    rd: pd.Timestamp,
) -> bool:
    """C1: At least one fundamental record available before rebalance date."""
    if fundamentals_df.empty:
        return False
    date_col = next(
        (c for c in ["feature_available_date", "per_end_date"] if c in fundamentals_df.columns),
        None,
    )
    ticker_col = next(
        (c for c in ["ticker", "ticker_price"] if c in fundamentals_df.columns),
        None,
    )
    if date_col is None or ticker_col is None:
        return False
    dates = pd.to_datetime(
        fundamentals_df.loc[fundamentals_df[ticker_col] == ticker, date_col],
        errors="coerce",
    )
    return bool((dates < rd).any())


def _check_price_history(
    prices_df: pd.DataFrame,
    ticker: str,
    start_ts: pd.Timestamp,
) -> bool:
    """C2: Price data exists going back to at least start_ts."""
    if prices_df.empty:
        return False
    ticker_col = next(
        (c for c in ["ticker", "ticker_price"] if c in prices_df.columns), None
    )
    date_col = "date" if "date" in prices_df.columns else None
    if ticker_col is None or date_col is None:
        return False
    dates = pd.to_datetime(
        prices_df.loc[prices_df[ticker_col] == ticker, date_col], errors="coerce"
    )
    return bool((not dates.empty) and (dates.min() <= start_ts))


def _check_liquidity(
    prices_df: pd.DataFrame,
    ticker: str,
    rd: pd.Timestamp,
    lookback_days: int = 63,
) -> bool:
    """C3: Average daily dollar volume over trailing lookback_days >= threshold."""
    if prices_df.empty:
        return False
    ticker_col = next(
        (c for c in ["ticker", "ticker_price"] if c in prices_df.columns), None
    )
    date_col = "date" if "date" in prices_df.columns else None
    vol_col = "volume" if "volume" in prices_df.columns else None
    price_col = next(
        (c for c in ["adj_close", "close"] if c in prices_df.columns), None
    )
    if any(c is None for c in [ticker_col, date_col, vol_col, price_col]):
        return False
    sub = prices_df[
        (prices_df[ticker_col] == ticker)
        & (pd.to_datetime(prices_df[date_col], errors="coerce") < rd)
    ]
    if sub.empty:
        return False
    recent = sub.sort_values(date_col).tail(lookback_days)
    dollar_vol = (
        pd.to_numeric(recent[vol_col], errors="coerce")
        * pd.to_numeric(recent[price_col], errors="coerce")
    )
    avg = dollar_vol.mean()
    return bool(pd.notna(avg) and avg >= _MIN_AVG_DOLLAR_VOL)


def _check_beta_stability(
    beta_df: pd.DataFrame,
    ticker: str,
    rd: pd.Timestamp,
) -> bool:
    """C4: Most recent rolling beta is finite and within [_BETA_MIN, _BETA_MAX]."""
    if beta_df.empty:
        return False
    ticker_col = "ticker" if "ticker" in beta_df.columns else None
    date_col = "date" if "date" in beta_df.columns else None
    beta_col = "beta_252d" if "beta_252d" in beta_df.columns else None
    if any(c is None for c in [ticker_col, date_col, beta_col]):
        return False
    sub = beta_df[
        (beta_df[ticker_col] == ticker)
        & (pd.to_datetime(beta_df[date_col], errors="coerce") < rd)
    ]
    if sub.empty:
        return False
    latest = pd.to_numeric(sub[beta_col], errors="coerce").dropna()
    if latest.empty:
        return False
    val = float(latest.iloc[-1])
    return bool(np.isfinite(val) and _BETA_MIN <= val <= _BETA_MAX)


# ---------------------------------------------------------------------------
# Universe construction
# ---------------------------------------------------------------------------


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