from __future__ import annotations

import pandas as pd


# 
# 30-stock static universe — S&P 500 members as of 2012-12-31, ranked by
# market cap, max 4 per sector, continuous price history from 2006-01-01,
# <20% missingness on key fundamental fields, $50M+ ADDV, and
# σ(β_252d) ≤ 0.40 across the 2006-2011 training window (beta stability).
# Full selection rationale: docs/universe_selection_rationale.md
# 

_UNIVERSE_RECORDS: list[dict] = [
    # ── Technology (4 / 4) ──────────────────────────────────────────────────
    # AAPL / MSFT / GOOGL: no long-term debt in early years; Zacks stores
    # zero-debt quarters as NULL rather than 0.  de_impute_zero=True means
    # fill NaN tot_debt_tot_equity with 0 — not forward-fill from a later value.
    {
        "ticker": "AAPL",  "ticker_price": "AAPL",
        "sector": "Technology",             "mkt_val": 499613.81, "ipo_year": 1980,
        "de_impute_zero": True,
        "notes": "No debt 2006-2012; first bond Oct-2013.",
    },
    {
        "ticker": "MSFT",  "ticker_price": "MSFT",
        "sector": "Technology",             "mkt_val": 224801.13, "ipo_year": 1986,
        "de_impute_zero": True,
        "notes": "Near-zero debt pre-2009; 36% D/E NULL = zero-debt quarters.",
    },
    {
        "ticker": "GOOGL", "ticker_price": "GOOGL",
        "sector": "Technology",             "mkt_val": 232440.81, "ipo_year": 2004,
        "de_impute_zero": True,
        "notes": "No material debt pre-2011; 61% D/E NULL = zero-debt.",
    },
    {
        "ticker": "IBM",   "ticker_price": "IBM",
        "sector": "Technology",             "mkt_val": 216438.08, "ipo_year": 1915,
        "de_impute_zero": False,
        "notes": "Clean fundamental coverage throughout sample.",
    },
    # ── Healthcare (4 / 4) ──────────────────────────────────────────────────
    # ABT: spun off AbbVie (ABBV) 2013-01-01; fundamental step-change at
    # start of test period.  Prices split-adjusted; EPS/margin drop is real.
    {
        "ticker": "JNJ",   "ticker_price": "JNJ",
        "sector": "Healthcare",             "mkt_val": 194265.33, "ipo_year": 1944,
        "de_impute_zero": False,
        "notes": "Clean fundamental coverage throughout sample.",
    },
    {
        "ticker": "PFE",   "ticker_price": "PFE",
        "sector": "Healthcare",             "mkt_val": 195800.00, "ipo_year": 1972,
        "de_impute_zero": False,
        "notes": "Largest pharma by revenue in period; full Zacks coverage.",
    },
    {
        "ticker": "MRK",   "ticker_price": "MRK",
        "sector": "Healthcare",             "mkt_val": 120000.00, "ipo_year": 1946,
        "de_impute_zero": False,
        "notes": "Major diversified pharma; full Zacks coverage.",
    },
    {
        "ticker": "ABT",   "ticker_price": "ABT",
        "sector": "Healthcare",             "mkt_val": 55000.00,  "ipo_year": 1929,
        "de_impute_zero": False,
        "notes": "CORP EVENT: AbbVie (ABBV) spin 2013-01-01; fundamental break at test boundary.",
    },
    # ── Energy (4 / 4) ──────────────────────────────────────────────────────
    # COP: spun off Phillips 66 (PSX) 2012-05-01; revenue/D/E drop mid-training.
    # Pre-spin COP includes refining; post-spin is pure E&P.
    {
        "ticker": "XOM",   "ticker_price": "XOM",
        "sector": "Energy",                 "mkt_val": 394610.88, "ipo_year": 1882,
        "de_impute_zero": False,
        "notes": "Largest US energy by market cap; full Zacks coverage.",
    },
    {
        "ticker": "CVX",   "ticker_price": "CVX",
        "sector": "Energy",                 "mkt_val": 211649.45, "ipo_year": 1879,
        "de_impute_zero": False,
        "notes": "Second largest US integrated oil; full Zacks coverage.",
    },
    {
        "ticker": "COP",   "ticker_price": "COP",
        "sector": "Energy",                 "mkt_val": 85000.00,  "ipo_year": 2002,
        "de_impute_zero": False,
        "notes": "CORP EVENT: Phillips 66 (PSX) spin 2012-05-01; D/E step-change mid-training.",
    },
    {
        "ticker": "SLB",   "ticker_price": "SLB",
        "sector": "Energy",                 "mkt_val": 100000.00, "ipo_year": 1962,
        "de_impute_zero": False,
        "notes": "Largest oilfield-services company; full Zacks coverage.",
    },
    # ── Consumer Staples (4 / 4) ────────────────────────────────────────────
    {
        "ticker": "WMT",   "ticker_price": "WMT",
        "sector": "Consumer Staples",       "mkt_val": 229351.06, "ipo_year": 1972,
        "de_impute_zero": False,
        "notes": "Lowest missingness of current 10; clean fundamentals.",
    },
    {
        "ticker": "PG",    "ticker_price": "PG",
        "sector": "Consumer Staples",       "mkt_val": 210000.00, "ipo_year": 1890,
        "de_impute_zero": False,
        "notes": "Largest consumer staples by mkt cap after WMT.",
    },
    {
        "ticker": "KO",    "ticker_price": "KO",
        "sector": "Consumer Staples",       "mkt_val": 175000.00, "ipo_year": 1892,
        "de_impute_zero": False,
        "notes": "Stable cash-flow profile; full Zacks coverage.",
    },
    {
        "ticker": "PEP",   "ticker_price": "PEP",
        "sector": "Consumer Staples",       "mkt_val": 110000.00, "ipo_year": 1919,
        "de_impute_zero": False,
        "notes": "Comparable fundamental profile to KO; within-sector variation.",
    },
    # ── Industrials (4 / 4) ─────────────────────────────────────────────────
    {
        "ticker": "GE",    "ticker_price": "GE",
        "sector": "Industrials",            "mkt_val": 220107.44, "ipo_year": 1892,
        "de_impute_zero": False,
        "notes": "Largest industrial conglomerate by mkt cap in period.",
    },
    {
        "ticker": "BA",    "ticker_price": "BA",
        "sector": "Industrials",            "mkt_val": 55000.00,  "ipo_year": 1934,
        "de_impute_zero": False,
        "notes": "Major aerospace/defense; full price and fundamental history.",
    },
    {
        "ticker": "MMM",   "ticker_price": "MMM",
        "sector": "Industrials",            "mkt_val": 65000.00,  "ipo_year": 1916,
        "de_impute_zero": False,
        "notes": "Diversified industrial with stable margin profile.",
    },
    {
        "ticker": "HON",   "ticker_price": "HON",
        "sector": "Industrials",            "mkt_val": 60000.00,  "ipo_year": 1920,
        "de_impute_zero": False,
        "notes": "Aerospace/automation; full price and fundamental history.",
    },
    # ── Consumer Discretionary (4 / 4) ──────────────────────────────────────
    # AMZN: near-zero/negative margins 2006-2012 by design (reinvestment model).
    # profit_margin and diluted_net_eps near zero is real data, not a gap.
    {
        "ticker": "MCD",   "ticker_price": "MCD",
        "sector": "Consumer Discretionary", "mkt_val": 95000.00,  "ipo_year": 1965,
        "de_impute_zero": False,
        "notes": "Franchise model; stable high-margin fundamentals.",
    },
    {
        "ticker": "HD",    "ticker_price": "HD",
        "sector": "Consumer Discretionary", "mkt_val": 80000.00,  "ipo_year": 1981,
        "de_impute_zero": False,
        "notes": "Home improvement; cyclically sensitive; useful for ranking.",
    },
    {
        "ticker": "DIS",   "ticker_price": "DIS",
        "sector": "Consumer Discretionary", "mkt_val": 90000.00,  "ipo_year": 1957,
        "de_impute_zero": False,
        "notes": "Diversified media/entertainment; full Zacks coverage.",
    },
    {
        "ticker": "AMZN",  "ticker_price": "AMZN",
        "sector": "Consumer Discretionary", "mkt_val": 115000.00, "ipo_year": 1997,
        "de_impute_zero": False,
        "notes": "S&P 500 since Nov-2005; near-zero margin is real, not missing.",
    },
    # ── Communication Services (3) ───────────────────────────────────────────
    # Sector (Telecom Services in 2012 GICS) has only 3 eligible names at
    # liquid-cap scale without compromising fundamental coverage.
    {
        "ticker": "T",     "ticker_price": "T",
        "sector": "Communication Services", "mkt_val": 200000.00, "ipo_year": 1983,
        "de_impute_zero": False,
        "notes": "AT&T; largest telecom by mkt cap in period.",
    },
    {
        "ticker": "VZ",    "ticker_price": "VZ",
        "sector": "Communication Services", "mkt_val": 110000.00, "ipo_year": 2000,
        "de_impute_zero": False,
        "notes": "Verizon; formed 2000 (Bell Atlantic + GTE); full history.",
    },
    {
        "ticker": "CMCSA", "ticker_price": "CMCSA",
        "sector": "Communication Services", "mkt_val": 70000.00,  "ipo_year": 1972,
        "de_impute_zero": False,
        "notes": "Comcast; S&P 500 since 2002; full coverage.",
    },
    # ── Materials (2) ───────────────────────────────────────────────────────
    {
        "ticker": "DD",    "ticker_price": "DD",
        "sector": "Materials",              "mkt_val": 45000.00,  "ipo_year": 1935,
        "de_impute_zero": False,
        "notes": "DuPont; largest US chemicals by mkt cap in period.",
    },
    {
        "ticker": "PX",    "ticker_price": "PX",
        "sector": "Materials",              "mkt_val": 35000.00,  "ipo_year": 1992,
        "de_impute_zero": False,
        "notes": "Praxair industrial gases; spun from Union Carbide 1992.",
    },
    # ── Utilities (1) ───────────────────────────────────────────────────────
    # NEE excluded: traded as FPL Group pre-2010; ticker join risk in PRICES.csv.
    # SO (Southern Company) has a stable ticker throughout the full sample.
    {
        "ticker": "SO",    "ticker_price": "SO",
        "sector": "Utilities",              "mkt_val": 40000.00,  "ipo_year": 1949,
        "de_impute_zero": False,
        "notes": "Largest regulated US utility; stable ticker throughout sample.",
    },
]

# Pre-built DataFrame, sorted by market cap descending (mirrors top-10 ordering).
UNIVERSE_30: pd.DataFrame = (
    pd.DataFrame(_UNIVERSE_RECORDS)
    .assign(rank_date=pd.Timestamp("2012-12-31"))
    .sort_values("mkt_val", ascending=False)
    .reset_index(drop=True)
)

# Audit trail for excluded candidates.
# Criterion references match docs/universe_selection_rationale.md.
EXCLUDED_CANDIDATES: dict[str, str] = {
    "BRK_B":  "book_val_per_share 39% missing (C1); beta σ ~0.55 during 2008-2009 distress (C4).",
    "JPM":    "Beta σ > 1.0 during financial crisis; equity behaved as distressed option (C4).",
    "BAC":    "Beta σ > 1.0 during financial crisis; equity behaved as distressed option (C4).",
    "WFC":    "Beta σ > 1.0 during financial crisis; equity behaved as distressed option (C4).",
    "C":      "Beta σ > 1.2 during crisis; near-insolvency 2008 (C4).",
    "GS":     "Beta σ > 1.0 during crisis; investment bank distress dynamics (C4).",
    "AIG":    "Beta σ > 1.5 during crisis; near-insolvency 2008 (C4).",
    "PM":     "IPO March 2008 (MO spinoff); no 2006-2007 price/fundamental history (C2).",
    "INTC":   "Technology sector at cap (C5); ranked below IBM by 2012 mkt cap.",
    "ORCL":   "Technology sector at cap (C5); ranked below IBM by 2012 mkt cap.",
    "QCOM":   "Technology sector at cap (C5); ranked below IBM by 2012 mkt cap.",
    "NEE":    "Traded as FPL Group pre-2010; ticker join risk in PRICES.csv (C2).",
}


def build_static_top30_universe(rank_date: str = "2012-12-31") -> pd.DataFrame:
    """Return the 30-stock static universe as a DataFrame.

    Drop-in replacement for build_static_top10_universe.  Returns columns:
    ticker, ticker_price, mkt_val, rank_date, sector, de_impute_zero,
    ipo_year, notes — sorted by mkt_val descending.
    """
    out = UNIVERSE_30.copy()
    out["rank_date"] = pd.Timestamp(rank_date)
    return out


def get_zero_debt_tickers() -> list[str]:
    """Return ticker_price values whose NaN tot_debt_tot_equity should be filled with 0.

    These companies had genuinely zero long-term debt; Zacks stores those
    quarters as NULL rather than 0, producing spurious missingness.
    """
    return UNIVERSE_30.loc[UNIVERSE_30["de_impute_zero"], "ticker_price"].tolist()


def apply_zero_debt_imputation(
    panel_df: pd.DataFrame,
    col: str = "tot_debt_tot_equity",
) -> pd.DataFrame:
    """Fill NaN ``col`` with 0 for zero-debt tickers in the daily feature panel.

    Call after asof_join_point_in_time and before build_lstm_tensors.
    Only rows where the ticker is in get_zero_debt_tickers() and the
    column is NaN are modified.
    """
    out = panel_df.copy()
    zero_debt = set(get_zero_debt_tickers())
    mask = out["ticker"].isin(zero_debt) & out[col].isna()
    out.loc[mask, col] = 0.0
    return out


def _print_universe_summary() -> None:
    cols = ["ticker", "sector", "mkt_val", "de_impute_zero", "ipo_year"]
    summary = UNIVERSE_30[cols].copy()
    summary["mkt_val"] = summary["mkt_val"].map(lambda x: f"${x:,.0f}M")
    summary.index = range(1, len(summary) + 1)

    sector_counts = UNIVERSE_30["sector"].value_counts()

    print("=" * 72)
    print("30-Stock Static Universe (ranked by 2012-12-31 mkt cap)")
    print("=" * 72)
    print(summary.to_string())
    print()
    print("Sector breakdown:")
    for sector, count in sector_counts.sort_values(ascending=False).items():
        bar = "#" * count
        print(f"  {sector:<28}  {bar}  ({count})")
    print()
    print(f"Total stocks              : {len(UNIVERSE_30)}")
    print(f"Sectors represented       : {UNIVERSE_30['sector'].nunique()}")
    print(f"D/E zero-imputation needed: {UNIVERSE_30['de_impute_zero'].sum()} "
          f"({', '.join(get_zero_debt_tickers())})")
    print("=" * 72)


if __name__ == "__main__":
    _print_universe_summary()
    print("\nExcluded candidates:")
    for ticker, reason in EXCLUDED_CANDIDATES.items():
        print(f"  {ticker:<8}: {reason}")
