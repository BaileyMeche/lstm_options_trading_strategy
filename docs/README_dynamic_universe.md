# README: Dynamic Universe (src/dynamic_universe.py)

## What this file is

A quarterly reconstitution of the 30-stock panel. At each rebalance date the
same five criteria from `docs/universe_selection_rationale.md` are applied
point-in-time, so no future information enters the selection decision.

The static universe in `src/universe_selection.py` is a snapshot frozen at
2012-12-31. This module re-runs that selection at every quarter-end, letting
the portfolio adapt to changes in fundamental data quality, market cap rank,
liquidity, and beta stability as they occur.

---

## The Five Criteria (Point-in-Time)

| ID | Rule | Threshold | Implementation |
|----|------|-----------|----------------|
| C1 | Fundamental missingness on five key fields in trailing 2-year window | Below 20% | `_check_fundamental_coverage()` |
| C2 | Continuous daily price history available | From 2006-01-01 | `_check_price_history()` |
| C3 | Trailing average daily dollar volume | At or above $50M | `_check_liquidity()` |
| C4 | Standard deviation of 252-day rolling beta in trailing 2-year window | At or below 0.40 | `_check_beta_stability()` |
| C5 | Sector diversity cap; Financials and Real Estate excluded entirely | Max 4 per GICS sector | Inside `build_dynamic_universe()` |

Five fundamental fields checked under C1:

| Field | Economic meaning |
|-------|-----------------|
| `tot_debt_tot_equity` | Financial leverage |
| `ret_equity` | Return on equity |
| `profit_margin` | Net income / revenue |
| `book_val_per_share` | Equity book value per share |
| `diluted_net_eps` | Diluted earnings per share |

---

## Turnover Buffer

`rank_buffer=10` (default) prevents churn. An incumbent is retained as long
as it ranks within position 40 (30 + buffer) among eligible names after
applying C1 through C5. A new stock only enters when a slot cannot be filled
by any retained incumbent, or when an incumbent falls entirely out of the
eligible set (fails C1 through C4, or its sector is at cap).

This mirrors the banding rule used in index reconstitution (e.g., Russell
1000 buffer zone methodology).

---

## Excluded Sectors

**Financials** and **Real Estate** are excluded at every rebalance:

- Financials: D/E is structurally incomparable; bank leverage is an
  operational input, not a financing choice. Large banks (JPM, BAC, WFC, C,
  GS, AIG) also fail C4, with rolling-beta sigma above 1.0 during the
  2008-2009 crisis.

- Real Estate: REITs report D/E and EPS on a funds-from-operations basis
  not comparable to the GAAP fields in the model. Beta is interest-rate
  driven, producing high sigma over rate-cycle turns.

---

## API

### `build_dynamic_universe(fundamentals_df, prices_df, beta_df, candidates_df, rebalance_dates, rank_buffer=10, price_history_start="2006-01-01")`

Main entry point. Applies C1 through C5 at each period in `rebalance_dates`,
applies the turnover buffer, and returns a long DataFrame:
`[rebalance_date, ticker, sector, mkt_val, rank]`. Populates `REBALANCE_LOG`.

`candidates_df` requires columns `ticker`, `sector`, `mkt_val`. If a `date`
or `rank_date` column is present the function filters per period; otherwise
the full DataFrame is used at every period.

### `expand_to_daily_universe(quarterly_universe_df, trading_dates)`

Forward-fills quarterly membership to every trading date. Returns
`[date, ticker, sector, rebalance_date]`.

### `filter_panel_to_active_universe(panel_df, daily_universe_df)`

Keeps only rows in `panel_df` where `(date, ticker)` is in the daily
universe. Use this to restrict the LSTM feature panel to active members.

### `compute_turnover_stats(rebalance_log=None)`

Returns `[rebalance_date, n_stocks, n_added, n_removed, turnover_rate]`.
Uses module-level `REBALANCE_LOG` if no argument is passed.

### `compare_static_vs_dynamic(dynamic_log=None)`

Compares static `UNIVERSE_30` against each rebalance period. Returns
`[rebalance_date, in_static, in_dynamic, static_only, dynamic_only, overlap]`.

### `print_rebalance_log_summary(rebalance_log=None)`

Prints a formatted table of per-period turnover statistics to stdout.

---

## Typical Usage

```python
import pandas as pd
from src.dynamic_universe import (
    build_dynamic_universe,
    expand_to_daily_universe,
    filter_panel_to_active_universe,
    print_rebalance_log_summary,
)

rebalance_dates = pd.date_range("2006-12-31", "2013-06-30", freq="QE").tolist()

quarterly_universe = build_dynamic_universe(
    fundamentals_df=fundamentals,
    prices_df=prices,
    beta_df=beta,
    candidates_df=sp500_candidates,
    rebalance_dates=rebalance_dates,
    rank_buffer=10,
)

trading_dates = pd.DatetimeIndex(sorted(prices["date"].unique()))
daily_universe = expand_to_daily_universe(quarterly_universe, trading_dates)
panel_dynamic = filter_panel_to_active_universe(panel_df, daily_universe)

print_rebalance_log_summary()
```

---

## Relationship to Static Universe

The static universe (`src/universe_selection.py`) is the correct baseline
for the primary backtest: the 2012-12-31 ranking date is prior to the test
period (2013-2015) and introduces no forward-looking bias.

The dynamic universe is a robustness check. Similar performance between the
two confirms the static universe is not a lucky draw of 2012 survivors.
Divergence reveals which stocks drive the difference and why.
