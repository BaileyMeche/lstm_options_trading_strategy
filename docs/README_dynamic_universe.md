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

## Data Flow and Portfolio Construction

### Sample Timeline

```
2006-01-01          2011-12-31  2012-12-31  2013-01-01          2013-12-31
    |                    |           |           |                    |
    |<--- TRAINING ------>|<-- DEV -->|<--- TEST PERIOD ------------>|
    |                    |           |           |                    |
    | Beta burn-in (252d)|           |           |                    |
    |=====>|             |           |           |                    |
```

First rolling beta is available approximately 2006-12-31 (252 trading days
in). All C4 checks before that date return False; stocks are excluded until
the beta window is fully populated.

---

### Step 1: Quarterly Universe Reconstitution

Runs at each quarter-end date (e.g. 2006-12-31, 2007-03-31, ...).

```
S&P 500 constituent list at rebalance date
            |
            v
  [Remove Financials and Real Estate]  <- C5 pre-filter
            |
            v
  For each remaining candidate:
    +-----------------------------------------------+
    | C1: fundamentals_df, trailing 504 days         |
    |     < 20% missing on 5 fields?                 |
    | C2: prices_df from 2006-01-01                  |
    |     first price <= required_start + 5 days?    |
    | C3: prices_df, trailing 45 days                |
    |     ADDV = mean(close x volume) >= $50M?       |
    | C4: beta_df, trailing 504 days                 |
    |     std(beta_252d) <= 0.40?                    |
    +-----------------------------------------------+
            |
            v
  Rank eligible stocks by market cap (descending)
            |
            v
  Apply sector cap: max 4 per GICS sector  <- C5 post-filter
            |
            v
  Apply turnover buffer (rank_buffer=10):
    Incumbents retained if rank <= 40
    New entrants fill remaining slots
            |
            v
  Output: up to 30 stocks for this quarter
  Logged to REBALANCE_LOG
```

Functions: `build_dynamic_universe()` calls `_check_fundamental_coverage()`,
`_check_price_history()`, `_check_liquidity()`, `_check_beta_stability()`.

---

### Step 2: Expand to Daily Membership

```
Quarterly universe DataFrame
  [rebalance_date, ticker, sector, mkt_val, rank]
            |
            v
  expand_to_daily_universe(quarterly_universe_df, trading_dates)
            |
            v
  Daily universe DataFrame
  [date, ticker, sector, rebalance_date]

  Each trading day carries forward the most recent
  quarter-end universe until the next rebalance date.
```

---

### Step 3: Filter Feature Panel

```
Full LSTM feature panel (all tickers, all dates)
            |
            v
  filter_panel_to_active_universe(panel_df, daily_universe_df)
            |
            v
  Filtered panel: only (date, ticker) pairs
  where ticker was active in that date's quarter
```

---

### Step 4: Daily Signal and Trade

Runs every trading day within the test period.

```
Active universe (30 stocks today)
            |
            v
  LSTM model produces predicted open-to-close
  log return for each stock
            |
            v
  Rank all 30 stocks by predicted return
            |
       +----+----+
       |         |
  Top 20%    Bottom 20%
  (6 stocks)  (6 stocks)
  LONG leg    SHORT leg
       |         |
       +----+----+
            |
            v
  Beta-dollar hedge per leg:
    hedge_value_i = beta_252d_i x (shares_i x price_i)
    Offset with SPY (or QQQ) position
            |
            v
  Enter at open, exit at close (intraday, no overnight)
```

---

### Step 5: Holding Period and Position Sizing

| Parameter | Value |
|-----------|-------|
| Holding period | Intraday (open to close, same day) |
| Overnight exposure | None |
| Signal frequency | Daily (new LSTM prediction each morning) |
| Universe update | Quarterly (quarter-end rebalance) |
| Long leg size | Top quintile: 6 stocks out of 30 |
| Short leg size | Bottom quintile: 6 stocks out of 30 |
| Middle stocks | Held flat (positions 7 through 24 not traded) |
| Hedge | Beta x position value, executed in SPY or QQQ |

---

### Data Required at Each Stage

| Stage | Data source | Fields used | Frequency |
|-------|-------------|-------------|-----------|
| C1 check | Zacks FR table (fundamentals_df) | tot_debt_tot_equity, ret_equity, profit_margin, book_val_per_share, diluted_net_eps | Quarterly, PIT (45-day filing lag) |
| C2 check | PRICES.csv (prices_df) | date, ticker | Daily |
| C3 check | PRICES.csv (prices_df) | close, volume | Daily, trailing 45 days |
| C4 check | Derived beta_df | beta_252d from compute_rolling_beta_vs_spy() | Daily, trailing 504 days |
| LSTM features | Merged panel (panel_df) | All 9 features including beta_252d, price_to_book | Daily |
| Signal target | PRICES.csv | adj_open, adj_close (open-to-close log return) | Daily |
| Hedge | Derived beta_df + index prices | beta_252d, SPY or QQQ close | Daily |

---

## The Five Criteria (Point-in-Time)

| ID | Rule | Threshold | Implementation |
|----|------|-----------|----------------|
| C1 | Fundamental missingness on five key fields in trailing 2-year window | Below 20% | `_check_fundamental_coverage()` |
| C2 | Continuous daily price history available | From 2006-01-01 | `_check_price_history()` |
| C3 | Trailing average daily dollar volume over the prior 45 calendar days | At or above $50M | `_check_liquidity()` |
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

## Window Choice: Empirical Basis

| Check | Window | Rationale |
|-------|--------|-----------|
| C1: Fundamental coverage | 504 calendar days (2 years) | Need at least 8 quarterly filings to measure missingness rate meaningfully. A 2-year window covers two full fiscal cycles and one full crisis-recovery period. |
| C2: Price history | Fixed from 2006-01-01 | Absolute requirement for LSTM training window completeness; no rolling window applies. |
| C3: Liquidity | 45 calendar days (half a quarter) | Captures current trading conditions at the rebalance date rather than trailing annual averages. Approximately 31 trading days, sufficient to smooth daily volume noise while remaining within the current business quarter. |
| C4: Beta stability | 504 calendar days (2 years) | The 252-day rolling beta itself requires a 252-day burn-in. A 2-year trailing window gives two non-overlapping annual beta estimates, enough to measure cross-date standard deviation reliably. |

The 45-day liquidity window can be defended by showing that 45-day and
252-day ADDV rankings agree on 95%+ of stocks in the sample, making the
shorter window sufficient for exclusion screening at large-cap scale.

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

Financials and Real Estate are excluded at every rebalance:

- Financials: D/E is structurally incomparable; bank leverage is an
  operational input, not a financing choice. Large banks (JPM, BAC, WFC, C,
  GS, AIG) also fail C4, with rolling-beta sigma above 1.0 during the
  2008-2009 crisis.

- Real Estate: REITs report D/E and EPS on a funds-from-operations basis
  not comparable to the GAAP fields in the model. Beta is interest-rate
  driven, producing high sigma over rate-cycle turns.

---

## Hedging

The strategy uses beta-dollar hedging to remove market exposure from each leg.

Hedge calculation:

```
hedge_value = beta_i x position_value_i
           = beta_i x (shares_i x price_i)
```

Index choice: SPY (S&P 500 ETF) is the default because all 30 stocks are
S&P 500 constituents and their betas are already estimated against SPY in
`compute_rolling_beta_vs_spy()`. QQQ (Nasdaq-100) is an acceptable alternative
but introduces basis risk for non-tech names. Either index is valid as long
as the same one is used consistently for both beta estimation and hedge execution.

Why beta-dollar hedging: neutralizing beta x position value removes the
systematic return component from each stock leg. The LSTM signal is trained
to predict idiosyncratic return, so the hedge converts gross positions into
market-neutral bets. The residual P&L reflects only the quality of the LSTM
ranking signal.

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
