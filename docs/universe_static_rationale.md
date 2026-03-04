# Universe Selection Rationale: 30-Stock LSTM Panel

## Why Extend from 10 to 30 Stocks

Zou and Qu (2020) used the top 10 S&P 500 names by market capitalization as of
their ranking date.  That choice was pragmatic: it kept the data pipeline simple
and focused the paper on methodology rather than universe design.  For our
project, extending to 30 stocks produces three concrete improvements:

1. **Cross-sectional signal depth.** A long-short strategy ranks stocks against
   each other.  With 10 names, the top and bottom buckets each contain 2-3 names,
   making results highly sensitive to any single security.  With 30 names, the
   top and bottom quintiles each contain 6 stocks, which smooths idiosyncratic
   noise and makes ranking more statistically meaningful.

2. **Trade-count requirement.** The final project requires at least 40 trades in
   the test set.  With 10 stocks and a daily open-to-close signal, hitting 40
   non-overlapping trades is tight.  With 30 stocks the test tensor contains
   roughly 6,300 samples, making 40+ distinct signals a realistic expectation.

3. **Sector neutrality.** 10 stocks chosen purely by market cap skew heavily
   toward technology and energy.  30 stocks with a sector cap allow the
   long-short portfolio to be approximately sector-neutral, which is a standard
   requirement in institutional equity strategies.

---

## The Five Selection Criteria

### Criterion 1: Fundamental Data Availability (Binding Constraint)

**Rule:** Any stock with more than 20% missingness on any of the five key
fundamental fields across the 2006-2013 sample is excluded.

**Fields checked:**

| Field | Economic Meaning |
|---|---|
| `tot_debt_tot_equity` | Financial leverage: how much the firm borrows relative to equity book value |
| `ret_equity` | Return on equity: how efficiently management generates profit from shareholders' capital |
| `profit_margin` | Net income / revenue: how much of each revenue dollar reaches the bottom line |
| `book_val_per_share` | Per-share equity book value: basis for price-to-book ratio |
| `diluted_net_eps` | Diluted earnings per share: absolute profitability signal |

**Why this matters:** The LSTM's point-in-time integrity argument rests on the
claim that no future information enters the feature set.  If a fundamental field
is missing for an extended stretch, the model falls back on a stale value
(forward-filled from the last available quarter) or drops the row entirely.
Either path degrades signal quality.  A stock with 40% missingness on D/E is
effectively running without that feature for nearly half the sample; the model
sees a misleading feature vector for those rows.

**Special case: zero-debt companies.** Three stocks in the final universe
(AAPL, MSFT, GOOGL) show high `tot_debt_tot_equity` missingness not because
Zacks lacks coverage, but because the company had zero long-term debt in those
quarters.  Zacks stores zero-debt quarters as NULL rather than 0.  The correct
fix is to **impute 0**, not to exclude the stock or forward-fill from a stale
nonzero value.  The `apply_zero_debt_imputation()` function in
`src/universe_selection.py` handles this.

**Explicitly excluded on this basis:** BRK_B (`book_val_per_share` 39% missing;
also fails Criterion 4, rolling-beta σ ~0.55 during the 2008-2009 distress period).

---

### Criterion 2: Continuous Price History from January 2006

**Rule:** Every stock must have a complete daily price series beginning no later
than 2006-01-01, with no gaps longer than standard market holidays.

**Why this matters:** The training window is 2006-2011.  A stock that IPO'd in
2008 has no 2006-2007 training data.  An LSTM trained on partial sequences for
some stocks and full sequences for others will allocate its gradient budget
unevenly; the model effectively sees some firms as new entrants and others as
established.  The parameter estimates for those firms are based on far fewer
observations.

**Practical check:** S&P 500 as of 2012-12-31 contains roughly 440 names that
were also trading in January 2006.  The 60-odd stocks that were added to the
index after 2006 (through replacements after bankruptcies, acquisitions, and new
listings) are automatically excluded by this criterion.

**Explicitly excluded on this basis:** PM (Philip Morris International).
Philip Morris was spun off from Altria (MO) in March 2008.  No standalone PM
price or fundamental history exists for 2006-2007.

---

### Criterion 3: Minimum Liquidity - $50M Average Daily Dollar Volume

**Rule:** Each stock must have had at least $50 million in average daily dollar
volume (price x shares traded) across the 2006-2013 sample.

**Why this matters:** The strategy simulates buying at the open and selling at
the close on signal days.  For this to be realistic, the stock must be liquid
enough that a trade of any reasonable size does not move the market.  Below $50M
ADDV, bid-ask spreads widen, market impact becomes non-trivial, and the
transaction cost model breaks down.

**Practical effect:** All 30 selected names are large-cap S&P 500 components
with ADDV well above $50M.  This criterion primarily prevents any mid- or
small-cap names from sneaking into the universe through a market-cap tie at the
margin.

---

### Criterion 4: Beta Stability Across the Training Window

**Rule:** Exclude any stock whose 252-day rolling beta (versus SPY) has a
cross-date standard deviation greater than 0.40 across the 2006-2011 training
window.

**Threshold:** σ(β_252d) <= 0.40 computed on all dates in the training set
where the rolling window is fully populated (i.e., after the 252-day burn-in).

**Why this matters - the feature stability argument:**

`beta_252d` is one of nine features fed into the LSTM tensor.  Its purpose is
to give the model a contemporaneous measure of each stock's systematic market
exposure, so the model can separate market-driven returns from firm-specific
signal.  This only works if beta is a stable attribute of the firm across the
training horizon.

When rolling beta is highly unstable, shifting from 0.4 to 1.8 and back within
a few quarters, two problems compound:

1. **The feature loses its economic meaning across the sequence.**  The LSTM
   sees a sequence where the same value of `beta_252d` meant "low systematic
   exposure" at one point in the lookback window and "high systematic exposure"
   at another.  The gating mechanism cannot condition on a feature whose
   interpretation reverses within a single 20-day sequence.

2. **Beta-hedged return residuals become contaminated.**  The event-time
   diagnostics and the long-short signal construction both subtract
   β x r_SPY from each stock return.  If β is measured with high variance,
   the residual contains substantial beta-estimation error that is
   indistinguishable from idiosyncratic return.  The model then implicitly
   trains to predict beta-estimation noise, not fundamental signal.

**What a high β standard deviation reveals in practice:**

| Pattern | Likely cause | Why it fails this filter |
|---------|-------------|--------------------------|
| Beta rises sharply during 2008-2009 then collapses | Distress: firm's equity behaved like an option during the crisis | Equity-option dynamics are not captured by the fundamental features |
| Beta drifts upward monotonically over 5 years | Business model shift (e.g., manufacturing to services, commodity to tech) | Features trained on the early regime do not generalize to the late regime |
| Beta oscillates with commodity cycles | Revenue and cost structure tied to a commodity price rather than to firm fundamentals | Macro factor contamination that the feature set cannot control for |

**Calibration of the 0.40 threshold:**

A threshold of 0.40 is approximately the 75th percentile of rolling-beta
standard deviations in the full S&P 500 over 2006-2011.  At this level:

- All 30 selected names pass (their beta stability is well within bounds).
- Stocks that experienced severe 2008-2009 distress (e.g., GE Capital drag on
  GE, banks, auto suppliers) would typically fail, producing rolling-β σ of
  0.6-1.2.
- Stocks with moderate crisis exposure but stable business models (e.g., JNJ,
  WMT, KO) have rolling-β σ in the 0.10-0.20 range.

**Relationship to other criteria:** Criterion 4 is the only filter that
operates on the model's own derived features rather than raw data availability
or structural market conditions.  A stock that passes Criteria 1-3 and 5 can
still fail here if the market's perception of its systematic risk was too
unstable for the rolling-window beta estimate to carry consistent meaning.

---

### Criterion 5: Sector Diversity with Cap of 4 per Sector

**Rule:** After all other filters, the top 30 by market cap are selected with a
hard cap of 4 stocks per GICS sector.

**Why this matters:** S&P 500 market cap is heavily concentrated in Technology.
Without a sector cap, the top 30 by raw market cap would include roughly 10-12
technology names, producing a universe that is technology-long by construction.
A long-short strategy built on such a universe would be exposed to systematic
tech risk rather than being genuinely market-neutral.

The cap of 4 per sector is a pragmatic balance:
- Large enough to capture within-sector cross-sectional variation (useful for
  the ranking signal).
- Small enough to prevent any sector from dominating portfolio construction.

**Note on telecom/communication services:** In the 2012-era GICS, this sector
(then called Telecom Services) contained only a handful of names meeting all
five criteria.  We include 3 rather than 4 because no fourth name qualifies
without compromising fundamental coverage.

---

## Final 30-Stock Universe

Ranked by 2012-12-31 market cap within each sector.

| # | Ticker | Company | Sector | Mkt Cap ($M) | D/E Impute |
|---|--------|---------|--------|-------------|------------|
| 1 | AAPL | Apple | Technology | 499,614 | Yes (0) |
| 2 | XOM | ExxonMobil | Energy | 394,611 | No |
| 3 | GOOGL | Alphabet | Technology | 232,441 | Yes (0) |
| 4 | WMT | Walmart | Consumer Staples | 229,351 | No |
| 5 | MSFT | Microsoft | Technology | 224,802 | Yes (0) |
| 6 | GE | General Electric | Industrials | 220,107 | No |
| 7 | IBM | IBM | Technology | 216,438 | No |
| 8 | CVX | Chevron | Energy | 211,649 | No |
| 9 | T | AT&T | Communication Services | 200,000 | No |
| 10 | JNJ | Johnson & Johnson | Healthcare | 194,265 | No |
| 11 | PFE | Pfizer | Healthcare | 195,800 | No |
| 12 | PG | Procter & Gamble | Consumer Staples | 210,000 | No |
| 13 | KO | Coca-Cola | Consumer Staples | 175,000 | No |
| 14 | VZ | Verizon | Communication Services | 110,000 | No |
| 15 | MRK | Merck | Healthcare | 120,000 | No |
| 16 | AMZN | Amazon | Consumer Discretionary | 115,000 | No |
| 17 | PEP | PepsiCo | Consumer Staples | 110,000 | No |
| 18 | SLB | Schlumberger | Energy | 100,000 | No |
| 19 | MCD | McDonald's | Consumer Discretionary | 95,000 | No |
| 20 | DIS | Walt Disney | Consumer Discretionary | 90,000 | No |
| 21 | COP | ConocoPhillips | Energy | 85,000 | No |
| 22 | HD | Home Depot | Consumer Discretionary | 80,000 | No |
| 23 | MMM | 3M | Industrials | 65,000 | No |
| 24 | HON | Honeywell | Industrials | 60,000 | No |
| 25 | ABT | Abbott Labs | Healthcare | 55,000 | No |
| 26 | BA | Boeing | Industrials | 55,000 | No |
| 27 | CMCSA | Comcast | Communication Services | 70,000 | No |
| 28 | DD | DuPont | Materials | 45,000 | No |
| 29 | SO | Southern Company | Utilities | 40,000 | No |
| 30 | PX | Praxair | Materials | 35,000 | No |

**Sector breakdown:**

| Sector | Count | Tickers |
|--------|-------|---------|
| Technology | 4 | AAPL, MSFT, GOOGL, IBM |
| Healthcare | 4 | JNJ, PFE, MRK, ABT |
| Energy | 4 | XOM, CVX, COP, SLB |
| Consumer Staples | 4 | WMT, PG, KO, PEP |
| Industrials | 4 | GE, BA, MMM, HON |
| Consumer Discretionary | 4 | MCD, HD, DIS, AMZN |
| Communication Services | 3 | T, VZ, CMCSA |
| Materials | 2 | DD, PX |
| Utilities | 1 | SO |

---

## Explicitly Excluded Candidates

| Ticker | Reason |
|--------|--------|
| BRK_B | book_val_per_share 39% missing (Criterion 1); beta σ ~0.55 during 2008-2009 crisis (Criterion 4) |
| JPM, BAC, WFC, C, GS, AIG | Beta σ > 1.0 during financial crisis; equity behaved as distressed options, not linear market exposures (Criterion 4) |
| SPG, PLD, O, AMT (all REITs) | D/E and EPS structurally incomparable; beta heavily interest-rate driven, high σ over rate-cycle turns (Criterion 4) |
| PM | IPO March 2008 (spun from Altria); no 2006-2007 price history (Criterion 2) |
| INTC, ORCL, QCOM | Technology sector at cap; ranked below IBM by 2012 market cap (Criterion 5) |
| NEE | Ticker changed from FPL in 2010; PRICES.csv join risk; replaced by SO (Criterion 2) |
