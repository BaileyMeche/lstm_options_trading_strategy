# README — data

# Description for `tips_treasury_implied_rf_2010.csv`
## What this file is

A **daily panel of curve-implied yields and the TIPS–Treasury wedge** at four standard tenors (2y, 5y, 10y, 20y). Each row is one trading date (`date`). The dataset contains:

* **Real TIPS yields** (`real_cc*`) in **decimal yield units** (e.g., `0.0070` = 0.70%).
* **Nominal Treasury zero-coupon yields** (`nom_zc*`) in **basis points** (bp).
* **TIPS-implied nominal (“synthetic nominal”) yields** (`tips_treas_*_rf`) in **bp**.
* **Arbitrage wedge** (`arb_*`) in **bp**, defined mechanically as the difference between the synthetic nominal yield and the nominal Treasury yield at the same tenor.

The first few rows you shared (starting 2010-01-04) already verify these identities numerically.

---

## Key identity (verified from your sample rows)

For each tenor (x \in {2,5,10,20}),
[
\texttt{arb_x} = \texttt{tips_treas_x_rf} - \texttt{nom_zcx}
]

Example (2010-01-04, 5y), using your pasted values:

* `tips_treas_5_rf = 306.99941054930235`
* `nom_zc5 = 268.43978045888895`
* Difference = `38.5596300904134` ≈ `arb_5 = 38.55963009041341`

So the wedge columns are not “estimated separately”—they are **exactly constructed** from the two yield series.

---

## Economic interpretation (consistent with the sign convention in this file)

* **Positive `arb_x`** means:
  **TIPS-implied nominal yield > nominal Treasury yield** at tenor (x).
  Equivalently, the **nominal Treasury is “rich”** relative to the synthetic nominal constructed from TIPS (the synthetic looks “cheap” on a yield basis).

* **Negative `arb_x`** means the opposite: the synthetic is rich / Treasury is cheap.

In your sample, `arb_5`, `arb_10`, and `arb_20` are positive on all shown dates; `arb_2` is also positive in the sample.

---

## Units and scale

Your sample confirms mixed units:

### Decimal yields

* `real_cc2`, `real_cc5`, `real_cc10`, `real_cc20`
  These are **real yields in decimals** (e.g., `-0.00225` = -0.225%).
  Example (2010-01-04): `real_cc5 = 0.007009999752...` ≈ 0.701%.

### Basis points (bp)

* `nom_zc2`, `nom_zc5`, `nom_zc10`, `nom_zc20`
* `tips_treas_2_rf`, `tips_treas_5_rf`, `tips_treas_10_rf`, `tips_treas_20_rf`
* `arb_2`, `arb_5`, `arb_10`, `arb_20`

Example (2010-01-04): `nom_zc10 = 428.0479` bp ≈ 4.280%.

This “decimals for real curve, bp for nominal/synthetic/wedge” convention is critical: don’t mix them in regressions without converting.

---

## Column dictionary (exactly as in your header)

### Date

* `date`
  Trading date in `YYYY-MM-DD`.

### Real yield curve inputs (decimal yields)

* `real_cc2`  — 2-year real yield (decimal)
* `real_cc5`  — 5-year real yield (decimal)
* `real_cc10` — 10-year real yield (decimal)
* `real_cc20` — 20-year real yield (decimal)

### Nominal Treasury zero-coupon yields (bp)

* `nom_zc2`  — 2-year nominal zero-coupon yield (bp)
* `nom_zc5`  — 5-year nominal zero-coupon yield (bp)
* `nom_zc10` — 10-year nominal zero-coupon yield (bp)
* `nom_zc20` — 20-year nominal zero-coupon yield (bp)

### TIPS-implied nominal (“synthetic nominal”, bp)

* `tips_treas_2_rf`  — 2-year TIPS-implied nominal yield (bp)
* `tips_treas_5_rf`  — 5-year TIPS-implied nominal yield (bp)
* `tips_treas_10_rf` — 10-year TIPS-implied nominal yield (bp)
* `tips_treas_20_rf` — 20-year TIPS-implied nominal yield (bp)

Interpretation: the nominal yield you would infer from the TIPS side (plus inflation compensation construction in your pipeline), expressed in bp.

### Arbitrage wedge (bp)

* `arb_2`  — 2-year wedge (bp) = `tips_treas_2_rf  - nom_zc2`
* `arb_5`  — 5-year wedge (bp) = `tips_treas_5_rf  - nom_zc5`
* `arb_10` — 10-year wedge (bp) = `tips_treas_10_rf - nom_zc10`
* `arb_20` — 20-year wedge (bp) = `tips_treas_20_rf - nom_zc20`

---

## What you can do with this file (and what you can’t)

### Supported cleanly

* Event studies / policy-window analyses using `arb_*` as outcomes (daily frequency).
* Tenor-heterogeneous dynamics (short end vs long end).
* Decompositions/diagnostics using `nom_zc*` vs `tips_treas_*_rf`.

### Not contained here

* No CUSIPs, no issue-level identifiers, no dealer IDs, no TRACE fields, no transaction volumes.
  This is **curve-level** data only.

---

## Minimal QA checks you should run on load

1. Verify wedge identity for each tenor (within floating rounding):

* `arb_x` equals `tips_treas_x_rf - nom_zcx`

2. Verify unit sanity:

* `real_cc*` should look like small decimals (e.g., -0.03 to +0.05-ish).
* `nom_zc*` and `tips_treas_*_rf` should look like bp (tens to hundreds or thousands depending on sample period).
* `arb_*` should look like bp (often tens, can spike during stress).
