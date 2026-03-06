# Stage 8 — Cross-Sectional Ranking
# Converts LSTM predictions into ranked signals using SUE-style cross-sectional normalization.
# Predictions are already produced by the walk-forward LSTM and embed fundamental momentum
# (EPS growth, ROE change, etc.), so we rank directly on the prediction value.
# Timing: prediction available after close t → rank formed after close t → entry at open t+1.

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def rank_predictions(
    predictions_df: pd.DataFrame,
    date_col: str = "date",
    pred_col: str = "prediction",
    ticker_col: str = "ticker",
) -> pd.DataFrame:
    """Rank LSTM predictions cross-sectionally per date.

    Rank 1 = highest predicted return (strongest long signal).
    Within ties, the first occurrence in sorted order wins (method='first').
    The LSTM walk-forward predictions are implicitly cross-sectionally normalized
    because the model is trained on z-scored features, mirroring SUE-style
    standardization without requiring explicit earnings surprise data.

    Parameters
    ----------
    predictions_df : DataFrame with columns [date_col, ticker_col, pred_col].
    date_col, pred_col, ticker_col : Column name overrides.

    Returns
    -------
    Input DataFrame with added columns ``rank`` (int, 1 = best) and
    ``n_assets`` (int, number of assets ranked on that date).
    Sorted by date asc, rank asc.
    """
    df = predictions_df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, ticker_col, pred_col]).reset_index(drop=True)

    df = df.sort_values([date_col, pred_col], ascending=[True, False])
    df["rank"] = (
        df.groupby(date_col)[pred_col]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    df["n_assets"] = df.groupby(date_col)[pred_col].transform("count").astype(int)

    return df.sort_values([date_col, "rank"]).reset_index(drop=True)


def select_top_k(
    ranked_df: pd.DataFrame,
    K: int = 3,
    date_col: str = "date",
    rank_col: str = "rank",
    n_assets_col: str = "n_assets",
) -> pd.DataFrame:
    """Filter to top-K ranked stocks per date (long-only signal for options strategy).

    Dates with fewer than K available assets are skipped with a warning.

    Parameters
    ----------
    ranked_df : Output of :func:`rank_predictions`.
    K : Number of top stocks to select per date.

    Returns
    -------
    Filtered DataFrame containing only rows where rank <= K, with added column ``K``.
    """
    parts: list[pd.DataFrame] = []

    for date_val, group in ranked_df.groupby(date_col, sort=True):
        n = int(group[n_assets_col].iloc[0])
        if n < K:
            warnings.warn(
                f"[ranking] {date_val}: only {n} assets available, need {K}. Skipping.",
                stacklevel=2,
            )
            continue
        top = group[group[rank_col] <= K].copy()
        top["K"] = K
        parts.append(top)

    if not parts:
        return pd.DataFrame()

    return pd.concat(parts, ignore_index=True)


def build_earnings_signals(
    predictions_df: pd.DataFrame,
    feature_panel_df: pd.DataFrame,
    K: int = 3,
    date_col: str = "date",
    pred_col: str = "prediction",
    ticker_col: str = "ticker",
    earnings_col: str = "feature_available_date",
) -> pd.DataFrame:
    """Generate entry signals triggered by earnings announcement dates.

    For each earnings event (ticker + announcement date):
      - signal_date = last prediction date strictly BEFORE the announcement
      - `date` col is set to signal_date so build_entry_table finds
        the next option date (= earnings day) as the actual entry
      - Stocks are ranked cross-sectionally among those sharing the same
        announcement date; top-K are selected.

    Parameters
    ----------
    predictions_df   : LSTM predictions with columns [date, ticker, prediction].
    feature_panel_df : Feature panel with columns [ticker, feature_available_date].
    K                : Number of top stocks to select per earnings date.
    earnings_col     : Column in feature_panel_df containing announcement dates.

    Returns
    -------
    DataFrame with columns: date (signal_date), entry_date_hint, ticker,
    prediction, rank, n_assets, K.
    """
    predictions_df = predictions_df.copy()
    predictions_df[date_col] = pd.to_datetime(predictions_df[date_col], errors="coerce")

    pred_dates = sorted(predictions_df[date_col].dropna().unique())

    # Build fast (ticker, date) -> prediction lookup
    pred_index = (
        predictions_df.dropna(subset=[date_col, ticker_col, pred_col])
        .set_index([ticker_col, date_col])[pred_col]
        .to_dict()
    )

    # Extract unique earnings events
    earnings = (
        feature_panel_df[[ticker_col, earnings_col]]
        .drop_duplicates(subset=[ticker_col, earnings_col])
        .dropna(subset=[earnings_col])
        .copy()
    )
    earnings[earnings_col] = pd.to_datetime(earnings[earnings_col], errors="coerce")
    earnings = earnings.dropna(subset=[earnings_col])

    # Restrict to dates covered by predictions
    min_pred = min(pred_dates) if pred_dates else pd.Timestamp("2100-01-01")
    max_pred = max(pred_dates) if pred_dates else pd.Timestamp("1900-01-01")
    earnings = earnings[
        (earnings[earnings_col] > min_pred) &
        (earnings[earnings_col] <= max_pred)
    ]

    if earnings.empty:
        warnings.warn("[ranking] No earnings events overlap with prediction dates.", stacklevel=2)
        return pd.DataFrame()

    records: list[dict] = []
    for _, row in earnings.iterrows():
        ticker = str(row[ticker_col])
        ann_date = row[earnings_col]

        # Find the last prediction date strictly before the announcement
        signal_date = None
        for d in reversed(pred_dates):
            if d < ann_date:
                signal_date = d
                break
        if signal_date is None:
            continue

        pred_val = pred_index.get((ticker, signal_date))
        if pred_val is None:
            continue

        records.append({
            "signal_date":    signal_date,
            "date":           signal_date,   # build_entry_table uses this to find next option date
            "entry_date_hint": ann_date,
            "ticker":         ticker,
            pred_col:         float(pred_val),
        })

    if not records:
        warnings.warn("[ranking] No earnings signals generated — check date overlap.", stacklevel=2)
        return pd.DataFrame()

    signals_df = pd.DataFrame(records)

    # Cross-sectional rank within each announcement date group
    signals_df["rank"] = (
        signals_df.groupby("entry_date_hint")[pred_col]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    signals_df["n_assets"] = (
        signals_df.groupby("entry_date_hint")[ticker_col]
        .transform("count")
        .astype(int)
    )

    # Keep top-K per announcement date
    top_k = signals_df[signals_df["rank"] <= K].copy()
    top_k["K"] = K

    n_events = top_k["entry_date_hint"].nunique()
    print(f"[ranking] earnings signals: K={K} | earnings events={n_events} | total rows={len(top_k)}")
    return top_k.sort_values("date").reset_index(drop=True)


def build_signal_table(
    predictions_df: pd.DataFrame,
    K: int = 3,
    date_col: str = "date",
    pred_col: str = "prediction",
    ticker_col: str = "ticker",
) -> pd.DataFrame:
    """Orchestrate ranking → top-K selection into a single signal table.

    Parameters
    ----------
    predictions_df : Raw predictions CSV as DataFrame.
    K : Number of long positions per date.

    Returns
    -------
    DataFrame with columns: date, ticker, prediction, rank, n_assets, K.
    """
    ranked = rank_predictions(predictions_df, date_col=date_col, pred_col=pred_col, ticker_col=ticker_col)
    top_k = select_top_k(ranked, K=K, date_col=date_col)

    n_dates = top_k[date_col].nunique() if not top_k.empty else 0
    print(f"[ranking] K={K} | signal dates={n_dates} | total rows={len(top_k)}")
    return top_k
