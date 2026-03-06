
# Stage 10 — Exposure Utilities
# Merges risk_utils outputs into canonical Stage 10 CSV outputs:
#   risk_exposure_daily.csv   — per-date wide metrics table
#   risk_events.csv           — de-risking trigger events

from __future__ import annotations

import os
from typing import Any

import pandas as pd

from .risk_utils import compute_all_risk_metrics


def _merge_optional(df: pd.DataFrame, other: pd.DataFrame | None, cols: list[str]) -> pd.DataFrame:
    if other is None or other.empty:
        for col in cols:
            if col not in df.columns:
                df[col] = float("nan")
        return df
    tmp = other.copy()
    tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
    present = ["date"] + [c for c in cols if c in tmp.columns]
    return df.merge(tmp[present], on="date", how="left")


def build_risk_exposure_daily(risk_results: dict[str, Any]) -> pd.DataFrame:
    """Merge all Stage 10 risk metrics into a single daily table."""
    equity = risk_results["equity"]
    dd_stats = risk_results["drawdown_stats"]
    rolling_risk = risk_results["rolling_risk"]
    daily_exp = risk_results.get("daily_exposure")
    concentration = risk_results.get("concentration")
    sector_conc = risk_results.get("sector_concentration")
    beta_exp = risk_results.get("beta_exposure")
    greeks_exp = risk_results.get("greeks_exposure")
    limits_df = risk_results.get("limits_df")

    spine = equity.rename_axis("date").reset_index()
    spine.columns = ["date", "equity"]
    spine["date"] = pd.to_datetime(spine["date"], errors="coerce")

    dd_series = dd_stats["drawdown_series"].rename_axis("date").reset_index()
    dd_series.columns = ["date", "drawdown"]
    dd_series["date"] = pd.to_datetime(dd_series["date"], errors="coerce")

    rr = rolling_risk.rename_axis("date").reset_index()
    rr.columns = ["date"] + list(rolling_risk.columns)
    rr["date"] = pd.to_datetime(rr["date"], errors="coerce")

    df = spine.merge(dd_series, on="date", how="left")
    df = df.merge(rr, on="date", how="left")

    df = _merge_optional(
        df, daily_exp,
        ["n_positions", "gross_exposure", "net_exposure", "net_delta_exposure", "long_option_value", "short_stock_value"],
    )
    df = _merge_optional(df, concentration, ["hhi", "max_weight_pct", "max_weight_ticker"])
    df = _merge_optional(df, sector_conc, ["sector_hhi", "max_sector_weight_pct", "max_weight_sector"])
    df = _merge_optional(df, beta_exp, ["net_beta_exposure", "gross_beta_exposure"])
    df = _merge_optional(df, greeks_exp, ["net_dollar_delta", "net_dollar_gamma", "net_dollar_vega", "net_dollar_theta"])
    df = _merge_optional(
        df, limits_df,
        [
            "gross_exposure_pct_equity", "net_exposure_pct_equity",
            "long_option_value_pct_equity", "short_stock_value_pct_equity",
            "net_beta_exposure_pct_equity", "gross_beta_exposure_pct_equity",
            "gross_exposure_limit_breach", "net_exposure_limit_breach",
            "beta_limit_breach", "single_name_limit_breach", "sector_limit_breach",
        ],
    )

    df["drawdown_regime"] = "normal"
    df.loc[df["drawdown"] <= -0.05, "drawdown_regime"] = "watch"
    df.loc[df["drawdown"] <= -0.10, "drawdown_regime"] = "de-risk"
    df.loc[df["drawdown"] <= -0.15, "drawdown_regime"] = "capital_preservation"

    return df.sort_values("date").reset_index(drop=True)


def run_stage10(
    trade_log_path: str,
    position_log_path: str | None = None,
    daily_pnl_path: str | None = None,
    output_dir: str = "data/results",
    beta_lookup: dict[str, float] | None = None,
    sector_lookup: dict[str, str] | None = None,
    initial_capital: float = 100_000.0,
    drawdown_threshold: float = -0.10,
    max_gross_exposure_pct: float = 0.10,
    max_net_exposure_pct: float = 0.05,
    max_abs_beta_exposure: float = 0.25,
    max_single_name_weight: float = 0.50,
    max_sector_weight: float = 0.60,
) -> dict[str, Any]:
    """Full Stage 10 pipeline: load data, compute metrics, write outputs."""
    trade_log = pd.read_csv(trade_log_path)
    for col in ["date", "signal_date", "expiry", "entry_date"]:
        if col in trade_log.columns:
            trade_log[col] = pd.to_datetime(trade_log[col], errors="coerce")

    position_log: pd.DataFrame | None = None
    if position_log_path is not None and os.path.exists(position_log_path):
        position_log = pd.read_csv(position_log_path)
        if "date" in position_log.columns:
            position_log["date"] = pd.to_datetime(position_log["date"], errors="coerce")

    daily_pnl_df: pd.DataFrame | None = None
    if daily_pnl_path is not None and os.path.exists(daily_pnl_path):
        daily_pnl_df = pd.read_csv(daily_pnl_path)
        if "date" in daily_pnl_df.columns:
            daily_pnl_df["date"] = pd.to_datetime(daily_pnl_df["date"], errors="coerce")

    risk_results = compute_all_risk_metrics(
        trade_log=trade_log,
        position_log=position_log,
        daily_pnl_df=daily_pnl_df,
        beta_lookup=beta_lookup,
        sector_lookup=sector_lookup,
        initial_capital=initial_capital,
        drawdown_threshold=drawdown_threshold,
        max_gross_exposure_pct=max_gross_exposure_pct,
        max_net_exposure_pct=max_net_exposure_pct,
        max_abs_beta_exposure=max_abs_beta_exposure,
        max_single_name_weight=max_single_name_weight,
        max_sector_weight=max_sector_weight,
    )

    risk_exposure_daily = build_risk_exposure_daily(risk_results)

    os.makedirs(output_dir, exist_ok=True)
    risk_exposure_path = os.path.join(output_dir, "risk_exposure_daily.csv")
    risk_events_path = os.path.join(output_dir, "risk_events.csv")
    stress_path = os.path.join(output_dir, "stress_scenarios.csv")

    risk_exposure_daily.to_csv(risk_exposure_path, index=False)
    risk_events_df = risk_results["risk_events"]
    risk_events_df.to_csv(risk_events_path, index=False)
    risk_results["stress_table"].to_csv(stress_path, index=False)

    print(
        f"[Stage 10] Outputs written:\n"
        f"  {risk_exposure_path} ({len(risk_exposure_daily)} rows)\n"
        f"  {risk_events_path} ({len(risk_events_df)} rows)\n"
        f"  {stress_path} ({len(risk_results['stress_table'])} rows)"
    )

    return {
        "risk_exposure_daily": risk_exposure_daily,
        "risk_events": risk_events_df,
        "trade_risk_stats": risk_results["trade_risk_stats"],
        "var_cvar_table": risk_results["var_cvar_table"],
        "drawdown_stats": risk_results["drawdown_stats"],
        "stress_table": risk_results["stress_table"],
        "paths": {
            "risk_exposure_path": risk_exposure_path,
            "risk_events_path": risk_events_path,
            "stress_path": stress_path,
        },
    }
