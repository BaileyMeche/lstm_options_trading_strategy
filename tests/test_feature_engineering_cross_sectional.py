import numpy as np
import pandas as pd

from src.feature_engineering import (
    get_cross_section_rank_feature_columns,
    rank_cross_sectional,
    zscore_cross_sectional,
)


def test_get_cross_section_rank_feature_columns_default_exclusions() -> None:
    feature_cols = [
        "roe_change",
        "margin_change",
        "log_return",
        "rolling_beta",
        "vol_regime_ratio_20_60",
        "reaction_speed",
    ]

    rank_cols = get_cross_section_rank_feature_columns(feature_cols)
    assert rank_cols == ["roe_change", "margin_change", "reaction_speed"]


def test_rank_cross_sectional_matches_half_step_formula() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-03",
                    "2024-01-03",
                ]
            ),
            "ticker": ["A", "B", "C", "D", "A", "B", "C"],
            "eps_growth": [0.10, 0.20, 0.30, np.nan, 0.02, 0.02, 0.15],
        }
    )

    ranked = rank_cross_sectional(df, feature_cols=["eps_growth"], date_col="date", suffix="_rank", center=False)
    centered = rank_cross_sectional(df, feature_cols=["eps_growth"], date_col="date", suffix="_rank_c", center=True)

    d1 = ranked[ranked["date"] == pd.Timestamp("2024-01-02")].set_index("ticker")["eps_growth_rank"]
    d2 = ranked[ranked["date"] == pd.Timestamp("2024-01-03")].set_index("ticker")["eps_growth_rank"]

    assert np.isclose(d1["A"], (1.0 - 0.5) / 3.0)
    assert np.isclose(d1["B"], (2.0 - 0.5) / 3.0)
    assert np.isclose(d1["C"], (3.0 - 0.5) / 3.0)
    assert np.isnan(d1["D"])

    # Tie on 2024-01-03: A and B both get average rank 1.5.
    assert np.isclose(d2["A"], (1.5 - 0.5) / 3.0)
    assert np.isclose(d2["B"], (1.5 - 0.5) / 3.0)
    assert np.isclose(d2["C"], (3.0 - 0.5) / 3.0)

    centered_vals = centered[centered["date"] == pd.Timestamp("2024-01-03")].set_index("ticker")["eps_growth_rank_c"]
    assert np.isclose(centered_vals["C"], ((3.0 - 0.5) / 3.0) - 0.5)


def test_zscore_cross_sectional_uses_source_overrides() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-02", "2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]
            ),
            "ticker": ["A", "B", "C", "A", "B"],
            "foo": [1.0, 2.0, 2.0, 5.0, 7.0],
            "foo_rank": [0.1, 0.6, 0.9, 0.2, 0.8],
            "bar": [10.0, 20.0, 30.0, 40.0, 60.0],
        }
    )

    out = zscore_cross_sectional(
        panel_df=df,
        feature_cols=["foo", "bar"],
        date_col="date",
        suffix="_z",
        source_overrides={"foo": "foo_rank"},
    )

    d1 = out[out["date"] == pd.Timestamp("2024-01-02")].set_index("ticker")
    d2 = out[out["date"] == pd.Timestamp("2024-01-03")].set_index("ticker")

    # Expected from foo_rank (not from foo raw): mean=0.5333, std=0.32998.
    assert np.isclose(d1.loc["B", "foo_z"], (0.6 - (0.1 + 0.6 + 0.9) / 3.0) / np.std([0.1, 0.6, 0.9], ddof=0))
    # 2-point z-score for 2024-01-03 foo_rank values [0.2, 0.8].
    assert np.isclose(d2.loc["A", "foo_z"], -1.0)
    assert np.isclose(d2.loc["B", "foo_z"], 1.0)

    # bar_z is still sourced from bar.
    assert np.isclose(d1.loc["A", "bar_z"], -1.224744871391589)


def test_rank_cross_sectional_respects_universe_col() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"] * 4),
            "ticker": ["A", "B", "C", "OUT"],
            "in_universe": [True, True, True, False],
            "eps_growth": [0.10, 0.20, 0.30, 999.0],  # out-of-universe outlier
        }
    )

    ranked = rank_cross_sectional(
        df,
        feature_cols=["eps_growth"],
        date_col="date",
        suffix="_rank",
        center=True,
        universe_col="in_universe",
    )
    vals = ranked.set_index("ticker")["eps_growth_rank"]

    # Universe members are ranked against each other only, then centered.
    assert np.isclose(vals["A"], (1.0 - 0.5) / 3.0 - 0.5)
    assert np.isclose(vals["B"], (2.0 - 0.5) / 3.0 - 0.5)
    assert np.isclose(vals["C"], (3.0 - 0.5) / 3.0 - 0.5)
    # Non-universe row is excluded.
    assert np.isnan(vals["OUT"])
