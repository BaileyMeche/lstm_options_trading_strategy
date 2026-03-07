from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.signal_construction import (
    build_long_only_signal_book,
    build_long_short_signal_book,
    generate_signal_books,
    rank_predictions_cross_sectionally,
    validate_no_lookahead,
)


def test_validate_no_lookahead_blocks_forward_columns() -> None:
    ok = pd.DataFrame({"date": ["2020-01-01"], "ticker": ["A"], "y_pred": [0.1]})
    validate_no_lookahead(ok)

    bad = ok.copy()
    bad["next_close"] = [123]
    with pytest.raises(ValueError, match="forward-look"):
        validate_no_lookahead(bad)


def test_rank_predictions_cross_sectionally_tie_breaks_and_handles_missing_predictions() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02"] * 4),
            "ticker": ["A", "B", "C", "D"],
            "y_pred": [0.5, 0.5, np.nan, 0.1],
            "volume": [100, 200, 300, 50],
        }
    )
    ranked = rank_predictions_cross_sectionally(df)
    by_ticker = ranked.set_index("ticker")

    assert by_ticker.loc["B", "pred_rank"] == 1
    assert by_ticker.loc["A", "pred_rank"] == 2
    assert by_ticker.loc["D", "pred_rank"] == 3
    assert np.isnan(by_ticker.loc["C", "pred_rank"])
    assert (ranked["n_assets"] == 3).all()


def test_build_long_short_and_long_only_signal_books() -> None:
    ranked = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-02", "2020-01-02"]),
            "ticker": ["A", "B", "C"],
            "pred_rank": [1, 2, 3],
            "n_assets": [3, 3, 3],
            "y_pred": [0.9, 0.1, -0.2],
            "y_true": [0.05, 0.0, -0.01],
        }
    )

    ls = build_long_short_signal_book(ranked, K=1)
    lo = build_long_only_signal_book(ranked, K=1)

    assert ls["ticker"].tolist() == ["A", "C"]
    assert ls["signal"].tolist() == [1, -1]
    assert lo["ticker"].tolist() == ["A"]
    assert (lo["signal"] == 1).all()


def test_build_long_short_skips_when_assets_insufficient() -> None:
    ranked = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-02", "2020-01-03", "2020-01-03"]),
            "ticker": ["A", "B", "A", "B"],
            "pred_rank": [1, 2, 1, 2],
            "n_assets": [2, 2, 2, 2],
            "y_pred": [0.4, 0.1, 0.3, 0.2],
        }
    )

    with pytest.warns(UserWarning, match="Skipping"):
        out = build_long_short_signal_book(ranked, K=2)
    assert out.empty


def test_generate_signal_books_writes_outputs(tmp_path: Path) -> None:
    pred_path = tmp_path / "preds.csv"
    out_dir = tmp_path / "signals"

    preds = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02"] * 4 + ["2020-01-03"] * 4),
            "ticker": ["A", "B", "C", "D", "A", "B", "C", "D"],
            "y_pred": [0.5, 0.2, -0.1, 0.4, 0.1, 0.3, 0.0, -0.2],
            "y_true": [0.01, 0.0, -0.02, 0.03, -0.01, 0.02, 0.0, -0.03],
            "volume": [100, 90, 80, 70, 110, 95, 85, 75],
            "split": ["test"] * 8,
        }
    )
    preds.to_csv(pred_path, index=False)

    books = generate_signal_books(str(pred_path), output_dir=str(out_dir), K=1)
    assert set(books.keys()) == {"long_short", "long_only"}
    assert (out_dir / "signal_book_long_short.csv").exists()
    assert (out_dir / "signal_book_long_only.csv").exists()
    assert not books["long_short"].empty


def test_generate_signal_books_accepts_prediction_alias_columns(tmp_path: Path) -> None:
    pred_path = tmp_path / "preds_alias.csv"
    out_dir = tmp_path / "signals_alias"

    preds = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02"] * 3),
            "ticker": ["A", "B", "C"],
            "prediction": [0.5, 0.1, -0.2],
            "actual_return": [0.01, 0.0, -0.01],
            "volume": [100, 90, 80],
        }
    )
    preds.to_csv(pred_path, index=False)

    books = generate_signal_books(str(pred_path), output_dir=str(out_dir), K=1)
    assert not books["long_only"].empty
