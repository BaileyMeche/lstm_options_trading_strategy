from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import model_utils
from src.model_utils import (
    PooledLSTMRegressor,
    build_sequence_dataset,
    load_sequence_dataset_npz,
    predict_pooled_lstm,
    save_sequence_dataset_npz,
    train_pooled_lstm,
    walk_forward_lstm_predictions,
)


def test_build_sequence_dataset_outputs_alignment_and_validates_lookback() -> None:
    panel = pd.DataFrame(
        {
            "ticker": ["A", "A", "A", "A", "B", "B", "B"],
            "date": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06", "2020-01-01", "2020-01-02", "2020-01-03"]
            ),
            "f1": [1, 2, 3, 4, 1, np.nan, 3],
            "f2": [2, 3, 4, 5, 2, 3, 4],
            "target_return": [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3],
        }
    )

    out = build_sequence_dataset(panel, feature_cols=["f1", "f2"], lookback=2)
    assert out["X_sequences"].shape[1:] == (2, 2)
    assert out["X_sequences"].shape[0] == 2
    assert out["y_targets"].shape == (2,)
    assert out["sample_dates"].shape == out["label_dates"].shape
    assert np.all(out["label_dates"] > out["sample_dates"])

    with pytest.raises(ValueError, match="lookback"):
        build_sequence_dataset(panel, feature_cols=["f1", "f2"], lookback=0)


def test_sequence_dataset_npz_round_trip(tmp_path: Path) -> None:
    seq = {
        "X_sequences": np.ones((2, 3, 2), dtype=np.float32),
        "y_targets": np.array([0.1, -0.2], dtype=np.float32),
        "sample_dates": np.array(["2020-01-03", "2020-01-06"], dtype="datetime64[ns]"),
        "label_dates": np.array(["2020-01-06", "2020-01-07"], dtype="datetime64[ns]"),
        "tickers": np.array(["A", "B"], dtype=str),
        "lookback": 3,
        "feature_cols": ["f1", "f2"],
        "target_col": "target_return",
    }
    path = tmp_path / "seq.npz"
    save_sequence_dataset_npz(seq, path)

    loaded = load_sequence_dataset_npz(path)
    assert loaded["X_sequences"].shape == (2, 3, 2)
    assert loaded["feature_cols"] == ["f1", "f2"]
    assert loaded["lookback"] == 3


def test_train_and_predict_pooled_lstm_smoke() -> None:
    X = np.random.RandomState(0).randn(8, 4, 3).astype(np.float32)
    y = np.random.RandomState(1).randn(8).astype(np.float32)

    model = train_pooled_lstm(X, y, input_size=3, epochs=1, batch_size=4, hidden_size=8, seed=7)
    assert isinstance(model, PooledLSTMRegressor)

    pred = predict_pooled_lstm(model, X[:3], batch_size=2)
    assert pred.shape == (3,)


def test_walk_forward_lstm_predictions_with_label_dates(monkeypatch: pytest.MonkeyPatch) -> None:
    X = np.ones((4, 2, 2), dtype=np.float32)
    y = np.array([0.1, 0.2, -0.1, 0.3], dtype=np.float32)
    sample_dates = np.array(["2008-01-02", "2009-01-02", "2010-01-04", "2010-06-01"], dtype="datetime64[ns]")
    label_dates = np.array(["2008-01-03", "2009-01-05", "2010-01-05", "2010-06-02"], dtype="datetime64[ns]")
    tickers = np.array(["A", "A", "A", "B"], dtype=str)

    monkeypatch.setattr(model_utils, "train_pooled_lstm", lambda **kwargs: "dummy_model")
    monkeypatch.setattr(
        model_utils,
        "predict_pooled_lstm",
        lambda model, X, batch_size=1024, device=None: np.full(X.shape[0], 0.5, dtype=np.float32),
    )

    pred_df, summary = walk_forward_lstm_predictions(
        X_sequences=X,
        y_targets=y,
        sample_dates=sample_dates,
        label_dates=label_dates,
        tickers=tickers,
        train_start_year=2008,
        first_predict_year=2010,
        min_train_samples=2,
        min_predict_samples=1,
        epochs=1,
    )

    assert len(pred_df) == 2
    assert set(pred_df.columns) >= {"date", "label_date", "ticker", "prediction", "actual_return"}
    assert summary.loc[0, "status"] == "trained"


def test_walk_forward_lstm_predictions_rejects_mismatched_lengths() -> None:
    X = np.ones((2, 2, 1), dtype=np.float32)
    y = np.array([0.1, 0.2], dtype=np.float32)
    sample_dates = np.array(["2010-01-01", "2010-01-02"], dtype="datetime64[ns]")
    tickers = np.array(["A", "B"], dtype=str)
    bad_label_dates = np.array(["2010-01-02"], dtype="datetime64[ns]")

    with pytest.raises(ValueError, match="label_dates"):
        walk_forward_lstm_predictions(
            X_sequences=X,
            y_targets=y,
            sample_dates=sample_dates,
            label_dates=bad_label_dates,
            tickers=tickers,
        )
