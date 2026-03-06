from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def build_sequence_dataset(
    panel_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "target_return",
    lookback: int = 60,
    date_col: str = "date",
    ticker_col: str = "ticker",
) -> dict[str, object]:
    """Build pooled LSTM sequences and metadata from a daily panel.

    Each output sample uses:
    - X: feature rows from t-lookback+1 to t (inclusive)
    - y: target at t (already encoded as return from t to t+1 in the panel)
    - sample_date: sequence end date t
    - label_date: target realization date t+1
    - ticker: asset identifier for sample row t
    """
    required = {date_col, ticker_col, target_col, *feature_cols}
    missing = sorted(required - set(panel_df.columns))
    if missing:
        raise KeyError(f"Missing columns for sequence build: {missing}")

    panel = panel_df.copy()
    panel[date_col] = pd.to_datetime(panel[date_col], errors="coerce")
    panel = panel.dropna(subset=[date_col, ticker_col]).sort_values([ticker_col, date_col]).reset_index(drop=True)

    x_parts: list[np.ndarray] = []
    y_parts: list[float] = []
    date_parts: list[np.datetime64] = []
    label_date_parts: list[np.datetime64] = []
    ticker_parts: list[str] = []

    for ticker, group in panel.groupby(ticker_col, sort=False):
        group = group.sort_values(date_col).copy()
        for col in feature_cols:
            group[col] = pd.to_numeric(group[col], errors="coerce")
        group[target_col] = pd.to_numeric(group[target_col], errors="coerce")

        feature_arr = group[feature_cols].to_numpy(dtype=np.float32)
        target_arr = group[target_col].to_numpy(dtype=np.float32)
        date_arr = group[date_col].to_numpy(dtype="datetime64[ns]")

        # X through t predicts return t->t+1; idx+1 must exist to capture label_date.
        for idx in range(lookback - 1, len(group) - 1):
            x_window = feature_arr[idx - lookback + 1 : idx + 1]
            y_val = target_arr[idx]
            sample_date = date_arr[idx]
            label_date = date_arr[idx + 1]

            if not np.isfinite(x_window).all() or not np.isfinite(y_val):
                continue
            if np.isnat(sample_date) or np.isnat(label_date):
                continue

            x_parts.append(x_window)
            y_parts.append(float(y_val))
            date_parts.append(sample_date)
            label_date_parts.append(label_date)
            ticker_parts.append(str(ticker))

    if not x_parts:
        return {
            "X_sequences": np.empty((0, lookback, len(feature_cols)), dtype=np.float32),
            "y_targets": np.empty((0,), dtype=np.float32),
            "sample_dates": np.empty((0,), dtype="datetime64[ns]"),
            "label_dates": np.empty((0,), dtype="datetime64[ns]"),
            "tickers": np.empty((0,), dtype=str),
            "lookback": int(lookback),
            "feature_cols": list(feature_cols),
            "target_col": target_col,
        }

    return {
        "X_sequences": np.stack(x_parts).astype(np.float32),
        "y_targets": np.asarray(y_parts, dtype=np.float32),
        "sample_dates": np.asarray(date_parts, dtype="datetime64[ns]"),
        "label_dates": np.asarray(label_date_parts, dtype="datetime64[ns]"),
        "tickers": np.asarray(ticker_parts, dtype=str),
        "lookback": int(lookback),
        "feature_cols": list(feature_cols),
        "target_col": target_col,
    }


def save_sequence_dataset_npz(sequence_data: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_dates_int = pd.to_datetime(sequence_data["sample_dates"]).asi8
    save_payload: dict[str, np.ndarray] = {
        "X_sequences": np.asarray(sequence_data["X_sequences"], dtype=np.float32),
        "y_targets": np.asarray(sequence_data["y_targets"], dtype=np.float32),
        "sample_dates_ns": np.asarray(sample_dates_int, dtype=np.int64),
        "tickers": np.asarray(sequence_data["tickers"], dtype=str),
        "lookback": np.asarray([int(sequence_data["lookback"])], dtype=np.int32),
        "feature_cols": np.asarray(sequence_data["feature_cols"], dtype=str),
        "target_col": np.asarray([str(sequence_data["target_col"])], dtype=str),
    }

    label_dates = sequence_data.get("label_dates")
    if label_dates is not None:
        label_dates_int = pd.to_datetime(label_dates).asi8
        save_payload["label_dates_ns"] = np.asarray(label_dates_int, dtype=np.int64)

    np.savez_compressed(output_path, **save_payload)


def load_sequence_dataset_npz(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as npz:
        sample_dates = pd.to_datetime(npz["sample_dates_ns"], unit="ns").to_numpy(dtype="datetime64[ns]")
        label_dates: np.ndarray | None = None
        if "label_dates_ns" in npz.files:
            label_dates = pd.to_datetime(npz["label_dates_ns"], unit="ns").to_numpy(dtype="datetime64[ns]")
        return {
            "X_sequences": np.asarray(npz["X_sequences"], dtype=np.float32),
            "y_targets": np.asarray(npz["y_targets"], dtype=np.float32),
            "sample_dates": sample_dates,
            "label_dates": label_dates,
            "tickers": np.asarray(npz["tickers"], dtype=str),
            "lookback": int(npz["lookback"][0]),
            "feature_cols": [str(x) for x in npz["feature_cols"].tolist()],
            "target_col": str(npz["target_col"][0]),
        }


class PooledLSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        h_last = out[:, -1, :]
        h_last = self.dropout(h_last)
        return self.output(h_last).squeeze(-1)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_pooled_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    input_size: int,
    hidden_size: int = 64,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
    epochs: int = 10,
    batch_size: int = 256,
    seed: int = 42,
    device: str | None = None,
) -> PooledLSTMRegressor:
    import random

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if X_train.shape[0] == 0:
        raise ValueError("Cannot train LSTM with zero training samples.")

    target_device = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = PooledLSTMRegressor(input_size=input_size, hidden_size=hidden_size, dropout=dropout).to(target_device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    dataset = TensorDataset(
        torch.from_numpy(X_train.astype(np.float32)),
        torch.from_numpy(y_train.astype(np.float32)),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(target_device)
            yb = yb.to(target_device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

    return model


def predict_pooled_lstm(
    model: PooledLSTMRegressor,
    X: np.ndarray,
    batch_size: int = 1024,
    device: str | None = None,
) -> np.ndarray:
    if X.shape[0] == 0:
        return np.empty((0,), dtype=np.float32)

    target_device = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model.eval()

    loader = DataLoader(TensorDataset(torch.from_numpy(X.astype(np.float32))), batch_size=batch_size, shuffle=False)
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(target_device)
            out = model(xb).detach().cpu().numpy().astype(np.float32)
            preds.append(out)

    return np.concatenate(preds, axis=0) if preds else np.empty((0,), dtype=np.float32)


def walk_forward_lstm_predictions(
    X_sequences: np.ndarray,
    y_targets: np.ndarray,
    sample_dates: np.ndarray,
    tickers: np.ndarray,
    label_dates: np.ndarray | None = None,
    train_start_year: int = 2006,
    first_predict_year: int = 2010,
    max_predict_year: int | None = None,
    min_train_samples: int = 200,
    min_predict_samples: int = 1,
    hidden_size: int = 64,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
    epochs: int = 10,
    batch_size: int = 256,
    seed: int = 42,
    device: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(X_sequences) != len(y_targets) or len(X_sequences) != len(sample_dates) or len(X_sequences) != len(tickers):
        raise ValueError("Input arrays must share the same sample count.")
    if label_dates is not None and len(X_sequences) != len(label_dates):
        raise ValueError("label_dates must share the same sample count as sequences.")
    if len(X_sequences) == 0:
        raise ValueError("Sequence dataset is empty.")

    decision_dates = pd.to_datetime(sample_dates)
    if label_dates is not None:
        dates_for_split = pd.to_datetime(label_dates)
        output_label_dates = pd.to_datetime(label_dates).to_numpy(dtype="datetime64[ns]")
    else:
        print(
            "[walk-forward] WARNING: label_dates not provided; using sample_dates for split. "
            "Year-boundary leakage may exist."
        )
        dates_for_split = decision_dates
        output_label_dates = np.full(len(sample_dates), np.datetime64("NaT"), dtype="datetime64[ns]")

    years = sorted(pd.Index(dates_for_split.year).unique().tolist())
    predict_years = [year for year in years if year >= first_predict_year]
    if max_predict_year is not None:
        predict_years = [year for year in predict_years if year <= max_predict_year]

    if not predict_years:
        raise ValueError("No prediction years available after filtering by first_predict_year/max_predict_year.")

    predictions_parts: list[pd.DataFrame] = []
    walk_rows: list[dict[str, object]] = []
    train_start_date = pd.Timestamp(f"{train_start_year}-01-01")

    for pred_year in predict_years:
        train_end_date = pd.Timestamp(f"{pred_year - 1}-12-31")
        pred_start_date = pd.Timestamp(f"{pred_year}-01-01")
        pred_end_date = pd.Timestamp(f"{pred_year}-12-31")

        train_mask = (dates_for_split >= train_start_date) & (dates_for_split <= train_end_date)
        pred_mask = (dates_for_split >= pred_start_date) & (dates_for_split <= pred_end_date)

        n_train = int(train_mask.sum())
        n_pred = int(pred_mask.sum())

        print(
            "[walk-forward] "
            f"train={train_start_year}-{pred_year - 1} "
            f"predict_year={pred_year} "
            f"train_samples={n_train:,} "
            f"predict_samples={n_pred:,}"
        )

        if n_train < min_train_samples or n_pred < min_predict_samples:
            print(
                "[walk-forward] skip "
                f"predict_year={pred_year} due to insufficient data "
                f"(min_train_samples={min_train_samples}, min_predict_samples={min_predict_samples})"
            )
            walk_rows.append(
                {
                    "prediction_year": pred_year,
                    "train_period": f"{train_start_year}-{pred_year - 1}",
                    "predict_period": f"{pred_year}",
                    "train_samples": n_train,
                    "predict_samples": n_pred,
                    "status": "skipped_insufficient_data",
                }
            )
            continue

        model = train_pooled_lstm(
            X_train=X_sequences[train_mask],
            y_train=y_targets[train_mask],
            input_size=X_sequences.shape[2],
            hidden_size=hidden_size,
            dropout=dropout,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
            device=device,
        )
        pred_values = predict_pooled_lstm(model=model, X=X_sequences[pred_mask], batch_size=1024, device=device)

        fold_df = pd.DataFrame(
            {
                "date": pd.to_datetime(decision_dates[pred_mask]).to_numpy(dtype="datetime64[ns]"),
                "label_date": pd.to_datetime(output_label_dates[pred_mask]).to_numpy(dtype="datetime64[ns]"),
                "ticker": np.asarray(tickers[pred_mask], dtype=str),
                "prediction": pred_values.astype(np.float32),
                "actual_return": np.asarray(y_targets[pred_mask], dtype=np.float32),
                "prediction_year": pred_year,
            }
        )
        predictions_parts.append(fold_df)

        walk_rows.append(
            {
                "prediction_year": pred_year,
                "train_period": f"{train_start_year}-{pred_year - 1}",
                "predict_period": f"{pred_year}",
                "train_samples": n_train,
                "predict_samples": n_pred,
                "status": "trained",
            }
        )

    if predictions_parts:
        predictions_df = pd.concat(predictions_parts, ignore_index=True)
        predictions_df["date"] = pd.to_datetime(predictions_df["date"], errors="coerce")
        predictions_df = predictions_df.sort_values(["date", "ticker"]).reset_index(drop=True)
    else:
        predictions_df = pd.DataFrame(
            columns=["date", "label_date", "ticker", "prediction", "actual_return", "prediction_year"]
        )

    walk_summary = pd.DataFrame(walk_rows)
    return predictions_df, walk_summary
