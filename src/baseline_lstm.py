"""
baseline_lstm.py
================
Baseline LSTM for BIST 100 log return forecasting.

Project: Uncertainty-Aware Financial Forecasting (COM0415 Applied Deep Learning)
Authors: Lara Yıldırım, Belkıs Derin Yüksel

Purpose
-------
Interim-submission baseline. Predicts next-day BIST 100 log return using a
30-day lookback over 5 features (BIST100 Close, S&P500 Close, CPI, FEDFUNDS, INDPRO).

Usage
-----
    python baseline_lstm.py --data path/to/financial_data.csv

Expects a CSV with a DateTimeIndex and these columns (order-insensitive):
    BIST100_Close, SP500_Close, CPIAUCSL, FEDFUNDS, INDPRO
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
FEATURES = ["BIST100_Close", "SP500_Close", "CPIAUCSL", "FEDFUNDS", "INDPRO"]
LOOKBACK = 30
TRAIN_END = "2022-12-31"
VAL_END = "2023-12-31"
BATCH = 64
EPOCHS = 50
LR = 1e-3
PATIENCE = 7
SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------
def load_and_prepare(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True).sort_index()
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df[FEATURES].dropna()
    # Next-day BIST log return as target
    df["log_return_next"] = np.log(df["BIST100_Close"].shift(-1) / df["BIST100_Close"])
    return df.dropna()


def chronological_split(df: pd.DataFrame):
    train = df.loc[:TRAIN_END]
    val = df.loc[(df.index > TRAIN_END) & (df.index <= VAL_END)]
    test = df.loc[df.index > VAL_END]
    return train, val, test


def make_windows(X: np.ndarray, y: np.ndarray, T: int = LOOKBACK):
    """Sliding windows: sample ending at index i uses X[i-T+1 : i+1] to predict y[i]."""
    X_win, y_win = [], []
    for i in range(T - 1, len(X)):
        X_win.append(X[i - T + 1 : i + 1])
        y_win.append(y[i])
    return np.stack(X_win), np.array(y_win, dtype=np.float32)


class TSDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        return self.X[i], self.y[i]


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------
class LSTMBaseline(nn.Module):
    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


# ------------------------------------------------------------------
# Train / Eval
# ------------------------------------------------------------------
def train_model(model, train_loader, val_loader, device, epochs=EPOCHS, lr=LR, patience=PATIENCE):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = float("inf")
    best_state = None
    wait = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                val_losses.append(criterion(model(Xb), yb).item())

        tl, vl = float(np.mean(train_losses)), float(np.mean(val_losses))
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        print(f"Epoch {epoch:3d} | Train MSE: {tl:.6f} | Val MSE: {vl:.6f}")

        if vl < best_val:
            best_val = vl
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch} (best val MSE: {best_val:.6f})")
                break

    model.load_state_dict(best_state)
    return model, history, best_val


def predict(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            preds.append(model(Xb).cpu().numpy())
            trues.append(yb.numpy())
    return np.concatenate(preds), np.concatenate(trues)


def compute_metrics(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mask = y_true != 0
    dir_acc = float((np.sign(y_pred[mask]) == np.sign(y_true[mask])).mean() * 100) if mask.sum() > 0 else float("nan")
    print(f"{label:25s} | MAE: {mae:.6f} | RMSE: {rmse:.6f} | Dir. Acc: {dir_acc:.2f}%")
    return mae, rmse, dir_acc


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to financial_data.csv")
    ap.add_argument("--out_dir", default="results", help="Output directory")
    args = ap.parse_args()

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load & split
    df = load_and_prepare(args.data)
    train_df, val_df, test_df = chronological_split(df)
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Scale
    scaler = StandardScaler().fit(train_df[FEATURES].values)
    X_train = scaler.transform(train_df[FEATURES].values).astype(np.float32)
    X_val = scaler.transform(val_df[FEATURES].values).astype(np.float32)
    X_test = scaler.transform(test_df[FEATURES].values).astype(np.float32)

    y_train = train_df["log_return_next"].values.astype(np.float32)
    y_val = val_df["log_return_next"].values.astype(np.float32)
    y_test = test_df["log_return_next"].values.astype(np.float32)

    # Windows
    Xtr, ytr = make_windows(X_train, y_train)
    Xva, yva = make_windows(X_val, y_val)
    Xte, yte = make_windows(X_test, y_test)
    print(f"Windows — train: {Xtr.shape}, val: {Xva.shape}, test: {Xte.shape}")

    # Loaders
    train_loader = DataLoader(TSDataset(Xtr, ytr), batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(TSDataset(Xva, yva), batch_size=BATCH, shuffle=False)
    test_loader = DataLoader(TSDataset(Xte, yte), batch_size=BATCH, shuffle=False)

    # Model
    model = LSTMBaseline(input_dim=len(FEATURES)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,}")

    # Train
    model, history, best_val = train_model(model, train_loader, val_loader, device)

    # Evaluate
    y_pred_lstm, y_true = predict(model, test_loader, device)
    y_pred_naive = np.zeros_like(y_true)

    print("\n" + "=" * 72)
    print("TEST SET RESULTS")
    print("=" * 72)
    m_naive = compute_metrics(y_true, y_pred_naive, "Naive Baseline (r̂=0)")
    m_lstm = compute_metrics(y_true, y_pred_lstm, "LSTM Baseline")
    print("=" * 72)

    # Save
    pd.DataFrame(
        [
            {"Model": "Naive Baseline (r̂=0)", "MAE": m_naive[0], "RMSE": m_naive[1], "Dir. Acc (%)": m_naive[2]},
            {"Model": "LSTM Baseline", "MAE": m_lstm[0], "RMSE": m_lstm[1], "Dir. Acc (%)": m_lstm[2]},
        ]
    ).to_csv(out_dir / "baseline_results.csv", index=False)
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
    print(f"\n✅ Saved results to {out_dir}")


if __name__ == "__main__":
    main()
