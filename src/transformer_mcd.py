"""
transformer_mcd.py
==================
Uncertainty-Aware Transformer for BIST 100 log return forecasting.

Project: Uncertainty-Aware Financial Forecasting (COM0415 Applied Deep Learning)
Authors: Lara Yıldırım, Belkıs Derin Yüksel

Architecture
------------
- Encoder-only Transformer with sinusoidal positional encoding
- Monte Carlo Dropout in the uncertainty head (active at inference)
- Output: predictive mean μ (log return) + epistemic uncertainty σ²

Usage
-----
    python transformer_mcd.py --data path/to/financial_data.csv
    python transformer_mcd.py --data path/to/financial_data.csv --out_dir results/ --n_mc 50

References
----------
- Gal & Ghahramani (2016): Dropout as a Bayesian Approximation
- Asgharnezhad et al. (2025): Enhancing MC Dropout for Uncertainty Quantification
- Liu (JRFM 2026): Transformer-Based Financial Time-Series Forecasting
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
FEATURES   = ["BIST100_Close", "SP500_Close", "CPIAUCSL", "FEDFUNDS", "INDPRO"]
LOOKBACK   = 30
TRAIN_END  = "2022-12-31"
VAL_END    = "2023-12-31"
BATCH      = 64
EPOCHS     = 80
LR         = 1e-3
PATIENCE   = 10
SEED       = 42

# Transformer hyperparameters
D_MODEL    = 64
N_HEADS    = 4
N_LAYERS   = 2
D_FF       = 128
DROPOUT    = 0.1
MC_DROPOUT = 0.2
N_MC       = 50   # Monte Carlo stochastic passes at inference


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
    df["log_return_next"] = np.log(df["BIST100_Close"].shift(-1) / df["BIST100_Close"])
    return df.dropna()


def chronological_split(df: pd.DataFrame):
    train = df.loc[:TRAIN_END]
    val   = df.loc[(df.index > TRAIN_END) & (df.index <= VAL_END)]
    test  = df.loc[df.index > VAL_END]
    return train, val, test


def make_windows(X: np.ndarray, y: np.ndarray, T: int = LOOKBACK):
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
class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class MCDropout(nn.Dropout):
    """Dropout that remains active during model.eval() for MC inference."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.dropout(x, self.p, training=True)


class UncertaintyAwareTransformer(nn.Module):
    """
    Encoder-only Transformer with Monte Carlo Dropout uncertainty head.

    Args:
        input_dim  : number of input features (D=5)
        d_model    : Transformer internal dimension
        n_heads    : number of attention heads (must divide d_model)
        n_layers   : number of Transformer encoder layers
        d_ff       : feed-forward hidden dimension
        dropout    : standard dropout (encoder + positional encoding)
        mc_dropout : MCDropout probability in uncertainty head
    """

    def __init__(
        self,
        input_dim: int = 5,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        n_layers: int = N_LAYERS,
        d_ff: int = D_FF,
        dropout: float = DROPOUT,
        mc_dropout: float = MC_DROPOUT,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc    = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Uncertainty head — MCDropout stays ON at inference
        self.uncertainty_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            MCDropout(mc_dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            MCDropout(mc_dropout),
        )
        self.mu_head = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x = self.input_proj(x)        # (B, T, d_model)
        x = self.pos_enc(x)            # (B, T, d_model)
        x = self.encoder(x)            # (B, T, d_model)
        x = x[:, -1, :]               # last timestep (B, d_model)
        x = self.uncertainty_head(x)   # (B, 32)
        return self.mu_head(x).squeeze(-1)  # (B,)


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------
def train_model(model, train_loader, val_loader, device):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    best_val   = float("inf")
    best_state = None
    wait       = 0
    history    = {"train_loss": [], "val_loss": []}

    for epoch in range(1, EPOCHS + 1):
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

        tl = float(np.mean(train_losses))
        vl = float(np.mean(val_losses))
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        scheduler.step(vl)

        print(f"Epoch {epoch:3d} | Train MSE: {tl:.6f} | Val MSE: {vl:.6f}")

        if vl < best_val:
            best_val   = vl
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} (best val MSE: {best_val:.6f})")
                break

    model.load_state_dict(best_state)
    return model, history, best_val


# ------------------------------------------------------------------
# Monte Carlo Inference
# ------------------------------------------------------------------
def mc_predict(model, loader, device, n_mc: int = N_MC):
    """
    Run n_mc stochastic forward passes.
    MCDropout remains active during eval().

    Returns:
        mu     : mean prediction  (n_samples,)
        sigma2 : variance         (n_samples,)  — epistemic uncertainty
        y_true : ground truth     (n_samples,)
    """
    model.eval()
    all_preds = []
    all_true  = []

    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            batch_preds = [model(Xb).cpu().numpy() for _ in range(n_mc)]
            all_preds.append(np.stack(batch_preds, axis=0))  # (n_mc, batch)
            all_true.append(yb.numpy())

    all_preds = np.concatenate(all_preds, axis=1)  # (n_mc, n_samples)
    y_true    = np.concatenate(all_true)
    return all_preds.mean(axis=0), all_preds.var(axis=0), y_true


# ------------------------------------------------------------------
# Metrics & Reporting
# ------------------------------------------------------------------
def compute_metrics(y_true, y_pred, label: str) -> dict:
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mask = y_true != 0
    dir_acc = float((np.sign(y_pred[mask]) == np.sign(y_true[mask])).mean() * 100) if mask.sum() > 0 else float("nan")
    print(f"{label:35s} | MAE: {mae:.6f} | RMSE: {rmse:.6f} | Dir. Acc: {dir_acc:.2f}%")
    return {"Model": label, "MAE": mae, "RMSE": rmse, "Dir. Acc (%)": dir_acc}


def save_plots(history, mu_test, sigma2_test, y_true_test, out_dir: Path):
    fig, axes = plt.subplots(3, 1, figsize=(13, 12))
    n_show = min(200, len(y_true_test))
    x_idx  = np.arange(n_show)
    mu_s   = mu_test[:n_show]
    sig_s  = np.sqrt(sigma2_test[:n_show])

    # Training curves
    axes[0].plot(history["train_loss"], label="Train MSE")
    axes[0].plot(history["val_loss"],   label="Val MSE")
    axes[0].set(xlabel="Epoch", ylabel="MSE", title="Transformer — Training Curves")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # Predictions + uncertainty band
    axes[1].plot(x_idx, y_true_test[:n_show], label="Actual",        alpha=0.7, linewidth=1)
    axes[1].plot(x_idx, mu_s,                 label="Transformer μ", alpha=0.9, linewidth=1, color="orange")
    axes[1].fill_between(x_idx, mu_s - 2 * sig_s, mu_s + 2 * sig_s,
                         alpha=0.2, color="orange", label="±2σ (epistemic)")
    axes[1].set(xlabel="Test day", ylabel="Log return",
                title=f"Predictions vs Actual + Uncertainty Band (first {n_show} days)")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    # Epistemic uncertainty over time
    axes[2].fill_between(x_idx, 0, sig_s, alpha=0.5, color="red", label="σ (epistemic)")
    axes[2].set(xlabel="Test day", ylabel="Uncertainty σ", title="Epistemic Uncertainty over Test Period")
    axes[2].legend(); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / "transformer_results_plot.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅  Saved {out_path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Uncertainty-Aware Transformer for BIST 100 forecasting")
    ap.add_argument("--data",    required=True,    help="Path to financial_data.csv")
    ap.add_argument("--out_dir", default="results", help="Output directory")
    ap.add_argument("--n_mc",    type=int, default=N_MC, help="MC inference passes (default 50)")
    args = ap.parse_args()

    set_seed()
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device : {device}")
    print(f"Out dir: {out_dir}")

    # ---- Data ----
    df = load_and_prepare(args.data)
    train_df, val_df, test_df = chronological_split(df)
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    scaler  = StandardScaler().fit(train_df[FEATURES].values)
    X_train = scaler.transform(train_df[FEATURES].values).astype(np.float32)
    X_val   = scaler.transform(val_df[FEATURES].values).astype(np.float32)
    X_test  = scaler.transform(test_df[FEATURES].values).astype(np.float32)

    y_train = train_df["log_return_next"].values.astype(np.float32)
    y_val   = val_df["log_return_next"].values.astype(np.float32)
    y_test  = test_df["log_return_next"].values.astype(np.float32)

    Xtr, ytr = make_windows(X_train, y_train)
    Xva, yva = make_windows(X_val,   y_val)
    Xte, yte = make_windows(X_test,  y_test)
    print(f"Windows — train: {Xtr.shape} | val: {Xva.shape} | test: {Xte.shape}")

    train_loader = DataLoader(TSDataset(Xtr, ytr), batch_size=BATCH, shuffle=True)
    val_loader   = DataLoader(TSDataset(Xva, yva), batch_size=BATCH, shuffle=False)
    test_loader  = DataLoader(TSDataset(Xte, yte), batch_size=BATCH, shuffle=False)

    # ---- Model ----
    model = UncertaintyAwareTransformer(input_dim=len(FEATURES)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # ---- Train ----
    model, history, best_val = train_model(model, train_loader, val_loader, device)

    # ---- MC Inference ----
    print(f"\nRunning {args.n_mc} Monte Carlo passes...")
    mu_test, sigma2_test, y_true_test = mc_predict(model, test_loader, device, n_mc=args.n_mc)
    print(f"Mean σ²: {sigma2_test.mean():.6f}")

    # ---- Evaluate ----
    y_pred_naive = np.zeros_like(y_true_test)
    print("\n" + "=" * 78)
    print("TEST SET RESULTS")
    print("=" * 78)
    r_naive       = compute_metrics(y_true_test, y_pred_naive, "Naive Baseline (r̂=0)")
    r_transformer = compute_metrics(y_true_test, mu_test,      "Transformer + MCD (μ)")
    print("=" * 78)

    for metric in ["MAE", "RMSE"]:
        impr = (r_naive[metric] - r_transformer[metric]) / r_naive[metric] * 100
        print(f"  Improvement over naive — {metric}: {impr:+.2f}%")

    # ---- Save ----
    pd.DataFrame([r_naive, r_transformer]).to_csv(out_dir / "transformer_results.csv", index=False)
    pd.DataFrame(history).to_csv(out_dir / "transformer_training_history.csv", index=False)
    pd.DataFrame({
        "y_true": y_true_test,
        "mu_pred": mu_test,
        "sigma2": sigma2_test,
        "sigma": np.sqrt(sigma2_test),
    }).to_csv(out_dir / "transformer_uncertainty.csv", index=False)

    save_plots(history, mu_test, sigma2_test, y_true_test, out_dir)
    print(f"\n✅  All outputs saved to {out_dir}/")


if __name__ == "__main__":
    main()
