# Uncertainty-Aware Financial Forecasting

**Multivariate Time-Series Forecasting with Monte Carlo Dropout and Planned Regime Detection**

A transformer-based forecasting system for the Borsa Istanbul 100 Index (BIST 100) that
integrates Turkish equity data, global market signals, and US macroeconomic indicators.
The model is designed to (i) forecast next-day log returns, (ii) quantify epistemic
uncertainty via Monte Carlo Dropout, and (iii) adapt to changing market regimes via a
hybrid K-Means + Hidden Markov Model pipeline (planned for final submission).

---

## Authors

- **Lara Yıldırım** (2100003941)
- **Belkıs Derin Yüksel** (2200001724)

**Course:** COM0415 Applied Deep Learning
**Instructor:** Doç. Dr. Fatma Patlar Akbulut
**Institution:** Istanbul Kültür University

---

## Project Status

This repository contains the **interim submission**. Progress so far:

| Component            | Status      | Notes                                                                 |
|----------------------|-------------|-----------------------------------------------------------------------|
| Related work review  | Complete    | Six papers, three pillars (Transformer / MCD / Regime Detection)      |
| Dataset pipeline     | Complete    | 4,223 daily observations, 5 features, chronological splits            |
| Exploratory analysis | Complete    | Price history + correlation heatmap; UNRATE excluded (ρ with INDPRO = −0.85) |
| Methodology design   | Complete    | Encoder-only Transformer + MC Dropout head, D = 5, T = 30             |
| Baseline (LSTM)      | Complete    | 2-layer LSTM, ~53k params, honest results on 2024–2026 test window    |
| Transformer model    | Planned     | Final submission                                                      |
| MCD inference loop   | Planned     | Final submission                                                      |
| Regime detection     | Planned     | Hybrid K-Means + HMM on log returns + rolling volatility              |
| Risk-aware strategy  | Planned     | Uncertainty-gated trading evaluation                                  |

---

## Repository Structure

```
ADL-financial-forecasting/
├── data/                             # Cleaned datasets (not tracked if large)
├── notebooks/
│   ├── data_collection_cleaning.ipynb  # Builds the 5-feature merged dataset
│   ├── eda_initial.ipynb               # Exploratory analysis & correlation heatmap
│   └── baseline_lstm_colab.ipynb       # End-to-end LSTM baseline (Colab-ready)
├── papers/                           # Reference papers supporting the methodology
├── results/
│   ├── bist_price_history.png          # Figure 1 of the interim report
│   ├── correlation_heatmap.png         # Figure 2 of the interim report
│   ├── transformer_architecture.png    # Original architecture diagram (v1)
│   ├── transformer_architecture_v2.png # Updated architecture diagram (log-return output)
│   ├── lstm_baseline_plot.png          # Training curves + predictions vs. actuals
│   └── interim_report.docx             # Interim submission document
├── src/
│   └── baseline_lstm.py              # Standalone PyTorch LSTM baseline
├── requirements.txt                  # Pinned Python dependencies
├── .gitignore
└── README.md                         # This file
```

---

## Data Sources

This project integrates three public data sources:

| Source                                                               | Variables                                                          |
|----------------------------------------------------------------------|--------------------------------------------------------------------|
| [Borsa Istanbul](https://www.borsaistanbul.com/) via Yahoo Finance   | BIST 100 daily close (ticker `XU100.IS`)                           |
| [Yahoo Finance](https://finance.yahoo.com/)                          | S&P 500 daily close (ticker `^GSPC`)                                |
| [FRED-MD](https://research.stlouisfed.org/econ/mccracken/fred-databases/) | CPI (`CPIAUCSL`), Fed Funds Rate (`FEDFUNDS`), Industrial Production (`INDPRO`) |

Daily price data covers **2010-01-04 to 2026-04-13** (N = 4,223 trading days).
Macroeconomic series are published monthly and forward-filled to daily frequency; within
each month their values are held constant.

### Feature Selection Note

An initial candidate set included the US Unemployment Rate (`UNRATE`). It was examined
during EDA and **excluded** from the final feature set due to its strong negative correlation
with Industrial Production (ρ = −0.85) — retaining both would have introduced statistical
redundancy without adding information. See Section 3.3 of `results/interim_report.docx`.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/derinyuksel/ADL-financial-forecasting.git
cd ADL-financial-forecasting
```

### 2. Create a virtual environment (recommended)

```bash
# Using venv
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Tested with Python 3.11. A GPU is not required for the LSTM baseline but will be useful
for the full Transformer in the final submission.

---

## Running the Baseline

### Option A — Google Colab (recommended for first run)

1. Upload `notebooks/baseline_lstm_colab.ipynb` to [Google Colab](https://colab.research.google.com/).
2. When prompted, upload the cleaned dataset `financial_data.csv` produced by
   `notebooks/data_collection_cleaning.ipynb`.
3. Run all cells. Training takes ~2–5 minutes on a free CPU runtime.

### Option B — Local

```bash
python src/baseline_lstm.py --data path/to/financial_data.csv --out_dir results/
```

Outputs written to `results/`:
- `baseline_results.csv` — MAE, RMSE, and directional accuracy for the LSTM and a
  naive (r̂ = 0) baseline, evaluated on the 2024–2026 test window.
- `training_history.csv` — per-epoch train/validation MSE.

---

## Methodology (Summary)

**Target:** one-day-ahead log return of BIST 100:
`r_{t+1} = ln(P_{t+1} / P_t)`

**Input:** 30-day lookback over 5 features — BIST 100 Close, S&P 500 Close, CPI,
Federal Funds Rate, and Industrial Production. All features are z-score normalised
using statistics fit only on the training partition.

**Split (strictly chronological, no shuffling):**
- Train: 2010-01-04 → 2022-12-31
- Validation: 2023-01-01 → 2023-12-31
- Test: 2024-01-01 → 2026-04-13

**Evaluation:** Mean Absolute Error, Root Mean Squared Error, and Directional Accuracy,
with a naive `r̂ = 0` baseline for context.

Full methodology in `results/interim_report.docx`.

---

## References

1. T. Liu, "A Comparative Study of Transformer-Based and Classical Models for Financial Time-Series Forecasting," *Journal of Risk and Financial Management*, vol. 19, no. 203, Mar. 2026. doi: [10.3390/jrfm19030203](https://doi.org/10.3390/jrfm19030203)

2. A. Hadizadeh, M. J. Tarokh, and M. M. Ghazani, "A Novel Transformer-Based Dual Attention Architecture for the Prediction of Financial Time Series," *Journal of King Saud University — Computer and Information Sciences*, vol. 37, article no. 72, Jun. 2025. doi: [10.1007/s44443-025-00045-y](https://doi.org/10.1007/s44443-025-00045-y)

3. Y. Gal and Z. Ghahramani, "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning," in *Proc. 33rd International Conference on Machine Learning (ICML)*, New York, NY, USA, 2016, pp. 1050–1059.

4. H. Asgharnezhad, A. Shamsi, R. Alizadehsani, A. Mohammadi, and H. Alinejad-Rokny, "Enhancing Monte Carlo Dropout Performance for Uncertainty Quantification," *arXiv preprint* arXiv:2505.15671, May 2025.

5. C. A. Haryani, Chandra, and R. E. Tarigan, "Market Regime Detection in Bitcoin Time Series Using K-Means Clustering and Hidden Markov Models," *Journal of Digital Market and Digital Currency*, vol. 3, no. 1, pp. 75–95, 2026. doi: [10.47738/jdmdc.v3i1.57](https://doi.org/10.47738/jdmdc.v3i1.57)

6. M. W. McCracken and S. Ng, "FRED-MD: A Monthly Database for Macroeconomic Research," *Journal of Business & Economic Statistics*, vol. 34, no. 4, pp. 574–589, 2016. doi: [10.1080/07350015.2015.1086655](https://doi.org/10.1080/07350015.2015.1086655)

---

## License

This repository is coursework submitted for academic evaluation. Code and text are
provided for reviewer access; third-party papers in `/papers/` remain under their
original publisher licenses.
