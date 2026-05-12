# ADL-financial-forecasting

## Uncertainty-Aware Financial Forecasting

**Multivariate Time-Series Forecasting with Monte Carlo Dropout and Regime Detection**

A Transformer-based forecasting system for the Borsa Istanbul 100 Index (BIST 100) that
integrates Turkish equity data, global market signals, and US macroeconomic indicators.
The model (i) forecasts next-day log returns, (ii) quantifies epistemic uncertainty via
Monte Carlo Dropout, and (iii) adapts to changing market regimes via a hybrid
K-Means + Hidden Markov Model pipeline.

---

## Authors

- **Lara Yıldırım**
- **Belkıs Derin Yüksel**
- **Course:** COM0415 Applied Deep Learning
- **Institution:** Istanbul Kültür University

---

## Project Status

| Component | Status | Notes |
|---|---|---|
| Related work review | ✅ Complete | Six papers, three pillars (Transformer / MCD / Regime Detection) |
| Dataset pipeline | ✅ Complete | 4,231 daily observations, 5 features, chronological splits |
| Exploratory analysis | ✅ Complete | Price history + correlation heatmap; UNRATE excluded (ρ with INDPRO = −0.85) |
| Methodology design | ✅ Complete | Encoder-only Transformer + MC Dropout head, D=5, T=30 |
| Baseline (LSTM) | ✅ Complete | 2-layer LSTM, ~53k params, honest results on 2024–2026 test window |
| Transformer + MCD | ✅ Complete | Encoder-only Transformer, N=50 MC passes, μ + σ² output |
| Regime detection | ✅ Complete | Hybrid K-Means (K=3) + Gaussian HMM on log returns + rolling volatility |
| Final report | ✅ Complete | Full methodology, results, and analysis |

---

## Repository Structure

```
ADL-financial-forecasting/
├── data/
│   ├── 2026-03-MD.csv                      # March 2026 FRED-MD monthly data (manual upload)
│   └── financial_data.csv                  # Final merged dataset (4,231 × 5 features)
├── notebooks/
│   ├── data_collection_cleaning.ipynb      # Builds the 5-feature merged dataset
│   ├── eda_initial.ipynb                   # Exploratory analysis & correlation heatmap
│   ├── baseline_lstm_colab.ipynb           # End-to-end LSTM baseline (Colab-ready)
│   ├── transformer_mcd_colab.ipynb         # Transformer + Monte Carlo Dropout (Colab-ready)
│   ├── regime_detection_colab.ipynb        # K-Means + HMM market regime detection (Colab-ready)
│   └── demo_live_forecast.ipynb            # Live demo: real-time forecast + regime (Colab-ready)
├── papers/                                 # Reference papers supporting the methodology
│   ├── jrfm-19-00203.pdf                   # Liu (2026): Transformer vs classical models
│   ├── A novel transformer-based dual attention architecture for the prediction of financial time series.pdf  # Hadizadeh et al. (2025): Dual Attention
│   ├── gal16.pdf                           # Gal & Ghahramani (2016): MC Dropout = Bayesian inference
│   ├── 2505.15671v1.pdf                    # Asgharnezhad et al. (2025): Enhanced MC Dropout
│   └── jdmdc-57.pdf                        # Haryani et al. (2026): K-Means + HMM regimes
├── results/
│   ├── bist_price_history.png              # Figure 1: BIST 100 historical price (2010–2026)
│   ├── correlation_heatmap.png             # Figure 2: Feature correlation heatmap
│   ├── transformer_architecture.png        # Architecture diagram v1 (price output)
│   ├── transformer_architecture_v2.png     # Architecture diagram v2 (log-return + uncertainty output)
│   ├── lstm_baseline_plot.png              # LSTM training curves + predictions vs actuals
│   ├── transformer_results_plot.png        # Transformer training curves + predictions + uncertainty band
│   ├── regime_detection_plot.png           # K-Means & HMM regimes on BIST 100 price
│   └── Uncertainty_Aware_Financial_Forecasting_FinalReport.pdf  # Final submission document
├── src/
│   ├── baseline_lstm.py                    # Standalone PyTorch LSTM baseline
│   └── transformer_mcd.py                  # Standalone Transformer + MC Dropout
├── COM0415_ADL_Presentation.pptx           # Final presentation slides
├── requirements.txt                        # Pinned Python dependencies (incl. hmmlearn>=0.3)
├── .gitignore
└── README.md
```

---

## Data Sources

| Source | Variables |
|---|---|
| [Borsa Istanbul](https://www.borsaistanbul.com/) via Yahoo Finance | BIST 100 daily close (`XU100.IS`) |
| [Yahoo Finance](https://finance.yahoo.com/) | S&P 500 daily close (`^GSPC`) |
| [FRED-MD](https://research.stlouisfed.org/econ/mccracken/fred-databases/) | CPI (`CPIAUCSL`), Fed Funds Rate (`FEDFUNDS`), Industrial Production (`INDPRO`) |

Daily price data covers **2010-01-04 to 2026-04-23** (N = 4,231 trading days).
Macroeconomic series are published monthly and forward-filled to daily frequency.

**Feature Selection Note:** UNRATE was excluded due to strong negative correlation
with INDPRO (ρ = −0.85), which would introduce multicollinearity without adding information.
See Section 3.3 of the final report.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/derinyuksel/ADL-financial-forecasting.git
cd ADL-financial-forecasting
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Tested with Python 3.11. GPU recommended for Transformer training.

---

## Running the Models

### Option A — Google Colab (recommended)

Upload the relevant notebook to [Google Colab](https://colab.research.google.com/)
and run all cells. Each notebook prompts for `financial_data.csv` upload.

| Notebook | Description |
|---|---|
| `baseline_lstm_colab.ipynb` | 2-layer LSTM baseline, ~2–5 min on CPU |
| `transformer_mcd_colab.ipynb` | Transformer + MC Dropout, GPU recommended |
| `regime_detection_colab.ipynb` | K-Means + HMM regime detection |
| `demo_live_forecast.ipynb` | Live demo: next-day forecast + uncertainty + regime |

### Option B — Local (standalone scripts)

```bash
# LSTM baseline
python src/baseline_lstm.py --data data/financial_data.csv --out_dir results/

# Transformer + MC Dropout
python src/transformer_mcd.py --data data/financial_data.csv --out_dir results/ --n_mc 50
```

---

## Methodology Summary

**Target:** one-day-ahead log return of BIST 100:
`r_{t+1} = ln(P_{t+1} / P_t)`

**Input:** 30-day lookback over 5 z-score-normalised features.

**Split (strictly chronological, no shuffling):**
- Train: 2010-01-04 → 2022-12-31
- Validation: 2023-01-01 → 2023-12-31
- Test: 2024-01-01 → 2026-04-23

### Model Architecture

```
Input (B, T=30, D=5)
        ↓
Linear Projection → d_model=64
        ↓
Positional Encoding (sinusoidal)
        ↓
Transformer Encoder × 2 layers
  ├── Multi-Head Self-Attention (4 heads)
  ├── Add & Norm
  ├── Feed-Forward Network (dim=128, ReLU)
  └── Add & Norm
        ↓
Uncertainty Head (MCDropout, always active)
  ├── Linear(64→64) → ReLU → MCDropout(0.2)
  └── Linear(64→32) → ReLU → MCDropout(0.2)
        ↓
Output: μ (predicted log return)  +  σ² (epistemic uncertainty)
        via N=50 Monte Carlo stochastic passes
```

### Regime Detection

```
Features: log return + 20-day rolling volatility
        ↓
K-Means (K=3) → Bull / Sideways / Bear initial labels
        ↓
Gaussian HMM (N=3 states) → smooth probabilistic regime sequence
        ↓
Outputs: Viterbi state sequence + posterior probabilities + transition matrix
```

**Evaluation metrics:** MAE, RMSE, Directional Accuracy — compared against
naive baseline (r̂ = 0) and LSTM baseline.

---

## References

1. T. Liu, "A Comparative Study of Transformer-Based and Classical Models for Financial Time-Series Forecasting," *JRFM*, vol. 19, no. 203, Mar. 2026. doi: [10.3390/jrfm19030203](https://doi.org/10.3390/jrfm19030203)
2. A. Hadizadeh et al., "A Novel Transformer-Based Dual Attention Architecture for the Prediction of Financial Time Series," *J. King Saud Univ.*, vol. 37, art. 72, Jun. 2025. doi: [10.1007/s44443-025-00045-y](https://doi.org/10.1007/s44443-025-00045-y)
3. Y. Gal and Z. Ghahramani, "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning," in *Proc. ICML*, 2016, pp. 1050–1059.
4. H. Asgharnezhad et al., "Enhancing Monte Carlo Dropout Performance for Uncertainty Quantification," *arXiv* 2505.15671, May 2025.
5. C. A. Haryani et al., "Market Regime Detection in Bitcoin Time Series Using K-Means Clustering and Hidden Markov Models," *JDMDC*, vol. 3, no. 1, pp. 75–95, 2026. doi: [10.47738/jdmdc.v3i1.57](https://doi.org/10.47738/jdmdc.v3i1.57)
6. M. W. McCracken and S. Ng, "FRED-MD: A Monthly Database for Macroeconomic Research," *J. Bus. Econ. Stat.*, vol. 34, no. 4, pp. 574–589, 2016.

---

## License

This repository is coursework submitted for academic evaluation. Code and text are
provided for reviewer access; third-party papers in `/papers/` remain under their
original publisher licenses.

All project files are available via the GitHub repository:
https://github.com/derinyuksel/ADL-financial-forecasting
