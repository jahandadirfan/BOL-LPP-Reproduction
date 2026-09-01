# BOL-LPP Reproduction and Extension

## Overview

This project reproduces and extends the research presented in:

**BOL-LPP: A Bayesian-Optimized LSTM Model for Day-Ahead Load Price Forecasting in the ERCOT Market**
Y. Mohamed, M. M. Fouda, Z. M. Fadlullah, R. Abdelfattah and M. I. Ibrahem, *IEEE Open Journal of the Computer Society*, vol. 6, pp. 1001-1011, 2025. [doi: 10.1109/OJCS.2025.3580107](https://doi.org/10.1109/OJCS.2025.3580107)

The original research uses a Long Short-Term Memory (LSTM) neural network, with hyperparameters tuned via Bayesian Optimization (BO), to forecast day-ahead electricity prices for ERCOT's zonal markets, using historical load, price, and weather data.

My goal is to (1) faithfully reproduce the original methodology and evaluate whether the reported results can be independently replicated, and (2) extend the model with the exogenous renewable-generation variables the original authors identify as future work.

## Project Goals

1. Obtain and preprocess ERCOT electricity load/price data (North, South, and Coast zones) and matching NOAA weather data.
2. Build baseline forecasting models for comparison: ARMA, ARIMA, SARIMAX, XGBoost, and BiLSTM.
3. Implement the BOL-LPP LSTM architecture (3 stacked LSTM layers with ReLU-gated cell/hidden states, dropout, dense output layer).
4. Implement Bayesian Optimization (Gaussian Process surrogate, UCB acquisition) for LSTM hyperparameter tuning.
5. Reproduce and compare results with the original BOL-LPP methodology, including the look-back window sweep (1–24 hours) and cross-zone generalization test (train North, test South/Coast).
6. Investigate and correct a measurement inconsistency identified in the original paper's results (see below).
7. Extend the model with exogenous renewable-generation variables (wind and solar output, and derived net load) — the primary future-work direction named by the original authors.

## Reproduction Notes & Known Issues

While extracting exact figures from the paper, a unit/scaling inconsistency was identified in the original results table: classical models (ARMA, ARIMA, SARIMAX, XGBoost) report MAE in real dollar units (e.g., SARIMAX MAE = $8.003/MWh), while the deep learning models (BiLSTM, BOL-LPP) report MAE three orders of magnitude smaller (e.g., BOL-LPP MAE = $0.0044/MWh) — consistent with those being computed on normalized (0–1 scaled) data rather than inverse-transformed dollar values. This project reports **both** scaled and inverse-transformed (real-dollar) metrics for every model to enable a fair, consistent comparison, and treats resolving this discrepancy as part of the reproduction's contribution.

## Technologies

* Python
* PyTorch
* Pandas
* NumPy
* Scikit-learn
* Optuna (Bayesian Optimization)
* Matplotlib
* Google Colab

## Data Sources

* **ERCOT DAM Settlement Point Prices & hourly zonal load** (North, South, Coast zones), Jan 1, 2013 – Dec 31, 2018 — [ERCOT MIS](https://www.ercot.com/) public reports
* **Weather** (temperature, dew temperature, wind direction, wind speed) — [NOAA NCEI](https://www.ncei.noaa.gov/access/)
* **Wind & solar generation** (extension only) — ERCOT system-wide actual generation reports

## Project Structure

```text
BOL-LPP-Reproduction/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
├── models/
├── results/
│   ├── figures/
│   └── tables/
├── .gitignore
└── README.md
```

## Current Status

🚧 Project setup complete; data acquisition in progress.

## Research Extension: Exogenous Renewable Variables

The original paper's Conclusion identifies integrating exogenous variables — specifically renewable energy generation — as its main direction for future work. This project implements that extension directly:

* Add ERCOT-wide wind and solar generation as input features.
* Engineer a **net load** feature (`load − wind − solar`), motivated by the fact that ERCOT prices are driven primarily by net load rather than raw demand: when renewable output drops while demand holds steady, expensive marginal generation sets a higher price, which is the mechanism behind many ERCOT price spikes.
* Retrain the same BOL-LPP architecture with and without these features (ablation-style comparison) and evaluate whether they measurably improve forecasting accuracy — particularly during high-volatility/price-spike hours, where the original model's error is likely concentrated.
* Use the original paper's permutation feature importance (PFI) method to rank the new features against the original ones (load, previous price, temperature, etc.).

## Author

Jahandad Irfan
