# BOL-LPP Reproduction and Extension

## Overview

This project aims to reproduce and extend the research presented in:

**BOL-LPP: A Bayesian-Optimized LSTM Model for Day-Ahead Load Price Forecasting in the ERCOT Market**

The original research uses a Long Short-Term Memory (LSTM) neural network combined with Bayesian Optimization to improve day-ahead electricity price forecasting.

My initial goal is to reproduce the methodology and evaluate whether the reported results can be independently replicated. After reproducing the baseline model, I plan to investigate potential improvements and extensions to the approach.

## Project Goals

1. Obtain and preprocess the relevant ERCOT electricity market and weather data.
2. Build baseline forecasting models for comparison.
3. Implement an LSTM-based electricity price forecasting model.
4. Implement Bayesian Optimization for LSTM hyperparameter tuning.
5. Reproduce and compare results with the original BOL-LPP methodology.
6. Analyze model weaknesses and investigate a potential research extension.

## Technologies

* Python
* PyTorch
* Pandas
* NumPy
* Scikit-learn
* Optuna / Bayesian Optimization
* Matplotlib
* Google Colab

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

🚧 Project setup in progress.

## Future Research Extension

After reproducing the original methodology, I plan to analyze its limitations and investigate potential improvements, such as improved forecasting performance during high-volatility electricity price events or uncertainty-aware forecasting.

## Author

Jahandad Irfan
