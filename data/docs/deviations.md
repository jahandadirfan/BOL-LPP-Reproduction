# Deviations and Resolved Ambiguities

This document records every point where this reproduction departs from the original paper, or where the paper's description was ambiguous and a choice had to be made. Each entry states what the paper says, what was done here, and why.

**Reference:** Y. Mohamed, M. M. Fouda, Z. M. Fadlullah, R. Abdelfattah and M. I. Ibrahem,
"BOL-LPP: A Bayesian-Optimized LSTM Model for Day-Ahead Load Price Forecasting in the ERCOT
Market," *IEEE Open Journal of the Computer Society*, vol. 6, pp. 1001-1011, 2025.

---

## 1. Weather-zone to load-zone mapping

**Paper:** Section III states that hourly zonal load and prices are used for ERCOT's North zone, with South and Coast zones used for testing. It does not specify how ERCOT's load zones (used for pricing) map to ERCOT's weather zones (used for load reporting).

**Issue:** ERCOT maintains two distinct zonal systems. Prices are published for load zones
(`LZ_NORTH`, `LZ_SOUTH`, `LZ_HOUSTON`); load is published for eight weather zones (Coast, East, Far West, North, North Central, South, South Central, West). The names overlap but the
boundaries differ. ERCOT has no load zone named "Coast" — the coastal region is `LZ_HOUSTON`.

**Resolution:** Single best geographic match per zone:

| Paper zone | Price series | Load series |
|---|---|---|
| North | `LZ_NORTH` | North Central weather zone |
| South | `LZ_SOUTH` | South Central weather zone |
| Coast | `LZ_HOUSTON` | Coast weather zone |

**Justification:** An initial implementation paired `LZ_NORTH` with the "North" weather zone by name. Inspection showed North weather zone averaging 838 MW against Coast at 11,498 MW — 
implausible for the zone serving the Dallas-Fort Worth metroplex. A one-week sample across all eight weather zones (July 2016) confirmed North Central at 12,953 MW and North at 823 MW,
establishing that DFW falls in North Central. The corrected mapping yields North 13,116 MW,
South 6,508 MW, Coast 11,498 MW.

---

## 2. Weather data source

**Paper:** Section III cites NOAA's National Centers for Environmental Information for temperature, dew temperature, wind direction, and wind speed. No station is named.

**Deviation:** Open-Meteo historical archive API (ERA5 reanalysis), queried at 32.90 N, 97.04 W (Dallas-Fort Worth), hourly, 2013-01-01 to 2018-12-31, timezone America/Chicago.

**Justification:** NOAA ISD data is distributed in a fixed-width legacy format requiring
position-based parsing per station-year. Open-Meteo provides the same four variables at hourly resolution in a single request. Since the paper does not name its station, exact replication of its weather series is not possible under either source.

**Expected impact:** Low. ERA5 reanalysis and station observations agree closely for temperature and dewpoint. The largest divergence is in wind speed, which is sensitive to local terrain and instrument height — and the paper's own permutation feature importance (Figure 5) ranks wind variables near the bottom of the feature set, with load, previous price, and temperature
dominating.

**Possible robustness check:** Compare one year of NOAA station temperature against the
Open-Meteo series to quantify divergence directly.

---

## 3. Scale of reported error metrics

**Paper:** Table 2 reports MAE of 0.0044 and MSE of 0.00017 for BOL-LPP, and MAE 0.0065 for
BiLSTM, alongside SARIMAX MAE 8.003 and XGBoost MAE 6.88, all labelled as $/MWh. The abstract
states MAE of $0.0044/MWh.

**Issue:** Empirical inspection of ERCOT North zone DAM prices for 2013-2018 gives mean $29.41, standard deviation $39.36, minimum $1.48, maximum $2,241.43. An MAE of $0.0044 corresponds to
0.015% of the mean, which is not plausible for a volatile price series.

Two further observations indicate the deep-learning metrics are not on the original price scale:

1. Within Table 2, classical models report values consistent with dollar units (8.003, 6.88)
   while deep-learning models report values three orders of magnitude smaller (0.0065, 0.0044) on the same dataset. A unit change between model families is more parsimonious than a genuine accuracy difference of that magnitude.
2. BOL-LPP reports MAE 0.0044 alongside MAPE 48.7%. These cannot both describe the same
   predictions in the same units. Both become internally consistent if MAE is computed on
   normalised values and MAPE against actual prices.

**Resolution:** This reproduction reports all error metrics twice — on the normalised scale(comparable to the paper's deep-learning figures) and inverse-transformed to $/MWh(comparable to the classical baselines, and economically interpretable). The paper states that "feature normalization is employed" without specifying the method, so the exact transform cannot be recovered from the text.

---

## 4. Train / validation / test split strategy

**Paper:** Section IV-A states the dataset is split "in a stratified manner, with 70% of the dataset allocated for training, 15% for validation, and 15% for testing." Algorithm 1, line 2, states only "Split D into training, validation, and test."

**Issue:** Stratified sampling conventionally means random sampling that preserves class proportions. Applied to a time series, this interleaves future and past observations across splits, allowing the model to train on hours adjacent to test hours. Given the strong autocorrelation of hourly electricity prices, this would substantially inflate reported accuracy.

**Resolution:** Chronological split — the first 70% of the timeline for training, the next 15% for validation, the final 15% for testing. The 70/15/15 ratios follow the paper.

**Note:** If the original authors did use a randomised or per-period stratified split, this alone could account for a large share of the difference between their reported metrics and any obtained here.

---

## 5. Feature timing and look-ahead leakage

**Paper:** Figure 4 lists model inputs as Load, Previous Load prices, Temperature, Dew temperature, Wind direction, Wind speed, Hour, Day, Month, Year. It does not state whether "Load" refers to the target hour's load or to load from prior hours.

**Issue:** In genuine day-ahead forecasting, the forecast is produced before the delivery day. Actual load for the target hour is not known at bid submission time. Using it as an input would constitute look-ahead leakage.

**Resolution:** All input features are drawn from hours strictly prior to the target hour. The target hour's own load and weather are excluded from its feature window.

---

## 6. LSTM cell activations

**Paper:** Equations (3) and (6) specify ReLU for the candidate cell state and for the hidden
state output, in place of the tanh used in the standard LSTM formulation. Gates (equations 1, 2,
5) use sigmoid as usual.

**Implication:** PyTorch's built-in `nn.LSTM` hardcodes tanh at both positions and cannot be
configured otherwise. Faithful reproduction requires a hand-implemented LSTM cell following the
paper's equations.

**Resolution:** Custom LSTM cell implemented in PyTorch per equations (1)-(6).

---

## 7. Number of LSTM layers

**Paper:** Table 1 lists "LSTM middle hidden layer" with search space 1-4 and optimal value 1.
However, the same table gives three values for hidden units per layer ([224, 256, 64]), three
activation functions ([ReLU, ReLU, swish]), and three dropout rates ([0.4, 0.6, 0.7]). Figure 4
depicts three stacked LSTM layers.

**Resolution:** Three stacked LSTM layers implemented, consistent with Figure 4 and with the
triplets of per-layer values in Table 1. The reported optimal value of 1 is treated as a
labelling inconsistency.

---

## 8. Hyperparameter search space values not fully specified

**Paper:** Table 1 gives search ranges and selected values for the LSTM hyperparameters. Where a
range is stated (e.g. learning rate 0.001-0.1), it is used directly.

**Note:** The Bayesian Optimization implementation in the paper uses a Gaussian Process surrogate
with UCB acquisition (beta = 2.6, kernel noise 1e-4), 10 trials maximum, 10 epochs per trial,
batch size 32, early stopping after 5 epochs without improvement. Where this reproduction uses
Optuna, the default sampler differs (TPE rather than GP-UCB); any substitution is noted in the
relevant notebook.

---

## 9 Rows interpolation to handle time duplicates

We get 52,578 rows during preprocessing spanning 2013-01-01 to 2018-12-31, with no missing values after interpolation.

This is six rows fewer than the 52,584 hourly observations reported in the paper (Section III).
The difference is the six daylight-saving fall-back hours (one per year, 2013-2018), where the 1:00-2:00 am hour occurs twice. Deduplicating by naive timestamp retains the first occurrence and discards the second. The paper does not state how it handled these hours; retaining both would require timezone-aware timestamps throughout, which is incompatible with merging against the timezone-naive weather series.

---

## 10. The `year` feature under a chronological split

**Paper:** Figure 4 lists Year among the ten model inputs.

**Issue:** With min-max scaling fitted on the training split (2013-2017), the scaler maps 2013 to 0.0 and 2017 to 1.0. The test set is entirely 2018, which scales to 1.25 — a constant value outside the range the model observed during training. The feature therefore carries no discriminative signal at test time, and introduces a small risk of the model extrapolating a learned year trend beyond its training range.

This is structural to using raw year as a feature with any chronological split, since each split necessarily covers different years.

**Resolution:** Retained, to follow the paper's stated input set. The paper's own permutation feature importance analysis (Figure 5) ranks Year lowest of all ten inputs, which is consistent with the model assigning it little weight.

**Planned check:** An ablation comparing model performance with and without the `year` feature is listed as a Phase 8 experiment.

## Pending

- Walk-forward validation as an additional evaluation protocol (considered, deferred to Phase 8)
- NOAA temperature comparison as a robustness check on the weather substitution
- Ablation on the `year` feature (see entry 10)
- Cyclical (sine/cosine) encoding of hour-of-day as an alternative to raw values