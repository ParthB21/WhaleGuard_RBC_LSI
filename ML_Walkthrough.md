# Machine Learning — Technical Walkthrough

**WhaleGuard NARW Species Distribution Model**  
*RBC Borealis — Large-Scale Intelligence*

> This document provides a detailed technical walkthrough of the model training, evaluation, and comparison conducted in [`train_logistic_regression.py`](train_logistic_regression.py), [`train_xgboost.py`](train_xgboost.py), [`train_random_forest.py`](train_random_forest.py), and [`manual_test.py`](manual_test.py). For the exploratory data analysis, see the [EDA Walkthrough](EDA_Walkthrough.md). For a high-level project overview, see the [README](README.md).

---

## Table of Contents

1. [Modelling Strategy](#1-modelling-strategy)
2. [Data Preparation & Temporal Split](#2-data-preparation--temporal-split)
3. [Model A: Logistic Regression (Baseline)](#3-model-a-logistic-regression-baseline)
4. [Model B: XGBoost (Primary)](#4-model-b-xgboost-primary)
5. [Model C: Random Forest](#5-model-c-random-forest)
6. [Threshold Optimisation for Conservation](#6-threshold-optimisation-for-conservation)
7. [Head-to-Head Comparison](#7-head-to-head-comparison)
8. [Feature Importance & Interpretability](#8-feature-importance--interpretability)
9. [Manual Inference Testing](#9-manual-inference-testing)
10. [Model Artefacts & Deployment](#10-model-artefacts--deployment)

---

## 1. Modelling Strategy

The WhaleGuard modelling strategy follows established ML research practice: **train a simple baseline model first, then compare it against more complex models to empirically justify the additional complexity.**

| Aspect | Logistic Regression (Baseline) | Random Forest | XGBoost (Primary) |
|---|---|---|---|
| **Role** | Establishes interpretable lower bound | Bagged ensemble comparator | Production classifier |
| **Why chosen** | Provides coefficients and odds ratios for ecological interpretability | Quantifies the benefit of boosting over bagging; provides OOB error estimate | Handles non-linear interactions, missing values, and feature correlations natively |
| **Missing values** | Median imputation required | Median imputation required | Sparsity-Aware Split Finding (Ji et al., 2024) — no imputation |
| **Class imbalance** | `class_weight="balanced"` (auto-adjusts) | `class_weight="balanced_subsample"` (per-tree rebalancing) | `scale_pos_weight=4.0` (explicit ratio) |
| **Regularisation** | L2 Ridge (C=1.0) | `max_depth=20`, `min_samples_leaf=5` + bagging variance reduction | L1 + L2 (`reg_alpha=0.1`, `reg_lambda=1.0`) + `gamma=0.1` |

### Feature Exclusions

Both models train on the same **10 features** and explicitly **exclude** `Lat`, `Lon`, and `Date` from the feature set. This is a deliberate design decision:

- **Lat/Lon exclusion** forces the model to learn *ocean physics* (temperature, productivity, bathymetry) rather than memorise geographic coordinates. A model that memorises "this lat/lon had a whale" cannot generalise to new areas or account for range shifts under climate change.
- **Date exclusion** prevents temporal leakage. The `Month` feature is retained as a cyclical seasonal proxy, but the exact date is dropped to avoid overfitting to specific survey events.

---

## 2. Data Preparation & Temporal Split

### Split Strategy: Chronological (80/20)

Unlike random train/test splits, a **temporal split** places all data before a cutoff date into training and all data after it into testing. This proves the model can generalise to *future* conditions it has never seen — a much more demanding evaluation than random cross-validation.

| Set | Rows | Date Range | Presence | Absence |
|---|---|---|---|---|
| **Train** | 51,920 | 2002-01-07 → 2015-07-09 | ~10,200 | ~41,720 |
| **Test** | 12,981 | 2015-07-10 → 2018-01-31 | ~2,780 | ~10,200 |

**Split date:** 2015-07-10 (80th percentile chronologically)

The temporal split is critical for this application. Ship strike management requires predictions about *where whales will be tomorrow*, not where they were in historical data. A model that interpolates known dates is operationally useless; a model that extrapolates to unseen future conditions is deployable.

### Preprocessing Differences

| Step | Logistic Regression | XGBoost |
|---|---|---|
| Missing value handling | `SimpleImputer(strategy="median")` | Native (no imputation) |
| Feature scaling | `StandardScaler(mean=0, std=1)` | Not required (tree-based) |
| Boolean encoding | `Is_Thermal_Front` cast to int | `Is_Thermal_Front` cast to int |

The logistic regression pipeline chains these steps using `sklearn.Pipeline` to prevent data leakage — the scaler is fit only on training data and applied to test data.

---

## 3. Model A: Logistic Regression (Baseline)

### Hyperparameter Tuning

The regularisation strength `C` is tuned via `RandomizedSearchCV` (40 iterations) with `TimeSeriesSplit(n_splits=4)`, sampling from a log-uniform distribution to ensure equal coverage across orders of magnitude.

| Parameter | Value | Tuned? |
|---|---|---|
| `C` | **0.93** (selected by RandomizedSearchCV) | ✓ `loguniform(0.001, 100)` — continuous sampling |
| `penalty` | L2 (Ridge) | Fixed — keeps all features for interpretability |
| `solver` | lbfgs | Fixed — efficient for L2 penalty |
| `max_iter` | 1000 | Fixed — ensures convergence |
| `class_weight` | balanced | Fixed — auto-compensates class imbalance |

**Best CV ROC-AUC: 0.8046.** The tuned C≈0.93 is very close to the default (1.0), confirming the original choice was near-optimal.

### Performance

| Metric | Value |
|---|---|
| **ROC-AUC** | 0.8049 |
| **Recall** | 0.8316 |
| **Precision** | 0.3389 |
| **F1-Score** | 0.4816 |
| **Accuracy** | 0.6396 |

The logistic regression achieves a respectable AUC of 0.804, confirming that the feature set carries meaningful signal. However, the low precision (33.9%) at high recall (83.4%) indicates that the linear decision boundary produces many false positives — locations where the model predicts whale presence but no whale is found.

![ROC curve for the logistic regression baseline — AUC = 0.8044, well above the random classifier diagonal](images/lr_roc_curve.png)

### Coefficient Analysis

Logistic regression coefficients (standardised) reveal the **direction and magnitude** of each feature's influence on whale presence probability:

![Logistic regression coefficients — Dist_to_Shore_km has the strongest negative coefficient (-2.044), Bathymetry the strongest positive (+1.552)](images/lr_coefficients.png)

**Top coefficients by magnitude:**

| Rank | Feature | Coefficient | Odds Ratio | Interpretation |
|---|---|---|---|---|
| 1 | **Dist_to_Shore_km** | -2.044 | 0.130 | Each 1σ increase in distance to shore reduces whale odds by **87%** |
| 2 | **Bathymetry** | +1.552 | 4.721 | Each 1σ increase (shallower) increases odds by **372%** |
| 3 | **Dist_to_Shelf_km** | -0.496 | 0.609 | Farther from shelf break → 39% odds reduction |
| 4 | **Chlorophyll** | +0.144 | 1.155 | Higher productivity → 15.5% odds increase |
| 5 | **Salinity** | -0.131 | 0.877 | Saltier open-ocean water → 12.3% odds reduction |

**Ecological coherence check:** The coefficient signs match known NARW ecology — closer to shore (+), shallower water (+), more productive (+), fresher (+). This validates that the model is learning biologically meaningful relationships rather than artefacts.

**Month (coefficient = +0.003)** is nearly zero, confirming the Month Paradox discussed in the [EDA Walkthrough](EDA_Walkthrough.md#10-mann-whitney-u-tests--statistical-significance) — a linear model cannot exploit the interaction effects that make seasonality informative.

---

## 4. Model B: XGBoost

### Hyperparameter Tuning

XGBoost hyperparameters are tuned via `RandomizedSearchCV` (40 iterations) with `TimeSeriesSplit(n_splits=4)`, scoring on ROC-AUC. Search distributions use `scipy.stats` for dense, continuous coverage of the parameter space.

**Best CV ROC-AUC: 0.9499**

| Parameter | Tuned Value | Search Distribution |
|---|---|---|
| `n_estimators` | **460** | `randint(100, 800)` |
| `max_depth` | **7** | `randint(3, 12)` |
| `learning_rate` | **0.034** | `loguniform(0.005, 0.3)` |
| `subsample` | **0.82** | `uniform(0.5, 1.0)` |
| `colsample_bytree` | **0.88** | `uniform(0.5, 1.0)` |
| `min_child_weight` | **8** | `randint(1, 15)` |
| `gamma` | **0.15** | `uniform(0, 0.5)` |
| `reg_alpha` | **0.42** | `loguniform(0.001, 5.0)` |
| `reg_lambda` | **0.63** | `loguniform(0.1, 10.0)` |
| `scale_pos_weight` | 4.0 | Fixed — compensates 1:4 class imbalance |
| `objective` | binary:logistic | Fixed |
| `tree_method` | hist | Fixed |

Using continuous distributions instead of discrete lists allows the search to find more precise parameter values (e.g., learning_rate=0.034 rather than being limited to {0.01, 0.03, 0.05}).

### Why XGBoost Over Logistic Regression?

Three architectural advantages make XGBoost the superior choice for this problem:

1. **Sparsity-Aware Split Finding** — XGBoost natively handles the ~1–5% NaN values in SST, Chlorophyll, and Salinity by learning an optimal default branch direction at each tree node. This eliminates the need for imputation, which can introduce bias (Ji et al., 2024).

2. **Non-Linear Feature Interactions** — Tree-based models capture conditional relationships like "SST < 10°C AND Month ∈ {3,4,5} AND Dist_to_Shore < 30km" that logistic regression cannot represent without manual feature engineering.

3. **Robustness to Multicollinearity** — The moderate correlations between Bathymetry/Dist_to_Shore (r = -0.63) and SST_Gradient/Is_Thermal_Front (r = 0.73) do not destabilise XGBoost as they would in a linear model.

### Default Threshold Performance (τ = 0.50)

| Metric | Value |
|---|---|
| **ROC-AUC** | 0.8991 |
| **Recall** | 0.7011 |
| **Precision** | 0.6285 |
| **F1-Score** | 0.6628 |
| **Accuracy** | 0.8564 |

At the default threshold, the tuned XGBoost achieves a **+9.4 pp AUC improvement** over logistic regression (0.8991 vs. 0.8049). However, the recall of 69.4% means ~31% of whale locations would be missed — unacceptable for endangered species management.

---

## 5. Model C: Random Forest

### Motivation

Random Forest serves as a critical **middle-ground comparator** between the linear baseline and the boosted ensemble. Both RF and XGBoost are tree-based, but they differ fundamentally in how they combine trees:

- **Random Forest (bagging):** Trains independent trees on bootstrap samples and averages their predictions. Each tree is intentionally deep and high-variance; averaging reduces that variance.
- **XGBoost (boosting):** Trains sequential trees, where each new tree corrects the errors of the previous ensemble. Trees are intentionally shallow and low-variance; boosting reduces bias.

By comparing RF against XGBoost, we can determine whether the sequential error-correction of boosting provides a meaningful advantage over the parallel averaging of bagging for NARW habitat prediction.

### Hyperparameter Tuning

Random Forest hyperparameters are tuned via `RandomizedSearchCV` (40 iterations) with `TimeSeriesSplit(n_splits=4)`, scoring on ROC-AUC. Integer parameters use `scipy.stats.randint` for dense coverage; `max_features` remains a discrete list due to mixed types.

**Best CV ROC-AUC: 0.9447**

| Parameter | Tuned Value | Search Distribution |
|---|---|---|
| `n_estimators` | **712** | `randint(100, 800)` |
| `max_depth` | **26** | `randint(5, 40)` |
| `min_samples_split` | **2** | `randint(2, 30)` |
| `min_samples_leaf` | **3** | `randint(1, 15)` |
| `max_features` | **0.3** | {sqrt, log2, 0.3, 0.5} |
| `class_weight` | balanced_subsample | Fixed — per-tree class rebalancing |
| `bootstrap` | True | Fixed — standard bagging with OOB scoring |

Notable tuning outcomes: the search selected `max_features=0.3` (30% of features per split) instead of the conventional `sqrt`, and a larger ensemble (712 trees) with deep, lightly regularised trees (max_depth=26, min_samples_leaf=3).

### Preprocessing

Like logistic regression, Random Forest in scikit-learn cannot handle NaN values natively. The training pipeline uses `SimpleImputer(strategy="median")` to fill missing values. Unlike logistic regression, **no feature scaling is required** — tree-based splits are invariant to monotonic transformations.

### Default Threshold Performance (τ = 0.50)

| Metric | Value |
|---|---|
| **ROC-AUC** | **0.9041** |
| **Recall** | 0.6410 |
| **Precision** | 0.7603 |
| **F1-Score** | 0.6956 |
| **Accuracy** | 0.8871 |

The tuned Random Forest achieves the **highest AUC of all three models** (0.9041), a **+0.5 pp improvement over XGBoost** (0.8991) and **+9.9 pp over logistic regression** (0.8049). The OOB score of 0.9264 provides an independent validation estimate.

### Optimised Threshold Performance (τ = 0.2018)

| Metric | Default (τ=0.50) | Optimised (τ=0.20) | Change |
|---|---|---|---|
| **ROC-AUC** | 0.9041 | 0.9041 | — (threshold-independent) |
| **Recall** | 0.6410 | **0.8006** ✓ | +15.9 pp |
| **Precision** | 0.7603 | 0.4572 | -30.3 pp |
| **F1-Score** | 0.6956 | 0.5820 | -11.4 pp |
| **Accuracy** | 0.8871 | 0.7685 | -11.9 pp |

**Confusion Matrix (Optimised):**

|  | Predicted Absence | Predicted Presence |
|---|---|---|
| **Actual Absence** | 7,884 TN | 2,484 FP |
| **Actual Presence** | 521 FN | 2,092 TP |

At the conservation-optimised threshold, the tuned Random Forest correctly identifies **80.1% of whale locations** while producing 2,484 false alarms — comparable to XGBoost (2,496).

![ROC curve — Random Forest with AUC = 0.9031, the green dot marks the operating point at the optimised threshold τ=0.179](images/rf_roc_curve.png)

![Precision-recall tradeoff curve for Random Forest — the green operating point (τ=0.179) achieves 81% recall at 47% precision](images/rf_precision_recall_tradeoff.png)

---

## 6. Threshold Optimisation for Conservation

### The Conservation Calculus

In endangered species management, the consequences of prediction errors are **asymmetric**:

- **False Negative** (missed whale) → A real whale location is not flagged → Potential ship strike fatality for a species with ~350 remaining individuals
- **False Positive** (false alarm) → An empty location triggers a speed restriction → Minor economic inconvenience

This asymmetry demands that we **prioritise recall** (minimising false negatives) over precision, even at the cost of more false alarms.

### Threshold Sweep

We sweep all classification thresholds using the precision-recall curve and select the threshold that achieves **≥80% recall** with the highest possible precision:

![Precision-recall tradeoff curve — the green operating point (τ=0.172) achieves 80% recall at 42% precision, meeting the conservation target](images/precision_recall_tradeoff.png)

**Optimal threshold (XGBoost): τ = 0.2229** (lowered from the default 0.50)

### Optimised Performance (XGBoost)

| Metric | Default (τ=0.50) | Optimised (τ=0.22) | Change |
|---|---|---|---|
| **ROC-AUC** | 0.8991 | 0.8991 | — (threshold-independent) |
| **Recall** | 0.7011 | **0.8002** ✓ | +9.9 pp |
| **Precision** | 0.6285 | 0.4559 | -17.3 pp |
| **F1-Score** | 0.6628 | 0.5808 | -8.2 pp |
| **Accuracy** | 0.8564 | 0.7675 | -8.9 pp |

**Confusion Matrix (Optimised):**

|  | Predicted Absence | Predicted Presence |
|---|---|---|
| **Actual Absence** | 7,872 TN | 2,496 FP |
| **Actual Presence** | 522 FN | 2,091 TP |

The optimised model correctly identifies **80% of whale locations** while generating ~2,671 false alarms per test period. In operational terms, this means a conservative alerting system that errs on the side of caution — consistent with the precautionary principle applied in marine mammal management.

![ROC curve — XGBoost with AUC = 0.8916, the green dot marks the operating point at the optimised threshold τ=0.185](images/roc_curve.png)

---

## 7. Head-to-Head Comparison

### Performance Summary (All Models Tuned)

All hyperparameters were tuned using `RandomizedSearchCV` (40 iterations) with `TimeSeriesSplit` (4 folds) and `scipy.stats` distributions for dense, continuous parameter space coverage.

| Metric | LR (Tuned) | XGBoost (τ=0.50) | XGBoost (τ=0.22) | RF (τ=0.50) | RF (τ=0.20) |
|---|---|---|---|---|---|
| **ROC-AUC** | 0.8049 | 0.8991 | 0.8991 | **0.9041** | **0.9041** |
| **Recall** | 0.8316 | 0.7011 | 0.8002 ✓ | 0.6410 | **0.8006** ✓ |
| **Precision** | 0.3389 | 0.6285 | 0.4559 | **0.7603** | 0.4572 |
| **F1-Score** | 0.4816 | 0.6628 | 0.5808 | **0.6956** | 0.5820 |
| **Accuracy** | 0.6396 | 0.8564 | 0.7675 | **0.8871** | 0.7685 |

![ROC comparison — XGBoost dominates LR across all operating points](images/roc_comparison.png)

![Model comparison bar chart showing the progression from baseline to optimised model](images/model_comparison.png)

### Key Observations

1. **Random Forest achieves the highest AUC (0.9041)** — A **+0.5 pp improvement** over XGBoost (0.8991) and **+9.9 pp over LR** (0.8049). Using `scipy.stats` distributions improved XGBoost by +0.8 pp (0.8916→0.8991) and RF by +0.1 pp (0.9031→0.9041) compared to discrete-list tuning.

2. **At 80% recall, both tree models are nearly identical** — RF produces 2,484 FP vs. XGBoost's 2,496 FP, and precision is 45.7% vs. 45.6%. The distribution-based tuning narrowed the gap between the two models.

3. **The LR recall "advantage" is misleading** — Logistic regression achieves 83% recall but at only 34% precision. Both tree-based models reach comparable recall at substantially higher precision.

4. **Continuous distributions improve XGBoost more than RF** — XGBoost has 9 interacting hyperparameters where precise values matter (e.g., learning_rate=0.034). RF's integer-valued params (tree depth, leaf size) benefit less from continuous sampling.

5. **Bagging vs. Boosting** — RF's AUC advantage over XGBoost persists but is now marginal (+0.5 pp). With better tuning, XGBoost closes the gap, suggesting the original discrete-list search was under-exploring XGBoost's parameter space.

---

## 8. Feature Importance & Interpretability

### XGBoost Gain-Based Importance

XGBoost's native gain metric measures the average improvement in loss function (AUC) when a feature is used for a split. Higher gain = more useful for separating whale presence from absence.

![XGBoost feature importance — Dist_to_Shore_km dominates (0.322), followed by Dist_to_Shelf_km (0.135) and Month (0.125)](images/feature_importance.png)

### Random Forest MDI-Based Importance

Random Forest uses Mean Decrease in Impurity (MDI) — the total reduction in Gini impurity averaged across all 500 trees. Error bars show inter-tree variability.

![Random Forest feature importance — Dist_to_Shore_km dominates (0.278), followed by Dist_to_Shelf_km (0.148) and Chlorophyll (0.137)](images/rf_feature_importance.png)

### Importance Ranking Comparison

| Rank | XGBoost (Gain) | | Random Forest (MDI) | |
|---|---|---|---|---|
| 1 | **Dist_to_Shore_km** | 0.322 | **Dist_to_Shore_km** | 0.278 |
| 2 | Dist_to_Shelf_km | 0.135 | Dist_to_Shelf_km | 0.148 |
| 3 | Month | 0.125 | Chlorophyll | 0.137 |
| 4 | Chlorophyll | 0.093 | Bathymetry | 0.117 |
| 5 | Salinity | 0.086 | Salinity | 0.104 |
| 6 | Bathymetry | 0.082 | SST | 0.093 |
| 7 | SST | 0.063 | Month | 0.054 |
| 8 | Bathy_Slope | 0.043 | Bathy_Slope | 0.037 |
| 9 | SST_Gradient | 0.029 | SST_Gradient | 0.030 |
| 10 | Is_Thermal_Front | 0.022 | Is_Thermal_Front | 0.003 |

### Cross-Model Consistency

All three models agree on the most important features, despite their fundamentally different architectures:

| Feature | LR Coefficient Rank | XGBoost Gain Rank | RF MDI Rank |
|---|---|---|---|
| **Dist_to_Shore_km** | **#1** (-2.044) | **#1** (0.322) | **#1** (0.278) |
| Dist_to_Shelf_km | #3 (-0.496) | #2 (0.135) | #2 (0.148) |
| Chlorophyll | #4 (+0.144) | #4 (0.093) | #3 (0.137) |
| Bathymetry | #2 (+1.552) | #6 (0.082) | #4 (0.117) |

The agreement across a linear model, a boosted ensemble, and a bagged ensemble on `Dist_to_Shore_km` as the top predictor provides very strong evidence that coastal proximity is genuinely the most informative predictor of NARW habitat, not an artefact of any single modelling approach.

### The Month Divergence

`Month` ranks #3 in XGBoost gain but only #7 in Random Forest and dead last in logistic regression (coefficient ≈ 0). XGBoost's sequential boosting architecture is better at extracting conditional interactions (e.g., "Month=4 AND SST<10") because each tree builds on previous errors, making it more sensitive to interaction effects. Random Forest's independent trees capture some of this signal but less efficiently. Logistic regression cannot represent interactions at all without manual feature engineering.

---

## 9. Manual Inference Testing

To validate ecological plausibility beyond statistical metrics, we test the trained XGBoost model against four hand-crafted scenarios representing known habitats and non-habitats:

### Test Scenarios

| Scenario | Location | Season | Expected | Probability | Prediction |
|---|---|---|---|---|---|
| 🐳 Cape Cod Bay | Shallow shelf, 8 km from shore | April | HIGH | High | **WHALE HABITAT ✓** |
| 🐳 Bay of Fundy | Deep basin, 30 km from shore | July | HIGH | High | **WHALE HABITAT ✓** |
| ❌ Mid-Atlantic Ridge | Abyssal depth, 400 km offshore | June | LOW | Low | **NOT HABITAT ✗** |
| ❌ Florida Keys | Tropical shallow, August | LOW | Borderline | ~21% | **Borderline/Positive** |

### Ecological Plausibility Assessment

**Cape Cod Bay (April):** The model correctly identifies this as prime habitat — shallow shelf water (60m), cold SST (8°C), high chlorophyll (6.0 mg/m³), active thermal fronts, and very close to shore (8 km). This is the most well-documented NARW spring feeding ground.

**Bay of Fundy (July):** Correctly identified as habitat. The combination of moderate depth (170m), summer SST (12°C), productive water, and proximity to the continental shelf edge creates ideal *Calanus* aggregation conditions.

**Mid-Atlantic Ridge:** Correctly rejected. Abyssal depth (4,500m), warm oligotrophic water (22°C), extreme distance from shore (400 km), and no thermal front activity are incompatible with NARW ecology.

**Florida Keys (August):** This scenario produces a probability of ~21.3%, which exceeds the 17.2% operational threshold. This is a known edge case: the model errs on the side of caution for a location that is warm (28°C) and in the wrong season but still close to shore (5 km). This is the expected behaviour of a high-recall, cautious model — a minor false positive that is preferable to missing a real whale.

---

## 10. Model Artefacts & Deployment

### Saved Models

| File | Format | Size | Contents |
|---|---|---|---|
| `models/xgb_narw_sdm.json` | XGBoost JSON | 2.4 MB | Full trained XGBoost ensemble (500 trees) |
| `models/rf_narw_sdm.joblib` | joblib Pipeline | ~50 MB | Complete RF pipeline (imputer + 500-tree classifier) |
| `models/lr_narw_sdm.joblib` | joblib Pipeline | 2.4 KB | Complete LR pipeline (imputer + scaler + classifier) |
| `models/optimal_threshold.txt` | Plain text | 247 B | XGBoost: τ = 0.1718, with associated metrics |
| `models/rf_optimal_threshold.txt` | Plain text | 260 B | RF: τ = 0.2454, with associated metrics |

### Inference Pipeline

To generate a prediction for a new (lat, lon, date) observation:

```
1. Extract 10 environmental features using the same ETL pipeline
2. Load model:  model = xgb.XGBClassifier(); model.load_model("models/xgb_narw_sdm.json")
3. Predict:     probability = model.predict_proba(features)[0][1]
4. Classify:    is_habitat = probability >= 0.1718
```

### Generated Visualisations

All model evaluation plots are saved to `images/` at 200 DPI:

| File | Description |
|---|---|
| `roc_curve.png` | XGBoost ROC curve with operating point |
| `rf_roc_curve.png` | Random Forest ROC curve with operating point |
| `lr_roc_curve.png` | Logistic regression ROC curve |
| `roc_comparison.png` | Side-by-side ROC comparison (LR vs XGBoost) |
| `precision_recall_tradeoff.png` | XGBoost PR curve with recall target line |
| `rf_precision_recall_tradeoff.png` | Random Forest PR curve with recall target line |
| `feature_importance.png` | XGBoost gain-based feature ranking |
| `rf_feature_importance.png` | Random Forest MDI-based feature ranking (with error bars) |
| `lr_coefficients.png` | LR coefficient bar chart |
| `model_comparison.png` | Multi-metric comparison bar chart |

---

*For exploratory data analysis and ecological justification, see the [EDA Walkthrough](EDA_Walkthrough.md).*

*For the high-level project overview, pipeline architecture, and references, see the [README](README.md).*
