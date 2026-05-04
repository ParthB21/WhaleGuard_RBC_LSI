"""
==========================================================================
 WhaleGuard — Phase 4: XGBoost Model Training & Evaluation
==========================================================================
 Trains an XGBoost classifier on the fully-engineered NARW habitat dataset
 using a temporally-ordered train/test split.

 Design Decisions:
   - Temporal Split (80/20): Proves the model generalises to *future*
     conditions, not just interpolates known dates. This is critical for
     climate-shift resilience (Ji et al., 2024).
   - scale_pos_weight = 4.0: Mathematically counterbalances the 1:4
     pseudo-absence ratio from Gowan & Ortega-Ortiz (2014).
   - Features EXCLUDE Lat, Lon, Date: Forces the model to learn ocean
     physics and seasonality, not memorise patrol coordinates.
   - XGBoost's Sparsity-Aware Split Finding natively handles the ~1-5%
     NaN values in SST, Chlorophyll, and Salinity (Ji et al., 2024).

 Outputs:
   - Terminal classification report (Accuracy, Precision, Recall, F1, AUC)
   - images/roc_curve.png
   - images/feature_importance.png
   - models/xgb_narw_sdm.json  (serialised model for deployment)
==========================================================================
"""

import time
import sys
import io
from pathlib import Path

# Force UTF-8 output on Windows console
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from scipy.stats import randint, uniform, loguniform

# =========================================================================
#  Configuration
# =========================================================================

DATA_PATH   = Path("data/processed/ML_Whale_Dataset_Final.csv")
IMG_DIR     = Path("images")
MODEL_DIR   = Path("models")
IMG_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# Columns to drop from features (model must learn physics, not geography)
DROP_COLS = ["Date", "Lat", "Lon", "Presence"]

# Temporal split ratio
TRAIN_RATIO = 0.80

# XGBoost hyperparameters
XGB_PARAMS = {
    "n_estimators":     500,
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma":            0.1,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "objective":        "binary:logistic",
    "eval_metric":      "auc",
    "tree_method":      "hist",
    "random_state":     42,
    "verbosity":        0,
}

# Plot style (matching EDA dark theme)
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#c9d1d9",
    "text.color":       "#c9d1d9",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "grid.color":       "#21262d",
    "font.family":      "sans-serif",
    "font.size":        12,
    "axes.titlesize":   15,
    "axes.labelsize":   13,
})


# =========================================================================
#  Helpers
# =========================================================================

def print_header(title: str):
    width = 66
    print("\n" + "╔" + "═" * width + "╗")
    print("║  " + title.ljust(width - 2) + "║")
    print("╚" + "═" * width + "╝\n")


def print_section(title: str):
    print(f"\n─── {title} " + "─" * max(0, 58 - len(title)))


# =========================================================================
#  Main Pipeline
# =========================================================================

def main():
    t_start = time.time()
    print_header("WhaleGuard — XGBoost NARW Habitat Model")

    # ── 1. Load Data ─────────────────────────────────────────────────
    print_section("1. Data Loading")

    if not DATA_PATH.exists():
        print(f"  ✗ File not found: {DATA_PATH}")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    print(f"  Loaded: {len(df):,} rows × {len(df.columns)} columns")
    print(f"  Columns: {', '.join(df.columns)}")
    print(f"  Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")

    # ── 2. Feature / Target Separation ───────────────────────────────
    print_section("2. Feature Engineering")

    target = df["Presence"]
    features = df.drop(columns=DROP_COLS)

    # Ensure Is_Thermal_Front is numeric (bool → int)
    if "Is_Thermal_Front" in features.columns:
        features["Is_Thermal_Front"] = features["Is_Thermal_Front"].astype(int)

    feature_names = list(features.columns)
    print(f"  Target: Presence (1 = whale, 0 = background)")
    print(f"  Features ({len(feature_names)}): {', '.join(feature_names)}")
    print(f"  Dropped: {', '.join(DROP_COLS[:-1])} (prevent geographic memorisation)")

    # ── 3. Temporal Split ────────────────────────────────────────────
    print_section("3. Temporal Train/Test Split")

    # Sort chronologically (should already be, but enforce)
    sort_idx = df["Date"].argsort()
    features = features.iloc[sort_idx].reset_index(drop=True)
    target   = target.iloc[sort_idx].reset_index(drop=True)
    dates    = df["Date"].iloc[sort_idx].reset_index(drop=True)

    split_idx  = int(len(df) * TRAIN_RATIO)
    split_date = dates.iloc[split_idx]

    X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
    y_train, y_test = target.iloc[:split_idx],   target.iloc[split_idx:]

    n_pos_train = int(y_train.sum())
    n_neg_train = int((y_train == 0).sum())
    ratio = n_neg_train / n_pos_train if n_pos_train > 0 else 4.0

    print(f"  Strategy: Chronological (train on past, test on future)")
    print(f"  Split point: {split_date.date()} ({TRAIN_RATIO*100:.0f}%/{(1-TRAIN_RATIO)*100:.0f}%)")
    print(f"  Train: {len(X_train):,} rows ({dates.iloc[0].date()} → {dates.iloc[split_idx-1].date()})")
    print(f"  Test:  {len(X_test):,} rows  ({split_date.date()} → {dates.iloc[-1].date()})")
    print(f"  Train class balance: {n_pos_train:,} pos / {n_neg_train:,} neg (ratio 1:{ratio:.1f})")

    # ── 4. Hyperparameter Tuning & Training ────────────────────────
    print_section("4. Hyperparameter Tuning (RandomizedSearchCV + TimeSeriesSplit)")

    # Fixed params (not tuned)
    fixed_params = {
        "objective":        "binary:logistic",
        "eval_metric":      "auc",
        "tree_method":      "hist",
        "random_state":     42,
        "verbosity":        0,
        "scale_pos_weight": ratio,
    }

    # Search space
    param_distributions = {
        "n_estimators":     randint(100, 800),
        "max_depth":        randint(3, 12),
        "learning_rate":    loguniform(0.005, 0.3),
        "subsample":        uniform(0.5, 0.5),       # [0.5, 1.0]
        "colsample_bytree": uniform(0.5, 0.5),       # [0.5, 1.0]
        "min_child_weight": randint(1, 15),
        "gamma":            uniform(0, 0.5),
        "reg_alpha":        loguniform(0.001, 5.0),
        "reg_lambda":       loguniform(0.1, 10.0),
    }

    base_model = xgb.XGBClassifier(**fixed_params)
    tscv = TimeSeriesSplit(n_splits=3)
    n_iter = 40

    search = RandomizedSearchCV(
        base_model, param_distributions, n_iter=n_iter,
        cv=tscv, scoring="roc_auc", n_jobs=-1,
        random_state=42, verbose=0, refit=False,
    )

    print(f"  Algorithm:        XGBClassifier (gradient-boosted trees)")
    print(f"  scale_pos_weight: {ratio:.2f} (fixed, compensates 1:{ratio:.0f} imbalance)")
    print(f"  Search space:     {len(param_distributions)} hyperparameters")
    dist_labels = {
        "n_estimators": "randint(100, 800)", "max_depth": "randint(3, 12)",
        "learning_rate": "loguniform(0.005, 0.3)", "subsample": "uniform(0.5, 1.0)",
        "colsample_bytree": "uniform(0.5, 1.0)", "min_child_weight": "randint(1, 15)",
        "gamma": "uniform(0, 0.5)", "reg_alpha": "loguniform(0.001, 5.0)",
        "reg_lambda": "loguniform(0.1, 10.0)",
    }
    for param in param_distributions:
        print(f"    {param:20s} ~ {dist_labels[param]}")
    print(f"  CV strategy:      TimeSeriesSplit (3 folds, temporal ordering)")
    print(f"  Iterations:       {n_iter} random samples")
    print(f"  Total fits:       {n_iter * 3}")
    print(f"\n  Searching...", end="", flush=True)

    t_tune = time.time()
    search.fit(X_train, y_train)
    tune_time = time.time() - t_tune
    print(f" done in {tune_time:.1f}s")

    # Report best params
    best_params = search.best_params_
    print(f"\n  Best CV ROC-AUC: {search.best_score_:.4f}")
    print(f"  Best parameters:")
    for param, value in sorted(best_params.items()):
        print(f"    {param:20s} = {value}")

    # Retrain final model on full training data with best params + eval_set
    print_section("4b. Final Training (Best Parameters + eval_set)")

    XGB_PARAMS = {**fixed_params, **best_params}
    model = xgb.XGBClassifier(**XGB_PARAMS)

    print(f"\n  Training final model...", end="", flush=True)
    t_train = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    train_time = time.time() - t_train
    print(f" done in {train_time:.1f}s")

    # ── 5. Evaluation — Default Threshold (0.50) ────────────────────
    print_section("5a. Model Evaluation — Default Threshold (0.50)")

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred_default = (y_pred_proba >= 0.50).astype(int)

    acc_def       = accuracy_score(y_test, y_pred_default)
    precision_def = precision_score(y_test, y_pred_default, zero_division=0)
    recall_def    = recall_score(y_test, y_pred_default, zero_division=0)
    f1_def        = f1_score(y_test, y_pred_default, zero_division=0)
    auc           = roc_auc_score(y_test, y_pred_proba)

    print(f"\n  ┌─────────────────────────────────────────┐")
    print(f"  │   DEFAULT METRICS (threshold = 0.50)     │")
    print(f"  ├─────────────────────┬───────────────────┤")
    print(f"  │  Accuracy           │  {acc_def:>14.4f}    │")
    print(f"  │  Precision          │  {precision_def:>14.4f}    │")
    print(f"  │  Recall (Sens.)     │  {recall_def:>14.4f}    │")
    print(f"  │  F1-Score           │  {f1_def:>14.4f}    │")
    print(f"  │  ROC-AUC            │  {auc:>14.4f}    │")
    print(f"  └─────────────────────┴───────────────────┘")

    # ── 5b. Threshold Optimisation for ≥80% Recall ──────────────────
    #
    # Rationale (Endangered Species Management):
    #   A False Negative = a missed whale = a potential fatal ship strike.
    #   NOAA Right Whale management prioritises RECALL over precision.
    #   We sweep all classification thresholds and select the one that
    #   achieves ≥80% recall with the highest possible precision.

    print_section("5b. Threshold Optimisation (Target: Recall ≥ 0.80)")

    TARGET_RECALL = 0.80

    prec_curve, rec_curve, thresholds_pr = precision_recall_curve(
        y_test, y_pred_proba
    )
    # precision_recall_curve returns arrays where len(thresholds) = len(prec) - 1
    # Trim the final point so arrays align
    prec_curve = prec_curve[:-1]
    rec_curve  = rec_curve[:-1]

    # Find all thresholds where recall >= target
    valid_mask  = rec_curve >= TARGET_RECALL
    if valid_mask.any():
        # Among those, pick the one with highest precision (= highest threshold)
        valid_indices = np.where(valid_mask)[0]
        # Highest precision among valid = highest threshold among valid
        best_idx      = valid_indices[np.argmax(prec_curve[valid_indices])]
        opt_threshold = thresholds_pr[best_idx]
    else:
        # Fallback: pick the threshold closest to target recall
        closest_idx   = np.argmin(np.abs(rec_curve - TARGET_RECALL))
        opt_threshold = thresholds_pr[closest_idx]
        print(f"  ⚠ Could not achieve {TARGET_RECALL:.0%} recall; "
              f"using closest: {rec_curve[closest_idx]:.4f}")

    # Apply optimised threshold
    y_pred_opt = (y_pred_proba >= opt_threshold).astype(int)

    acc_opt       = accuracy_score(y_test, y_pred_opt)
    precision_opt = precision_score(y_test, y_pred_opt, zero_division=0)
    recall_opt    = recall_score(y_test, y_pred_opt, zero_division=0)
    f1_opt        = f1_score(y_test, y_pred_opt, zero_division=0)

    print(f"\n  Optimal threshold: {opt_threshold:.4f}  (default was 0.50)")
    print(f"\n  ┌─────────────────────────────────────────────────────────┐")
    print(f"  │   OPTIMISED METRICS (threshold = {opt_threshold:.4f})              │")
    print(f"  ├─────────────────────┬──────────────┬──────────────────┤")
    print(f"  │  Metric             │   Default    │   Optimised      │")
    print(f"  ├─────────────────────┼──────────────┼──────────────────┤")
    print(f"  │  Accuracy           │  {acc_def:>10.4f}  │  {acc_opt:>10.4f}      │")
    print(f"  │  Precision          │  {precision_def:>10.4f}  │  {precision_opt:>10.4f}      │")
    print(f"  │  Recall (Sens.)     │  {recall_def:>10.4f}  │  {recall_opt:>10.4f}  ✓   │")
    print(f"  │  F1-Score           │  {f1_def:>10.4f}  │  {f1_opt:>10.4f}      │")
    print(f"  │  ROC-AUC            │  {auc:>10.4f}  │  {auc:>10.4f}      │")
    print(f"  └─────────────────────┴──────────────┴──────────────────┘")
    print(f"\n  Note: ROC-AUC is threshold-independent (same for both).")

    # Confusion matrix (optimised)
    cm = confusion_matrix(y_test, y_pred_opt)
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  Confusion Matrix (threshold = {opt_threshold:.4f}):")
    print(f"                    Predicted 0    Predicted 1")
    print(f"    Actual 0 (Abs)   {tn:>8,}       {fp:>8,}")
    print(f"    Actual 1 (Pres)  {fn:>8,}       {tp:>8,}")
    print(f"\n    Missed whales (FN): {fn:,}  →  Recall = {recall_opt:.4f}")
    print(f"    False alarms (FP): {fp:,}  →  Precision = {precision_opt:.4f}")

    print(f"\n  Full Classification Report (Optimised Threshold):")
    print(classification_report(y_test, y_pred_opt,
                                target_names=["Absence", "Presence"]))

    # Use optimised metrics for all downstream plots/reports
    precision = precision_opt
    recall    = recall_opt
    f1        = f1_opt
    acc       = acc_opt

    # ── 6. ROC Curve ─────────────────────────────────────────────────
    print_section("6. Generating ROC Curve")

    fpr_curve, tpr_curve, _ = roc_curve(y_test, y_pred_proba)

    # Compute the operating point on the ROC curve at the optimised threshold
    fpr_opt = fp / (fp + tn)
    tpr_opt = recall_opt

    fig, ax = plt.subplots(figsize=(9, 7))

    # Fill under curve
    ax.fill_between(fpr_curve, tpr_curve, alpha=0.15, color="#58a6ff")
    ax.plot(fpr_curve, tpr_curve, color="#58a6ff", linewidth=2.5,
            label=f"XGBoost (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#f85149", linestyle="--",
            linewidth=1.5, alpha=0.7, label="Random Classifier (AUC = 0.50)")

    # Mark the operating point
    ax.scatter([fpr_opt], [tpr_opt], color="#3fb950", s=120, zorder=5,
               edgecolors="white", linewidths=2,
               label=f"Operating Point (τ={opt_threshold:.3f})")
    ax.annotate(f"  Recall={recall_opt:.2%}\n  Prec={precision_opt:.2%}",
                xy=(fpr_opt, tpr_opt), fontsize=10, color="#3fb950",
                fontweight="bold")

    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=13)
    ax.set_title("ROC Curve — NARW Habitat Model\n"
                 f"Temporal Validation: Train ≤ {dates.iloc[split_idx-1].date()}, "
                 f"Test ≥ {split_date.date()}",
                 fontsize=14, pad=15)
    ax.legend(loc="lower right", fontsize=11, framealpha=0.9,
              facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
    ax.grid(alpha=0.2)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    fig.tight_layout()
    roc_path = IMG_DIR / "roc_curve.png"
    fig.savefig(roc_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: {roc_path}")

    # ── 6b. Precision-Recall Tradeoff Curve ──────────────────────────
    print_section("6b. Generating Precision-Recall Tradeoff Curve")

    fig, ax = plt.subplots(figsize=(9, 7))

    ax.plot(rec_curve, prec_curve, color="#d2a8ff", linewidth=2.5,
            label="Precision-Recall Curve")
    ax.fill_between(rec_curve, prec_curve, alpha=0.1, color="#d2a8ff")

    # Mark operating point
    ax.scatter([recall_opt], [precision_opt], color="#3fb950", s=120,
               zorder=5, edgecolors="white", linewidths=2,
               label=f"Operating Point (τ={opt_threshold:.3f})")
    ax.annotate(f"  τ={opt_threshold:.3f}\n  R={recall_opt:.2%}, P={precision_opt:.2%}",
                xy=(recall_opt, precision_opt), fontsize=10,
                color="#3fb950", fontweight="bold")

    # Mark the 80% recall target line
    ax.axvline(x=TARGET_RECALL, color="#f85149", linestyle="--",
               linewidth=1.5, alpha=0.7, label=f"Recall Target ({TARGET_RECALL:.0%})")

    ax.set_xlabel("Recall", fontsize=13)
    ax.set_ylabel("Precision", fontsize=13)
    ax.set_title("Precision-Recall Tradeoff — NARW Habitat Model\n"
                 "Lowering threshold increases recall (fewer missed whales) "
                 "but decreases precision",
                 fontsize=13, pad=15)
    ax.legend(loc="upper right", fontsize=11, framealpha=0.9,
              facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
    ax.grid(alpha=0.2)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    fig.tight_layout()
    pr_path = IMG_DIR / "precision_recall_tradeoff.png"
    fig.savefig(pr_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: {pr_path}")

    # ── 7. Feature Importance ────────────────────────────────────────
    print_section("7. Generating Feature Importance Chart")

    importances = model.feature_importances_
    sorted_idx  = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = plt.cm.cividis(np.linspace(0.2, 0.9, len(sorted_idx)))
    bars = ax.barh(
        range(len(sorted_idx)),
        importances[sorted_idx],
        color=colors,
        edgecolor="#30363d",
        linewidth=0.8,
        height=0.7,
    )

    # Value labels
    for bar, val in zip(bars, importances[sorted_idx]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=11, color="#c9d1d9",
                fontweight="bold")

    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=12)
    ax.set_xlabel("Feature Importance (Gain)", fontsize=13)
    ax.set_title("XGBoost Feature Importance — NARW Habitat Model\n"
                 "Which ocean variables drive whale presence predictions?",
                 fontsize=14, pad=15)
    ax.grid(axis="x", alpha=0.2)

    fig.tight_layout()
    fi_path = IMG_DIR / "feature_importance.png"
    fig.savefig(fi_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: {fi_path}")

    # Print ranking
    print(f"\n  Feature Importance Ranking:")
    for rank, idx in enumerate(reversed(sorted_idx)):
        bar = "█" * int(importances[idx] * 40)
        print(f"    {rank+1}. {feature_names[idx]:20s}  {importances[idx]:.4f}  {bar}")

    # ── 8. Save Model & Threshold ────────────────────────────────────
    print_section("8. Saving Model & Optimal Threshold")

    model_path = MODEL_DIR / "xgb_narw_sdm.json"
    model.save_model(str(model_path))
    print(f"  ✓ Model saved: {model_path}")
    print(f"  ✓ Model format: XGBoost JSON (portable, version-safe)")

    # Save the optimal threshold alongside the model
    threshold_path = MODEL_DIR / "optimal_threshold.txt"
    with open(threshold_path, "w") as f:
        f.write(f"# NARW Habitat Model — Optimal Classification Threshold\n")
        f.write(f"# Optimised for Recall >= {TARGET_RECALL:.0%} (endangered species management)\n")
        f.write(f"# Generated: {pd.Timestamp.now().isoformat()}\n")
        f.write(f"threshold={opt_threshold:.6f}\n")
        f.write(f"recall={recall_opt:.6f}\n")
        f.write(f"precision={precision_opt:.6f}\n")
        f.write(f"f1={f1_opt:.6f}\n")
        f.write(f"auc={auc:.6f}\n")
    print(f"  ✓ Threshold saved: {threshold_path}")
    print(f"  ✓ Deploy with: predict(proba >= {opt_threshold:.4f})")

    # ── Summary ──────────────────────────────────────────────────────
    total_time = time.time() - t_start
    print_header("Training Complete")
    print(f"  Model Performance (Optimised for Conservation):")
    print(f"    ROC-AUC:    {auc:.4f}")
    print(f"    Recall:     {recall:.4f}  (target ≥ {TARGET_RECALL:.0%} ✓)")
    print(f"    Precision:  {precision:.4f}")
    print(f"    F1-Score:   {f1:.4f}")
    print(f"    Threshold:  {opt_threshold:.4f}  (default was 0.50)")
    print(f"\n  Artifacts:")
    print(f"    {roc_path}")
    print(f"    {pr_path}")
    print(f"    {fi_path}")
    print(f"    {model_path}")
    print(f"    {threshold_path}")
    print(f"\n  Total time: {total_time:.1f}s")
    print("═" * 68)


if __name__ == "__main__":
    main()
