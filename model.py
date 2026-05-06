# Suppress sklearn/joblib parallel worker UserWarnings BEFORE any imports
import os, warnings
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"
warnings.filterwarnings("ignore")

"""
PART 5 — Model Training & Evaluation (4-Model Ablation)
=========================================================
Trains four binary classifiers and compares them in a nested ablation:
  A. LR Baseline         — Logistic Regression on rank + CTFI
  B. RF — No CTFI        — Random Forest on rank only (ablation control)
  C. RF — Traditional    — Random Forest on rank + CTFI
  D. RF — Full           — Random Forest on rank + CTFI + NLP

Additional outputs:
  - McNemar's paired significance test (CTFI ablation, NLP ablation)
  - SHAP feature importance (full model)
  - Calibration curves, precision-recall curves, ROC curves, confusion matrices
  - Metric summary bar chart

Usage:
    python model.py
"""

import warnings, joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2

from sklearn.ensemble  import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline  import Pipeline
from sklearn.compose   import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute    import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, recall_score,
    roc_curve, precision_recall_curve, confusion_matrix,
    ConfusionMatrixDisplay, brier_score_loss
)
from sklearn.calibration import calibration_curve

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[WARN] shap not installed — SHAP plot will be skipped.")

warnings.filterwarnings("ignore", category=UserWarning)

FEATURES_CSV = "features.csv"
MODEL_OUT    = "upset_model.pkl"
PLOTS_DIR    = "."

# ── Feature sets ──────────────────────────────────────────────────────────────

RANK_FEATURES = [
    "rank", "opp_rank", "rank_ratio", "log_rank_diff",
    "is_underdog", "round_num", "best_of",
]

CTFI_FEATURES = ["ctfi_minutes", "ctfi_sets"]

NLP_FEATURES = [
    "sentiment_polarity", "fatigue_total", "fatigue_word_density",
    "fatigue_physical", "fatigue_mental", "fatigue_schedule",
    "fatigue_injury", "fatigue_motivation",
    "first_person_rate", "negation_rate", "llm_is_fatigued",
]

CATEGORICAL_FEATURES = ["rank_bin"]
TARGET = "upset"

# Columns to drop (identifiers / non-features)
DROP_COLS = [
    "player_id", "player_name", "tourney_date", "slam_name",
    "surface", "tour",
]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_features(path: str = FEATURES_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns from {path}")
    # Drop identifier / non-numeric non-feature columns
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
    return df


def time_split(df: pd.DataFrame, test_frac: float = 0.20):
    n_test = int(len(df) * test_frac)
    return df.iloc[:-n_test].copy(), df.iloc[-n_test:].copy()


# ── Preprocessing ─────────────────────────────────────────────────────────────

def build_preprocessor(feature_cols: list, cat_cols: list):
    num_cols = [c for c in feature_cols if c not in cat_cols]
    steps = [("num", Pipeline([
        ("imp",   SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), num_cols)]
    if cat_cols:
        steps.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), cat_cols))
    return ColumnTransformer(steps, remainder="drop")


# ── Model builders ────────────────────────────────────────────────────────────

def make_lr_pipeline(feature_cols, cat_cols):
    return Pipeline([
        ("prep",  build_preprocessor(feature_cols, cat_cols)),
        ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
    ])


def make_rf_pipeline(feature_cols, cat_cols):
    return Pipeline([
        ("prep",  build_preprocessor(feature_cols, cat_cols)),
        ("model", RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )),
    ])


# ── Tuning ────────────────────────────────────────────────────────────────────

RF_GRID = {
    "model__n_estimators":     [200, 400],
    "model__max_depth":        [None, 8, 15],
    "model__min_samples_leaf": [3, 7],
    "model__max_features":     ["sqrt", 0.5],
}
LR_GRID = {
    "model__C":       [0.01, 0.1, 1.0, 10.0],
    "model__solver":  ["lbfgs", "saga"],
    "model__max_iter": [1000],
}


def tune(pipeline, X, y, param_grid, cv=3, scoring="roc_auc"):
    cv_strat = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    gs = GridSearchCV(pipeline, param_grid, cv=cv_strat, scoring=scoring,
                      n_jobs=-1, refit=True)
    gs.fit(X, y)
    print(f"\nBest CV {scoring} : {gs.best_score_:.4f}")
    print(f"Best params     : {gs.best_params_}")
    return gs.best_estimator_


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(pipe, X_test, y_test, label):
    y_prob = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    roc  = roc_auc_score(y_test, y_prob)
    ap   = average_precision_score(y_test, y_prob)
    f1   = f1_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    brier= brier_score_loss(y_test, y_prob)
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  ROC-AUC           : {roc:.4f}")
    print(f"  PR-AUC            : {ap:.4f}")
    print(f"  F1 (upset)        : {f1:.4f}")
    print(f"  Recall (upset)    : {rec:.4f}")
    print(f"  Brier Score       : {brier:.4f}")
    return {"label": label, "roc_auc": roc, "pr_auc": ap, "f1": f1,
            "recall": rec, "brier": brier, "y_prob": y_prob, "y_pred": y_pred}


# ── McNemar's test ────────────────────────────────────────────────────────────

def mcnemar_test(y_test, y_pred_a, y_pred_b, label_a, label_b):
    """McNemar's continuity-corrected chi-squared test."""
    correct_a = (y_pred_a == y_test.values)
    correct_b = (y_pred_b == y_test.values)
    n01 = (~correct_a & correct_b).sum()  # b correct, a wrong
    n10 = (correct_a & ~correct_b).sum()  # a correct, b wrong
    denom = n01 + n10
    if denom == 0:
        chi2_stat, p = 0.0, 1.0
    else:
        chi2_stat = (abs(n01 - n10) - 1) ** 2 / denom
        p = 1 - chi2.cdf(chi2_stat, df=1)
    sig = "✓ SIGNIFICANT" if p < 0.05 else "✗ not significant"
    print(f"\n  McNemar ({label_b} vs {label_a}): "
          f"n01={n01}, n10={n10}, χ²={chi2_stat:.2f}, p={p:.3f}  {sig}")
    return {"n01": n01, "n10": n10, "chi2": chi2_stat, "p": p}


# ── Plots ─────────────────────────────────────────────────────────────────────

COLORS = {"LR Baseline": "#6b7280", "RF-No-CTFI": "#f59e0b",
          "RF-Traditional": "#3b82f6", "RF-Full": "#10b981"}


def plot_roc_curves(results_list, y_test, path="roc_curves_all.png"):
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in results_list:
        fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
        ax.plot(fpr, tpr, label=f"{r['label']} (AUC={r['roc_auc']:.3f})",
                color=COLORS.get(r["label"], "#555"), lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random (0.500)")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Model Variants")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{path}", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def plot_pr_curves(results_list, y_test, base_rate, path="precision_recall_curves.png"):
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in results_list:
        prec, rec, _ = precision_recall_curve(y_test, r["y_prob"])
        ax.plot(rec, prec, label=f"{r['label']} (AP={r['pr_auc']:.3f})",
                color=COLORS.get(r["label"], "#555"), lw=2)
    ax.axhline(base_rate, color="gray", ls="--", lw=1, label=f"No-skill ({base_rate:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{path}", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def plot_confusion_matrices(results_list, y_test, path="confusion_matrices.png"):
    n = len(results_list)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, r in zip(axes, results_list):
        cm = confusion_matrix(y_test, r["y_pred"])
        disp = ConfusionMatrixDisplay(cm, display_labels=["No Upset", "Upset"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(r["label"], fontsize=9)
    plt.suptitle("Confusion Matrices (threshold = 0.5)", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{path}", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def plot_calibration_curves(results_list, y_test, path="calibration_curves.png"):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    for r in results_list:
        prob_true, prob_pred = calibration_curve(y_test, r["y_prob"], n_bins=8)
        ax.plot(prob_pred, prob_true, "o-", label=r["label"],
                color=COLORS.get(r["label"], "#555"), lw=1.5, ms=5)
    ax.set_xlabel("Mean Predicted Probability"); ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration (Reliability) Curves")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{path}", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def plot_metric_summary(results_list, path="metric_summary.png"):
    labels = [r["label"] for r in results_list]
    metrics = {"PR-AUC": "pr_auc", "ROC-AUC": "roc_auc", "F1 (Upset)": "f1"}
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (metric_name, key) in enumerate(metrics.items()):
        vals = [r[key] for r in results_list]
        ax.bar(x + i * width, vals, width, label=metric_name,
               color=["#3b82f6", "#10b981", "#f59e0b"][i], edgecolor="none")
    ax.axhline(0.182, color="gray", ls="--", lw=1, alpha=0.7, label="PR no-skill baseline")
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, rotation=12, fontsize=9)
    ax.set_ylim(0, 0.9); ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Key Metrics")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{path}", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def plot_shap(pipe, X_test_df, feature_cols, cat_cols, label, path="shap_importance.png"):
    if not SHAP_AVAILABLE:
        return
    try:
        X_transformed = pipe.named_steps["prep"].transform(X_test_df)
        rf = pipe.named_steps["model"]
        explainer  = shap.TreeExplainer(rf)
        shap_vals  = explainer.shap_values(X_transformed)

        # SHAP API changed across versions:
        # Old (<=0.41): returns list [neg_class_array, pos_class_array]
        # New (>=0.42): returns single array of shape (n, features) or (n, features, 2)
        if isinstance(shap_vals, list):
            sv = shap_vals[1]                        # binary: take positive class
        elif isinstance(shap_vals, np.ndarray):
            if shap_vals.ndim == 3:
                sv = shap_vals[:, :, 1]              # shape (n, features, 2)
            else:
                sv = shap_vals                       # shape (n, features) — already pos class
        else:
            sv = np.array(shap_vals)

        # Feature names after transform
        prep = pipe.named_steps["prep"]
        num_names = list(prep.transformers_[0][2])
        cat_names = list(prep.transformers_[1][2]) if len(prep.transformers_) > 1 else []
        all_names = num_names + cat_names

        mean_abs = np.abs(sv).mean(axis=0)
        idx = np.argsort(mean_abs)[::-1][:20]
        names = [all_names[i] if i < len(all_names) else f"f{i}" for i in idx[::-1]]

        fig, ax = plt.subplots(figsize=(9, 7))
        ax.barh(names, mean_abs[idx[::-1]], color="#2563eb", edgecolor="none")
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"SHAP Feature Importance — {label}")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{PLOTS_DIR}/{path}", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved → {path}")
    except Exception as e:
        print(f"  [WARN] SHAP plot failed: {e}")


def plot_ctfi_by_surface(df: pd.DataFrame, path="ctfi_upset_by_surface.png"):
    """CTFI decile vs upset rate, faceted by surface."""
    surfaces = [s for s in ["Hard", "Clay", "Grass"] if f"surface_{s}" in df.columns]
    if not surfaces:
        return
    fig, axes = plt.subplots(1, len(surfaces), figsize=(5 * len(surfaces), 4), sharey=True)
    if len(surfaces) == 1:
        axes = [axes]
    ctfi_col = "ctfi_minutes" if "ctfi_minutes" in df.columns else "ctfi"
    for ax, surf in zip(axes, surfaces):
        mask = df[f"surface_{surf}"] == 1
        sub  = df[mask].copy()
        if len(sub) < 50:
            continue
        sub["ctfi_decile"] = pd.qcut(sub[ctfi_col], q=10, duplicates="drop",
                                      labels=False)
        stats = sub.groupby("ctfi_decile")["upset"].agg(["mean", "count"])
        # 95% binomial CI
        ci = 1.96 * np.sqrt(stats["mean"] * (1 - stats["mean"]) / stats["count"])
        ax.errorbar(stats.index, stats["mean"] * 100, yerr=ci * 100,
                    fmt="o-", capsize=4, color="#2563eb", lw=2, ms=5)
        ax.axhline(df["upset"].mean() * 100, color="gray", ls="--", lw=1,
                   alpha=0.6, label="Overall rate")
        ax.set_title(surf); ax.set_xlabel("CTFI Decile")
        if ax == axes[0]:
            ax.set_ylabel("Upset Rate (%)")
        ax.grid(alpha=0.3)
    plt.suptitle("CTFI Decile vs Upset Rate by Surface", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{path}", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def plot_correlation_heatmap(df: pd.DataFrame, path="correlation_heatmap.png"):
    num_df = df.select_dtypes(include=[np.number]).drop(
        columns=[c for c in ["upset", "won_match"] if c in df.columns], errors="ignore"
    )
    corr = num_df.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(corr.columns, fontsize=7)
    plt.colorbar(im, ax=ax, shrink=0.7)
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{path}", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  PART 5 — Model Training & Evaluation")
    print("=" * 60)

    df = load_features(FEATURES_CSV)
    df = df.dropna(subset=[TARGET])

    # Surface dummies discovered from data
    surface_dummy_cols = [c for c in df.columns if c.startswith("surface_")]

    train, test = time_split(df)
    y_train = train[TARGET]
    y_test  = test[TARGET]
    base_rate = y_train.mean()
    print(f"\nTrain: {len(train):,} rows | Test: {len(test):,} rows")
    print(f"Upset base rate: {base_rate:.3f}")

    # Available feature sets
    available_rank  = [c for c in RANK_FEATURES + surface_dummy_cols if c in df.columns]
    available_ctfi  = [c for c in CTFI_FEATURES if c in df.columns]
    available_nlp   = [c for c in NLP_FEATURES  if c in df.columns]
    available_cat   = [c for c in CATEGORICAL_FEATURES if c in df.columns]

    # ── Model A: LR Baseline (rank + CTFI) ──────────────────────────────────
    print("\n[A] LR Baseline (rank + CTFI)")
    cols_a = available_rank + available_ctfi + available_cat
    cols_a = [c for c in cols_a if c in train.columns]
    cat_a  = [c for c in available_cat if c in cols_a]
    pipe_a = make_lr_pipeline(cols_a, cat_a)
    pipe_a = tune(pipe_a, train[cols_a], y_train, LR_GRID, cv=3)
    res_a  = evaluate(pipe_a, test[cols_a], y_test, "LR Baseline")

    # ── Model B: RF — No CTFI (rank only) ───────────────────────────────────
    print("\n[B] RF — No CTFI (rank only)")
    cols_b = available_rank + available_cat
    cols_b = [c for c in cols_b if c in train.columns]
    cat_b  = [c for c in available_cat if c in cols_b]
    pipe_b = make_rf_pipeline(cols_b, cat_b)
    pipe_b = tune(pipe_b, train[cols_b], y_train, RF_GRID, cv=3)
    res_b  = evaluate(pipe_b, test[cols_b], y_test, "RF-No-CTFI")

    # ── Model C: RF — Traditional (rank + CTFI) ──────────────────────────────
    print("\n[C] RF — Traditional (rank + CTFI)")
    cols_c = available_rank + available_ctfi + available_cat
    cols_c = [c for c in cols_c if c in train.columns]
    cat_c  = [c for c in available_cat if c in cols_c]
    pipe_c = make_rf_pipeline(cols_c, cat_c)
    pipe_c = tune(pipe_c, train[cols_c], y_train, RF_GRID, cv=3)
    res_c  = evaluate(pipe_c, test[cols_c], y_test, "RF-Traditional")

    # ── Model D: RF — Full (rank + CTFI + NLP) ────────────────────────────────
    print("\n[D] RF — Full (rank + CTFI + NLP)")
    cols_d = available_rank + available_ctfi + available_nlp + available_cat
    cols_d = [c for c in cols_d if c in train.columns]
    cat_d  = [c for c in available_cat if c in cols_d]
    pipe_d = make_rf_pipeline(cols_d, cat_d)
    pipe_d = tune(pipe_d, train[cols_d], y_train, RF_GRID, cv=3)
    res_d  = evaluate(pipe_d, test[cols_d], y_test, "RF-Full")

    all_results = [res_a, res_b, res_c, res_d]

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n── Model comparison ──────────────────────────────────────────────")
    print(f"  {'Model':<22} {'PR-AUC':>8} {'ROC-AUC':>9} {'F1':>7} {'Recall':>8} {'Brier':>8}")
    for r in all_results:
        print(f"  {r['label']:<22} {r['pr_auc']:>8.4f} {r['roc_auc']:>9.4f} "
              f"{r['f1']:>7.4f} {r['recall']:>8.4f} {r['brier']:>8.4f}")
    print("──────────────────────────────────────────────────────────────────")

    # ── McNemar's tests ───────────────────────────────────────────────────────
    print("\n── McNemar's Significance Tests ──────────────────────────────────")
    mn_ctfi = mcnemar_test(y_test, res_b["y_pred"], res_c["y_pred"],
                           "RF-No-CTFI", "RF-Traditional")
    mn_nlp  = mcnemar_test(y_test, res_c["y_pred"], res_d["y_pred"],
                           "RF-Traditional", "RF-Full")
    print("──────────────────────────────────────────────────────────────────")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots …")
    plot_roc_curves(all_results, y_test)
    plot_pr_curves(all_results, y_test, base_rate)
    plot_confusion_matrices(all_results, y_test)
    plot_calibration_curves(all_results, y_test)
    plot_metric_summary(all_results)
    plot_ctfi_by_surface(df)
    plot_correlation_heatmap(df)
    plot_shap(pipe_d, test[cols_d], cols_d, cat_d, "RF-Full")

    # ── Save model package ────────────────────────────────────────────────────
    package = {
        "model_lr":         pipe_a, "cols_lr":   cols_a,
        "model_rf_noct":    pipe_b, "cols_noct":  cols_b,
        "model_rf_trad":    pipe_c, "cols_trad":  cols_c,
        "model_rf_full":    pipe_d, "cols_full":  cols_d,
        "results": {r["label"]: {k: v for k, v in r.items()
                                 if k not in ("y_prob", "y_pred")}
                    for r in all_results},
        "mcnemar_ctfi": mn_ctfi,
        "mcnemar_nlp":  mn_nlp,
        "base_rate":    float(base_rate),
        "traditional_cols": cols_c,   # kept for backward compat
        "full_cols":        cols_d,
    }
    joblib.dump(package, MODEL_OUT)
    print(f"\nModel saved to: {MODEL_OUT}")
    print("\nPart 5 complete! 🎾")


if __name__ == "__main__":
    main()