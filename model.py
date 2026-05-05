"""
PART 5 — Model Training & Evaluation  

Trains classifiers to predict Grand Slam upsets.

Four model variants are trained:
  LR    —  Logistic Regression baseline (rank + CTFI)
  RF-A  — Random Forest, NO CTFI            ("rank only")
  RF-B  — Random Forest, rank + CTFI        ("traditional", primary model)
  RF-C  — Random Forest, rank + CTFI + NLP  ("full")

Methodology (per proposal):
  - SMOTE applied INSIDE each CV fold (imblearn Pipeline) — no leakage
  - 5-fold stratified CV for hyper-parameter tuning
  - Temporal train/test split (oldest 80% / newest 20%) for the final
    held-out evaluation — prevents future-data leakage (proposal Phase 4)
  - Feature ablation: RF-A vs RF-B isolates CTFI's contribution;
    RF-B vs RF-C isolates NLP's contribution
  - McNemar's test reports the statistical significance of the
    CTFI ablation (proposal Phase 4 explicitly asks for this)
  - Metrics: ROC-AUC · PR-AUC · F1 · Recall · Brier Score

Plots produced (proposal Phase 5):
  - ROC curves               (all variants on one axes)
  - Precision-Recall curves  (all variants on one axes, with no-skill)
  - Confusion matrices       (one panel per variant)
  - Correlation heatmap
  - CTFI vs upset-rate scatter, faceted by surface
  - Upset rate by NLP fatigue quartile
  - Calibration curve        (reliability diagram, all variants)
  - Metric summary bar chart
  - SHAP feature importances (RF-Full)


"""

from __future__ import annotations
import warnings

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# imbalanced-learn (SMOTE)
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("[WARN] imbalanced-learn not installed — SMOTE will be skipped.")

# SHAP (
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[WARN] shap not installed — SHAP plot will be skipped.")

warnings.filterwarnings("ignore", category=UserWarning)

# paths 
FEATURES_CSV = "features.csv"
MODEL_OUT    = "upset_model.pkl"
PLOTS_DIR    = "."

# feature groups 

# Use ctfi_minutes as the canonical CTFI; ctfi_sets is the auxiliary fallback.
RANK_ONLY_FEATURES = [
    "rank", "opp_rank", "rank_ratio", "log_rank_diff",
    "is_underdog", "round_num",
]

CTFI_FEATURES = ["ctfi_minutes", "ctfi_sets", "ctfi_diff_minutes", "ctfi_diff_sets"]

TRADITIONAL_FEATURES = RANK_ONLY_FEATURES + CTFI_FEATURES

NLP_FEATURES = [
    "sentiment_polarity",
    "fatigue_total", "fatigue_word_density",
    "fatigue_physical", "fatigue_mental", "fatigue_schedule",
    "fatigue_injury", "fatigue_motivation",
    "first_person_rate", "negation_rate",
    "llm_is_fatigued",
]

CATEGORICAL_FEATURES = ["rank_bin"]
TARGET = "upset"


# backward-compat shim for the old "ctfi" column name 
def _ensure_ctfi_columns(df: pd.DataFrame) -> pd.DataFrame:
    """If the legacy `ctfi` column exists but new ones don't, alias them."""
    df = df.copy()
    if "ctfi" in df.columns and "ctfi_minutes" not in df.columns:
        df["ctfi_minutes"] = df["ctfi"]
        df["ctfi_sets"]    = df["ctfi"]
        df["ctfi_diff_minutes"] = 0.0
        df["ctfi_diff_sets"]    = 0.0
    return df


# data loading 

def load_features(path: str = FEATURES_CSV) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        print(f"Loaded {len(df):,} rows, {len(df.columns)} columns from {path}")
    except FileNotFoundError:
        print(f"[WARN] {path} not found — generating synthetic data.")
        df = _generate_synthetic_data()
    return _ensure_ctfi_columns(df)


def _generate_synthetic_data(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Synthetic dataset with realistic structure for sanity testing."""
    rng = np.random.default_rng(seed)
    rank = rng.integers(1, 200, n);  opp_rank = rng.integers(1, 200, n)
    rank_ratio    = rank / opp_rank
    log_rank_diff = np.sign(rank - opp_rank) * np.log1p(np.abs(rank - opp_rank))
    is_underdog   = (rank > opp_rank).astype(int)
    round_num = rng.integers(1, 8, n);  best_of = np.full(n, 5)

    ctfi_min  = rng.integers(0, 700, n).astype(float)
    ctfi_sets = (ctfi_min / 50).astype(int)
    ctfi_diff_min  = rng.integers(-500, 500, n).astype(float)
    ctfi_diff_sets = (ctfi_diff_min / 50).astype(int)

    has_t = rng.random(n) < 0.45
    nan = np.where(has_t, 0, np.nan)

    log_odds = (
        0.6 * (rank / 100) - 0.9 * (opp_rank / 100) - 0.1 * log_rank_diff
        + 0.001 * ctfi_min + 0.0008 * ctfi_diff_min
        + rng.normal(0, 0.4, n)
    )
    upset = (rng.random(n) < 1 / (1 + np.exp(-log_odds))).astype(int)

    rank_bin = pd.cut(rank, bins=[0, 10, 30, 100, np.inf],
                      labels=["top10", "top30", "top100", "outside100"])

    return pd.DataFrame({
        "rank": rank, "opp_rank": opp_rank, "rank_ratio": rank_ratio,
        "log_rank_diff": log_rank_diff, "is_underdog": is_underdog,
        "round_num": round_num, "best_of": best_of,
        "ctfi_minutes": ctfi_min, "ctfi_sets": ctfi_sets,
        "ctfi_diff_minutes": ctfi_diff_min, "ctfi_diff_sets": ctfi_diff_sets,
        "sentiment_polarity": np.where(has_t, rng.uniform(-1, 1, n), np.nan),
        "fatigue_total": np.where(has_t, rng.integers(0, 12, n).astype(float), np.nan),
        "fatigue_word_density": np.where(has_t, rng.uniform(0, 3, n), np.nan),
        "fatigue_physical":  np.where(has_t, rng.integers(0, 5, n).astype(float), np.nan),
        "fatigue_mental":    np.where(has_t, rng.integers(0, 3, n).astype(float), np.nan),
        "fatigue_schedule":  np.where(has_t, rng.integers(0, 4, n).astype(float), np.nan),
        "fatigue_injury":    np.where(has_t, rng.integers(0, 3, n).astype(float), np.nan),
        "fatigue_motivation":np.where(has_t, rng.integers(0, 2, n).astype(float), np.nan),
        "first_person_rate": np.where(has_t, rng.uniform(0.05, 0.25, n), np.nan),
        "negation_rate":     np.where(has_t, rng.uniform(0.01, 0.10, n), np.nan),
        "llm_is_fatigued":   np.where(has_t, rng.choice([0, 1], n, p=[0.7, 0.3]).astype(float), np.nan),
        "rank_bin": rank_bin, "upset": upset,
        "surface": rng.choice(["Hard", "Clay", "Grass"], n, p=[0.5, 0.3, 0.2]),
    })


# preprocessing 

def build_preprocessor(feature_cols: list, categorical_cols: list) -> ColumnTransformer:
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]
    numeric_t = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    cat_t = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    steps = [("num", numeric_t, numeric_cols)]
    if categorical_cols:
        steps.append(("cat", cat_t, categorical_cols))
    return ColumnTransformer(steps, remainder="drop")


def _make_pipeline(estimator, feature_cols, categorical_cols):
    prep = build_preprocessor(feature_cols, categorical_cols)
    if IMBLEARN_AVAILABLE:
        return ImbPipeline([
            ("prep",  prep),
            ("smote", SMOTE(random_state=42, k_neighbors=5)),
            ("model", estimator),
        ])
    return Pipeline([("prep", prep), ("model", estimator)])


def build_rf_pipeline(feature_cols, categorical_cols):
    rf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    return _make_pipeline(rf, feature_cols, categorical_cols)


def build_lr_pipeline(feature_cols, categorical_cols):
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    return _make_pipeline(lr, feature_cols, categorical_cols)


# train/test split 

def time_split(df: pd.DataFrame, test_frac: float = 0.20):
    """Temporal holdout — oldest (1-frac) for train, newest frac for test."""
    n_test = int(len(df) * test_frac)
    train  = df.iloc[:-n_test].copy()
    test   = df.iloc[-n_test:].copy()
    print(f"Train: {len(train):,} rows | Test: {len(test):,} rows")
    return train, test


# hyperparameter search 

RF_PARAM_GRID = {
    "model__n_estimators":     [200, 400],
    "model__max_depth":        [None, 8, 15],
    "model__min_samples_leaf": [3, 7],
    "model__max_features":     ["sqrt", 0.5],
}

LR_PARAM_GRID = {
    "model__C":       [0.01, 0.1, 1.0, 10.0],
    "model__solver":  ["lbfgs", "saga"],
    "model__max_iter":[1000],
}


def tune_model(pipeline, X_train, y_train, param_grid, label="Model"):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(pipeline, param_grid, cv=skf,
                      scoring="roc_auc", n_jobs=-1, verbose=0, refit=True)
    gs.fit(X_train, y_train)
    print(f"  Best 5-fold CV ROC-AUC [{label}] : {gs.best_score_:.4f}")
    print(f"  Best params              : {gs.best_params_}")
    return gs.best_estimator_


# evaluation 

def evaluate(model, X_test, y_test, label="Model") -> dict:
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    auc   = roc_auc_score(y_test, y_prob)
    ap    = average_precision_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)

    print(f"\n{'='*55}\n  {label}\n{'='*55}")
    print(f"  ROC-AUC              : {auc:.4f}")
    print(f"  Precision-Recall AUC : {ap:.4f}")
    print(f"  Brier Score          : {brier:.4f}  (lower=better)")
    print("\n  Classification report (threshold=0.5):")
    print(classification_report(y_test, y_pred, target_names=["No upset", "Upset"]))

    return {
        "label": label, "roc_auc": auc, "avg_precision": ap, "brier": brier,
        "y_prob": y_prob, "y_pred": y_pred,
    }


# mcNemar's test for paired classifier comparison 

def mcnemar_test(y_true, y_pred_a, y_pred_b) -> dict:
    """
    McNemar's test: tests whether classifier A and B have the same error
    distribution on the held-out test set. Used to assess significance of
    the CTFI ablation uplift.
    """
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    n01 = int(((~correct_a) & ( correct_b)).sum())   # A wrong, B right
    n10 = int(((  correct_a) & (~correct_b)).sum())  # A right, B wrong

    if n01 + n10 == 0:
        return {"n01": 0, "n10": 0, "stat": 0.0, "p_value": 1.0}

    # continuity-corrected chi-squared with 1 df
    stat = (abs(n01 - n10) - 1.0) ** 2 / (n01 + n10)
    # survival function of chi^2 with 1 df at stat
    from math import erfc, sqrt
    p_value = erfc(sqrt(stat / 2.0))   # exact for chi^2_1
    return {"n01": n01, "n10": n10, "stat": stat, "p_value": p_value}


# plots 

def _save(name: str) -> None:
    out = f"{PLOTS_DIR}/{name}"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


def plot_roc_curves(results, y_test) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#5C5CE0", "#E05C5C", "#5CE0A0", "#E0B85C"]
    for res, c in zip(results, colors):
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        ax.plot(fpr, tpr, lw=2, color=c,
                label=f"{res['label']}  (AUC={res['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (0.500)")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Model Variants")
    ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout(); _save("roc_curves_all.png")


def plot_precision_recall_curves(results, y_test) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    baseline = y_test.mean()
    ax.axhline(baseline, color="k", linestyle="--", lw=1,
               label=f"No-skill ({baseline:.2f})")
    colors = ["#5C5CE0", "#E05C5C", "#5CE0A0", "#E0B85C"]
    for res, c in zip(results, colors):
        precision, recall, _ = precision_recall_curve(y_test, res["y_prob"])
        ax.plot(recall, precision, lw=2, color=c,
                label=f"{res['label']}  (AP={res['avg_precision']:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — All Model Variants")
    ax.legend(loc="upper right", fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout(); _save("precision_recall_curves.png")


def plot_confusion_matrices(results, y_test) -> None:
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1: axes = [axes]
    for ax, res in zip(axes, results):
        cm = confusion_matrix(y_test, res["y_pred"])
        ConfusionMatrixDisplay(cm, display_labels=["No upset", "Upset"]).plot(
            ax=ax, colorbar=False)
        ax.set_title(res["label"], fontsize=10)
    plt.suptitle("Confusion Matrices — All Model Variants", fontsize=13, y=1.02)
    plt.tight_layout(); _save("confusion_matrices.png")


def plot_calibration_curves(results, y_test) -> None:
    """Reliability diagram — checks whether predicted probabilities are calibrated."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfectly calibrated")
    colors = ["#5C5CE0", "#E05C5C", "#5CE0A0", "#E0B85C"]
    for res, c in zip(results, colors):
        try:
            frac_pos, mean_pred = calibration_curve(
                y_test, res["y_prob"], n_bins=10, strategy="quantile",
            )
            ax.plot(mean_pred, frac_pos, "o-", color=c,
                    label=f"{res['label']}  (Brier={res['brier']:.3f})")
        except Exception:
            continue
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Fraction of upsets")
    ax.set_title("Calibration Curves — All Model Variants")
    ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout(); _save("calibration_curves.png")


def plot_correlation_heatmap(df) -> None:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    drop_cols = [c for c in numeric_cols if c.startswith("surface_")]
    plot_cols = [c for c in numeric_cols if c not in drop_cols]
    corr = df[plot_cols].corr()

    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                annot=True, fmt=".2f", annot_kws={"size": 7},
                linewidths=0.4, ax=ax, square=True)
    ax.set_title("Feature Correlation Heatmap", pad=14)
    plt.tight_layout(); _save("correlation_heatmap.png")


def plot_ctfi_by_surface(df) -> None:
    ctfi_col = "ctfi_minutes" if "ctfi_minutes" in df.columns else "ctfi"
    if ctfi_col not in df.columns or "upset" not in df.columns:
        return

    surfaces = ["Hard", "Clay", "Grass"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    colors = {"Hard": "#5C5CE0", "Clay": "#C96A2B", "Grass": "#2E8B57"}

    for ax, surface in zip(axes, surfaces):
        col = f"surface_{surface}"
        if col in df.columns:
            sub = df[df[col] == 1].copy()
        elif "surface" in df.columns:
            sub = df[df["surface"] == surface].copy()
        else:
            ax.set_title(f"{surface}\n(no data)"); continue
        if len(sub) < 20:
            ax.set_title(f"{surface}\n(insufficient data)"); continue

        try:
            sub["bin"] = pd.qcut(sub[ctfi_col], q=10, duplicates="drop")
        except ValueError:
            sub["bin"] = pd.cut(sub[ctfi_col], bins=5, duplicates="drop")

        g = (sub.groupby("bin", observed=True)["upset"]
                .agg(rate="mean", n="count").reset_index())
        g["mid"] = g["bin"].apply(
            lambda x: (x.left + x.right) / 2 if hasattr(x, "left") else x)

        ax.scatter(g["mid"], g["rate"] * 100,
                   s=g["n"] / g["n"].max() * 200 + 30,
                   color=colors.get(surface, "#555"),
                   alpha=0.75, edgecolors="white", linewidth=0.5)

        if len(g) >= 3:
            z = np.polyfit(g["mid"].astype(float), g["rate"] * 100, 1)
            xs = np.linspace(g["mid"].min(), g["mid"].max(), 100)
            ax.plot(xs, np.poly1d(z)(xs), "--",
                    color=colors.get(surface, "#555"), lw=1.5, alpha=0.6)

        ax.set_title(f"{surface}  (n={len(sub):,})")
        ax.set_xlabel(f"CTFI ({'minutes' if ctfi_col=='ctfi_minutes' else 'sets'})")
        if ax == axes[0]: ax.set_ylabel("Upset rate (%)")
        ax.grid(alpha=0.3)

    plt.suptitle("CTFI vs Upset Rate by Surface  (bubble size ∝ N)",
                 fontsize=12, y=1.02)
    plt.tight_layout(); _save("ctfi_upset_by_surface.png")


def plot_upset_rate_by_fatigue(df) -> None:
    if "fatigue_total" not in df.columns or df["fatigue_total"].isna().all():
        return
    sub = df[df["fatigue_total"].notna()].copy()
    if sub["fatigue_total"].nunique() < 4:
        return
    sub["q"] = pd.qcut(sub["fatigue_total"], q=4,
                       labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"],
                       duplicates="drop")
    g = (sub.groupby("q", observed=True)["upset"]
            .agg(["mean", "count"]).rename(columns={"mean":"rate","count":"n"}))

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(g.index, g["rate"] * 100,
                  color=["#9FE1CB", "#5DCAA5", "#1D9E75", "#0F6E56"])
    for bar, (_, row) in zip(bars, g.iterrows()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                f"n={row['n']:.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("NLP Fatigue Keyword Quartile")
    ax.set_ylabel("Upset rate (%)")
    ax.set_title("Upset Rate by Pre-Match Fatigue Language Level")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); _save("upset_rate_by_fatigue.png")


def plot_shap_importance(model, X_test, label="Model") -> None:
    if not SHAP_AVAILABLE: return
    try:
        prep = model.named_steps["prep"]
        rf   = model.named_steps["model"]
        Xt   = prep.transform(X_test)
        explainer = shap.TreeExplainer(rf)
        sv = explainer.shap_values(Xt)
        sv = sv[1] if isinstance(sv, list) else sv

        num_names = list(prep.transformers_[0][2])
        cat_names = list(prep.transformers_[1][2]) if len(prep.transformers_) > 1 else []
        names = num_names + cat_names

        mean_abs = np.abs(sv).mean(axis=0)
        if mean_abs.ndim > 1:
            mean_abs = mean_abs.mean(axis=0)
        order = np.argsort(mean_abs)[::-1][:20]

        fig, ax = plt.subplots(figsize=(9, 7))
        ax.barh([names[int(i)] if int(i) < len(names) else f"f{int(i)}"
                 for i in order[::-1]],
                mean_abs[order[::-1]], color="#5C5CE0")
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"SHAP Feature Importance — {label}")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout(); _save("shap_importance.png")
    except Exception as e:
        print(f"[WARN] SHAP plot skipped: {e}")


def plot_metric_summary(results) -> None:
    labels = [r["label"] for r in results]
    roc    = [r["roc_auc"]       for r in results]
    pr     = [r["avg_precision"] for r in results]
    brier  = [r["brier"]         for r in results]

    x, w = np.arange(len(labels)), 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w, roc,   w, label="ROC-AUC",         color="#5C5CE0")
    ax.bar(x,     pr,    w, label="PR-AUC",          color="#5CE0A0")
    ax.bar(x + w, brier, w, label="Brier (↓)",       color="#E05C5C")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1); ax.set_ylabel("Score")
    ax.set_title("Model Comparison — ROC-AUC · PR-AUC · Brier")
    ax.legend(); ax.axhline(0.5, color="k", linestyle="--", lw=0.8, alpha=0.5)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); _save("metric_summary.png")


# main 

def main() -> None:
    print("=" * 60)
    print("  PART 5 — Model Training & Evaluation  (FIXED)")
    print("=" * 60)
    print(f"  SMOTE          : {'ENABLED' if IMBLEARN_AVAILABLE else 'DISABLED'}")
    print(f"  CV strategy    : 5-fold StratifiedKFold")
    print(f"  Class balance  : class_weight='balanced' + SMOTE")
    print("=" * 60)

    df = load_features(FEATURES_CSV)
    df = df.dropna(subset=[TARGET])

    train, test = time_split(df)
    y_train, y_test = train[TARGET], test[TARGET]
    print(f"\n  Upset rate — train: {y_train.mean()*100:.1f}%  | "
          f" test: {y_test.mean()*100:.1f}%")

    all_results = []

    # 1. LR baseline
    print("\n" + "─"*55)
    print("[1/4] Logistic Regression — interpretable baseline")
    print("─"*55)
    trad_cols  = [c for c in TRADITIONAL_FEATURES if c in df.columns]
    cat_cols_t = [c for c in CATEGORICAL_FEATURES if c in trad_cols]
    pipe_lr = build_lr_pipeline(trad_cols, cat_cols_t)
    pipe_lr = tune_model(pipe_lr, train[trad_cols], y_train,
                         LR_PARAM_GRID, label="LR")
    res_lr = evaluate(pipe_lr, test[trad_cols], y_test,
                      label="LR Baseline (rank+CTFI)")
    all_results.append(res_lr)

    # 2. RF - no CTFI (ablation baseline)
    print("\n" + "─"*55)
    print("[2/4] Random Forest — NO CTFI  (ablation baseline)")
    print("─"*55)
    rank_cols  = [c for c in RANK_ONLY_FEATURES if c in df.columns]
    cat_cols_r = [c for c in CATEGORICAL_FEATURES if c in rank_cols]
    pipe_noctfi = build_rf_pipeline(rank_cols, cat_cols_r)
    pipe_noctfi = tune_model(pipe_noctfi, train[rank_cols], y_train,
                             RF_PARAM_GRID, label="RF — no CTFI")
    res_noctfi = evaluate(pipe_noctfi, test[rank_cols], y_test,
                          label="RF — no CTFI")
    all_results.append(res_noctfi)

    # 3. RF - Traditional (rank + CTFI)
    print("\n" + "─"*55)
    print("[3/4] Random Forest — Traditional  (rank + CTFI)")
    print("─"*55)
    pipe_trad = build_rf_pipeline(trad_cols, cat_cols_t)
    pipe_trad = tune_model(pipe_trad, train[trad_cols], y_train,
                           RF_PARAM_GRID, label="RF Traditional")
    res_trad = evaluate(pipe_trad, test[trad_cols], y_test,
                        label="RF Traditional (rank+CTFI)")
    all_results.append(res_trad)

    # 4. RF - Full
    print("\n" + "─"*55)
    print("[4/4] Random Forest — Full  (rank + CTFI + NLP)")
    print("─"*55)
    full_cols  = [c for c in TRADITIONAL_FEATURES + NLP_FEATURES if c in df.columns]
    cat_cols_f = [c for c in CATEGORICAL_FEATURES if c in full_cols]
    pipe_full  = build_rf_pipeline(full_cols, cat_cols_f)
    pipe_full  = tune_model(pipe_full, train[full_cols], y_train,
                            RF_PARAM_GRID, label="RF Full")
    res_full   = evaluate(pipe_full, test[full_cols], y_test,
                          label="RF Full (rank+CTFI+NLP)")
    all_results.append(res_full)

    # summary 
    print("\n" + "="*60)
    print("  MODEL COMPARISON SUMMARY")
    print("="*60)
    print(f"  {'Model':<32} {'ROC-AUC':>8} {'PR-AUC':>8} {'Brier':>7}")
    print(f"  {'-'*60}")
    for r in all_results:
        print(f"  {r['label']:<32} {r['roc_auc']:>8.4f} "
              f"{r['avg_precision']:>8.4f} {r['brier']:>7.4f}")

    # CTFI ablation + significance
    print("\n  — CTFI Ablation —")
    delta_ctfi = res_trad["roc_auc"] - res_noctfi["roc_auc"]
    print(f"  ROC-AUC uplift (RF trad − RF no-CTFI) : {delta_ctfi:+.4f}")
    mc_ctfi = mcnemar_test(y_test.values, res_noctfi["y_pred"], res_trad["y_pred"])
    print(f"  McNemar's test : χ²={mc_ctfi['stat']:.3f}, "
          f"p={mc_ctfi['p_value']:.4f}  "
          f"(n01={mc_ctfi['n01']}, n10={mc_ctfi['n10']})")

    # NLP ablation + significance
    print("\n  — NLP Ablation —")
    delta_nlp = res_full["roc_auc"] - res_trad["roc_auc"]
    print(f"  ROC-AUC uplift (RF full − RF trad)     : {delta_nlp:+.4f}")
    mc_nlp = mcnemar_test(y_test.values, res_trad["y_pred"], res_full["y_pred"])
    print(f"  McNemar's test : χ²={mc_nlp['stat']:.3f}, "
          f"p={mc_nlp['p_value']:.4f}  "
          f"(n01={mc_nlp['n01']}, n10={mc_nlp['n10']})")
    print("="*60)

    # plots
    print("\nGenerating plots …")
    plot_roc_curves(all_results, y_test)
    plot_precision_recall_curves(all_results, y_test)
    plot_confusion_matrices(all_results, y_test)
    plot_calibration_curves(all_results, y_test)
    plot_correlation_heatmap(df)
    plot_ctfi_by_surface(df)
    plot_upset_rate_by_fatigue(df)
    plot_metric_summary(all_results)
    plot_shap_importance(pipe_full, test[full_cols], label="RF Full")

    # save model package 
    save_pkg = {
        "model_lr":         pipe_lr,
        "model_a":          pipe_trad,
        "model_b":          pipe_full,
        "model_noctfi":     pipe_noctfi,
        "traditional_cols": trad_cols,
        "full_cols":        full_cols,
        "rank_only_cols":   rank_cols,
        "results_lr":       {k: v for k, v in res_lr.items()    if k not in ("y_prob","y_pred")},
        "results_a":        {k: v for k, v in res_trad.items()  if k not in ("y_prob","y_pred")},
        "results_b":        {k: v for k, v in res_full.items()  if k not in ("y_prob","y_pred")},
        "results_noctfi":   {k: v for k, v in res_noctfi.items()if k not in ("y_prob","y_pred")},
        "mcnemar_ctfi":     mc_ctfi,
        "mcnemar_nlp":      mc_nlp,
    }
    joblib.dump(save_pkg, MODEL_OUT)
    print(f"\nModel package saved → {MODEL_OUT}")

    print("\nPart 5 complete.")


# inference helper 

def predict_upset_probability(
    model_path: str = MODEL_OUT,
    player_rank: int = 45, opponent_rank: int = 12,
    ctfi_minutes: float = 480.0, ctfi_diff_minutes: float = 100.0,
    sentiment_polarity: float = -0.3, fatigue_total: int = 5,
    round_num: int = 4, best_of: int = 5,
) -> float:
    pkg  = joblib.load(model_path)
    pipe = pkg["model_b"]; cols = pkg["full_cols"]

    row = {
        "rank": player_rank, "opp_rank": opponent_rank,
        "rank_ratio": player_rank / max(opponent_rank, 1),
        "log_rank_diff": np.sign(player_rank - opponent_rank) *
                        np.log1p(abs(player_rank - opponent_rank)),
        "is_underdog": int(player_rank > opponent_rank),
        "round_num": round_num, "best_of": best_of,
        "ctfi_minutes": ctfi_minutes,
        "ctfi_sets": ctfi_minutes / 50,
        "ctfi_diff_minutes": ctfi_diff_minutes,
        "ctfi_diff_sets": ctfi_diff_minutes / 50,
        "rank_bin": "top100" if player_rank <= 100 else "outside100",
        "sentiment_polarity": sentiment_polarity,
        "fatigue_total": fatigue_total,
        "fatigue_word_density": fatigue_total / 100,
        "fatigue_physical": max(0, fatigue_total - 2),
        "fatigue_mental":   max(0, fatigue_total - 3),
        "fatigue_schedule": max(0, fatigue_total - 4),
        "fatigue_injury": 0, "fatigue_motivation": 0,
        "first_person_rate": 0.15, "negation_rate": 0.05,
        "llm_is_fatigued": 0.5,
    }
    df = pd.DataFrame([row])[[c for c in cols if c in row]]
    prob = pipe.predict_proba(df)[0, 1]
    print(f"\nUpset probability: {prob:.3f}")
    return prob


if __name__ == "__main__":
    main()