from __future__ import annotations  # must be line 1
import os, warnings
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"
warnings.filterwarnings("ignore")

import sqlite3
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

DB_PATH      = "tennis_upsets.db"
FEATURES_CSV = "features.csv"
OUT_PATH     = "player_clusters.csv"
PLOTS_DIR    = "."


# ── Step 1: Build player-level aggregates ────────────────────────────────────

def build_player_features() -> pd.DataFrame:
    try:
        df = pd.read_csv(FEATURES_CSV)
    except FileNotFoundError:
        print(f"[ERROR] {FEATURES_CSV} not found. Run features.py first.")
        return pd.DataFrame()

    ctfi_col = "ctfi_minutes" if "ctfi_minutes" in df.columns else "ctfi"

    # Resolve player identity
    if "player_id" in df.columns:
        id_col = "player_id"
    elif "player_name" in df.columns:
        id_col = "player_name"
    else:
        # Try joining from DB via positional index (last resort)
        print("[WARN] No player identifier in features.csv — attempting DB join.")
        try:
            conn = sqlite3.connect(DB_PATH)
            db_players = pd.read_sql("""
                SELECT winner_id AS player_id, winner_name AS player_name FROM matches
                UNION
                SELECT loser_id,  loser_name  FROM matches
            """, conn)
            conn.close()
            df = df.merge(db_players, left_index=True, right_index=True, how="left")
            id_col = "player_id" if "player_id" in df.columns else "player_name"
        except Exception as e:
            print(f"[ERROR] DB join failed: {e}")
            return pd.DataFrame()

    # Surface columns
    pct_clay  = df.get("surface_Clay",  pd.Series(0, index=df.index))
    pct_grass = df.get("surface_Grass", pd.Series(0, index=df.index))
    pct_hard  = df.get("surface_Hard",  pd.Series(0, index=df.index))

    agg = df.groupby(id_col).agg(
        mean_rank        = ("rank",     "mean"),
        n_matches        = ("rank",     "count"),
        upset_rate       = ("upset",    "mean"),
        mean_ctfi        = (ctfi_col,   "mean"),
        mean_log_rank_diff=("log_rank_diff", "mean"),
        pct_clay         = ("surface_Clay",  "mean") if "surface_Clay"  in df.columns else ("rank", lambda x: 0),
        pct_grass        = ("surface_Grass", "mean") if "surface_Grass" in df.columns else ("rank", lambda x: 0),
        pct_hard         = ("surface_Hard",  "mean") if "surface_Hard"  in df.columns else ("rank", lambda x: 0),
    ).reset_index()

    # Keep only players with ≥5 matches
    agg = agg[agg["n_matches"] >= 5].copy()
    print(f"Players with ≥5 matches: {len(agg):,}")
    return agg


# ── Step 2: Silhouette-optimal K ─────────────────────────────────────────────

def find_best_k(X_scaled: np.ndarray, k_range=(2, 8)) -> int:
    scores = {}
    for k in range(*k_range):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        if len(np.unique(labels)) < 2:
            continue
        scores[k] = silhouette_score(X_scaled, labels)
    if not scores:
        return 4
    best_k = max(scores, key=scores.get)
    print(f"Best k = {best_k} (silhouette = {scores[best_k]:.3f})")
    return best_k, scores


# ── Step 3: Plots ─────────────────────────────────────────────────────────────

PALETTE = ["#2563eb", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899"]


def plot_silhouette_curve(scores: dict, best_k: int, path="silhouette_curve.png"):
    ks = sorted(scores.keys())
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, [scores[k] for k in ks], "o-", color="#2563eb", lw=2)
    ax.axvline(best_k, color="#ef4444", ls="--", lw=1.5, label=f"Best k={best_k}")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Score vs k (Player Clustering)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{path}", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def plot_pca_scatter(pca2, labels, agg_df, path="cluster_pca_scatter.png"):
    fig, ax = plt.subplots(figsize=(8, 6))
    for k in sorted(np.unique(labels)):
        mask = labels == k
        ax.scatter(pca2[mask, 0], pca2[mask, 1],
                   c=PALETTE[k % len(PALETTE)], label=f"Cluster {k}",
                   alpha=0.65, s=30, edgecolors="none")
    ax.set_xlabel("PCA Component 1"); ax.set_ylabel("PCA Component 2")
    ax.set_title("Player Archetypes — PCA Projection")
    ax.legend(fontsize=9); ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{path}", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def plot_cluster_upset_rates(agg_df: pd.DataFrame, path="cluster_upset_rates.png"):
    stats = agg_df.groupby("cluster").agg(
        upset_rate=("upset_rate", "mean"),
        mean_ctfi=("mean_ctfi", "mean"),
        mean_rank=("mean_rank", "mean"),
        n=("upset_rate", "count"),
    ).reset_index().sort_values("cluster")

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(
        [f"Cluster {c}" for c in stats["cluster"]],
        stats["upset_rate"] * 100,
        color=[PALETTE[c % len(PALETTE)] for c in stats["cluster"]],
        edgecolor="none",
    )
    for bar, row in zip(bars, stats.itertuples()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"n={row.n}\nCTFI={row.mean_ctfi:.0f}\nRank={row.mean_rank:.0f}",
                ha="center", va="bottom", fontsize=8, color="#333")
    ax.set_ylabel("Upset Rate (%)"); ax.set_ylim(0, max(stats["upset_rate"]) * 120)
    ax.set_title("Upset Rate by Player Archetype Cluster")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{path}", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  PART 6 — Unsupervised Player Archetype Clustering")
    print("=" * 60)

    agg = build_player_features()
    if agg.empty or len(agg) < 10:
        print("[WARN] Insufficient data for clustering. Run features.py first.")
        return

    feature_cols = ["mean_rank", "n_matches", "mean_ctfi",
                    "mean_log_rank_diff", "pct_clay", "pct_grass", "pct_hard"]
    feature_cols = [c for c in feature_cols if c in agg.columns]

    X = agg[feature_cols].fillna(agg[feature_cols].median()).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_k, sil_scores = find_best_k(X_scaled)

    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    agg["cluster"] = km.fit_predict(X_scaled)

    # PCA for visualization
    pca  = PCA(n_components=2, random_state=42)
    pca2 = pca.fit_transform(X_scaled)

    # Print summary
    print("\n── Cluster Summary ───────────────────────────────────────────────")
    print(f"  {'Cluster':>8} {'N':>6} {'Mean Rank':>10} {'Mean CTFI':>12} {'Upset Rate':>12}")
    for _, row in (agg.groupby("cluster").agg(
        n=("upset_rate","count"),
        mean_rank=("mean_rank","mean"),
        mean_ctfi=("mean_ctfi","mean"),
        upset_rate=("upset_rate","mean")
    ).reset_index().iterrows()):
        print(f"  {int(row['cluster']):>8} {int(row['n']):>6} {row['mean_rank']:>10.1f} "
              f"{row['mean_ctfi']:>12.1f} {row['upset_rate']*100:>11.1f}%")
    print("──────────────────────────────────────────────────────────────────")

    # Save outputs
    agg.to_csv(OUT_PATH, index=False)
    print(f"\nClusters saved to: {OUT_PATH}")

    # Plots
    print("\nGenerating clustering plots …")
    plot_silhouette_curve(sil_scores, best_k)
    plot_pca_scatter(pca2, agg["cluster"].values, agg)
    plot_cluster_upset_rates(agg)

    print("\nPart 6 complete.")


if __name__ == "__main__":
    main()