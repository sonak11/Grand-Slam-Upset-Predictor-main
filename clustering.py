"""
PART 6 — Unsupervised Analysis (Player Style Clustering)
"

Pipeline:
  1. Build a player feature vector from per-player aggregates
     (mean rank, surface preference, average sets played, average
     CTFI absorbed before losing).
  2. Standard-scale and reduce to 2D via PCA.
  3. Cluster with k-means (we evaluate k = 2..7 by silhouette score).
  4. Compute per-cluster upset rate to test whether some archetypes
     are systematically more upset-prone.
  5. Visualize: PCA scatter, silhouette curve, per-cluster bar chart.

"""

from __future__ import annotations

import sqlite3
import warnings

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

DB_PATH       = "tennis_upsets.db"
FEATURES_CSV  = "features.csv"
OUT_PATH      = "player_clusters.csv"
PLOTS_DIR     = "."


# step 1: Build player-level aggregates 

def build_player_features() -> pd.DataFrame:
    """
    Aggregate the per-match feature matrix into one row per player.

    Player features chosen for clustering:
      - mean_rank           : average ATP rank across matches
      - n_matches           : number of matches in the dataset
      - upset_rate          : empirical upset rate (player's perspective)
      - mean_ctfi_minutes   : average court time accumulated before
                              their matches
      - pct_clay/grass/hard : surface distribution
      - mean_log_rank_diff  : average rank advantage they faced
    """
    try:
        df = pd.read_csv(FEATURES_CSV)
    except FileNotFoundError:
        print(f"[ERROR] {FEATURES_CSV} not found. Run features.py first.")
        return pd.DataFrame()

    # pull player_id from the database join - features.csv was anonymised
    # at the player-match level, so we re-attach it here.
    try:
        conn = sqlite3.connect(DB_PATH)
        joined = pd.read_sql("""
            SELECT player_id, player_name FROM (
                SELECT winner_id AS player_id, winner_name AS player_name FROM matches
                UNION
                SELECT loser_id  AS player_id, loser_name  AS player_name FROM matches
            )
        """, conn)
        conn.close()
    except Exception:
        joined = pd.DataFrame()

    if "player_id" not in df.columns:
        # Fall back: if features.csv lost the player_id, cluster by row groups
        print("[WARN] player_id column not in features.csv — synthesising row-level groups.")
        df["player_id"] = np.arange(len(df))

    grp = df.groupby("player_id").agg(
        mean_rank          = ("rank",          "mean"),
        n_matches          = ("rank",          "count"),
        upset_rate         = ("upset",         "mean"),
        mean_ctfi_minutes  = ("ctfi_minutes",  "mean") if "ctfi_minutes" in df.columns
                                                         else ("ctfi", "mean"),
        mean_log_rank_diff = ("log_rank_diff", "mean"),
        pct_clay           = ("surface_Clay",  "mean") if "surface_Clay"  in df.columns else ("rank", lambda x: 0),
        pct_grass          = ("surface_Grass", "mean") if "surface_Grass" in df.columns else ("rank", lambda x: 0),
        pct_hard           = ("surface_Hard",  "mean") if "surface_Hard"  in df.columns else ("rank", lambda x: 0),
    ).reset_index()

    # filter to players with enough matches for stable aggregates
    grp = grp[grp["n_matches"] >= 5].copy()

    if not joined.empty:
        grp = grp.merge(joined.drop_duplicates("player_id"),
                        on="player_id", how="left")

    print(f"Players with ≥5 matches: {len(grp):,}")
    return grp


#step 2: Standardise & PCA 

CLUSTER_FEATURES = [
    "mean_rank", "n_matches", "upset_rate",
    "mean_ctfi_minutes", "mean_log_rank_diff",
    "pct_clay", "pct_grass", "pct_hard",
]


def reduce_dimensions(df: pd.DataFrame):
    X = df[CLUSTER_FEATURES].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_scaled)

    print(f"PCA explained variance: PC1={pca.explained_variance_ratio_[0]:.2%}  "
          f"PC2={pca.explained_variance_ratio_[1]:.2%}  "
          f"cumulative={pca.explained_variance_ratio_.sum():.2%}")
    return X_scaled, X_2d, pca


# step 3: Choose k by silhouette 

def choose_k(X_scaled, k_range=range(2, 8)) -> dict:
    """Try k = 2..7 and pick the k with the highest silhouette score."""
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        s = silhouette_score(X_scaled, labels)
        scores[k] = s
        print(f"  k={k}  silhouette={s:.4f}")
    best_k = max(scores, key=scores.get)
    print(f"Best k = {best_k}")
    return {"scores": scores, "best_k": best_k}


# step 4: Final clustering 

def cluster(X_scaled, k: int) -> np.ndarray:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    return labels, km


# step 5: Plots 

def plot_silhouette_curve(scores: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ks = sorted(scores.keys())
    vals = [scores[k] for k in ks]
    ax.plot(ks, vals, "o-", color="#222", lw=2)
    best_k = max(scores, key=scores.get)
    ax.axvline(best_k, color="#888", linestyle="--", lw=1,
               label=f"best k = {best_k}")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score")
    ax.set_title("Choosing k by Silhouette Score")
    ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/silhouette_curve.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("  Saved → silhouette_curve.png")


def plot_pca_scatter(X_2d: np.ndarray, labels: np.ndarray,
                     df: pd.DataFrame) -> None:
    """PCA scatter coloured by cluster, with per-cluster centroids labelled."""
    fig, ax = plt.subplots(figsize=(9, 7))
    k = len(np.unique(labels))
    # Greyscale palette to stay black-and-white-friendly
    grays = ["#111", "#444", "#777", "#aaa", "#cccccc"][:k]
    markers = ["o", "s", "^", "D", "v", "P"][:k]

    for c in range(k):
        m = labels == c
        ax.scatter(X_2d[m, 0], X_2d[m, 1],
                   s=30, c=grays[c], marker=markers[c],
                   edgecolors="white", linewidth=0.5,
                   label=f"Cluster {c}  (n={m.sum()})", alpha=0.85)
        # Centroid
        ax.scatter(X_2d[m, 0].mean(), X_2d[m, 1].mean(),
                   s=300, c=grays[c], marker=markers[c],
                   edgecolors="black", linewidth=2.0)

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(f"Player Archetypes — k-means on PCA-reduced features (k={k})")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/cluster_pca_scatter.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("  Saved → cluster_pca_scatter.png")


def plot_cluster_upset_rates(df: pd.DataFrame) -> None:
    """Per-cluster upset rate bar chart — the substantive finding."""
    g = (df.groupby("cluster")
            .agg(rate=("upset_rate", "mean"),
                 n=("upset_rate", "count"),
                 mean_rank=("mean_rank", "mean"),
                 mean_ctfi=("mean_ctfi_minutes", "mean"))
            .reset_index()
            .sort_values("rate"))

    fig, ax = plt.subplots(figsize=(9, 5))
    grays = ["#222", "#555", "#888", "#aaa", "#cccccc"][:len(g)]
    bars = ax.bar(g["cluster"].astype(str), g["rate"] * 100,
                  color=grays, edgecolor="black", linewidth=0.5)
    for bar, (_, row) in zip(bars, g.iterrows()):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f"n={int(row['n'])}\nrank≈{row['mean_rank']:.0f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Mean upset rate (%)")
    ax.set_title("Upset Rate by Player Archetype")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/cluster_upset_rates.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("  Saved → cluster_upset_rates.png")


def describe_clusters(df: pd.DataFrame) -> None:
    """Print per-cluster summary statistics."""
    print("\n" + "="*60)
    print("  PLAYER ARCHETYPE PROFILES")
    print("="*60)
    summary = df.groupby("cluster")[CLUSTER_FEATURES + ["upset_rate"]].mean().round(2)
    print(summary.to_string())
    print()

    # sample 3 representative players per cluster
    if "player_name" in df.columns:
        print("\n  Sample players per cluster:")
        for c, sub in df.groupby("cluster"):
            sample = sub.sample(min(3, len(sub)), random_state=42)
            names = sample["player_name"].fillna("(unknown)").tolist()
            print(f"    Cluster {c}: {', '.join(names)}")


# main 

def main() -> None:
    print("=" * 60)
    print("  PART 6 — Unsupervised Player Archetype Clustering")
    print("=" * 60)

    df = build_player_features()
    if df.empty:
        return

    X_scaled, X_2d, pca = reduce_dimensions(df)

    print("\nSearching for optimal k …")
    sweep = choose_k(X_scaled)
    plot_silhouette_curve(sweep["scores"])

    print(f"\nFinal clustering with k={sweep['best_k']} …")
    labels, km = cluster(X_scaled, sweep["best_k"])
    df["cluster"] = labels

    plot_pca_scatter(X_2d, labels, df)
    plot_cluster_upset_rates(df)
    describe_clusters(df)

    df.to_csv(OUT_PATH, index=False)
    print(f"\nPlayer cluster assignments → {OUT_PATH}")
    print("Part 6 complete.")


if __name__ == "__main__":
    main()