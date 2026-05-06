import os, warnings
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"
warnings.filterwarnings("ignore")

import sqlite3, re
import numpy as np
import pandas as pd

DB_PATH       = "tennis_upsets.db"
FEATURES_OUT  = "features.csv"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


# step 1: Load matches (player-centric) 

def load_matches() -> pd.DataFrame:
    conn = get_connection()
    matches = pd.read_sql("""
        SELECT
            rowid           AS match_db_id,
            winner_id       AS player_id,
            winner_name     AS player_name,
            loser_id        AS opponent_id,
            loser_name      AS opponent_name,
            winner_rank     AS rank,
            loser_rank      AS opp_rank,
            tourney_id, tourney_name, slam_name, tourney_date,
            match_num, round, round_num, surface, best_of,
            minutes, score, upset, rank_diff, tour,
            1               AS won_match
        FROM matches

        UNION ALL

        SELECT
            rowid,
            loser_id,  loser_name,  winner_id, winner_name,
            loser_rank, winner_rank,
            tourney_id, tourney_name, slam_name, tourney_date,
            match_num, round, round_num, surface, best_of,
            minutes, score, upset, -rank_diff, tour,
            0
        FROM matches
        ORDER BY tourney_date, tourney_id, round_num
    """, conn)
    conn.close()

    matches["tourney_date"] = pd.to_datetime(matches["tourney_date"], errors="coerce")
    matches["rank"]         = pd.to_numeric(matches["rank"],     errors="coerce")
    matches["opp_rank"]     = pd.to_numeric(matches["opp_rank"], errors="coerce")
    matches["minutes"]      = pd.to_numeric(matches["minutes"],  errors="coerce")

    print(f"Player-match rows loaded: {len(matches):,}")
    return matches


# step 2: Parse score 

def parse_score(score: str) -> dict:
    if not isinstance(score, str) or not score.strip():
        return {"sets_played": 3, "total_games": 30, "sets_won": 2, "sets_lost": 1}
    sets = re.findall(r"(\d+)-(\d+)", score)
    if not sets:
        return {"sets_played": 3, "total_games": 30, "sets_won": 2, "sets_lost": 1}
    sets_played = len(sets)
    sets_won    = sum(1 for w, l in sets if int(w) > int(l))
    return {
        "sets_played": sets_played,
        "total_games": sum(int(w) + int(l) for w, l in sets),
        "sets_won":    sets_won,
        "sets_lost":   sets_played - sets_won,
    }


def enrich_score_features(df: pd.DataFrame) -> pd.DataFrame:
    parsed = df["score"].apply(parse_score).apply(pd.Series)
    df = pd.concat([df, parsed], axis=1)
    loser_mask = df["won_match"] == 0
    df.loc[loser_mask, ["sets_won", "sets_lost"]] = (
        df.loc[loser_mask, ["sets_lost", "sets_won"]].values
    )
    return df


# step 3: CTFI — minutes-based, sets fallback 
#  CTFI(player, tournament, round) = SUM(minutes played in all prior rounds)
#  Uses actual match duration; falls back to sets_played proxy if minutes=NULL.

CTFI_QUERY = """
WITH player_match AS (
    -- winner perspective
    SELECT
        winner_id   AS player_id,
        tourney_id,
        match_num,
        round_num,
        CAST(COALESCE(minutes, 0) AS REAL) AS match_minutes,
        -- Sets proxy for fallback: count hyphens / 2
        CAST((LENGTH(COALESCE(score,'')) - LENGTH(REPLACE(COALESCE(score,''), '-', ''))) / 2 AS REAL) AS approx_sets
    FROM matches
    UNION ALL
    -- loser perspective (same minutes / sets)
    SELECT
        loser_id,
        tourney_id,
        match_num,
        round_num,
        CAST(COALESCE(minutes, 0) AS REAL),
        CAST((LENGTH(COALESCE(score,'')) - LENGTH(REPLACE(COALESCE(score,''), '-', ''))) / 2 AS REAL)
    FROM matches
),
ctfi_raw AS (
    SELECT
        player_id,
        tourney_id,
        match_num,
        round_num,
        -- Minutes-based CTFI (gold standard)
        SUM(match_minutes) OVER (
            PARTITION BY player_id, tourney_id
            ORDER BY round_num, match_num
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS ctfi_minutes,
        -- Sets-based CTFI (fallback / validation)
        SUM(approx_sets) OVER (
            PARTITION BY player_id, tourney_id
            ORDER BY round_num, match_num
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS ctfi_sets
    FROM player_match
)
SELECT
    player_id,
    tourney_id,
    match_num,
    COALESCE(ctfi_minutes, 0) AS ctfi_minutes,
    COALESCE(ctfi_sets,    0) AS ctfi_sets
FROM ctfi_raw
"""


def compute_ctfi() -> pd.DataFrame:
    conn = get_connection()
    ctfi = pd.read_sql(CTFI_QUERY, conn)
    conn.close()
    print(f"CTFI rows computed: {len(ctfi):,}")
    return ctfi


# step 4: Rank features 

def add_rank_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rank_ratio"]    = df["rank"] / df["opp_rank"].replace(0, np.nan)
    df["log_rank_diff"] = np.sign(df["rank_diff"]) * np.log1p(np.abs(df["rank_diff"]))
    df["is_underdog"]   = (df["rank"] > df["opp_rank"]).astype(int)
    df["rank_bin"]      = pd.cut(
        df["rank"], bins=[0, 10, 30, 100, np.inf],
        labels=["top10", "top30", "top100", "outside100"], right=True,
    )
    return df


# step 5: Surface encoding 

def encode_surface(df: pd.DataFrame) -> pd.DataFrame:
    surface_dummies = pd.get_dummies(df["surface"], prefix="surface", dtype=int)
    return pd.concat([df, surface_dummies], axis=1)


# step 6: NLP merge 

NLP_COLS = [
    "sentiment_polarity", "fatigue_total", "fatigue_word_density",
    "fatigue_physical", "fatigue_mental", "fatigue_schedule",
    "fatigue_injury", "fatigue_motivation",
    "first_person_rate", "negation_rate", "llm_is_fatigued",
]


def load_transcript_features() -> pd.DataFrame:
    conn = get_connection()
    cols_info = conn.execute("PRAGMA table_info(transcripts)").fetchall()
    col_names = {row[1] for row in cols_info}
    if "sentiment_polarity" not in col_names:
        conn.close()
        print("[WARN] NLP columns not found — run nlp.py first.")
        return pd.DataFrame()

    transcript_df = pd.read_sql("""
        SELECT player_id, player_name, tourney_name, tourney_date, round,
               sentiment_polarity, fatigue_total, fatigue_word_density,
               fatigue_physical, fatigue_mental, fatigue_schedule,
               fatigue_injury, fatigue_motivation,
               first_person_rate, negation_rate,
               llm_fatigue_label, llm_fatigue_confidence
        FROM transcripts WHERE nlp_processed = 1
    """, conn)
    conn.close()

    if "llm_fatigue_label" in transcript_df.columns:
        transcript_df["llm_is_fatigued"] = (
            transcript_df["llm_fatigue_label"] == "FATIGUED"
        ).astype(float)

    print(f"Transcript NLP rows loaded: {len(transcript_df):,}")
    return transcript_df


def merge_transcripts(matches: pd.DataFrame, transcripts: pd.DataFrame) -> pd.DataFrame:
    if transcripts.empty:
        print("[INFO] No transcript data — NLP columns will be NaN.")
        return matches

    available = [c for c in NLP_COLS if c in transcripts.columns]
    if not available:
        return matches

    trans_subset = transcripts[["player_name", "tourney_name"] + available].copy()
    trans_subset["player_name"]  = trans_subset["player_name"].str.strip().str.title()
    trans_subset["tourney_name"] = trans_subset["tourney_name"].str.strip()
    matches = matches.copy()
    matches["player_name"]  = matches["player_name"].str.strip().str.title()
    matches["tourney_name"] = matches["tourney_name"].str.strip()

    trans_agg = (
        trans_subset.groupby(["player_name", "tourney_name"], as_index=False)[available].mean()
    )
    merged = matches.merge(trans_agg, on=["player_name", "tourney_name"], how="left",
                           suffixes=("", "_t"))

    # fallback via slam_name
    if "slam_name" in matches.columns:
        unmatched = merged[available[0]].isna()
        if unmatched.sum() > 0:
            ts2 = trans_subset.copy().rename(columns={"tourney_name": "slam_name"})
            ts2_agg = ts2.groupby(["player_name", "slam_name"], as_index=False)[available].mean()
            fallback = matches[unmatched].merge(ts2_agg, on=["player_name", "slam_name"],
                                                how="left", suffixes=("", "_t"))
            for col in available:
                if col in fallback.columns:
                    merged.loc[unmatched, col] = fallback[col].values

    n_matched = merged[available[0]].notna().sum()
    print(f"Transcript features matched: {n_matched:,}/{len(merged):,} rows "
          f"({n_matched / len(merged) * 100:.1f}%)")
    return merged


def impute_nlp_surface_median(df: pd.DataFrame) -> pd.DataFrame:
    """Surface-stratified median imputation for NLP features."""
    available = [c for c in NLP_COLS if c in df.columns]
    if not available or "surface" not in df.columns:
        return df
    df = df.copy()
    for surface in df["surface"].dropna().unique():
        mask = df["surface"] == surface
        for col in available:
            median_val = df.loc[mask & df[col].notna(), col].median()
            if pd.notna(median_val):
                df.loc[mask & df[col].isna(), col] = median_val
    # global fallback for any remaining
    for col in available:
        df[col] = df[col].fillna(df[col].median())
    return df


#  step 7: final assembly 

FEATURE_COLS = [
    # identifiers (kept for clustering, dropped from model features)
    "player_id", "player_name", "tourney_date", "slam_name",
    # match context
    "surface", "round_num", "best_of", "tour",
    # rank features
    "rank", "opp_rank", "rank_ratio", "log_rank_diff", "is_underdog", "rank_bin",
    # CTFI (both variants)
    "ctfi_minutes", "ctfi_sets",
    # NLP features
    "sentiment_polarity", "fatigue_total", "fatigue_word_density",
    "fatigue_physical", "fatigue_mental", "fatigue_schedule",
    "fatigue_injury", "fatigue_motivation",
    "first_person_rate", "negation_rate", "llm_is_fatigued",
    # target
    "upset",
]


def build_final_features(matches, ctfi, transcripts) -> pd.DataFrame:
    df = enrich_score_features(matches)
    df = add_rank_features(df)
    df = encode_surface(df)

    # merge CTFI (both minutes and sets)
    df = df.merge(
        ctfi[["player_id", "tourney_id", "match_num", "ctfi_minutes", "ctfi_sets"]],
        on=["player_id", "tourney_id", "match_num"],
        how="left",
    )
    df["ctfi_minutes"] = df["ctfi_minutes"].fillna(0)
    df["ctfi_sets"]    = df["ctfi_sets"].fillna(0)

    # merge NLP
    df = merge_transcripts(df, transcripts)

    # surface-stratified NLP imputation
    df = impute_nlp_surface_median(df)

    # surface dummies
    surface_dummies = [c for c in df.columns if c.startswith("surface_")]
    keep = [c for c in FEATURE_COLS + surface_dummies if c in df.columns]
    df_features = df[keep].copy()
    df_features = df_features.dropna(subset=["upset"])

    return df_features


def describe_features(df: pd.DataFrame) -> None:
    print("\n── Feature matrix summary ────────────────────────")
    print(f"  Rows     : {len(df):,}")
    print(f"  Columns  : {len(df.columns)}")
    print(f"  Upset rate: {df['upset'].mean()*100:.1f}%")
    missing = df.isnull().mean()
    missing = missing[missing > 0].round(3)
    print("\n  Missing values:")
    print(missing.to_string() if not missing.empty else "    None")
    print("──────────────────────────────────────────────────\n")


def main() -> None:
    print("=" * 60)
    print("  PART 4 — Feature Engineering")
    print("=" * 60)
    matches     = load_matches()
    ctfi        = compute_ctfi()
    transcripts = load_transcript_features()
    df_features = build_final_features(matches, ctfi, transcripts)
    describe_features(df_features)
    df_features.to_csv(FEATURES_OUT, index=False)
    print(f"Feature matrix exported to: {FEATURES_OUT}")
    print("\nPart 4 complete. Run model.py next.")


if __name__ == "__main__":
    main()