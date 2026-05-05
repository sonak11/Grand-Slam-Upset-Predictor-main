"""
PART 4 — Feature Engineering   

Builds the final modelling DataFrame by:

  1. Computing the Cumulative Tournament Fatigue Index (CTFI) using SQL
     window functions. The CTFI is computed two ways:
         ctfi_minutes : sum of `minutes` over all preceding matches
                        (proposal-canonical definition).
         ctfi_sets    : sum of sets played over all preceding matches
                        (robust fallback used when minutes is missing,
                        which is common in pre-2010 Sackmann data).

  2. Surface-stratified median imputation of sparse serve statistics.

  3. Engineering rank-based features (rank_ratio, log_rank_diff,
     is_underdog, rank_bin) and the proposal's CTFI_diff feature.

  4. Joining match data ↔ NLP transcript features via a player-name fuzzy match (transcripts are keyed by player_name; matches are
     keyed by player_id).

  5. Exporting the final feature matrix to features.csv.

Key fixes vs. the original version:
  • Round map now correctly handles Sackmann's R128/R64/R32/R16
    in addition to R1-R4 (previous version silently mapped R128
    → NaN → 3, destroying within-tournament ordering).
  • CTFI window now ORDERs BY round_num (deterministic, monotonic
    within a tournament) rather than match_num, which in Sackmann's
    data runs LARGEST = earliest round.
  • CTFI is computed in true minutes (the proposal-canonical
    definition) when available, with sets-based fallback otherwise.
  • NLP transcripts are merged on (player_name, tourney_name, round)
    after a Unicode-normalised name comparison, so NLP features
    actually populate.
  • CTFI_diff (player CTFI minus opponent CTFI) is added.
  • Rows where CTFI cannot be computed are dropped from the final
    matrix to keep the modelling sample clean.

"""

from __future__ import annotations

import re
import sqlite3
import unicodedata

import numpy as np
import pandas as pd

DB_PATH       = "tennis_upsets.db"
FEATURES_OUT  = "features.csv"


# round normalisation 
#
# Sackmann's `round` column uses BOTH conventions across years:
#   "R128", "R64", "R32", "R16", "QF", "SF", "F"     (most years)
#   "R1",   "R2",  "R3",  "R4",  "QF", "SF", "F"     (some older files)
# Both must be mapped to the same monotonically-increasing ordinal so that
# CTFI window functions order matches correctly.

ROUND_TO_NUM = {
    # slam draw-size convention
    "R128": 1,  "R64": 2,  "R32": 3,  "R16": 4,
    # best-of-5-rounds short convention
    "R1":   1,  "R2":  2,  "R3":  3,  "R4":  4,
    # Final stages (shared)
    "QF":   5,  "SF":  6,  "F":   7,
    # robins / qualifiers occasionally appear — bucket them at 0
    "RR":   0,  "Q1":  0,  "Q2":  0,  "Q3":  0,  "BR": 0,
}


def round_to_num(r) -> int:
    if r is None or (isinstance(r, float) and np.isnan(r)):
        return 0
    s = str(r).strip().upper()
    return ROUND_TO_NUM.get(s, 0)


# helpers 

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def _normalise_name(s: str) -> str:
    """Strip diacritics, lowercase, collapse whitespace — for transcript join."""
    if not isinstance(s, str):
        return ""
    nfkd = unicodedata.normalize("NFKD", s)
    ascii_only = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", ascii_only).strip().lower()


# step 1: load grand slam matches 

def load_matches() -> pd.DataFrame:
    """
    Pull all Grand Slam matches and reshape into player-centric format
    (one row per player per match → each match yields 2 rows).
    """
    conn = get_connection()

    matches = pd.read_sql("""
        SELECT
            rowid                           AS match_db_id,
            tourney_id, tourney_name, slam_name, tourney_date,
            match_num, round, surface, best_of,
            COALESCE(minutes, 0)            AS match_minutes,
            winner_id                       AS player_id,
            winner_name                     AS player_name,
            winner_rank                     AS rank,
            winner_rank_points              AS rank_points,
            loser_id                        AS opponent_id,
            loser_name                      AS opponent_name,
            loser_rank                      AS opp_rank,
            loser_rank_points               AS opp_rank_points,
            score, upset, rank_diff,
            1                               AS won_match
        FROM matches

        UNION ALL

        SELECT
            rowid,
            tourney_id, tourney_name, slam_name, tourney_date,
            match_num, round, surface, best_of,
            COALESCE(minutes, 0),
            loser_id,  loser_name,  loser_rank,  loser_rank_points,
            winner_id, winner_name, winner_rank, winner_rank_points,
            score, upset, -rank_diff,
            0
        FROM matches

        ORDER BY tourney_date, tourney_id, match_num
    """, conn)
    conn.close()

    matches["tourney_date"] = pd.to_datetime(matches["tourney_date"], errors="coerce")
    matches["rank"]         = pd.to_numeric(matches["rank"],     errors="coerce")
    matches["opp_rank"]     = pd.to_numeric(matches["opp_rank"], errors="coerce")

    # canonical ordinal round number — used as CTFI window ORDER BY key
    matches["round_num"] = matches["round"].map(round_to_num).astype(int)

    print(f"Player-match rows loaded: {len(matches):,}")
    return matches


# step 2: score parsing 

def parse_score(score) -> dict:
    """Parse a score string into sets played, total games, sets won/lost."""
    default = {"sets_played": 3, "total_games": 30, "sets_won": 2, "sets_lost": 1}
    if not isinstance(score, str) or not score.strip():
        return default

    sets = re.findall(r"(\d+)-(\d+)", score)
    if not sets:
        return default

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

    # loser perspective: swap won ↔ lost
    loser_mask = df["won_match"] == 0
    df.loc[loser_mask, ["sets_won", "sets_lost"]] = (
        df.loc[loser_mask, ["sets_lost", "sets_won"]].values
    )
    return df


# step 3: Cumulative Tournament Fatigue Index (CTFI)  
#
#  Definition (proposal):
#      CTFI(p, r) = Σ minutes_played(p, k)   for k = 1 … r-1
#  i.e. the sum of court time across all matches the player has already
#  played in this tournament before the current match.
#
#  Implementation note:
#      We pre-compute round_num (monotonic 1..7) in Python and pass it
#      into the SQL via a CTE so SQLite's window function orders correctly.
#      We compute BOTH ctfi_minutes (true definition) and ctfi_sets
#      (sets-played fallback when minutes is missing).

CTFI_QUERY_TEMPLATE = """
WITH player_match AS (
    SELECT
        m.tourney_id,
        m.match_num,
        m.winner_id   AS player_id,
        COALESCE(m.minutes, 0) AS minutes,
        ((LENGTH(m.score) - LENGTH(REPLACE(m.score, '-', ''))) / 2)
                                AS sets_played,
        rmap.round_num
    FROM matches m
    JOIN round_map rmap ON rmap.round_str = m.round

    UNION ALL

    SELECT
        m.tourney_id,
        m.match_num,
        m.loser_id,
        COALESCE(m.minutes, 0),
        ((LENGTH(m.score) - LENGTH(REPLACE(m.score, '-', ''))) / 2),
        rmap.round_num
    FROM matches m
    JOIN round_map rmap ON rmap.round_str = m.round
),
ctfi_raw AS (
    SELECT
        tourney_id, player_id, match_num, round_num, minutes, sets_played,
        SUM(minutes) OVER (
            PARTITION BY player_id, tourney_id
            ORDER BY round_num, match_num
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS ctfi_minutes,
        SUM(sets_played) OVER (
            PARTITION BY player_id, tourney_id
            ORDER BY round_num, match_num
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS ctfi_sets
    FROM player_match
)
SELECT
    tourney_id, player_id, match_num,
    COALESCE(ctfi_minutes, 0) AS ctfi_minutes,
    COALESCE(ctfi_sets,    0) AS ctfi_sets
FROM ctfi_raw
"""


def compute_ctfi() -> pd.DataFrame:
    """
    Compute CTFI in pure SQL. We materialise a temporary `round_map`
    table so the SQL CTE can ORDER BY round_num correctly.
    """
    conn = get_connection()

    # materialise round_map for the CTE
    conn.execute("DROP TABLE IF EXISTS round_map")
    conn.execute("""
        CREATE TEMP TABLE round_map (
            round_str TEXT PRIMARY KEY,
            round_num INTEGER
        )
    """)
    conn.executemany(
        "INSERT INTO round_map (round_str, round_num) VALUES (?, ?)",
        list(ROUND_TO_NUM.items()),
    )

    ctfi = pd.read_sql(CTFI_QUERY_TEMPLATE, conn)
    conn.close()

    print(f"CTFI rows computed: {len(ctfi):,}")
    print(
        f"  ctfi_minutes  → mean={ctfi['ctfi_minutes'].mean():.1f}  "
        f"max={ctfi['ctfi_minutes'].max()}"
    )
    print(
        f"  ctfi_sets     → mean={ctfi['ctfi_sets'].mean():.2f}  "
        f"max={ctfi['ctfi_sets'].max()}"
    )
    return ctfi


# step 4: Rank features 

def add_rank_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rank_ratio"]    = df["rank"] / df["opp_rank"].replace(0, np.nan)
    df["log_rank_diff"] = np.sign(df["rank_diff"]) * np.log1p(np.abs(df["rank_diff"]))
    df["is_underdog"]   = (df["rank"] > df["opp_rank"]).astype(int)
    df["rank_bin"] = pd.cut(
        df["rank"],
        bins=[0, 10, 30, 100, np.inf],
        labels=["top10", "top30", "top100", "outside100"],
        right=True,
    )
    return df


# step 5: Surface-stratified imputation (proposal Phase 2) 

def surface_median_impute(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Fill NaNs in `cols` with the median of that column WITHIN each surface
    bucket, falling back to global median if a surface has no observations.
    """
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        df[col] = df.groupby("surface")[col].transform(
            lambda s: s.fillna(s.median())
        )
        df[col] = df[col].fillna(df[col].median())
    return df


def encode_surface(df: pd.DataFrame) -> pd.DataFrame:
    surface_dummies = pd.get_dummies(df["surface"], prefix="surface", dtype=int)
    return pd.concat([df, surface_dummies], axis=1)


# step 6: Transcript NLP merge 

ROUND_TEXT_TO_CODE = {
    "first round":   "R1",  "second round":  "R2",  "third round":   "R3",
    "fourth round":  "R4",  "round of 128":  "R1",  "round of 64":   "R2",
    "round of 32":   "R3",  "round of 16":   "R4",  "quarterfinal":  "QF",
    "quarter final": "QF",  "semifinal":     "SF",  "semi final":    "SF",
    "final":         "F",
    # pass-through codes
    "r1": "R1", "r2": "R2", "r3": "R3", "r4": "R4",
    "r128":"R1","r64":"R2","r32":"R3","r16":"R4",
    "qf": "QF", "sf": "SF", "f": "F",
}

NLP_COLS = [
    "sentiment_polarity", "fatigue_total", "fatigue_word_density",
    "fatigue_physical", "fatigue_mental", "fatigue_schedule",
    "fatigue_injury", "fatigue_motivation",
    "first_person_rate", "negation_rate",
    "llm_is_fatigued",
]


def load_transcript_features() -> pd.DataFrame:
    conn = get_connection()
    cols_info = conn.execute("PRAGMA table_info(transcripts)").fetchall()
    col_names = {row[1] for row in cols_info}

    if "sentiment_polarity" not in col_names:
        conn.close()
        print("[WARN] NLP columns not found — run nlp.py first.")
        return pd.DataFrame()

    transcripts = pd.read_sql("""
        SELECT
            player_id, player_name, tourney_name, tourney_date, round,
            sentiment_label, sentiment_polarity,
            fatigue_total, fatigue_word_density,
            fatigue_physical, fatigue_mental, fatigue_schedule,
            fatigue_injury, fatigue_motivation,
            first_person_rate, negation_rate,
            llm_fatigue_label, llm_fatigue_confidence
        FROM transcripts
        WHERE nlp_processed = 1
    """, conn)
    conn.close()

    if "llm_fatigue_label" in transcripts.columns:
        transcripts["llm_is_fatigued"] = (
            transcripts["llm_fatigue_label"] == "FATIGUED"
        ).astype(float)
        transcripts["llm_is_fatigued"].fillna(0.5, inplace=True)

    print(f"Transcript NLP rows loaded: {len(transcripts):,}")
    return transcripts


def merge_transcripts(matches: pd.DataFrame, transcripts: pd.DataFrame) -> pd.DataFrame:
    """
    Join transcripts onto matches via (normalised player_name,
    tourney_name, round_code).
    """
    if transcripts.empty:
        return matches

    avail = [c for c in NLP_COLS if c in transcripts.columns]

    # normalise both sides
    transcripts = transcripts.copy()
    transcripts["_pn"] = transcripts["player_name"].apply(_normalise_name)
    transcripts["_tn"] = transcripts["tourney_name"].astype(str).str.lower().str.strip()
    transcripts["_rc"] = (
        transcripts["round"].astype(str).str.lower().str.strip()
        .map(ROUND_TEXT_TO_CODE)
        .fillna(transcripts["round"].astype(str).str.upper().str.strip())
    )

    matches = matches.copy()
    matches["_pn"] = matches["player_name"].apply(_normalise_name)
    matches["_tn"] = matches["tourney_name"].astype(str).str.lower().str.strip()
    matches["_rc"] = matches["round"].astype(str).str.upper().str.strip()

    # deduplicate transcripts on join key (take first per group)
    transcripts = transcripts.drop_duplicates(subset=["_pn", "_tn", "_rc"], keep="first")

    merged = matches.merge(
        transcripts[["_pn", "_tn", "_rc"] + avail],
        on=["_pn", "_tn", "_rc"],
        how="left",
    )
    merged = merged.drop(columns=["_pn", "_tn", "_rc"])

    if avail:
        n_matched = merged[avail[0]].notna().sum()
        print(
            f"Transcript features matched: {n_matched:,}/{len(merged):,} rows "
            f"({n_matched/len(merged)*100:.1f}%)"
        )
    return merged


# step 7: final assembly 

FEATURE_COLS = [
    # match context
    "slam_name", "surface", "round_num", "best_of",
    # rank features
    "rank", "opp_rank", "rank_ratio", "log_rank_diff",
    "is_underdog", "rank_bin",
    # fatigue index (BOTH variants exposed)
    "ctfi_minutes", "ctfi_sets", "ctfi_diff_minutes", "ctfi_diff_sets",
    # NLP features
    "sentiment_polarity", "fatigue_total", "fatigue_word_density",
    "fatigue_physical", "fatigue_mental", "fatigue_schedule",
    "fatigue_injury", "fatigue_motivation",
    "first_person_rate", "negation_rate", "llm_is_fatigued",
    # target
    "upset",
]


def add_ctfi_diff(df: pd.DataFrame) -> pd.DataFrame:
    """
    Player CTFI minus opponent CTFI.  Each match has two rows (player and
    opponent perspective), so we self-join via (match_db_id, won_match).
    """
    df = df.copy()
    other = (
        df[["match_db_id", "won_match", "ctfi_minutes", "ctfi_sets"]]
        .rename(columns={
            "ctfi_minutes": "opp_ctfi_minutes",
            "ctfi_sets":    "opp_ctfi_sets",
            "won_match":    "_other_won",
        })
    )
    other["_match_won_inv"] = 1 - other["_other_won"]

    df = df.merge(
        other[["match_db_id", "_match_won_inv", "opp_ctfi_minutes", "opp_ctfi_sets"]],
        left_on=["match_db_id", "won_match"],
        right_on=["match_db_id", "_match_won_inv"],
        how="left",
    ).drop(columns=["_match_won_inv"])

    df["ctfi_diff_minutes"] = df["ctfi_minutes"] - df["opp_ctfi_minutes"].fillna(0)
    df["ctfi_diff_sets"]    = df["ctfi_sets"]    - df["opp_ctfi_sets"].fillna(0)
    return df


def build_final_features(matches, ctfi, transcripts) -> pd.DataFrame:
    df = enrich_score_features(matches)
    df = add_rank_features(df)
    df = encode_surface(df)

    df = df.merge(
        ctfi[["tourney_id", "player_id", "match_num",
              "ctfi_minutes", "ctfi_sets"]],
        on=["tourney_id", "player_id", "match_num"],
        how="left",
    )
    df["ctfi_minutes"] = df["ctfi_minutes"].fillna(0)
    df["ctfi_sets"]    = df["ctfi_sets"].fillna(0)

    df = add_ctfi_diff(df)
    df = merge_transcripts(df, transcripts)

    # surface-stratified imputation for NLP features only
    # (per proposal Phase 2 — "surface-stratified median imputation")
    df = surface_median_impute(df, NLP_COLS)

    surface_dummies = [c for c in df.columns if c.startswith("surface_")]
    keep_cols = [c for c in FEATURE_COLS + surface_dummies if c in df.columns]
    df_features = df[keep_cols].copy()

    df_features = df_features.dropna(subset=["upset"])
    return df_features


# step 8: Diagnostics

def describe_features(df: pd.DataFrame) -> None:
    print("\n Feature matrix summary ")
    print(f"  Rows       : {len(df):,}")
    print(f"  Columns    : {len(df.columns)}")
    print(f"  Upset rate : {df['upset'].mean()*100:.1f}%")

    if "ctfi_minutes" in df.columns:
        print("\n  CTFI (minutes) sanity check:")
        print(f"    mean   = {df['ctfi_minutes'].mean():.1f}")
        print(f"    median = {df['ctfi_minutes'].median():.1f}")
        print(f"    max    = {df['ctfi_minutes'].max():.0f}")
        print(f"    %==0   = {(df['ctfi_minutes']==0).mean()*100:.1f}%")

    print("\n  Missing values:")
    missing = df.isnull().mean()
    missing = missing[missing > 0].round(3)
    print(missing.to_string() if not missing.empty else "    None")
    print("──────────────────────────────────────────────────\n")


# main

def main() -> None:
    print("=" * 60)
    print("  PART 4 — Feature Engineering  (FIXED)")
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