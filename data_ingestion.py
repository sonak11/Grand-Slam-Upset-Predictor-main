

import os, io, sqlite3, requests
import pandas as pd
from tqdm import tqdm

# config 
DB_PATH = "tennis_upsets.db"

ATP_BASE = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master"
WTA_BASE = "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master"

GRAND_SLAM_IDS = {
    "520": "Australian Open",
    "540": "Roland Garros",
    "560": "Wimbledon",
    "580": "US Open",
}

# WTA Grand Slam IDs differ slightly — both 4-digit suffix sets covered
WTA_SLAM_IDS = {
    "520": "Australian Open",
    "540": "Roland Garros",
    "560": "Wimbledon",
    "580": "US Open",
}

YEARS = list(range(1990, 2025))


# helpers 

def fetch_csv(url: str) -> "pd.DataFrame | None":
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        return pd.read_csv(io.StringIO(resp.text), low_memory=False)
    except Exception as exc:
        print(f"  [WARN] Could not fetch {url}: {exc}")
        return None


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


# step 1: Download match data 

def download_matches(years: list, base_url: str, prefix: str, tour: str) -> pd.DataFrame:
    """Download {prefix}_matches_YYYY.csv for each year and concatenate."""
    frames = []
    for year in tqdm(years, desc=f"Downloading {tour} match CSVs"):
        url = f"{base_url}/{prefix}_matches_{year}.csv"
        df = fetch_csv(url)
        if df is not None:
            df["year"] = year
            df["tour"] = tour
            frames.append(df)

    if not frames:
        print(f"[WARN] No {tour} match data downloaded.")
        return pd.DataFrame()

    matches = pd.concat(frames, ignore_index=True)
    print(f"\n{tour} matches downloaded: {len(matches):,}")
    return matches


# step 2: Filter Grand Slams and clean 

def filter_grand_slams(matches: pd.DataFrame, slam_ids: dict) -> pd.DataFrame:
    if matches.empty:
        return matches

    slam_suffixes = set(slam_ids.keys())
    mask = (
        matches["tourney_id"].astype(str).str[-3:].isin(slam_suffixes) |
        matches["tourney_id"].astype(str).str.split("-").str[-1].isin(slam_suffixes)
    )
    slams = matches[mask].copy()

    for col in ["winner_rank", "loser_rank"]:
        slams[col] = pd.to_numeric(slams[col], errors="coerce")

    slams = slams.dropna(subset=["winner_rank", "loser_rank"])

    # Upset label
    slams["upset"] = (slams["winner_rank"] > slams["loser_rank"]).astype(int)
    slams["rank_diff"] = slams["winner_rank"] - slams["loser_rank"]

    # minutes - coerce to numeric (many older matches have no duration)
    if "minutes" in slams.columns:
        slams["minutes"] = pd.to_numeric(slams["minutes"], errors="coerce")
    else:
        slams["minutes"] = float("nan")

    # Normalise tourney_date to YYYY-MM-DD
    slams["tourney_date"] = pd.to_datetime(
        slams["tourney_date"].astype(str), format="%Y%m%d", errors="coerce"
    )

    # clean slam name
    slams["slam_name"] = (
        slams["tourney_id"].astype(str).str.split("-").str[-1].map(slam_ids)
    )

    # WTA uses best_of=3; ATP uses best_of=5
    if "best_of" not in slams.columns:
        slams["best_of"] = 3 if slams["tour"].iloc[0] == "WTA" else 5
    slams["best_of"] = pd.to_numeric(slams["best_of"], errors="coerce").fillna(
        3 if (slams["tour"] == "WTA").all() else 5
    )

    # round ordering: assign an ordinal for correct CTFI ordering
    round_order = {
        "R128": 1, "R64": 2, "R32": 3, "R16": 4, "QF": 5, "SF": 6, "F": 7,
        "R1": 1, "R2": 2, "R3": 3, "R4": 4,  # older convention
    }
    slams["round_num"] = slams["round"].map(round_order).fillna(3)

    return slams


# step 3: Download player info 

def download_players(atp_base: str, wta_base: str) -> pd.DataFrame:
    frames = []
    for base, tour in [(atp_base, "ATP"), (wta_base, "WTA")]:
        prefix = "atp" if tour == "ATP" else "wta"
        url = f"{base}/{prefix}_players.csv"
        df = fetch_csv(url)
        if df is None:
            continue
        # standardise columns (WTA file may have slightly different structure)
        if df.shape[1] >= 6:
            df = df.iloc[:, :8]
            df.columns = (
                ["player_id", "first_name", "last_name", "hand", "dob", "ioc",
                 "height", "wikidata_id"][: df.shape[1]]
            )
        else:
            df.columns = ["player_id", "first_name", "last_name", "hand", "dob", "ioc"][
                : df.shape[1]
            ]
        df["full_name"] = (
            df["first_name"].fillna("") + " " + df["last_name"].fillna("")
        ).str.strip()
        df["tour"] = tour
        df["dob"] = pd.to_datetime(
            df["dob"].astype(str), format="%Y%m%d", errors="coerce"
        )
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    players = pd.concat(frames, ignore_index=True)
    players = players.drop_duplicates(subset=["player_id"])
    print(f"Players loaded: {len(players):,}")
    return players


# step 4: Write to SQLite 

def write_to_db(matches: pd.DataFrame, players: pd.DataFrame) -> None:
    conn = get_connection()

    print("\nWriting matches …")
    matches.to_sql("matches", conn, if_exists="replace", index=False)

    if not players.empty:
        print("Writing players …")
        players.to_sql("players", conn, if_exists="replace", index=False)

    # covering indexes for CTFI window function, transcript join, etc.
    for sql in [
        "CREATE INDEX IF NOT EXISTS idx_m_winner   ON matches(winner_id, tourney_id, round_num);",
        "CREATE INDEX IF NOT EXISTS idx_m_loser    ON matches(loser_id,  tourney_id, round_num);",
        "CREATE INDEX IF NOT EXISTS idx_m_date     ON matches(tourney_date);",
        "CREATE INDEX IF NOT EXISTS idx_m_slam     ON matches(slam_name);",
        "CREATE INDEX IF NOT EXISTS idx_m_tour     ON matches(tour);",
        "CREATE INDEX IF NOT EXISTS idx_p_name     ON players(full_name);",
    ]:
        conn.execute(sql)

    conn.commit()
    conn.close()
    print(f"\nDatabase written to: {os.path.abspath(DB_PATH)}")


# step 5: Sanity check 

def sanity_check() -> None:
    conn = get_connection()

    total  = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    upsets = conn.execute("SELECT COUNT(*) FROM matches WHERE upset=1").fetchone()[0]
    by_tour = conn.execute(
        "SELECT tour, COUNT(*) FROM matches GROUP BY tour ORDER BY tour"
    ).fetchall()
    by_slam = conn.execute(
        "SELECT slam_name, COUNT(*) FROM matches GROUP BY slam_name ORDER BY 2 DESC"
    ).fetchall()

    print("\n── Sanity check ──────────────────────────────")
    print(f"  Total Grand Slam matches : {total:,}")
    print(f"  Upset matches            : {upsets:,}  ({upsets/max(total,1)*100:.1f}%)")
    print("  Breakdown by tour:")
    for tour, n in by_tour:
        print(f"    {tour or 'Unknown':<8} {n:>6,}")
    print("  Breakdown by slam:")
    for name, n in by_slam:
        print(f"    {name or 'Unknown':<20} {n:>6,}")
    print("─────────────────────────────────────────────\n")
    conn.close()


#  main 

def main() -> None:
    print("=" * 60)
    print("  PART 1 — Tennis Data Ingestion")
    print("=" * 60)

    # ATP
    atp_raw    = download_matches(YEARS, ATP_BASE, "atp", "ATP")
    atp_slams  = filter_grand_slams(atp_raw, GRAND_SLAM_IDS) if not atp_raw.empty else pd.DataFrame()

    # WTA
    wta_raw    = download_matches(YEARS, WTA_BASE, "wta", "WTA")
    wta_slams  = filter_grand_slams(wta_raw, WTA_SLAM_IDS) if not wta_raw.empty else pd.DataFrame()

    # combine
    all_matches = pd.concat(
        [df for df in [atp_slams, wta_slams] if not df.empty], ignore_index=True
    )
    print(f"\nGrand Slam matches found: {len(all_matches):,}")

    # players
    players = download_players(ATP_BASE, WTA_BASE)

    write_to_db(all_matches, players)
    sanity_check()

    print("Part 1 complete. Run scraping.py next.")


if __name__ == "__main__":
    main()