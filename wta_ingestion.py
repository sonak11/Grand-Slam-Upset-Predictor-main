"""
wta_ingestion.py
=================
Adds WTA Grand Slam match data + press conference transcripts
to the existing tennis_upsets.db database.

Run AFTER the full ATP pipeline is complete:
    python data_ingestion.py
    python scraping.py
    python nlp.py
    python features.py
    python model.py
    python wta_ingestion.py   ← run this, then re-run features.py + model.py

Usage:
    source venv/bin/activate
    python wta_ingestion.py
"""

import re
import io
import time
import random
import sqlite3
import logging
from datetime import datetime, timedelta
from urllib.parse import urljoin

import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────

DB_PATH   = "tennis_upsets.db"
DELAY_MIN = 1.5
DELAY_MAX = 3.2
MAX_RETRIES = 3

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.asapsports.com",
}

ASAP_BASE    = "https://www.asapsports.com"
ASAP_DAY_URL = f"{ASAP_BASE}/show_event.php"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── WTA CSV source ────────────────────────────────────────────────────────────

WTA_CSV_URLS = {
    2022: "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_matches_2022.csv",
    2023: "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_matches_2023.csv",
    2024: "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_matches_2024.csv",
}

# WTA Grand Slam tourney IDs (last 3 digits of tourney_id)
WTA_SLAM_IDS = {"580", "520", "540", "560"}

WTA_SLAM_NAMES = {
    "580": "Australian Open",
    "520": "Roland Garros",
    "540": "Wimbledon",
    "560": "US Open",
}

# ── WTA ASAP Sports tournament dates ─────────────────────────────────────────

WTA_TOURNAMENTS = [
    {"name":"Australian Open","title":"AUSTRALIAN+OPEN","start":"2022-01-17","end":"2022-01-29"},
    {"name":"Roland Garros",  "title":"ROLAND+GARROS",  "start":"2022-05-23","end":"2022-06-04"},
    {"name":"Wimbledon",      "title":"THE+CHAMPIONSHIPS","start":"2022-06-28","end":"2022-07-10"},
    {"name":"US Open",        "title":"US+OPEN",         "start":"2022-08-29","end":"2022-09-10"},
    {"name":"Australian Open","title":"AUSTRALIAN+OPEN","start":"2023-01-16","end":"2023-01-28"},
    {"name":"Roland Garros",  "title":"ROLAND+GARROS",  "start":"2023-05-28","end":"2023-06-10"},
    {"name":"Wimbledon",      "title":"THE+CHAMPIONSHIPS","start":"2023-07-03","end":"2023-07-15"},
    {"name":"US Open",        "title":"US+OPEN",         "start":"2023-08-28","end":"2023-09-09"},
    {"name":"Australian Open","title":"AUSTRALIAN+OPEN","start":"2024-01-14","end":"2024-01-27"},
    {"name":"Roland Garros",  "title":"ROLAND+GARROS",  "start":"2024-05-26","end":"2024-06-08"},
    {"name":"Wimbledon",      "title":"THE+CHAMPIONSHIPS","start":"2024-07-01","end":"2024-07-13"},
    {"name":"US Open",        "title":"US+OPEN",         "start":"2024-08-26","end":"2024-09-07"},
]


# ── Database ──────────────────────────────────────────────────────────────────

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def get_matches_columns():
    """Return the actual column names of the existing matches table."""
    conn = get_connection()
    cols = [r[1] for r in conn.execute("PRAGMA table_info(matches)").fetchall()]
    conn.close()
    return cols


def add_tour_column_if_missing():
    """Add tour column to matches and transcripts if not already there."""
    conn = get_connection()
    existing = [r[1] for r in conn.execute("PRAGMA table_info(matches)").fetchall()]
    if "tour" not in existing:
        conn.execute("ALTER TABLE matches ADD COLUMN tour TEXT DEFAULT 'ATP'")
        conn.execute("UPDATE matches SET tour='ATP' WHERE tour IS NULL")
        log.info("Added 'tour' column to matches table")

    existing_t = [r[1] for r in conn.execute("PRAGMA table_info(transcripts)").fetchall()]
    if "tour" not in existing_t:
        conn.execute("ALTER TABLE transcripts ADD COLUMN tour TEXT DEFAULT 'ATP'")
        conn.execute("UPDATE transcripts SET tour='ATP' WHERE tour IS NULL")
        log.info("Added 'tour' column to transcripts table")

    conn.commit()
    conn.close()


def row_exists(tourney_id, match_num):
    """Check if a match already exists (ATP or WTA)."""
    conn = get_connection()
    row = conn.execute(
        "SELECT 1 FROM matches WHERE tourney_id=? AND match_num=?",
        (str(tourney_id), str(match_num))
    ).fetchone()
    conn.close()
    return row is not None


def transcript_exists(url):
    conn = get_connection()
    row = conn.execute(
        "SELECT 1 FROM transcripts WHERE source_url=?", (url,)
    ).fetchone()
    conn.close()
    return row is not None


def save_transcript(record):
    conn = get_connection()
    conn.execute("""
        INSERT OR IGNORE INTO transcripts
            (player_name, tourney_name, tourney_date, round,
             source_url, scraped_at, raw_text, tour)
        VALUES
            (:player_name, :tourney_name, :tourney_date, :round,
             :source_url, :scraped_at, :raw_text, :tour)
    """, record)
    conn.commit()
    conn.close()


# ── HTTP ──────────────────────────────────────────────────────────────────────

def polite_get(url):
    for attempt in range(1, MAX_RETRIES + 1):
        time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))
        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            log.warning(f"  Attempt {attempt}/{MAX_RETRIES}: {e}")
            time.sleep(2 ** attempt)
    return None


# ── Part 1: WTA match data ────────────────────────────────────────────────────

ROUND_REMAP = {
    "R128": "R1", "R64": "R2", "R32": "R3", "R16": "R4",
    "QF": "QF", "SF": "SF", "F": "F",
}

SLAM_NAME_MAP = {
    "australian open": "Australian Open",
    "roland garros":   "Roland Garros",
    "french open":     "Roland Garros",
    "wimbledon":       "Wimbledon",
    "us open":         "US Open",
}


def _get_slam_name(tourney_name, tourney_id):
    """Map tourney_name to our canonical slam_name."""
    t = str(tourney_name).lower()
    for key, val in SLAM_NAME_MAP.items():
        if key in t:
            return val
    # Fallback: use tourney_id suffix
    suffix = str(tourney_id).split("-")[-1][-3:]
    return WTA_SLAM_NAMES.get(suffix, tourney_name)


def ingest_wta_matches():
    """Download WTA CSVs and append Grand Slam rows to matches table."""
    log.info("=" * 55)
    log.info("  Part 1: WTA Match Data")
    log.info("=" * 55)

    # Get actual column names from existing matches table
    existing_cols = get_matches_columns()
    log.info(f"  Existing matches columns: {existing_cols}")

    total_saved = 0

    for year, url in WTA_CSV_URLS.items():
        log.info(f"\nDownloading WTA {year}…")
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            df   = pd.read_csv(io.StringIO(resp.text), low_memory=False)
        except Exception as e:
            log.error(f"  Failed: {e}")
            continue

        log.info(f"  {len(df)} total WTA matches in {year}")

        # Filter to Grand Slams by tourney_id suffix
        def is_slam(tid):
            return str(tid).split("-")[-1][-3:] in WTA_SLAM_IDS
        df = df[df["tourney_id"].apply(is_slam)].copy()
        log.info(f"  {len(df)} Grand Slam matches")

        if df.empty:
            continue

        # Compute derived columns to match ATP schema
        df["winner_rank"] = pd.to_numeric(df["winner_rank"], errors="coerce")
        df["loser_rank"]  = pd.to_numeric(df["loser_rank"],  errors="coerce")
        df               = df.dropna(subset=["winner_rank", "loser_rank"])

        df["upset"]     = (df["winner_rank"] > df["loser_rank"]).astype(int)
        df["rank_diff"] = df["winner_rank"] - df["loser_rank"]
        df["slam_name"] = df.apply(
            lambda r: _get_slam_name(r["tourney_name"], r["tourney_id"]), axis=1
        )
        df["year"]      = year
        df["best_of"]   = 3   # WTA Grand Slams best of 3
        df["tour"]      = "WTA"

        # Normalise tourney_date
        df["tourney_date"] = pd.to_datetime(
            df["tourney_date"].astype(str), format="%Y%m%d", errors="coerce"
        ).dt.strftime("%Y-%m-%d")

        # Remap round values to match ATP schema
        if "round" in df.columns:
            df["round"] = df["round"].map(ROUND_REMAP).fillna(df["round"])

        # Only keep columns that exist in the target table
        cols_to_use = [c for c in df.columns if c in existing_cols + ["tour"]]
        df = df[cols_to_use]

        # Append to DB row-by-row (checking for duplicates)
        conn = get_connection()
        for _, row in df.iterrows():
            try:
                if row_exists(row.get("tourney_id",""), row.get("match_num","")):
                    continue
                row_dict = {k: (None if pd.isna(v) else v) for k, v in row.items()}
                cols     = list(row_dict.keys())
                placeholders = ", ".join(["?"] * len(cols))
                col_names    = ", ".join(cols)
                conn.execute(
                    f"INSERT OR IGNORE INTO matches ({col_names}) VALUES ({placeholders})",
                    list(row_dict.values())
                )
                total_saved += 1
            except Exception as e:
                log.debug(f"  Row insert error: {e}")
        conn.commit()
        conn.close()
        log.info(f"  Saved so far: {total_saved}")

    conn = get_connection()
    n_wta = conn.execute("SELECT COUNT(*) FROM matches WHERE tour='WTA'").fetchone()[0]
    conn.close()
    log.info(f"\n✅ WTA matches in DB: {n_wta:,}")
    return total_saved


# ── Part 2: WTA transcript scraping ──────────────────────────────────────────

def _extract_round(text):
    t = text.lower()
    if re.search(r"\bsemifinale?\b|\bsf\b",   t): return "SF"
    if re.search(r"\bquarterfinale?\b|\bqf\b", t): return "QF"
    if re.search(r"\bfourth round\b|\br4\b",   t): return "R4"
    if re.search(r"\bthird round\b|\br3\b",    t): return "R3"
    if re.search(r"\bsecond round\b|\br2\b",   t): return "R2"
    if re.search(r"\bfirst round\b|\br1\b",    t): return "R1"
    if re.search(r"\bfinal\b",                 t): return "F"
    return "Unknown"


def _extract_player_name(text):
    for line in text.split("\n")[:15]:
        line = line.strip()
        m = re.match(r"AN INTERVIEW WITH[:\s]+(.+)", line, re.IGNORECASE)
        if m:
            return m.group(1).title().strip()
        if re.match(r"^[A-Z][A-Z\s\-\.]{4,39}$", line) and len(line.split()) >= 2:
            return line.title().strip()
    return "Unknown"


def _clean_transcript(text):
    text = re.sub(r"&[a-zA-Z]+;",      " ", text)
    text = re.sub(r"&#\d+;",            " ", text)
    text = re.sub(r"FastScripts.*",     "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"Copyright.*",       "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"Browse by Sport.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}",  " ",    text)
    return text.strip()


def generate_date_urls(tournament):
    start   = datetime.strptime(tournament["start"], "%Y-%m-%d")
    end     = datetime.strptime(tournament["end"],   "%Y-%m-%d")
    urls    = []
    current = start
    while current <= end:
        date_str = f"{current.year}-{current.month}-{current.day}"
        url      = (f"{ASAP_DAY_URL}?category=7"
                    f"&date={date_str}&title={tournament['title']}")
        urls.append((url, current.strftime("%Y-%m-%d")))
        current += timedelta(days=1)
    return urls


def get_interviews_from_day(day_url, date_str, tournament):
    resp = polite_get(day_url)
    if resp is None:
        return []

    soup       = BeautifulSoup(resp.text, "html.parser")
    interviews = []
    year       = date_str[:4]
    title_tag  = soup.find("title")
    page_title = title_tag.get_text(strip=True) if title_tag else ""

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "show_interview.php" not in href:
            continue
        full_url = urljoin(ASAP_BASE, href)
        if transcript_exists(full_url):
            continue

        player_name = a.get_text(strip=True)
        row_text    = page_title
        if a.parent:
            row_text += " " + a.parent.get_text(" ", strip=True)

        interviews.append({
            "url":          full_url,
            "player_name":  player_name if len(player_name) > 2 else "Unknown",
            "tourney_name": tournament["name"],
            "tourney_year": year,
            "round":        _extract_round(row_text),
        })
    return interviews


def scrape_interview(url, meta):
    resp = polite_get(url)
    if resp is None:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # Primary: <p> tags (confirmed working)
    paragraphs = [
        p.get_text(" ", strip=True)
        for p in soup.find_all("p")
        if len(p.get_text(strip=True)) > 20
    ]
    raw_text = "\n\n".join(paragraphs)

    # Fallback: largest meaningful <td>
    if len(raw_text) < 200:
        tds = soup.find_all("td")
        for td in sorted(tds, key=lambda t: len(t.get_text()), reverse=True):
            td_text = td.get_text("\n", strip=True)
            if "THE MODERATOR" in td_text or "Press Conference" in td_text:
                raw_text = td_text
                break

    raw_text = _clean_transcript(raw_text)
    if len(raw_text) < 150:
        return None

    player_name = meta.get("player_name", "Unknown")
    if not player_name or player_name == "Unknown":
        player_name = _extract_player_name(raw_text)

    return {
        "player_name":  player_name,
        "tourney_name": meta["tourney_name"],
        "tourney_date": meta["tourney_year"],
        "round":        meta["round"],
        "source_url":   url,
        "scraped_at":   datetime.utcnow().isoformat(),
        "raw_text":     raw_text,
        "tour":         "WTA",
    }


def scrape_wta_transcripts():
    """
    ASAP Sports mixes ATP and WTA on the same event pages.
    We scrape all interviews and save only those matching known WTA player names.
    """
    log.info("\n" + "=" * 55)
    log.info("  Part 2: WTA Transcript Scraping")
    log.info("=" * 55)

    # Build set of known WTA player names from DB
    conn      = get_connection()
    wta_rows  = conn.execute(
        "SELECT DISTINCT winner_name FROM matches WHERE tour='WTA' "
        "UNION SELECT DISTINCT loser_name FROM matches WHERE tour='WTA'"
    ).fetchall()
    conn.close()
    wta_names = {r[0].lower() for r in wta_rows if r[0]}
    log.info(f"  Known WTA players: {len(wta_names)}")

    if not wta_names:
        log.warning("  No WTA players found — run Part 1 first!")
        return 0

    total_saved = 0

    for tournament in tqdm(WTA_TOURNAMENTS, desc="Tournaments"):
        name  = tournament["name"]
        year  = tournament["start"][:4]
        label = f"{name} {year}"

        log.info(f"\n🎾  {label}")
        date_urls      = generate_date_urls(tournament)
        all_interviews = []

        for day_url, date_str in date_urls:
            interviews = get_interviews_from_day(day_url, date_str, tournament)
            all_interviews.extend(interviews)

        log.info(f"  {len(all_interviews)} interview links found")

        saved = 0
        for meta in tqdm(all_interviews, desc=f"  {label}", leave=False):
            record = scrape_interview(meta["url"], meta)
            if not record:
                continue

            # Save if player name matches a known WTA player
            player_lower = (record["player_name"] or "").lower()
            is_wta = any(
                wn in player_lower or player_lower in wn
                for wn in wta_names
                if len(wn) > 4  # avoid short name false matches
            )
            if is_wta:
                save_transcript(record)
                saved       += 1
                total_saved += 1

        log.info(f"  WTA transcripts saved: {saved}")

    return total_saved


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  WTA Data Ingestion — Matches + Transcripts")
    print("=" * 60)
    print()

    # Step 0: add tour column to existing tables
    add_tour_column_if_missing()

    # Step 1: WTA match data
    match_count = ingest_wta_matches()

    # Step 2: WTA transcripts
    transcript_count = scrape_wta_transcripts()

    # Final summary
    conn     = get_connection()
    n_atp_m  = conn.execute("SELECT COUNT(*) FROM matches WHERE tour='ATP' OR tour IS NULL").fetchone()[0]
    n_wta_m  = conn.execute("SELECT COUNT(*) FROM matches WHERE tour='WTA'").fetchone()[0]
    n_atp_t  = conn.execute("SELECT COUNT(*) FROM transcripts WHERE tour='ATP' OR tour IS NULL").fetchone()[0]
    n_wta_t  = conn.execute("SELECT COUNT(*) FROM transcripts WHERE tour='WTA'").fetchone()[0]
    conn.close()

    print(f"\n{'='*60}")
    print(f"  FINAL DATABASE STATE")
    print(f"{'='*60}")
    print(f"  ATP matches:     {n_atp_m:>7,}")
    print(f"  WTA matches:     {n_wta_m:>7,}  ← new")
    print(f"  ATP transcripts: {n_atp_t:>7,}")
    print(f"  WTA transcripts: {n_wta_t:>7,}  ← new")
    print(f"{'='*60}")
    print(f"\n✅ Done! Now re-run:")
    print(f"     python nlp.py       ← process new WTA transcripts")
    print(f"     python features.py  ← rebuild feature matrix")
    print(f"     python model.py     ← retrain with combined data")


if __name__ == "__main__":
    main()