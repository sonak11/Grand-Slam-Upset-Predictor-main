"""
PART 2 — Transcript Scraping

Scrapes Grand Slam press conference transcripts from asapsports.com.

Transcript text is stored in <p> tags on interview pages.
Day pages list interviews with show_interview.php links.


"""

import re
import time
import random
import sqlite3
import logging
from datetime import datetime, timedelta
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

DB_PATH     = "tennis_upsets.db"
DELAY_MIN   = 1.5
DELAY_MAX   = 3.0
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ASAP_BASE     = "https://www.asapsports.com"
ASAP_DAY_URL  = f"{ASAP_BASE}/show_event.php"

TOURNAMENTS = [
    {"name": "Australian Open", "title": "AUSTRALIAN+OPEN",
     "start": "2022-01-17", "end": "2022-01-30"},
    {"name": "Roland Garros",   "title": "ROLAND+GARROS",
     "start": "2022-05-23", "end": "2022-06-05"},
    {"name": "Wimbledon",       "title": "THE+CHAMPIONSHIPS",
     "start": "2022-06-28", "end": "2022-07-11"},
    {"name": "US Open",         "title": "US+OPEN",
     "start": "2022-08-29", "end": "2022-09-11"},

    {"name": "Australian Open", "title": "AUSTRALIAN+OPEN",
     "start": "2023-01-16", "end": "2023-01-29"},
    {"name": "Roland Garros",   "title": "ROLAND+GARROS",
     "start": "2023-05-28", "end": "2023-06-11"},
    {"name": "Wimbledon",       "title": "THE+CHAMPIONSHIPS",
     "start": "2023-07-03", "end": "2023-07-16"},
    {"name": "US Open",         "title": "US+OPEN",
     "start": "2023-08-28", "end": "2023-09-10"},

    {"name": "Australian Open", "title": "AUSTRALIAN+OPEN",
     "start": "2024-01-14", "end": "2024-01-28"},
    {"name": "Roland Garros",   "title": "ROLAND+GARROS",
     "start": "2024-05-26", "end": "2024-06-09"},
    {"name": "Wimbledon",       "title": "THE+CHAMPIONSHIPS",
     "start": "2024-07-01", "end": "2024-07-14"},
    {"name": "US Open",         "title": "US+OPEN",
     "start": "2024-08-26", "end": "2024-09-08"},
]


# database

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def ensure_transcript_table():
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS transcripts (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            player_name  TEXT NOT NULL,
            player_id    INTEGER,
            tourney_name TEXT,
            tourney_date TEXT,
            round        TEXT,
            source_url   TEXT UNIQUE,
            scraped_at   TEXT,
            raw_text     TEXT
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_trans_player ON transcripts(player_name);"
    )
    conn.commit()
    conn.close()


def transcript_exists(url):
    conn = get_connection()
    row  = conn.execute(
        "SELECT 1 FROM transcripts WHERE source_url=?", (url,)
    ).fetchone()
    conn.close()
    return row is not None


def save_transcript(record):
    conn = get_connection()
    conn.execute("""
        INSERT OR IGNORE INTO transcripts
            (player_name, player_id, tourney_name, tourney_date,
             round, source_url, scraped_at, raw_text)
        VALUES
            (:player_name, :player_id, :tourney_name, :tourney_date,
             :round, :source_url, :scraped_at, :raw_text)
    """, record)
    conn.commit()
    conn.close()


#HTTP 

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


# date URL generation 

def generate_date_urls(tournament):
    start   = datetime.strptime(tournament["start"], "%Y-%m-%d")
    end     = datetime.strptime(tournament["end"],   "%Y-%m-%d")
    urls    = []
    current = start
    while current <= end:
        date_str = f"{current.year}-{current.month}-{current.day}"
        url = (
            f"{ASAP_DAY_URL}?category=7"
            f"&date={date_str}"
            f"&title={tournament['title']}"
        )
        urls.append((url, current.strftime("%Y-%m-%d")))
        current += timedelta(days=1)
    return urls


# day page → interview links 

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


# interview page → transcript 

def scrape_interview(url, meta):
    """
    Transcript text is in <p> tags on ASAP interview pages.
    We collect all <p> text that comes after the 'Press Conference'
    header, which is inside an <h3> tag.
    """
    resp = polite_get(url)
    if resp is None:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # Strategy: collect all <p> tag text — the transcript IS the <p> content
    # Filter out short navigation paragraphs (< 20 chars)
    paragraphs = [
        p.get_text(" ", strip=True)
        for p in soup.find_all("p")
        if len(p.get_text(strip=True)) > 20
    ]
    raw_text = "\n\n".join(paragraphs)

    # Fallback: if no <p> tags with content, try <td>
    if len(raw_text) < 200:
        tds = soup.find_all("td")
        if tds:
            # Skip the nav sidebar (it starts with "Browse by Sport")
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
        player_name = _extract_player_name_from_transcript(raw_text)

    return {
        "player_name":  player_name,
        "player_id":    None,
        "tourney_name": meta["tourney_name"],
        "tourney_date": meta["tourney_year"],
        "round":        meta["round"],
        "source_url":   url,
        "scraped_at":   datetime.utcnow().isoformat(),
        "raw_text":     raw_text,
    }


# text utilities 

def _clean_transcript(text):
    text = re.sub(r"&[a-zA-Z]+;",  " ", text)
    text = re.sub(r"&#\d+;",        " ", text)
    text = re.sub(r"FastScripts.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"Copyright.*",   "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"Browse by Sport.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}",  " ",    text)
    return text.strip()


def _extract_player_name_from_transcript(text):
    for line in text.split("\n")[:15]:
        line = line.strip()
        m = re.match(r"AN INTERVIEW WITH[:\s]+(.+)", line, re.IGNORECASE)
        if m:
            return m.group(1).title().strip()
        if re.match(r"^[A-Z][A-Z\s\-\.]{4,39}$", line) and len(line.split()) >= 2:
            return line.title().strip()
    return "Unknown"


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


# player linking 

def link_transcripts_to_players():
    conn    = get_connection()
    players = conn.execute("SELECT player_id, full_name FROM players").fetchall()
    if not players:
        log.warning("No players table — run data_ingestion.py first.")
        conn.close()
        return

    name_to_id = {name.lower(): pid for pid, name in players}
    unlinked   = conn.execute(
        "SELECT id, player_name FROM transcripts WHERE player_id IS NULL"
    ).fetchall()

    linked = 0
    for row_id, player_name in unlinked:
        name = (player_name or "").lower()
        pid  = name_to_id.get(name)
        if pid is None:
            parts = name.split()
            if parts:
                last  = parts[-1]
                cands = [i for n, i in name_to_id.items() if last and last in n]
                if len(cands) == 1:
                    pid = cands[0]
        if pid:
            conn.execute(
                "UPDATE transcripts SET player_id=? WHERE id=?", (pid, row_id)
            )
            linked += 1

    conn.commit()
    conn.close()
    log.info(f"Linked {linked}/{len(unlinked)} transcripts to player IDs.")


# main 

def main():
    print("=" * 60)
    print("  PART 2 — Transcript Scraping")
    print("=" * 60)
    print(f"  Tournaments : {len(TOURNAMENTS)}")
    print(f"  Method      : <p> tag extraction (confirmed from debug)")
    print()

    ensure_transcript_table()
    total_saved = 0

    for tournament in tqdm(TOURNAMENTS, desc="Tournaments"):
        name  = tournament["name"]
        year  = tournament["start"][:4]
        label = f"{name} {year}"

        log.info(f"\n🎾  {label}")

        date_urls      = generate_date_urls(tournament)
        all_interviews = []

        for day_url, date_str in date_urls:
            interviews = get_interviews_from_day(day_url, date_str, tournament)
            if interviews:
                log.info(f"  {date_str}: {len(interviews)} interviews")
            all_interviews.extend(interviews)

        log.info(f"  Total for {label}: {len(all_interviews)} interviews")

        saved_this_event = 0
        for meta in tqdm(all_interviews, desc=f"  {label}", leave=False):
            record = scrape_interview(meta["url"], meta)
            if record:
                save_transcript(record)
                total_saved      += 1
                saved_this_event += 1

        log.info(f"  Saved: {saved_this_event}")

    log.info("\nLinking transcripts to player IDs…")
    link_transcripts_to_players()

    conn     = get_connection()
    n_total  = conn.execute("SELECT COUNT(*) FROM transcripts").fetchone()[0]
    n_linked = conn.execute(
        "SELECT COUNT(*) FROM transcripts WHERE player_id IS NOT NULL"
    ).fetchone()[0]
    conn.close()

    print(f"\nScraping summary ")
    print(f"  Transcripts saved this run : {total_saved:,}")
    print(f"  Total in database          : {n_total:,}")
    print(f"  Linked to player ID        : {n_linked:,}")
    print("\n✅  Done. Run nlp.py next.")


if __name__ == "__main__":
    main()