#import everything 

import re
import time
import random
import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

ASAP_BASE = "https://www.asapsports.com"
OUTPUT_JSON = "real_event_ids.json"
SCRAPING_FILE = "scraping.py"

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

# keywords to identify Grand Slam events in page text
SLAM_PATTERNS = {
    "Australian Open": [
        "australian open", "aus open"
    ],
    "Roland Garros": [
        "roland garros", "french open"
    ],
    "Wimbledon": [
        "wimbledon", "the championships"
    ],
    "US Open": [
        "us open", "u.s. open", "u.s open"
    ],
}

YEARS = list(range(2015, 2025))


def polite_get(url):
    time.sleep(random.uniform(1.5, 3.0))
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        return resp
    except Exception as e:
        print(f"  [WARN] Failed to fetch {url}: {e}")
        return None


def is_grand_slam(title: str) -> tuple[str | None, int | None]:
    """
    Returns (slam_name, year) if title matches a Grand Slam, else (None, None).
    Example: "AUSTRALIAN OPEN 2023" → ("Australian Open", 2023)
    """
    title_lower = title.lower()

    # extract year
    year_match = re.search(r"\b(20\d{2})\b", title)
    if not year_match:
        return None, None
    year = int(year_match.group(1))
    if year not in YEARS:
        return None, None

    for slam_name, patterns in SLAM_PATTERNS.items():
        if any(p in title_lower for p in patterns):
            return slam_name, year

    return None, None


def crawl_events_page(url: str) -> dict:
    """
    fetch a page and extract all show_event.php?id=X links that are Grand Slams.
    Returns dict of "Slam Year" → event_id.
    """
    found = {}
    resp  = polite_get(url)
    if resp is None:
        return found

    soup = BeautifulSoup(resp.text, "lxml")

    for a in soup.find_all("a", href=True):
        href = a["href"]
        m    = re.search(r"show_event\.php\?id=(\d+)", href)
        if not m:
            continue

        event_id    = int(m.group(1))
        link_text   = a.get_text(strip=True)

        # also check surrounding row text for more context
        row_text = link_text
        if a.parent:
            row_text = a.parent.get_text(" ", strip=True)

        slam_name, year = is_grand_slam(row_text)
        if slam_name and year:
            key = f"{slam_name} {year}"
            if key not in found:
                found[key] = event_id
                print(f"  ✅  [{event_id}]  {key}  (link: '{link_text[:50]}')")

    return found


def discover_ids() -> dict:
   
 
    all_found = {}

    # approach 1: Year-by-year event listing pages 
    # ASAP Sports has yearly archives at ?q=YEAR or category pages
    print("\n🔍  Approach 1: Browsing yearly event listing pages...")
    for year in YEARS:
        urls_to_try = [
            f"{ASAP_BASE}/show_events.php?q={year}",
            f"{ASAP_BASE}/show_events.php?year={year}",
            f"{ASAP_BASE}/show_events.php?date={year}",
            f"{ASAP_BASE}/show_events.php?cat=0&q={year}",
        ]
        for url in urls_to_try:
            results = crawl_events_page(url)
            if results:
                all_found.update(results)
                break  # found something for this year, move on

    # approach 2: ASAP main page and top-level listing 
    print("\n🔍  Approach 2: Crawling main page and direct event listing...")
    for url in [
        ASAP_BASE,
        f"{ASAP_BASE}/show_events.php",
        f"{ASAP_BASE}/show_events.php?cat=0",
        f"{ASAP_BASE}/show_events.php?type=tennis",
    ]:
        results = crawl_events_page(url)
        all_found.update(results)

    # approach 3: Search for each slam directly 
    print("\n🔍  Approach 3: Searching for each Grand Slam by name...")
    search_terms = [
        "australian open", "roland garros", "wimbledon", "us open"
    ]
    for term in search_terms:
        for year in [2023, 2024, 2022]:   # try recent years first
            urls_to_try = [
                f"{ASAP_BASE}/show_events.php?q={term.replace(' ', '+')}+{year}",
                f"{ASAP_BASE}/show_events.php?search={term.replace(' ', '+')}&year={year}",
            ]
            for url in urls_to_try:
                results = crawl_events_page(url)
                all_found.update(results)
                if results:
                    break

    return all_found


def patch_scraping_py(real_ids: dict) -> None:
    #rewite the ASAP event_IDS  dict in scraping.py with real IDs. and only includes GS events and not other tournaments 

    if not real_ids:
        print("\n⚠️  No real IDs found to patch into scraping.py")
        return

    try:
        with open(SCRAPING_FILE, "r") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"\n⚠️  {SCRAPING_FILE} not found — skipping auto-patch")
        return

    # build the new dict string
    lines = ["ASAP_EVENT_IDS = {\n"]
    for key in sorted(real_ids.keys()):
        lines.append(f'    "{key}": {real_ids[key]},\n')
    lines.append("}\n")
    new_dict = "".join(lines)

    # replace the existing ASAP_EVENT_IDS block
    # match from "ASAP_EVENT_IDS = {" to the closing "}"
    pattern = r"ASAP_EVENT_IDS\s*=\s*\{[^}]*(?:\{[^}]*\}[^}]*)?\}"
    new_content = re.sub(pattern, new_dict.rstrip(), content, flags=re.DOTALL)

    if new_content == content:
        # simpler fallback: replace everything between the dict braces
        start = content.find("ASAP_EVENT_IDS = {")
        if start != -1:
            # find the matching closing brace
            depth = 0
            end   = start
            for i, ch in enumerate(content[start:]):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = start + i + 1
                        break
            new_content = content[:start] + new_dict.rstrip() + content[end:]

    with open(SCRAPING_FILE, "w") as f:
        f.write(new_content)

    print(f"\n✅  Patched {SCRAPING_FILE} with {len(real_ids)} real event IDs")


def manual_fallback_instructions(found: dict) -> None:
  
    slams    = ["Australian Open", "Roland Garros", "Wimbledon", "US Open"]
    missing  = [
        f"{slam} {year}"
        for slam in slams
        for year in [2022, 2023, 2024]
        if f"{slam} {year}" not in found
    ]
    if not missing:
        return

    print(f"\n⚠️  Could not auto-find IDs for {len(missing)} events:")
    for m in missing:
        print(f"    • {m}")

    print("""
to find  manually:
  1. Go to https://www.asapsports.com
  2. in the search box, type e.g. "Australian Open 2023"
  3. click the result — the URL will show:
       show_event.php?id=XXXXX
  4. copy that number into real_event_ids.json like:
       "Australian Open 2023": 163421

  then run:
       python get_real_ids.py --patch-only
""")


def main():
    import sys
    patch_only = "--patch-only" in sys.argv

    if patch_only:
        # jjust load the saved JSON and patch scraping.py
        try:
            with open(OUTPUT_JSON) as f:
                real_ids = json.load(f)
            print(f"Loaded {len(real_ids)} IDs from {OUTPUT_JSON}")
        except FileNotFoundError:
            print(f"  {OUTPUT_JSON} not found. Run without --patch-only first.")
            return
    else:
        print("=" * 55)
        print("  Discovering real ASAP Sports event IDs")
        print("=" * 55)

        real_ids = discover_ids()

        print(f"\n{'='*55}")
        print(f"  Found {len(real_ids)} Grand Slam events")
        print(f"{'='*55}")

        if real_ids:
            for k, v in sorted(real_ids.items()):
                print(f"  {k:<30} → {v}")
        else:
            print("""
    Found 0 events automatically.
  
  ASAP Sports may require JavaScript or has changed their layout.
  Use the manual method below, then run:
      python get_real_ids.py --patch-only
""")

        # save regardless so user can edit manually
        with open(OUTPUT_JSON, "w") as f:
            json.dump(real_ids, f, indent=2)
        print(f"\n  Saved to {OUTPUT_JSON} (edit this file manually if needed)")

        manual_fallback_instructions(real_ids)

    # patch scraping.py
    patch_scraping_py(real_ids)

    if real_ids:
        print("\n🎾  Next steps:")
        print("     python scraping.py    ← scrape the transcripts")
        print("     python nlp.py         ← run NLP")
        print("     python features.py")
        print("     python model.py")


if __name__ == "__main__":
    main()