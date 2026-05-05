"""
run this to diagnose exactly why transcripts aren't saving.
It fetches one real day page, gets the first interview link,
fetches that, and prints exactly what HTML elements are found
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time

ASAP_BASE = "https://www.asapsports.com"

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

# step 1: fetch a known day page and get first interview link 

DAY_URL = "http://www.asapsports.com/show_event.php?category=7&date=2023-1-29&title=AUSTRALIAN+OPEN"

print("=" * 60)
print("Step 1: Fetching day page")
print(f"  URL: {DAY_URL}")
print("=" * 60)

time.sleep(2)
resp = requests.get(DAY_URL, headers=HEADERS, timeout=20)
print(f"  Status: {resp.status_code}")
print(f"  Content length: {len(resp.text)} chars")

soup = BeautifulSoup(resp.text, "lxml")

# find all interview links
interview_links = []
for a in soup.find_all("a", href=True):
    if "show_interview.php" in a["href"]:
        full_url = urljoin(ASAP_BASE, a["href"])
        interview_links.append((full_url, a.get_text(strip=True)))

print(f"\n  Interview links found: {len(interview_links)}")
for url, name in interview_links[:5]:
    print(f"    [{name}] {url}")

if not interview_links:
    print("\n❌ No interview links found on day page!")
    print("   Raw HTML snippet:")
    print(resp.text[:2000])
    exit()

# step 2: fetch the first interview page 

interview_url, player_name = interview_links[0]
print(f"\n{'='*60}")
print(f"Step 2: Fetching interview page for: {player_name}")
print(f"  URL: {interview_url}")
print("=" * 60)

time.sleep(2)
resp2 = requests.get(interview_url, headers=HEADERS, timeout=20)
print(f"  Status: {resp2.status_code}")
print(f"  Content length: {len(resp2.text)} chars")

soup2 = BeautifulSoup(resp2.text, "lxml")

# step 3: check what HTML elements exist 

print(f"\n{'='*60}")
print("Step 3: Checking HTML elements")
print("=" * 60)

pre_tags  = soup2.find_all("pre")
div_tags  = soup2.find_all("div")
td_tags   = soup2.find_all("td")
p_tags    = soup2.find_all("p")

print(f"  <pre> tags  : {len(pre_tags)}")
print(f"  <div> tags  : {len(div_tags)}")
print(f"  <td> tags   : {len(td_tags)}")
print(f"  <p> tags    : {len(p_tags)}")

# show the longest content from each tag type
print(f"\n{'='*60}")
print("Step 4: Longest text found per tag type")
print("=" * 60)

for tag_name, tags in [("<pre>", pre_tags), ("<div>", div_tags),
                        ("<td>", td_tags),  ("<p>", p_tags)]:
    if tags:
        best = max(tags, key=lambda t: len(t.get_text()))
        text = best.get_text().strip()
        print(f"\n  {tag_name} longest ({len(text)} chars):")
        print(f"  {repr(text[:300])}")
    else:
        print(f"\n  {tag_name}: none found")

# step 4: show raw HTML around "THE MODERATOR" or "Q." 

print(f"\n{'='*60}")
print("Step 5: Looking for transcript markers in raw HTML")
print("=" * 60)

raw = resp2.text
for marker in ["THE MODERATOR", "Press Conference", "Q.", "DJOKOVIC", "NOVAK"]:
    idx = raw.find(marker)
    if idx >= 0:
        print(f"\n  Found '{marker}' at position {idx}")
        print(f"  Context: {repr(raw[max(0,idx-100):idx+200])}")
        break
else:
    print("  No transcript markers found in raw HTML")
    print("\n  First 3000 chars of raw HTML:")
    print(raw[:3000])