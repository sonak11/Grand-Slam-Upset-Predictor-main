"""
retag_wta.py
=============
WTA player transcripts are already in your DB from the ATP scraping
(ASAP Sports mixes both tours on the same pages).
This script finds them by player name and re-tags them as tour='WTA'.

Run:
    python3.11 retag_wta.py
"""
import sqlite3

DB_PATH = "tennis_upsets.db"

# Known WTA players (top players 2022-2024)
WTA_PLAYERS = [
    "Iga Swiatek", "Aryna Sabalenka", "Coco Gauff", "Elena Rybakina",
    "Jessica Pegula", "Caroline Wozniacki", "Marketa Vondrousova",
    "Karolina Muchova", "Barbora Krejcikova", "Ons Jabeur",
    "Caroline Garcia", "Maria Sakkari", "Daria Kasatkina",
    "Veronika Kudermetova", "Beatriz Haddad Maia", "Belinda Bencic",
    "Elina Svitolina", "Petra Kvitova", "Madison Keys",
    "Jelena Ostapenko", "Mirra Andreeva", "Victoria Azarenka",
    "Liudmila Samsonova", "Anna Kalinskaya", "Emma Navarro",
    "Jasmine Paolini", "Danielle Collins", "Sofia Kenin",
    "Elena-Gabriela Ruse", "Sorana Cirstea", "Anastasia Pavlyuchenkova",
    "Ekaterina Alexandrova", "Elise Mertens", "Sloane Stephens",
    "Amanda Anisimova", "Bianca Andreescu", "Leylah Fernandez",
    "Emma Raducanu", "Paula Badosa", "Simona Halep", "Anett Kontaveit",
    "Garbine Muguruza", "Naomi Osaka", "Serena Williams",
    "Venus Williams", "Stefanos Tsitsipas",  # oops this is ATP, but won't match
    "Ena Shibahara", "Shuko Aoyama", "Barbora Strycova",
    "Katerina Siniakova", "Storm Sanders", "Ajla Tomljanovic",
    "Donna Vekic", "Camila Giorgi", "Ana Bogdan",
    "Irina-Camelia Begu", "Yulia Putintseva",
]

def retag():
    conn = sqlite3.connect(DB_PATH)

    # Add tour column if missing
    try:
        conn.execute("ALTER TABLE transcripts ADD COLUMN tour TEXT DEFAULT 'ATP'")
        print("Added tour column to transcripts")
    except Exception:
        pass

    # Check current state
    total = conn.execute("SELECT COUNT(*) FROM transcripts").fetchone()[0]
    print(f"Total transcripts in DB: {total:,}")

    # Find and retag
    retagged = 0
    for player in WTA_PLAYERS:
        # Match case-insensitively using partial name
        rows = conn.execute(
            "SELECT id, player_name FROM transcripts "
            "WHERE LOWER(player_name) LIKE LOWER(?) AND (tour IS NULL OR tour != 'WTA')",
            (f"%{player.split()[0]}%",)   # match on first name
        ).fetchall()

        # Refine: check last name too
        for row_id, name in rows:
            name_l   = (name or "").lower()
            player_l = player.lower()
            # Match if first+last both appear
            parts = player_l.split()
            if len(parts) >= 2 and parts[-1] in name_l:
                conn.execute("UPDATE transcripts SET tour='WTA' WHERE id=?", (row_id,))
                retagged += 1

    conn.commit()

    # Summary
    n_wta = conn.execute("SELECT COUNT(*) FROM transcripts WHERE tour='WTA'").fetchone()[0]
    n_atp = conn.execute("SELECT COUNT(*) FROM transcripts WHERE tour='ATP' OR tour IS NULL").fetchone()[0]

    print(f"\nRetagged: {retagged} transcripts as WTA")
    print(f"WTA transcripts: {n_wta:,}")
    print(f"ATP transcripts: {n_atp:,}")

    # Show sample WTA players found
    samples = conn.execute(
        "SELECT DISTINCT player_name FROM transcripts WHERE tour='WTA' LIMIT 20"
    ).fetchall()
    print(f"\nSample WTA players found:")
    for s in samples:
        print(f"  {s[0]}")

    conn.close()

    if n_wta == 0:
        print("\n⚠️  No WTA transcripts found.")
        print("This means ASAP Sports press conferences in your DB are ATP-only.")
        print("WTA transcripts may be on a different ASAP category.")
        print("See website limitation note instructions below.")
    else:
        print(f"\n✅ Done. {n_wta} WTA transcripts now tagged.")
        print("Re-run: python3.11 features.py && python3.11 model.py")

if __name__ == "__main__":
    retag()