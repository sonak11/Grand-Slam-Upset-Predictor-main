"""
fix_features_join.py
=====================
Patches features.py to join transcripts on player_name instead of player_id.
The transcripts table has player_id=NULL for scraped rows, so the join was
producing 0 matches. This fix uses player_name + slam_name for the merge.

Run once:
    python3.11 fix_features_join.py
    python3.11 features.py
    python3.11 model.py
"""

FEATURES_FILE = "features.py"

OLD_CODE = '''    trans_subset = transcripts[["player_id", "tourney_name", "round"] + available].copy()

    # Standardise round labels for joining
    round_map = {
        "first round": "R1", "second round": "R2", "third round": "R3",
        "fourth round": "R4", "quarterfinal": "QF", "semifinal": "SF",
        "final": "F", "r1": "R1", "r2": "R2", "r3": "R3", "r4": "R4",
    }
    trans_subset["round"] = (
        trans_subset["round"].str.lower().map(round_map).fillna(trans_subset["round"])
    )

    merged = matches.merge(
        trans_subset,
        on=["player_id", "tourney_name", "round"],
        how="left",
        suffixes=("", "_transcript"),
    )'''

NEW_CODE = '''    # Use player_name as join key (player_id is NULL for most scraped transcripts)
    trans_subset = transcripts[["player_name", "tourney_name", "round"] + available].copy()

    # Standardise round labels for joining
    round_map = {
        "first round": "R1", "second round": "R2", "third round": "R3",
        "fourth round": "R4", "quarterfinal": "QF", "semifinal": "SF",
        "final": "F", "r1": "R1", "r2": "R2", "r3": "R3", "r4": "R4",
        "unknown": None,
    }
    trans_subset["round"] = (
        trans_subset["round"].str.lower().map(round_map).fillna(trans_subset["round"])
    )

    # Normalise player names: strip whitespace, title-case
    trans_subset["player_name"] = trans_subset["player_name"].str.strip().str.title()
    matches["player_name"]      = matches["player_name"].str.strip().str.title()

    # Aggregate: if multiple transcripts per player+tourney+round, take mean of NLP cols
    trans_agg = (
        trans_subset
        .groupby(["player_name", "tourney_name", "round"], as_index=False)
        [available]
        .mean()
    )

    # Primary join: player_name + tourney_name + round
    merged = matches.merge(
        trans_agg,
        on=["player_name", "tourney_name", "round"],
        how="left",
        suffixes=("", "_transcript"),
    )

    # Fallback join on player_name + slam_name (some transcripts use slam_name not tourney_name)
    unmatched_mask = merged[available[0]].isna() if available else pd.Series(False, index=merged.index)
    if unmatched_mask.sum() > 0 and "slam_name" in trans_subset.columns:
        trans_slam = trans_subset.copy()
        trans_slam = trans_slam.rename(columns={"tourney_name": "slam_name"})
        fallback   = matches[unmatched_mask].merge(
            trans_slam.groupby(["player_name","slam_name","round"],as_index=False)[available].mean(),
            on=["player_name","slam_name","round"], how="left", suffixes=("","_t")
        )
        for col in available:
            merged.loc[unmatched_mask, col] = fallback[col].values'''

def apply_patch():
    with open(FEATURES_FILE, "r") as f:
        content = f.read()

    if OLD_CODE not in content:
        print("❌ Could not find the target code block.")
        print("   features.py may have already been patched, or the code changed.")
        print("   Check features.py manually around the merge_transcripts function.")
        return False

    new_content = content.replace(OLD_CODE, NEW_CODE)

    with open(FEATURES_FILE, "w") as f:
        f.write(new_content)

    print("✅ features.py patched successfully.")
    print("   The transcript join now uses player_name instead of player_id.")
    return True

if __name__ == "__main__":
    import sys
    success = apply_patch()
    if success:
        print("\nNow run:")
        print("    python3.11 features.py")
        print("    python3.11 model.py")
    else:
        sys.exit(1)