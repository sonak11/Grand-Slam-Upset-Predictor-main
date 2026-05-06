
import os
import re
import json
import sqlite3
import logging
import time
from typing import Optional
import requests
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DB_PATH     = "tennis_upsets.db"
NLP_CSV_OUT = "nlp_features.csv"

# fatigue lexicon 
#
#  domain-specific word list: words/phrases strongly associated with physical or mental fatigue in an athlete's press conference speech.
#  grouped by category so we can create sub-scores too.

FATIGUE_LEXICON: dict[str, list[str]] = {
    "physical_fatigue": [
        "tired", "exhausted", "fatigue", "fatigued", "heavy legs",
        "body is sore", "sore body", "aching", "drained", "burnt out",
        "run down", "worn out", "stiff", "cramping", "cramps",
        "physically tough", "my body", "not 100", "not 100%",
        "not feeling great", "feeling the effects",
    ],
    "mental_fatigue": [
        "mentally drained", "mentally tough", "mentally tired",
        "mentally exhausting", "struggled to focus", "lost focus",
        "hard to concentrate", "couldn't focus", "distracted",
        "not mentally there", "my head wasn't",
    ],
    "schedule_burden": [
        "back to back", "back-to-back", "quick turnaround", "long match",
        "three sets", "five sets", "played a lot", "deep run",
        "many matches", "tough schedule", "played yesterday",
        "haven't had much rest", "tight schedule", "no rest",
        "tough week", "lot of matches",
    ],
    "injury_concern": [
        "injury", "injured", "pain", "painful", "hurts", "it hurts",
        "my knee", "my shoulder", "my back", "my wrist", "my ankle",
        "rolled my", "pulled my", "tight hamstring", "blister",
        "medical timeout", "trainer", "strapping", "hurt", "stomach", "my abs" ,
    ],
    "motivation_doubt": [
        "not sure", "doubt", "doubts", "difficult to stay motivated",
        "hard to believe", "not confident", "lacking confidence",
        "don't know if", "question mark", "uncertain", "hard to be positive",
    ],
}

# flatten for quick counting (keep category breakdown too)
ALL_FATIGUE_PHRASES = [
    phrase
    for phrases in FATIGUE_LEXICON.values()
    for phrase in phrases
]


# sentiment model (DistilBERT) 
_sentiment_pipeline = None  # lazy-load


def get_sentiment_pipeline():
    """Load the DistilBERT sentiment pipeline once and cache it."""
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        log.info("Loading DistilBERT sentiment model (first run may download ~250 MB) …")
        from transformers import pipeline  # import here so startup is fast
        _sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            truncation=True,
            max_length=512,
        )
        log.info("Sentiment model loaded.")
    return _sentiment_pipeline


def compute_sentiment(text: str) -> dict:
    """
    Return sentiment label and confidence for the first 512 tokens of text.
    For longer transcripts we chunk and average.

    Returns:
        {"label": "NEGATIVE"|"POSITIVE", "score": float, "polarity": float}
        polarity is in [-1, +1]  (NEGATIVE = negative, POSITIVE = positive)
    """
    if not text or len(text.strip()) < 10:
        return {"label": "NEUTRAL", "score": 0.5, "polarity": 0.0}

    nlp = get_sentiment_pipeline()

    # split into abut 400-token chunks (rough: 400 words ≈ 500 tokens)
    words  = text.split()
    chunks = [
        " ".join(words[i : i + 400])
        for i in range(0, min(len(words), 1600), 400)  # max 4 chunks
    ]

    scores = []
    for chunk in chunks:
        result = nlp(chunk)[0]
        sign   = 1 if result["label"] == "POSITIVE" else -1
        scores.append(sign * result["score"])

    polarity     = sum(scores) / len(scores)
    label        = "POSITIVE" if polarity > 0 else "NEGATIVE"
    confidence   = abs(polarity)

    return {"label": label, "score": confidence, "polarity": round(polarity, 4)}


# fatigue keyword scoring 

def count_fatigue_keywords(text: str) -> dict:
    """
    Count occurrences of fatigue-related keywords/phrases in text.
    Returns totals by category plus an overall score.
    """
    text_lower = text.lower()
    counts: dict[str, int] = {}

    total = 0
    for category, phrases in FATIGUE_LEXICON.items():
        cat_count = sum(
            len(re.findall(r"\b" + re.escape(ph) + r"\b", text_lower))
            for ph in phrases
        )
        counts[f"fatigue_{category}"] = cat_count
        total += cat_count

    counts["fatigue_total"]       = total
    counts["fatigue_word_density"] = round(
        total / max(len(text.split()), 1) * 100, 4
    )  # fatigue words per 100 words

    return counts


# spaCy preprocessing 

_spacy_nlp = None


def get_spacy():
    global _spacy_nlp
    if _spacy_nlp is None:
        import spacy
        _spacy_nlp = spacy.load("en_core_web_sm", disable=["ner"])
    return _spacy_nlp


def extract_text_features(text: str) -> dict:
    """
    extract higher-level linguistic features:
      - avg sentence length
      - type-token ratio (lexical diversity)
      - first-person pronoun rate  (I, me, my, myself)
      - negation rate              (not, never, can't, won't …)
    """
    if not text or len(text) < 20:
        return {
            "avg_sentence_len": 0.0,
            "type_token_ratio": 0.0,
            "first_person_rate": 0.0,
            "negation_rate": 0.0,
            "word_count": 0,
        }

    nlp   = get_spacy()
    doc   = nlp(text[:50000])  # cap at 50k chars for speed
    sents = list(doc.sents)
    tokens = [t for t in doc if not t.is_space]
    words  = [t for t in tokens if t.is_alpha]

    if not words:
        return {
            "avg_sentence_len":  0.0,
            "type_token_ratio":  0.0,
            "first_person_rate": 0.0,
            "negation_rate":     0.0,
            "word_count":        0,
        }

    fp_pronouns = {"i", "me", "my", "myself", "we", "our"}
    neg_words   = {"not", "never", "no", "cannot", "can't", "won't",
                   "don't", "didn't", "isn't", "wasn't", "couldn't"}

    fp_count  = sum(1 for w in words if w.lower_ in fp_pronouns)
    neg_count = sum(1 for w in words if w.lower_ in neg_words)

    return {
        "avg_sentence_len":  round(len(words) / max(len(sents), 1), 2),
        "type_token_ratio":  round(
            len({w.lower_ for w in words}) / len(words), 4
        ),
        "first_person_rate": round(fp_count / len(words), 4),
        "negation_rate":     round(neg_count / len(words), 4),
        "word_count":        len(words),
    }


# mistral zero-shot labelling 


HF_API_URL = (
    "https://api-inference.huggingface.co/models/"
    "mistralai/Mistral-7B-Instruct-v0.2"
)

FATIGUE_PROMPT_TEMPLATE = """<s>[INST]
You are a sports psychology researcher analysing tennis press conference transcripts.
Read the following excerpt and decide whether the player shows signs of physical or mental fatigue.

Transcript excerpt (first 800 words):

{excerpt}


Answer with a JSON object only, no other text. Use this exact format:
{{"fatigue_label": "FATIGUED" or "NOT_FATIGUED", "confidence": 0.0 to 1.0, "reason": "one sentence"}}
[/INST]
"""


def zero_shot_fatigue_label(text: str, api_token: Optional[str] = None) -> dict:
    """
    Call Mistral-7B via HF Inference API for zero-shot fatigue classification.
    Falls back gracefully if the API is unavailable or rate-limited.
    """
    token = api_token or os.getenv("HF_API_TOKEN")
    if not token:
        return {
            "llm_fatigue_label":      "UNAVAILABLE",
            "llm_fatigue_confidence": None,
            "llm_reason":             "HF_API_TOKEN not set",
        }

    excerpt = " ".join(text.split()[:800])
    prompt  = FATIGUE_PROMPT_TEMPLATE.format(excerpt=excerpt)

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type":  "application/json",
    }
    payload = {
        "inputs":      prompt,
        "parameters":  {"max_new_tokens": 80, "temperature": 0.1, "return_full_text": False},
    }

    for attempt in range(3):
        try:
            resp = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)
            if resp.status_code == 503:
                log.info("  HF model loading … waiting 20s")
                time.sleep(20)
                continue
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 60))
                log.info(f"  HF rate-limited, waiting {wait}s …")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            raw = resp.json()

            # The API returns a list of generation dicts
            generated = raw[0]["generated_text"] if isinstance(raw, list) else str(raw)
            parsed    = _parse_llm_json(generated)
            return parsed

        except Exception as exc:
            log.warning(f"  LLM call attempt {attempt+1} failed: {exc}")
            time.sleep(5)

    return {
        "llm_fatigue_label":      "ERROR",
        "llm_fatigue_confidence": None,
        "llm_reason":             "API error after retries",
    }


def _parse_llm_json(text: str) -> dict:
    """Extract JSON from LLM output (handles stray text around the JSON)."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {
            "llm_fatigue_label":      "PARSE_ERROR",
            "llm_fatigue_confidence": None,
            "llm_reason":             text[:100],
        }
    try:
        data = json.loads(m.group())
        return {
            "llm_fatigue_label":      data.get("fatigue_label", "UNKNOWN"),
            "llm_fatigue_confidence": float(data.get("confidence", 0)),
            "llm_reason":             data.get("reason", ""),
        }
    except json.JSONDecodeError:
        return {
            "llm_fatigue_label":      "PARSE_ERROR",
            "llm_fatigue_confidence": None,
            "llm_reason":             m.group()[:100],
        }


# database update
def ensure_nlp_columns() -> None:
    """Add NLP result columns to transcripts table if they don't exist."""
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.execute("PRAGMA table_info(transcripts)")
    existing = {row[1] for row in cursor.fetchall()}

    new_cols = {
        "sentiment_label":        "TEXT",
        "sentiment_score":        "REAL",
        "sentiment_polarity":     "REAL",
        "fatigue_total":          "INTEGER",
        "fatigue_word_density":   "REAL",
        "fatigue_physical":       "INTEGER",
        "fatigue_mental":         "INTEGER",
        "fatigue_schedule":       "INTEGER",
        "fatigue_injury":         "INTEGER",
        "fatigue_motivation":     "INTEGER",
        "avg_sentence_len":       "REAL",
        "type_token_ratio":       "REAL",
        "first_person_rate":      "REAL",
        "negation_rate":          "REAL",
        "word_count":             "INTEGER",
        "llm_fatigue_label":      "TEXT",
        "llm_fatigue_confidence": "REAL",
        "llm_reason":             "TEXT",
        "nlp_processed":          "INTEGER DEFAULT 0",
    }
    for col, dtype in new_cols.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE transcripts ADD COLUMN {col} {dtype}")

    conn.commit()
    conn.close()


def process_transcript_row(row: dict, use_llm: bool = False) -> dict:
    """Run all NLP steps on a single transcript row. Return update dict."""
    text = row["raw_text"] or ""

    sentiment    = compute_sentiment(text)
    fatigue      = count_fatigue_keywords(text)
    text_feats   = extract_text_features(text)
    llm_result   = zero_shot_fatigue_label(text) if use_llm else {
        "llm_fatigue_label": None,
        "llm_fatigue_confidence": None,
        "llm_reason": None,
    }

    return {
        "id": row["id"],
        # sentiment
        "sentiment_label":        sentiment["label"],
        "sentiment_score":        sentiment["score"],
        "sentiment_polarity":     sentiment["polarity"],
        # fatigue counts
        "fatigue_total":          fatigue["fatigue_total"],
        "fatigue_word_density":   fatigue["fatigue_word_density"],
        "fatigue_physical":       fatigue.get("fatigue_physical_fatigue", 0),
        "fatigue_mental":         fatigue.get("fatigue_mental_fatigue", 0),
        "fatigue_schedule":       fatigue.get("fatigue_schedule_burden", 0),
        "fatigue_injury":         fatigue.get("fatigue_injury_concern", 0),
        "fatigue_motivation":     fatigue.get("fatigue_motivation_doubt", 0),
        # linguistic features
        "avg_sentence_len":       text_feats["avg_sentence_len"],
        "type_token_ratio":       text_feats["type_token_ratio"],
        "first_person_rate":      text_feats["first_person_rate"],
        "negation_rate":          text_feats["negation_rate"],
        "word_count":             text_feats["word_count"],
        # LLM
        "llm_fatigue_label":      llm_result["llm_fatigue_label"],
        "llm_fatigue_confidence": llm_result["llm_fatigue_confidence"],
        "llm_reason":             llm_result.get("llm_reason"),
        "nlp_processed":          1,
    }


def update_transcript_row(conn: sqlite3.Connection, updates: dict) -> None:
    cols    = [k for k in updates.keys() if k != "id"]
    placeholders = ", ".join(f"{c}=?" for c in cols)
    values  = [updates[c] for c in cols] + [updates["id"]]
    conn.execute(f"UPDATE transcripts SET {placeholders} WHERE id=?", values)


# demo data generator (if no real transcripts exist yet) 

DEMO_TRANSCRIPTS = [
    {
        "id": -1,
        "raw_text": """
        I'm honestly feeling quite tired today. The match yesterday was incredibly
        long — five sets, nearly four hours. My legs are really heavy and I've been
        cramping since last night. I haven't slept well, and mentally I'm drained.
        I'm not sure I have enough left for the next round. My back is a bit stiff too.
        It's tough to stay positive when you're this exhausted. I'll try my best but
        the schedule hasn't been kind to us.
        """,
        "player_name": "Demo Player A",
    },
    {
        "id": -2,
        "raw_text": """
        I feel great, honestly. I had a good rest yesterday, ate well, slept for
        nine hours. I'm feeling very confident going into tomorrow. My serve is
        working really well in practice today, and I feel sharp mentally.
        I've been on this court many times and I know exactly what I need to do.
        Looking forward to the challenge.
        """,
        "player_name": "Demo Player B",
    },
]


# main 

def main(use_llm: bool = False, demo_mode: bool = False) -> None:
    print("=" * 60)
    print("  PART 3 — NLP Processing")
    print("=" * 60)

    if demo_mode:
        print("\n[DEMO MODE] Running on synthetic transcripts …\n")
        rows = DEMO_TRANSCRIPTS
    else:
        ensure_nlp_columns()
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute(
            "SELECT id, raw_text, player_name FROM transcripts WHERE nlp_processed=0"
        ).fetchall()
        conn.close()
        rows = [{"id": r[0], "raw_text": r[1], "player_name": r[2]} for r in rows]
        print(f"Transcripts to process: {len(rows)}")

    if not rows:
        print("No unprocessed transcripts found. (Run part2_scraping.py first, or use --demo)")
        return

    results = []
    conn    = sqlite3.connect(DB_PATH) if not demo_mode else None

    for row in tqdm(rows, desc="Processing transcripts"):
        updates = process_transcript_row(row, use_llm=use_llm)

        if conn and row["id"] > 0:
            update_transcript_row(conn, updates)
            conn.commit()

        # pretty-print demo results
        if demo_mode:
            print(f"\n  Player  : {row['player_name']}")
            print(f"  Sentiment: {updates['sentiment_label']} (polarity={updates['sentiment_polarity']:.3f})")
            print(f"  Fatigue total      : {updates['fatigue_total']}")
            print(f"  Fatigue density    : {updates['fatigue_word_density']:.2f} per 100 words")
            print(f"    Physical  : {updates['fatigue_physical']}")
            print(f"    Mental    : {updates['fatigue_mental']}")
            print(f"    Schedule  : {updates['fatigue_schedule']}")
            print(f"    Injury    : {updates['fatigue_injury']}")
            print(f"  Negation rate      : {updates['negation_rate']:.4f}")
            print(f"  First-person rate  : {updates['first_person_rate']:.4f}")
            if updates.get("llm_fatigue_label"):
                print(f"  LLM label : {updates['llm_fatigue_label']} ({updates['llm_fatigue_confidence']})")
                print(f"  LLM reason: {updates['llm_reason']}")

        results.append(updates)

    if conn:
        conn.close()

    # export to CSV for inspection
    df_out = pd.DataFrame(results)
    df_out.drop(columns=["id"], errors="ignore").to_csv(NLP_CSV_OUT, index=False)
    print(f"\nNLP features exported to: {NLP_CSV_OUT}")
    print("\nPart 3 complete. Run part4_features.py next.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NLP processing for tennis transcripts")
    parser.add_argument("--llm",  action="store_true", help="Enable Mistral zero-shot labelling")
    parser.add_argument("--demo", action="store_true", help="Run on demo transcripts only")
    args = parser.parse_args()

    main(use_llm=args.llm, demo_mode=args.demo)