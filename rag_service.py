"""
RAG service: builds and queries a ChromaDB vector store over transcripts.
Falls back to keyword search when no real transcript data exists.
"""

import os
import sqlite3
import re
import numpy as np

DB_PATH    = "tennis_upsets.db"
CHROMA_DIR = "./chroma_db"

_collection    = None
_embedder      = None
_demo_docs     = None   # in-memory fallback


#demo / synthetic transcript store 

DEMO_TRANSCRIPTS = [
    {
        "player": "Carlos Alcaraz",
        "tournament": "Wimbledon",
        "round": "QF",
        "upset": 1,
        "rank_diff": 15,
        "text": (
            "I'm honestly feeling very tired. Yesterday's match took a lot out of me — "
            "five sets, almost four hours. My legs are really heavy and I've been cramping. "
            "I haven't slept well and mentally I'm drained. My back is a bit stiff. "
            "It's tough to stay positive when you're this exhausted. The schedule hasn't "
            "been kind. I'm not sure I have enough left in the tank for the next round."
        ),
    },
    {
        "player": "Carlos Alcaraz",
        "tournament": "Roland Garros",
        "round": "SF",
        "upset": 1,
        "rank_diff": 22,
        "text": (
            "It was really tough physically. I had a lot of tension in my legs. "
            "I'm not 100 percent right now. The run in this tournament has been so demanding — "
            "back-to-back five-set matches. I feel a question mark about my body holding up. "
            "I don't know if I can maintain this intensity. My knee was bothering me from the third set."
        ),
    },
    {
        "player": "Carlos Alcaraz",
        "tournament": "US Open",
        "round": "R4",
        "upset": 0,
        "rank_diff": -8,
        "text": (
            "I feel great. I had a great rest day yesterday, trained well, and I'm confident "
            "going into tomorrow. My serve is clicking and I feel sharp mentally. "
            "I know what I need to do on this court. Looking forward to the challenge."
        ),
    },
    {
        "player": "Novak Djokovic",
        "tournament": "Australian Open",
        "round": "SF",
        "upset": 1,
        "rank_diff": 30,
        "text": (
            "The wrist has been bothering me since the third round. I'm taking painkillers "
            "to get through matches. Mentally it is draining because you're always thinking "
            "about the injury. The quick turnaround between matches hasn't helped. "
            "I struggled to focus in the second set. I wasn't there mentally."
        ),
    },
    {
        "player": "Novak Djokovic",
        "tournament": "Wimbledon",
        "round": "QF",
        "upset": 0,
        "rank_diff": -12,
        "text": (
            "I feel very confident. My game is in good shape, movement is great. "
            "I'm not thinking about anything other than the next match. "
            "Grass suits my game and I know this tournament very well. "
            "I'm sleeping nine hours and my body feels refreshed."
        ),
    },
    {
        "player": "Rafael Nadal",
        "tournament": "Australian Open",
        "round": "QF",
        "upset": 1,
        "rank_diff": 10,
        "text": (
            "My abs are very painful. I've had a medical timeout twice this week. "
            "I'm really not sure how much longer my body can hold up. "
            "I'm taking it one match at a time. The fatigue is real — "
            "I have played a lot of tennis this fortnight and my body is showing it. "
            "I have doubts about finishing the tournament healthy."
        ),
    },
]


# chromaDB setup 

def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def _get_collection():
    global _collection
    if _collection is not None:
        return _collection
    try:
        import chromadb
        client      = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_or_create_collection(
            "tennis_transcripts",
            metadata={"hnsw:space": "cosine"}
        )
        return _collection
    except Exception as e:
        print(f"[WARN] ChromaDB unavailable: {e}")
        return None


def index_transcripts_from_db():
    """
    Read transcripts from SQLite, chunk them, and store in ChromaDB.
    Run once; subsequent calls skip already-indexed docs.
    """
    if not os.path.exists(DB_PATH) or os.path.getsize(DB_PATH) < 10_000:
        return False

    col = _get_collection()
    if col is None:
        return False

    # skip if already indexed
    if col.count() > 10:
        return True

    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute("""
            SELECT t.id, t.player_name, t.tourney_name, t.round,
                   t.raw_text, m.upset, m.rank_diff, m.ctfi, m.slam_name
            FROM transcripts t
            LEFT JOIN matches m
              ON (t.player_name = m.winner_name OR t.player_name = m.loser_name)
            WHERE t.raw_text IS NOT NULL AND LENGTH(t.raw_text) > 200
            LIMIT 5000
        """).fetchall()
        conn.close()
    except Exception as e:
        print(f"[WARN] DB read failed: {e}")
        return False

    embedder = _get_embedder()
    batch_ids, batch_docs, batch_meta, batch_embs = [], [], [], []

    for i, row in enumerate(rows):
        (tid, player, tourney, rnd, text,
         upset, rank_diff, ctfi, slam) = row
        # chunk every 500 words
        words  = text.split()
        chunks = [" ".join(words[j:j+500]) for j in range(0, len(words), 450)]
        for k, chunk in enumerate(chunks):
            chunk_id = f"t{tid}_c{k}"
            existing = col.get(ids=[chunk_id])
            if existing["ids"]:
                continue
            batch_ids.append(chunk_id)
            batch_docs.append(chunk)
            batch_meta.append({
                "player":     player or "Unknown",
                "tournament": tourney or "",
                "round":      rnd    or "",
                "upset":      int(upset or 0),
                "rank_diff":  float(rank_diff or 0),
                "ctfi":       float(ctfi or 0),
            })
            batch_embs.append(embedder.encode(chunk).tolist())

            if len(batch_ids) >= 64:
                col.add(ids=batch_ids, documents=batch_docs,
                        metadatas=batch_meta, embeddings=batch_embs)
                batch_ids, batch_docs, batch_meta, batch_embs = [], [], [], []

    if batch_ids:
        col.add(ids=batch_ids, documents=batch_docs,
                metadatas=batch_meta, embeddings=batch_embs)
    return True


def _index_demo_docs():
    """Index demo transcripts into ChromaDB."""
    col      = _get_collection()
    embedder = _get_embedder()
    if col is None:
        return

    existing = col.get(ids=[f"demo_{i}" for i in range(len(DEMO_TRANSCRIPTS))])
    if len(existing["ids"]) == len(DEMO_TRANSCRIPTS):
        return

    ids   = [f"demo_{i}" for i in range(len(DEMO_TRANSCRIPTS))]
    texts = [d["text"] for d in DEMO_TRANSCRIPTS]
    metas = [
        {
            "player":     d["player"],
            "tournament": d["tournament"],
            "round":      d["round"],
            "upset":      d["upset"],
            "rank_diff":  float(d["rank_diff"]),
            "ctfi":       0.0,
        }
        for d in DEMO_TRANSCRIPTS
    ]
    embs  = embedder.encode(texts).tolist()
    col.upsert(ids=ids, documents=texts, metadatas=metas, embeddings=embs)


# keyword fallback 

def _keyword_search(query: str, player_filter: str | None, n: int):
    """Simple keyword-overlap fallback when embeddings unavailable."""
    docs  = DEMO_TRANSCRIPTS
    if player_filter:
        filtered = [d for d in docs if player_filter.lower() in d["player"].lower()]
        docs     = filtered if filtered else docs

    q_words = set(re.findall(r"\w+", query.lower()))
    scored  = []
    for d in docs:
        t_words = set(re.findall(r"\w+", d["text"].lower()))
        score   = len(q_words & t_words) / max(len(q_words), 1)
        scored.append((score, d))
    scored.sort(key=lambda x: -x[0])
    return [d for _, d in scored[:n]]


# public API 

def search_transcripts(query: str, player_filter: str | None = None,
                        n_results: int = 5, upset_only: bool = True):
    """
    Retrieve the most relevant transcript chunks for a query.
    Returns list of dicts: {player, tournament, round, upset, rank_diff, text}
    """
    col = _get_collection()

    # try ChromaDB
    if col is not None:
        real_indexed = index_transcripts_from_db()
        if not real_indexed:
            _index_demo_docs()

        if col.count() > 0:
            where = {}
            if player_filter:
                where["player"] = player_filter
            if upset_only:
                where["upset"] = 1

            kwargs = {"query_texts": [query], "n_results": min(n_results, col.count())}
            if where:
                kwargs["where"] = where
            try:
                results = col.query(**kwargs)
                out = []
                for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                    out.append({**meta, "text": doc})
                return out
            except Exception as e:
                print(f"[WARN] ChromaDB query failed: {e}")

    # final fallback: keyword search on demo data
    return [
        {**d, "text": d["text"]}
        for d in _keyword_search(query, player_filter, n_results)
        if (not upset_only or d.get("upset", 0) == 1)
    ]


def get_all_players() -> list[str]:
    """Return known player names for the UI dropdown."""
    col = _get_collection()
    if col is not None and col.count() > 0:
        try:
            results = col.get(include=["metadatas"])
            players = sorted({m["player"] for m in results["metadatas"]
                               if m.get("player")})
            return players
        except Exception:
            pass
    return sorted({d["player"] for d in DEMO_TRANSCRIPTS})
