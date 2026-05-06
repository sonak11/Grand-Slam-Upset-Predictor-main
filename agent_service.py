#import libraries
import os
import re
import sqlite3
import pandas as pd

DB_PATH = "tennis_upsets.db"


# direct SQL helpers (always available) 

DEMO_STATS = {
    "total_matches": 2847,
    "total_upsets":  743,
    "upset_rate":    0.261,
    "top_upset_slam": "Wimbledon",
    "slam_breakdown": {
        "Australian Open": {"matches": 712, "upsets": 178, "rate": 0.250},
        "Roland Garros":   {"matches": 705, "upsets": 196, "rate": 0.278},
        "Wimbledon":       {"matches": 722, "upsets": 205, "rate": 0.284},
        "US Open":         {"matches": 708, "upsets": 164, "rate": 0.232},
    },
}


def _db_available():
    return os.path.exists(DB_PATH) and os.path.getsize(DB_PATH) > 10_000


def run_sql(query: str) -> pd.DataFrame | str:
    """Run a SQL query against the tennis DB. Returns DataFrame or error string."""
    if not _db_available():
        return "Database not available — running in demo mode."
    try:
        conn = sqlite3.connect(DB_PATH)
        df   = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        return f"SQL error: {e}"


def _safe_sql(query: str):
    result = run_sql(query)
    if isinstance(result, str):
        return result
    return result.to_string(index=False)


# question routing (keyword-based) 

def _classify_question(question: str) -> str:
    q = question.lower()
    if any(w in q for w in ["transcript", "said", "mention", "fatigue", "tired",
                              "cramping", "cramp", "quote", "words", "press"]):
        return "vector"
    if any(w in q for w in ["most", "least", "count", "how many", "which player",
                              "top", "rank", "win", "lost", "upset rate", "surface",
                              "wimbledon", "roland", "australian", "us open"]):
        return "sql"
    return "hybrid"


# SQL question answering 

SQL_TEMPLATES = {
    r"upset rate.*(by|per)\s+(slam|tournament)": """
        SELECT slam_name, COUNT(*) as matches,
               SUM(upset) as upsets,
               ROUND(AVG(upset)*100,1) as upset_rate_pct
        FROM matches GROUP BY slam_name ORDER BY upset_rate_pct DESC
    """,
    r"most.*upset.*favourite|lost.*favourite|favourite.*lost": """
        SELECT loser_name as player, COUNT(*) as times_lost_as_favourite
        FROM matches WHERE upset=1 AND loser_rank < winner_rank
        GROUP BY loser_name ORDER BY times_lost_as_favourite DESC LIMIT 10
    """,
    r"upset rate.*(wimbledon|grass)": """
        SELECT slam_name, ROUND(AVG(upset)*100,1) as upset_pct
        FROM matches WHERE lower(slam_name) LIKE '%wimbledon%'
        GROUP BY slam_name
    """,
    r"upset rate.*(round|early|late)": """
        SELECT round, COUNT(*) as matches,
               ROUND(AVG(upset)*100,1) as upset_rate_pct
        FROM matches GROUP BY round ORDER BY upset_rate_pct DESC
    """,
    r"biggest.*(upset|rank diff|rank difference)": """
        SELECT winner_name, loser_name, rank_diff,
               slam_name, round, tourney_date
        FROM matches WHERE upset=1
        ORDER BY rank_diff DESC LIMIT 10
    """,
    r"how many (matches|upsets)": """
        SELECT COUNT(*) as total_matches,
               SUM(upset) as total_upsets,
               ROUND(AVG(upset)*100,1) as upset_rate_pct
        FROM matches
    """,
}


def _answer_sql(question: str) -> str:
    if not _db_available():
        return _demo_sql_answer(question)
    for pattern, sql in SQL_TEMPLATES.items():
        if re.search(pattern, question.lower()):
            return _safe_sql(sql.strip())
    # generic fallback
    return _safe_sql("""
        SELECT slam_name, COUNT(*) as matches,
               ROUND(AVG(upset)*100,1) as upset_rate
        FROM matches GROUP BY slam_name
    """)


def _demo_sql_answer(question: str) -> str:
    q = question.lower()
    if "wimbledon" in q:
        s = DEMO_STATS["slam_breakdown"]["Wimbledon"]
        return f"Wimbledon (demo data): {s['upsets']} upsets in {s['matches']} matches ({s['rate']*100:.1f}%)"
    if "rate" in q and "slam" in q:
        lines = ["Upset rate by slam (demo data):"]
        for k, v in DEMO_STATS["slam_breakdown"].items():
            lines.append(f"  {k:<20} {v['rate']*100:.1f}%")
        return "\n".join(lines)
    return (f"Demo stats — {DEMO_STATS['total_matches']} matches, "
            f"{DEMO_STATS['total_upsets']} upsets ({DEMO_STATS['upset_rate']*100:.1f}% rate)")


# vector question answering 

def _answer_vector(question: str) -> tuple[str, list[dict]]:
    """Find transcript snippets relevant to the question."""
    from rag_service import search_transcripts
    q = question.lower()
    player_filter = None
    for name in ["alcaraz", "djokovic", "nadal", "federer", "sinner", "medvedev"]:
        if name in q:
            player_filter = name.title()
            break

    results = search_transcripts(
        query=question,
        player_filter=player_filter,
        n_results=4,
        upset_only=False,
    )
    return results


# langChain agent 

def _build_langchain_agent(groq_key: str):
    """
    Build a LangChain ReAct agent with SQL + vector tools.
    Returns None if dependencies unavailable.
    """
    try:
        from langchain_groq import ChatGroq
        from langchain.agents import initialize_agent, AgentType
        from langchain.tools  import Tool
        from langchain_community.utilities import SQLDatabase
        from langchain_community.agent_toolkits import SQLDatabaseToolkit
        from rag_service import search_transcripts

        llm = ChatGroq(
            api_key=groq_key,
            model_name="llama3-70b-8192",
            temperature=0.1,
        )

        tools = []

        # SQL tool
        if _db_available():
            db      = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
            toolkit = SQLDatabaseToolkit(db=db, llm=llm)
            tools  += toolkit.get_tools()

        # Vector / transcript tool
        def vector_search(q: str) -> str:
            results = search_transcripts(q, n_results=3, upset_only=False)
            if not results:
                return "No relevant transcripts found."
            lines = []
            for r in results:
                lines.append(
                    f"[{r.get('player','?')} – {r.get('tournament','?')} {r.get('round','')}] "
                    f"{'UPSET' if r.get('upset') else 'WIN'}: {r['text'][:300]}…"
                )
            return "\n\n".join(lines)

        tools.append(Tool(
            name="TranscriptSearch",
            func=vector_search,
            description=(
                "Search player press conference transcripts for fatigue signals, "
                "quotes, or emotional patterns. Input: a search query string."
            ),
        ))

        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=5,
        )
        return agent
    except Exception as e:
        print(f"[WARN] LangChain agent unavailable: {e}")
        return None


# public API 

_lc_agent = None


def answer_question(question: str, groq_key: str | None = None):
    """
    Route a natural language question to SQL, vector search, or LangChain agent.
    Returns (answer_text: str, snippets: list[dict], used_agent: bool)
    """
    global _lc_agent

    # try LangChain agent when Groq key provided
    if groq_key:
        if _lc_agent is None:
            _lc_agent = _build_langchain_agent(groq_key)
        if _lc_agent is not None:
            try:
                answer = _lc_agent.run(question)
                return answer, [], True
            except Exception as e:
                _lc_agent = None
                print(f"[WARN] Agent run failed: {e}")

    # fallback: rule-based routing
    route = _classify_question(question)

    if route == "vector":
        snippets = _answer_vector(question)
        if snippets:
            parts = [f"**{s.get('player','?')} — {s.get('tournament','?')} {s.get('round','')}** "
                     f"({'UPSET' if s.get('upset') else 'win'}): {s['text'][:350]}…"
                     for s in snippets[:3]]
            return "\n\n".join(parts), snippets, False
        return "No matching transcripts found.", [], False

    if route == "sql":
        ans = _answer_sql(question)
        return ans, [], False

    # SQL + vector
    sql_ans  = _answer_sql(question)
    snippets = _answer_vector(question)
    combo    = sql_ans
    if snippets:
        combo += "\n\n**Relevant transcript snippets:**\n"
        for s in snippets[:2]:
            combo += f"\n> *{s.get('player','?')}:* {s['text'][:250]}…"
    return combo, snippets, False
