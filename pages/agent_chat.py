"""Page 3 — Ask the Model: beautiful chat redesign."""

import streamlit as st

EXAMPLES = [
    ("📊", "Upset rates",       "What is the upset rate at each Grand Slam?"),
    ("🏆", "Biggest upsets",    "Show the 5 biggest rank-gap upsets in the dataset"),
    ("🔴", "Favourite losses",  "Which top-10 player lost most often as a favourite?"),
    ("🎾", "Round analysis",    "Which round has the highest upset rate?"),
    ("💬", "Fatigue quotes",    "Find matches where cramping was mentioned before an upset"),
    ("🌿", "Surface stats",     "Compare upset rates on grass vs clay"),
    ("📈", "Match count",       "How many Grand Slam matches are in the dataset?"),
    ("😓", "Djokovic fatigue",  "What did Djokovic say about fatigue before upset losses?"),
]


def _fmt_answer(text: str) -> str:
    """Wrap table-like multi-line text in code block."""
    lines = text.strip().split("\n")
    if len(lines) > 3 and any(
        any(c.isdigit() for c in l) for l in lines[1:4]
    ):
        return f"```\n{text}\n```"
    return text


def render(groq_key, db_ok):
    st.markdown("""
    <div style='padding:1.5rem 0 0.5rem'>
      <p class='muted' style='text-transform:uppercase;letter-spacing:0.12em;
                               font-family:DM Mono,monospace;margin-bottom:0.3rem'>Tool 3 of 3</p>
      <h1 style='font-family:Syne,sans-serif;font-size:2rem;font-weight:800;
                  letter-spacing:-0.03em;margin:0'>Ask the <span style='color:#ff6b6b'>Model</span></h1>
      <p style='color:#6868a0;font-size:0.9rem;margin:0.4rem 0 0;line-height:1.5'>
        Chat naturally with 10 years of Grand Slam data. The AI routes your question to
        a SQL query, a transcript search, or both.
      </p>
    </div>
    """, unsafe_allow_html=True)

    if not db_ok:
        st.markdown("""
        <div style='background:#0d0d1a;border:1px solid #1e1e3a;border-radius:10px;
                    padding:0.8rem 1.2rem;margin:0.8rem 0;font-size:0.85rem;color:#5050a0'>
        ℹ️ Database not found — answering from demo statistics.
        </div>""", unsafe_allow_html=True)

    # rxample question grid 
    st.markdown("<p class='muted' style='margin:1rem 0 0.5rem;text-transform:uppercase;"
                "letter-spacing:0.1em;font-family:DM Mono,monospace;font-size:0.72rem'>"
                "Try an example</p>", unsafe_allow_html=True)

    cols = st.columns(4)
    for i, (icon, label, q) in enumerate(EXAMPLES):
        with cols[i % 4]:
            if st.button(f"{icon} {label}", key=f"ex_{i}", use_container_width=True):
                st.session_state.pending_q = q

    st.markdown("<hr style='margin:1rem 0'>", unsafe_allow_html=True)

    # conversation history 
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role":     "assistant",
            "content":  (
                "Hi! I can answer questions about Grand Slam upsets — statistics, "
                "player patterns, and what players said in press conferences.\n\n"
                "Try one of the examples above, or type your own question below."
            ),
            "snippets": [],
            "source":   "",
        }]

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f"<div class='chat-user'>{msg['content']}</div>",
                unsafe_allow_html=True,
            )
        else:
            content = _fmt_answer(msg["content"])
            source  = msg.get("source", "")
            src_tag = (
                f"<div style='font-family:DM Mono,monospace;font-size:0.68rem;"
                f"color:#3030a0;margin-top:0.5rem;text-transform:uppercase;"
                f"letter-spacing:0.08em'>{source}</div>"
                if source else ""
            )
            st.markdown(
                f"<div class='chat-bot'>{content}{src_tag}</div>",
                unsafe_allow_html=True,
            )
            if msg.get("snippets"):
                with st.expander("📑 Relevant excerpts"):
                    for s in msg["snippets"][:3]:
                        outcome = "🔴 UPSET" if s.get("upset") else "🟢 Win"
                        st.markdown(
                            f"<div class='snippet-card'>"
                            f"<div class='snippet-meta'>"
                            f"{s.get('player','?')} · {s.get('tournament','?')} "
                            f"{s.get('round','')} · {outcome}</div>"
                            f"{s['text'][:380]}…</div>",
                            unsafe_allow_html=True,
                        )

    # input 
    pending = st.session_state.pop("pending_q", None)
    user_input = st.chat_input("Ask anything about tennis upsets…")
    question   = pending or user_input

    if question:
        st.session_state.messages.append(
            {"role": "user", "content": question, "snippets": [], "source": ""}
        )
        st.markdown(
            f"<div class='chat-user'>{question}</div>",
            unsafe_allow_html=True,
        )

        with st.spinner("Thinking…"):
            from agent_service import answer_question
            answer, snippets, used_agent = answer_question(question, groq_key)

        source = (
            "⚡ LangChain agent" if used_agent
            else "🔍 Transcript search" if snippets
            else "🗄️ SQL query"
        )
        display = _fmt_answer(answer)
        src_tag = (
            f"<div style='font-family:DM Mono,monospace;font-size:0.68rem;"
            f"color:#3030a0;margin-top:0.5rem;text-transform:uppercase;"
            f"letter-spacing:0.08em'>{source}</div>"
        )
        st.markdown(
            f"<div class='chat-bot'>{display}{src_tag}</div>",
            unsafe_allow_html=True,
        )

        if snippets:
            with st.expander("📑 Relevant excerpts"):
                for s in snippets[:3]:
                    outcome = "🔴 UPSET" if s.get("upset") else "🟢 Win"
                    st.markdown(
                        f"<div class='snippet-card'>"
                        f"<div class='snippet-meta'>"
                        f"{s.get('player','?')} · {s.get('tournament','?')} "
                        f"{s.get('round','')} · {outcome}</div>"
                        f"{s['text'][:380]}…</div>",
                        unsafe_allow_html=True,
                    )

        st.session_state.messages.append({
            "role":     "assistant",
            "content":  answer,
            "snippets": snippets,
            "source":   source,
        })

    # clear 
    if len(st.session_state.get("messages", [])) > 1:
        if st.button("🗑️  Clear conversation", key="clear"):
            st.session_state.messages = []
            st.rerun()