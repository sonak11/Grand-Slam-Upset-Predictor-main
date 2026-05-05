"""Page 2 — Scouting Report: beautiful redesign."""

import streamlit as st
import re


def _groq_report(player, query, snippets, groq_key):
    try:
        from groq import Groq
        client  = Groq(api_key=groq_key)
        context = "\n\n---\n\n".join(
            f"[{s.get('tournament','?')} · {s.get('round','')} · "
            f"{'UPSET' if s.get('upset') else 'WIN'}]\n{s['text'][:500]}"
            for s in snippets[:5]
        )
        prompt = (
            f"You are a professional tennis scout writing for a coaching team. "
            f"Based on these transcript excerpts from {player}'s matches, "
            f"answer this question and produce a scouting report:\n\n"
            f"QUESTION: {query}\n\nTRANSCRIPTS:\n{context}\n\n"
            f"Write a sharp, actionable scouting report in bullet points (max 180 words). "
            f"Focus on patterns, recurring phrases, and what they mean for match preparation."
        )
        resp = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2, max_tokens=320,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return None


def _rule_report(player, snippets):
    CATS = {
        "Physical fatigue": ["tired","exhausted","heavy legs","cramping","drained","sore","not 100"],
        "Mental fatigue":   ["mentally","focus","distracted","lost focus"],
        "Schedule burden":  ["back-to-back","five sets","long match","tough schedule"],
        "Injury concern":   ["injury","pain","knee","shoulder","back","ankle","blister"],
        "Motivation doubt": ["doubt","question mark","uncertain","not sure"],
    }
    combined = " ".join(s["text"] for s in snippets).lower()
    lines    = []
    for cat, words in CATS.items():
        hits = sum(len(re.findall(r"\b" + re.escape(w) + r"\b", combined)) for w in words)
        if hits:
            lines.append(f"• **{cat}**: {hits} mention{'s' if hits > 1 else ''} detected")

    n_upsets = sum(1 for s in snippets if s.get("upset"))
    report   = (
        f"**Pattern analysis for {player}** *(from {len(snippets)} excerpts, "
        f"{n_upsets} from upset matches)*\n\n"
        + ("\n".join(lines) if lines else "• No strong fatigue signals detected in retrieved excerpts.")
        + "\n\n*Add a Groq key for a full AI-written scouting report.*"
    )
    return report


FATIGUE_CATS = {
    "Physical":   ["tired","exhausted","heavy legs","cramping","drained","sore","not 100"],
    "Mental":     ["mentally","focus","distracted","lost focus","concentration"],
    "Schedule":   ["back-to-back","five sets","long match","tough schedule","no rest"],
    "Injury":     ["injury","pain","knee","shoulder","back","ankle","blister"],
    "Motivation": ["doubt","question mark","uncertain","not sure","hard to believe"],
}
CAT_COLORS = ["#ff6b6b", "#c8a0ff", "#5ee7df", "#f5a623", "#c8f23d"]


def render(groq_key, db_ok):
    st.markdown("""
    <div style='padding:1.5rem 0 0.5rem'>
      <p class='muted' style='text-transform:uppercase;letter-spacing:0.12em;
                               font-family:DM Mono,monospace;margin-bottom:0.3rem'>Tool 2 of 3</p>
      <h1 style='font-family:Syne,sans-serif;font-size:2rem;font-weight:800;
                  letter-spacing:-0.03em;margin:0'>Player <span style='color:#5ee7df'>Scouting</span></h1>
      <p style='color:#6868a0;font-size:0.9rem;margin:0.4rem 0 0;line-height:1.5'>
        Ask about a player's fatigue history. The AI searches real press conference transcripts
        and writes a scouting report.
      </p>
    </div>
    """, unsafe_allow_html=True)

    if not db_ok:
        st.markdown("""
        <div style='background:#0d0d1a;border:1px solid #1e1e3a;border-radius:10px;
                    padding:0.8rem 1.2rem;margin:0.8rem 0;font-size:0.85rem;color:#5050a0'>
        ℹ️ Using demo transcripts — run the full pipeline for real player data.
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    from rag_service import get_all_players
    players = get_all_players()

    col1, col2 = st.columns([1, 2])
    with col1:
        if players:
            player = st.selectbox("Select a player", players,
                                   help="Players with transcript data")
        else:
            player = st.text_input("Player name", "Carlos Alcaraz")
    with col2:
        query = st.text_input(
            "What do you want to know?",
            "What are this player's fatigue signals before upset losses?",
            help="Ask anything about their physical state, mental patterns, or match history",
        )

    adv1, adv2 = st.columns(2)
    with adv1:
        n_results  = st.slider("Excerpts to retrieve", 2, 10, 5)
    with adv2:
        upset_only = st.toggle("Only upset matches", value=True,
                                help="Focus on matches where this player lost as favourite")

    if st.button("📋  Generate scouting report", use_container_width=True):
        from rag_service import search_transcripts
        with st.spinner("Searching transcripts…"):
            snippets = search_transcripts(
                query=query, player_filter=player,
                n_results=n_results, upset_only=upset_only,
            )

        if not snippets:
            st.markdown("""
            <div style='background:#1a0808;border:1px solid #ff6b6b33;border-radius:10px;
                        padding:1rem 1.2rem;color:#a06060;font-size:0.9rem'>
            No matching transcripts found. Try turning off "Only upset matches" or
            searching for a different player.
            </div>""", unsafe_allow_html=True)
            return

        st.markdown(f"<p class='muted' style='margin:0.8rem 0'>✅ Retrieved "
                    f"<strong style='color:#e8e8f0'>{len(snippets)}</strong> relevant excerpts</p>",
                    unsafe_allow_html=True)

        # Report
        with st.spinner("Writing scouting report…"):
            if groq_key:
                report = _groq_report(player, query, snippets, groq_key)
                if not report:
                    report = _rule_report(player, snippets)
            else:
                report = _rule_report(player, snippets)

        st.markdown("<div class='gs-card'>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-family:Syne,sans-serif;font-weight:700;"
                    f"font-size:1rem;color:#5ee7df;margin-bottom:0.8rem'>"
                    f"📄 Scouting Report — {player}</p>", unsafe_allow_html=True)
        st.markdown(report)
        st.markdown("</div>", unsafe_allow_html=True)

        # Signal breakdown bar chart
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<p style='font-family:Syne,sans-serif;font-weight:700;"
                    "font-size:0.9rem;margin-bottom:0.8rem'>📊 Fatigue signal breakdown</p>",
                    unsafe_allow_html=True)

        combined = " ".join(s["text"] for s in snippets).lower()
        counts   = {
            cat: sum(len(re.findall(r"\b" + re.escape(w) + r"\b", combined)) for w in words)
            for cat, words in FATIGUE_CATS.items()
        }
        max_val  = max(counts.values()) if max(counts.values()) > 0 else 1

        for (cat, val), color in zip(counts.items(), CAT_COLORS):
            pct = int(val / max_val * 100)
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:12px;margin:6px 0'>"
                f"<span style='width:90px;font-size:0.78rem;color:#6868a0;"
                f"font-family:DM Mono,monospace;flex-shrink:0'>{cat}</span>"
                f"<div style='flex:1;background:#1a1a2e;border-radius:4px;height:14px'>"
                f"<div style='width:{pct}%;background:{color};height:14px;"
                f"border-radius:4px'></div></div>"
                f"<span style='color:{color};font-family:DM Mono,monospace;"
                f"font-size:0.78rem;width:24px;flex-shrink:0'>{val}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # Summary metrics
        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        n_upsets  = sum(1 for s in snippets if s.get("upset"))
        avg_rdiff = sum(s.get("rank_diff", 0) for s in snippets) / len(snippets)
        m1.metric("Excerpts found", len(snippets))
        m2.metric("From upsets",    n_upsets)
        m3.metric("Avg rank gap",   f"{avg_rdiff:.0f}")
        m4.metric("Total signals",  sum(counts.values()))

        # Raw excerpts
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📑 View all retrieved excerpts"):
            for i, s in enumerate(snippets, 1):
                outcome = "🔴 UPSET" if s.get("upset") else "🟢 Win"
                st.markdown(
                    f"<div class='snippet-card'>"
                    f"<div class='snippet-meta'>{i} · {s.get('player','?')} · "
                    f"{s.get('tournament','?')} {s.get('round','')} · {outcome}</div>"
                    f"{s['text'][:500]}…</div>",
                    unsafe_allow_html=True,
                )