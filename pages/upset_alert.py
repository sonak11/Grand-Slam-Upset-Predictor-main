"""Page 1 — Upset Alert: beautiful redesign."""

import streamlit as st
import numpy as np

ROUND_MAP    = {"R1": 1, "R2": 2, "R3": 3, "R4": 4, "QF": 5, "SF": 6, "F": 7}
ROUND_LABELS = list(ROUND_MAP.keys())
ROUND_FULL   = {
    "R1": "First Round", "R2": "Second Round", "R3": "Third Round",
    "R4": "Fourth Round", "QF": "Quarterfinal", "SF": "Semifinal", "F": "Final"
}

FATIGUE_EXAMPLES = [
    "tired", "heavy legs", "cramping", "mentally drained",
    "back-to-back", "five sets", "injury", "my knee",
    "not 100%", "doubt", "hard to focus",
]


def _groq_explain(prob, shap_dict, player, opponent, groq_key):
    try:
        from groq import Groq
        client = Groq(api_key=groq_key)
        top    = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        lines  = [f"  • {f.replace('_',' ')}: {v:+.3f}" for f, v in top if abs(v) > 0.005]
        prompt = (
            f"You are an elite tennis analyst. The model predicts a {prob*100:.1f}% upset "
            f"probability for {player} (underdog) vs {opponent} (favourite) at a Grand Slam.\n"
            f"Key SHAP feature contributions (positive = more likely upset):\n"
            + "\n".join(lines)
            + "\n\nWrite a clear, confident 2-sentence explanation for a general audience. "
              "Avoid jargon. Say what's driving the risk and what it means for the match."
        )
        resp = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=130,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return None


def _rule_explain(prob, shap_dict, player, opponent):
    top   = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:2]
    names = [f.replace("_", " ") for f, _ in top]
    level = "high" if prob > 0.6 else "moderate" if prob > 0.4 else "low"
    return (
        f"The model assigns a **{level}** upset risk ({prob*100:.1f}%) for {player}. "
        f"The strongest drivers are **{names[0]}** and **{names[1] if len(names)>1 else 'rank gap'}**."
    )


def _analyse_transcript(text: str) -> dict:
    import re
    text_l = text.lower()
    LEXICON = {
        "physical":   ["tired","exhausted","heavy legs","cramping","drained","sore",
                        "worn out","stiff","not 100","my body","physically"],
        "mental":     ["mentally","focus","distracted","lost focus","concentrate","head"],
        "schedule":   ["back to back","back-to-back","five sets","long match",
                        "tough schedule","no rest","played a lot","deep run"],
        "injury":     ["injury","pain","hurts","my knee","my shoulder","my back",
                        "my ankle","blister","trainer","pulled","tight"],
        "motivation": ["doubt","not sure","question mark","uncertain",
                        "lacking confidence","hard to believe"],
    }
    counts = {}
    total  = 0
    for cat, phrases in LEXICON.items():
        c = sum(len(re.findall(r"\b" + re.escape(ph) + r"\b", text_l)) for ph in phrases)
        counts[cat] = c
        total += c

    words    = text.split()
    n        = max(len(words), 1)
    pos      = sum(text_l.count(w) for w in ["confident","great","good","ready","strong","sharp"])
    neg      = sum(text_l.count(w) for w in ["not","never","tired","exhausted","pain","doubt","sore"])
    polarity = round((pos - neg) / max(pos + neg, 1), 3)

    return {
        "fatigue_total":    total,
        "fatigue_density":  round(total / n * 100, 2),
        "sentiment_polarity": polarity,
        **{f"fatigue_{k}": v for k, v in counts.items()},
        "word_count": len(words),
    }


def _shap_html(shap_dict):
    top    = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
    mx     = max(abs(v) for _, v in top) if top else 1
    html   = "<div style='margin-top:0.5rem'>"
    for feat, val in top:
        pct   = int(abs(val) / mx * 100)
        color = "#ff6b6b" if val > 0 else "#5ee7df"
        label = feat.replace("_", " ")
        sign  = "+" if val > 0 else "−"
        html += (
            f"<div class='shap-row'>"
            f"<span class='shap-label'>{label}</span>"
            f"<div class='shap-track'>"
            f"<div class='{'shap-fill-pos' if val > 0 else 'shap-fill-neg'}' "
            f"style='width:{pct}%'></div></div>"
            f"<span class='shap-val' style='color:{color}'>{sign}{abs(val):.3f}</span>"
            f"</div>"
        )
    html += "</div>"
    return html


def _prob_color(prob):
    if prob >= 0.65: return "#ff6b6b", "HIGH UPSET RISK"
    if prob >= 0.40: return "#f5a623", "MODERATE RISK"
    return "#5ee7df", "LOW RISK"


def render(groq_key, model_ok, feats_ok):
    st.markdown("""
    <div style='padding:1.5rem 0 0.5rem'>
      <p class='muted' style='text-transform:uppercase;letter-spacing:0.12em;
                               font-family:DM Mono,monospace;margin-bottom:0.3rem'>Tool 1 of 3</p>
      <h1 style='font-family:Syne,sans-serif;font-size:2rem;font-weight:800;
                  letter-spacing:-0.03em;margin:0'>Upset <span class='accent'>Alert</span></h1>
      <p style='color:#6868a0;font-size:0.9rem;margin:0.4rem 0 0;line-height:1.5'>
        Enter match details → get an AI-powered upset probability with a plain-English explanation.
      </p>
    </div>
    """, unsafe_allow_html=True)

    if not model_ok:
        st.markdown("""
        <div style='background:#1a1008;border:1px solid #f5a62333;border-radius:10px;
                    padding:0.8rem 1.2rem;margin-bottom:1rem;font-size:0.85rem;color:#a07030'>
        ⚠️ No trained model found — using demo mode. Run the pipeline scripts to train on real data.
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # player inputs 
    col1, mid, col2 = st.columns([5, 1, 5])

    with col1:
        st.markdown("<p style='font-family:Syne,sans-serif;font-weight:700;"
                    "font-size:0.85rem;color:#c8f23d;text-transform:uppercase;"
                    "letter-spacing:0.08em;margin-bottom:0.8rem'>🟢 Underdog</p>",
                    unsafe_allow_html=True)
        player      = st.text_input("Player name", "Carlos Alcaraz", key="p_name",
                                     label_visibility="collapsed",
                                     placeholder="Underdog player name")
        player_rank = st.number_input("ATP Ranking", 1, 1000, 45, key="p_rank",
                                       label_visibility="collapsed")
        st.markdown("<p class='muted'>ATP Ranking</p>", unsafe_allow_html=True)

    with mid:
        st.markdown("<div style='text-align:center;padding-top:1.5rem;"
                    "font-family:Syne,sans-serif;font-weight:800;font-size:1.2rem;"
                    "color:#2e2e5a'>vs</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<p style='font-family:Syne,sans-serif;font-weight:700;"
                    "font-size:0.85rem;color:#ff6b6b;text-transform:uppercase;"
                    "letter-spacing:0.08em;margin-bottom:0.8rem'>🔴 Favourite</p>",
                    unsafe_allow_html=True)
        opponent  = st.text_input("Opponent name", "Novak Djokovic", key="o_name",
                                   label_visibility="collapsed",
                                   placeholder="Favourite player name")
        opp_rank  = st.number_input("Opponent Ranking", 1, 1000, 3, key="o_rank",
                                     label_visibility="collapsed")
        st.markdown("<p class='muted'>ATP Ranking</p>", unsafe_allow_html=True)

    # match context 
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<p style='font-family:Syne,sans-serif;font-weight:700;font-size:0.85rem;"
                "color:#8888b0;text-transform:uppercase;letter-spacing:0.08em;"
                "margin-bottom:0.8rem'>Match Context</p>", unsafe_allow_html=True)

    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        round_sel = st.selectbox("Round", ROUND_LABELS, index=4,
                                  format_func=lambda r: f"{r} — {ROUND_FULL[r]}")
    with mc2:
        best_of = st.radio("Format", [3, 5], index=1, horizontal=True,
                            format_func=lambda x: f"Best of {x}")
    with mc3:
        ctfi = st.slider("Fatigue Index (CTFI)",
                          0, 30, 8,
                          help="Sets the underdog has already played in this tournament")

    # transcript 
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='border-top:1px solid #1e1e3a;padding-top:1.2rem;margin-bottom:0.8rem'>
      <p style='font-family:Syne,sans-serif;font-weight:700;font-size:1rem;margin:0'>
        Press conference transcript
        <span style='font-weight:400;font-size:0.8rem;color:#4040a0;margin-left:0.5rem'>optional</span>
      </p>
      <p class='muted' style='margin:0.3rem 0 0'>
        Paste the underdog's pre-match press conference text to extract fatigue signals automatically.
        The more text you add, the better the NLP features.
      </p>
    </div>
    """, unsafe_allow_html=True)

    use_transcript = st.toggle("Include transcript NLP", value=True)
    nlp_feats      = {}

    if use_transcript:
        st.markdown("<p class='muted' style='margin-bottom:0.4rem'>Quick-add fatigue signals:</p>",
                    unsafe_allow_html=True)
        quick = st.multiselect("Quick signals", FATIGUE_EXAMPLES,
                                label_visibility="collapsed")
        transcript_text = st.text_area(
            "Transcript",
            placeholder='e.g. "I\'m feeling really tired, my legs are heavy after yesterday\'s five-set match. '
                         'My back is a bit stiff but I\'ll give everything tomorrow."',
            height=130,
            label_visibility="collapsed",
        )
        full_text = " ".join(quick) + " " + (transcript_text or "")
        if full_text.strip():
            nlp_feats = _analyse_transcript(full_text)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Fatigue signals", nlp_feats["fatigue_total"])
            c2.metric("Density / 100w",  f"{nlp_feats['fatigue_density']:.1f}")
            c3.metric("Sentiment",        f"{nlp_feats['sentiment_polarity']:+.2f}")
            c4.metric("Word count",        nlp_feats["word_count"])

    # predict 
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("⚡  Predict upset probability", use_container_width=True):
        from prediction_service import predict_with_explanation

        sentiment = nlp_feats.get("sentiment_polarity", 0.0)
        fatigue   = nlp_feats.get("fatigue_total",      0)

        with st.spinner("Running model…"):
            prob, shap_dict = predict_with_explanation(
                player_rank=player_rank, opp_rank=opp_rank, ctfi=ctfi,
                sentiment_polarity=sentiment, fatigue_total=fatigue,
                round_num=ROUND_MAP[round_sel], best_of=best_of,
            )

        color, risk_label = _prob_color(prob)

        # result card
        r1, r2 = st.columns([2, 3])
        with r1:
            st.markdown(f"""
            <div class='gs-card' style='text-align:center;border-color:{color}33'>
              <div class='prob-number' style='color:{color}'>{prob*100:.0f}%</div>
              <div class='prob-label'>upset probability</div>
              <div style='margin-top:0.8rem;font-family:DM Mono,monospace;font-size:0.7rem;
                          color:{color};letter-spacing:0.12em'>{risk_label}</div>
              <div style='margin-top:0.6rem;font-size:0.8rem;color:#6868a0'>
                {player} vs {opponent}<br>{ROUND_FULL[round_sel]}
              </div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(prob)

        with r2:
            # explanation
            st.markdown("<p style='font-family:Syne,sans-serif;font-weight:700;"
                        "font-size:0.9rem;margin-bottom:0.6rem'>🧠 AI Explanation</p>",
                        unsafe_allow_html=True)
            if groq_key:
                with st.spinner("Generating explanation…"):
                    explanation = _groq_explain(prob, shap_dict, player, opponent, groq_key)
                if not explanation:
                    explanation = _rule_explain(prob, shap_dict, player, opponent)
            else:
                explanation = _rule_explain(prob, shap_dict, player, opponent)
                st.markdown("<p class='muted'>Add a Groq key for richer explanations.</p>",
                            unsafe_allow_html=True)

            st.markdown(f"""
            <div style='background:#0d0d1a;border-left:3px solid {color};border-radius:0 10px 10px 0;
                        padding:1rem 1.2rem;font-size:0.9rem;color:#c8c8e0;line-height:1.65'>
              {explanation}
            </div>
            """, unsafe_allow_html=True)

            # SHAP
            if shap_dict:
                st.markdown("<br><p style='font-family:Syne,sans-serif;font-weight:700;"
                            "font-size:0.9rem;margin-bottom:0.3rem'>📊 Feature importance</p>",
                            unsafe_allow_html=True)
                st.markdown("<p class='muted' style='margin-bottom:0.5rem'>"
                            "Red = increases upset chance · Teal = decreases it</p>",
                            unsafe_allow_html=True)
                st.markdown(_shap_html(shap_dict), unsafe_allow_html=True)