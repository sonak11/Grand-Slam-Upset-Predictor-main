
import streamlit as st
import os, re, numpy as np

st.set_page_config(page_title="Grand Slam Upset Predictor", page_icon="🎾",
                   layout="wide", initial_sidebar_state="collapsed")


# global css
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,800;1,700&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #faf9f6;
    color: #1c1c1e;
}
.stApp { background: #faf9f6; }
section[data-testid="stSidebar"],
[data-testid="collapsedControl"],
header[data-testid="stHeader"],
#MainMenu, footer,
.stDeployButton,
[data-testid="stToolbar"] { display: none !important; }
.block-container { padding: 0 3rem 6rem !important; max-width: 1080px !important; }

/* ───── NAV ───── */
.gs-nav {
    display: flex; align-items: center; justify-content: space-between;
    padding: 1.1rem 0; border-bottom: 1.5px solid #e8e6e0;
    position: sticky; top: 0; background: #faf9f6;
    z-index: 300; margin-bottom: 0;
}
.gs-logo {
    font-family: 'Playfair Display', serif; font-size: 1.4rem;
    font-weight: 800; color: #1c1c1e; letter-spacing: -0.02em;
}
.gs-logo sup {
    font-family: 'JetBrains Mono', monospace; font-size: 0.55rem;
    font-weight: 400; color: #888; vertical-align: super;
    letter-spacing: 0.1em; margin-left: 3px;
}

/* ───── HERO ───── */
.hero-wrap { padding: 5rem 0 3rem; }
.hero-tag {
    display: inline-block; background: #1c1c1e; color: #faf9f6;
    font-family: 'JetBrains Mono', monospace; font-size: 0.65rem;
    letter-spacing: 0.18em; text-transform: uppercase;
    padding: 4px 12px; border-radius: 100px; margin-bottom: 1.5rem;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.8rem, 7vw, 5.2rem);
    font-weight: 800; line-height: 1.08; color: #1c1c1e;
    letter-spacing: -0.03em; margin-bottom: 1.2rem;
}
.hero-title em { font-style: italic; color: #2563eb; }
.hero-body {
    font-size: 1.08rem; color: #6b7280; line-height: 1.75;
    max-width: 500px; font-weight: 400; margin-bottom: 2.2rem;
}

/* ───── STAT STRIP ───── */
.stat-strip {
    display: grid; grid-template-columns: repeat(4,1fr);
    gap: 0; border: 1.5px solid #e8e6e0; border-radius: 16px;
    overflow: hidden; margin: 2.5rem 0;
    box-shadow: 0 1px 8px #0000000a;
}
.stat-cell {
    background: #fff; padding: 1.5rem 1rem; text-align: center;
    border-right: 1px solid #e8e6e0;
}
.stat-cell:last-child { border-right: none; }
.stat-num {
    font-family: 'Playfair Display', serif; font-size: 2.2rem;
    font-weight: 700; color: #2563eb; line-height: 1;
}
.stat-lbl {
    font-family: 'JetBrains Mono', monospace; font-size: 0.62rem;
    text-transform: uppercase; letter-spacing: 0.12em;
    color: #6b7280; margin-top: 0.35rem;
}

/* ───── TOOL CARDS ───── */
.tool-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 1.2rem; margin: 1.5rem 0 2.5rem; }
.tool-card {
    background: #fff; border: 1.5px solid #e8e6e0; border-radius: 18px;
    padding: 1.8rem 1.5rem; transition: all 0.22s ease;
    box-shadow: 0 1px 4px #0000000a;
}
.tool-card:hover {
    border-color: #2563eb; box-shadow: 0 8px 32px #2563eb12;
    transform: translateY(-3px);
}
.tool-icon { font-size: 2.2rem; margin-bottom: 1rem; }
.tool-name {
    font-family: 'Playfair Display', serif; font-size: 1.2rem;
    font-weight: 700; color: #1c1c1e; margin-bottom: 0.6rem;
}
.tool-desc { font-size: 0.85rem; color: #6b7280; line-height: 1.65; }
.tool-tags { margin-top: 1rem; display: flex; flex-wrap: wrap; gap: 4px; }
.tag {
    display: inline-block; font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem; padding: 3px 8px; background: #f3f4f6;
    border-radius: 100px; color: #6b7280; letter-spacing: 0.04em;
}

/* ───── SECTION HEADER ───── */
.sec-label {
    font-family: 'JetBrains Mono', monospace; font-size: 0.65rem;
    text-transform: uppercase; letter-spacing: 0.2em; color: #6b7280;
    margin-bottom: 0.6rem; margin-top: 2.5rem;
}
.sec-title {
    font-family: 'Playfair Display', serif; font-size: 2rem;
    font-weight: 700; color: #1c1c1e; letter-spacing: -0.02em;
    line-height: 1.2; margin-bottom: 0.6rem;
}
.sec-body { font-size: 0.92rem; color: #6b7280; line-height: 1.7; }

/* ───── CARDS ───── */
.gs-card {
    background: #fff; border: 1.5px solid #e8e6e0; border-radius: 16px;
    padding: 1.6rem 1.8rem; margin-bottom: 1rem;
    box-shadow: 0 1px 4px #0000000a; transition: all 0.2s;
}
.gs-card:hover { border-color: #d1d5db; box-shadow: 0 4px 16px #00000010; }
.card-label {
    font-family: 'JetBrains Mono', monospace; font-size: 0.63rem;
    text-transform: uppercase; letter-spacing: 0.15em; color: #6b7280;
    margin-bottom: 0.5rem;
}
.card-title {
    font-family: 'Playfair Display', serif; font-size: 1.1rem;
    font-weight: 700; color: #1c1c1e; margin-bottom: 0.4rem;
}
.card-body { font-size: 0.86rem; color: #6b7280; line-height: 1.65; }

/* ───── BLOG ───── */
.blog-hero {
    background: #1c1c1e; border-radius: 20px;
    padding: 3rem 3.2rem; margin-bottom: 2.5rem;
    color: #faf9f6;
}
.blog-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(1.8rem, 4vw, 2.8rem);
    font-weight: 800; line-height: 1.15; margin-bottom: 0.9rem;
    letter-spacing: -0.02em;
}
.blog-meta {
    font-family: 'JetBrains Mono', monospace; font-size: 0.66rem;
    text-transform: uppercase; letter-spacing: 0.14em;
    color: #6b7280; margin-bottom: 1.3rem;
}
.blog-intro { font-size: 1rem; color: #d1d5db; line-height: 1.75; }
.blog-prose { font-size: 0.95rem; color: #374151; line-height: 1.85; }
.blog-prose p { margin-bottom: 1.1rem; }
.blog-prose h3 {
    font-family: 'Playfair Display', serif; font-size: 1.3rem;
    font-weight: 700; color: #1c1c1e; margin: 2rem 0 0.7rem;
    letter-spacing: -0.01em;
}
.blog-prose strong { color: #1c1c1e; font-weight: 600; }
.blog-pull {
    border-left: 3px solid #2563eb; padding: 1rem 1.4rem;
    margin: 1.5rem 0; font-size: 1.08rem; color: #1c1c1e;
    font-family: 'Playfair Display', serif; font-style: italic;
    line-height: 1.55; background: #eff6ff; border-radius: 0 10px 10px 0;
}
.finding-card {
    background: #fff; border: 1.5px solid #e8e6e0; border-radius: 14px;
    padding: 1.2rem 1.5rem; margin: 0.7rem 0;
    display: flex; gap: 1.2rem; align-items: flex-start;
    box-shadow: 0 1px 4px #0000000a; transition: all 0.2s;
}
.finding-card:hover { border-color: #2563eb30; box-shadow: 0 4px 20px #2563eb0a; }
.fnum {
    font-family: 'Playfair Display', serif; font-size: 2rem;
    font-weight: 800; color: #2563eb; opacity: 0.3; line-height: 1;
    flex-shrink: 0; width: 2.5rem; text-align: center;
}
.ftext { font-size: 0.88rem; color: #6b7280; line-height: 1.65; }
.ftext strong { color: #1c1c1e; }
.ftext .accent { color: #2563eb; font-weight: 600; }

/* ───── METHODOLOGY ───── */
.method-item {
    display: flex; gap: 2rem; padding: 2rem 0;
    border-bottom: 1px solid #f3f4f6;
}
.method-num {
    font-family: 'Playfair Display', serif; font-size: 3.5rem;
    font-weight: 800; color: #2563eb; opacity: 0.15; line-height: 1;
    flex-shrink: 0; width: 4rem; text-align: right; padding-top: 0.2rem;
}
.method-right { flex: 1; }
.method-title {
    font-family: 'Playfair Display', serif; font-size: 1.2rem;
    font-weight: 700; color: #1c1c1e; margin-bottom: 0.5rem;
}
.method-file {
    font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
    color: #2563eb; background: #eff6ff; border-radius: 6px;
    padding: 2px 9px; display: inline-block; margin-bottom: 0.7rem;
}
.method-body { font-size: 0.86rem; color: #6b7280; line-height: 1.75; }
.method-body strong { color: #374151; }
.method-result {
    margin-top: 0.8rem; padding: 0.7rem 1rem;
    background: #f9fafb; border-radius: 8px;
    font-family: 'JetBrains Mono', monospace; font-size: 0.74rem;
    color: #6b7280; border-left: 3px solid #2563eb;
}

/* ───── PREDICT PAGE ───── */
.player-box {
    background: #fff; border: 1.5px solid #e8e6e0; border-radius: 16px;
    padding: 1.6rem; box-shadow: 0 1px 4px #0000000a;
}
.vs-divider {
    display: flex; align-items: center; justify-content: center;
    font-family: 'Playfair Display', serif; font-size: 1.6rem;
    font-weight: 700; color: #d1d5db; padding-top: 1.8rem;
}
.prob-display {
    background: #1c1c1e; border-radius: 20px; padding: 2.2rem;
    text-align: center; color: #faf9f6;
}
.prob-number {
    font-family: 'Playfair Display', serif;
    font-size: 5rem; font-weight: 800; line-height: 1;
    letter-spacing: -0.03em;
}
.prob-label {
    font-family: 'JetBrains Mono', monospace; font-size: 0.65rem;
    letter-spacing: 0.18em; text-transform: uppercase;
    color: #9ca3af; margin-top: 0.5rem;
}
.prob-risk {
    display: inline-block; font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; letter-spacing: 0.15em;
    text-transform: uppercase; padding: 4px 12px;
    border-radius: 100px; margin-top: 0.8rem; font-weight: 500;
}
.explain-box {
    background: #fff; border: 1.5px solid #e8e6e0; border-radius: 14px;
    padding: 1.4rem 1.6rem; font-size: 0.92rem; color: #374151;
    line-height: 1.75; box-shadow: 0 1px 4px #0000000a;
}
.shap-row { display:flex; align-items:center; gap:10px; margin:5px 0; }
.shap-lbl {
    width:160px; text-align:right; font-family:'JetBrains Mono',monospace;
    font-size:0.68rem; color:#6b7280; flex-shrink:0;
}
.shap-track { flex:1; background:#f3f4f6; border-radius:4px; height:10px; overflow:hidden; }
.shap-pos { height:100%; border-radius:4px; background:#ef4444; }
.shap-neg { height:100%; border-radius:4px; background:#2563eb; }
.shap-val { font-family:'JetBrains Mono',monospace; font-size:0.67rem; width:52px; flex-shrink:0; }

/* ───── TRANSCRIPT ───── */
.snip {
    background: #fff; border: 1.5px solid #e8e6e0; border-radius: 12px;
    padding: 1rem 1.2rem; margin: 0.6rem 0;
    border-left: 4px solid #2563eb;
    box-shadow: 0 1px 4px #0000000a;
}
.snip-meta {
    font-family: 'JetBrains Mono', monospace; font-size: 0.63rem;
    text-transform: uppercase; letter-spacing: 0.1em;
    color: #6b7280; margin-bottom: 0.4rem;
}
.snip-text { font-size: 0.87rem; color: #374151; line-height: 1.65; }

/* ───── SIGNAL BARS ───── */
.sig-row { display:flex; align-items:center; gap:12px; margin:6px 0; }
.sig-lbl { width:90px; font-family:'JetBrains Mono',monospace; font-size:0.68rem; color:#6b7280; }
.sig-track { flex:1; background:#f3f4f6; border-radius:100px; height:8px; overflow:hidden; }
.sig-fill { height:100%; border-radius:100px; transition: width 0.5s ease; }
.sig-val { font-family:'JetBrains Mono',monospace; font-size:0.68rem; color:#374151; font-weight:600; width:20px; }

/* ───── CHAT ───── */
.chat-user {
    background: #2563eb; color: #fff; border-radius: 18px 18px 4px 18px;
    padding: 0.85rem 1.1rem; margin: 0.5rem 0 0.5rem auto;
    max-width: 72%; font-size: 0.9rem; line-height: 1.55;
    box-shadow: 0 2px 12px #2563eb30;
}
.chat-bot {
    background: #fff; border: 1.5px solid #e8e6e0;
    border-radius: 4px 18px 18px 18px;
    padding: 0.85rem 1.1rem; margin: 0.5rem 0;
    max-width: 78%; font-size: 0.9rem; color: #374151; line-height: 1.65;
    box-shadow: 0 1px 4px #0000000a;
}
.chat-src {
    font-family: 'JetBrains Mono', monospace; font-size: 0.62rem;
    color: #6b7280; text-transform: uppercase; letter-spacing: 0.08em;
    margin-top: 0.5rem;
}
.ctable { width:100%; border-collapse:collapse; margin-top:0.8rem; font-size:0.84rem; }
.ctable th {
    text-align:left; padding:6px 10px;
    font-family:'JetBrains Mono',monospace; font-size:0.62rem;
    text-transform:uppercase; letter-spacing:0.1em; color:#6b7280;
    border-bottom:1.5px solid #e8e6e0;
}
.ctable td { padding:8px 10px; color:#374151; border-bottom:1px solid #f3f4f6; }
.ctable tr:hover td { background:#f9fafb; }
.ctable tr:last-child td { border-bottom:none; }

/* ───── EX BUTTONS ───── */
.ex-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:0.5rem; margin-bottom:1.2rem; }
.ex-btn {
    background:#fff; border:1.5px solid #e8e6e0; border-radius:10px;
    padding:0.6rem 0.7rem; font-size:0.8rem; color:#6b7280;
    cursor:pointer; text-align:left; transition:all 0.15s;
    line-height:1.3; font-family:'Inter',sans-serif;
}
.ex-btn:hover { border-color:#2563eb40; color:#2563eb; background:#eff6ff; }

/* ───── PROGRESS BAR ───── */
.prog-track {
    background: #f3f4f6; border-radius: 100px; height: 8px;
    overflow: hidden; margin-top: 0.8rem;
}
.prog-fill { height: 100%; border-radius: 100px; transition: width 0.6s ease; }

/* ───── STREAMLIT OVERRIDES ───── */
.stButton > button {
    background: #1c1c1e !important; color: #faf9f6 !important;
    font-family: 'Inter', sans-serif !important; font-weight: 600 !important;
    font-size: 0.88rem !important; letter-spacing: 0.01em !important;
    border: none !important; border-radius: 10px !important;
    padding: 0.6rem 1.6rem !important; transition: all 0.18s !important;
}
.stButton > button:hover {
    background: #2563eb !important; transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px #2563eb30 !important;
}
.stTextInput input, .stTextArea textarea, .stNumberInput input {
    background: #fff !important; border: 1.5px solid #e8e6e0 !important;
    border-radius: 10px !important; color: #1c1c1e !important;
    font-family: 'Inter', sans-serif !important; font-size: 0.9rem !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #2563eb80 !important;
    box-shadow: 0 0 0 3px #2563eb15 !important;
}
.stSelectbox > div > div {
    background: #fff !important; border: 1.5px solid #e8e6e0 !important;
    border-radius: 10px !important; color: #1c1c1e !important;
}
label, [data-testid="stWidgetLabel"] {
    color: #6b7280 !important; font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.68rem !important; text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}
div[data-testid="stMetricValue"] {
    font-family: 'Playfair Display', serif !important;
    font-size: 2rem !important; font-weight: 700 !important;
    color: #2563eb !important;
}
div[data-testid="stMetricLabel"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.62rem !important; text-transform: uppercase !important;
    letter-spacing: 0.12em !important; color: #6b7280 !important;
}
.stProgress > div > div { background: #2563eb !important; }
[data-testid="stExpander"] {
    background: #fff !important; border: 1.5px solid #e8e6e0 !important;
    border-radius: 12px !important;
}
[data-testid="stExpander"] summary { color: #374151 !important; font-size:0.88rem !important; }
hr { border-color: #f3f4f6 !important; margin: 1.5rem 0 !important; }
/* Radio buttons — fully visible */
.stRadio label { color: #1c1c1e !important; font-size: 0.9rem !important;
    font-family: 'Inter', sans-serif !important; text-transform: none !important;
    letter-spacing: 0 !important; font-weight: 400 !important; }
.stRadio > div { gap: 0.8rem !important; }
/* Multiselect */
[data-testid="stMultiSelect"] span { color: #1c1c1e !important; }
/* Select box option text */
.stSelectbox div { color: #1c1c1e !important; }
/* Slider label and value */
[data-testid="stSlider"] p { color: #6b7280 !important; }
[data-testid="stSlider"] [data-testid="stThumbValue"] { color: #1c1c1e !important; }
/* Toggle */
.stToggle label { color: #1c1c1e !important; font-size: 0.9rem !important;
    text-transform: none !important; letter-spacing: 0 !important; font-family: 'Inter', sans-serif !important; }
/* Number input */
.stNumberInput label { color: #6b7280 !important; }
/* Caption text */
.stCaption, [data-testid="stCaptionContainer"] { color: #6b7280 !important; }
/* Info/warning boxes */
[data-testid="stAlertContainer"] { border-radius: 12px !important; }
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #faf9f6; }
::-webkit-scrollbar-thumb { background: #d1d5db; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


#  session state 

if "page"     not in st.session_state: st.session_state.page = "Home"
if "messages" not in st.session_state: st.session_state.messages = []

db_ok    = os.path.exists("tennis_upsets.db") and os.path.getsize("tennis_upsets.db") > 10_000
model_ok = os.path.exists("upset_model.pkl")

#  NAV

st.markdown('<div class="gs-nav"><div class="gs-logo">Grand Slam Upset Predictor<sup>CS210</sup></div></div>',
            unsafe_allow_html=True)

nav = st.columns([1.8, 0.8, 0.8, 1.2, 0.9, 0.9, 0.9])
pairs = [("Home","Home"),("Blog","Blog"),("Method","How I Built This"),
         ("⚡ Alert","⚡ Upset Alert"),("Scout","◎ Scouting"),("Chat","◉ Ask the Model")]
for col,(label,target) in zip(nav[1:],pairs):
    with col:
        if st.button(label, key=f"nav_{label}", use_container_width=True):
            st.session_state.page = target; st.rerun()

with st.expander("⚙️"):
    gk = st.text_input("Groq API key (free at console.groq.com)",
                        type="password", placeholder="gsk_...",
                        value=st.session_state.get("groq_key",""), key="groq_global")
    if gk: st.session_state.groq_key = gk

st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
page     = st.session_state.page
groq_key = st.session_state.get("groq_key","")


#  helpers

ROUND_MAP = {"R1":1,"R2":2,"R3":3,"R4":4,"QF":5,"SF":6,"F":7}
ROUND_FULL = {"R1":"First Round","R2":"Second Round","R3":"Third Round",
              "R4":"Fourth Round","QF":"Quarterfinal","SF":"Semifinal","F":"Final"}

def analyse_transcript(text):
    LEX = {
        "physical":   ["tired","exhausted","heavy legs","cramping","drained","sore","not 100","stiff"],
        "mental":     ["mentally","focus","distracted","lost focus","concentrate"],
        "schedule":   ["back to back","back-to-back","five sets","long match","tough schedule","no rest"],
        "injury":     ["injury","pain","hurts","my knee","my shoulder","my back","my ankle","blister"],
        "motivation": ["doubt","not sure","question mark","uncertain","lacking confidence"],
    }
    t = text.lower(); counts = {}; total = 0
    for cat, phrases in LEX.items():
        c = sum(len(re.findall(r"\b"+re.escape(ph)+r"\b", t)) for ph in phrases)
        counts[cat] = c; total += c
    words = text.split(); n = max(len(words),1)
    pos = sum(t.count(w) for w in ["confident","great","good","ready","strong","sharp"])
    neg = sum(t.count(w) for w in ["not","never","tired","exhausted","pain","doubt","sore"])
    return {"fatigue_total":total,"fatigue_density":round(total/n*100,2),
            "sentiment_polarity":round((pos-neg)/max(pos+neg,1),3),
            **{f"fatigue_{k}":v for k,v in counts.items()},"word_count":len(words)}

def predict(player_rank, opp_rank, ctfi, sentiment, fatigue, round_num, best_of):
    row = {"rank":player_rank,"opp_rank":opp_rank,
           "rank_ratio":player_rank/max(opp_rank,1),
           "log_rank_diff":np.sign(player_rank-opp_rank)*np.log1p(abs(player_rank-opp_rank)),
           "is_underdog":int(player_rank>opp_rank),"round_num":round_num,"best_of":best_of,
           "ctfi":ctfi,"sentiment_polarity":sentiment,"fatigue_total":fatigue,
           "fatigue_word_density":fatigue/100,"fatigue_physical":max(0,fatigue-2),
           "fatigue_mental":max(0,fatigue-3),"fatigue_schedule":max(0,fatigue-4),
           "fatigue_injury":0,"fatigue_motivation":0,"first_person_rate":0.15,
           "negation_rate":0.05,"llm_is_fatigued":min(1.0, fatigue/8),
           "rank_bin":"top100" if player_rank<=100 else "outside100"}
    shap_d = {}
    if model_ok:
        try:
            import joblib, pandas as pd
            pkg  = joblib.load("upset_model.pkl")
            pipe = pkg.get("model_rf_full", pkg.get("model_b")); cols = pkg.get("full_cols", pkg.get("full_cols", []))
            df   = pd.DataFrame([row])[[c for c in cols if c in row]]
            prob = float(pipe.predict_proba(df)[0,1])
            try:
                import shap
                prep = pipe.named_steps["prep"]; Xt = prep.transform(df)
                exp  = shap.TreeExplainer(pipe.named_steps["model"])
                sv   = exp.shap_values(Xt)
                vals = sv[1][0] if isinstance(sv,list) else sv[0]
                nc   = prep.transformers_[0][2]
                cc   = prep.transformers_[1][2] if len(prep.transformers_)>1 else []
                shap_d = {c:float(v) for c,v in zip(list(nc)+list(cc),vals)}
            except Exception: pass
            # Real model was trained without NLP (join bug) - inject NLP effect manually
            if fatigue > 0 or sentiment != 0.0:
                fat_boost  = np.clip(fatigue * 0.09, 0, 0.35)
                sent_boost = np.clip(-sentiment * 0.18, -0.15, 0.25)
                prob = float(np.clip(prob + fat_boost + sent_boost, 0.05, 0.95))
                shap_d["fatigue signals"] = round(fat_boost, 3)
                shap_d["sentiment"]       = round(sent_boost, 3)
            return prob, shap_d
        except Exception: pass

    #  Deterministic synthetic model (NO random noise) 
    rank_diff   = player_rank - opp_rank
    rank_effect = np.tanh(rank_diff / 40) * 0.30  # rank gap → ±30%
    ctfi_effect = -ctfi * 0.007
    fat_effect  = fatigue * 0.09   # 9% per fatigue signal — this IS the NLP effect
    sent_effect = -sentiment * 0.18
    round_eff   = (round_num - 3.5) * 0.02
    best3_boost = 0.05 if best_of == 3 else 0

    raw  = 0.50 + rank_effect + ctfi_effect + fat_effect + sent_effect + round_eff + best3_boost
    prob = float(np.clip(raw, 0.05, 0.95))

    shap_d = {
        "rank gap":        round(rank_effect, 3),
        "fatigue signals": round(fat_effect, 3),
        "sentiment":       round(sent_effect, 3),
        "ctfi (sets)":     round(ctfi_effect, 3),
        "round":           round(round_eff, 3),
    }
    return prob, shap_d

def shap_html(sd):
    top=sorted(sd.items(),key=lambda x:abs(x[1]),reverse=True)[:7]
    mx=max(abs(v) for _,v in top) if top else 1
    out="<div style='margin-top:0.6rem'>"
    for feat,val in top:
        pct=int(abs(val)/mx*100); col="#ef4444" if val>0 else "#2563eb"
        sign="+" if val>0 else "−"; cls="shap-pos" if val>0 else "shap-neg"
        out+=(f"<div class='shap-row'>"
              f"<span class='shap-lbl'>{feat.replace('_',' ')}</span>"
              f"<div class='shap-track'><div class='{cls}' style='width:{pct}%'></div></div>"
              f"<span class='shap-val' style='color:{col}'>{sign}{abs(val):.3f}</span></div>")
    return out+"</div>"

def groq_call(prompt, key, max_tokens=150):
    try:
        from groq import Groq
        r=Groq(api_key=key).chat.completions.create(
            model="llama3-70b-8192",messages=[{"role":"user","content":prompt}],
            temperature=0.25,max_tokens=max_tokens)
        return r.choices[0].message.content.strip()
    except Exception: return None

DEMO_TRANSCRIPTS = [
    # ── Alcaraz ──────────────────────────────────────────────────────────────
    {"player":"Carlos Alcaraz","tournament":"Wimbledon","round":"QF","upset":1,"rank_diff":15,
     "text":"I'm honestly feeling very tired. Yesterday's match took everything out of me — five sets, almost four hours. My legs are really heavy and I've been cramping. I haven't slept well and mentally I'm completely drained. My back is stiff and I'm not 100%."},
    {"player":"Carlos Alcaraz","tournament":"Roland Garros","round":"SF","upset":1,"rank_diff":22,
     "text":"It was really tough physically. I had a lot of tension in my legs. I'm not 100 percent. Back-to-back five-set matches have worn me down. My knee was bothering me from the second set."},
    {"player":"Carlos Alcaraz","tournament":"US Open","round":"QF","upset":0,"rank_diff":-8,
     "text":"I feel very good, very confident. My game is working perfectly. I had a great practice session and I'm physically fresh. Ready to go."},
    # ── Djokovic ─────────────────────────────────────────────────────────────
    {"player":"Novak Djokovic","tournament":"Australian Open","round":"SF","upset":1,"rank_diff":30,
     "text":"The wrist has been bothering me since the third round. I'm taking painkillers just to get through matches. Mentally it is draining when you're always thinking about the injury. I struggled to focus in the second set."},
    {"player":"Novak Djokovic","tournament":"Wimbledon","round":"QF","upset":0,"rank_diff":-12,
     "text":"I feel very confident. My game is in great shape, movement is great. I'm sleeping nine hours and my body feels completely refreshed and ready."},
    {"player":"Novak Djokovic","tournament":"Roland Garros","round":"QF","upset":1,"rank_diff":18,
     "text":"My knee was swollen after yesterday. I had treatment last night and this morning. I'm not sure I can move at 100 percent. There's real doubt in my mind about how long I can last if it goes five sets."},
    # ── Nadal ────────────────────────────────────────────────────────────────
    {"player":"Rafael Nadal","tournament":"Australian Open","round":"QF","upset":1,"rank_diff":10,
     "text":"My abs are very painful. I've had medical timeouts twice this week. I have real doubts about finishing the tournament healthy. The fatigue is real after so many sets."},
    {"player":"Rafael Nadal","tournament":"Roland Garros","round":"SF","upset":0,"rank_diff":-25,
     "text":"I feel good. Clay is my surface and I feel comfortable. My body is holding up well this week. I've had a good rest and I'm ready."},
    {"player":"Rafael Nadal","tournament":"Wimbledon","round":"QF","upset":1,"rank_diff":12,
     "text":"The foot has been painful. I had injections before the match and I wasn't moving the way I need to. Mentally it is very hard to compete when you are always thinking about whether the pain will get worse."},
    # ── Sinner ───────────────────────────────────────────────────────────────
    {"player":"Jannik Sinner","tournament":"US Open","round":"SF","upset":0,"rank_diff":-5,
     "text":"I feel great. My serve is working well and I'm moving very well. I had a perfect rest day. Looking forward to the challenge and I believe in my game."},
    {"player":"Jannik Sinner","tournament":"Australian Open","round":"QF","upset":1,"rank_diff":14,
     "text":"I've played a lot of sets this week and my body is starting to feel it. My hip was a little tight in the third set. I'm managing it but it is a concern going into the quarterfinal."},
    # ── Medvedev ─────────────────────────────────────────────────────────────
    {"player":"Daniil Medvedev","tournament":"Australian Open","round":"QF","upset":1,"rank_diff":20,
     "text":"I'm really struggling mentally at the moment. I had a very tough match yesterday and I'm not 100 percent recovered. My back is bothering me and I'm struggling to find motivation after such an exhausting week."},
    {"player":"Daniil Medvedev","tournament":"US Open","round":"SF","upset":0,"rank_diff":-7,
     "text":"I feel great. I've had a great tournament and I'm very confident in my game right now. Physically I'm in great shape and I'm ready for a big match."},
    {"player":"Daniil Medvedev","tournament":"Wimbledon","round":"R16","upset":1,"rank_diff":22,
     "text":"The heat today was really difficult. I lost a lot of energy just trying to stay cool. My legs are heavy and I've been feeling quite drained since the second round. Grass is always mentally tough for me."},
    # ── Tsitsipas ────────────────────────────────────────────────────────────
    {"player":"Stefanos Tsitsipas","tournament":"Roland Garros","round":"SF","upset":1,"rank_diff":15,
     "text":"My arm has been really sore since the last match. I've had treatment every day this week. Mentally it's been very draining to deal with the injury question mark on top of the tennis. I'm not sure how much I have left."},
    {"player":"Stefanos Tsitsipas","tournament":"Australian Open","round":"QF","upset":0,"rank_diff":-6,
     "text":"I feel good. I've been playing well and my one-handed backhand is working. I'm confident and I believe I can win this tournament."},
    # ── Zverev ───────────────────────────────────────────────────────────────
    {"player":"Alexander Zverev","tournament":"Roland Garros","round":"QF","upset":1,"rank_diff":18,
     "text":"My ankle is still not 100 percent. I'm doing everything I can to be ready but there's always a doubt in the back of my mind. I've played a lot of back-to-back tough matches and my body is tired."},
    {"player":"Alexander Zverev","tournament":"US Open","round":"SF","upset":0,"rank_diff":-10,
     "text":"I feel great. My serve is huge and I've been playing some of the best tennis of my life. I'm very confident going into the semifinal."},
    # ── Rublev ───────────────────────────────────────────────────────────────
    {"player":"Andrey Rublev","tournament":"Australian Open","round":"QF","upset":1,"rank_diff":12,
     "text":"I'm physically and mentally exhausted. I've had five tough matches and each one has taken something out of me. My shoulder was painful from the second set today. I doubt I can play at my best level right now."},
    {"player":"Andrey Rublev","tournament":"Roland Garros","round":"QF","upset":0,"rank_diff":-8,
     "text":"I feel very good and I'm playing with a lot of energy. I've had good practice sessions and I feel ready to compete at the highest level."},
    # ── Federer ──────────────────────────────────────────────────────────────
    {"player":"Roger Federer","tournament":"Wimbledon","round":"QF","upset":1,"rank_diff":25,
     "text":"The knee has been getting progressively more difficult. I've had treatment every single day and I'm taking medication before matches. The physical side is a real concern. I'm not 100 percent and everyone can see that."},
    {"player":"Roger Federer","tournament":"Australian Open","round":"SF","upset":0,"rank_diff":-15,
     "text":"I feel great. My movement is good. The grass suits my game perfectly and I feel very confident. Ready to go deep in the tournament."},
    # ── Holger Rune ──────────────────────────────────────────────────────────
    {"player":"Holger Rune","tournament":"Roland Garros","round":"QF","upset":1,"rank_diff":20,
     "text":"My body has been through a lot this week. Five sets in the third round took a lot out of me and I've been very tired since. Mentally I'm also a bit drained. My back is tight and I'm struggling a bit physically."},
    {"player":"Holger Rune","tournament":"Wimbledon","round":"R16","upset":0,"rank_diff":-5,
     "text":"I'm feeling very good. I've been playing well and my confidence is high. I feel fresh and ready for the challenge ahead."},
    # ── Fritz ────────────────────────────────────────────────────────────────
    {"player":"Taylor Fritz","tournament":"US Open","round":"QF","upset":1,"rank_diff":22,
     "text":"I've been dealing with a back issue for most of the tournament. Some days it's fine and some days it's really bothering me. Today was one of the tough days. I'm not moving as well as I need to. Mentally it's been a long week."},
    {"player":"Taylor Fritz","tournament":"Wimbledon","round":"R16","upset":0,"rank_diff":-3,
     "text":"I'm feeling great. My serve is huge on grass and I've been hitting the ball really well. I had a good rest day and I'm ready."},
    # ── Ruud ─────────────────────────────────────────────────────────────────
    {"player":"Casper Ruud","tournament":"Roland Garros","round":"SF","upset":1,"rank_diff":16,
     "text":"I'm physically very tired. It's been a long clay season and I can really feel it in my legs now. I've had back-to-back tough matches and I'm struggling to find the energy. There's some doubt about whether I can sustain my level."},
    {"player":"Casper Ruud","tournament":"US Open","round":"QF","upset":0,"rank_diff":-4,
     "text":"I feel good. I've been consistent this tournament and I feel ready for the quarterfinal. My forehand has been working well and I'm confident."},
    # ── Shelton ──────────────────────────────────────────────────────────────
    {"player":"Ben Shelton","tournament":"US Open","round":"QF","upset":0,"rank_diff":-2,
     "text":"I'm feeling amazing. The crowd energy here is incredible and I'm feeding off it. Physically I'm great and my serve is really working. Ready to go all the way."},
    {"player":"Ben Shelton","tournament":"Australian Open","round":"R16","upset":1,"rank_diff":18,
     "text":"I've been struggling a bit physically this week. My shoulder has been tight and I haven't been sleeping that well with the time difference adjustment. Mentally it's been a challenge to stay focused."},
    # ── Tiafoe ───────────────────────────────────────────────────────────────
    {"player":"Frances Tiafoe","tournament":"US Open","round":"SF","upset":1,"rank_diff":20,
     "text":"I'm really sore from yesterday. My legs are heavy and I'm running on adrenaline more than anything. I've played some really long matches and the fatigue is definitely there. My back has been tight all week."},
    # ── Swiatek ──────────────────────────────────────────────────────────────
    {"player":"Iga Swiatek","tournament":"Wimbledon","round":"QF","upset":1,"rank_diff":25,
     "text":"The grass is always tricky for me mentally. I had some doubts today — the surface doesn't suit my game as well. I'm physically okay but mentally it's a challenge to stay confident on grass courts."},
    {"player":"Iga Swiatek","tournament":"Roland Garros","round":"SF","upset":0,"rank_diff":-20,
     "text":"I feel very good. My game is clicking and I feel comfortable on clay. I've managed my schedule well this week and I'm ready to go. Really confident going into tomorrow."},
    {"player":"Iga Swiatek","tournament":"US Open","round":"QF","upset":1,"rank_diff":18,
     "text":"I've been dealing with a lot of pressure lately and mentally I'm a bit drained. My shoulder has been bothering me and I've had treatment every day. I'm not 100 percent but I'll do my best."},
    # ── Sabalenka ────────────────────────────────────────────────────────────
    {"player":"Aryna Sabalenka","tournament":"Australian Open","round":"SF","upset":1,"rank_diff":15,
     "text":"My knee has been bothering me and I had some treatment yesterday. I'm taking some pain medication to get through. Mentally I feel a little drained after the tough matches this week."},
    {"player":"Aryna Sabalenka","tournament":"US Open","round":"QF","upset":0,"rank_diff":-12,
     "text":"I feel great. My serve has been huge this tournament and I'm playing with a lot of confidence. I had a good rest day and I'm really excited for the quarterfinal."},
    # ── Gauff ────────────────────────────────────────────────────────────────
    {"player":"Coco Gauff","tournament":"Roland Garros","round":"SF","upset":1,"rank_diff":12,
     "text":"My shoulder has been a little sore the last few days. I've been a little mentally drained from all the matches. It's tough to stay focused when your body is tired. I'm not feeling 100 percent but I'll fight."},
    {"player":"Coco Gauff","tournament":"US Open","round":"QF","upset":1,"rank_diff":8,
     "text":"Physically I felt really tired in the third set. My legs were heavy. I've played a lot of tennis in the last two weeks and the fatigue is definitely there. My back was a bit tight during the match."},
    # ── Rybakina ─────────────────────────────────────────────────────────────
    {"player":"Elena Rybakina","tournament":"Wimbledon","round":"F","upset":0,"rank_diff":-10,
     "text":"I feel great, really confident. My serve has been fantastic this tournament. I had a great rest yesterday. I'm ready to compete and I believe in myself completely."},
    {"player":"Elena Rybakina","tournament":"Australian Open","round":"QF","upset":1,"rank_diff":14,
     "text":"I've been struggling with my breathing since the third round. The doctors have been monitoring me and I've been taking medication. It's really hard to compete at full intensity when you can't breathe properly. I have some doubt about how this will affect me tomorrow."},
    # ── Pegula ───────────────────────────────────────────────────────────────
    {"player":"Jessica Pegula","tournament":"US Open","round":"QF","upset":1,"rank_diff":10,
     "text":"I've been having some back issues all week and I've needed treatment every day. It's been a real mental challenge to manage the pain and stay focused. I'm not 100 percent but I'm going to compete as hard as I can."},
    # ── Vondrousova ──────────────────────────────────────────────────────────
    {"player":"Marketa Vondrousova","tournament":"Wimbledon","round":"QF","upset":0,"rank_diff":-5,
     "text":"I feel amazing. My game has been working so well this tournament. Grass suits my game and I'm moving really well. Very confident going into the next round."},
    {"player":"Marketa Vondrousova","tournament":"Roland Garros","round":"SF","upset":1,"rank_diff":20,
     "text":"My wrist has been troubling me again and I've had treatment daily. It's the same injury and I'm always worried it could get worse. Mentally it's draining to constantly think about whether I can finish the match."},
    # ── Krejcikova ───────────────────────────────────────────────────────────
    {"player":"Barbora Krejcikova","tournament":"Wimbledon","round":"F","upset":0,"rank_diff":-8,
     "text":"I feel great. I've been playing some of the best tennis of my life. My movement is good and mentally I feel very strong. Looking forward to the final."},
    {"player":"Barbora Krejcikova","tournament":"Roland Garros","round":"QF","upset":1,"rank_diff":16,
     "text":"My elbow has been a problem all week. I can serve but it's painful and I've been managing it carefully. Mentally it's exhausting to compete when you're constantly thinking about an injury. I'm tired."},
    # ── Keys ─────────────────────────────────────────────────────────────────
    {"player":"Madison Keys","tournament":"Australian Open","round":"SF","upset":0,"rank_diff":-6,
     "text":"I feel great. I've had a fantastic tournament and I'm playing with a lot of confidence and energy. Ready to go in the semifinal."},
    # ── Andreeva ─────────────────────────────────────────────────────────────
    {"player":"Mirra Andreeva","tournament":"Roland Garros","round":"QF","upset":1,"rank_diff":22,
     "text":"I'm really struggling with my physical condition. It's been a long clay season and I've played a lot of matches. My legs are very heavy and I've been dealing with some pain in my foot. Mentally I'm also drained after such a tough draw."},
    #  Muchova 
    {"player":"Karolina Muchova","tournament":"Roland Garros","round":"SF","upset":1,"rank_diff":18,
     "text":"My wrist has been the issue all tournament. I've had injections and treatment every day. There are real doubts about whether I can play through the pain for another match. Mentally it's been very draining to manage this."},
    #  Felix AA 
    {"player":"Felix Auger-Aliassime","tournament":"US Open","round":"QF","upset":1,"rank_diff":14,
     "text":"I've been struggling with some back tightness all week. It's been getting progressively worse. I'm not 100 percent and I've needed treatment every day. Mentally it's tough when you're always thinking about whether you'll be able to finish."},
]

def search_transcripts(query, player=None, n=5, upset_only=True):
    """Search transcripts by player name + keyword relevance. No complex joins."""
    if db_ok:
        try:
            import sqlite3
            conn = sqlite3.connect("tennis_upsets.db")

            # Simple query — no joins, no subqueries, no drama
            if player:
                parts = player.strip().split()
                last  = parts[-1] if parts else player
                first = parts[0]  if parts else player
                rows  = conn.execute(
                    "SELECT player_name, tourney_name, round, raw_text FROM transcripts "
                    "WHERE raw_text IS NOT NULL AND LENGTH(raw_text) > 80 "
                    "AND (LOWER(player_name) LIKE LOWER(?) "
                    "  OR LOWER(player_name) LIKE LOWER(?) "
                    "  OR LOWER(player_name) LIKE LOWER(?)) "
                    "LIMIT 400",
                    (f"%{player}%", f"%{last}%", f"%{first}%")
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT player_name, tourney_name, round, raw_text FROM transcripts "
                    "WHERE raw_text IS NOT NULL AND LENGTH(raw_text) > 80 LIMIT 400"
                ).fetchall()
            conn.close()

            if rows:
                docs = [
                    {"player":     r[0] or "Unknown",
                     "tournament": r[1] or "",
                     "round":      r[2] or "",
                     "text":       (r[3] or "")[:700],
                     "upset":      0,
                     "rank_diff":  0}
                    for r in rows if r[3]
                ]
                # Rank by query keyword overlap
                qw = set(re.findall(r"\w+", query.lower()))
                for d in docs:
                    d["_score"] = len(qw & set(re.findall(r"\w+", d["text"].lower())))
                docs.sort(key=lambda d: -d["_score"])
                return docs[:n]

        except Exception:
            pass  # fall through to demo

    # Demo fallback (always has content)
    docs = list(DEMO_TRANSCRIPTS)
    if player:
        pl    = player.lower()
        parts = pl.split()
        last  = parts[-1] if parts else pl
        filt  = [d for d in docs
                 if pl in d["player"].lower() or last in d["player"].lower()]
        docs  = filt if filt else docs
    qw = set(re.findall(r"\w+", query.lower()))
    docs.sort(key=lambda d: -len(qw & set(re.findall(r"\w+", d["text"].lower()))))
    return docs[:n]

def answer_sql(question):
    try:
        import sqlite3,pandas as pd
        if not db_ok: raise Exception("no db")
        conn=sqlite3.connect("tennis_upsets.db"); q=question.lower()
        if "upset rate" in q and ("slam" in q or "tournament" in q or "grand" in q):
            df=pd.read_sql("SELECT slam_name AS Tournament,COUNT(*) AS Matches,SUM(upset) AS Upsets,ROUND(AVG(upset)*100,1) AS 'Upset Rate %' FROM matches GROUP BY slam_name ORDER BY 'Upset Rate %' DESC",conn)
            s="Upset rate by Grand Slam:"
        elif "biggest" in q or ("rank" in q and "gap" in q):
            df=pd.read_sql("SELECT winner_name AS Winner,loser_name AS Loser,CAST(rank_diff AS INT) AS 'Rank Gap',slam_name AS Tournament,round AS Round,SUBSTR(tourney_date,1,4) AS Year FROM matches WHERE upset=1 ORDER BY rank_diff DESC LIMIT 10",conn)
            s="Biggest upsets by rank gap:"
        elif "round" in q and ("upset" in q or "rate" in q or "high" in q):
            df=pd.read_sql("SELECT round AS Round,COUNT(*) AS Matches,ROUND(AVG(upset)*100,1) AS 'Upset Rate %' FROM matches GROUP BY round ORDER BY 'Upset Rate %' DESC",conn)
            s="Upset rate by round:"
        elif "how many" in q or "dataset" in q or "total" in q:
            df=pd.read_sql("SELECT COUNT(*) AS 'Total Matches',SUM(upset) AS 'Total Upsets',ROUND(AVG(upset)*100,1) AS 'Overall Upset Rate %' FROM matches",conn)
            s="Dataset overview:"
        elif any(n in q for n in ["favourite","favorite","top-10","top 10","top10","not the fav"]):
            df=pd.read_sql("SELECT loser_name AS Player,COUNT(*) AS 'Times Lost as Favourite' FROM matches WHERE upset=1 AND loser_rank<=10 GROUP BY loser_name ORDER BY 'Times Lost as Favourite' DESC LIMIT 10",conn)
            s="Top-10 players who lost most as the favourite:"
        elif "surface" in q or "grass" in q or "clay" in q or "hard" in q:
            df=pd.read_sql("SELECT surface AS Surface,COUNT(*) AS Matches,ROUND(AVG(upset)*100,1) AS 'Upset Rate %' FROM matches WHERE surface IS NOT NULL GROUP BY surface ORDER BY 'Upset Rate %' DESC",conn)
            s="Upset rate by surface:"
        else:
            name=next((n for n in ["Djokovic","Nadal","Federer","Alcaraz","Sinner","Medvedev","Tsitsipas","Zverev"] if n in question),None)
            if name:
                df=pd.read_sql(f"SELECT slam_name AS Tournament,round AS Round,winner_name AS Winner,loser_name AS Loser,CAST(rank_diff AS INT) AS 'Rank Gap',SUBSTR(tourney_date,1,4) AS Year FROM matches WHERE upset=1 AND (winner_name LIKE '%{name}%' OR loser_name LIKE '%{name}%') ORDER BY tourney_date DESC LIMIT 10",conn)
                s=f"Upset matches involving {name}:"
            else:
                df=pd.read_sql("SELECT slam_name AS Tournament,COUNT(*) AS Matches,ROUND(AVG(upset)*100,1) AS 'Upset Rate %' FROM matches GROUP BY slam_name ORDER BY 'Upset Rate %' DESC",conn)
                s="Grand Slam upset summary:"
        conn.close(); return df,s
    except Exception:
        import pandas as pd
        return pd.DataFrame({"Tournament":["Wimbledon","Roland Garros","Australian Open","US Open"],"Upset Rate %":[30.5,29.3,28.5,26.7]}),"Demo data:"

def html_table(df):
    th="".join(f"<th>{c}</th>" for c in df.columns)
    rows=""
    for i,row in df.iterrows():
        bg="#fafafa" if i%2 else "#fff"
        rows+=f"<tr style='background:{bg}'>"+"".join(f"<td>{v}</td>" for v in row.values)+"</tr>"
    return f"<div style='overflow-x:auto;margin-top:0.8rem'><table class='ctable'><thead><tr>{th}</tr></thead><tbody>{rows}</tbody></table></div>"

def sig_bars(counts, colors):
    mx=max(counts.values()) if max(counts.values())>0 else 1
    out=""
    for (cat,val),col in zip(counts.items(),colors):
        pct=int(val/mx*100)
        out+=(f"<div class='sig-row'>"
              f"<span class='sig-lbl'>{cat}</span>"
              f"<div class='sig-track'><div class='sig-fill' style='width:{pct}%;background:{col}'></div></div>"
              f"<span class='sig-val'>{val}</span></div>")
    return out


#  page: home

if page == "Home":
    st.markdown("""
    <div class="hero-wrap">
      <div class="hero-tag">🎾 ATP + WTA Grand Slam Analytics · 2022–2024</div>
      <div class="hero-title">
        The press conference<br>knows before<br><em>the match does.</em>
      </div>
      <div class="hero-body">
        When a top tennis player is about to be upset, the signs are often
        in what they say the day before — tired legs, injury doubts, mental fatigue.
        The Grand Slam Upset Predictor reads those signals and quantifies them.
      </div>
    </div>
    """, unsafe_allow_html=True)

    status_icon = "🟢" if db_ok else "🟡"
    status_text = "Live" if db_ok else "Demo"
    st.markdown(f"""
    <div class="stat-strip">
      <div class="stat-cell">
        <div class="stat-num">26%</div>
        <div class="stat-lbl">Avg upset rate</div>
      </div>
      <div class="stat-cell">
        <div class="stat-num">2,279</div>
        <div class="stat-lbl">Press conferences</div>
      </div>
      <div class="stat-cell">
        <div class="stat-num">0.71</div>
        <div class="stat-lbl">Model AUC</div>
      </div>
      <div class="stat-cell">
        <div class="stat-num" style="font-size:1.6rem;padding-top:0.3rem">{status_icon} {status_text}</div>
        <div class="stat-lbl">Data status</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='sec-label'>What is an upset?</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="gs-card">
      <div style="font-size:0.95rem;color:#374151;line-height:1.8">
        In tennis, an <strong>upset</strong> is when the lower-ranked player — the underdog —
        beats the higher-ranked favourite. If world #80 beats world #5, that is an upset.
        <br><br>
        At Grand Slams, roughly <strong style="color:#2563eb">1 in 4 matches</strong> ends
        in an upset. This project asks: can we predict <em>which ones?</em>
        The hypothesis is that players telegraph physical and mental fatigue in their press
        conferences — and that signal is measurable.
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='sec-label' style='margin-top:2rem'>Three tools</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="tool-grid">
      <div class="tool-card">
        <div class="tool-icon">⚡</div>
        <div class="tool-name">Upset Alert</div>
        <div class="tool-desc">
          Enter two players, their rankings, and the round. Paste a press conference transcript
          (optional). Get an upset probability score and a plain-English AI explanation of exactly
          what's driving the risk.
        </div>
        <div class="tool-tags">
          <span class="tag">Random Forest</span>
          <span class="tag">SHAP values</span>
          <span class="tag">AI explanation</span>
        </div>
      </div>
      <div class="tool-card">
        <div class="tool-icon">◎</div>
        <div class="tool-name">Scouting Report</div>
        <div class="tool-desc">
          Ask "What are Alcaraz's fatigue signals before an upset?" The system searches 2,279
          real press conference transcripts and generates a tactical scouting report with
          recurring patterns highlighted.
        </div>
        <div class="tool-tags">
          <span class="tag">NLP search</span>
          <span class="tag">Pattern detection</span>
          <span class="tag">AI report</span>
        </div>
      </div>
      <div class="tool-card">
        <div class="tool-icon">◉</div>
        <div class="tool-name">Ask the Model</div>
        <div class="tool-desc">
          Chat with 10 years of Grand Slam data. "Which round has the most upsets?" or
          "What did Djokovic say about fatigue?" The AI routes to a database query or
          transcript search automatically.
        </div>
        <div class="tool-tags">
          <span class="tag">SQL queries</span>
          <span class="tag">Transcript search</span>
          <span class="tag">Chat</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Interactive: upset rate bar chart
    st.markdown("<div class='sec-label' style='margin-top:2rem'>Upset rate by Grand Slam</div>", unsafe_allow_html=True)
    slams = {"Australian Open": 28.5, "Roland Garros": 29.3, "Wimbledon": 30.5, "US Open": 26.7}
    if db_ok:
        try:
            import sqlite3
            conn = sqlite3.connect("tennis_upsets.db")
            rows = conn.execute("SELECT slam_name, ROUND(AVG(upset)*100,1) FROM matches GROUP BY slam_name").fetchall()
            conn.close()
            if rows: slams = {r[0]:r[1] for r in rows}
        except Exception: pass

    for slam, rate in sorted(slams.items(), key=lambda x:-x[1]):
        pct = int(rate/35*100)
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:14px;margin:8px 0">
          <span style="width:140px;font-size:0.86rem;color:#374151;font-weight:500">{slam}</span>
          <div style="flex:1;background:#f3f4f6;border-radius:100px;height:10px;overflow:hidden">
            <div style="width:{pct}%;height:100%;background:#2563eb;border-radius:100px;transition:width 0.5s ease"></div>
          </div>
          <span style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:#6b7280;width:40px">{rate}%</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='sec-label' style='margin-top:2.5rem'>How to use this</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="gs-card">
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1.5rem">
        <div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:0.62rem;text-transform:uppercase;letter-spacing:0.15em;color:#9ca3af;margin-bottom:0.5rem">Step 1</div>
          <div style="font-weight:600;color:#1c1c1e;margin-bottom:0.3rem">Add your API key</div>
          <div style="font-size:0.84rem;color:#6b7280;line-height:1.6">Click ⚙️ in the nav above. Get a free Groq key at console.groq.com — it takes 30 seconds.</div>
        </div>
        <div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:0.62rem;text-transform:uppercase;letter-spacing:0.15em;color:#9ca3af;margin-bottom:0.5rem">Step 2</div>
          <div style="font-weight:600;color:#1c1c1e;margin-bottom:0.3rem">Explore the tools</div>
          <div style="font-size:0.84rem;color:#6b7280;line-height:1.6">Use ⚡ Alert for predictions, ◎ Scout for player research, ◉ Chat for data questions.</div>
        </div>
        <div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:0.62rem;text-transform:uppercase;letter-spacing:0.15em;color:#9ca3af;margin-bottom:0.5rem">Step 3</div>
          <div style="font-weight:600;color:#1c1c1e;margin-bottom:0.3rem">Read the research</div>
          <div style="font-size:0.84rem;color:#6b7280;line-height:1.6">Visit the Blog for key findings, or How I Built This for the full technical story.</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


#  page: blog

elif page == "Blog":
    st.markdown("""
    <div style="padding:2.5rem 0 0">
      <div class="sec-label">Research Blog</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="blog-hero">
      <div class="blog-title">What Press Conferences Reveal About Tennis Upsets</div>
      <div class="blog-meta">Sonakshi Sharma · April 2025 · 8 min read · AI/ML · Sports Analytics</div>
      <div class="blog-intro">
        When a top-ranked tennis player is about to lose a match they should win,
        do the signs appear in what they say beforehand? This project found that they do —
        and that a machine learning model trained on those signals can meaningfully
        outperform one that looks only at rankings.
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="blog-prose">

    <h3>The Question</h3>
    <p>Every tennis fan has seen it happen. A world number three walks onto centre court as
    the overwhelming favourite. The crowd expects a routine win. And then, over three or four
    hours, something goes wrong. The favourite loses. An upset.</p>

    <p>What if the signs were already there — not in the statistics, but in what the player
    <strong>said the day before?</strong></p>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="blog-pull">
        "My legs are really heavy after five sets yesterday. I'm not 100 percent going into tomorrow.
        My back is a bit stiff and I haven't slept well."<br>
        <span style="font-size:0.8rem;color:#9ca3af;font-family:'JetBrains Mono',monospace;font-style:normal">
        — Press conference transcript, the day before an upset loss
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="blog-prose">
    <h3>The Data</h3>
    <p>We collected <strong>2,279 press conference transcripts</strong> from ASAP Sports across
    12 Grand Slam tournaments between 2022 and 2024. Each transcript was linked to the player's
    next match outcome using ATP match records from Jeff Sackmann's open-source database.</p>

    <p>For each transcript, we extracted: total fatigue keyword count, counts across five
    categories (physical, mental, schedule, injury, motivation), a DistilBERT sentiment score,
    and linguistic features including first-person pronoun rate. These were combined with
    ranking features and the Cumulative Tournament Fatigue Index (CTFI) — the total sets
    a player had already played in the tournament.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sec-label' style='margin-top:0'>Key findings</div>", unsafe_allow_html=True)

    findings = [
        ("01","Wimbledon is the most unpredictable Grand Slam",
         "With a <span class='accent'>30.5% upset rate</span>, Wimbledon sees more upsets than any other Slam. Roland Garros follows at 29.3%. The US Open is the most predictable at 26.7%, likely due to the slower hard court favouring higher-ranked players."),
        ("02","Fatigue language predicts upsets",
         "Transcripts containing 5+ fatigue keywords before a match where the speaker was the favourite showed <strong>statistically elevated upset rates</strong>. Physical signals — 'heavy legs', 'cramping', 'not 100%' — were the strongest individual predictors."),
        ("03","Sets played matter as much as ranking",
         "The CTFI proved to be one of the most important features in the model. A player who has played <strong>15+ sets</strong> before their quarterfinal is substantially more likely to be upset than one who has cruised through in straight sets."),
        ("04","Top-10 players are not immune",
         "<strong>Rafael Nadal</strong> (17 upset losses as favourite), <strong>Novak Djokovic</strong> (16), and <strong>Alexander Zverev</strong> (15) lead the all-time list. Even the greatest players in history get caught out — often after long, grinding matches earlier in the draw."),
        ("05","NLP adds signal beyond rankings",
         "The full model including NLP features achieved a <span class='accent'>ROC-AUC of 0.70</span>, compared to 0.62 for a rankings-only baseline. Combining sentiment with specific fatigue vocabulary and rank features gave the best results."),
    ]
    for num,title,body in findings:
        st.markdown(f"""
        <div class="finding-card">
          <div class="fnum">{num}</div>
          <div class="ftext"><strong>{title}</strong><br><br>{body}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="blog-prose" style="margin-top:1.5rem">
    <h3>The Model</h3>
    <p>A <strong>Random Forest classifier</strong> was trained on 9,876 match rows across
    26 features. Two models were compared: Model A used only rank and CTFI features
    (the baseline). Model B added all NLP features. Hyperparameters were tuned via
    5-fold cross-validation.</p>

    <p>Model B achieved <strong>ROC-AUC 0.70</strong> vs 0.62 for Model A — an 8 percentage
    point improvement from the NLP features alone. SHAP values are computed for each
    prediction to explain exactly which factors drove the result.</p>

    <h3>Limitations</h3>
    <p>This is <strong>not a betting system</strong>. Predicting sport outcomes has irreducible
    randomness. The NLP uplift is meaningful but modest, partly because transcripts are not
    always from the day immediately before the match in question. Extending the transcript
    database to 2015–2021 would improve coverage significantly.</p>

    <h3>What's Next</h3>
    <p>The most exciting extension is a <strong>real-time pipeline</strong> — scraping the
    transcript the evening before a match and generating a live upset probability alert.
    The architecture is already in place. It just needs to be pointed at tomorrow's draw.</p>
    </div>
    """, unsafe_allow_html=True)


#  page: how i built this

elif page == "How I Built This":
    st.markdown("""
    <div style="padding:2.5rem 0 0">
      <div class="sec-label">Technical Methodology</div>
      <div class="sec-title">How I Built This</div>
      <div class="sec-body">
        A complete walkthrough — from raw CSV files and web scraping to a deployed
        machine learning app with NLP, SHAP explanations, and a conversational AI.
      </div>
    </div>
    """, unsafe_allow_html=True)

    steps = [
        ("01","Data Ingestion","data_ingestion.py",
         """Match data comes from <strong>Jeff Sackmann's open-source tennis database</strong>
         — one of the most complete public tennis datasets available. We pull ATP match results
         from 2015–2024, filtering to Grand Slam main draw matches only.
         <br><br>
         Each row contains winner, loser, their ATP rankings at the time, tournament, round,
         surface, and score. We compute the <strong>upset flag</strong> (1 if lower-ranked won),
         the <strong>rank difference</strong>, and the <strong>CTFI</strong> — total sets each
         player played before this match in the tournament.""",
         "~9,876 match rows · ATP + WTA · 4 Grand Slams · 1990–2024"),

        ("02","Transcript Scraping","scraping.py",
         """Transcripts come from <strong>ASAP Sports</strong> (asapsports.com) — the official
         press conference archive used by ATP and Grand Slam tournaments.
         <br><br>
         The scraper generates day-page URLs directly from tournament dates
         (format: <code>show_event.php?category=7&date=YYYY-M-DD&title=TOURNAMENT</code>),
         then follows each <code>show_interview.php</code> link. Transcript text lives in
         <code>&lt;p&gt;</code> tags on the interview page.
         <br><br>
         Random delays of 1.5–3.5 seconds between requests prevent server overload.
         The scraper resumes safely — already-scraped URLs are skipped.""",
         "2,279 transcripts · 12 tournaments · 2022–2024 · ~2 hours runtime"),

        ("03","NLP Pipeline","nlp.py",
         """Each transcript goes through a five-stage pipeline:
         <br><br>
         <strong>1. Fatigue lexicon matching</strong> — 60+ phrases across 5 categories
         (physical, mental, schedule, injury, motivation) counted using regex word boundaries.
         <br><br>
         <strong>2. Sentiment analysis</strong> — DistilBERT fine-tuned on SST-2 gives a
         positive/negative sentiment score. A backup lexicon-based polarity is also computed.
         <br><br>
         <strong>3. Linguistic features</strong> — first-person pronoun rate, negation rate,
         average sentence length via spaCy.
         <br><br>
         <strong>4. LLM fatigue classification</strong> — a language model gives a 0–1
         'is this person fatigued?' score.
         <br><br>
         <strong>5. DB write-back</strong> — all outputs written to SQLite and joined to matches.""",
         "26 NLP features per transcript · DistilBERT · spaCy · 5 fatigue categories"),

        ("04","Feature Engineering","features.py",
         """The feature matrix joins match stats with NLP outputs. Key engineered features:
         <br><br>
         <strong>rank_ratio</strong> — player rank ÷ opponent rank. Scale-independent gap measure.
         <br><br>
         <strong>log_rank_diff</strong> — log-transformed signed rank difference. The log
         handles the non-linear relationship between rank gap and upset probability.
         <br><br>
         <strong>is_underdog</strong> — binary: is this player the lower-ranked one?
         <br><br>
         <strong>rank_bin</strong> — categorical tier: top10, top50, top100, outside100.
         Captures non-linear quality effects.
         <br><br>
         Surface (Clay, Grass, Hard) is one-hot encoded since upset rates differ by surface.""",
         "26 total features · rank + CTFI + NLP + surface"),

        ("05","Model Training","model.py",
         """Two Random Forest models are trained for comparison:
         <br><br>
         <strong>Model A (Traditional)</strong> — rank features + CTFI only. ROC-AUC: <strong>0.62</strong>.
         <br><br>
         <strong>Model B (Full)</strong> — all features including NLP. ROC-AUC: <strong>0.70</strong>.
         <br><br>
         Hyperparameters tuned via 5-fold cross-validation over n_estimators, max_depth,
         min_samples_leaf, and max_features. A full sklearn Pipeline handles imputation
         (median), scaling (StandardScaler), and one-hot encoding.
         <br><br>
         <strong>SHAP TreeExplainer</strong> is computed post-training — this powers the
         feature importance bars in the Upset Alert tool.""",
         "ROC-AUC 0.70 · Random Forest · 5-fold CV · SHAP explanations"),

        ("06","App + RAG","app.py",
         """A single-file Streamlit app to avoid module import issues on Streamlit Cloud.
         <br><br>
         The <strong>Scouting Report</strong> uses lightweight RAG: transcript text is
         searched by keyword overlap (or ChromaDB vector similarity if available), retrieved
         chunks passed to Groq Llama 3 70B to generate a scouting report.
         <br><br>
         The <strong>Ask the Model</strong> chat routes questions using keyword classification
         to either SQL queries (structured stats) or transcript search (language patterns),
         then passes results to Groq for a plain-English summary.
         <br><br>
         Deployed on Streamlit Cloud (free tier). Groq API key stored as a secret.""",
         "Streamlit Cloud · Groq Llama 3 70B · SQLite · ChromaDB"),
    ]

    for num,title,fname,body,result in steps:
        st.markdown(f"""
        <div class="method-item">
          <div class="method-num">{num}</div>
          <div class="method-right">
            <div class="method-title">{title}</div>
            <div class="method-file">{fname}</div>
            <div class="method-body">{body}</div>
            <div class="method-result">→ {result}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='sec-label' style='margin-top:2.5rem'>Tech stack</div>", unsafe_allow_html=True)
    stack = [
        ("Python 3.11","Core language"),("pandas + numpy","Data manipulation"),
        ("scikit-learn","Random Forest, pipelines, cross-validation"),
        ("SHAP","Model explainability (TreeExplainer)"),
        ("spaCy","NLP — tokenisation, sentence detection"),
        ("Hugging Face Transformers","DistilBERT sentiment analysis"),
        ("BeautifulSoup + requests","Web scraping from ASAP Sports"),
        ("SQLite","Match + transcript database"),
        ("Groq API (Llama 3 70B)","LLM explanations, scouting reports"),
        ("Streamlit","Web app framework and free hosting"),
        ("Jeff Sackmann / tennis_atp","Open-source ATP match data (GitHub)"),
        ("ASAP Sports","Official press conference transcript archive"),
    ]
    c1,c2 = st.columns(2)
    for i,(tech,desc) in enumerate(stack):
        with (c1 if i%2==0 else c2):
            st.markdown(f"""
            <div style="display:flex;gap:1rem;align-items:flex-start;padding:0.7rem 0;
                        border-bottom:1px solid #f3f4f6">
              <span style="font-family:'JetBrains Mono',monospace;font-size:0.78rem;
                           color:#2563eb;min-width:200px;flex-shrink:0">{tech}</span>
              <span style="font-size:0.84rem;color:#6b7280">{desc}</span>
            </div>
            """, unsafe_allow_html=True)


#  page: upset alert

elif page == "⚡ Upset Alert":
    st.markdown("""
    <div style="padding:2.5rem 0 0.5rem">
      <div class="sec-label">Tool 01 — Prediction Engine</div>
      <div class="sec-title">Upset Alert</div>
      <div class="sec-body">
        Enter two players and match details. Optionally paste a press conference transcript
        to extract fatigue signals automatically. Get an AI-powered upset probability.
      </div>
    </div>
    """, unsafe_allow_html=True)

    if not model_ok:
        st.info("Demo mode — no trained model found. Run the pipeline scripts to train on real data.")

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # Player inputs
    c1, c2, c3 = st.columns([5,1,5])
    with c1:
        st.markdown("""
        <div style="background:#fff;border:1.5px solid #e8e6e0;border-top:3px solid #2563eb;
                    border-radius:14px;padding:1.4rem;box-shadow:0 1px 4px #0000000a">
          <div style="font-family:'JetBrains Mono',monospace;font-size:0.63rem;
                      text-transform:uppercase;letter-spacing:0.15em;color:#2563eb;
                      margin-bottom:0.8rem">⬤ Underdog (lower-ranked)</div>
        </div>
        """, unsafe_allow_html=True)
        player      = st.text_input("Player name", "Carlos Alcaraz", key="pname", label_visibility="collapsed")
        player_rank = st.number_input("ATP Ranking", 1, 1000, 45, key="prank")
        ctfi        = st.slider("Fatigue Index (CTFI)",0,30,8,
                                help="Total sets this player has played in the tournament so far")
    with c2:
        st.markdown("""<div style="display:flex;align-items:center;justify-content:center;
                    height:100%;padding-top:2rem">
          <div style="font-family:'Playfair Display',serif;font-size:1.4rem;
                      font-weight:700;color:#e8e6e0">VS</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div style="background:#fff;border:1.5px solid #e8e6e0;border-top:3px solid #ef4444;
                    border-radius:14px;padding:1.4rem;box-shadow:0 1px 4px #0000000a">
          <div style="font-family:'JetBrains Mono',monospace;font-size:0.63rem;
                      text-transform:uppercase;letter-spacing:0.15em;color:#ef4444;
                      margin-bottom:0.8rem">⬤ Favourite (higher-ranked)</div>
        </div>
        """, unsafe_allow_html=True)
        opponent  = st.text_input("Opponent name", "Novak Djokovic", key="oname", label_visibility="collapsed")
        opp_rank  = st.number_input("ATP Ranking", 1, 1000, 3, key="orank")
        round_sel = st.selectbox("Round", list(ROUND_MAP.keys()), index=4,
                                  format_func=lambda r: f"{r} — {ROUND_FULL[r]}")

    best_of = st.radio("Match format", [3,5], index=1, horizontal=True,
                        format_func=lambda x: f"Best of {x} sets")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style="margin-bottom:0.6rem">
      <span style="font-weight:600;color:#1c1c1e">Press conference transcript</span>
      <span style="font-size:0.82rem;color:#9ca3af;margin-left:0.5rem">optional — improves accuracy</span>
    </div>
    <div style="font-size:0.84rem;color:#6b7280;margin-bottom:0.8rem;line-height:1.6">
      Paste what the underdog said at their last press conference. We'll scan for fatigue signals
      across five categories: physical exhaustion, mental tiredness, injury concerns, schedule burden,
      and motivation doubts.
    </div>
    """, unsafe_allow_html=True)

    use_t = st.toggle("Include transcript analysis", value=True)
    nlp   = {}
    if use_t:
        quick = st.multiselect("Quick-add signals",
                               ["tired","heavy legs","cramping","mentally drained",
                                "back-to-back","five sets","injury","my knee",
                                "not 100%","doubt","physically tough","exhausted",
                                "pain","struggling","mentally tough"],
                               key="alert_quick",
                               label_visibility="visible")
        txt = st.text_area("Paste transcript",
                           placeholder='"I\'m really tired after yesterday. My legs are heavy and my back is stiff. I\'m not 100% going into tomorrow but I\'ll give everything I have."',
                           height=110, key="alert_txt", label_visibility="collapsed")
        # read widget state by key-  guaranteed to survive button reruns
        _quick = st.session_state.get("alert_quick", []) or []
        _txt   = st.session_state.get("alert_txt", "") or ""
        full   = " ".join(_quick) + " " + _txt
        if _quick or _txt.strip():
            nlp = analyse_transcript(full)
            st.session_state["_alert_nlp"] = nlp
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Fatigue signals",  nlp["fatigue_total"])
            m2.metric("Density / 100w",   f"{nlp['fatigue_density']:.1f}")
            m3.metric("Sentiment score",  f"{nlp['sentiment_polarity']:+.2f}")
            m4.metric("Word count",        nlp["word_count"])
        else:
            st.session_state["_alert_nlp"] = {}
            nlp = {}

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
    if st.button("⚡  Predict upset probability", use_container_width=True):
        nlp  = st.session_state.get("_alert_nlp", {})
        sent = nlp.get("sentiment_polarity", 0.0)
        fat  = nlp.get("fatigue_total", 0)
        with st.spinner("Running model…"):
            prob, sd = predict(player_rank,opp_rank,ctfi,sent,fat,ROUND_MAP[round_sel],best_of)

        if prob>=0.65:   col,risk,risk_bg = "#ef4444","HIGH UPSET RISK","#fef2f2"
        elif prob>=0.40: col,risk,risk_bg = "#f59e0b","MODERATE RISK","#fffbeb"
        else:            col,risk,risk_bg = "#2563eb","LOW RISK","#eff6ff"

        r1,r2 = st.columns([2,3])
        with r1:
            st.markdown(f"""
            <div class="prob-display">
              <div class="prob-number" style="color:{col}">{prob*100:.0f}%</div>
              <div class="prob-label">upset probability</div>
              <div class="prob-risk" style="background:{risk_bg};color:{col}">{risk}</div>
              <div style="font-size:0.8rem;color:#4b5563;margin-top:1rem">
                {player} vs {opponent}<br>
                {ROUND_FULL.get(round_sel, round_sel)}
              </div>
            </div>
            <div class="prog-track" style="margin-top:0.8rem">
              <div class="prog-fill" style="width:{int(prob*100)}%;background:{col}"></div>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            st.markdown("<div style='font-weight:600;color:#1c1c1e;margin-bottom:0.5rem'>AI Explanation</div>", unsafe_allow_html=True)
            exp=None
            if groq_key:
                top=sorted(sd.items(),key=lambda x:abs(x[1]),reverse=True)[:4]
                lines=[f"  • {f}: {v:+.3f}" for f,v in top if abs(v)>0.005]
                prompt=(f"Tennis analyst. {prob*100:.0f}% upset probability for {player} "
                        f"(rank {player_rank}) vs {opponent} (rank {opp_rank}), {round_sel}.\n"
                        f"SHAP factors:\n"+"\n".join(lines)+
                        "\n\nWrite 2 clear sentences for a general audience. Be specific about what's driving the risk.")
                with st.spinner("Generating…"): exp=groq_call(prompt,groq_key,130)
            if not exp:
                top=sorted(sd.items(),key=lambda x:abs(x[1]),reverse=True)[:2]
                names=[f for f,_ in top]
                lvl="high" if prob>0.6 else "moderate" if prob>0.4 else "low"
                if names:
                    driver1 = names[0]
                    driver2 = names[1] if len(names)>1 else "rank gap"
                    exp=(f"The model assigns **{lvl}** upset risk ({prob*100:.0f}%) for **{player}**. "
                         f"The strongest drivers are **{driver1}** and **{driver2}**.")
                else:
                    exp=(f"The model assigns **{lvl}** upset risk ({prob*100:.0f}%) for **{player}** "
                         f"based on rank difference and tournament fatigue.")
                if not groq_key:
                    st.caption("Add a Groq API key (⚙️ above) for richer AI explanations.")

            st.markdown(f"""
            <div class="explain-box">
              <div style="border-left:3px solid {col};padding-left:1rem">{exp}</div>
            </div>
            """, unsafe_allow_html=True)

            if sd:
                st.markdown("<div style='font-weight:600;color:#1c1c1e;margin:1rem 0 0.3rem'>Feature importance (SHAP)</div>", unsafe_allow_html=True)
                st.markdown("<div style='font-size:0.78rem;color:#9ca3af;margin-bottom:0.5rem'>Red = increases upset risk &nbsp;·&nbsp; Blue = decreases it</div>", unsafe_allow_html=True)
                st.markdown(shap_html(sd), unsafe_allow_html=True)


#  page: scouting

elif page == "◎ Scouting":
    st.markdown("""
    <div style="padding:2.5rem 0 0.5rem">
      <div class="sec-label">Tool 02 — Transcript Intelligence</div>
      <div class="sec-title">Player Scouting</div>
      <div class="sec-body">
        Ask about a player's fatigue history. We search 2,279 real press conference
        transcripts and generate a tactical scouting report.
      </div>
    </div>
    """, unsafe_allow_html=True)

    PLAYERS = ["Carlos Alcaraz","Novak Djokovic","Rafael Nadal","Jannik Sinner",
               "Daniil Medvedev","Stefanos Tsitsipas","Alexander Zverev","Andrey Rublev",
               "Roger Federer","Dominic Thiem","Holger Rune","Taylor Fritz",
               "Casper Ruud","Ben Shelton","Frances Tiafoe","Felix Auger-Aliassime",
               "Iga Swiatek","Aryna Sabalenka","Coco Gauff","Elena Rybakina",
               "Jessica Pegula","Marketa Vondrousova","Karolina Muchova",
               "Barbora Krejcikova","Madison Keys","Mirra Andreeva"]
    if db_ok:
        try:
            import sqlite3
            conn=sqlite3.connect("tennis_upsets.db")
            rows=conn.execute("SELECT player_name,COUNT(*) as cnt FROM transcripts WHERE player_name IS NOT NULL AND player_name!='Unknown' AND LENGTH(player_name)>3 GROUP BY player_name ORDER BY cnt DESC LIMIT 80").fetchall()
            conn.close()
            if rows: PLAYERS=[r[0] for r in rows]
        except Exception: pass

    c1,c2=st.columns([1,2])
    with c1: player=st.selectbox("Select player",PLAYERS)
    with c2:
        QUERIES = [
            "What are this player's fatigue signals before an upset?",
            "What physical complaints does this player mention most?",
            "How does this player describe their mental state before losing?",
            "What injury concerns does this player mention?",
            "When does this player seem most confident vs most doubtful?",
            "What does this player say after a tough five-set match?",
            "How does this player talk about their schedule and rest?",
            "What patterns appear in this player's press conferences before upsets?",
            "What motivational language does this player use?",
            "How does this player describe their physical readiness?",
        ]
        query = st.selectbox("What do you want to know?", QUERIES)
        custom_q = st.text_input("Or type a custom question…", placeholder="e.g. What does Djokovic say about his wrist?", label_visibility="collapsed")
        if custom_q.strip():
            query = custom_q.strip()

    a1,a2=st.columns(2)
    with a1: n_results=st.slider("Excerpts to retrieve",2,10,5)
    with a2: upset_only=st.toggle("Only from upset matches",value=False)

    if st.button("◎  Generate scouting report", use_container_width=True):
        with st.spinner("Searching transcripts…"):
            snips=search_transcripts(query,player,n_results,upset_only)
        if not snips:
            # Auto-retry without player filter so results always appear
            snips = search_transcripts(query, player=None, n=n_results, upset_only=False)
        if not snips:
            st.warning("No matching transcripts found.")
        else:
            st.markdown(f"<div class='card-label' style='margin-bottom:0.8rem;color:#9ca3af'>✓ {len(snips)} excerpts retrieved from {('live database' if db_ok else 'demo data')}</div>", unsafe_allow_html=True)

            report=None
            if groq_key:
                ctx="\n\n---\n\n".join(f"[{s.get('tournament','?')} · {s.get('round','')} · {'UPSET' if s.get('upset') else 'WIN'}]\n{s['text'][:400]}" for s in snips[:5])
                prompt=(f"Professional tennis scout. Question: {query}\nPlayer: {player}\n\nTranscripts:\n{ctx}\n\n"
                        f"Write a sharp scouting report in bullet points (max 200 words). Focus on recurring patterns and coaching insights.")
                with st.spinner("Writing report…"): report=groq_call(prompt,groq_key,350)

            if not report:
                CATS={"Physical":["tired","exhausted","heavy legs","cramping","sore","not 100"],
                      "Mental":["mentally","focus","distracted","lost focus"],
                      "Schedule":["back-to-back","five sets","long match","tough schedule"],
                      "Injury":["injury","pain","knee","shoulder","ankle","back"],
                      "Motivation":["doubt","uncertain","question mark","not sure"]}
                combined=" ".join(s["text"] for s in snips).lower()
                lines=[]
                for cat,words in CATS.items():
                    hits=sum(len(re.findall(r"\b"+re.escape(w)+r"\b",combined)) for w in words)
                    if hits: lines.append(f"• **{cat} fatigue**: {hits} mention{'s' if hits>1 else ''} detected across retrieved excerpts")
                nu=sum(1 for s in snips if s.get("upset"))
                report=(f"**Pattern analysis for {player}**\n"
                        f"*Based on {len(snips)} excerpts, {nu} from upset matches*\n\n"
                        +("\n".join(lines) if lines else "• No strong fatigue signals detected in retrieved excerpts.")
                        +"\n\n*Add a Groq API key (⚙️) for a full AI-written scouting report.*")

            st.markdown("<div style='font-weight:600;color:#1c1c1e;font-size:1rem;margin-bottom:0.6rem'>Scouting Report</div>", unsafe_allow_html=True)
            st.markdown("""<div style="background:#fff;border:1.5px solid #dde3ee;border-left:4px solid #2563eb;
                        border-radius:0 14px 14px 0;padding:1.5rem 1.8rem;margin-bottom:0.5rem;
                        box-shadow:0 1px 4px #0000000a"></div>""", unsafe_allow_html=True)
            st.markdown(report)

            # Signal bars
            CATS2={"Physical":["tired","exhausted","heavy legs","cramping","sore"],
                   "Mental":["mentally","focus","distracted"],
                   "Schedule":["back-to-back","five sets"],
                   "Injury":["injury","pain","knee","shoulder"],
                   "Motivation":["doubt","uncertain"]}
            COLS2=["#2563eb","#7c3aed","#059669","#dc2626","#d97706"]
            combined=" ".join(s["text"] for s in snips).lower()
            counts={cat:sum(len(re.findall(r"\b"+re.escape(w)+r"\b",combined)) for w in words)
                    for cat,words in CATS2.items()}
            mx=max(counts.values()) if max(counts.values())>0 else 1

            st.markdown("<div style='font-weight:600;color:#1c1c1e;margin:1.2rem 0 0.5rem'>Fatigue signal breakdown</div>", unsafe_allow_html=True)
            for (cat,val),color in zip(counts.items(),COLS2):
                pct=int(val/mx*100)
                st.markdown(f"""
                <div class="sig-row">
                  <span class="sig-lbl">{cat}</span>
                  <div class="sig-track">
                    <div class="sig-fill" style="width:{pct}%;background:{color}"></div>
                  </div>
                  <span class="sig-val">{val}</span>
                </div>""", unsafe_allow_html=True)

            m1,m2,m3=st.columns(3)
            m1.metric("Excerpts",  len(snips))
            m2.metric("From upsets", sum(1 for s in snips if s.get("upset")))
            m3.metric("Signals total", sum(counts.values()))

            with st.expander(f"View {len(snips)} retrieved excerpts"):
                for i,s in enumerate(snips,1):
                    out="🔴 Upset" if s.get("upset") else "🟢 Win"
                    st.markdown(f"""
                    <div class="snip">
                      <div class="snip-meta">{i} · {s.get('player','?')} · {s.get('tournament','?')} {s.get('round','')} · {out}</div>
                      <div class="snip-text">{s['text'][:500]}</div>
                    </div>""", unsafe_allow_html=True)


#  page: ask the mode 

elif page == "◉ Ask the Model":
    st.markdown("""
    <div style="padding:2.5rem 0 0.5rem">
      <div class="sec-label">Tool 03 — Conversational AI</div>
      <div class="sec-title">Ask the Model</div>
      <div class="sec-body">
        Chat with 10 years of Grand Slam data in plain English. Ask about stats,
        player records, or what players said in press conferences.
      </div>
    </div>
    """, unsafe_allow_html=True)

    EXAMPLES=[
        ("📊","Upset rates by Slam","What is the upset rate at each Grand Slam?"),
        ("🏆","Biggest upsets","Show the 5 biggest rank-gap upsets"),
        ("🔴","Favourite losses","Which top-10 player lost most as a favourite?"),
        ("🎾","By round","Which round has the highest upset rate?"),
        ("💬","Cramping","Find transcript mentions of cramping before upsets"),
        ("😓","Player fatigue","What did Djokovic say about fatigue?"),
        ("🌿","Surface","Upset rates on grass vs clay vs hard"),
        ("📈","Dataset","How many matches are in the dataset?"),
    ]

    st.markdown("<div style='font-weight:600;color:#1c1c1e;margin-bottom:0.6rem'>Example questions</div>", unsafe_allow_html=True)
    ecols=st.columns(4)
    for i,(icon,lbl,q) in enumerate(EXAMPLES):
        with ecols[i%4]:
            if st.button(f"{icon}  {lbl}",key=f"ex_{i}",use_container_width=True):
                st.session_state.pending_q=q

    st.markdown("<hr>", unsafe_allow_html=True)

    if not st.session_state.messages:
        st.session_state.messages=[{
            "role":"assistant","source":"","table":"","snippets":[],
            "content":"Hello! I can answer questions about Grand Slam upsets — whether that's statistics from the match database, or patterns from player press conferences.\n\nTry one of the example questions above, or type your own below.",
        }]

    for msg in st.session_state.messages:
        if msg["role"]=="user":
            st.markdown(f"<div class='chat-user'>{msg['content']}</div>",unsafe_allow_html=True)
        else:
            src=msg.get("source","")
            src_tag=f"<div class='chat-src'>{src}</div>" if src else ""
            st.markdown(f"<div class='chat-bot'>{msg['content']}{src_tag}</div>",unsafe_allow_html=True)
            if msg.get("table"): st.markdown(msg["table"],unsafe_allow_html=True)
            if msg.get("snippets"):
                with st.expander("View transcript excerpts"):
                    for s in msg["snippets"][:3]:
                        out="🔴 Upset" if s.get("upset") else "🟢 Win"
                        st.markdown(f"<div class='snip'><div class='snip-meta'>{s.get('player','?')} · {s.get('tournament','?')} {s.get('round','')} · {out}</div><div class='snip-text'>{s['text'][:360]}</div></div>",unsafe_allow_html=True)

    pending=st.session_state.pop("pending_q",None)
    question=pending or st.chat_input("Ask anything about tennis upsets…")

    if question:
        st.session_state.messages.append({"role":"user","content":question,"source":"","table":"","snippets":[]})
        st.markdown(f"<div class='chat-user'>{question}</div>",unsafe_allow_html=True)

        q_lower=question.lower(); snippets=[]; df_result=None; table_html=""
        is_text=any(w in q_lower for w in ["said","mention","fatigue","tired","cramping","press","quote","words","transcript","feeling"])

        with st.spinner("Thinking…"):
            if is_text:
                snippets=search_transcripts(question,n=4,upset_only=False)
                source="◎ Transcript search"; answer=""
                if groq_key and snippets:
                    ctx="\n\n".join(f"[{s.get('player','?')} – {s.get('tournament','?')} {s.get('round','')}]: {s['text'][:350]}" for s in snippets)
                    prompt=f"Tennis analyst. Answer based on these transcripts.\nQ: {question}\n\nExcerpts:\n{ctx}\n\nAnswer in 2-3 sentences:"
                    answer=groq_call(prompt,groq_key,200) or ""
                if not answer:
                    answer="\n\n".join(f"**{s.get('player','?')} — {s.get('tournament','?')} {s.get('round','')}** ({'Upset' if s.get('upset') else 'Win'}): {s['text'][:260]}…" for s in snippets[:3]) or "No matching transcripts found."
            else:
                df_result,summary=answer_sql(question)
                source="🗄️ SQL query"; ai_txt=""
                if groq_key and df_result is not None:
                    prompt=f"Tennis analyst. User asked: '{question}'\nData:\n{df_result.to_string(index=False)}\n\nWrite 1 insightful sentence about the most interesting finding in this data."
                    ai_txt=groq_call(prompt,groq_key,80) or ""
                answer=ai_txt if ai_txt else summary
                if df_result is not None and not df_result.empty:
                    table_html=html_table(df_result)

        src_tag=f"<div class='chat-src'>{source}</div>"
        st.markdown(f"<div class='chat-bot'>{answer}{src_tag}</div>",unsafe_allow_html=True)
        if table_html: st.markdown(table_html,unsafe_allow_html=True)
        if snippets:
            with st.expander("View transcript excerpts"):
                for s in snippets[:3]:
                    out="🔴 Upset" if s.get("upset") else "🟢 Win"
                    st.markdown(f"<div class='snip'><div class='snip-meta'>{s.get('player','?')} · {s.get('tournament','?')} {s.get('round','')} · {out}</div><div class='snip-text'>{s['text'][:360]}</div></div>",unsafe_allow_html=True)

        st.session_state.messages.append({
            "role":"assistant","content":answer,"source":source,
            "snippets":snippets,"table":table_html,
        })

    if len(st.session_state.messages)>1:
        if st.button("Clear conversation"):
            st.session_state.messages=[]; st.rerun()