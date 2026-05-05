"""
GrandSlam IQ - Complete Single-File Streamlit App
Pages: Home · Blog · How I Built This · Upset Alert · Scouting · Ask the Model
"""

import streamlit as st
import os, re, numpy as np

st.set_page_config(
    page_title="GrandSlam IQ",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="collapsed",
)


#  CSS

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&family=DM+Serif+Display:ital@0;1&family=JetBrains+Mono:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: #f5f4f0;
    color: #1a1a2e;
}
.stApp { background: #f5f4f0; }
section[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
header[data-testid="stHeader"] { display: none !important; }
#MainMenu { display: none !important; }
footer { display: none !important; }
.stDeployButton { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
.block-container { padding: 0 2.5rem 5rem !important; max-width: 1100px !important; }

/* ── NAV ── */
.topnav {
    display: flex; align-items: center; justify-content: space-between;
    padding: 1rem 0 0.9rem; border-bottom: 1px solid #ddddd8;
    margin-bottom: 0; position: sticky; top: 0;
    background: #f5f4f0; z-index: 200;
}
.logo {
    font-family: 'Bebas Neue', sans-serif; font-size: 1.5rem;
    letter-spacing: 0.1em; color: #1a1a2e; line-height: 1;
}
.logo em { color: #2d2dff; font-style: normal; }
.navlinks { display: flex; gap: 0.2rem; flex-wrap: wrap; }
.nb {
    padding: 0.38rem 0.85rem; border-radius: 7px; font-size: 0.82rem;
    font-weight: 500; color: #888880; border: 1px solid transparent;
    cursor: pointer; transition: all 0.15s; white-space: nowrap;
}
.nb:hover { color: #1a1a2e; background: #ebebeb; border-color: #ddddd8; }
.nb.on { color: #2d2dff; background: #2d2dff0f; border-color: #2d2dff30; }

/* ── HERO ── */
.hero { padding: 4rem 0 2.5rem; }
.eyebrow {
    font-family: 'JetBrains Mono', monospace; font-size: 0.68rem;
    letter-spacing: 0.2em; text-transform: uppercase; color: #aaaaaa;
    margin-bottom: 0.8rem;
}
.headline {
    font-family: 'Bebas Neue', sans-serif;
    font-size: clamp(3rem, 8vw, 6rem);
    letter-spacing: 0.04em; line-height: 0.93; color: #1a1a2e;
    margin-bottom: 1rem;
}
.headline .y  { color: #2d2dff; }
.headline .b  { color: #2d2dff; }
.headline .r  { color: #e63946; }
.subhead {
    font-size: 1.05rem; color: #777770; line-height: 1.7;
    max-width: 520px; font-weight: 300; margin-bottom: 2rem;
}

/* ── STAT ROW ── */
.stats {
    display: grid; grid-template-columns: repeat(4,1fr);
    gap: 1px; background: #ddddd8;
    border-radius: 12px; overflow: hidden; margin: 2rem 0;
}
.sc { background: #ffffff; padding: 1.3rem 1.2rem; text-align: center; }
.sn {
    font-family: 'Bebas Neue', sans-serif; font-size: 2.4rem;
    letter-spacing: 0.04em; color: #2d2dff; line-height: 1;
}
.sl {
    font-family: 'JetBrains Mono', monospace; font-size: 0.62rem;
    text-transform: uppercase; letter-spacing: 0.12em;
    color: #bbbbbb; margin-top: 0.3rem;
}

/* ── CARDS ── */
.card {
    background: #ffffff; border: 1px solid #e8e8e4;
    border-radius: 14px; padding: 1.5rem 1.7rem;
    margin-bottom: 1rem; transition: border-color 0.2s;
}
.card:hover { border-color: #cccccc; }
.card-hi { border-color: #2d2dff20; background: #fafaff; }

.clabel {
    font-family: 'JetBrains Mono', monospace; font-size: 0.66rem;
    text-transform: uppercase; letter-spacing: 0.14em; margin-bottom: 0.5rem;
    color: #aaaaaa;
}
.ctitle {
    font-family: 'Bebas Neue', sans-serif; font-size: 1.15rem;
    letter-spacing: 0.05em; margin-bottom: 0.4rem; color: #1a1a2e;
}
.cbody { font-size: 0.84rem; color: #888880; line-height: 1.65; }

/* ── THREE TOOLS GRID ── */
.tools { display: grid; grid-template-columns: repeat(3,1fr); gap: 1rem; margin: 1.5rem 0; }
.tool-card {
    background: #ffffff; border: 1px solid #e8e8e4;
    border-radius: 14px; padding: 1.5rem 1.4rem;
    transition: all 0.2s; cursor: default;
}
.tool-card:hover { border-color: #cccccc; box-shadow: 0 4px 24px #0000000a; transform: translateY(-2px); }
.ti { font-size: 2rem; margin-bottom: 0.9rem; }
.tn { font-family: 'Bebas Neue', sans-serif; font-size: 1.1rem; letter-spacing: 0.06em; margin-bottom: 0.5rem; color: #1a1a2e; }
.td { font-size: 0.82rem; color: #888880; line-height: 1.6; }
.tags { margin-top: 0.9rem; }
.tag {
    display: inline-block; font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem; padding: 2px 7px; background: #f0f0ee;
    border: 1px solid #e0e0dc; border-radius: 4px; color: #aaaaaa;
    margin: 2px 2px 0 0; letter-spacing: 0.04em;
}

/* ── BLOG ── */
.blog-hero {
    background: #ffffff; border: 1px solid #e8e8e4;
    border-radius: 16px; padding: 2.5rem 2.8rem; margin-bottom: 2rem;
}
.blog-title {
    font-family: 'DM Serif Display', serif; font-size: clamp(1.6rem, 4vw, 2.8rem);
    line-height: 1.2; color: #1a1a2e; margin-bottom: 0.8rem;
}
.blog-meta {
    font-family: 'JetBrains Mono', monospace; font-size: 0.68rem;
    color: #bbbbbb; text-transform: uppercase; letter-spacing: 0.12em;
    margin-bottom: 1.2rem;
}
.blog-body { font-size: 0.94rem; color: #555550; line-height: 1.8; }
.blog-body p { margin-bottom: 1rem; }
.blog-body h3 {
    font-family: 'Bebas Neue', sans-serif; font-size: 1.1rem;
    letter-spacing: 0.08em; color: #2d2dff; margin: 1.5rem 0 0.5rem;
}
.blog-body strong { color: #1a1a2e; }
.blog-body em { color: #2d2dff; font-style: normal; font-weight: 600; }
.blog-pull {
    border-left: 3px solid #2d2dff; padding: 0.8rem 1.2rem;
    margin: 1.2rem 0; font-size: 1.05rem; color: #333330;
    font-style: italic; line-height: 1.6; background: #f5f5ff;
    border-radius: 0 8px 8px 0;
}
.finding {
    background: #ffffff; border: 1px solid #e8e8e4; border-radius: 10px;
    padding: 1rem 1.2rem; margin: 0.6rem 0;
    display: flex; gap: 1rem; align-items: flex-start;
}
.fnum {
    font-family: 'Bebas Neue', sans-serif; font-size: 1.6rem;
    color: #2d2dff; opacity: 0.25; line-height: 1; flex-shrink: 0;
    width: 2rem; text-align: center;
}
.ftext { font-size: 0.86rem; color: #666660; line-height: 1.6; }
.ftext strong { color: #1a1a2e; }

/*  METHODOLOGY  */
.method-step {
    display: flex; gap: 1.5rem; align-items: flex-start;
    padding: 1.5rem 0; border-bottom: 1px solid #eeeeea;
}
.method-num {
    font-family: 'Bebas Neue', sans-serif; font-size: 3rem;
    color: #2d2dff; opacity: 0.12; line-height: 1;
    flex-shrink: 0; width: 3.5rem; text-align: right;
}
.method-content { flex: 1; }
.method-title {
    font-family: 'Bebas Neue', sans-serif; font-size: 1.1rem;
    letter-spacing: 0.06em; margin-bottom: 0.4rem; color: #1a1a2e;
}
.method-body { font-size: 0.85rem; color: #666660; line-height: 1.7; }
.method-body strong { color: #333330; }
.code-pill {
    display: inline-block; background: #f0f0ee; border: 1px solid #e0e0dc;
    border-radius: 5px; padding: 2px 8px;
    font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
    color: #2d2dff; margin: 2px;
}
.result-box {
    background: #f8f8ff; border-left: 3px solid #2d2dff;
    border-radius: 0 8px 8px 0; padding: 0.8rem 1rem;
    font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;
    color: #6666aa; line-height: 1.6; margin: 0.8rem 0;
}

/*  PROB  */
.probnum {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 5.5rem; letter-spacing: 0.03em; line-height: 1;
}
.problbl {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem; letter-spacing: 0.18em; text-transform: uppercase;
    color: #bbbbbb; margin-top: 0.4rem;
}

/*  SHAP  */
.shap-row { display:flex; align-items:center; gap:8px; margin:4px 0; }
.shap-lbl {
    width:155px; text-align:right; font-family:'JetBrains Mono',monospace;
    font-size:0.68rem; color:#bbbbbb; flex-shrink:0;
}
.shap-track { flex:1; background:#eeeeea; border-radius:3px; height:11px; overflow:hidden; }
.shap-pos { height:100%; border-radius:3px; background:#e63946; }
.shap-neg { height:100%; border-radius:3px; background:#2d2dff; }
.shap-val { font-family:'JetBrains Mono',monospace; font-size:0.67rem; width:50px; flex-shrink:0; }

/*  SNIPPET  */
.snip {
    background: #fafaf8; border-left: 2px solid #2d2dff;
    padding: 0.8rem 1rem; margin: 0.5rem 0;
    border-radius: 0 8px 8px 0; font-size: 0.83rem;
    line-height: 1.65; color: #555550;
}
.snip-meta {
    font-family: 'JetBrains Mono', monospace; font-size: 0.63rem;
    color: #cccccc; text-transform: uppercase;
    letter-spacing: 0.1em; margin-bottom: 0.35rem;
}

/*  CHAT  */
.cu {
    background: #2d2dff; border-radius: 12px 12px 3px 12px;
    padding: 0.75rem 1rem; margin: 0.4rem 0 0.4rem auto;
    max-width: 74%; font-size: 0.88rem; color: #ffffff;
}
.cb {
    background: #ffffff; border: 1px solid #e8e8e4;
    border-radius: 3px 12px 12px 12px;
    padding: 0.8rem 1rem; margin: 0.4rem 0;
    max-width: 80%; font-size: 0.87rem; line-height: 1.65; color: #444440;
}
.csrc {
    font-family: 'JetBrains Mono', monospace; font-size: 0.6rem;
    color: #cccccc; margin-top: 0.4rem; letter-spacing: 0.08em;
    text-transform: uppercase;
}
.ctable { width: 100%; border-collapse: collapse; margin-top: 0.8rem; font-size: 0.82rem; }
.ctable th {
    text-align: left; padding: 5px 10px;
    font-family: 'JetBrains Mono', monospace; font-size: 0.62rem;
    text-transform: uppercase; letter-spacing: 0.1em; color: #bbbbbb;
    border-bottom: 1px solid #e8e8e4;
}
.ctable td { padding: 7px 10px; color: #555550; border-bottom: 1px solid #f0f0ee; }
.ctable tr:last-child td { border-bottom: none; }

/*  STREAMLIT OVERRIDES  */
.stButton > button {
    background: #1a1a2e !important; color: #ffffff !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important; font-weight: 600 !important;
    border: none !important; border-radius: 8px !important;
    padding: 0.55rem 1.5rem !important; transition: all 0.15s !important;
    letter-spacing: 0.02em !important;
}
.stButton > button:hover { background: #2d2dff !important; transform: translateY(-1px) !important; }
.stTextInput input, .stTextArea textarea, .stNumberInput input {
    background: #ffffff !important; border: 1px solid #e0e0dc !important;
    border-radius: 8px !important; color: #1a1a2e !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #2d2dff88 !important;
    box-shadow: 0 0 0 2px #2d2dff18 !important;
}
.stSelectbox > div > div {
    background: #ffffff !important; border: 1px solid #e0e0dc !important;
    border-radius: 8px !important; color: #1a1a2e !important;
}
label, [data-testid="stWidgetLabel"] {
    color: #999990 !important; font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important; text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}
div[data-testid="stMetricValue"] {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 2rem !important; color: #2d2dff !important;
}
div[data-testid="stMetricLabel"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.63rem !important; text-transform: uppercase !important;
    letter-spacing: 0.12em !important; color: #bbbbbb !important;
}
.stProgress > div > div { background: #2d2dff !important; }
[data-testid="stExpander"] {
    background: #ffffff !important; border: 1px solid #e8e8e4 !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary { color: #888880 !important; }
hr { border-color: #e8e8e4 !important; }
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #f5f4f0; }
::-webkit-scrollbar-thumb { background: #ddddd8; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


#  STATE & DATA STATUS

if "page"     not in st.session_state: st.session_state.page     = "Home"
if "messages" not in st.session_state: st.session_state.messages = []

db_ok    = os.path.exists("tennis_upsets.db") and os.path.getsize("tennis_upsets.db") > 10_000
model_ok = os.path.exists("upset_model.pkl")


#  NAV

PAGES = ["Home","Blog","How I Built This","⚡ Upset Alert","◎ Scouting","◉ Ask the Model"]

st.markdown("""
<div class="topnav">
  <div class="logo">Grand<em>Slam</em> IQ</div>
  <div class="navlinks" id="navlinks"></div>
</div>
""", unsafe_allow_html=True)

nav_cols = st.columns([2,1,1,1,1,1,1])
labels   = ["", "Home", "Blog", "How I Built This", "⚡ Alert", "◎ Scout", "◉ Chat"]
page_map = {"": None, "Home":"Home", "Blog":"Blog",
            "How I Built This":"How I Built This",
            "⚡ Alert":"⚡ Upset Alert",
            "◎ Scout":"◎ Scouting",
            "◉ Chat":"◉ Ask the Model"}

for i, (col, label) in enumerate(zip(nav_cols, labels)):
    if not label: continue
    with col:
        if st.button(label, key=f"nav_{i}", use_container_width=True):
            st.session_state.page = page_map[label]
            st.rerun()

# settings expander - only place Groq key appears
with st.expander("⚙️ Settings", expanded=False):
    gk = st.text_input(
        "Groq API key",
        type="password",
        placeholder="gsk_... — free at console.groq.com",
        value=st.session_state.get("groq_key", ""),
        key="groq_global",
        help="Powers AI explanations and scouting reports. Get a free key at console.groq.com"
    )
    if gk:
        st.session_state.groq_key = gk

st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
page     = st.session_state.page
groq_key = st.session_state.get("groq_key", "")

#  SHARED HELPERS
ROUND_MAP = {"R1":1,"R2":2,"R3":3,"R4":4,"QF":5,"SF":6,"F":7}

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
           "negation_rate":0.05,"llm_is_fatigued":0.5,
           "rank_bin":"top100" if player_rank<=100 else "outside100"}
    shap_dict = {}
    if model_ok:
        try:
            import joblib, pandas as pd
            pkg  = joblib.load("upset_model.pkl")
            pipe = pkg["model_b"]; cols = pkg["full_cols"]
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
                shap_dict = {c:float(v) for c,v in zip(list(nc)+list(cc),vals)}
            except Exception: pass
            return prob, shap_dict
        except Exception: pass
    rank_diff = player_rank-opp_rank
    raw  = 0.5+0.003*rank_diff-0.005*ctfi+0.05*fatigue-0.1*sentiment+np.random.normal(0,0.05)
    prob = float(np.clip(1/(1+np.exp(-raw*0.5)),0.05,0.95))
    shap_dict = {"rank_ratio":round(0.3*(player_rank/max(opp_rank,1)-1),3),
                 "ctfi":round(-0.02*ctfi,3),"fatigue_total":round(0.04*fatigue,3),
                 "sentiment_polarity":round(-0.08*sentiment,3),"round_num":round(0.01*round_num,3)}
    return prob, shap_dict

def shap_html(sd):
    top = sorted(sd.items(), key=lambda x:abs(x[1]), reverse=True)[:7]
    mx  = max(abs(v) for _,v in top) if top else 1
    out = "<div style='margin-top:0.5rem'>"
    for feat,val in top:
        pct = int(abs(val)/mx*100); col = "#ff4466" if val>0 else "#5252ff"
        sign= "+" if val>0 else "−"; cls = "shap-pos" if val>0 else "shap-neg"
        out += (f"<div class='shap-row'><span class='shap-lbl'>{feat.replace('_',' ')}</span>"
                f"<div class='shap-track'><div class='{cls}' style='width:{pct}%'></div></div>"
                f"<span class='shap-val' style='color:{col}'>{sign}{abs(val):.3f}</span></div>")
    return out+"</div>"

def groq_call(prompt, groq_key, max_tokens=150):
    try:
        from groq import Groq
        r = Groq(api_key=groq_key).chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role":"user","content":prompt}],
            temperature=0.25, max_tokens=max_tokens)
        return r.choices[0].message.content.strip()
    except Exception: return None

def search_transcripts(query, player=None, n=5, upset_only=True):
    DEMO = [
        {"player":"Carlos Alcaraz","tournament":"Wimbledon","round":"QF","upset":1,"rank_diff":15,
         "text":"I'm honestly feeling very tired. Yesterday's match took everything out of me — five sets, almost four hours. My legs are really heavy and I've been cramping. I haven't slept well and mentally I'm completely drained."},
        {"player":"Carlos Alcaraz","tournament":"Roland Garros","round":"SF","upset":1,"rank_diff":22,
         "text":"It was really tough physically. I had a lot of tension in my legs. I'm not 100 percent right now. Back-to-back five-set matches — my knee was really bothering me from the second set."},
        {"player":"Novak Djokovic","tournament":"Australian Open","round":"SF","upset":1,"rank_diff":30,
         "text":"The wrist has been bothering me since the third round. I'm taking painkillers just to get through matches. Mentally it is draining when you're always thinking about the injury."},
        {"player":"Novak Djokovic","tournament":"Wimbledon","round":"QF","upset":0,"rank_diff":-12,
         "text":"I feel very confident. My game is in good shape, movement is great. I'm sleeping nine hours and my body feels completely refreshed."},
        {"player":"Rafael Nadal","tournament":"Australian Open","round":"QF","upset":1,"rank_diff":10,
         "text":"My abs are very painful. I've had medical timeouts twice this week. I have real doubts about finishing the tournament healthy."},
        {"player":"Stefanos Tsitsipas","tournament":"Australian Open","round":"F","upset":0,"rank_diff":-8,
         "text":"My body is holding up well. I feel fresh and my serve has been clicking all tournament. Looking forward to the challenge tomorrow."},
    ]
    try:
        import sqlite3
        if db_ok:
            conn = sqlite3.connect("tennis_upsets.db")
            q    = "SELECT player_name, tourney_name, round, raw_text FROM transcripts WHERE raw_text IS NOT NULL"
            rows = conn.execute(q).fetchall(); conn.close()
            docs = [{"player":r[0],"tournament":r[1],"round":r[2],"upset":0,"rank_diff":0,"text":(r[3] or "")[:600]}
                    for r in rows]
            if player:
                filtered = [d for d in docs if player.lower() in (d["player"] or "").lower()]
                docs = filtered if filtered else docs
            qw = set(re.findall(r"\w+", query.lower()))
            docs.sort(key=lambda d:-len(qw&set(re.findall(r"\w+",d["text"].lower()))))
            return docs[:n]
    except Exception: pass
    docs = DEMO
    if player:
        filtered = [d for d in docs if player.lower() in d["player"].lower()]
        docs = filtered if filtered else docs
    if upset_only:
        docs2 = [d for d in docs if d.get("upset")]
        docs  = docs2 if docs2 else docs
    return docs[:n]

def answer_sql(question):
    try:
        import sqlite3, pandas as pd
        if not db_ok: raise Exception("no db")
        conn = sqlite3.connect("tennis_upsets.db"); q = question.lower()
        if "upset rate" in q and ("slam" in q or "tournament" in q or "grand" in q):
            df = pd.read_sql("""SELECT slam_name AS Tournament, COUNT(*) AS Matches,
                SUM(upset) AS Upsets, ROUND(AVG(upset)*100,1) AS "Upset Rate %"
                FROM matches GROUP BY slam_name ORDER BY "Upset Rate %" DESC""", conn)
            summary = "Upset rate by Grand Slam — higher means more unpredictable:"
        elif "biggest" in q or ("rank" in q and "gap" in q) or "largest" in q:
            df = pd.read_sql("""SELECT winner_name AS Winner, loser_name AS Loser,
                CAST(rank_diff AS INT) AS "Rank Gap", slam_name AS Tournament,
                round AS Round, SUBSTR(tourney_date,1,4) AS Year
                FROM matches WHERE upset=1 ORDER BY rank_diff DESC LIMIT 10""", conn)
            summary = "Biggest upsets by rank gap (2022–2024):"
        elif "round" in q and ("upset" in q or "rate" in q or "high" in q):
            df = pd.read_sql("""SELECT round AS Round, COUNT(*) AS Matches,
                ROUND(AVG(upset)*100,1) AS "Upset Rate %"
                FROM matches GROUP BY round ORDER BY "Upset Rate %" DESC""", conn)
            summary = "Upset rate by tournament round:"
        elif "how many" in q or "count" in q or "total" in q or "dataset" in q:
            df = pd.read_sql("""SELECT COUNT(*) AS "Total Matches",
                SUM(upset) AS "Total Upsets",
                ROUND(AVG(upset)*100,1) AS "Overall Upset Rate %"
                FROM matches""", conn)
            summary = "Dataset overview:"
        elif any(n in q for n in ["favourite","favorite","top-10","top 10","top10","not the fav"]):
            df = pd.read_sql("""SELECT loser_name AS Player,
                COUNT(*) AS "Times Lost as Favourite"
                FROM matches WHERE upset=1 AND loser_rank<=10
                GROUP BY loser_name ORDER BY "Times Lost as Favourite" DESC LIMIT 10""", conn)
            summary = "Top-10 players who lost most often as the favourite:"
        elif "surface" in q or "grass" in q or "clay" in q or "hard" in q:
            df = pd.read_sql("""SELECT surface AS Surface, COUNT(*) AS Matches,
                ROUND(AVG(upset)*100,1) AS "Upset Rate %"
                FROM matches WHERE surface IS NOT NULL
                GROUP BY surface ORDER BY "Upset Rate %" DESC""", conn)
            summary = "Upset rate by surface:"
        else:
            name = next((n for n in ["Djokovic","Nadal","Federer","Alcaraz","Sinner",
                                      "Medvedev","Tsitsipas","Zverev","Rublev","Murray"]
                         if n in question), None)
            if name:
                df = pd.read_sql(f"""SELECT slam_name AS Tournament, round AS Round,
                    winner_name AS Winner, loser_name AS Loser,
                    CAST(rank_diff AS INT) AS "Rank Gap",
                    SUBSTR(tourney_date,1,4) AS Year
                    FROM matches WHERE upset=1
                    AND (winner_name LIKE '%{name}%' OR loser_name LIKE '%{name}%')
                    ORDER BY tourney_date DESC LIMIT 10""", conn)
                summary = f"Recent upset matches involving {name}:"
            else:
                df = pd.read_sql("""SELECT slam_name AS Tournament, COUNT(*) AS Matches,
                    ROUND(AVG(upset)*100,1) AS "Upset Rate %"
                    FROM matches GROUP BY slam_name ORDER BY "Upset Rate %" DESC""", conn)
                summary = "Upset summary by Grand Slam:"
        conn.close()
        return df, summary
    except Exception:
        import pandas as pd
        return pd.DataFrame({"Tournament":["Wimbledon","Roland Garros","Australian Open","US Open"],
                             "Upset Rate %":[30.5,29.3,28.5,26.7]}), "Demo data (run pipeline for live stats):"

def html_table(df):
    th = "".join(f"<th>{c}</th>" for c in df.columns)
    rows = ""
    for i,row in df.iterrows():
        bg = "#0a0a18" if i%2==0 else "#080815"
        rows += f"<tr style='background:{bg}'>"+"".join(f"<td>{v}</td>" for v in row.values)+"</tr>"
    return (f"<div style='overflow-x:auto;margin-top:0.8rem'>"
            f"<table class='ctable'><thead><tr>{th}</tr></thead><tbody>{rows}</tbody></table></div>")


#  PAGE: HOME
if page == "Home":
    st.markdown("""
    <div class="hero">
      <div class="eyebrow">Grand Slam Analytics · 2022–2024</div>
      <div class="headline">
        Can AI predict<br>a <span class="y">tennis upset</span><br>before it happens?
      </div>
      <div class="subhead">
        GrandSlam IQ analyses 10 years of match data and thousands of player
        press conferences to forecast upsets at the four major Grand Slams —
        Australian Open, Roland Garros, Wimbledon, and the US Open.
      </div>
    </div>
    """, unsafe_allow_html=True)

    status = "🟢 Live Data" if db_ok else "🟡 Demo Mode"
    st.markdown(f"""
    <div class="stats">
      <div class="sc"><div class="sn">26%</div><div class="sl">Average upset rate</div></div>
      <div class="sc"><div class="sn">2,279</div><div class="sl">Real transcripts</div></div>
      <div class="sc"><div class="sn">0.70</div><div class="sl">Model ROC-AUC</div></div>
      <div class="sc"><div class="sn" style="font-size:1.3rem;padding-top:0.4rem">{status}</div><div class="sl">Data status</div></div>
    </div>
    """, unsafe_allow_html=True)

    # simple explainer 
    st.markdown("<div class='eyebrow' style='margin-top:2rem'>What is an upset?</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="card card-hi">
      <div style="font-size:0.95rem;color:#6666aa;line-height:1.75">
        In tennis, an <strong style="color:#aaaacc">upset</strong> is when the lower-ranked player
        (the underdog) beats the higher-ranked player (the favourite). For example, if a player
        ranked <em>#85 in the world beats the #12 player</em>, that's an upset — the favourite was expected to win.
        <br><br>
        At Grand Slams, about <strong style="color:#e2ff47">1 in 4 matches</strong> is an upset.
        This project asks: <em style="color:#aaaacc">can we predict which matches are likely to flip?</em>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='eyebrow' style='margin-top:2rem'>Three tools</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="tools">
      <div class="tool-card">
        <div class="ti">⚡</div>
        <div class="tn" style="color:#e2ff47">Upset Alert</div>
        <div class="td">Enter two players before a match. The AI gives you a probability of an upset happening — plus a plain-English explanation of why. Great for pre-match analysis.</div>
        <div class="tags"><span class="tag">Random Forest</span><span class="tag">SHAP</span><span class="tag">AI Explanation</span></div>
      </div>
      <div class="tool-card">
        <div class="ti">◎</div>
        <div class="tn" style="color:#5252ff">Scouting Report</div>
        <div class="td">Ask about a specific player's history. The system reads through real press conference transcripts and finds patterns in how they talk when they're about to lose.</div>
        <div class="tags"><span class="tag">Transcript Search</span><span class="tag">NLP</span><span class="tag">AI Report</span></div>
      </div>
      <div class="tool-card">
        <div class="ti">◉</div>
        <div class="tn" style="color:#ff4466">Ask the Model</div>
        <div class="td">Chat naturally with 10 years of Grand Slam data. Ask "Which round has the most upsets?" or "What did Djokovic say about fatigue?" — the AI finds the answer.</div>
        <div class="tags"><span class="tag">SQL Agent</span><span class="tag">Transcript Search</span><span class="tag">Chat</span></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Groq key
    st.markdown("<div class='eyebrow' style='margin-top:2rem'>API key</div>", unsafe_allow_html=True)
    groq_in = st.text_input("Groq API key — powers all AI explanations (free at console.groq.com)",
                             type="password", placeholder="gsk_...",
                             value=st.session_state.get("groq_key",""), key="groq_home")
    if groq_in:
        st.session_state.groq_key = groq_in
        st.success("✓ AI explanations enabled across all tools")

    st.markdown("""
    <div class="card" style="margin-top:1.5rem;border-color:#13132a">
      <div class="clabel" style="color:#2a2a5a">Quick start</div>
      <div style="font-size:0.84rem;color:#3a3a6a;line-height:1.75">
        1. Get a <strong style="color:#6666aa">free Groq API key</strong> at console.groq.com (takes 30 seconds, no credit card)<br>
        2. Paste it in the box above<br>
        3. Use the navigation buttons at the top to explore the three tools<br>
        4. Read the <strong style="color:#6666aa">Blog</strong> to understand the findings, or
           <strong style="color:#6666aa">How I Built This</strong> for the full technical story
      </div>
    </div>
    """, unsafe_allow_html=True)


#  PAGE: BLOG

elif page == "Blog":
    st.markdown("""
    <div style="padding:2rem 0 0">
      <div class="eyebrow">Research Blog</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="blog-hero">
      <div class="blog-title">What Press Conferences Reveal About Tennis Upsets</div>
      <div class="blog-meta">By Sonakshi Sharma · April 2025 · 8 min read · AI/ML · Sports Analytics</div>
      <div class="blog-body">
        <p>Every tennis fan has seen it happen. A top seed walks onto centre court as the overwhelming favourite.
        The crowd expects a routine win. And then, over two or three hours, something goes wrong.
        The favourite loses. An upset.</p>
        <p>What if the signs were already there — not in the match statistics, but in what the player
        <strong>said the day before</strong>?</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="blog-body">

    <h3>The Observation</h3>
    <p>Tennis press conferences are remarkable things. Players speak for 10–20 minutes after every match,
    answering questions about their physical state, their mental preparation, their confidence going into
    the next round. And they're surprisingly honest — especially when they're tired.</p>

    <div class="blog-pull">
      "My legs are really heavy after five sets yesterday. I'm not 100 percent going into tomorrow."<br>
      — A player who lost as the heavy favourite the next day
    </div>

    <p>The question driving this project was simple: <strong>can we detect these signals systematically
    and use them to predict upsets?</strong></p>

    <h3>The Data</h3>
    <p>We collected <strong>2,279 press conference transcripts</strong> from ASAP Sports across 12
    Grand Slam tournaments between 2022 and 2024 — Australian Open, Roland Garros, Wimbledon, and the US Open.
    Each transcript was scraped directly from the official press conference archive and linked to match
    outcomes from the ATP database.</p>

    </div>
    """, unsafe_allow_html=True)

    # key findings
    st.markdown("<div class='eyebrow' style='margin:1.5rem 0 0.8rem'>Key findings</div>", unsafe_allow_html=True)

    for n, title, body in [
        ("01", "Wimbledon is the most unpredictable Grand Slam",
         "With a <strong>30.5% upset rate</strong>, Wimbledon upsets the seedings more than any other Slam. Roland Garros is close behind at 29.3%. The US Open is the most predictable at 26.7%."),
        ("02", "Fatigue language is a real signal",
         "Transcripts containing 5+ fatigue keywords the day before a match showed <strong>measurably higher upset rates</strong> when the speaker was the favourite. Physical fatigue phrases — 'heavy legs', 'cramping', 'not 100%' — were the strongest predictors."),
        ("03", "The Cumulative Fatigue Index (CTFI) matters",
         "A player who has already played <strong>15+ sets</strong> in a tournament before their quarterfinal is significantly more likely to be upset than one who cruised through in straight sets. The body keeps score even when the mind tries to hide it."),
        ("04", "Sentiment alone isn't enough",
         "We tried using simple positive/negative sentiment. It didn't work well on its own. But <strong>combining sentiment with specific fatigue vocabulary</strong> and rank features gave a meaningful improvement over baseline."),
        ("05", "Top-10 players are not immune",
         "Rafael Nadal (17 upset losses as favourite), Novak Djokovic (16), and Alexander Zverev (15) top the list of players who've been upset the most despite being favourites. Even the best players in history get caught out."),
    ]:
        st.markdown(f"""
        <div class="finding">
          <div class="fnum">{n}</div>
          <div class="ftext"><strong>{title}</strong><br>{body}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="blog-body" style="margin-top:1.5rem">

    <h3>The Model</h3>
    <p>We trained a <strong>Random Forest classifier</strong> on 9,876 match rows across 26 features —
    covering rank differences, tournament fatigue, round depth, surface, and the NLP features extracted
    from pre-match transcripts.</p>

    <p>The final model achieved a <strong>ROC-AUC of 0.70</strong> — meaning it correctly ranks
    an actual upset above a non-upset 70% of the time. For context, random guessing gives 0.50,
    and predicting purely from rank differences gives about 0.62. The NLP features provide a
    meaningful additional signal.</p>

    <div class="blog-pull">
      A model that reads press conferences outperforms one that only looks at rankings.
      The words matter.
    </div>

    <h3>What This Isn't</h3>
    <p>This is not a betting system. Predicting upsets in sport is genuinely hard — there's irreducible
    randomness in every match. What this project demonstrates is that <strong>language carries
    information about physical and mental state</strong> that statistical records don't capture,
    and that information is measurably useful for prediction.</p>

    <h3>What's Next</h3>
    <p>The most exciting extension would be <strong>real-time scraping</strong> — pulling the press
    conference transcript the evening before a match and generating a live upset probability. The
    pipeline is already built. It just needs to be pointed at tomorrow's draw.</p>

    </div>
    """, unsafe_allow_html=True)


#  PAGE: HOW I BUILT THIS

elif page == "How I Built This":
    st.markdown("""
    <div style="padding:2rem 0 0">
      <div class="eyebrow">Technical Methodology</div>
      <div class="headline" style="font-size:clamp(2.2rem,6vw,4rem)">
        How I <span class="b">Built</span> This
      </div>
      <div class="subhead">
        A complete walkthrough of the data pipeline, NLP approach, model architecture,
        and engineering decisions — from raw CSV files to a live Streamlit app.
      </div>
    </div>
    """, unsafe_allow_html=True)

    steps = [
        ("01","#e2ff47","Data Ingestion",
         "data_ingestion.py",
         """Match data comes from <strong>Jeff Sackmann's open-source tennis database</strong> on GitHub —
         one of the most complete public tennis datasets available. We pull ATP match results from 2015–2024,
         filtering to Grand Slam main draw matches only. Each row contains: winner, loser, their ATP rankings
         at the time of the match, the tournament, round, surface, and score.
         <br><br>
         We then compute the <strong>upset flag</strong> (1 if the lower-ranked player won),
         <strong>rank difference</strong>, and the <strong>Cumulative Tournament Fatigue Index (CTFI)</strong>
         — the total sets each player played in the tournament before this match.""",
         "~9,876 match rows · 4 Grand Slams · 2015–2024"),

        ("02","#5252ff","Transcript Scraping",
         "scraping.py",
         """Press conference transcripts come from <strong>ASAP Sports</strong> (asapsports.com) —
         the official transcript archive used by ATP and Grand Slam tournaments.
         <br><br>
         The scraper works in two levels: first fetching the daily interview listing page for each
         tournament date (using a URL pattern of <code>show_event.php?category=7&date=YYYY-M-DD&title=TOURNAMENT</code>),
         then following each <code>show_interview.php?id=X</code> link and extracting the <code>&lt;p&gt;</code>
         tag content which contains the full Q&amp;A transcript.
         <br><br>
         We use random delays between requests (1.5–3.5 seconds) to be polite to the server.
         The scraper runs resume-safe — already-scraped transcripts are skipped on repeat runs.""",
         "2,279 transcripts · 12 tournaments · 2022–2024"),

        ("03","#ff4466","NLP Processing",
         "nlp.py",
         """Each transcript goes through a five-stage NLP pipeline:
         <br><br>
         <strong>1. Fatigue lexicon matching</strong> — 60+ phrases across 5 categories
         (physical, mental, schedule, injury, motivation) are counted using regex boundary matching.
         <br><br>
         <strong>2. Sentiment analysis</strong> — DistilBERT fine-tuned on SST-2 gives a
         positive/negative sentiment score for each transcript. We also compute a simple
         lexicon-based polarity as a backup feature.
         <br><br>
         <strong>3. Linguistic features</strong> — first-person pronoun rate, negation rate,
         and average sentence length are computed using spaCy.
         <br><br>
         <strong>4. LLM fatigue classification</strong> — we ask a small language model to
         give a 0–1 'is this person fatigued?' score as an additional feature.
         <br><br>
         <strong>5. Feature storage</strong> — all NLP outputs are written back to the
         SQLite database and then joined to match data to build the feature matrix.""",
         "26 features per match · 5 fatigue categories · DistilBERT sentiment"),

        ("04","#e2ff47","Feature Engineering",
         "features.py",
         """The feature matrix combines match statistics with NLP outputs.
         Key engineered features include:
         <br><br>
         <strong>rank_ratio</strong> — player rank divided by opponent rank. Captures the
         relative ranking gap in a scale-independent way.
         <br><br>
         <strong>log_rank_diff</strong> — log-transformed rank difference with sign preserved.
         The log transform handles the non-linear relationship between rank gap and upset probability.
         <br><br>
         <strong>is_underdog</strong> — binary flag: is this player the lower-ranked one?
         <br><br>
         <strong>rank_bin</strong> — categorical: top10, top50, top100, outside100.
         Captures non-linear tier effects in player quality.
         <br><br>
         We also one-hot encode the surface (Clay, Grass, Hard) since upset rates differ significantly.""",
         "26 total features · rank + CTFI + NLP + surface"),

        ("05","#5252ff","Model Training",
         "model.py",
         """We train two models to isolate the NLP contribution:
         <br><br>
         <strong>Model A (Traditional)</strong> — rank features + CTFI only. No NLP. This is the baseline.
         <br><br>
         <strong>Model B (Full)</strong> — all features including NLP transcript signals.
         <br><br>
         Both use a <strong>Random Forest classifier</strong> with hyperparameter tuning via
         5-fold cross-validation over a grid of n_estimators, max_depth, min_samples_leaf,
         and max_features. The preprocessor pipeline handles numeric imputation (median),
         scaling (StandardScaler), and one-hot encoding of categoricals.
         <br><br>
         <strong>SHAP values</strong> (TreeExplainer) are computed post-training to explain
         individual predictions — this powers the feature importance bars in the Upset Alert tool.""",
         "ROC-AUC: 0.70 · Random Forest · 5-fold CV · SHAP explanations"),

        ("06","#ff4466","Streamlit App + RAG",
         "app.py · rag_service.py",
         """The app is a single-file Streamlit application — all pages in one file to avoid
         module import issues on Streamlit Cloud.
         <br><br>
         The <strong>Scouting Report</strong> tool uses a lightweight RAG (Retrieval-Augmented Generation)
         approach: transcript text is searched using keyword overlap scoring (or ChromaDB vector
         similarity if available), and retrieved chunks are passed to the Groq LLM API
         (Llama 3 70B) to generate a natural language scouting report.
         <br><br>
         The <strong>Ask the Model</strong> chat routes questions to either a SQL query
         (for structured stats questions) or transcript search (for language/quote questions)
         using keyword classification, then optionally passes the results to Groq for a
         plain-English summary.
         <br><br>
         Deployment is on <strong>Streamlit Cloud</strong> (free tier) with the Groq API
         key stored as a secret. The DB and model file are committed to GitHub.""",
         "Streamlit Cloud · Groq Llama 3 70B · SQLite · ChromaDB"),
    ]

    for num, color, title, file_label, body, result in steps:
        pills = " ".join(f"<span class='code-pill'>{f.strip()}</span>" for f in file_label.split("·"))
        st.markdown(f"""
        <div class="method-step">
          <div class="method-num">{num}</div>
          <div class="method-content">
            <div class="method-title" style="color:{color}">{title}</div>
            <div style="margin-bottom:0.6rem">{pills}</div>
            <div class="method-body">{body}</div>
            <div class="result-box">→ {result}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # tech stack
    st.markdown("<div class='eyebrow' style='margin:2rem 0 1rem'>Tech stack</div>", unsafe_allow_html=True)
    stack = [
        ("Python 3.11", "Core language"),
        ("pandas + numpy", "Data manipulation"),
        ("scikit-learn", "Random Forest, pipelines, CV"),
        ("SHAP", "Model explainability"),
        ("spaCy", "NLP tokenisation, sentence detection"),
        ("Hugging Face Transformers", "DistilBERT sentiment model"),
        ("BeautifulSoup + requests", "Web scraping"),
        ("SQLite", "Local database for matches + transcripts"),
        ("ChromaDB", "Vector store for RAG"),
        ("Groq API (Llama 3 70B)", "LLM explanations and reports"),
        ("Streamlit", "Web app framework + free hosting"),
        ("Jeff Sackmann tennis_atp", "Source match data (open-source GitHub)"),
        ("ASAP Sports", "Press conference transcript archive"),
    ]
    cols = st.columns(2)
    for i,(tech,desc) in enumerate(stack):
        with cols[i%2]:
            st.markdown(f"""
            <div style="display:flex;gap:0.8rem;align-items:center;padding:0.5rem 0;
                        border-bottom:1px solid #0e0e1e">
              <span style="font-family:'JetBrains Mono',monospace;font-size:0.78rem;
                           color:#5252ff;min-width:220px">{tech}</span>
              <span style="font-size:0.82rem;color:#3a3a6a">{desc}</span>
            </div>
            """, unsafe_allow_html=True)


#  PAGE: UPSET ALERT

elif page == "⚡ Upset Alert":
    st.markdown("""
    <div style="padding:2rem 0 0.5rem">
      <div class="eyebrow">Tool 01 — Prediction</div>
      <div class="headline" style="font-size:clamp(2rem,5vw,3.5rem)">
        <span class="y">Upset</span> Alert
      </div>
      <div class="subhead">
        Enter two players and match details. Optionally paste a press conference transcript
        to extract fatigue signals. Get an AI-powered upset probability.
      </div>
    </div>
    """, unsafe_allow_html=True)

    if not model_ok:
        st.warning("No trained model found — using demo mode. Run the pipeline to train on real data.")

    groq_key = st.session_state.get("groq_key","")

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

    c1, vmid, c2 = st.columns([5,1,5])
    with c1:
        st.markdown("<div class='clabel' style='color:#e2ff47;margin-bottom:0.5rem'>⬤ Underdog (lower-ranked)</div>", unsafe_allow_html=True)
        player      = st.text_input("Underdog name", "Carlos Alcaraz", label_visibility="collapsed")
        player_rank = st.number_input("ATP Ranking", 1, 1000, 45, key="pr")
        ctfi        = st.slider("Fatigue Index (CTFI)", 0, 30, 8,
                                help="Total sets this player has already played in this tournament")
    with vmid:
        st.markdown("<div style='text-align:center;padding-top:2.2rem;font-family:Bebas Neue,sans-serif;font-size:1.4rem;color:#1a1a3a'>VS</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='clabel' style='color:#ff4466;margin-bottom:0.5rem'>⬤ Favourite (higher-ranked)</div>", unsafe_allow_html=True)
        opponent = st.text_input("Favourite name", "Novak Djokovic", label_visibility="collapsed")
        opp_rank = st.number_input("ATP Ranking", 1, 1000, 3, key="or")
        round_sel = st.selectbox("Round", list(ROUND_MAP.keys()), index=4)

    best_of = st.radio("Match format", [3,5], index=1, horizontal=True, format_func=lambda x:f"Best of {x}")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div class="clabel">Press conference transcript
    <span style="color:#1a1a3a;font-size:0.75em"> — optional but improves accuracy</span></div>
    <div style="font-size:0.82rem;color:#2a2a5a;margin-bottom:0.6rem;line-height:1.5">
    Paste what the underdog player said at their last press conference. The system scans for
    fatigue signals (physical exhaustion, mental tiredness, injury concerns, schedule burden).
    </div>
    """, unsafe_allow_html=True)

    use_t = st.toggle("Include transcript NLP", value=True)
    nlp   = {}
    if use_t:
        quick = st.multiselect("Quick-add signals (or type transcript below)",
                               ["tired","heavy legs","cramping","mentally drained",
                                "back-to-back","five sets","injury","my knee","not 100%","doubt"])
        txt = st.text_area("Paste transcript here",
                           placeholder='"I\'m really tired. My legs are heavy after five sets yesterday and my back is stiff. I\'m not 100% going into tomorrow."',
                           height=110, label_visibility="collapsed")
        full = " ".join(quick)+" "+(txt or "")
        if full.strip():
            nlp = analyse_transcript(full)
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Fatigue signals", nlp["fatigue_total"])
            c2.metric("Density/100w", f"{nlp['fatigue_density']:.1f}")
            c3.metric("Sentiment", f"{nlp['sentiment_polarity']:+.2f}")
            c4.metric("Words", nlp["word_count"])

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
    if st.button("⚡  PREDICT UPSET PROBABILITY", use_container_width=True):
        sent = nlp.get("sentiment_polarity",0.0); fat = nlp.get("fatigue_total",0)
        with st.spinner("Running model…"):
            prob, sd = predict(player_rank, opp_rank, ctfi, sent, fat, ROUND_MAP[round_sel], best_of)

        col = "#ff4466" if prob>=0.65 else "#f5a623" if prob>=0.40 else "#5252ff"
        risk= "HIGH UPSET RISK" if prob>=0.65 else "MODERATE RISK" if prob>=0.40 else "LOW RISK"

        r1,r2 = st.columns([2,3])
        with r1:
            st.markdown(f"""
            <div class="card card-hi" style="text-align:center;border-color:{col}22">
              <div class="probnum" style="color:{col}">{prob*100:.0f}%</div>
              <div class="problbl">upset probability</div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:0.62rem;
                          letter-spacing:0.15em;color:{col};margin-top:0.5rem">{risk}</div>
              <div style="font-size:0.8rem;color:#2a2a5a;margin-top:0.5rem">
                {player} vs {opponent} · {round_sel}
              </div>
            </div>""", unsafe_allow_html=True)
            st.progress(prob)

        with r2:
            st.markdown("<div class='clabel'>AI Explanation</div>", unsafe_allow_html=True)
            exp = None
            if groq_key:
                top   = sorted(sd.items(),key=lambda x:abs(x[1]),reverse=True)[:4]
                lines = [f"  • {f.replace('_',' ')}: {v:+.3f}" for f,v in top if abs(v)>0.005]
                prompt= (f"Tennis analyst. {prob*100:.0f}% upset probability for {player} (rank {player_rank}) vs "
                         f"{opponent} (rank {opp_rank}), {round_sel}.\nTop SHAP features:\n"+"\n".join(lines)+
                         "\n\nWrite 2 clear sentences for a general audience. Why is this the predicted risk?")
                with st.spinner("Generating…"): exp = groq_call(prompt, groq_key, 130)
            if not exp:
                top   = sorted(sd.items(),key=lambda x:abs(x[1]),reverse=True)[:2]
                names = [f.replace("_"," ") for f,_ in top]
                lvl   = "high" if prob>0.6 else "moderate" if prob>0.4 else "low"
                exp   = (f"The model assigns **{lvl}** upset risk ({prob*100:.0f}%) for {player}. "
                         f"Key drivers: **{names[0]}** and **{names[1] if len(names)>1 else 'rank gap'}**.")
                if not groq_key: st.caption("Add a Groq key for richer AI explanations.")

            st.markdown(f"""
            <div style="border-left:3px solid {col};border-radius:0 8px 8px 0;
                        padding:1rem 1.2rem;font-size:0.88rem;color:#8888aa;
                        line-height:1.65;background:#0a0a18;margin:0.5rem 0">{exp}</div>
            """, unsafe_allow_html=True)

            if sd:
                st.markdown("<div class='clabel' style='margin-top:1rem'>Feature importance (SHAP)</div>", unsafe_allow_html=True)
                st.markdown("<div style='font-size:0.74rem;color:#1e1e42;margin-bottom:0.3rem'>Red = increases upset risk · Blue = decreases it</div>", unsafe_allow_html=True)
                st.markdown(shap_html(sd), unsafe_allow_html=True)


#  PAGE: SCOUTING

elif page == "◎ Scouting":
    st.markdown("""
    <div style="padding:2rem 0 0.5rem">
      <div class="eyebrow">Tool 02 — Transcript Intelligence</div>
      <div class="headline" style="font-size:clamp(2rem,5vw,3.5rem)">
        Player <span class="b">Scouting</span>
      </div>
      <div class="subhead">
        Ask about a player's fatigue history. The system searches through real press
        conference transcripts and generates a tactical scouting report.
      </div>
    </div>
    """, unsafe_allow_html=True)

    groq_key = st.session_state.get("groq_key","")

    PLAYERS = ["Carlos Alcaraz","Novak Djokovic","Rafael Nadal","Jannik Sinner",
               "Daniil Medvedev","Stefanos Tsitsipas","Alexander Zverev","Andrey Rublev",
               "Roger Federer","Dominic Thiem","Holger Rune","Taylor Fritz",
               "Casper Ruud","Ben Shelton","Frances Tiafoe","Felix Auger-Aliassime"]
    if db_ok:
        try:
            import sqlite3
            conn = sqlite3.connect("tennis_upsets.db")
            rows = conn.execute("""
                SELECT player_name, COUNT(*) as cnt FROM transcripts
                WHERE player_name IS NOT NULL AND player_name != 'Unknown' AND LENGTH(player_name)>3
                GROUP BY player_name ORDER BY cnt DESC LIMIT 80
            """).fetchall(); conn.close()
            if rows: PLAYERS = [r[0] for r in rows]
        except Exception: pass

    c1,c2 = st.columns([1,2])
    with c1:
        player = st.selectbox("Select player", PLAYERS)
    with c2:
        query  = st.text_input("What do you want to know?",
                               "What are this player's fatigue signals before upset losses?")
    a1,a2 = st.columns(2)
    with a1: n_results = st.slider("Excerpts to retrieve", 2, 10, 5)
    with a2: upset_only = st.toggle("Only from upset matches", value=True)

    if st.button("◎  GENERATE SCOUTING REPORT", use_container_width=True):
        with st.spinner("Searching transcripts…"):
            snips = search_transcripts(query, player, n_results, upset_only)
        if not snips:
            st.warning("No matching transcripts. Try turning off 'Only from upset matches'.")
        else:
            st.markdown(f"<div class='clabel' style='margin-bottom:0.8rem'>✓ {len(snips)} excerpts retrieved</div>", unsafe_allow_html=True)
            report = None
            if groq_key:
                ctx    = "\n\n---\n\n".join(f"[{s.get('tournament','?')} · {s.get('round','')} · {'UPSET' if s.get('upset') else 'WIN'}]\n{s['text'][:400]}" for s in snips[:5])
                prompt = (f"Professional tennis scout writing for a coaching team.\nQuestion: {query}\nPlayer: {player}\n\nTranscripts:\n{ctx}\n\n"
                          f"Write a sharp, bullet-point scouting report (max 200 words). Focus on recurring patterns and preparation insights.")
                with st.spinner("Writing report…"): report = groq_call(prompt, groq_key, 350)
            if not report:
                CATS = {"Physical":["tired","exhausted","heavy legs","cramping","sore"],
                        "Mental":["mentally","focus","distracted"],"Schedule":["back-to-back","five sets"],
                        "Injury":["injury","pain","knee","shoulder","ankle"],"Motivation":["doubt","uncertain","not sure"]}
                combined = " ".join(s["text"] for s in snips).lower()
                lines    = []
                for cat,words in CATS.items():
                    hits = sum(len(re.findall(r"\b"+re.escape(w)+r"\b",combined)) for w in words)
                    if hits: lines.append(f"• **{cat}**: {hits} mention{'s' if hits>1 else ''} detected")
                nu     = sum(1 for s in snips if s.get("upset"))
                report = (f"**Pattern analysis for {player}** *(from {len(snips)} excerpts, {nu} from upsets)*\n\n"
                          +("\n".join(lines) if lines else "• No strong fatigue signals detected.")
                          +"\n\n*Add a Groq key for a full AI-written report.*")

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='clabel' style='color:#5252ff;margin-bottom:0.6rem'>Scouting Report — {player}</div>", unsafe_allow_html=True)
            st.markdown(report)
            st.markdown("</div>", unsafe_allow_html=True)

            # Signal bars
            CATS2  = {"Physical":["tired","exhausted","heavy legs","cramping","sore"],
                      "Mental":["mentally","focus","distracted"],"Schedule":["back-to-back","five sets"],
                      "Injury":["injury","pain","knee","shoulder"],"Motivation":["doubt","uncertain"]}
            COLS2  = ["#ff4466","#5252ff","#e2ff47","#f5a623","#888"]
            combined = " ".join(s["text"] for s in snips).lower()
            counts   = {cat:sum(len(re.findall(r"\b"+re.escape(w)+r"\b",combined)) for w in words)
                        for cat,words in CATS2.items()}
            mx = max(counts.values()) if max(counts.values())>0 else 1
            st.markdown("<div class='clabel' style='margin-top:1.2rem;margin-bottom:0.5rem'>Fatigue signal breakdown</div>", unsafe_allow_html=True)
            for (cat,val),color in zip(counts.items(),COLS2):
                pct = int(val/mx*100)
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:10px;margin:5px 0">
                  <span style="width:80px;font-family:'JetBrains Mono',monospace;font-size:0.67rem;color:#2a2a5a">{cat}</span>
                  <div style="flex:1;background:#0a0a18;border-radius:3px;height:10px">
                    <div style="width:{pct}%;background:{color};height:10px;border-radius:3px"></div></div>
                  <span style="color:{color};font-family:'JetBrains Mono',monospace;font-size:0.7rem;width:18px">{val}</span>
                </div>""", unsafe_allow_html=True)

            m1,m2,m3 = st.columns(3)
            m1.metric("Excerpts", len(snips))
            m2.metric("From upsets", sum(1 for s in snips if s.get("upset")))
            m3.metric("Total signals", sum(counts.values()))
            with st.expander("View retrieved excerpts"):
                for i,s in enumerate(snips,1):
                    out = "🔴 UPSET" if s.get("upset") else "🟢 Win"
                    st.markdown(f"<div class='snip'><div class='snip-meta'>{i} · {s.get('player','?')} · {s.get('tournament','?')} {s.get('round','')} · {out}</div>{s['text'][:500]}</div>", unsafe_allow_html=True)


#  PAGE: ASK THE MODEL

elif page == "◉ Ask the Model":
    st.markdown("""
    <div style="padding:2rem 0 0.5rem">
      <div class="eyebrow">Tool 03 — Conversational AI</div>
      <div class="headline" style="font-size:clamp(2rem,5vw,3.5rem)">
        Ask the <span class="r">Model</span>
      </div>
      <div class="subhead">
        Chat with 10 years of Grand Slam data. Ask stats questions or
        search what players said in press conferences.
      </div>
    </div>
    """, unsafe_allow_html=True)

    groq_key = st.session_state.get("groq_key","")

    EXAMPLES = [
        ("📊","Upset rates","What is the upset rate at each Grand Slam?"),
        ("🏆","Biggest upsets","Show the 5 biggest rank-gap upsets"),
        ("🔴","Favourite losses","Which top-10 player lost most as a favourite?"),
        ("🎾","By round","Which round has the highest upset rate?"),
        ("💬","Cramping","Find mentions of cramping before an upset"),
        ("😓","Djokovic fatigue","What did Djokovic say about fatigue?"),
        ("🌿","Surface","Upset rates on grass vs clay"),
        ("📈","Dataset","How many matches are in the dataset?"),
    ]

    st.markdown("<div class='clabel' style='margin-bottom:0.5rem'>Try an example</div>", unsafe_allow_html=True)
    ecols = st.columns(4)
    for i,(icon,lbl,q) in enumerate(EXAMPLES):
        with ecols[i%4]:
            if st.button(f"{icon} {lbl}", key=f"ex_{i}", use_container_width=True):
                st.session_state.pending_q = q

    st.markdown("<hr>", unsafe_allow_html=True)

    if not st.session_state.messages:
        st.session_state.messages = [{
            "role":"assistant","source":"",
            "content":"Hi! Ask me anything about Grand Slam upsets — statistics from the match database, or patterns from player press conference transcripts.\n\nTry one of the examples above, or type your own question below.",
        }]

    for msg in st.session_state.messages:
        if msg["role"]=="user":
            st.markdown(f"<div class='cu'>{msg['content']}</div>", unsafe_allow_html=True)
        else:
            content = msg["content"]; source = msg.get("source","")
            src_tag = f"<div class='csrc'>{source}</div>" if source else ""
            st.markdown(f"<div class='cb'>{content}{src_tag}</div>", unsafe_allow_html=True)
            if msg.get("table"):
                st.markdown(msg["table"], unsafe_allow_html=True)
            if msg.get("snippets"):
                with st.expander("View transcript excerpts"):
                    for s in msg["snippets"][:3]:
                        out = "🔴 UPSET" if s.get("upset") else "🟢 Win"
                        st.markdown(f"<div class='snip'><div class='snip-meta'>{s.get('player','?')} · {s.get('tournament','?')} {s.get('round','')} · {out}</div>{s['text'][:380]}</div>", unsafe_allow_html=True)

    pending    = st.session_state.pop("pending_q", None)
    user_input = st.chat_input("Ask anything about tennis upsets…")
    question   = pending or user_input

    if question:
        st.session_state.messages.append({"role":"user","content":question,"source":""})
        st.markdown(f"<div class='cu'>{question}</div>", unsafe_allow_html=True)

        q_lower = question.lower(); snippets = []; df_result = None; table_html = ""
        is_text = any(w in q_lower for w in
                      ["said","mention","fatigue","tired","cramping","press","quote","words","transcript"])

        with st.spinner("Thinking…"):
            if is_text:
                snippets = search_transcripts(question, n=4, upset_only=False)
                source   = "◎ Transcript search"
                if groq_key and snippets:
                    ctx    = "\n\n".join(f"[{s.get('player','?')} – {s.get('tournament','?')} {s.get('round','')}]: {s['text'][:350]}" for s in snippets)
                    prompt = f"Tennis analyst. Answer this question based on these transcript excerpts.\nQ: {question}\n\nExcerpts:\n{ctx}\n\nAnswer in 2-3 sentences:"
                    answer = groq_call(prompt, groq_key, 200) or ""
                if not (is_text and answer if 'answer' in dir() else False):
                    answer = "\n\n".join(f"**{s.get('player','?')} — {s.get('tournament','?')} {s.get('round','')}** ({'UPSET' if s.get('upset') else 'win'}): {s['text'][:280]}…" for s in snippets[:3]) or "No matching transcripts found."
            else:
                df_result, summary = answer_sql(question)
                source = "🗄️ SQL query"
                ai_txt = ""
                if groq_key and df_result is not None:
                    prompt = f"Tennis analyst. User asked: '{question}'\nData:\n{df_result.to_string(index=False)}\n\nWrite 1 insightful sentence about the most interesting finding."
                    ai_txt = groq_call(prompt, groq_key, 80) or ""
                answer = ai_txt if ai_txt else summary
                if df_result is not None and not df_result.empty:
                    table_html = html_table(df_result)

        src_tag = f"<div class='csrc'>{source}</div>"
        st.markdown(f"<div class='cb'>{answer}{src_tag}</div>", unsafe_allow_html=True)
        if table_html:
            st.markdown(table_html, unsafe_allow_html=True)
        if snippets:
            with st.expander("View transcript excerpts"):
                for s in snippets[:3]:
                    out = "🔴 UPSET" if s.get("upset") else "🟢 Win"
                    st.markdown(f"<div class='snip'><div class='snip-meta'>{s.get('player','?')} · {s.get('tournament','?')} {s.get('round','')} · {out}</div>{s['text'][:380]}</div>", unsafe_allow_html=True)

        st.session_state.messages.append({
            "role":"assistant","content":answer,"source":source,
            "snippets":snippets,"table":table_html,
        })

    if len(st.session_state.messages)>1:
        if st.button("Clear conversation"): st.session_state.messages=[]; st.rerun()