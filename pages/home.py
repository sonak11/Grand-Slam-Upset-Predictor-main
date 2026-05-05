"""Home page — explains the project beautifully."""

import streamlit as st


def render(db_ok, model_ok, groq_key):

    # hero 
    st.markdown("""
    <div style='padding: 3rem 0 1rem'>
      <p style='font-family:DM Mono,monospace; font-size:0.75rem; color:#4040a0;
                text-transform:uppercase; letter-spacing:0.15em; margin-bottom:1rem'>
        Grand Slam Analytics · 2015–2024
      </p>
      <h1 class='hero-title'>
        Can we predict a<br>
        <span class='accent'>tennis upset</span><br>
        before it happens?
      </h1>
      <p class='hero-sub' style='max-width:560px'>
        GrandSlam IQ combines match statistics, player rankings, and
        AI-powered press conference analysis to detect fatigue signals
        and predict upsets at the four Grand Slams.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # stats row 
    st.markdown("""
    <div class='metric-row'>
      <div class='metric-box'>
        <div class='metric-val'>26%</div>
        <div class='metric-lbl'>Average upset rate</div>
      </div>
      <div class='metric-box'>
        <div class='metric-val'>10yr</div>
        <div class='metric-lbl'>Data coverage</div>
      </div>
      <div class='metric-box'>
        <div class='metric-val'>4</div>
        <div class='metric-lbl'>Grand Slams</div>
      </div>
      <div class='metric-box'>
        <div class='metric-val'>3</div>
        <div class='metric-lbl'>AI tools</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # what is an upset 
    st.markdown("""
    <div class='gs-card'>
      <p style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;
                margin-bottom:0.6rem'>What counts as an upset?</p>
      <p style='color:#8888b0;line-height:1.7;margin:0'>
        An upset is when the <strong style='color:#e8e8f0'>lower-ranked player beats the higher-ranked favourite</strong>.
        For example, if rank 85 beats rank 12, that's an upset — rank 12 was expected to win.
        We measure how big the upset was using the <em>rank gap</em> (85 − 12 = 73 places).
        This app focuses only on Grand Slam main-draw matches — the four most prestigious
        tournaments in tennis: Australian Open, Roland Garros, Wimbledon, and US Open.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # 3 features 
    st.markdown("""
    <p style='font-family:Syne,sans-serif;font-size:1.5rem;font-weight:700;
              margin: 1.5rem 0 1rem; letter-spacing:-0.02em'>
      Three tools, one question
    </p>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class='gs-card-accent' style='height:100%'>
          <div style='font-size:2rem;margin-bottom:0.8rem'>🚨</div>
          <p style='font-family:Syne,sans-serif;font-weight:700;font-size:1rem;
                    color:#c8f23d;margin-bottom:0.5rem'>Upset Alert</p>
          <p style='color:#8888b0;font-size:0.85rem;line-height:1.65;margin-bottom:1rem'>
            Enter two players and their stats. Paste a post-match press
            conference transcript (optional). Get an upset probability
            <strong style='color:#e8e8f0'>plus a plain-English AI explanation</strong>
            of exactly why the model thinks an upset is likely.
          </p>
          <div>
            <span class='pill'>Random Forest</span>
            <span class='pill'>SHAP</span>
            <span class='pill'>Groq LLM</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='gs-card-accent' style='height:100%'>
          <div style='font-size:2rem;margin-bottom:0.8rem'>📋</div>
          <p style='font-family:Syne,sans-serif;font-weight:700;font-size:1rem;
                    color:#5ee7df;margin-bottom:0.5rem'>Scouting Report</p>
          <p style='color:#8888b0;font-size:0.85rem;line-height:1.65;margin-bottom:1rem'>
            Ask <em>"What are Djokovic's fatigue signals before an upset?"</em>
            The system searches through hundreds of real press conference transcripts,
            finds matches that fit your question, and <strong style='color:#e8e8f0'>
            generates a tactical scouting report</strong> for a coach or analyst.
          </p>
          <div>
            <span class='pill'>ChromaDB RAG</span>
            <span class='pill'>Embeddings</span>
            <span class='pill'>Groq LLM</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class='gs-card-accent' style='height:100%'>
          <div style='font-size:2rem;margin-bottom:0.8rem'>💬</div>
          <p style='font-family:Syne,sans-serif;font-weight:700;font-size:1rem;
                    color:#ff6b6b;margin-bottom:0.5rem'>Ask the Model</p>
          <p style='color:#8888b0;font-size:0.85rem;line-height:1.65;margin-bottom:1rem'>
            Chat naturally with your data. Ask <em>"Which round has the most
            upsets at Wimbledon?"</em> or <em>"Show me matches where cramping was
            mentioned."</em> The AI decides whether to run a <strong style='color:#e8e8f0'>
            database query or a transcript search</strong> — and shows you the answer.
          </p>
          <div>
            <span class='pill'>LangChain</span>
            <span class='pill'>SQL Agent</span>
            <span class='pill'>Vector Search</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # how it works 
    st.markdown("""
    <p style='font-family:Syne,sans-serif;font-size:1.5rem;font-weight:700;
              margin: 0.5rem 0 1rem; letter-spacing:-0.02em'>
      How the prediction works
    </p>
    """, unsafe_allow_html=True)

    steps = [
        ("01", "#c8f23d", "Rank features",
         "We compare both players' ATP rankings, compute the rank gap, log-transform it, and flag who is the underdog. A rank-500 player beating rank-1 is a far bigger story than rank-40 beating rank-30."),
        ("02", "#5ee7df", "Fatigue Index (CTFI)",
         "The Cumulative Tournament Fatigue Index counts how many sets a player has already played in this tournament before the current match. Going to five sets twice before a quarterfinal is a huge physical cost that shows up in the data."),
        ("03", "#ff6b6b", "NLP transcript analysis",
         "We scan pre-match press conference transcripts for 60+ fatigue signals across five categories: physical ('heavy legs'), mental ('couldn't focus'), schedule ('back-to-back'), injury ('my knee'), and motivation ('question mark'). Sentiment analysis adds a polarity score."),
        ("04", "#c8a0ff", "Random Forest + SHAP",
         "A Random Forest trained on 10 years of Grand Slam data combines all features. SHAP values then explain exactly which factor drove the prediction — so you always know why, not just what."),
    ]

    for num, color, title, desc in steps:
        st.markdown(f"""
        <div class='gs-card' style='display:flex;gap:1.5rem;align-items:flex-start'>
          <div style='font-family:Syne,sans-serif;font-size:2rem;font-weight:800;
                      color:{color};opacity:0.4;flex-shrink:0;line-height:1'>{num}</div>
          <div>
            <p style='font-family:Syne,sans-serif;font-weight:700;font-size:0.95rem;
                      color:{color};margin:0 0 0.4rem'>{title}</p>
            <p style='color:#8888b0;font-size:0.87rem;line-height:1.65;margin:0'>{desc}</p>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # getting started 
    st.markdown("<br>", unsafe_allow_html=True)

    if not groq_key:
        st.markdown("""
        <div style='background:linear-gradient(135deg,#1a1a0d,#121208);
                    border:1px solid #c8f23d55;border-radius:16px;padding:1.5rem 2rem'>
          <p style='font-family:Syne,sans-serif;font-weight:700;color:#c8f23d;margin:0 0 0.5rem'>
            ⚡ Add your Groq key to unlock AI explanations
          </p>
          <p style='color:#a0a060;font-size:0.87rem;line-height:1.6;margin:0'>
            All three tools work without a key (using rule-based fallbacks), but adding a
            <strong style='color:#c8f23d'>free Groq API key</strong> in the sidebar unlocks
            natural-language explanations powered by Llama 3 70B.
            Get one free in 30 seconds at
            <strong style='color:#c8f23d'>console.groq.com</strong>
          </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background:linear-gradient(135deg,#0d1a0d,#081208);
                    border:1px solid #5ee7df55;border-radius:16px;padding:1.5rem 2rem'>
          <p style='font-family:Syne,sans-serif;font-weight:700;color:#5ee7df;margin:0 0 0.3rem'>
            ✅ AI explanations enabled
          </p>
          <p style='color:#608060;font-size:0.85rem;margin:0'>
            Groq key detected. Use the sidebar to navigate to any tool.
          </p>
        </div>
        """, unsafe_allow_html=True)