import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Lungwise | Lung cancer risk check", page_icon="🫁", layout="wide")

DEEP, MUTED, LINE = "#123B40", "#5B7176", "#DCE6E7"
TEAL, ROSE = "#1F8A84", "#D1495B"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Instrument+Serif&display=swap');
html { scroll-behavior: smooth; }
html, body, .stApp, .stApp p, .stApp label, .stApp span, .stApp div, .stApp button { font-family: 'DM Sans', sans-serif; }
[data-testid="stIconMaterial"], .material-symbols-rounded, .material-icons,
.stApp [data-testid="stIconMaterial"] { font-family: 'Material Symbols Rounded', 'Material Icons' !important; }
.stApp { background: #F3F7F7; color: #123B40; }
#MainMenu, footer, header { visibility: hidden; }

/* full-bleed page, content columns centred */
.block-container, [data-testid="stMainBlockContainer"] { max-width: 100% !important; padding: 0 0 2rem 0 !important; }
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"],
.block-container > [data-testid="stVerticalBlock"] { gap: 1.2rem; }
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"],
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > [data-testid="stForm"],
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > [data-testid="stExpander"],
.block-container > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"],
.block-container > [data-testid="stVerticalBlock"] > [data-testid="stForm"],
.block-container > [data-testid="stVerticalBlock"] > [data-testid="stExpander"] {
  width: min(820px, calc(100% - 2.5rem)); margin-left: auto; margin-right: auto; }
[data-testid="stElementContainer"]:has(.fullbleed) { width: 100% !important; max-width: 100% !important; }

/* nav */
.topnav { position: fixed; top: 0; left: 0; right: 0; z-index: 999; display: flex; align-items: center;
  justify-content: space-between; padding: 1rem 5vw; background: rgba(243,247,247,.85);
  backdrop-filter: blur(10px); border-bottom: 1px solid #DCE6E7; }
.brand { font-family: 'Instrument Serif', serif !important; font-size: 1.6rem; color: #123B40; text-decoration: none; }
.navlinks { display: flex; align-items: center; gap: 1.8rem; }
.navlinks a { color: #5B7176; text-decoration: none; font-weight: 500; font-size: .95rem; }
.navlinks a:hover { color: #123B40; }
.navlinks a.pill { background: #123B40; color: #fff; padding: .5rem 1.1rem; border-radius: 99px; }
.navlinks a.pill:hover { background: #1F8A84; color: #fff; }

/* hero */
.hero { display: flex; align-items: center; justify-content: space-between; gap: 2rem; flex-wrap: wrap;
  padding: 9rem 5vw 5rem 5vw;
  background: radial-gradient(900px 420px at 88% 10%, #D3ECEC 0%, rgba(243,247,247,0) 70%); }
.hero-text { flex: 1 1 420px; max-width: 640px; }
.hero h1 { font-family: 'Instrument Serif', serif !important; font-weight: 400; font-size: clamp(2.8rem, 6vw, 4.8rem);
  line-height: 1.04; letter-spacing: -0.015em; margin: 0 0 1.4rem 0; color: #123B40; }
.hero p { font-size: 1.15rem; line-height: 1.6; color: #5B7176; max-width: 34rem; margin: 0 0 2rem 0; }
.btn { display: inline-block; background: #1F8A84; color: #fff !important; padding: .9rem 1.8rem; border-radius: 99px;
  text-decoration: none; font-weight: 600; margin-right: .8rem; }
.btn:hover { background: #123B40; }
.btn.ghost { background: transparent; color: #123B40 !important; border: 1px solid #123B40; }
.btn.ghost:hover { background: #123B40; color: #fff !important; }
.rings { flex: 0 0 340px; height: 340px; margin: 0 auto;
  background: repeating-radial-gradient(circle at center, rgba(31,138,132,0) 0 20px, rgba(31,138,132,.28) 21px 22px);
  -webkit-mask-image: radial-gradient(circle, #000 35%, transparent 72%);
  mask-image: radial-gradient(circle, #000 35%, transparent 72%);
  animation: breathe 7s ease-in-out infinite; }
@keyframes breathe { 0%,100% { transform: scale(.94); } 50% { transform: scale(1.06); } }
@media (prefers-reduced-motion: reduce) { .rings { animation: none; } }

/* how it works */
.how { padding: 3rem 5vw 3rem 5vw; border-top: 1px solid #DCE6E7; scroll-margin-top: 80px; }
.how h2, .sec h2 { font-family: 'Instrument Serif', serif !important; font-weight: 400; font-size: 2.4rem;
  letter-spacing: -0.01em; margin: 0 0 2rem 0; color: #123B40; }
.steps { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 2.5rem; }
.step { border-top: 2px solid #123B40; padding-top: 1rem; }
.step .n { font-family: 'Instrument Serif', serif !important; font-size: 1.6rem; color: #1F8A84; }
.step h3 { font-size: 1.1rem; font-weight: 600; margin: .3rem 0 .4rem 0; color: #123B40; }
.step p { color: #5B7176; line-height: 1.55; margin: 0; }

/* sections inside the centred column */
.sec { scroll-margin-top: 90px; padding-top: 2.5rem; }
.sec h2 { margin-bottom: .6rem; }
.sec p { color: #5B7176; line-height: 1.6; margin: 0 0 .5rem 0; }
[data-testid="stForm"] { background: #fff; border: 1px solid #DCE6E7; border-radius: 18px; padding: 1.8rem; }
[data-testid="stFormSubmitButton"] button { background: #1F8A84; color: #fff; border: 0; border-radius: 99px;
  padding: .75rem 1.6rem; font-weight: 600; width: 100%; }
[data-testid="stFormSubmitButton"] button:hover { background: #123B40; color: #fff; }

/* result */
.result { background: #fff; border: 1px solid #DCE6E7; border-radius: 18px; padding: 1.8rem; }
.result .label { font-family: 'Instrument Serif', serif !important; font-size: 2.4rem; line-height: 1.1; }
.result .meta { color: #5B7176; margin-top: .3rem; }
.bar { height: 8px; background: #E6EEEF; border-radius: 99px; overflow: hidden; margin: 1.1rem 0; }
.bar > span { display: block; height: 100%; border-radius: 99px; }

/* footer */
.foot { margin-top: 3rem; padding: 2rem 5vw; border-top: 1px solid #DCE6E7; color: #5B7176; font-size: .9rem;
  display: flex; justify-content: space-between; gap: 1rem; flex-wrap: wrap; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

model = joblib.load("disease_model.pkl")
features = joblib.load("features.pkl")
explainer = shap.TreeExplainer(model)
nice = {c: c.replace("_", " ").capitalize() for c in features}

questions = {
    "SMOKING": "I smoke",
    "YELLOW_FINGERS": "My fingers have yellow stains",
    "ANXIETY": "I often feel anxious",
    "PEER_PRESSURE": "I feel peer pressure to smoke or drink",
    "CHRONIC DISEASE": "I have a long-term illness",
    "FATIGUE": "I feel tired very often",
    "ALLERGY": "I have allergies",
    "WHEEZING": "I wheeze when breathing",
    "ALCOHOL CONSUMING": "I drink alcohol regularly",
    "COUGHING": "I have a persistent cough",
    "SHORTNESS OF BREATH": "I get short of breath easily",
    "SWALLOWING DIFFICULTY": "I find it hard to swallow",
    "CHEST PAIN": "I have chest pain",
}


def risk_shap_values(row):
    """SHAP values for the 'cancer' class, robust to SHAP version differences."""
    sv = explainer.shap_values(row)
    if isinstance(sv, list):
        return sv[1][0]
    if getattr(sv, "ndim", 2) == 3:
        return sv[0, :, 1]
    return sv[0]


def clean_axes(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(LINE)
    ax.tick_params(colors=DEEP, length=0)
    ax.xaxis.label.set_color(MUTED)


# ---------- nav + hero + how it works ----------
st.markdown("""<div class="fullbleed"></div>
<div class="topnav">
<a class="brand" href="#top">Lungwise</a>
<div class="navlinks">
<a href="#how">How it works</a>
<a href="#check" class="pill">Start the check</a>
</div>
</div>
<div class="hero" id="top">
<div class="hero-text">
<h1>Check your lung cancer risk in under a minute.</h1>
<p>Answer thirteen simple yes/no questions. An AI model estimates your risk and shows exactly which answers influenced the result.</p>
<a class="btn" href="#check">Start the check</a><a class="btn ghost" href="#how">How it works</a>
</div>
<div class="rings"></div>
</div>
<div class="how" id="how">
<h2>Three steps, no medical terms</h2>
<div class="steps">
<div class="step"><div class="n">1</div><h3>Answer simple questions</h3><p>Tell us your age, gender and which everyday symptoms and habits apply to you.</p></div>
<div class="step"><div class="n">2</div><h3>Get a risk estimate</h3><p>A Random Forest model trained on survey data gives you a lower or higher risk result.</p></div>
<div class="step"><div class="n">3</div><h3>See why</h3><p>SHAP explains which answers raised or lowered your result, so it is never a black box.</p></div>
</div>
</div>""", unsafe_allow_html=True)

# ---------- checker ----------
st.markdown("""<div class="sec" id="check">
<h2>Your check</h2>
<p>Switch on everything that applies to you right now.</p>
</div>""", unsafe_allow_html=True)

with st.form("check"):
    c1, c2 = st.columns([1, 2])
    gender = c1.radio("Gender", ["Male", "Female"], horizontal=True)
    age = c2.slider("Age", 10, 100, 40)

    st.markdown("**Which of these apply to you?**")
    answers = {"GENDER": 1 if gender == "Male" else 0, "AGE": age}
    cols = st.columns(2)
    for i, (col, text) in enumerate(questions.items()):
        answers[col] = int(cols[i % 2].toggle(text))

    submitted = st.form_submit_button("Check my risk")

if submitted:
    row = pd.DataFrame([answers])[features]
    prob = model.predict_proba(row)[0][1]
    high = prob >= 0.5
    color = ROSE if high else TEAL
    label = "Higher risk" if high else "Lower risk"
    advice = ("Please consider seeing a doctor about a chest X-ray or CT scan, and avoid smoking."
              if high else
              "Keep up healthy habits, avoid smoking, and get regular check-ups.")

    st.markdown(f"""<div class="result">
<div class="label" style="color:{color}">{label}</div>
<div class="meta">Model estimate: {prob:.0%} chance</div>
<div class="bar"><span style="width:{prob*100:.0f}%; background:{color}"></span></div>
<div>{advice}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("""<div class="sec"><h2>Why the model said this</h2></div>""", unsafe_allow_html=True)
    contrib = pd.Series(risk_shap_values(row), index=[nice[f] for f in features])
    contrib = contrib[contrib.abs() > 1e-6].sort_values()
    up = contrib[contrib > 0].sort_values(ascending=False).head(3)
    down = contrib[contrib < 0].sort_values().head(3)
    if len(up):
        st.markdown(f"**Raised the estimate:** {', '.join(up.index)}")
    if len(down):
        st.markdown(f"**Lowered it:** {', '.join(down.index)}")

    fig, ax = plt.subplots(figsize=(6.4, max(2, 0.34 * len(contrib))))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.barh(contrib.index, contrib.values,
            color=[ROSE if v > 0 else TEAL for v in contrib.values], height=0.6)
    ax.axvline(0, color=LINE, linewidth=1)
    ax.set_xlabel("Effect on risk (red raises, teal lowers)")
    clean_axes(ax)
    plt.tight_layout()
    st.pyplot(fig, transparent=True)

with st.expander("Which factors matter most overall?"):
    imp = pd.Series(model.feature_importances_, index=[nice[f] for f in features]).sort_values()
    fig2, ax2 = plt.subplots(figsize=(6.4, 5))
    fig2.patch.set_alpha(0)
    ax2.set_facecolor("none")
    ax2.barh(imp.index, imp.values, color=TEAL, height=0.6)
    ax2.set_xlabel("Feature importance")
    clean_axes(ax2)
    plt.tight_layout()
    st.pyplot(fig2, transparent=True)

st.markdown("""<div class="fullbleed"></div>
<div class="foot">
<span>Lungwise is an educational screening tool, not a medical diagnosis.</span>
<span>Built with scikit-learn, SHAP and Streamlit</span>
</div>""", unsafe_allow_html=True)