import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import uuid
from datetime import datetime

from report import build_report

st.set_page_config(page_title="Lungwise | Lung cancer risk check", page_icon="🫁", layout="wide")

DEEP, MUTED, LINE = "#123B40", "#5B7176", "#DCE6E7"
TEAL, ROSE, AMBER = "#1F8A84", "#D1495B", "#B7791F"

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
[data-testid="stForm"] { background: #fff; border: 1px solid #DCE6E7; border-radius: 18px; padding: 1.8rem;
  width: min(820px, calc(100% - 2.5rem)); max-width: 820px; margin-left: auto; margin-right: auto; box-sizing: border-box; }
[data-testid="stExpander"] { width: min(820px, calc(100% - 2.5rem)); margin-left: auto; margin-right: auto; }
[data-testid="stFormSubmitButton"] button { background: #1F8A84; color: #fff; border: 0; border-radius: 99px;
  padding: .75rem 1.6rem; font-weight: 600; width: 100%; }
[data-testid="stFormSubmitButton"] button:hover { background: #123B40; color: #fff; }
[data-testid="stDownloadButton"] button { background: #123B40; color: #fff; border: 0; border-radius: 99px;
  padding: .75rem 1.6rem; font-weight: 600; }
[data-testid="stDownloadButton"] button:hover { background: #1F8A84; color: #fff; }

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

NICE = {
    "AGE": "Age", "GENDER": "Gender", "PACK_YEARS": "Smoking history",
    "RADON": "Radon exposure", "ASBESTOS": "Asbestos exposure",
    "SECONDHAND_SMOKE": "Secondhand smoke", "COPD": "COPD diagnosis",
    "ALCOHOL": "Alcohol use", "FAMILY_HISTORY": "Family history",
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
    for s_ in ("top", "right", "left"):
        ax.spines[s_].set_visible(False)
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
<p>Answer nine simple questions about your age, smoking and surroundings. An AI model estimates your risk and shows exactly which answers influenced the result.</p>
<a class="btn" href="#check">Start the check</a><a class="btn ghost" href="#how">How it works</a>
</div>
<div class="rings"></div>
</div>
<div class="how" id="how">
<h2>Three steps, no medical terms</h2>
<div class="steps">
<div class="step"><div class="n">1</div><h3>Answer simple questions</h3><p>Tell us your age, how much you have smoked, and what you have been exposed to. No lab values needed.</p></div>
<div class="step"><div class="n">2</div><h3>Get a risk estimate</h3><p>A Random Forest model trained on 50,000 patient profiles gives you a lower, moderate or higher risk result.</p></div>
<div class="step"><div class="n">3</div><h3>See why</h3><p>SHAP explains which answers raised or lowered your result, so it is never a black box.</p></div>
</div>
</div>""", unsafe_allow_html=True)

# ---------- checker ----------
st.markdown("""<div class="sec" id="check">
<h2>Your check</h2>
<p>Answer as best you can. If you are unsure about radon, choose Not sure.</p>
</div>""", unsafe_allow_html=True)

with st.form("check"):
    name = st.text_input("Full name (optional, only used on the PDF report)", max_chars=60,
                         placeholder="e.g. Asha Patil",
                         help="Use English letters so the name prints correctly in the report.")
    c1, c2 = st.columns(2)
    gender = c1.radio("Gender", ["Male", "Female"], horizontal=True)
    age = c2.slider("Age", 18, 100, 45)

    st.markdown("**Smoking**")
    s1, s2 = st.columns(2)
    cigs = s1.number_input("Cigarettes per day (0 if you never smoked)", 0, 80, 0)
    years = s2.number_input("Years you smoked", 0, 80, 0)

    st.markdown("**Exposure and health**")
    e1, e2 = st.columns(2)
    radon = e1.selectbox("Radon at home or work (a gas from soil; many homes are tested)",
                         ["Not sure", "Low", "Medium", "High"])
    alcohol = e2.selectbox("Alcohol use", ["None", "Moderate", "Heavy"])
    asbestos = e1.toggle("I have worked with or lived near asbestos")
    secondhand = e2.toggle("I am regularly around other people's smoke")
    copd = e1.toggle("A doctor has told me I have COPD")
    family = e2.toggle("Family history of cancer or lung disease")

    submitted = st.form_submit_button("Check my risk")

if submitted:
    pack_years = min(cigs / 20 * years, 100)   # 1 pack = 20 cigarettes
    answers = {
        "AGE": age,
        "GENDER": 1 if gender == "Male" else 0,
        "PACK_YEARS": pack_years,
        "RADON": {"Low": 0, "Not sure": 1, "Medium": 1, "High": 2}[radon],
        "ASBESTOS": int(asbestos),
        "SECONDHAND_SMOKE": int(secondhand),
        "COPD": int(copd),
        "ALCOHOL": {"None": 0, "Moderate": 1, "Heavy": 2}[alcohol],
        "FAMILY_HISTORY": int(family),
    }
    row = pd.DataFrame([answers])[features]
    prob = model.predict_proba(row)[0][1]
    if prob >= 0.75:
        color, label = ROSE, "Higher than average"
        advice = "Please consider talking to a doctor about screening, such as a low-dose CT scan, and avoid smoking."
    elif prob >= 0.50:
        color, label = AMBER, "Moderate risk"
        advice = "Several of your answers match patients in the data. Consider a check-up and reduce exposure where you can."
    else:
        color, label = TEAL, "Lower than average"
        advice = ("You have a smoking history, which raises lung cancer risk over time. Quitting helps at any age, and regular check-ups are wise."
                  if pack_years > 0 else
                  "Keep up healthy habits, avoid smoking, and get regular check-ups.")

    # Expert rule: the practice dataset understates smoking, so 10+ pack-years is never shown as "lower".
    floor_note = ""
    rule_applied = False
    if pack_years >= 10 and label == "Lower than average":
        rule_applied = True
        color, label = AMBER, "Moderate risk"
        advice = ("Your smoking history of 10 or more pack-years raises lung cancer risk over time. "
                  "Quitting helps at any age, and regular check-ups are wise.")
        floor_note = ('<div style="margin-top:.9rem;color:#5B7176;font-size:.92rem">'
                      'Raised to Moderate by a medical rule for smoking history, because the practice dataset understates smoking.</div>')

    contrib_all = dict(zip([NICE[f] for f in features], risk_shap_values(row)))

    if pack_years > 0:
        smoking_txt = f"{cigs} cigarettes/day for {years} years ({pack_years:.1f} pack-years)"
    else:
        smoking_txt = "No smoking history entered"
    parameters = [
        ("Age", f"{age} years"),
        ("Gender", gender),
        ("Smoking history", smoking_txt),
        ("Radon exposure", "Not sure (treated as Medium)" if radon == "Not sure" else radon),
        ("Asbestos exposure", "Yes" if asbestos else "No"),
        ("Secondhand smoke", "Yes" if secondhand else "No"),
        ("COPD diagnosis", "Yes" if copd else "No"),
        ("Alcohol use", alcohol),
        ("Family history", "Yes" if family else "No"),
    ]
    now = datetime.now()
    report_id = f"LW-{now:%Y%m%d}-{uuid.uuid4().hex[:4].upper()}"
    pdf_bytes = build_report(name, age, gender, parameters, contrib_all, prob, label, advice,
                             rule_applied, report_id, f"{now:%d %b %Y, %H:%M}")
    # Keep the result in session_state so it stays on screen after the download button is clicked.
    st.session_state["result"] = dict(prob=prob, color=color, label=label, advice=advice, floor_note=floor_note,
                                      contrib=contrib_all, pdf=pdf_bytes, report_id=report_id)

res = st.session_state.get("result")
if res:
    st.markdown(f"""<div class="result">
<div class="label" style="color:{res['color']}">{res['label']}</div>
<div class="meta">Model score: {res['prob']:.0%}, compared with patients in a synthetic practice dataset (not a real-world probability)</div>
<div class="bar"><span style="width:{res['prob']*100:.0f}%; background:{res['color']}"></span></div>
<div>{res['advice']}</div>
{res['floor_note']}
</div>""", unsafe_allow_html=True)

    st.markdown("""<div class="sec"><h2>Why the model said this</h2></div>""", unsafe_allow_html=True)
    contrib = pd.Series(res["contrib"])
    contrib = contrib[contrib.abs() > 1e-6].sort_values()
    up = contrib[contrib > 0].sort_values(ascending=False).head(3)
    down = contrib[contrib < 0].sort_values().head(3)
    if len(up):
        st.markdown(f"**Raised the estimate:** {', '.join(up.index)}")
    if len(down):
        st.markdown(f"**Lowered it:** {', '.join(down.index)}")

    fig, ax = plt.subplots(figsize=(6.4, max(2, 0.4 * len(contrib))))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.barh(contrib.index, contrib.values,
            color=[ROSE if v > 0 else TEAL for v in contrib.values], height=0.6)
    ax.axvline(0, color=LINE, linewidth=1)
    ax.set_xlabel("Effect on risk (red raises, teal lowers)")
    clean_axes(ax)
    plt.tight_layout()
    st.pyplot(fig, transparent=True)

    st.markdown("""<div class="sec"><h2>Your report</h2>
<p>Want a copy? Download a lab-style PDF with your details, results and the explanation. Your name and answers are not stored.</p></div>""",
                unsafe_allow_html=True)
    st.download_button("Download my report (PDF)", data=res["pdf"],
                       file_name=f"Lungwise_Report_{res['report_id']}.pdf", mime="application/pdf")

with st.expander("Which factors matter most overall?"):
    imp = pd.Series(model.feature_importances_, index=[NICE[f] for f in features]).sort_values()
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