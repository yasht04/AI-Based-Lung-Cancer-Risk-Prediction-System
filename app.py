import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Lung cancer risk check", page_icon="🫁", layout="centered")

INK, MUTED, LINE = "#1B2430", "#6B7685", "#E6E9EE"
ACCENT, WARN = "#0F766E", "#C2410C"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
html, body, .stApp, .stApp * {{ font-family: 'DM Sans', sans-serif; }}
.stApp {{ background: #F7F8FA; color: {INK}; }}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ max-width: 720px; padding-top: 3rem; padding-bottom: 4rem; }}
h1 {{ font-weight: 700; letter-spacing: -0.02em; font-size: 2.1rem; margin-bottom: .2rem; }}
h3 {{ font-weight: 600; font-size: 1.15rem; margin-top: 1.8rem; }}
.sub {{ color: {MUTED}; font-size: 1.02rem; line-height: 1.5; margin-bottom: 1.6rem; }}
[data-testid="stForm"] {{ background: #fff; border: 1px solid {LINE}; border-radius: 14px; padding: 1.6rem; }}
[data-testid="stFormSubmitButton"] button {{
  background: {ACCENT}; color: #fff; border: 0; border-radius: 10px;
  padding: .65rem 1.6rem; font-weight: 600; width: 100%; }}
[data-testid="stFormSubmitButton"] button:hover {{ background: #115E59; color: #fff; }}
.result {{ background: #fff; border: 1px solid {LINE}; border-radius: 14px; padding: 1.6rem; margin-top: 1.6rem; }}
.result .label {{ font-size: 1.6rem; font-weight: 700; letter-spacing: -0.01em; }}
.result .meta {{ color: {MUTED}; margin-top: .2rem; }}
.bar {{ height: 8px; background: #EAEEF2; border-radius: 99px; overflow: hidden; margin: 1rem 0; }}
.bar > span {{ display: block; height: 100%; border-radius: 99px; }}
.note {{ color: {MUTED}; font-size: .85rem; margin-top: 1.4rem; }}
</style>
""", unsafe_allow_html=True)

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
    ax.tick_params(colors=INK, length=0)
    ax.xaxis.label.set_color(MUTED)


st.title("Lung cancer risk check")
st.markdown('<p class="sub">Answer a few yes/no questions. The model estimates your risk '
            'and shows what influenced the result.</p>', unsafe_allow_html=True)

with st.form("check"):
    c1, c2 = st.columns([1, 2])
    gender = c1.radio("Gender", ["Male", "Female"], horizontal=True)
    age = c2.slider("Age", 10, 100, 40)

    st.markdown("**Which of these apply to you?**")
    answers = {"GENDER": 1 if gender == "Male" else 0, "AGE": age}
    cols = st.columns(2)
    for i, (col, text) in enumerate(questions.items()):
        answers[col] = int(cols[i % 2].toggle(text))

    submitted = st.form_submit_button("Check risk")

if submitted:
    row = pd.DataFrame([answers])[features]
    prob = model.predict_proba(row)[0][1]
    high = prob >= 0.5
    color = WARN if high else ACCENT
    label = "Higher risk" if high else "Lower risk"
    advice = ("Please consider seeing a doctor about a chest X-ray or CT scan, "
              "and avoid smoking." if high else
              "Keep up healthy habits, avoid smoking, and get regular check-ups.")

    st.markdown(f"""
    <div class="result">
      <div class="label" style="color:{color}">{label}</div>
      <div class="meta">Model estimate: {prob:.0%} chance</div>
      <div class="bar"><span style="width:{prob*100:.0f}%; background:{color}"></span></div>
      <div>{advice}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### What influenced this result")
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
            color=[WARN if v > 0 else ACCENT for v in contrib.values], height=0.6)
    ax.axvline(0, color=LINE, linewidth=1)
    ax.set_xlabel("Effect on risk (orange raises, teal lowers)")
    clean_axes(ax)
    plt.tight_layout()
    st.pyplot(fig, transparent=True)

    st.markdown('<p class="note">This is an educational screening tool, not a medical diagnosis.</p>',
                unsafe_allow_html=True)

with st.expander("Which factors matter most overall?"):
    imp = pd.Series(model.feature_importances_, index=[nice[f] for f in features]).sort_values()
    fig2, ax2 = plt.subplots(figsize=(6.4, 5))
    fig2.patch.set_alpha(0)
    ax2.set_facecolor("none")
    ax2.barh(imp.index, imp.values, color=ACCENT, height=0.6)
    ax2.set_xlabel("Feature importance")
    clean_axes(ax2)
    plt.tight_layout()
    st.pyplot(fig2, transparent=True)