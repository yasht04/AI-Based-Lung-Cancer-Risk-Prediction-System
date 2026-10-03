import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Explainable Lung Cancer Risk Checker", page_icon="🫁")
model = joblib.load("disease_model.pkl")
features = joblib.load("features.pkl")
explainer = shap.TreeExplainer(model)

st.title("🫁 Explainable Lung Cancer Risk Checker")
st.write("Answer a few simple questions. No medical values needed.")

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 10, 100, 40)

questions = {
    "SMOKING": "Do you smoke?",
    "YELLOW_FINGERS": "Do you have yellow stains on your fingers?",
    "ANXIETY": "Do you often feel anxious?",
    "PEER_PRESSURE": "Are you under peer pressure (to smoke/drink)?",
    "CHRONIC DISEASE": "Do you have a long-term (chronic) illness?",
    "FATIGUE": "Do you feel tired very often?",
    "ALLERGY": "Do you have allergies?",
    "WHEEZING": "Do you have wheezing (whistling sound while breathing)?",
    "ALCOHOL CONSUMING": "Do you drink alcohol regularly?",
    "COUGHING": "Do you have a persistent cough?",
    "SHORTNESS OF BREATH": "Do you get short of breath easily?",
    "SWALLOWING DIFFICULTY": "Do you have difficulty swallowing?",
    "CHEST PAIN": "Do you have chest pain?",
}
nice = {c: c.replace("_", " ").title() for c in features}

answers = {"GENDER": 1 if gender == "Male" else 0, "AGE": age}
for col, text in questions.items():
    answers[col] = int(st.checkbox(text))


def risk_shap_values(row):
    """SHAP values for the 'cancer' class, robust to SHAP version differences."""
    sv = explainer.shap_values(row)
    if isinstance(sv, list):
        return sv[1][0]
    if getattr(sv, "ndim", 2) == 3:
        return sv[0, :, 1]
    return sv[0]


if st.button("Check Risk"):
    row = pd.DataFrame([answers])[features]
    prob = model.predict_proba(row)[0][1]
    if prob >= 0.5:
        st.error(f"⚠️ High risk of lung cancer (confidence: {prob:.0%})")
        st.write("Recommendation: Please consult a doctor and consider a chest X-ray or CT scan. Avoid smoking.")
    else:
        st.success(f"✅ Low risk (confidence: {1 - prob:.0%})")
        st.write("Recommendation: Maintain a healthy lifestyle, avoid smoking, and get regular check-ups.")

    # ---- Explainability (SHAP) ----
    st.subheader("🔍 Why this prediction?")
    contrib = pd.Series(risk_shap_values(row), index=[nice[f] for f in features])
    contrib = contrib[contrib.abs() > 1e-6].sort_values()

    up = contrib[contrib > 0].sort_values(ascending=False).head(3)
    down = contrib[contrib < 0].sort_values().head(3)
    if len(up):
        st.write("**Pushed risk UP:** " + ", ".join(up.index))
    if len(down):
        st.write("**Pushed risk DOWN:** " + ", ".join(down.index))

    fig, ax = plt.subplots(figsize=(6, max(2, 0.35 * len(contrib))))
    ax.barh(contrib.index, contrib.values,
            color=["#d9534f" if v > 0 else "#5cb85c" for v in contrib.values])
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("Effect on cancer risk (red = increases, green = decreases)")
    plt.tight_layout()
    st.pyplot(fig)
    st.caption("This is an AI screening tool, not a medical diagnosis.")

with st.expander("📊 Which factors matter most overall?"):
    imp = pd.Series(model.feature_importances_, index=[nice[f] for f in features]).sort_values()
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.barh(imp.index, imp.values, color="#337ab7")
    ax2.set_xlabel("Feature importance")
    plt.tight_layout()
    st.pyplot(fig2)
