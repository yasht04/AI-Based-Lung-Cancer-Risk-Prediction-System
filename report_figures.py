"""Creates ROC, calibration and tier charts for the report (5-fold cross-validated)."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve

df = pd.read_csv("lung_cancer_dataset.csv", keep_default_na=False)
X = pd.DataFrame({
    "AGE": df["age"], "GENDER": (df["gender"] == "Male").astype(int), "PACK_YEARS": df["pack_years"],
    "RADON": df["radon_exposure"].map({"Low": 0, "Medium": 1, "High": 2}),
    "ASBESTOS": df["asbestos_exposure"].map({"No": 0, "Yes": 1}),
    "SECONDHAND_SMOKE": df["secondhand_smoke_exposure"].map({"No": 0, "Yes": 1}),
    "COPD": df["copd_diagnosis"].map({"No": 0, "Yes": 1}),
    "ALCOHOL": df["alcohol_consumption"].map({"None": 0, "Moderate": 1, "Heavy": 2}),
    "FAMILY_HISTORY": df["family_history"].map({"No": 0, "Yes": 1}),
})
y = (df["lung_cancer"] == "Yes").astype(int)
rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=50, random_state=42, n_jobs=-1,
                            monotonic_cst=[0 if c == "GENDER" else 1 for c in X.columns])
oof = cross_val_predict(rf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42), method="predict_proba")[:, 1]

os.makedirs("report_figures", exist_ok=True)
TEAL, ROSE, AMBER, GREY = "#1F8A84", "#D1495B", "#B7791F", "#9aa7aa"


def tidy(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


auc = roc_auc_score(y, oof)
fpr, tpr, _ = roc_curve(y, oof)
fig, ax = plt.subplots(figsize=(4.6, 4.2))
ax.plot(fpr, tpr, color=TEAL, lw=2, label=f"Random Forest (AUC = {auc:.2f})")
ax.plot([0, 1], [0, 1], "--", color=GREY, label="Chance")
ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve (5-fold cross-validated)")
ax.legend(loc="lower right", frameon=False); tidy(ax); plt.tight_layout()
plt.savefig("report_figures/roc_curve.png", dpi=200); plt.close()

frac, mean_p = calibration_curve(y, oof, n_bins=8, strategy="quantile")
brier, brier0 = brier_score_loss(y, oof), brier_score_loss(y, np.full(len(y), y.mean()))
fig, ax = plt.subplots(figsize=(4.6, 4.2))
ax.plot([0, 1], [0, 1], "--", color=GREY, label="Perfect calibration")
ax.plot(mean_p, frac, "o-", color=TEAL, lw=2, label=f"Model (Brier {brier:.3f})")
ax.set(xlabel="Mean model score", ylabel="Actual share with cancer", title="Calibration (5-fold cross-validated)")
ax.legend(loc="lower right", frameon=False); tidy(ax); plt.tight_layout()
plt.savefig("report_figures/calibration_curve.png", dpi=200); plt.close()

tier = pd.cut(oof, [0, 0.5, 0.75, 1.0], labels=["Lower", "Moderate", "Higher"], right=False)
t = pd.DataFrame({"tier": tier, "y": y}).groupby("tier", observed=True).y.agg(["mean", "count"])
fig, ax = plt.subplots(figsize=(4.6, 3.8))
bars = ax.bar(t.index.astype(str), t["mean"] * 100, color=[TEAL, AMBER, ROSE], width=0.55)
for b, (m, n) in zip(bars, zip(t["mean"], t["count"])):
    ax.text(b.get_x() + b.get_width() / 2, m * 100 + 1.5, f"{m:.0%}\n(n={n})", ha="center", fontsize=9)
ax.set(ylabel="Share with cancer (%)", ylim=(0, 115), title="Observed cancer rate by result tier"); tidy(ax)
plt.tight_layout(); plt.savefig("report_figures/tier_rates.png", dpi=200); plt.close()

print(f"AUC {auc:.3f} | Brier {brier:.3f} vs base-rate {brier0:.3f}")
print(t.round(3))
