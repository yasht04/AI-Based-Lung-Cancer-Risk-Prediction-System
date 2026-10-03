import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, classification_report

# Kaggle "Lung Cancer Risk" dataset (lung_cancer_dataset.csv, 50,000 rows)
# keep_default_na=False: otherwise pandas reads the alcohol answer "None" as a missing value
df = pd.read_csv("lung_cancer_dataset.csv", keep_default_na=False)

LEVEL3 = {"Low": 0, "Medium": 1, "High": 2}
ALCOHOL = {"None": 0, "Moderate": 1, "Heavy": 2}
YESNO = {"No": 0, "Yes": 1}

X = pd.DataFrame({
    "AGE": df["age"],
    "GENDER": (df["gender"] == "Male").astype(int),
    "PACK_YEARS": df["pack_years"],
    "RADON": df["radon_exposure"].map(LEVEL3),
    "ASBESTOS": df["asbestos_exposure"].map(YESNO),
    "SECONDHAND_SMOKE": df["secondhand_smoke_exposure"].map(YESNO),
    "COPD": df["copd_diagnosis"].map(YESNO),
    "ALCOHOL": df["alcohol_consumption"].map(ALCOHOL),
    "FAMILY_HISTORY": df["family_history"].map(YESNO),
})
y = (df["lung_cancer"] == "Yes").astype(int)
print("Rows:", len(X), "| missing values:", int(X.isna().sum().sum()), "| share with cancer:", round(y.mean(), 3))

# Domain-knowledge constraint: no factor may ever LOWER the risk (gender left free). Needs scikit-learn >= 1.4.
monotonic = [0 if c == "GENDER" else 1 for c in X.columns]

model = RandomForestClassifier(
    n_estimators=300,
    min_samples_leaf=50,     # large leaves give smooth, better-calibrated probabilities
    monotonic_cst=monotonic,
    random_state=42,
    n_jobs=-1,
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model.fit(X_train, y_train)
proba = model.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)
print("Accuracy:", round(accuracy_score(y_test, pred), 3))
print("ROC-AUC :", round(roc_auc_score(y_test, proba), 3))
print("Brier   :", round(brier_score_loss(y_test, proba), 3),
      "(always predicting the base rate:", round(brier_score_loss(y_test, [y_train.mean()] * len(y_test)), 3), ")")
print(classification_report(y_test, pred))

model.fit(X, y)  # final model on all data for the app
joblib.dump(model, "disease_model.pkl")
joblib.dump(list(X.columns), "features.pkl")
print("Saved disease_model.pkl and features.pkl")