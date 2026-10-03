import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Kaggle "Lung Cancer" survey dataset (survey lung cancer.csv)
df = pd.read_csv("survey lung cancer.csv")
df.columns = df.columns.str.strip()

# Encode: gender M=1/F=0, symptoms 1/2 -> 0/1, target YES=1/NO=0
df["GENDER"] = df["GENDER"].map({"M": 1, "F": 0})
df["LUNG_CANCER"] = df["LUNG_CANCER"].map({"YES": 1, "NO": 0})
symptom_cols = [c for c in df.columns if c not in ["GENDER", "AGE", "LUNG_CANCER"]]
df[symptom_cols] = df[symptom_cols] - 1

df = df.drop_duplicates()
X = df.drop(columns=["LUNG_CANCER"])
y = df["LUNG_CANCER"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# class_weight handles the imbalance (far more YES than NO rows)
model = RandomForestClassifier(
    n_estimators=200, class_weight="balanced", random_state=42
)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

joblib.dump(model, "disease_model.pkl")
joblib.dump(list(X.columns), "features.pkl")
print("Saved disease_model.pkl and features.pkl")
