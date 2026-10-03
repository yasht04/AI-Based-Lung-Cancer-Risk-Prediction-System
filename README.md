# Explainable AI for Lung Cancer Risk Prediction

A Streamlit web app that estimates lung cancer risk from nine plain-language questions
and explains each result with SHAP.

## Dataset
Kaggle "Lung Cancer Risk" (2025), file `lung_cancer_dataset.csv` (50,000 rows, 11 columns).
Columns used: age, gender, pack_years, radon_exposure, asbestos_exposure,
secondhand_smoke_exposure, copd_diagnosis, alcohol_consumption, family_history.
Target: lung_cancer (Yes/No). The data looks synthetic (uniform age and pack-years, 69% cancer rate).

## Files
- `train.py`: trains the Random Forest and saves the model
- `app.py`: Streamlit app (landing page, form, result, SHAP explanation)
- `report_figures.py`: makes the ROC, calibration and tier charts for the report
- `report_notes.md`: Methods, Results and Limitations text for the report
- `requirements.txt`: dependencies
- `survey_version/`: the earlier version built on the small survey dataset (needs `survey lung cancer.csv`)

## How to run
1. Put `lung_cancer_dataset.csv`, `train.py`, `app.py` in one folder
2. `pip install -r requirements.txt` (needs scikit-learn 1.4 or newer)
3. `python train.py`  (creates `disease_model.pkl` and `features.pkl`)
4. `python report_figures.py`  (optional, makes the charts)
5. `streamlit run app.py`
Optional: put `config.toml` inside a folder named `.streamlit` to lock the light theme.

## How the questions map to the model
- Cigarettes per day and years smoked become pack-years: (cigarettes / 20) x years, capped at 100
- Radon: Low / Medium / High (Not sure is treated as Medium)
- Asbestos, secondhand smoke, COPD, family history: yes/no toggles
- Alcohol: None / Moderate / Heavy

## Result tiers
Lower: score below 50%. Moderate: 50% to 75%. Higher: 75% and above.
Cross-validated, the share of people with cancer was 22%, 65% and 94% in the three tiers.

## Test cases for Step 9 (take a screenshot of each)
| # | Gender | Age | Smoking | Other answers | Score | Result |
|---|--------|-----|---------|---------------|-------|--------|
| 1 | Male | 68 | 20/day for 40 years | Radon High, Alcohol Heavy, asbestos, secondhand smoke, COPD, family history | 84% | Higher |
| 2 | Male | 55 | 20/day for 20 years | Radon Not sure, Alcohol Moderate | 61% | Moderate |
| 3 | Female | 45 | never smoked | Radon Not sure, secondhand smoke | 34% | Lower |
| 4 | Female | 62 | 10/day for 30 years | Radon Low, COPD, family history | 56% | Moderate |
| 5 | Female | 25 | never smoked | Radon Low, nothing else | 20% | Lower |
| 6 | Male | 70 | 30/day for 45 years | Radon High, Alcohol Heavy, asbestos, secondhand smoke, COPD, family history | 94% | Higher |

## Disclaimer
Educational screening demonstration, not a medical diagnosis.