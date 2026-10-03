# Explainable AI for Lung Cancer Risk Prediction

A Streamlit web app that predicts lung cancer risk from simple yes/no
symptom and lifestyle questions, and explains each prediction using SHAP.

## Dataset
Kaggle: "Lung Cancer" by Mysar Ahmad Bhat
File needed: `survey lung cancer.csv` (about 309 rows, 16 columns)
Search "survey lung cancer Kaggle" and download the CSV.

## Files
- `train.py`: trains a Random Forest, prints accuracy, saves the model
- `app.py`: Streamlit app (prediction + SHAP explanation)
- `requirements.txt`: dependencies

## How to run
1. Put `survey lung cancer.csv`, `train.py`, `app.py` in one folder
2. `pip install -r requirements.txt`
3. `python train.py`  (creates `disease_model.pkl` and `features.pkl`)
4. `streamlit run app.py`

## Google Colab
Run `!pip install -r requirements.txt`, upload the CSV, run train.py,
then `!streamlit run app.py & npx localtunnel --port 8501`.

## Test cases for Step 9 (take a screenshot of each)
| # | Gender | Age | Answers set to YES | Expected |
|---|--------|-----|--------------------|----------|
| 1 | Male | 65 | Smoking, Coughing, Wheezing, Chest pain, Shortness of breath | High |
| 2 | Female | 25 | none | Low |
| 3 | Male | 58 | Smoking, Yellow fingers, Fatigue, Swallowing difficulty | High |
| 4 | Female | 40 | Allergy, Anxiety | Low |
| 5 | Male | 70 | Smoking, Alcohol, Coughing, Chronic disease | High |
Actual results depend on the trained model, so record what the app shows.

## Disclaimer
Educational screening tool, not a medical diagnosis.
