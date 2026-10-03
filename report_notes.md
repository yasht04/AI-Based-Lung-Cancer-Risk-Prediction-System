# Report notes: Methods, Results and Limitations

## Methods
- Dataset: Kaggle "Lung Cancer Risk" (50,000 patient profiles, 9 input features, binary target). Smoking is captured as pack-years; the app asks for cigarettes per day and years smoked and converts them.
- Model: Random Forest (300 trees, min_samples_leaf=50) from scikit-learn.
- Monotonic constraints: age and every exposure/health factor can only raise or leave unchanged the score (gender is unconstrained). This encodes medical knowledge and prevents implausible behaviour.
- Explainability: SHAP TreeExplainer for each prediction; feature importance for the global view.
- Evaluation: 80/20 stratified split plus 5-fold stratified cross-validation.
- Earlier prototype: a 276-row symptom survey dataset (ROC-AUC 0.93). It was replaced because it was small, probably synthetic, and showed almost no effect for smoking (87% vs 85% cancer rate).

## Results
- ROC-AUC 0.76 (5-fold cross-validated; 0.765 on the held-out split).
- Brier score 0.187, better than always predicting the base rate (0.215), so the scores carry real information about probability within this dataset.
- Calibration: model scores track the observed cancer rate closely (calibration_curve.png).
- Result tiers (cross-validated): Lower (<50%) 22% had cancer; Moderate (50-75%) 65%; Higher (75%+) 94% (tier_rates.png).
- Strongest drivers: pack-years of smoking, age, asbestos, COPD, high radon, family history and secondhand smoke. Gender had no effect.

## Limitations (include these)
1. The dataset looks synthetic: age and pack-years are uniformly distributed, most binary features split exactly 50/50, and 69% of rows have cancer although the dataset description cites about 25%. Relationships were probably built in by the creator, so they reflect assumptions, not real patient outcomes.
2. Because 69% of rows have cancer, scores are anchored to that rate. The Moderate tier holds about 74% of all rows, and scores must not be read as real-world risk.
3. ROC-AUC of 0.76 is modest; the earlier 0.93 on the small survey data came from easy patterns in a tiny dataset, not a better model.
4. Some questions are hard for ordinary users (radon, pack-years). "Not sure" for radon is treated as Medium, which is an assumption.
5. Missing risk factors: air pollution, occupation details, time since quitting, and imaging are not included.
6. No clinical validation. This is an educational screening demonstration, not a diagnostic tool.

## Future scope
Validate on real clinical data, calibrate to a real population baseline, add chest X-ray or CT analysis, collect better exposure information, and review with clinicians before any real use.
