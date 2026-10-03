import numpy as np
import pandas as pd
import joblib

model = joblib.load("disease_model.pkl")
features = joblib.load("features.pkl")


def prob(**kw):
    row = {f: 0 for f in features}
    row.update({"GENDER": 1, "AGE": 40})
    row.update(kw)
    return model.predict_proba(pd.DataFrame([row])[features])[0][1]


print("Healthy 25yo female :", f"{prob(GENDER=0, AGE=25):.0%}")
all_yes = {f: 1 for f in features if f not in ("GENDER", "AGE")}
print("Everything YES, 70yo:", f"{prob(AGE=70, **all_yes):.0%}")

print("\nEffect of switching ONE answer on (baseline = all NO, age 40):")
base = prob()
print(f"baseline: {base:.0%}")
for f in features:
    if f in ("GENDER", "AGE"):
        continue
    p = prob(**{f: 1})
    print(f"{f:25s} {p:.0%}  ({p - base:+.0%})")

print("\nAge sweep (all NO):")
for a in range(20, 90, 10):
    print(a, f"{prob(AGE=a):.0%}")

# How many different outputs does the model produce for random inputs?
rng = np.random.default_rng(0)
rows = pd.DataFrame(rng.integers(0, 2, (2000, len(features))), columns=features)
rows["AGE"] = rng.integers(20, 85, 2000)
rows["GENDER"] = rng.integers(0, 2, 2000)
p = model.predict_proba(rows[features])[:, 1]
print(f"\n2000 random people -> {len(np.unique(p.round(2)))} distinct scores, "
      f"min {p.min():.0%}, max {p.max():.0%}, share above 50%: {(p > .5).mean():.0%}")
