"""
Step 19: Mixed-effects model with site as random effect.
FC ~ DX + age + FD + (1 | site), then + (DX | site).
"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUB_CSV = os.path.join(BASE, "code", "combined_subjects.csv")

def find_abide_dir():
    for f in os.listdir("E:/"):
        if f.startswith("ABIDE") and "ABIDE" in f.upper():
            return os.path.join("E:/", f)

ABIDE_DIR = find_abide_dir()
FC_CSV = os.path.join(ABIDE_DIR, "timeseries", "seed_connectivity.csv")
OUT_DIR = os.path.join(ABIDE_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

FC_COL = "PCC_mPFC_z"

fc = pd.read_csv(FC_CSV, dtype={"SUB_ID": str})
demo = pd.read_csv(SUB_CSV, dtype={"SUB_ID": str})[["SUB_ID", "SITE_ID", "DX", "AGE_AT_SCAN", "func_mean_fd"]]
df = fc.merge(demo, on=["SUB_ID", "SITE_ID", "DX"], how="left")
df["age"] = pd.to_numeric(df["AGE_AT_SCAN"], errors="coerce")
df["fd"] = pd.to_numeric(df["func_mean_fd"], errors="coerce")
df["dx_num"] = (df["DX"] == "ASD").astype(int)
df = df.dropna(subset=[FC_COL, "dx_num", "age", "fd", "SITE_ID"]).reset_index(drop=True)

n_total = len(df)
n_asd = (df["DX"] == "ASD").sum()
n_td = (df["DX"] == "TD").sum()
print(f"N = {n_total} (ASD={n_asd}, TD={n_td}), sites = {df['SITE_ID'].nunique()}")

# MixedLM needs categories as codes
df["site_code"] = df["SITE_ID"].astype("category").cat.codes

import statsmodels.api as sm
from statsmodels.formula.api import mixedlm

# Model 1: random intercept
m1 = mixedlm(f"{FC_COL} ~ dx_num + age + fd", data=df,
             groups=df["site_code"]).fit()
print(f"\n{'='*55}")
print("Mixed Model: random intercept")
print(f"{'='*55}")
print(m1.summary())

# Model 2: random intercept + random slope for DX
try:
    m2 = mixedlm(f"{FC_COL} ~ dx_num + age + fd", data=df,
                 groups=df["site_code"],
                 re_formula="~dx_num").fit()
    print(f"\n{'='*55}")
    print("Mixed Model: random intercept + random slope (DX)")
    print(f"{'='*55}")
    print(m2.summary())
    m2_result = m2
except Exception as e:
    print(f"\nRandom slope model failed: {e}")
    m2_result = None

# Save
results = []
for name, m in [("random_intercept", m1)]:
    for var in ["dx_num", "age", "fd"]:
        if var in m.params:
            results.append({
                "model": name, "term": var,
                "beta": round(m.params[var], 5),
                "SE": round(m.bse[var], 5),
                "z": round(m.tvalues[var], 4),
                "p": round(m.pvalues[var], 5),
                "ci_lower": round(m.params[var] - 1.96*m.bse[var], 5),
                "ci_upper": round(m.params[var] + 1.96*m.bse[var], 5),
            })

if m2_result is not None:
    for name, m in [("random_slope", m2_result)]:
        for var in ["dx_num", "age", "fd"]:
            if var in m.params:
                results.append({
                    "model": name, "term": var,
                    "beta": round(m.params[var], 5),
                    "SE": round(m.bse[var], 5),
                    "z": round(m.tvalues[var], 4),
                    "p": round(m.pvalues[var], 5),
                    "ci_lower": round(m.params[var] - 1.96*m.bse[var], 5),
                    "ci_upper": round(m.params[var] + 1.96*m.bse[var], 5),
                })

rdf = pd.DataFrame(results)
rdf.to_csv(os.path.join(OUT_DIR, "mixed_model.csv"), index=False)
print(f"\nResults saved.")
print("Done.")
