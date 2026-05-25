"""
Step 20: IQ sensitivity — check if PCC-mPFC ASD<TD survives with FIQ as covariate.
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
demo = pd.read_csv(SUB_CSV, dtype={"SUB_ID": str})[["SUB_ID", "SITE_ID", "DX", "AGE_AT_SCAN", "func_mean_fd", "FIQ"]]
df = fc.merge(demo, on=["SUB_ID", "SITE_ID", "DX"], how="left")
df["age"] = pd.to_numeric(df["AGE_AT_SCAN"], errors="coerce")
df["fd"] = pd.to_numeric(df["func_mean_fd"], errors="coerce")
df["fiq"] = pd.to_numeric(df["FIQ"], errors="coerce")
df["dx_num"] = (df["DX"] == "ASD").astype(int)
df["intercept"] = 1.0

import statsmodels.api as sm

def run_base(data, label):
    data = data.dropna(subset=[FC_COL, "dx_num", "age", "fd", "SITE_ID"]).reset_index(drop=True)
    X = pd.concat([data[["intercept", "dx_num", "age", "fd"]],
                   pd.get_dummies(data["SITE_ID"], prefix="site", drop_first=True)], axis=1).astype(float)
    y = data[FC_COL].values.astype(float)
    m = sm.OLS(y, X).fit()
    idx = list(X.columns).index("dx_num")
    return {"model": label, "N": len(data),
            "beta": round(m.params.iloc[idx], 5),
            "ci_lower": round(m.conf_int().iloc[idx, 0], 5),
            "ci_upper": round(m.conf_int().iloc[idx, 1], 5),
            "t": round(m.tvalues.iloc[idx], 4),
            "p": round(m.pvalues.iloc[idx], 5)}

def run_fiq(data, label):
    data = data.dropna(subset=[FC_COL, "dx_num", "age", "fd", "fiq", "SITE_ID"]).reset_index(drop=True)
    X = pd.concat([data[["intercept", "dx_num", "age", "fd", "fiq"]],
                   pd.get_dummies(data["SITE_ID"], prefix="site", drop_first=True)], axis=1).astype(float)
    y = data[FC_COL].values.astype(float)
    m = sm.OLS(y, X).fit()
    idx = list(X.columns).index("dx_num")
    return {"model": label, "N": len(data),
            "beta": round(m.params.iloc[idx], 5),
            "ci_lower": round(m.conf_int().iloc[idx, 0], 5),
            "ci_upper": round(m.conf_int().iloc[idx, 1], 5),
            "t": round(m.tvalues.iloc[idx], 4),
            "p": round(m.pvalues.iloc[idx], 5)}

rows = []
rows.append(run_base(df, "No IQ (full sample)"))
rows.append(run_base(df.dropna(subset=["fiq"]), "FIQ subset, no FIQ"))
rows.append(run_fiq(df, "With FIQ"))

# FIQ descriptive
iq_avail = df["fiq"].notna()
print(f"FIQ available: {iq_avail.sum()}/{len(df)}")
for grp in ["ASD", "TD"]:
    g = df[(df["DX"] == grp) & iq_avail]["fiq"]
    print(f"  {grp}: mean={g.mean():.1f}, SD={g.std():.1f}, N={len(g)}")

rdf = pd.DataFrame(rows)
rdf.to_csv(os.path.join(OUT_DIR, "iq_sensitivity.csv"), index=False)

print(f"\nIQ Sensitivity Results")
print(f"{'='*55}")
for _, r in rdf.iterrows():
    sig = " *" if r["p"] < 0.05 else ""
    print(f"  {r['model']:<30s} N={r['N']:>3d} | beta={r['beta']:>7.4f} [{r['ci_lower']:>7.4f}, {r['ci_upper']:>7.4f}] | p={r['p']:.4f}{sig}")
print("Done.")
