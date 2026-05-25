"""
Step 23: Balanced-site sensitivity analysis.
Exclude sites with <5 ASD or <5 TD and re-run all key models.
"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings("ignore")
import statsmodels.api as sm

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUB_CSV = os.path.join(BASE, "code", "combined_subjects.csv")

def find_abide_dir():
    for f in os.listdir("E:/"):
        if f.startswith("ABIDE") and "ABIDE" in f.upper():
            return os.path.join("E:/", f)

ABIDE_DIR = find_abide_dir()
FC_CSV = os.path.join(ABIDE_DIR, "timeseries", "seed_connectivity.csv")
QC_CSV = os.path.join(ABIDE_DIR, "results", "coverage_qc.csv")
OUT_DIR = os.path.join(ABIDE_DIR, "results")
EXCLUDE_SITES = ["ABIDEII-KUL_3", "ABIDEII-NYU_2", "ABIDEII-IP_1"]

# Load
fc = pd.read_csv(FC_CSV, dtype={"SUB_ID": str})
demo = pd.read_csv(SUB_CSV, dtype={"SUB_ID": str})
qc = pd.read_csv(QC_CSV) if os.path.exists(QC_CSV) else None

df = fc.merge(demo, on=["SUB_ID", "SITE_ID", "DX"], how="left", suffixes=("", "_y"))
df["age"] = pd.to_numeric(df["AGE_AT_SCAN"], errors="coerce")
df["fd"] = pd.to_numeric(df["func_mean_fd"], errors="coerce")
df["dx_num"] = (df["DX"] == "ASD").astype(int)

# Filter
balanced = df[~df["SITE_ID"].isin(EXCLUDE_SITES)].copy()
print(f"Full sample: {len(df)} ({df['SITE_ID'].nunique()} sites)")
print(f"Balanced:    {len(balanced)} ({balanced['SITE_ID'].nunique()} sites)")
print(f"Excluded:    {len(df) - len(balanced)} subjects from {EXCLUDE_SITES}")

# Pre-compute site dummies (same set for all models)
def run_model(data, fc_col, label, covariates, sample_desc):
    d = data.dropna(subset=[fc_col, "dx_num", "age", "fd", "SITE_ID"]).copy()
    # If specific covariate list doesn't include fd/age, adjust
    cov_keep = [c for c in covariates if c not in ["site", "dx_num"]]
    d = d.dropna(subset=[fc_col, "dx_num"] + cov_keep).reset_index(drop=True)
    d["intercept"] = 1.0

    X_parts = [d[["intercept", "dx_num"] + cov_keep]]
    X_parts.append(pd.get_dummies(d["SITE_ID"], prefix="site", drop_first=True))
    X = pd.concat(X_parts, axis=1).astype(float)
    y = d[fc_col].values.astype(float)

    m = sm.OLS(y, X).fit()
    idx = list(X.columns).index("dx_num")
    return {"model": label, "sample": sample_desc, "N": len(d),
            "ASD": int((d["DX"]=="ASD").sum()),
            "TD": int((d["DX"]=="TD").sum()),
            "beta": round(m.params.iloc[idx], 5),
            "ci_lower": round(m.conf_int().iloc[idx, 0], 5),
            "ci_upper": round(m.conf_int().iloc[idx, 1], 5),
            "t": round(m.tvalues.iloc[idx], 4),
            "p": round(m.pvalues.iloc[idx], 5)}

rows = []
pairs = [("PCC_mPFC_z", "PCC-mPFC"), ("PCC_zAI_z", "PCC-rAI"), ("rAI_mPFC_z", "rAI-mPFC")]

for fc_col, fc_name in pairs:
    # Full sample for comparison
    rows.append(run_model(df, fc_col, f"{fc_name} full", ["age", "fd"], "full sample"))
    # Balanced
    rows.append(run_model(balanced, fc_col, f"{fc_name} balanced", ["age", "fd"], "balanced"))
    # Balanced + no FD
    rows.append(run_model(balanced, fc_col, f"{fc_name} balanced noFD", ["age"], "balanced"))
    # ABIDE I balanced
    b1 = balanced[balanced["dataset"] == "ABIDE_I"]
    if len(b1) > 0:
        rows.append(run_model(b1, fc_col, f"{fc_name} ABIDE I", ["age", "fd"], "balanced"))
    # ABIDE II balanced
    b2 = balanced[balanced["dataset"] == "ABIDE_II"]
    if len(b2) > 0:
        rows.append(run_model(b2, fc_col, f"{fc_name} ABIDE II", ["age", "fd"], "balanced"))

rdf = pd.DataFrame(rows)
rdf.to_csv(os.path.join(OUT_DIR, "balanced_site_sensitivity.csv"), index=False)

print(f"\n{'='*60}")
print("Balanced-Site Sensitivity Analysis")
print(f"{'='*60}")
for _, r in rdf.iterrows():
    sig = " *" if r["p"] < 0.05 else ""
    print(f"  {r['model']:<28s} | {r['sample']:<12s} | N={r['N']:>3d} | "
          f"β={r['beta']:>7.4f} [{r['ci_lower']:>6.4f}, {r['ci_upper']:>6.4f}] | p={r['p']:.4f}{sig}")
print("Done.")
