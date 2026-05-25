"""
Step 22: Fix ABIDE II FD — re-run primary model with corrected FD.
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
OUT_DIR = os.path.join(ABIDE_DIR, "results")
FC_COL = "PCC_mPFC_z"

# Load
fc = pd.read_csv(FC_CSV, dtype={"SUB_ID": str})
demo = pd.read_csv(SUB_CSV, dtype={"SUB_ID": str})
real_fd = pd.read_csv(os.path.join(OUT_DIR, "abide2_real_fd.csv"))
real_fd["SUB_ID"] = real_fd["SUB_ID"].astype(str)

# Merge real FD into demo
demo = demo.merge(real_fd[["SUB_ID", "real_mean_fd"]], on="SUB_ID", how="left")

# Compute site-level means for imputation (ABIDE II only)
site_fd = {}
for site in demo[demo["dataset"]=="ABIDE_II"]["SITE_ID"].unique():
    vals = demo[(demo["SITE_ID"]==site) & demo["real_mean_fd"].notna()]["real_mean_fd"]
    site_fd[site] = vals.mean() if len(vals) > 0 else None

global_fd = demo[(demo["dataset"]=="ABIDE_II") & demo["real_mean_fd"].notna()]["real_mean_fd"].mean()

# Impute ABIDE II missing FD
demo["corrected_fd"] = np.where(demo["dataset"]=="ABIDE_II", np.nan, demo["func_mean_fd"])
for idx in demo.index:
    if demo.loc[idx, "dataset"] != "ABIDE_II":
        continue
    if pd.notna(demo.loc[idx, "real_mean_fd"]):
        demo.loc[idx, "corrected_fd"] = demo.loc[idx, "real_mean_fd"]
    else:
        site = demo.loc[idx, "SITE_ID"]
        if site_fd.get(site) is not None:
            demo.loc[idx, "corrected_fd"] = site_fd[site]
        else:
            demo.loc[idx, "corrected_fd"] = global_fd

# Report
print("FD Correction Summary:")
print(f"  ABIDE I: {len(demo[demo.dataset=='ABIDE_I'])} subjects, FD range [{demo[demo.dataset=='ABIDE_I']['corrected_fd'].min():.4f}, {demo[demo.dataset=='ABIDE_I']['corrected_fd'].max():.4f}]")
a2 = demo[demo.dataset=="ABIDE_II"]
print(f"  ABIDE II: {len(a2)} subjects")
print(f"    Real FD: {a2.real_mean_fd.notna().sum()}, site-imputed: {((a2.real_mean_fd.isna()) & (a2.corrected_fd.notna())).sum()}")
print(f"    Corrected FD: mean={a2['corrected_fd'].mean():.4f}, SD={a2['corrected_fd'].std():.4f}")
print(f"    Placeholder FD: mean=0.0922, SD=0.0000")

# Build analysis df: start from demo, merge FC values
df_all = demo.merge(fc[[FC_COL, "SUB_ID", "SITE_ID", "DX"]], on=["SUB_ID", "SITE_ID", "DX"], how="left")
df_all["age"] = pd.to_numeric(df_all["AGE_AT_SCAN"], errors="coerce")
df_all["corrected_fd"] = pd.to_numeric(df_all["corrected_fd"], errors="coerce")
df_all["func_mean_fd"] = pd.to_numeric(df_all["func_mean_fd"], errors="coerce")
df_all["dx_num"] = (df_all["DX"] == "ASD").astype(int)
df_all["intercept"] = 1.0

def run_model(data, fd_col, label):
    d = data.dropna(subset=[FC_COL, "dx_num", "age", fd_col, "SITE_ID"]).reset_index(drop=True)
    X = pd.concat([d[["intercept", "dx_num", "age", fd_col]],
                   pd.get_dummies(d["SITE_ID"], prefix="site", drop_first=True)], axis=1).astype(float)
    y = d[FC_COL].values.astype(float)
    m = sm.OLS(y, X).fit()
    idx = list(X.columns).index("dx_num")
    return {"model": label, "N": len(d),
            "ASD": int((d["DX"]=="ASD").sum()),
            "TD": int((d["DX"]=="TD").sum()),
            "beta": round(m.params.iloc[idx], 5),
            "ci_lower": round(m.conf_int().iloc[idx, 0], 5),
            "ci_upper": round(m.conf_int().iloc[idx, 1], 5),
            "t": round(m.tvalues.iloc[idx], 4),
            "p": round(m.pvalues.iloc[idx], 5)}

rows = []
rows.append(run_model(df_all, "func_mean_fd", "Original (placeholder FD)"))
rows.append(run_model(df_all, "corrected_fd", "Corrected FD (full sample)"))

# Complete case: ABIDE I + ABIDE II with real FD only
mask = (df_all["dataset"]=="ABIDE_I") | df_all["real_mean_fd"].notna()
rows.append(run_model(df_all[mask], "corrected_fd", "Complete case (real FD only)"))

# Corrected FD without FD covariate
d_no_fd = df_all.dropna(subset=[FC_COL, "dx_num", "age", "SITE_ID"]).reset_index(drop=True)
X_no_fd = pd.concat([d_no_fd[["intercept", "dx_num", "age"]],
                     pd.get_dummies(d_no_fd["SITE_ID"], prefix="site", drop_first=True)], axis=1).astype(float)
y_no_fd = d_no_fd[FC_COL].values.astype(float)
m_no_fd = sm.OLS(y_no_fd, X_no_fd).fit()
idx_no = list(X_no_fd.columns).index("dx_num")
rows.append({"model": "No FD covariate", "N": len(d_no_fd),
             "ASD": int((d_no_fd["DX"]=="ASD").sum()),
             "TD": int((d_no_fd["DX"]=="TD").sum()),
             "beta": round(m_no_fd.params.iloc[idx_no], 5),
             "ci_lower": round(m_no_fd.conf_int().iloc[idx_no, 0], 5),
             "ci_upper": round(m_no_fd.conf_int().iloc[idx_no, 1], 5),
             "t": round(m_no_fd.tvalues.iloc[idx_no], 4),
             "p": round(m_no_fd.pvalues.iloc[idx_no], 5)})

rdf = pd.DataFrame(rows)
rdf.to_csv(os.path.join(OUT_DIR, "fd_correction_comparison.csv"), index=False)

print(f"\n{'='*55}")
print("Primary Model: FD Correction Comparison")
print(f"{'='*55}")
for _, r in rdf.iterrows():
    sig = " *" if r["p"] < 0.05 else ""
    print(f"  {r['model']:<35s} N={r['N']:>3d} | β={r['beta']:>7.4f} [{r['ci_lower']:>7.4f}, {r['ci_upper']:>7.4f}] | p={r['p']:.4f}{sig}")
print("Done.")
