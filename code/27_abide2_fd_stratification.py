"""
Step 27: ABIDE II FD stratification.
Compare participants with real vs imputed FD within ABIDE II:
- Demographic/quality comparison between the two groups
- OLS models separately for each stratum
- Tests whether the null ABIDE II result is driven by imputed FD
"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUB_CSV = os.path.join(BASE, "code", "combined_subjects.csv")

def find_abide_dir():
    for f in os.listdir("E:/"):
        if f.startswith("ABIDE"):
            return os.path.join("E:/", f)
    raise FileNotFoundError("ABIDE dir not found on E:")

ABIDE_DIR = find_abide_dir()
FC_CSV = os.path.join(ABIDE_DIR, "timeseries", "seed_connectivity.csv")
REAL_FD_FILE = os.path.join(ABIDE_DIR, "results", "abide2_real_fd.csv")
OUT_DIR = os.path.join(ABIDE_DIR, "results")

FC_COL = "PCC_mPFC_z"

# ── Load ──
fc = pd.read_csv(FC_CSV, dtype={"SUB_ID": str})
demo = pd.read_csv(SUB_CSV, dtype={"SUB_ID": str})
real_fd = pd.read_csv(REAL_FD_FILE, dtype={"SUB_ID": str}) if os.path.exists(REAL_FD_FILE) else None

# Merge real FD
if real_fd is not None:
    demo = demo.merge(real_fd[["SUB_ID", "real_mean_fd"]], on="SUB_ID", how="left")

# Join FC + demo
df = fc.merge(demo, on=["SUB_ID", "SITE_ID", "DX"], how="left", suffixes=("", "_y"))
df["age"] = pd.to_numeric(df["AGE_AT_SCAN"], errors="coerce")
df["fd"] = pd.to_numeric(df["func_mean_fd"], errors="coerce")
df = df.dropna(subset=["age", "fd", "SITE_ID"]).reset_index(drop=True)
df["dx_num"] = (df["DX"] == "ASD").astype(int)

# ── ABIDE II stratification ──
abide2 = df[df["dataset"] == "ABIDE_II"].copy()
real_mask = abide2["real_mean_fd"].notna() if real_fd is not None else pd.Series(False, index=abide2.index)
abide2["fd_stratum"] = np.where(real_mask, "real_FD", "imputed_FD")

print(f"ABIDE II total: {len(abide2)}")
print(f"  Real FD:      {real_mask.sum()}")
print(f"  Imputed FD:   {(~real_mask).sum()}")

# ── Demographics comparison by stratum ──
print(f"\n{'='*55}")
print("Demographics by FD Stratum")
print(f"{'='*55}")
for stratum in ["real_FD", "imputed_FD"]:
    sub = abide2[abide2["fd_stratum"] == stratum]
    n_asd = (sub["DX"] == "ASD").sum()
    n_td = (sub["DX"] == "TD").sum()
    print(f"\n  {stratum} (N={len(sub)}, ASD={n_asd}, TD={n_td}):")
    print(f"    Age: {sub['age'].mean():.2f} ± {sub['age'].std():.2f}")
    print(f"    FD (func_mean_fd): {sub['fd'].mean():.4f} ± {sub['fd'].std():.4f}")
    if real_fd is not None and stratum == "real_FD":
        print(f"    Real FD: {sub['real_mean_fd'].mean():.4f} ± {sub['real_mean_fd'].std():.4f}")
        print(f"    Placeholder FD: {sub['fd'].mean():.4f} ± {sub['fd'].std():.4f}")
    print(f"    Sites: {sub['SITE_ID'].nunique()}")
    print(f"    PCC-mPFC_z (raw): {sub[FC_COL].mean():.4f} ± {sub[FC_COL].std():.4f}")

# ── Site distribution by stratum ──
print(f"\n  Site distribution:")
site_stratum = abide2.groupby(["SITE_ID", "fd_stratum"]).size().unstack(fill_value=0)
print(site_stratum.to_string())

# ── OLS models ──
import statsmodels.api as sm
abide2["intercept"] = 1.0

def run_model(data, label):
    d = data.dropna(subset=[FC_COL, "dx_num", "age", "fd", "SITE_ID"]).reset_index(drop=True)
    X = pd.concat([d[["intercept", "dx_num", "age", "fd"]],
                   pd.get_dummies(d["SITE_ID"], prefix="site", drop_first=True)], axis=1).astype(float)
    y = d[FC_COL].values.astype(float)
    m = sm.OLS(y, X).fit()
    idx = list(X.columns).index("dx_num")
    return {
        "stratum": label, "N": len(d),
        "ASD": int((d["DX"]=="ASD").sum()),
        "TD": int((d["DX"]=="TD").sum()),
        "beta": round(m.params.iloc[idx], 5),
        "CI_lower": round(m.conf_int().iloc[idx, 0], 5),
        "CI_upper": round(m.conf_int().iloc[idx, 1], 5),
        "t": round(m.tvalues.iloc[idx], 4),
        "p": round(m.pvalues.iloc[idx], 5),
    }

print(f"\n{'='*55}")
print("OLS Results by FD Stratum within ABIDE II")
print(f"{'='*55}")

strata_results = []

# Full ABIDE II
X_full = pd.concat([abide2[["intercept", "dx_num", "age", "fd"]],
                    pd.get_dummies(abide2["SITE_ID"], prefix="site", drop_first=True)], axis=1).astype(float)
y_full = abide2[FC_COL].values.astype(float)
m_full = sm.OLS(y_full, X_full).fit()
idx_f = list(X_full.columns).index("dx_num")
strata_results.append({
    "stratum": "ABIDE II (all)", "N": len(abide2),
    "ASD": int((abide2["DX"]=="ASD").sum()), "TD": int((abide2["DX"]=="TD").sum()),
    "beta": round(m_full.params.iloc[idx_f], 5),
    "CI_lower": round(m_full.conf_int().iloc[idx_f, 0], 5),
    "CI_upper": round(m_full.conf_int().iloc[idx_f, 1], 5),
    "t": round(m_full.tvalues.iloc[idx_f], 4),
    "p": round(m_full.pvalues.iloc[idx_f], 5),
})
print(f"  ABIDE II (all):  N={len(abide2):>3d} | β={strata_results[-1]['beta']:>7.4f} [{strata_results[-1]['CI_lower']:>7.4f}, {strata_results[-1]['CI_upper']:>7.4f}] | p={strata_results[-1]['p']:.4f}")

# Real FD stratum
if real_mask.sum() > 5:
    res = run_model(abide2[real_mask], "ABIDE II real FD")
    strata_results.append(res)
    print(f"  Real FD stratum: N={res['N']:>3d} | β={res['beta']:>7.4f} [{res['CI_lower']:>7.4f}, {res['CI_upper']:>7.4f}] | p={res['p']:.4f}")

# Imputed FD stratum
if (~real_mask).sum() > 5:
    res = run_model(abide2[~real_mask], "ABIDE II imputed FD")
    strata_results.append(res)
    print(f"  Imputed stratum: N={res['N']:>3d} | β={res['beta']:>7.4f} [{res['CI_lower']:>7.4f}, {res['CI_upper']:>7.4f}] | p={res['p']:.4f}")

# Real FD with real FD value (not func_mean_fd placeholder)
if real_fd is not None and real_mask.sum() > 5:
    abide2_real = abide2[real_mask].copy()
    abide2_real["fd_real"] = pd.to_numeric(abide2_real["real_mean_fd"], errors="coerce")
    d = abide2_real.dropna(subset=[FC_COL, "dx_num", "age", "fd_real", "SITE_ID"]).reset_index(drop=True)
    d["intercept"] = 1.0
    X_r = pd.concat([d[["intercept", "dx_num", "age", "fd_real"]],
                     pd.get_dummies(d["SITE_ID"], prefix="site", drop_first=True)], axis=1).astype(float)
    y_r = d[FC_COL].values.astype(float)
    m_r = sm.OLS(y_r, X_r).fit()
    idx_r = list(X_r.columns).index("dx_num")
    strata_results.append({
        "stratum": "ABIDE II real FD (using real FD value)", "N": len(d),
        "ASD": int((d["DX"]=="ASD").sum()), "TD": int((d["DX"]=="TD").sum()),
        "beta": round(m_r.params.iloc[idx_r], 5),
        "CI_lower": round(m_r.conf_int().iloc[idx_r, 0], 5),
        "CI_upper": round(m_r.conf_int().iloc[idx_r, 1], 5),
        "t": round(m_r.tvalues.iloc[idx_r], 4),
        "p": round(m_r.pvalues.iloc[idx_r], 5),
    })
    print(f"  Real FD (true FD): N={strata_results[-1]['N']:>3d} | β={strata_results[-1]['beta']:>7.4f} [{strata_results[-1]['CI_lower']:>7.4f}, {strata_results[-1]['CI_upper']:>7.4f}] | p={strata_results[-1]['p']:.4f}")

# ── Save ──
rdf = pd.DataFrame(strata_results)
rdf.to_csv(os.path.join(OUT_DIR, "abide2_fd_stratification.csv"), index=False)
print(f"\nSaved to abide2_fd_stratification.csv")
print("Done.")
