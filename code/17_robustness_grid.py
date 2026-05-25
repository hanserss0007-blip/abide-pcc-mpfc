"""
Step 17: Robustness grid — ASD<TD effect size under different analysis choices.
Produces a summary table for Supplementary Materials.
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
QC_CSV = os.path.join(ABIDE_DIR, "results", "coverage_qc.csv")
OUT_DIR = os.path.join(ABIDE_DIR, "results")

# ── Load FC + demographics ──
fc = pd.read_csv(FC_CSV, dtype={"SUB_ID": str})
demo = pd.read_csv(SUB_CSV, dtype={"SUB_ID": str})
df = fc.merge(demo, on=["SUB_ID", "SITE_ID", "DX", "dataset"], how="left", suffixes=("", "_y"))
df = df.dropna(subset=["AGE_AT_SCAN", "func_mean_fd", "SITE_ID"])
df["age"] = pd.to_numeric(df["AGE_AT_SCAN"], errors="coerce")
df["fd"] = pd.to_numeric(df["func_mean_fd"], errors="coerce")
df = df.dropna(subset=["age", "fd"]).reset_index(drop=True)
df["dx_num"] = (df["DX"] == "ASD").astype(int)

# QC
qc = pd.read_csv(QC_CSV, dtype={"SUB_ID": str}) if os.path.exists(QC_CSV) else None

import statsmodels.api as sm

def cohens_d_from_t(t_val, n):
    """Convert t-statistic to approx Cohen's d for two-group comparison."""
    return 2 * t_val / np.sqrt(n)

def run_model(data, fc_col, label, covariates, sample_desc="full sample"):
    """Run FC ~ DX + covariates + site, return DX effect summary."""
    data = data.dropna(subset=[fc_col, "dx_num"] + [c for c in covariates if c != "site"]).copy()
    data["intercept"] = 1.0
    X = [data[["intercept", "dx_num"] + [c for c in covariates if c != "site"]],
         pd.get_dummies(data["SITE_ID"], prefix="site", drop_first=True)]
    X = pd.concat(X, axis=1).astype(float)
    y = data[fc_col].values.astype(float)
    m = sm.OLS(y, X).fit()
    idx = list(X.columns).index("dx_num")
    beta = m.params.iloc[idx]
    ci = m.conf_int().iloc[idx].values
    t = m.tvalues.iloc[idx]
    p = m.pvalues.iloc[idx]
    n_asd = (data["DX"] == "ASD").sum()
    n_td = (data["DX"] == "TD").sum()
    return {
        "model": label, "sample": sample_desc,
        "N": len(data), "ASD": n_asd, "TD": n_td,
        "beta": round(beta, 4), "CI_lower": round(ci[0], 4), "CI_upper": round(ci[1], 4),
        "t": round(t, 3), "p": round(p, 5),
    }

rows = []
fc_pairs = [("PCC_mPFC_z", "PCC-mPFC"), ("PCC_zAI_z", "PCC-rAI"), ("rAI_mPFC_z", "rAI-mPFC")]

for fc_col, fc_name in fc_pairs:
    if fc_col not in df.columns:
        continue
    print(f"\n--- {fc_name} ---")

    # 1. Primary (full covariates + site)
    rows.append(run_model(df, fc_col, f"{fc_name} primary",
                          ["dx_num", "age", "fd", "site"]))

    # 2. No FD
    rows.append(run_model(df, fc_col, f"{fc_name} no FD",
                          ["dx_num", "age", "site"]))

    # 3. No age
    rows.append(run_model(df, fc_col, f"{fc_name} no age",
                          ["dx_num", "fd", "site"]))

    # 4. No age/FD (site only)
    rows.append(run_model(df, fc_col, f"{fc_name} site only",
                          ["dx_num", "site"]))

    # 5. M1 sample (coverage >= 80%)
    if qc is not None:
        m1_ids = qc[qc["included_M1"]]["subj_id"].astype(str)
        m1 = df[df["subj_id"].astype(str).isin(m1_ids)]
        rows.append(run_model(m1, fc_col, f"{fc_name} M1 (coverage>=80%)",
                              ["dx_num", "age", "fd", "site"]))

    # 6. M3 sample (coverage>=80% + FD<0.2)
    if qc is not None:
        m3_ids = qc[qc["included_M3"]]["subj_id"].astype(str)
        m3 = df[df["subj_id"].astype(str).isin(m3_ids)]
        rows.append(run_model(m3, fc_col, f"{fc_name} M3 (cov>=80%, FD<0.2)",
                              ["dx_num", "age", "fd", "site"]))

    # 7. ABIDE I only
    ab1 = df[df["dataset"] == "ABIDE_I"]
    rows.append(run_model(ab1, fc_col, f"{fc_name} ABIDE I",
                          ["dx_num", "age", "fd", "site"]))

    # 8. ABIDE II only
    ab2 = df[df["dataset"] == "ABIDE_II"]
    rows.append(run_model(ab2, fc_col, f"{fc_name} ABIDE II",
                          ["dx_num", "age", "fd", "site"]))

# ── Compile table ──
rdf = pd.DataFrame(rows)
rdf.to_csv(os.path.join(OUT_DIR, "robustness_grid.csv"), index=False)

print(f"\n{'='*60}")
print("Robustness Grid Summary")
print(f"{'='*60}")
for _, r in rdf.iterrows():
    sig = " *" if r["p"] < 0.05 else ""
    print(f"  {r['model']:<28s} | {r['sample']:<25s} | N={r['N']:>3d} | "
          f"beta={r['beta']:>7.4f} [{r['CI_lower']:>6.4f}, {r['CI_upper']:>6.4f}] | "
          f"p={r['p']:.4f}{sig}")
print(f"\nSaved to robustness_grid.csv")
print("Done.")
