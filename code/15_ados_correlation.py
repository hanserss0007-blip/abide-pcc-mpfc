"""
Step 15: ADOS clinical correlation analysis.
Harmonize ADOS across versions (Gotham/Classic/ADOS-2) via within-version z-score,
then regress PCC-mPFC FC ~ ADOS_z + age + FD + site in ASD subjects.
"""
import pandas as pd, numpy as np, os, warnings
from scipy.stats import pearsonr, t as t_dist
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUB_CSV = os.path.join(BASE, "code", "combined_subjects.csv")

# Build ABIDE project path dynamic (avoids Chinese char issues in some envs)
def find_abide_dir():
    for f in os.listdir("E:/"):
        if f.startswith("ABIDE"):
            return os.path.join("E:/", f)
    raise FileNotFoundError("ABIDE dir not found on E:")

ABIDE_DIR = find_abide_dir()
FC_CSV = os.path.join(ABIDE_DIR, "timeseries", "seed_connectivity.csv")
PHENO_ABIDE1 = os.path.join(BASE, "Phenotypic_V1_0b_preprocessed1.csv")
PHENO_DIR = os.path.join(ABIDE_DIR, "ABIDE_II")
OUT_DIR = os.path.join(ABIDE_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

FC_COL = "PCC_mPFC_z"
DX_COL = "DX"
SITE_COL = "SITE_ID"

# ── Load FC + demographics ──
fc = pd.read_csv(FC_CSV, dtype={"SUB_ID": str})
demo = pd.read_csv(SUB_CSV, dtype={"SUB_ID": str})[["SUB_ID", "AGE_AT_SCAN", "func_mean_fd", "SITE_ID", "DX"]]
fc = fc.merge(demo, on=["SUB_ID", "SITE_ID", "DX"], how="left")
fc = fc[fc[DX_COL] == "ASD"].copy()
print(f"ASD subjects with FC: {len(fc)}")

# ── ABIDE I ADOS (from PCP phenotype) ──
p1 = pd.read_csv(PHENO_ABIDE1, dtype={"SUB_ID": str})
p1 = p1[p1["SUB_ID"].notna()].copy()
p1["SUB_ID"] = p1["SUB_ID"].astype(str).str.strip()
p1["ados_gotham"] = pd.to_numeric(p1["ADOS_GOTHAM_TOTAL"], errors="coerce")
p1["ados_classic"] = pd.to_numeric(p1["ADOS_TOTAL"], errors="coerce")
p1["ados_severity"] = pd.to_numeric(p1["ADOS_GOTHAM_SEVERITY"], errors="coerce")
# -9999 as NA
p1["ados_gotham"] = p1["ados_gotham"].where(p1["ados_gotham"] != -9999)
p1["ados_classic"] = p1["ados_classic"].where(p1["ados_classic"] != -9999)
p1["ados_severity"] = p1["ados_severity"].where(p1["ados_severity"] != -9999)

# ── ABIDE II ADOS (from per-site phenotype files) ──
p2_rows = []
encodings = ["utf-8", "latin1", "cp1252"]
for fname in os.listdir(PHENO_DIR):
    if not fname.startswith("pheno_") or not fname.endswith(".tsv"):
        continue
    fpath = os.path.join(PHENO_DIR, fname)
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(fpath, sep="\t", encoding=enc, dtype={"participant_id": str})
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    if df is None:
        print(f"  WARN: cannot read {fname}, skipping")
        continue
    df.columns = [c.strip() for c in df.columns]
    # SUB_ID: participant_id matches combined_subjects numeric ABIDE II ID
    df["SUB_ID"] = df["participant_id"].astype(str).str.strip()
    df["ados_gotham"] = pd.to_numeric(df.get("ados_g_total"), errors="coerce") if "ados_g_total" in df.columns else np.nan
    df["ados_2_total"] = pd.to_numeric(df.get("ados_2_total"), errors="coerce") if "ados_2_total" in df.columns else np.nan
    p2_rows.append(df[["SUB_ID", "ados_gotham", "ados_2_total"]])

p2 = pd.concat(p2_rows, ignore_index=True) if p2_rows else pd.DataFrame()

# ── Merge ADOS into FC data ──
fc["SUB_ID"] = fc["SUB_ID"].astype(str).str.strip()

# ABIDE I
fc_ab1 = fc[fc["dataset"] == "ABIDE_I"].copy()
fc_ab1 = fc_ab1.merge(p1[["SUB_ID", "ados_gotham", "ados_classic", "ados_severity"]],
                       on="SUB_ID", how="left")

# ABIDE II
fc_ab2 = fc[fc["dataset"] == "ABIDE_II"].copy()
fc_ab2 = fc_ab2.merge(p2[["SUB_ID", "ados_gotham", "ados_2_total"]],
                       on="SUB_ID", how="left")

com = pd.concat([fc_ab1, fc_ab2], ignore_index=True)

# ── Assign ADOS version and create harmonized score ──
def assign_version(row):
    if pd.notna(row["ados_gotham"]):
        return "Gotham", row["ados_gotham"]
    elif pd.notna(row["ados_classic"]):
        return "Classic", row["ados_classic"]
    elif pd.notna(row["ados_2_total"]):
        return "ADOS-2", row["ados_2_total"]
    return None, np.nan

versions, scores = zip(*[assign_version(r) for _, r in com.iterrows()])
com["ados_version"] = versions
com["ados_score"] = scores

n_with_ados = com["ados_score"].notna().sum()
print(f"ASD with ADOS: {n_with_ados}/{len(com)} ({100*n_with_ados/len(com):.0f}%)")
print(f"  Version breakdown: {com['ados_version'].value_counts().to_dict()}")

# ── Within-version z-score ──
com["ados_z"] = np.nan
for ver in com["ados_version"].dropna().unique():
    mask = com["ados_version"] == ver
    m = com.loc[mask, "ados_score"].mean()
    s = com.loc[mask, "ados_score"].std()
    com.loc[mask, "ados_z"] = (com.loc[mask, "ados_score"] - m) / s
    print(f"  {ver}: mean={m:.2f}, SD={s:.2f}, N={mask.sum()}")

# ── Prepare regression data ──
keep = ["SUB_ID", "subj_id", SITE_COL, FC_COL, "ados_z", "ados_score",
        "ados_version", "AGE_AT_SCAN", "func_mean_fd", "dataset"]
reg = com[keep].dropna(subset=[FC_COL, "ados_z", "AGE_AT_SCAN", "func_mean_fd", SITE_COL]).copy()
reg["age"] = pd.to_numeric(reg["AGE_AT_SCAN"], errors="coerce")
reg["fd"] = pd.to_numeric(reg["func_mean_fd"], errors="coerce")
reg = reg.dropna(subset=["age", "fd"]).reset_index(drop=True)
print(f"\nFinal regression N: {len(reg)}")

# ── OLS ──
import statsmodels.api as sm
reg["intercept"] = 1.0
site_dummies = pd.get_dummies(reg[SITE_COL], prefix="site", drop_first=True)
X = pd.concat([reg[["intercept", "ados_z", "age", "fd"]], site_dummies], axis=1).astype(float)
y = reg[FC_COL].values.astype(float)

model = sm.OLS(y, X).fit()
print(f"\n{'='*55}")
print("ADOS Clinical Correlation: PCC-mPFC FC ~ ADOS_z + age + FD + site")
print(f"{'='*55}")
print(model.summary())

# Partial r
ados_idx = list(X.columns).index("ados_z")
t_val = model.tvalues.iloc[ados_idx]
coef = model.params.iloc[ados_idx]
ci_95 = model.conf_int().loc["ados_z"].values
p_val = model.pvalues.iloc[ados_idx]
partial_r = t_val / np.sqrt(t_val**2 + model.df_resid)

print(f"\n  ADOS_z beta: {coef:.4f}")
print(f"  95% CI: [{ci_95[0]:.4f}, {ci_95[1]:.4f}]")
print(f"  t({model.df_resid:.0f}) = {t_val:.3f}, p = {p_val:.4f}")
print(f"  Partial r = {partial_r:.4f}")

# ── Gotham-only sensitivity ──
gotham = reg[reg["ados_version"] == "Gotham"].copy()
print(f"\n{'='*55}")
print(f"Gotham-Only Sensitivity (N={len(gotham)})")
print(f"{'='*55}")
Xg = pd.concat([gotham[["intercept", "ados_z", "age", "fd"]],
                pd.get_dummies(gotham[SITE_COL], prefix="site", drop_first=True)], axis=1).astype(float)
# Ensure all columns from full model present
for c in X.columns:
    if c not in Xg.columns:
        Xg[c] = 0.0
Xg = Xg[X.columns]
yg = gotham[FC_COL].values.astype(float)

model_g = sm.OLS(yg, Xg).fit()
t_g = model_g.tvalues.iloc[ados_idx]
coef_g = model_g.params.iloc[ados_idx]
ci_g = model_g.conf_int().iloc[ados_idx].values
p_g = model_g.pvalues.iloc[ados_idx]
pr_g = t_g / np.sqrt(t_g**2 + model_g.df_resid)
print(f"  ADOS_z beta: {coef_g:.4f}, CI [{ci_g[0]:.4f}, {ci_g[1]:.4f}], t={t_g:.3f}, p={p_g:.4f}, r={pr_g:.4f}")

# ── Save results ──
results = pd.DataFrame([{
    "analysis": "ADOS correlation (full, within-version z-score)",
    "N": len(reg), "beta": round(coef, 5),
    "CI_lower": round(ci_95[0], 5), "CI_upper": round(ci_95[1], 5),
    "t": round(t_val, 4), "p": round(p_val, 5), "partial_r": round(partial_r, 5),
    "covariates": "age + FD + site", "group": "ASD only",
    "ados_versions": str(com["ados_version"].value_counts().to_dict()),
}])
results = pd.concat([results, pd.DataFrame([{
    "analysis": "ADOS correlation (Gotham only)",
    "N": len(gotham), "beta": round(coef_g, 5),
    "CI_lower": round(ci_g[0], 5), "CI_upper": round(ci_g[1], 5),
    "t": round(t_g, 4), "p": round(p_g, 5), "partial_r": round(pr_g, 5),
    "covariates": "age + FD + site", "group": "ASD only",
    "ados_versions": "Gotham only",
}])], ignore_index=True)
results.to_csv(os.path.join(OUT_DIR, "ados_correlation.csv"), index=False)
print(f"\nResults saved to ados_correlation.csv")

# ── Partial regression scatter plot ──
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Partial residuals
cov_cols = [c for c in X.columns if c != "ados_z"]
model_cov = sm.OLS(y, X[cov_cols].astype(float)).fit()
fc_resid = y - model_cov.predict(X[cov_cols].astype(float))

ados_cov = X[["intercept"] + list(site_dummies.columns) + ["age", "fd"]]
model_ados = sm.OLS(X["ados_z"].values.astype(float), ados_cov.astype(float)).fit()
ados_resid = X["ados_z"].values - model_ados.predict(ados_cov.astype(float))

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(ados_resid, fc_resid, alpha=0.5, s=25, c="steelblue", edgecolors="none")
b, a = np.polyfit(ados_resid, fc_resid, 1)
xx = np.linspace(ados_resid.min(), ados_resid.max(), 100)
ax.plot(xx, a * xx + b, "r-", lw=1.5)
ax.set_xlabel("ADOS (partial, z-score units)", fontsize=11)
ax.set_ylabel("PCC-mPFC FC (partial, Fisher Z)", fontsize=11)
ax.set_title(f"ADOS-Symptom Correlation\npartial r = {partial_r:.3f}, p = {p_val:.4f}, N = {len(reg)}",
             fontsize=10, fontweight="bold")
ax.axhline(0, color="gray", ls="--", lw=0.5)
ax.axvline(0, color="gray", ls="--", lw=0.5)
ax.text(0.05, 0.95, f"beta = {coef:.4f}\n95% CI [{ci_95[0]:.4f}, {ci_95[1]:.4f}]",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.6))
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_ados_correlation.png"), dpi=300)
print(f"Plot saved to fig_ados_correlation.png")
print("Done.")
