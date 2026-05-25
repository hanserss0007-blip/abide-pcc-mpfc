"""
Step 18: Leave-one-site-out ROI beta (OLS, no whole-brain, no OOM).
For each site, exclude → run FC ~ DX + age + FD + site → collect ASD<TD beta.
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

# Load
fc = pd.read_csv(FC_CSV, dtype={"SUB_ID": str})
demo = pd.read_csv(SUB_CSV, dtype={"SUB_ID": str})[["SUB_ID", "SITE_ID", "DX", "AGE_AT_SCAN", "func_mean_fd", "dataset"]]
df = fc.merge(demo, on=["SUB_ID", "SITE_ID", "DX", "dataset"], how="left")
df["age"] = pd.to_numeric(df["AGE_AT_SCAN"], errors="coerce")
df["fd"] = pd.to_numeric(df["func_mean_fd"], errors="coerce")
df["dx_num"] = (df["DX"] == "ASD").astype(int)
df["intercept"] = 1.0
df = df.dropna(subset=[FC_COL, "dx_num", "age", "fd", "SITE_ID"]).reset_index(drop=True)

import statsmodels.api as sm

def run_loso(data, exclude_site=None):
    if exclude_site:
        sub = data[data["SITE_ID"] != exclude_site]
    else:
        sub = data.copy()
    X = [sub[["intercept", "dx_num", "age", "fd"]],
         pd.get_dummies(sub["SITE_ID"], prefix="site", drop_first=True)]
    X = pd.concat(X, axis=1).astype(float)
    y = sub[FC_COL].values.astype(float)
    m = sm.OLS(y, X).fit()
    idx = list(X.columns).index("dx_num")
    beta = m.params.iloc[idx]
    ci = m.conf_int().iloc[idx].values
    t = m.tvalues.iloc[idx]
    p = m.pvalues.iloc[idx]
    return {"beta": beta, "ci_l": ci[0], "ci_u": ci[1], "t": t, "p": p,
            "n": len(sub), "n_asd": int((sub["DX"]=="ASD").sum()),
            "n_td": int((sub["DX"]=="TD").sum())}

all_sites = sorted(df["SITE_ID"].unique())
results = []

# Full sample
full = run_loso(df, exclude_site=None)
results.append({"Excluded": "None", **full})

# LOSO
for site in all_sites:
    n_site = (df["SITE_ID"] == site).sum()
    if n_site < 10:
        print(f"  Skip {site}: N={n_site} < 10")
        continue
    loso = run_loso(df, exclude_site=site)
    results.append({"Excluded": site, **loso})
    sig = " *" if loso["p"] < 0.05 else ""
    print(f"  Exclude {site}: beta={loso['beta']:.4f} [{loso['ci_l']:.4f}, {loso['ci_u']:.4f}], p={loso['p']:.4f}{sig}")

rdf = pd.DataFrame(results)
rdf.to_csv(os.path.join(OUT_DIR, "loso_roi_ols.csv"), index=False)

# Report
full_beta = rdf[rdf["Excluded"] == "None"]["beta"].values[0]
loso_betas = rdf[rdf["Excluded"] != "None"]["beta"].values
print(f"\n{'='*50}")
print("LOSO ROI Beta Stability (OLS)")
print(f"{'='*50}")
print(f"  Full sample beta: {full_beta:.4f}")
print(f"  LOSO mean: {loso_betas.mean():.4f}, SD: {loso_betas.std():.4f}")
print(f"  Range: [{loso_betas.min():.4f}, {loso_betas.max():.4f}]")
n_same_sign = ((loso_betas < 0) == (full_beta < 0)).sum()
print(f"  Same sign as full: {n_same_sign}/{len(loso_betas)}")
n_neg = sum(1 for b in loso_betas if b < 0)
print(f"  Negative (ASD<TD) betas: {n_neg}/{len(loso_betas)}")

# Plot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

loso_df = rdf[rdf["Excluded"] != "None"].copy()
fig, ax = plt.subplots(figsize=(9, 5))
sites_l = loso_df["Excluded"].tolist()
beta_l = loso_df["beta"].tolist()
ci_l = loso_df["ci_l"].tolist()
ci_u = loso_df["ci_u"].tolist()
colors = ["steelblue" if b < 0 else "salmon" for b in beta_l]

y_pos = range(len(sites_l))
for i in y_pos:
    ax.plot([ci_l[i], ci_u[i]], [i, i], color=colors[i], lw=1.5, alpha=0.6)
    ax.scatter(beta_l[i], i, color=colors[i], s=40, zorder=5)

ax.axvline(0, color="gray", ls="--", lw=0.8)
ax.axvline(full_beta, color="crimson", ls=":", lw=1.5, label=f"Full ({full_beta:.4f})")
ax.set_yticks(range(len(sites_l)))
ax.set_yticklabels(sites_l, fontsize=8)
ax.set_xlabel("ASD<TD beta (PCC-mPFC FC, Fisher Z)", fontsize=11)
ax.set_title("Leave-One-Site-Out Beta Stability", fontweight="bold")
ax.legend(fontsize=9)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_loso_ols.png"), dpi=300)
print(f"Plot saved.")
print("Done.")
