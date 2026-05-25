"""
Step 24: ADOS subscale analysis.
Test PCC-mPFC FC against ADOS subscales (Social Affect, RRB, Communication, Social)
using within-version z-standardization, separately for each subscale.
"""
import pandas as pd, numpy as np, os, warnings
from scipy.stats import pearsonr, t as t_dist
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
PHENO_ABIDE1 = os.path.join(BASE, "Phenotypic_V1_0b_preprocessed1.csv")
PHENO_DIR = os.path.join(ABIDE_DIR, "ABIDE_II")
OUT_DIR = os.path.join(ABIDE_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

FC_COL = "PCC_mPFC_z"

# ── Load FC + demographics ──
fc = pd.read_csv(FC_CSV, dtype={"SUB_ID": str})
demo = pd.read_csv(SUB_CSV, dtype={"SUB_ID": str})[["SUB_ID", "AGE_AT_SCAN", "func_mean_fd", "SITE_ID", "DX", "dataset"]]
fc = fc.merge(demo, on=["SUB_ID", "SITE_ID", "DX", "dataset"], how="left")
fc = fc[fc["DX"] == "ASD"].copy()
print(f"ASD subjects with FC: {len(fc)}")

# ── ABIDE I ADOS (from PCP phenotype) ──
p1 = pd.read_csv(PHENO_ABIDE1, dtype={"SUB_ID": str})
p1 = p1[p1["SUB_ID"].notna()].copy()
p1["SUB_ID"] = p1["SUB_ID"].astype(str).str.strip()

# Gotham subscales (ABIDE I)
for col in ["ADOS_GOTHAM_SOCAFFECT", "ADOS_GOTHAM_RRB", "ADOS_GOTHAM_TOTAL",
            "ADOS_COMM", "ADOS_SOCIAL", "ADOS_STEREO_BEHAV", "ADOS_TOTAL"]:
    p1[col] = pd.to_numeric(p1[col], errors="coerce")
    if col in ["ADOS_GOTHAM_SOCAFFECT", "ADOS_GOTHAM_RRB", "ADOS_COMM",
               "ADOS_SOCIAL", "ADOS_STEREO_BEHAV", "ADOS_TOTAL", "ADOS_GOTHAM_TOTAL"]:
        p1[col] = p1[col].where(p1[col] != -9999)

# Map ABIDE I: version classification
p1["ados_version"] = "Gotham"  # ABIDE I PCP uses Gotham algorithm
# Classic if has Classic scores but no Gotham
classic_mask = (p1["ADOS_TOTAL"].notna()) & (p1["ADOS_GOTHAM_TOTAL"].isna())
p1.loc[classic_mask, "ados_version"] = "Classic"

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
        continue
    df.columns = [c.strip() for c in df.columns]
    df["SUB_ID"] = df["participant_id"].astype(str).str.strip()
    # Gotham subscales
    df["ados_g_comm"] = pd.to_numeric(df.get("ados_g_comm"), errors="coerce") if "ados_g_comm" in df.columns else np.nan
    df["ados_g_social"] = pd.to_numeric(df.get("ados_g_social"), errors="coerce") if "ados_g_social" in df.columns else np.nan
    df["ados_g_stereo"] = pd.to_numeric(df.get("ados_g_stereo_behav"), errors="coerce") if "ados_g_stereo_behav" in df.columns else np.nan
    df["ados_g_total"] = pd.to_numeric(df.get("ados_g_total"), errors="coerce") if "ados_g_total" in df.columns else np.nan
    # ADOS-2 subscales
    df["ados2_sa"] = pd.to_numeric(df.get("ados_2_socaffect"), errors="coerce") if "ados_2_socaffect" in df.columns else np.nan
    df["ados2_rrb"] = pd.to_numeric(df.get("ados_2_rrb"), errors="coerce") if "ados_2_rrb" in df.columns else np.nan
    df["ados2_total"] = pd.to_numeric(df.get("ados_2_total"), errors="coerce") if "ados_2_total" in df.columns else np.nan

    keep_cols = ["SUB_ID", "ados_g_comm", "ados_g_social", "ados_g_stereo", "ados_g_total",
                 "ados2_sa", "ados2_rrb", "ados2_total"]
    p2_rows.append(df[[c for c in keep_cols if c in df.columns]])

p2 = pd.concat(p2_rows, ignore_index=True) if p2_rows else pd.DataFrame()

# ── Merge ADOS into FC data ──
fc["SUB_ID"] = fc["SUB_ID"].astype(str).str.strip()

fc_ab1 = fc[fc["dataset"] == "ABIDE_I"].copy()
fc_ab1 = fc_ab1.merge(p1[["SUB_ID", "ADOS_GOTHAM_SOCAFFECT", "ADOS_GOTHAM_RRB",
                           "ADOS_COMM", "ADOS_SOCIAL", "ADOS_STEREO_BEHAV",
                           "ADOS_TOTAL", "ados_version"]],
                       on="SUB_ID", how="left")

fc_ab2 = fc[fc["dataset"] == "ABIDE_II"].copy()
fc_ab2 = fc_ab2.merge(p2[["SUB_ID", "ados_g_comm", "ados_g_social", "ados_g_stereo",
                           "ados_g_total", "ados2_sa", "ados2_rrb", "ados2_total"]],
                       on="SUB_ID", how="left")

com = pd.concat([fc_ab1, fc_ab2], ignore_index=True)

# ── Define subscale hierarchy ──
def get_subscale(row, scale_name):
    """Return the appropriate subscale value based on available data and version."""
    dataset = row.get("dataset", "")
    # Social Affect: Gotham SOCAFFECT or ADOS-2 socaffect
    if scale_name == "social_affect":
        if pd.notna(row.get("ADOS_GOTHAM_SOCAFFECT")):
            return "Gotham", row["ADOS_GOTHAM_SOCAFFECT"]
        if pd.notna(row.get("ados2_sa")):
            return "ADOS-2", row["ados2_sa"]
        return None, np.nan

    # RRB: Gotham RRB or ADOS-2 rrb
    if scale_name == "rrb":
        if pd.notna(row.get("ADOS_GOTHAM_RRB")):
            return "Gotham", row["ADOS_GOTHAM_RRB"]
        if pd.notna(row.get("ados2_rrb")):
            return "ADOS-2", row["ados2_rrb"]
        return None, np.nan

    # Communication (Gotham only)
    if scale_name == "communication":
        if pd.notna(row.get("ADOS_COMM")):
            return "Gotham", row["ADOS_COMM"]
        if pd.notna(row.get("ados_g_comm")):
            return "Gotham", row["ados_g_comm"]
        return None, np.nan

    # Social (Gotham only)
    if scale_name == "social":
        if pd.notna(row.get("ADOS_SOCIAL")):
            return "Gotham", row["ADOS_SOCIAL"]
        if pd.notna(row.get("ados_g_social")):
            return "Gotham", row["ados_g_social"]
        return None, np.nan

    # Stereotyped behaviors (Gotham only)
    if scale_name == "stereo_behav":
        if pd.notna(row.get("ADOS_STEREO_BEHAV")):
            return "Gotham", row["ADOS_STEREO_BEHAV"]
        if pd.notna(row.get("ados_g_stereo")):
            return "Gotham", row["ados_g_stereo"]
        return None, np.nan

    return None, np.nan

import statsmodels.api as sm

def run_subscale_analysis(com, scale_name, display_name):
    """Run OLS: FC ~ ADOS_subscale_z + age + FD + site for a given ADOS subscale."""
    versions, scores = zip(*[get_subscale(row, scale_name) for _, row in com.iterrows()])
    com[f"{scale_name}_version"] = versions
    com[f"{scale_name}_score"] = scores

    n_avail = com[f"{scale_name}_score"].notna().sum()
    if n_avail < 20:
        print(f"\n  {display_name}: insufficient data (N={n_avail}), skipping")
        return None

    # Within-version z-score
    com[f"{scale_name}_z"] = np.nan
    for ver in com[f"{scale_name}_version"].dropna().unique():
        mask = com[f"{scale_name}_version"] == ver
        m = com.loc[mask, f"{scale_name}_score"].mean()
        s = com.loc[mask, f"{scale_name}_score"].std()
        com.loc[mask, f"{scale_name}_z"] = (com.loc[mask, f"{scale_name}_score"] - m) / s

    # Regression
    keep = ["SUB_ID", "SITE_ID", FC_COL, f"{scale_name}_z",
            "AGE_AT_SCAN", "func_mean_fd"]
    reg = com[keep].dropna(subset=[FC_COL, f"{scale_name}_z", "AGE_AT_SCAN",
                                    "func_mean_fd", "SITE_ID"]).copy()
    reg["age"] = pd.to_numeric(reg["AGE_AT_SCAN"], errors="coerce")
    reg["fd"] = pd.to_numeric(reg["func_mean_fd"], errors="coerce")
    reg = reg.dropna(subset=["age", "fd"]).reset_index(drop=True)

    if len(reg) < 20:
        print(f"  {display_name}: insufficient after listwise deletion (N={len(reg)}), skipping")
        return None

    reg["intercept"] = 1.0
    site_dummies = pd.get_dummies(reg["SITE_ID"], prefix="site", drop_first=True)
    X = pd.concat([reg[["intercept", f"{scale_name}_z", "age", "fd"]], site_dummies], axis=1).astype(float)
    y = reg[FC_COL].values.astype(float)

    model = sm.OLS(y, X).fit()
    idx = list(X.columns).index(f"{scale_name}_z")
    coef = model.params.iloc[idx]
    ci = model.conf_int().iloc[idx].values
    t_val = model.tvalues.iloc[idx]
    p_val = model.pvalues.iloc[idx]
    partial_r = t_val / np.sqrt(t_val ** 2 + model.df_resid)

    version_counts = com[f"{scale_name}_version"].value_counts().to_dict()

    result = {
        "analysis": f"ADOS subscale: {display_name}",
        "N": len(reg), "beta": round(coef, 5),
        "CI_lower": round(ci[0], 5), "CI_upper": round(ci[1], 5),
        "t": round(t_val, 4), "p": round(p_val, 5),
        "partial_r": round(partial_r, 5),
        "covariates": "age + FD + site", "group": "ASD only",
        "version_breakdown": str(version_counts),
    }

    print(f"\n  {display_name} (N={len(reg)}): β={coef:.4f}, CI=[{ci[0]:.4f}, {ci[1]:.4f}], "
          f"t={t_val:.3f}, p={p_val:.4f}, r={partial_r:.4f}")
    print(f"    Versions: {version_counts}")

    return result

# ── Run all subscale analyses ──
print(f"\n{'='*55}")
print("ADOS Subscale Analyses")
print(f"{'='*55}")

subscales = [
    ("social_affect", "Social Affect"),
    ("rrb", "RRB"),
    ("communication", "Communication"),
    ("social", "Social"),
    ("stereo_behav", "Stereotyped Behaviors"),
]

results = []
for scale_name, display_name in subscales:
    result = run_subscale_analysis(com.copy(), scale_name, display_name)
    if result is not None:
        results.append(result)

# ── Save ──
if results:
    rdf = pd.DataFrame(results)
    rdf.to_csv(os.path.join(OUT_DIR, "ados_subscales.csv"), index=False)
    print(f"\n{'='*55}")
    print("Summary:")
    print(f"{'='*55}")
    for _, r in rdf.iterrows():
        sig = " *" if r["p"] < 0.05 else ""
        print(f"  {r['analysis']:<30s} N={r['N']:>3d} | β={r['beta']:>7.4f} [{r['CI_lower']:>7.4f}, {r['CI_upper']:>7.4f}] | p={r['p']:.4f}{sig}")
    print(f"\nSaved to ados_subscales.csv")
else:
    print("No subscale results generated.")

print("Done.")
