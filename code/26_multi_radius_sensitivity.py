"""
Step 26: Multi-radius seed sensitivity.
Re-extract PCC-mPFC FC at 4mm and 8mm radii, then re-run primary OLS.
Preserves existing 6mm data as reference.
"""
import pandas as pd, numpy as np, os, gc, warnings
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUB_CSV = os.path.join(BASE, "code", "combined_subjects.csv")

def find_abide_dir():
    for f in os.listdir("E:/"):
        if f.startswith("ABIDE"):
            return os.path.join("E:/", f)
    raise FileNotFoundError("ABIDE dir not found on E:")

ABIDE_DIR = find_abide_dir()
OUT_DIR = os.path.join(ABIDE_DIR, "results")
RADII_OUT = os.path.join(OUT_DIR, "multi_radius_fc.csv")
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = {"PCC": [0, -52, 26], "mPFC": [0, 54, -2]}
RADII = [4, 8]  # 4mm and 8mm (6mm already available)

df = pd.read_csv(SUB_CSV, dtype={"SUB_ID": str})
print(f"Total subjects: {len(df)}", flush=True)

# Load existing 6mm FC for comparison
FC_CSV = os.path.join(ABIDE_DIR, "timeseries", "seed_connectivity.csv")
fc6 = pd.read_csv(FC_CSV, dtype={"SUB_ID": str}) if os.path.exists(FC_CSV) else None
if fc6 is not None:
    print(f"Existing 6mm FC: {len(fc6)} subjects")

# ── Check which subjects already processed ──
done = {}
if os.path.exists(RADII_OUT):
    existing = pd.read_csv(RADII_OUT, dtype={"SUB_ID": str})
    done = set(existing["SUB_ID"].astype(str))
    print(f"Already processed: {len(done)}")

results = []
n_total = len(df)

from nilearn.maskers import NiftiSpheresMasker

for i, (_, row) in enumerate(df.iterrows()):
    sub = str(row["SUB_ID"])
    if sub in done:
        continue

    nii = row["func_path"]
    msk = row["mask_path"] if isinstance(row["mask_path"], str) else ""
    if not os.path.exists(nii) or not os.path.exists(msk):
        continue

    try:
        for radius in RADII:
            masker = NiftiSpheresMasker(
                seeds=list(SEEDS.values()), radius=radius,
                mask_img=msk, detrend=True,
                standardize=False, verbose=0,
            )
            ts = masker.fit_transform(nii)
            pcc = ts[:, 0]; mpfc = ts[:, 1]

            r_val = float(np.corrcoef(pcc, mpfc)[0, 1])
            r_val = np.clip(r_val, -0.999, 0.999)
            z_val = float(np.arctanh(r_val))

            results.append({
                "SUB_ID": sub, "subj_id": row["subj_id"],
                "SITE_ID": row["SITE_ID"], "DX": row["DX"],
                "dataset": row["dataset"],
                "radius": radius,
                "PCC_mPFC_r": r_val,
                "PCC_mPFC_z": z_val,
            })
    except Exception as e:
        print(f"  [ERR] {row['subj_id']}: {e}")
        continue

    # Write every 100 subjects
    if len(results) >= 100:
        new_df = pd.DataFrame(results)
        if os.path.exists(RADII_OUT):
            old = pd.read_csv(RADII_OUT, dtype={"SUB_ID": str})
            new_df = pd.concat([old, new_df], ignore_index=True)
        new_df.to_csv(RADII_OUT, index=False)
        print(f"  Saved {len(new_df)}/{n_total} ({len(results)} in batch)", flush=True)
        results = []
        gc.collect()

# Final write
if results:
    new_df = pd.DataFrame(results)
    if os.path.exists(RADII_OUT):
        old = pd.read_csv(RADII_OUT, dtype={"SUB_ID": str})
        new_df = pd.concat([old, new_df], ignore_index=True)
    new_df.to_csv(RADII_OUT, index=False)

# ── Analysis ──
import statsmodels.api as sm

print(f"\n{'='*55}", flush=True)
print("Multi-Radius Seed Sensitivity Analysis", flush=True)
print(f"{'='*55}", flush=True)

multi = pd.read_csv(RADII_OUT, dtype={"SUB_ID": str})
demo = pd.read_csv(SUB_CSV, dtype={"SUB_ID": str})[["SUB_ID", "AGE_AT_SCAN", "func_mean_fd", "SITE_ID", "DX"]]
multi = multi.merge(demo, on=["SUB_ID", "SITE_ID", "DX"], how="left")
multi["age"] = pd.to_numeric(multi["AGE_AT_SCAN"], errors="coerce")
multi["fd"] = pd.to_numeric(multi["func_mean_fd"], errors="coerce")
multi["dx_num"] = (multi["DX"] == "ASD").astype(int)
multi["intercept"] = 1.0

radius_results = []
for radius in sorted(multi["radius"].unique()):
    d = multi[multi["radius"] == radius].dropna(
        subset=["PCC_mPFC_z", "dx_num", "age", "fd", "SITE_ID"]).reset_index(drop=True)
    X = pd.concat([d[["intercept", "dx_num", "age", "fd"]],
                   pd.get_dummies(d["SITE_ID"], prefix="site", drop_first=True)], axis=1).astype(float)
    y = d["PCC_mPFC_z"].values.astype(float)
    m = sm.OLS(y, X).fit()
    idx = list(X.columns).index("dx_num")
    beta = m.params.iloc[idx]
    ci = m.conf_int().iloc[idx].values
    t = m.tvalues.iloc[idx]
    p = m.pvalues.iloc[idx]
    n_asd = (d["DX"] == "ASD").sum()
    n_td = (d["DX"] == "TD").sum()
    radius_results.append({
        "radius": f"{radius}mm", "N": len(d), "ASD": n_asd, "TD": n_td,
        "beta": round(beta, 5), "CI_lower": round(ci[0], 5), "CI_upper": round(ci[1], 5),
        "t": round(t, 4), "p": round(p, 5),
    })
    sig = " *" if p < 0.05 else ""
    print(f"  {radius}mm radius: N={len(d):>3d}, β={beta:.4f} [{ci[0]:.4f}, {ci[1]:.4f}], t={t:.3f}, p={p:.4f}{sig}")

# 6mm reference from existing data
if fc6 is not None:
    ref = fc6.merge(demo, on=["SUB_ID", "SITE_ID", "DX"], how="left")
    ref["age"] = pd.to_numeric(ref["AGE_AT_SCAN"], errors="coerce")
    ref["fd"] = pd.to_numeric(ref["func_mean_fd"], errors="coerce")
    ref["dx_num"] = (ref["DX"] == "ASD").astype(int)
    ref["intercept"] = 1.0
    d6 = ref.dropna(subset=["PCC_mPFC_z", "dx_num", "age", "fd", "SITE_ID"]).reset_index(drop=True)
    X6 = pd.concat([d6[["intercept", "dx_num", "age", "fd"]],
                    pd.get_dummies(d6["SITE_ID"], prefix="site", drop_first=True)], axis=1).astype(float)
    y6 = d6["PCC_mPFC_z"].values.astype(float)
    m6 = sm.OLS(y6, X6).fit()
    idx6 = list(X6.columns).index("dx_num")
    radius_results.append({
        "radius": "6mm (reference)", "N": len(d6),
        "ASD": int((d6["DX"]=="ASD").sum()), "TD": int((d6["DX"]=="TD").sum()),
        "beta": round(m6.params.iloc[idx6], 5),
        "CI_lower": round(m6.conf_int().iloc[idx6, 0], 5),
        "CI_upper": round(m6.conf_int().iloc[idx6, 1], 5),
        "t": round(m6.tvalues.iloc[idx6], 4), "p": round(m6.pvalues.iloc[idx6], 5),
    })
    sig6 = " *" if m6.pvalues.iloc[idx6] < 0.05 else ""
    print(f"  6mm (reference): N={len(d6):>3d}, β={m6.params.iloc[idx6]:.4f}, p={m6.pvalues.iloc[idx6]:.4f}{sig6}")

rdf = pd.DataFrame(radius_results)
rdf.to_csv(os.path.join(OUT_DIR, "multi_radius_results.csv"), index=False)
print(f"\nSaved to multi_radius_results.csv")
print("Done.")
