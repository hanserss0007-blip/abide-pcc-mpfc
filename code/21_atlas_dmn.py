"""
Step 21: Atlas-based DMN validation.
Use Schaefer 100-parcel atlas to define mPFC ROI (vs 6mm sphere),
extract PCC-mPFC FC from existing PCC seed maps, re-run group comparison.
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
FC_MAP_DIR = os.path.join(ABIDE_DIR, "fc_maps")
OUT_DIR = os.path.join(ABIDE_DIR, "results")

SEED_ROI = [0, -52, 26]  # PCC seed (6mm)
RADIUS = 6

# ── Fetch Schaefer 100-parcel atlas ──
from nilearn import datasets
atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=2)
atlas_img = atlas.maps
labels = atlas.labels

import nibabel as nib
import numpy as np
from nilearn.image import resample_to_img

atlas_nii = nib.load(atlas_img)
atlas_data = atlas_nii.get_fdata()
atlas_affine = atlas_nii.affine

# ── Identify mPFC and PCC parcels ──
# Each parcel's center of mass
parcel_centers = {}
for i in range(1, 101):  # parcels are 1-indexed
    mask = atlas_data == i
    if mask.sum() == 0:
        continue
    coords = np.where(mask)
    # Center of mass in voxel space → MNI mm
    vox_center = np.array([c.mean() for c in coords])
    mni = atlas_affine[:3, :3] @ vox_center + atlas_affine[:3, 3]
    parcel_centers[i] = mni

# Classify based on label name and MNI coordinates
mpfc_parcels = []
pcc_parcels = []
for i in range(1, 101):
    idx = i - 1
    lbl = labels[idx] if idx < len(labels) else f"parcel_{i}"
    if "Default" not in lbl:
        continue
    mni = parcel_centers.get(i)
    if mni is None:
        continue
    # mPFC: y > 0, |x| < 20, z between -10 and 40
    if mni[1] > 0 and abs(mni[0]) < 25 and -10 < mni[2] < 45:
        mpfc_parcels.append(i)
    # PCC: y < -20, |x| < 20, z > 10
    if mni[1] < -20 and abs(mni[0]) < 25 and mni[2] > 10:
        pcc_parcels.append(i)

print(f"Schaefer 100-parcel atlas:")
print(f"  mPFC parcels ({len(mpfc_parcels)}): {mpfc_parcels}")
for p in mpfc_parcels:
    print(f"    P{p}: {labels[p-1]} @ {np.round(parcel_centers[p], 1)}")
print(f"  PCC parcels ({len(pcc_parcels)}): {pcc_parcels}")
for p in pcc_parcels:
    print(f"    P{p}: {labels[p-1]} @ {np.round(parcel_centers[p], 1)}")

# Create mPFC atlas mask
mpfc_mask = np.isin(atlas_data, mpfc_parcels).astype(np.float32)

# ── Load subject list ──
fc = pd.read_csv(os.path.join(ABIDE_DIR, "timeseries", "seed_connectivity.csv"), dtype={"SUB_ID": str})
demo = pd.read_csv(SUB_CSV, dtype={"SUB_ID": str})[["SUB_ID", "SITE_ID", "DX", "AGE_AT_SCAN", "func_mean_fd"]]
df = fc.merge(demo, on=["SUB_ID", "SITE_ID", "DX"], how="left")
df["age"] = pd.to_numeric(df["AGE_AT_SCAN"], errors="coerce")
df["fd"] = pd.to_numeric(df["func_mean_fd"], errors="coerce")
df = df.dropna(subset=["age", "fd", "SITE_ID"]).reset_index(drop=True)

# ── Extract atlas mPFC FC from PCC seed maps ──
results = []
missing = 0
for idx, row in df.iterrows():
    sid = row["subj_id"]
    fpath = os.path.join(FC_MAP_DIR, f"{sid}_PCC_fc.nii.gz")
    if not os.path.exists(fpath):
        missing += 1
        continue

    try:
        img = nib.load(fpath)
        img_data = img.get_fdata()
        img_affine = img.affine

        # Resample atlas mask to FC map space if needed
        if img.shape[:3] != mpfc_mask.shape:
            from nilearn.image import resample_to_img
            mpfc_nii = nib.Nifti1Image(mpfc_mask, atlas_affine)
            mpfc_resamp = resample_to_img(mpfc_nii, img, interpolation="nearest")
            mpfc_m = mpfc_resamp.get_fdata().astype(bool)
        else:
            mpfc_m = mpfc_mask.astype(bool)

        atlas_fc = float(img_data[mpfc_m].mean())
        results.append({"subj_id": sid, "SUB_ID": row["SUB_ID"],
                        "SITE_ID": row["SITE_ID"], "DX": row["DX"],
                        "atlas_mpfc_fc": atlas_fc,
                        "age": row["age"], "fd": row["fd"]})
    except Exception as e:
        missing += 1

    if (idx + 1) % 100 == 0:
        print(f"  {idx+1}/{len(df)} processed")

print(f"Processed: {len(results)}, missing: {missing}")
rdf = pd.DataFrame(results)
rdf.to_csv(os.path.join(OUT_DIR, "atlas_dmn_values.csv"), index=False)

# ── Group comparison ──
rdf["dx_num"] = (rdf["DX"] == "ASD").astype(int)
import statsmodels.api as sm

data = rdf.dropna(subset=["atlas_mpfc_fc", "dx_num", "age", "fd", "SITE_ID"]).reset_index(drop=True)
data["intercept"] = 1.0
X = pd.concat([data[["intercept", "dx_num", "age", "fd"]],
               pd.get_dummies(data["SITE_ID"], prefix="site", drop_first=True)], axis=1).astype(float)
y = data["atlas_mpfc_fc"].values.astype(float)

m = sm.OLS(y, X).fit()
idx_dx = list(X.columns).index("dx_num")
beta = m.params.iloc[idx_dx]
ci = m.conf_int().iloc[idx_dx].values
t = m.tvalues.iloc[idx_dx]
p = m.pvalues.iloc[idx_dx]

print(f"\n{'='*55}")
print("Atlas DMN Validation (Schaefer 100-parcel mPFC)")
print(f"{'='*55}")
print(f"  N = {len(data)} (ASD={int((data['DX']=='ASD').sum())}, TD={int((data['DX']=='TD').sum())})")
print(f"  DX beta = {beta:.4f}")
print(f"  95% CI = [{ci[0]:.4f}, {ci[1]:.4f}]")
print(f"  t = {t:.4f}, p = {p:.4f}")
print(f"\n  Comparison with 6mm sphere result: β ≈ -0.058, p ≈ 0.019")

# Save
summary = pd.DataFrame([{
    "analysis": "Atlas DMN (Schaefer100 mPFC parcels)",
    "N": len(data),
    "ASD": int((data["DX"]=="ASD").sum()),
    "TD": int((data["DX"]=="TD").sum()),
    "beta": round(beta, 5),
    "ci_lower": round(ci[0], 5),
    "ci_upper": round(ci[1], 5),
    "t": round(t, 4),
    "p": round(p, 5),
}])
summary.to_csv(os.path.join(OUT_DIR, "atlas_dmn_result.csv"), index=False)
print(f"\nDone.")
