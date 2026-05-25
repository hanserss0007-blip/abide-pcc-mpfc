"""
Step 25: Additional control connections — DMN-internal specificity.
Use existing PCC seed FC maps to extract FC at angular gyrus (AG) ROIs,
testing whether PCC-mPFC hypoconnectivity is specific within DMN.
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
FC_MAP_DIR = os.path.join(ABIDE_DIR, "fc_maps")
OUT_DIR = os.path.join(ABIDE_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Define angular gyrus ROIs (DMN control nodes) ──
# Left and right angular gyrus from standard DMN literature
# Using coordinates from Andrews-Hanna et al. 2010 and Schaefer atlas
AG_ROIS = {
    "AG_left": [-44, -62, 32],
    "AG_right": [48, -62, 32],
}
AG_RADIUS = 6  # same as primary seed radius

# ── Load subject list ──
fc = pd.read_csv(FC_CSV, dtype={"SUB_ID": str})
demo = pd.read_csv(SUB_CSV, dtype={"SUB_ID": str})[["SUB_ID", "subj_id", "SITE_ID", "DX",
                                                      "AGE_AT_SCAN", "func_mean_fd", "dataset"]]
# FC data already has subj_id, SITE_ID, DX, dataset
# Just merge AGE_AT_SCAN and func_mean_fd from demo
merge_cols = ["SUB_ID", "SITE_ID", "DX", "dataset"]
df = fc.merge(demo[merge_cols + ["AGE_AT_SCAN", "func_mean_fd"]],
              on=merge_cols, how="left", suffixes=("", "_y"))
# Drop any duplicate columns from the merge
for c in list(df.columns):
    if c.endswith("_y") and c.replace("_y", "") in df.columns:
        df.drop(columns=[c], inplace=True)
df["age"] = pd.to_numeric(df["AGE_AT_SCAN"], errors="coerce")
df["fd"] = pd.to_numeric(df["func_mean_fd"], errors="coerce")
df = df.dropna(subset=["age", "fd", "SITE_ID"]).reset_index(drop=True)
print(f"Total subjects: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# ── Extract AG FC from PCC seed FC maps ──
import nibabel as nib
from nilearn.image import resample_to_img
from scipy import ndimage

# Create AG sphere masks in MNI space
# We'll extract voxel coordinates from a reference FC map to determine the space
ref_fc_path = os.path.join(FC_MAP_DIR, f"{df.iloc[0]['subj_id']}_PCC_fc.nii.gz")
if not os.path.exists(ref_fc_path):
    # Try another subject
    for _, row in df.iterrows():
        ref_fc_path = os.path.join(FC_MAP_DIR, f"{row['subj_id']}_PCC_fc.nii.gz")
        if os.path.exists(ref_fc_path):
            break

ref_img = nib.load(ref_fc_path)
ref_data = ref_img.get_fdata()
ref_affine = ref_img.affine

# Create AG sphere masks
ag_masks = {}
for ag_name, ag_mni in AG_ROIS.items():
    # Convert MNI to voxel
    vox = nib.affines.apply_affine(np.linalg.inv(ref_affine), ag_mni)
    vox = np.round(vox).astype(int)

    # Create sphere
    xx, yy, zz = np.ogrid[:ref_data.shape[0], :ref_data.shape[1], :ref_data.shape[2]]
    dist = np.sqrt((xx - vox[0])**2 + (yy - vox[1])**2 + (zz - vox[2])**2)
    sphere = dist <= (AG_RADIUS / np.abs(ref_affine[0, 0]))  # convert mm to voxels
    ag_masks[ag_name] = sphere
    n_vox = sphere.sum()
    print(f"  {ag_name}: {ag_mni}, voxels in sphere: {n_vox}")

# Combine left + right AG
ag_mask_combined = ag_masks["AG_left"] | ag_masks["AG_right"]

# ── Extract AG FC values for all subjects ──
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

        # If different dimensions, resample mask
        if img_data.shape != ref_data.shape:
            from nilearn.image import resample_to_img
            ag_nii = nib.Nifti1Image(ag_mask_combined.astype(np.float32), ref_affine)
            ag_resamp = resample_to_img(ag_nii, img, interpolation="nearest")
            ag_m = ag_resamp.get_fdata().astype(bool)
        else:
            ag_m = ag_mask_combined

        # Extract mean FC in AG region
        ag_fc = float(img_data[ag_m].mean())

        # Also extract left/right separately
        if img_data.shape == ref_data.shape:
            ag_l = float(img_data[ag_masks["AG_left"]].mean())
            ag_r = float(img_data[ag_masks["AG_right"]].mean())
        else:
            ag_l_nii = nib.Nifti1Image(ag_masks["AG_left"].astype(np.float32), ref_affine)
            ag_r_nii = nib.Nifti1Image(ag_masks["AG_right"].astype(np.float32), ref_affine)
            ag_l_resamp = resample_to_img(ag_l_nii, img, interpolation="nearest")
            ag_r_resamp = resample_to_img(ag_r_nii, img, interpolation="nearest")
            ag_l = float(img_data[ag_l_resamp.get_fdata().astype(bool)].mean())
            ag_r = float(img_data[ag_r_resamp.get_fdata().astype(bool)].mean())

        results.append({
            "subj_id": sid, "SUB_ID": row["SUB_ID"],
            "SITE_ID": row["SITE_ID"], "DX": row["DX"],
            "dataset": row["dataset"],
            "PCC_AG_comb_z": float(np.arctanh(np.clip(ag_fc, -0.999, 0.999))),
            "PCC_AG_left_z": float(np.arctanh(np.clip(ag_l, -0.999, 0.999))),
            "PCC_AG_right_z": float(np.arctanh(np.clip(ag_r, -0.999, 0.999))),
            "PCC_AG_comb_r": float(ag_fc),
            "age": row["age"], "fd": row["fd"],
        })
    except Exception as e:
        missing += 1
        if missing <= 5:
            print(f"  [ERR] {sid}: {e}")
        continue

    if (idx + 1) % 200 == 0:
        print(f"  {idx+1}/{len(df)} processed")

rdf = pd.DataFrame(results)
print(f"\nProcessed: {len(rdf)}/{len(df)}, missing/mismatch: {missing}")

# ── Run OLS models for each AG metric ──
import statsmodels.api as sm
rdf["dx_num"] = (rdf["DX"] == "ASD").astype(int)
rdf["intercept"] = 1.0

def run_ag_model(data, fc_col, label):
    d = data.dropna(subset=[fc_col, "dx_num", "age", "fd", "SITE_ID"]).reset_index(drop=True)
    X = pd.concat([d[["intercept", "dx_num", "age", "fd"]],
                   pd.get_dummies(d["SITE_ID"], prefix="site", drop_first=True)], axis=1).astype(float)
    y = d[fc_col].values.astype(float)
    m = sm.OLS(y, X).fit()
    idx = list(X.columns).index("dx_num")
    beta = m.params.iloc[idx]
    ci = m.conf_int().iloc[idx].values
    t = m.tvalues.iloc[idx]
    p = m.pvalues.iloc[idx]
    n_asd = (d["DX"] == "ASD").sum()
    n_td = (d["DX"] == "TD").sum()
    return {"connection": label, "N": len(d), "ASD": n_asd, "TD": n_td,
            "beta": round(beta, 5), "CI_lower": round(ci[0], 5), "CI_upper": round(ci[1], 5),
            "t": round(t, 4), "p": round(p, 5)}

print(f"\n{'='*55}")
print("Additional Control Connections: PCC-Angular Gyrus")
print(f"{'='*55}")

ag_models = [
    ("PCC_AG_comb_z", "PCC-AG (combined)"),
    ("PCC_AG_left_z", "PCC-AG (left)"),
    ("PCC_AG_right_z", "PCC-AG (right)"),
]

rows = []
for fc_col, label in ag_models:
    res = run_ag_model(rdf, fc_col, label)
    rows.append(res)
    sig = " *" if res["p"] < 0.05 else ""
    print(f"  {res['connection']:<25s} N={res['N']:>3d} | β={res['beta']:>7.4f} [{res['CI_lower']:>7.4f}, {res['CI_upper']:>7.4f}] | p={res['p']:.4f}{sig}")

# Also run PCC-mPFC for comparison on this subsample
d_ref = df.dropna(subset=["PCC_mPFC_z", "age", "fd", "SITE_ID"]).reset_index(drop=True)
d_ref["dx_num"] = (d_ref["DX"] == "ASD").astype(int)
d_ref["intercept"] = 1.0
X_ref = pd.concat([d_ref[["intercept", "dx_num", "age", "fd"]],
                   pd.get_dummies(d_ref["SITE_ID"], prefix="site", drop_first=True)], axis=1).astype(float)
y_ref = d_ref["PCC_mPFC_z"].values.astype(float)
m_ref = sm.OLS(y_ref, X_ref).fit()
idx_ref = list(X_ref.columns).index("dx_num")
rows.append({"connection": "PCC-mPFC (reference)", "N": len(d_ref),
             "ASD": int((d_ref["DX"]=="ASD").sum()), "TD": int((d_ref["DX"]=="TD").sum()),
             "beta": round(m_ref.params.iloc[idx_ref], 5),
             "CI_lower": round(m_ref.conf_int().iloc[idx_ref, 0], 5),
             "CI_upper": round(m_ref.conf_int().iloc[idx_ref, 1], 5),
             "t": round(m_ref.tvalues.iloc[idx_ref], 4),
             "p": round(m_ref.pvalues.iloc[idx_ref], 5)})
print(f"  {'PCC-mPFC (reference)':<25s} N={d_ref.shape[0]:>3d} | β={rows[-1]['beta']:>7.4f} [{rows[-1]['CI_lower']:>7.4f}, {rows[-1]['CI_upper']:>7.4f}] | p={rows[-1]['p']:.4f}")

# ── Save ──
rdf_out = pd.DataFrame(results)
rdf_out.to_csv(os.path.join(OUT_DIR, "ag_control_connectivity.csv"), index=False)

rows_df = pd.DataFrame(rows)
rows_df.to_csv(os.path.join(OUT_DIR, "ag_control_results.csv"), index=False)
print(f"\nSaved to ag_control_connectivity.csv and ag_control_results.csv")
print("Done.")
