"""
Step 11: Coverage QC & subject screening (M0-M1).
For each subject, compute brain mask coverage of seed ROIs (PCC, mPFC, rAI).
Outputs exclusion recommendations based on coverage + motion thresholds.
"""
import pandas as pd, numpy as np, os, nibabel as nib
warnings = __import__('warnings'); warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUB_CSV = os.path.join(BASE, "code", "combined_subjects.csv")
OUT_DIR = "E:/ABIDE项目/results"
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = {"PCC": [0, -52, 26], "mPFC": [0, 54, -2], "rAI": [38, 22, -4]}
RADIUS_MM = 6

df = pd.read_csv(SUB_CSV)
print(f"Total subjects: {len(df)}")

def seed_coverage(mask_img, seed_mni, radius_mm):
    """Fraction of seed sphere voxels inside brain mask."""
    data = mask_img.get_fdata()
    affine = mask_img.affine
    inv_aff = np.linalg.inv(affine)
    cx, cy, cz = (inv_aff @ np.array([*seed_mni, 1]))[:3].round().astype(int)

    # Voxel radius (handle non-isotropic)
    vox_sizes = np.sqrt(np.sum(affine[:3, :3]**2, axis=0))
    r_vox = radius_mm / vox_sizes  # per-axis radius

    i, j, k = np.ogrid[:data.shape[0], :data.shape[1], :data.shape[2]]
    sphere = ((i-cx)/r_vox[0])**2 + ((j-cy)/r_vox[1])**2 + ((k-cz)/r_vox[2])**2 <= 1

    if sphere.sum() == 0:
        return 0.0
    return float((sphere & (data > 0)).sum() / sphere.sum())

results = []
errors = []
for idx, row in df.iterrows():
    mask_path = row["mask_path"]
    if not isinstance(mask_path, str) or not os.path.exists(mask_path):
        errors.append((row["subj_id"], "mask missing"))
        continue
    try:
        mask_img = nib.load(mask_path)
        cov = {}
        for name, coord in SEEDS.items():
            cov[name] = seed_coverage(mask_img, coord, RADIUS_MM)
        cov["subj_id"] = row["subj_id"]
        cov["SITE_ID"] = row["SITE_ID"]
        cov["DX"] = row["DX"]
        cov["dataset"] = row["dataset"]
        cov["FD"] = row["func_mean_fd"]
        cov["AGE"] = row["AGE_AT_SCAN"]
        results.append(cov)
    except Exception as e:
        errors.append((row["subj_id"], str(e)))

rdf = pd.DataFrame(results)
# Combine coverage: min across DMN seeds
rdf["min_coverage"] = rdf[["PCC", "mPFC", "rAI"]].min(axis=1)
rdf["mean_coverage"] = rdf[["PCC", "mPFC", "rAI"]].mean(axis=1)

# Exclusion flags
rdf["exclude_coverage"] = rdf["min_coverage"] < 0.80
rdf["exclude_fd_03"] = rdf["FD"] >= 0.3
rdf["exclude_fd_02"] = rdf["FD"] >= 0.2

# M0: no exclusions; M1: coverage >= 80%
rdf["included_M0"] = True
rdf["included_M1"] = ~rdf["exclude_coverage"]
rdf["included_M2"] = ~rdf["exclude_coverage"] & ~rdf["exclude_fd_03"]
rdf["included_M3"] = ~rdf["exclude_coverage"] & ~rdf["exclude_fd_02"]

rdf.to_csv(os.path.join(OUT_DIR, "coverage_qc.csv"), index=False)
print(f"  Processed: {len(rdf)}, errors: {len(errors)}")

print(f"\n{'='*50}")
print("Coverage Summary")
print(f"{'='*50}")
for s in ["PCC", "mPFC", "rAI"]:
    print(f"  {s}: mean={rdf[s].mean():.3f}, min={rdf[s].min():.3f}, "
          f"<80%: {(rdf[s] < 0.80).sum()} subjects")

print(f"\n  Min across seeds: mean={rdf['min_coverage'].mean():.3f}, "
      f"<80%: {rdf['exclude_coverage'].sum()} subjects ({rdf['exclude_coverage'].sum()/len(rdf)*100:.1f}%)")

print(f"\n{'='*50}")
print("Motion Summary")
print(f"{'='*50}")
print(f"  FD mean={rdf['FD'].mean():.4f}, median={rdf['FD'].median():.4f}")
print(f"  FD >= 0.3: {rdf['exclude_fd_03'].sum()} subjects")
print(f"  FD >= 0.2: {rdf['exclude_fd_02'].sum()} subjects")

print(f"\n{'='*50}")
print("Nested Sample Sizes (M0-M3)")
print(f"{'='*50}")
for m in ["M0", "M1", "M2", "M3"]:
    inc = rdf[rdf[f"included_{m}"]]
    n_asd = (inc["DX"] == "ASD").sum()
    n_td = (inc["DX"] == "TD").sum()
    print(f"  {m}: n={len(inc)} (ASD={n_asd}, TD={n_td})")
    # Per site
    site_counts = inc.groupby("SITE_ID").size()
    print(f"       sites={len(site_counts)}, min_site={site_counts.min()}")

# List excluded by coverage
exc = rdf[rdf["exclude_coverage"]]
if len(exc) > 0:
    print(f"\nExcluded by coverage (min < 80%):")
    for _, r in exc.iterrows():
        print(f"  {r['subj_id']} ({r['SITE_ID']}, {r['dataset']}): "
              f"PCC={r['PCC']:.2f}, mPFC={r['mPFC']:.2f}, rAI={r['rAI']:.2f}")

if errors:
    print(f"\nErrors ({len(errors)}):")
    for s, e in errors:
        print(f"  {s}: {e}")

print(f"\nQC results saved to {os.path.join(OUT_DIR, 'coverage_qc.csv')}")

# Site-level coverage summary
print(f"\n{'='*50}")
print("Per-site Coverage")
print(f"{'='*50}")
for site in sorted(rdf["SITE_ID"].unique()):
    sd = rdf[rdf["SITE_ID"] == site]
    print(f"  {site}: n={len(sd)}, min_cov={sd['min_coverage'].mean():.3f}, "
          f"excluded={sd['exclude_coverage'].sum()}/{len(sd)}")
