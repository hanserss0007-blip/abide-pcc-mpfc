"""
Step 26b: Manual sphere extraction for multi-radius sensitivity.
Avoids nilearn NiftiSpheresMasker hanging bug on certain subjects.
"""
import pandas as pd, numpy as np, os, gc, warnings, time
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

SEEDS_MM = np.array([[0, -52, 26], [0, 54, -2]])  # PCC, mPFC
RADII = [4, 8]

df = pd.read_csv(SUB_CSV, dtype={"SUB_ID": str})
print(f"Total subjects: {len(df)}", flush=True)

# Load existing 6mm FC
FC_CSV = os.path.join(ABIDE_DIR, "timeseries", "seed_connectivity.csv")
fc6 = pd.read_csv(FC_CSV, dtype={"SUB_ID": str}) if os.path.exists(FC_CSV) else None

# Already done
done = set()
if os.path.exists(RADII_OUT):
    existing = pd.read_csv(RADII_OUT, dtype={"SUB_ID": str})
    done = set(existing["SUB_ID"].astype(str))
    print(f"Already processed: {len(done)}", flush=True)

import nibabel as nib

def sphere_timeseries(img_data, affine, seed_mm, radius_mm, mask_data=None):
    """Extract mean time series from a spherical ROI."""
    # Convert seed from mm to voxel
    inv_affine = np.linalg.inv(affine)
    seed_vox = inv_affine[:3, :3] @ np.array(seed_mm) + inv_affine[:3, 3]
    seed_vox = np.round(seed_vox).astype(int)

    # Create voxel coordinate grid
    shape = img_data.shape[:3]
    x, y, z = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]),
        indexing='ij'
    )

    # Convert voxel coords to mm
    vox2mm = affine[:3, :3]
    offset = affine[:3, 3]
    x_mm = vox2mm[0, 0] * x + vox2mm[0, 1] * y + vox2mm[0, 2] * z + offset[0]
    y_mm = vox2mm[1, 0] * x + vox2mm[1, 1] * y + vox2mm[1, 2] * z + offset[1]
    z_mm = vox2mm[2, 0] * x + vox2mm[2, 1] * y + vox2mm[2, 2] * z + offset[2]

    # Distance from seed
    dist = np.sqrt((x_mm - seed_mm[0])**2 + (y_mm - seed_mm[1])**2 + (z_mm - seed_mm[2])**2)
    sphere_mask = dist <= radius_mm

    # Apply brain mask if provided
    if mask_data is not None:
        sphere_mask = sphere_mask & (mask_data > 0)

    idx = np.where(sphere_mask)
    if len(idx[0]) == 0:
        return None

    # Extract and mean time series
    ts = img_data[idx[0], idx[1], idx[2], :]  # (n_voxels, n_timepoints)
    return np.mean(ts, axis=0)  # (n_timepoints,)

results = []
n_total = len(df)
timeout_s = 120

for i, (_, row) in enumerate(df.iterrows()):
    sub = str(row["SUB_ID"])
    if sub in done:
        continue

    nii_path = row["func_path"]
    msk_path = row["mask_path"] if isinstance(row["mask_path"], str) else ""
    if not os.path.exists(nii_path):
        continue

    try:
        t0 = time.time()
        img = nib.load(nii_path)
        img_data = img.get_fdata(dtype=np.float32)
        affine = img.affine

        mask_data = None
        if msk_path and os.path.exists(msk_path):
            mask_img = nib.load(msk_path)
            mask_data = mask_img.get_fdata(dtype=np.float32)

        for radius in RADII:
            ts_list = []
            for seed in SEEDS_MM:
                ts = sphere_timeseries(img_data, affine, seed, radius, mask_data)
                if ts is None:
                    raise ValueError(f"Empty sphere at {seed}")
                ts_list.append(ts)

            pcc, mpfc = ts_list
            # Detrend (linear)
            t_pts = np.arange(len(pcc))
            pcc_det = pcc - np.polyval(np.polyfit(t_pts, pcc, 1), t_pts)
            mpfc_det = mpfc - np.polyval(np.polyfit(t_pts, mpfc, 1), t_pts)

            r_val = float(np.corrcoef(pcc_det, mpfc_det)[0, 1])
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

        elapsed = time.time() - t0
        if elapsed > timeout_s:
            print(f"  [WARN] {row['subj_id']}: slow ({elapsed:.0f}s)", flush=True)

    except Exception as e:
        print(f"  [ERR] {row['subj_id']}: {e}", flush=True)
        continue

    if (i + 1) % 50 == 0:
        print(f"  Progress: {i+1}/{n_total} ({len(results)} in batch)", flush=True)

    # Write every 100
    if len(results) >= 100:
        new_df = pd.DataFrame(results)
        if os.path.exists(RADII_OUT):
            old = pd.read_csv(RADII_OUT, dtype={"SUB_ID": str})
            new_df = pd.concat([old, new_df], ignore_index=True)
        new_df.to_csv(RADII_OUT, index=False)
        print(f"  Saved {len(new_df)}/{n_total}", flush=True)
        results = []
        gc.collect()

if results:
    new_df = pd.DataFrame(results)
    if os.path.exists(RADII_OUT):
        old = pd.read_csv(RADII_OUT, dtype={"SUB_ID": str})
        new_df = pd.concat([old, new_df], ignore_index=True)
    new_df.to_csv(RADII_OUT, index=False)

print(f"\nFinal check:", flush=True)
final = pd.read_csv(RADII_OUT, dtype={"SUB_ID": str})
print(f"  Total rows: {len(final)}", flush=True)
print(f"  Unique subjects: {final['SUB_ID'].nunique()}", flush=True)
print(f"  By dataset: {final.groupby('dataset')['SUB_ID'].nunique().to_dict()}", flush=True)
print("Done.", flush=True)
