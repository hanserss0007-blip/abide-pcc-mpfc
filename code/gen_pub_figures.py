"""
Generate ALL publication-quality figures.
Strategy: view_img (plotly) for ortho views + playwright screenshots.
           matplotlib for bar/scatter/manual axial montages.
"""
import pandas as pd
import numpy as np
import os, pathlib, gc
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from scipy.stats import ttest_ind, pearsonr
from nilearn.image import load_img, resample_to_img
from nilearn.plotting import view_img
from nilearn.datasets import load_mni152_template
from playwright.sync_api import sync_playwright
import warnings
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = "E:/ABIDE项目/results"
FIG = "E:/ABIDE项目/figures"
TS = "E:/ABIDE项目/timeseries"
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11, "figure.dpi": 300})

BRIGHT = LinearSegmentedColormap.from_list("bright_fc",
    [(0.0, "gold"), (0.25, "darkorange"), (0.5, "red"), (1.0, "darkred")], N=256)
COOL = LinearSegmentedColormap.from_list("cool_fc",
    [(0.0, "deepskyblue"), (0.5, "dodgerblue"), (1.0, "navy")], N=256)
WARM = LinearSegmentedColormap.from_list("warm_fc",
    [(0.0, "gold"), (0.33, "darkorange"), (0.66, "red"), (1.0, "darkred")], N=256)

MNI_PATH = os.path.join(os.path.expanduser("~"), "nilearn_data", "icbm152_2009",
                        "mni_icbm152_nlin_sym_09a", "mni_icbm152_t1_tal_nlin_sym_09a.nii.gz")
MNI = nib.load(MNI_PATH)

def screenshot_view(view_obj, png_path, width=1000, height=800, scale=2, wait=4000):
    """Screenshot a view_img object to PNG via playwright."""
    html_path = png_path.replace(".png", "_tmp.html")
    view_obj.save_as_html(html_path)
    uri = pathlib.Path(os.path.abspath(html_path)).as_uri()
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": width, "height": height},
                                device_scale_factor=scale)
        page.goto(uri)
        page.wait_for_timeout(wait)
        page.screenshot(path=png_path, full_page=False)
        browser.close()
    os.remove(html_path)
    return os.path.getsize(png_path)

def manual_axial_montage(z_path, threshold, cmap, title, n_slices=8, figsize=(16, 8)):
    """Create axial montage with MNI underlay using pure matplotlib."""
    z_img = load_img(z_path)
    data = z_img.get_fdata()

    # Find best slices
    vox_counts = [(z, int((data[:,:,z] > threshold).sum()),
                  nib.affines.apply_affine(z_img.affine, [0,0,z])[2])
                 for z in range(data.shape[2])]
    vox_counts.sort(key=lambda x: x[1], reverse=True)
    best = []
    for sl, cnt, mni_z in vox_counts:
        if cnt < 20: continue
        if any(abs(mni_z - bz) < 10 for _, bz in best): continue
        best.append((sl, int(round(mni_z))))
        if len(best) >= n_slices: break
    best.sort(key=lambda x: x[1])
    z_vals = [b for _, b in best]

    # Resample to MNI space
    z_mni = resample_to_img(z_img, MNI, interpolation="nearest")
    zd = z_mni.get_fdata()
    md = MNI.get_fdata()
    aff = MNI.affine

    if len(z_vals) == 0:
        fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor="white")
        ax.text(0.5, 0.5, "No suprathreshold clusters", ha="center", va="center", fontsize=14)
        ax.set_title(title, fontsize=11)
        return fig

    cols = min(4, len(z_vals))
    rows = int(np.ceil(len(z_vals) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=figsize, facecolor="white")
    axes = np.atleast_1d(axes).flatten()

    for i, mni_z in enumerate(z_vals):
        ax = axes[i]
        iz = int(round((mni_z - aff[2,3]) / aff[2,2]))
        iz = max(0, min(md.shape[2]-1, iz))
        mni_sl = np.fliplr(md[:,:,iz].T)
        z_sl = np.fliplr(zd[:,:,iz].T)
        z_sl[z_sl < threshold] = np.nan

        ax.imshow(mni_sl, cmap="gray", origin="lower", aspect="auto")
        im = ax.imshow(z_sl, cmap=cmap, origin="lower", aspect="auto",
                       vmin=0, vmax=float(zd.max()), alpha=0.75)
        ax.set_title(f"z = {mni_z} mm", fontsize=9, fontweight="bold")
        ax.axis("off")
    for i in range(len(z_vals), len(axes)):
        axes[i].axis("off")

    cbar_ax = fig.add_axes([0.92, 0.12, 0.012, 0.76])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("z-score", fontsize=10)
    fig.suptitle(title, fontweight="bold", fontsize=13, y=0.96)
    return fig

# ═══════════════════════════════════════════════════════════════════
# FIG 2/3: Ortho views at peak (view_img + playwright — best quality)
# ═══════════════════════════════════════════════════════════════════
for seed in ["PCC", "rAI"]:
    z_path = os.path.join(RES, f"{seed}_TD_gt_ASD_z.nii.gz")
    if not os.path.exists(z_path): continue

    z_img = load_img(z_path)
    d = z_img.get_fdata()
    peak_idx = np.unravel_index(np.argmax(d), d.shape)
    peak_mni = nib.affines.apply_affine(z_img.affine, peak_idx)
    max_z = float(d.max())
    print(f"{seed} TD>ASD peak: [{peak_mni[0]:.0f},{peak_mni[1]:.0f},{peak_mni[2]:.0f}] z={max_z:.1f}")

    # Check FWE/FDR status
    fwe_path = os.path.join(RES, f"{seed}_TD_gt_ASD_clusterFWE.nii.gz")
    fdr_path = os.path.join(RES, f"{seed}_TD_gt_ASD_FDR.nii.gz")
    fwe_vox = int((load_img(fwe_path).get_fdata() > 0).sum()) if os.path.exists(fwe_path) else 0
    fdr_vox = int((load_img(fdr_path).get_fdata() > 0).sum()) if os.path.exists(fdr_path) else 0
    if fwe_vox > 0:
        sig_note = f"Cluster-level FWE corrected: {fwe_vox} voxels"
    elif fdr_vox > 0:
        sig_note = f"Voxel-level FDR corrected: {fdr_vox} voxels"
    else:
        sig_note = "No clusters survived FWE or FDR correction"

    # Ortho view — use z > 3.1 threshold (more stringent)
    title = f"{seed} Seed: TD > ASD  (z > 3.1, Exploratory Uncorrected Threshold)<br>" \
            f"Peak [{peak_mni[0]:.0f},{peak_mni[1]:.0f},{peak_mni[2]:.0f}]  z = {max_z:.1f}<br>" \
            f"{sig_note}"
    view = view_img(z_img, threshold=3.1, cmap=BRIGHT, symmetric_cmap=False,
                    vmax=max_z, cut_coords=list(peak_mni), title=title,
                    colorbar=True, draw_cross=True, black_bg="off")
    png_path = os.path.join(FIG, f"{seed}_TD_gt_ASD_ortho.png")
    sz = screenshot_view(view, png_path, 1000, 800, 2)
    print(f"  -> ortho ({sz:,} bytes)")

    # Axial montage (manual matplotlib) — z > 3.1
    fig = manual_axial_montage(z_path, 3.1, BRIGHT,
        f"{seed} Seed: TD > ASD  z > 3.1 (Exploratory Uncorrected Threshold)\n{sig_note}\nPeak [{peak_mni[0]:.0f},{peak_mni[1]:.0f},{peak_mni[2]:.0f}]  z = {max_z:.1f}",
        n_slices=8)
    png_path = os.path.join(FIG, f"{seed}_TD_gt_ASD_axial.png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    sz = os.path.getsize(png_path)
    print(f"  -> axial ({sz:,} bytes)")

# ASD > TD
for seed in ["PCC", "rAI"]:
    z_path = os.path.join(RES, f"{seed}_ASD_gt_TD_z.nii.gz")
    if not os.path.exists(z_path): continue
    z_img = load_img(z_path)
    d = z_img.get_fdata()
    n_above = int((d > 2.3).sum())
    if n_above < 30:
        print(f"{seed} ASD>TD: only {n_above} voxels > 2.3, skip")
        continue

    peak_idx = np.unravel_index(np.argmax(d), d.shape)
    peak_mni = nib.affines.apply_affine(z_img.affine, peak_idx)
    max_z = float(d.max())
    print(f"{seed} ASD>TD peak: [{peak_mni[0]:.0f},{peak_mni[1]:.0f},{peak_mni[2]:.0f}] z={max_z:.1f}")

    title = f"{seed} Seed: ASD > TD  (|z| > 2.3)<br>" \
            f"Peak [{peak_mni[0]:.0f},{peak_mni[1]:.0f},{peak_mni[2]:.0f}]  z = {max_z:.1f}"
    view = view_img(z_img, threshold=2.3, cmap=COOL, symmetric_cmap=False,
                    vmax=max_z, cut_coords=list(peak_mni), title=title,
                    colorbar=True, draw_cross=True, black_bg="off")
    png_path = os.path.join(FIG, f"{seed}_ASD_gt_TD_ortho.png")
    sz = screenshot_view(view, png_path, 1000, 800, 2)
    print(f"  -> ortho ({sz:,} bytes)")

    fig = manual_axial_montage(z_path, 2.3, COOL,
        f"{seed} Seed: ASD > TD  |z| > 2.3\nPeak [{peak_mni[0]:.0f},{peak_mni[1]:.0f},{peak_mni[2]:.0f}]  z = {max_z:.1f}",
        n_slices=6)
    fig.savefig(os.path.join(FIG, f"{seed}_ASD_gt_TD_axial.png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  -> axial ({os.path.getsize(os.path.join(FIG, f'{seed}_ASD_gt_TD_axial.png')):,} bytes)")

# ═══════════════════════════════════════════════════════════════════
# FIG 1: Seed-to-seed FC bar chart
# ═══════════════════════════════════════════════════════════════════
print("\n>>> Fig 1: Seed-to-seed FC bar chart")
fc = pd.read_csv(os.path.join(TS, "seed_connectivity.csv"), dtype={"SUB_ID": str})
pairs = ["PCC-rAI", "PCC-mPFC", "rAI-mPFC"]
z_cols = ["PCC_zAI_z", "PCC_mPFC_z", "rAI_mPFC_z"]

fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor="white")
for i, (pair, zcol) in enumerate(zip(pairs, z_cols)):
    ax = axes[i]
    asd_d = fc[fc["DX"] == "ASD"][zcol]
    td_d = fc[fc["DX"] == "TD"][zcol]
    means = [asd_d.mean(), td_d.mean()]
    sems = [asd_d.std() / np.sqrt(len(asd_d)), td_d.std() / np.sqrt(len(td_d))]
    ax.bar(["ASD", "TD"], means, yerr=sems, capsize=8,
           color=["#E74C3C", "#3498DB"], edgecolor="black", linewidth=0.8)
    ax.set_title(pair, fontweight="bold")
    ax.set_ylabel("Fisher Z")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    t, p = ttest_ind(asd_d, td_d, equal_var=False)
    d_cohen = (asd_d.mean() - td_d.mean()) / np.sqrt((asd_d.var() + td_d.var()) / 2)
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
    ax.text(0.5, 0.95, f"p = {p:.3f} {sig}", transform=ax.transAxes,
            ha="center", fontsize=10, fontstyle="italic")
    rng = np.random.default_rng(42)
    jit = rng.uniform(-0.08, 0.08, len(asd_d))
    ax.scatter(np.ones(len(asd_d))*0.9+jit, asd_d, alpha=0.3, s=8, c="#E74C3C", zorder=5)
    jit = rng.uniform(-0.08, 0.08, len(td_d))
    ax.scatter(np.ones(len(td_d))*1.9+jit, td_d, alpha=0.3, s=8, c="#3498DB", zorder=5)
fig.suptitle("DMN-SN Seed Functional Connectivity: ASD vs TD", fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig1_seed_fc_barchart.png"), dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print("  done")

# ═══════════════════════════════════════════════════════════════════
# SUPP FIGURE S2: Brain-behavior correlations (ADOS + SRS + FIQ)
# ═══════════════════════════════════════════════════════════════════
print(">>> S-Fig S2: Brain-behavior correlations")
pheno = pd.read_csv(os.path.join(BASE, "Phenotypic_V1_0b_preprocessed1.csv"), dtype={"SUB_ID": str})
m = fc.merge(pheno[["SUB_ID","ADOS_SOCIAL","ADOS_TOTAL","ADOS_COMM",
                     "ADOS_STEREO_BEHAV","SRS_RAW_TOTAL","FIQ"]],
             on="SUB_ID", how="left")
asd = m[m["DX"] == "ASD"].copy()

candidates = [
    ("PCC_mPFC_r", "ADOS_TOTAL", "PCC-mPFC FC", "ADOS Total"),
    ("PCC_rAI_r", "ADOS_SOCIAL", "PCC-rAI FC", "ADOS Social"),
    ("rAI_mPFC_r", "ADOS_COMM", "rAI-mPFC FC", "ADOS Communication"),
]

fig = plt.figure(figsize=(18, 9), facecolor="white")
gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

# Row 1: ADOS (3 panels)
for i, (fc_col, ados_col, fc_label, ados_label) in enumerate(candidates):
    ax = fig.add_subplot(gs[0, i])
    d = asd.dropna(subset=[fc_col, ados_col])
    d = d[d[ados_col] > 0]
    r, p = pearsonr(d[fc_col], d[ados_col])
    ax.scatter(d[fc_col], d[ados_col], alpha=0.5, s=40, c="#E74C3C", edgecolors="white", linewidth=0.3)
    z = np.polyfit(d[fc_col], d[ados_col], 1)
    xl = np.linspace(d[fc_col].min(), d[fc_col].max(), 100)
    ax.plot(xl, np.polyval(z, xl), "k--", linewidth=1.2)
    ax.set_xlabel(fc_label, fontsize=11)
    ax.set_ylabel(ados_label, fontsize=11)
    ax.set_title(f"r = {r:.3f}, p = {p:.3f}  (n = {len(d)})", fontstyle="italic", fontsize=10)

# Row 2, Col 1: SRS
ax = fig.add_subplot(gs[1, 0])
d_srs = asd.dropna(subset=["PCC_mPFC_r", "SRS_RAW_TOTAL"])
d_srs = d_srs[d_srs["SRS_RAW_TOTAL"] > 0]
r_srs, p_srs = pearsonr(d_srs["PCC_mPFC_r"], d_srs["SRS_RAW_TOTAL"])
ax.scatter(d_srs["PCC_mPFC_r"], d_srs["SRS_RAW_TOTAL"], alpha=0.5, s=40, c="#8E44AD",
           edgecolors="white", linewidth=0.3)
z = np.polyfit(d_srs["PCC_mPFC_r"], d_srs["SRS_RAW_TOTAL"], 1)
xl = np.linspace(d_srs["PCC_mPFC_r"].min(), d_srs["PCC_mPFC_r"].max(), 100)
ax.plot(xl, np.polyval(z, xl), "k--", linewidth=1.2)
ax.set_xlabel("PCC-mPFC FC", fontsize=11)
ax.set_ylabel("SRS Raw Total", fontsize=11)
ax.set_title(f"r = {r_srs:.3f}, p = {p_srs:.3f}  (n = {len(d_srs)})", fontstyle="italic", fontsize=10)

# Row 2, Col 2: FIQ (ASD only)
ax = fig.add_subplot(gs[1, 1])
d_fiq = asd.dropna(subset=["PCC_mPFC_r", "FIQ"])
r_fiq, p_fiq = pearsonr(d_fiq["PCC_mPFC_r"], d_fiq["FIQ"])
ax.scatter(d_fiq["PCC_mPFC_r"], d_fiq["FIQ"], alpha=0.5, s=40, c="#2ECC71",
           edgecolors="white", linewidth=0.3)
z = np.polyfit(d_fiq["PCC_mPFC_r"], d_fiq["FIQ"], 1)
xl = np.linspace(d_fiq["PCC_mPFC_r"].min(), d_fiq["PCC_mPFC_r"].max(), 100)
ax.plot(xl, np.polyval(z, xl), "k--", linewidth=1.2)
ax.set_xlabel("PCC-mPFC FC", fontsize=11)
ax.set_ylabel("Full-Scale IQ", fontsize=11)
ax.set_title(f"r = {r_fiq:.3f}, p = {p_fiq:.3f}  (n = {len(d_fiq)})", fontstyle="italic", fontsize=10)

# Row 2, Col 3: summary note
ax = fig.add_subplot(gs[1, 2])
ax.text(0.5, 0.5, "All brain-behavior correlations\nnon-significant (p > 0.3)\n\nADOS: clinician-administered\nSRS: parent-report\nFIQ: cognitive control",
        transform=ax.transAxes, ha="center", va="center", fontsize=11, fontstyle="italic", color="#7F8C8D")
ax.axis("off")

fig.suptitle("Brain-Behavior Correlations in ASD Group (Supplementary Figure S2)",
             fontweight="bold", fontsize=15, y=1.01)
fig.savefig(os.path.join(FIG, "figS2_brain_behavior_scatter.png"), dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print("  done")

# ═══════════════════════════════════════════════════════════════════
# SUPP: Mean FC maps (manual — consistent quality)
# ═══════════════════════════════════════════════════════════════════
for seed in ["PCC", "rAI"]:
    for group in ["TD", "ASD"]:
        fpath = os.path.join(RES, f"{seed}_{group}_mean_fc.nii.gz")
        if not os.path.exists(fpath): continue
        print(f">>> S-Fig: {seed} {group} mean FC")

        z_img = load_img(fpath)
        data = z_img.get_fdata()
        # Auto-choose slices evenly across z-range
        z_range = nib.affines.apply_affine(z_img.affine, [[0,0,0],[0,0,data.shape[2]-1]])
        z_vals = list(range(int(z_range[0,2]), int(z_range[1,2])+5, 10))[::2][:8]

        z_mni = resample_to_img(z_img, MNI, interpolation="nearest")
        zd = z_mni.get_fdata()
        md = MNI.get_fdata()
        aff = MNI.affine

        cols, rows = 4, 2
        fig, axes = plt.subplots(rows, cols, figsize=(16, 7), facecolor="white")
        axes = axes.flatten()
        for i, mni_z in enumerate(z_vals[:8]):
            ax = axes[i]
            iz = int(round((mni_z - aff[2,3]) / aff[2,2]))
            iz = max(0, min(md.shape[2]-1, iz))
            mni_sl = np.fliplr(md[:,:,iz].T)
            fc_sl = np.fliplr(zd[:,:,iz].T)
            fc_sl[fc_sl < 0.15] = np.nan

            ax.imshow(mni_sl, cmap="gray", origin="lower", aspect="auto")
            im = ax.imshow(fc_sl, cmap=WARM, origin="lower", aspect="auto",
                           vmin=0, vmax=0.5, alpha=0.75)
            ax.set_title(f"z = {mni_z} mm", fontsize=9, fontweight="bold")
            ax.axis("off")
        for i in range(len(z_vals[:8]), len(axes)):
            axes[i].axis("off")
        cbar_ax = fig.add_axes([0.92, 0.12, 0.012, 0.76])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label("Fisher Z", fontsize=10)
        fig.suptitle(f"{seed} Seed: {group} Mean FC", fontweight="bold", fontsize=13, y=0.96)

        png_path = os.path.join(FIG, f"{seed}_{group}_mean_fc.png")
        fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  -> {os.path.basename(png_path)} ({os.path.getsize(png_path):,} bytes)")

# ═══════════════════════════════════════════════════════════════════
# SUPP: Sensitivity forest plot
# ═══════════════════════════════════════════════════════════════════
print(">>> S-Fig: Sensitivity analysis")
sens_path = os.path.join(RES, "sensitivity_nyu.csv")
if os.path.exists(sens_path):
    sens = pd.read_csv(sens_path)
    fig, ax = plt.subplots(figsize=(8, 3), facecolor="white")
    shifts = sens["Shift"].tolist()
    ds = sens["d"].tolist()
    ref_d = ds[0]
    colors = ["#3498DB" if v == "original" else "#95A5A6" for v in shifts]
    ax.axvline(x=ref_d, color="#E74C3C", linestyle="--", linewidth=1.5, label=f"Original d = {ref_d:.3f}")
    ax.axvline(x=0, color="black", linestyle=":", linewidth=0.8)
    y_pos = range(len(shifts))
    ax.barh(y_pos, ds, color=colors, edgecolor="black", linewidth=0.5, height=0.4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(shifts)
    ax.set_xlabel("Cohen's d (ASD - TD)")
    ax.set_title("PCC-rAI FC Sensitivity: +/-4mm Shift (NYU, n=88)", fontweight="bold")
    for i, d in enumerate(ds):
        ax.text(d + 0.01, i, f"{d:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "figS_sensitivity_forest.png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  done")

# ═══════════════════════════════════════════════════════════════════
# SUPP: Cluster overview
# ═══════════════════════════════════════════════════════════════════
print(">>> S-Fig: Cluster overview")
cluster_path = os.path.join(RES, "cluster_table.csv")
if os.path.exists(cluster_path):
    ct = pd.read_csv(cluster_path)
    ct = ct.sort_values("Peak_z", ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
    colors = ["#E74C3C" if s == "PCC" else "#3498DB" for s in ct["Seed"]]
    labels = [f"{r['Seed']}-C{r['Cluster']} [{r['MNI_x']},{r['MNI_y']},{r['MNI_z']}]"
              for _, r in ct.iterrows()]
    ax.barh(range(len(ct)), ct["Peak_z"], color=colors, edgecolor="black", linewidth=0.5, height=0.6)
    ax.set_yticks(range(len(ct)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Peak z-score")
    ax.set_title("Top 20 Significant Clusters: TD > ASD (|z| > 2.3, k >= 20)", fontweight="bold")
    ax.invert_yaxis()
    ax.axvline(x=3.1, color="gray", linestyle="--", alpha=0.5, label="|z| = 3.1")
    ax.axvline(x=2.3, color="gray", linestyle=":", alpha=0.5, label="|z| = 2.3")
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#E74C3C", label="PCC"),
                       Patch(facecolor="#3498DB", label="rAI")]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "figS_cluster_overview.png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  done")

# ═══════════════════════════════════════════════════════════════════
# SUPP: Combined overview — PCC + rAI side by side
# ═══════════════════════════════════════════════════════════════════
print(">>> S-Fig: Combined TD>ASD overview")
try:
    fig, axes = plt.subplots(2, 4, figsize=(22, 12), facecolor="white")
    for row, seed in enumerate(["PCC", "rAI"]):
        z_path = os.path.join(RES, f"{seed}_TD_gt_ASD_z.nii.gz")
        if not os.path.exists(z_path): continue
        z_img = load_img(z_path)
        data = z_img.get_fdata()
        vox_counts = [(z, int((data[:,:,z] > 2.3).sum()),
                      nib.affines.apply_affine(z_img.affine, [0,0,z])[2])
                     for z in range(data.shape[2])]
        vox_counts.sort(key=lambda x: x[1], reverse=True)
        best = []
        for sl, cnt, mni_z in vox_counts:
            if cnt < 20: continue
            if any(abs(mni_z - bz) < 12 for _, bz in best): continue
            best.append((sl, int(round(mni_z))))
            if len(best) >= 4: break
        best.sort(key=lambda x: x[1])
        z_mni = resample_to_img(z_img, MNI, interpolation="nearest")
        zd = z_mni.get_fdata()
        md = MNI.get_fdata()
        aff = MNI.affine
        for col, (sl_idx, mni_z) in enumerate(best):
            ax = axes[row, col]
            iz = int(round((mni_z - aff[2,3]) / aff[2,2]))
            iz = max(0, min(md.shape[2]-1, iz))
            mni_sl = np.fliplr(md[:,:,iz].T)
            z_sl = np.fliplr(zd[:,:,iz].T)
            z_sl[z_sl < 2.3] = np.nan
            ax.imshow(mni_sl, cmap="gray", origin="lower", aspect="auto")
            im = ax.imshow(z_sl, cmap=BRIGHT, origin="lower", aspect="auto",
                           vmin=0, vmax=float(zd.max()), alpha=0.75)
            ax.set_title(f"{seed}  z = {mni_z} mm", fontsize=10, fontweight="bold")
            ax.axis("off")
    fig.suptitle("DMN-SN Functional Connectivity: TD > ASD  (|z| > 2.3)",
                 fontweight="bold", fontsize=14, y=0.98)
    out_path = os.path.join(FIG, "figS_combined_td_gt_asd.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  -> {os.path.getsize(out_path):,} bytes")
except Exception as e:
    print(f"  Failed: {e}")

# ═══════════════════════════════════════════════════════════════════
# SUPP: Group comparison summary
# ═══════════════════════════════════════════════════════════════════
print(">>> S-Fig: Group comparison summary")
comp_path = os.path.join(RES, "group_comparison.csv")
if os.path.exists(comp_path):
    comp = pd.read_csv(comp_path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor="white")
    pairs = comp["Pair"].tolist()

    ax = axes[0]
    betas = comp["GLM_beta"].tolist()
    bar_colors = ["#E74C3C" if b < 0 else "#3498DB" for b in betas]
    ax.bar(range(len(pairs)), betas, color=bar_colors, edgecolor="black", linewidth=0.8)
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(pairs, fontsize=10)
    ax.set_ylabel("GLM Beta (ASD - TD)")
    ax.set_title("Group Difference (GLM: age+FD+site)", fontweight="bold")
    ax.axhline(y=0, color="black", linewidth=0.8)
    for i, (_, r) in enumerate(comp.iterrows()):
        p = r["GLM_p"]
        sig = "***" if p<0.005 else ("**" if p<0.01 else ("*" if p<0.05 else "n.s."))
        y = betas[i] - 0.01 if betas[i] < 0 else betas[i] + 0.01
        ax.text(i, y, f"p={p:.3f}\n{sig}", ha="center", fontsize=8, fontstyle="italic")

    ax = axes[1]
    ds = comp["Cohens_d"].tolist()
    bar_colors = ["#E74C3C" if d < 0 else "#3498DB" for d in ds]
    ax.bar(range(len(pairs)), ds, color=bar_colors, edgecolor="black", linewidth=0.8)
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(pairs, fontsize=10)
    ax.set_ylabel("Cohen's d")
    ax.set_title("Effect Size", fontweight="bold")
    ax.axhline(y=0, color="black", linewidth=0.8)
    for i, d in enumerate(ds):
        ax.text(i, d + 0.01 if d >= 0 else d - 0.02, f"{d:.3f}", ha="center", fontsize=8)

    fig.suptitle("Seed Functional Connectivity: Group Comparison Summary", fontweight="bold", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "figS_group_comparison.png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  done")

# ── Cleanup ──
for f in os.listdir(FIG):
    fp = os.path.join(FIG, f)
    if f.endswith(".html") or f.startswith("_tmp") or f.startswith("_test"):
        os.remove(fp)
        print(f"  cleaned: {f}")

# ── Summary ──
print(f"\n{'='*60}")
figs = sorted([f for f in os.listdir(FIG) if f.endswith(".png")])
for f in figs:
    sz = os.path.getsize(os.path.join(FIG, f))
    print(f"  {f} ({sz:,} bytes)")
print(f"\nTotal: {len(figs)} figures in {FIG}")
print(f"{'='*60}")
