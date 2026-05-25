"""
Step 16: Age x Diagnosis interaction on PCC-mPFC FC.
Tests whether the ASD<TD effect varies with age (linear and quadratic).
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

# ── Load data ──
fc = pd.read_csv(FC_CSV, dtype={"SUB_ID": str})
demo = pd.read_csv(SUB_CSV, dtype={"SUB_ID": str})[["SUB_ID", "AGE_AT_SCAN", "func_mean_fd", "SITE_ID", "DX", "dataset"]]
df = fc.merge(demo, on=["SUB_ID", "SITE_ID", "DX", "dataset"], how="left")
df = df.dropna(subset=[FC_COL, "AGE_AT_SCAN", "func_mean_fd", "SITE_ID"])
df["age"] = pd.to_numeric(df["AGE_AT_SCAN"], errors="coerce")
df["fd"] = pd.to_numeric(df["func_mean_fd"], errors="coerce")
df = df.dropna(subset=["age", "fd"]).reset_index(drop=True)

# Mean-center age
df["age_c"] = df["age"] - df["age"].mean()
df["age_c2"] = df["age_c"] ** 2
df["dx_num"] = (df["DX"] == "ASD").astype(int)

n_total = len(df)
n_asd = (df["DX"] == "ASD").sum()
n_td = (df["DX"] == "TD").sum()
print(f"Total N = {n_total} (ASD={n_asd}, TD={n_td})")

import statsmodels.api as sm
df["intercept"] = 1.0

def run_model(X, y, label):
    X = X.astype(float)
    m = sm.OLS(y, X).fit()
    print(f"\n{'='*55}")
    print(label)
    print(f"{'='*55}")
    print(f"  N = {len(y)}")
    print(m.summary())
    return m

# Model 1: main effect (DX + age + FD + site)
X1 = pd.concat([df[["intercept", "dx_num", "age_c", "fd"]],
                pd.get_dummies(df["SITE_ID"], prefix="site", drop_first=True)], axis=1)
y = df[FC_COL].values.astype(float)
m1 = run_model(X1, y, "Main: FC ~ DX + age + FD + site")

# Model 2: age x DX interaction (linear)
X2 = pd.concat([df[["intercept", "dx_num", "age_c", "fd"]],
                pd.get_dummies(df["SITE_ID"], prefix="site", drop_first=True)], axis=1)
X2["dx_age"] = df["dx_num"] * df["age_c"]
m2 = run_model(X2, y, "Interaction: FC ~ DX * age + FD + site")

# Model 3: quadratic age x DX interaction
X3 = pd.concat([df[["intercept", "dx_num", "age_c", "age_c2", "fd"]],
                pd.get_dummies(df["SITE_ID"], prefix="site", drop_first=True)], axis=1)
X3["dx_age"] = df["dx_num"] * df["age_c"]
X3["dx_age2"] = df["dx_num"] * df["age_c2"]
m3 = run_model(X3, y, "Quadratic: FC ~ DX * (age + age^2) + FD + site")

# Simple slopes at age mean and +/- 1SD
age_sd = df["age"].std()
age_mean = df["age"].mean()
print(f"\n{'='*55}")
print("Simple Slopes (Model 2): ASD<TD effect at different ages")
print(f"{'='*55}")
for label, age_at in [("-1SD", age_mean - age_sd), ("Mean", age_mean), ("+1SD", age_mean + age_sd)]:
    # At this age, the ASD<TD effect = dx_num_coef + dx_age_coef * (age_at - mean_age)
    effect = m2.params["dx_num"] + m2.params["dx_age"] * (age_at - age_mean)
    print(f"  At age {label} ({age_at:.1f}y): d = {effect:.4f}")

# ── Save results summary ──
results = []
for name, m in [("Main", m1), ("Age x DX", m2), ("Quadratic", m3)]:
    for var in m.params.index:
        if var == "intercept":
            continue
        results.append({
            "model": name, "term": var,
            "beta": round(m.params[var], 5),
            "SE": round(m.bse[var], 5),
            "t": round(m.tvalues[var], 4),
            "p": round(m.pvalues[var], 5),
            "ci_lower": round(m.conf_int().loc[var, 0], 5),
            "ci_upper": round(m.conf_int().loc[var, 1], 5),
        })

rdf = pd.DataFrame(results)
rdf.to_csv(os.path.join(OUT_DIR, "age_interaction.csv"), index=False)
print(f"\nResults saved.")

# ── Plot: FC vs age by group ──
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 5))
colors = {"ASD": "crimson", "TD": "steelblue"}
for grp, grp_df in df.groupby("DX"):
    ax.scatter(grp_df["age"], grp_df[FC_COL], c=colors[grp], alpha=0.3, s=15, label=grp)
    # Regression line from Model 2 (partial, holding FD+site at mean)
    # Predicted FC at each age for this group
    ages = np.linspace(df["age"].min(), df["age"].max(), 100)
    ages_c = ages - age_mean
    if grp == "ASD":
        pred = (m2.params["intercept"] + m2.params["dx_num"] +
                m2.params["age_c"] * ages_c +
                m2.params["dx_age"] * ages_c +
                m2.params["fd"] * df["fd"].mean())
    else:
        pred = (m2.params["intercept"] +
                m2.params["age_c"] * ages_c +
                m2.params["fd"] * df["fd"].mean())
    ax.plot(ages, pred, color=colors[grp], lw=2)

ax.set_xlabel("Age (years)", fontsize=11)
ax.set_ylabel("PCC-mPFC FC (Fisher Z)", fontsize=11)
interaction_p = m2.pvalues.get("dx_age", 1)
ax.set_title(f"Age x Diagnosis Interaction\np(interaction) = {interaction_p:.4f}", fontsize=11, fontweight="bold")
ax.legend(fontsize=10)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_age_interaction.png"), dpi=300)
print(f"Plot saved.")
print("Done.")
