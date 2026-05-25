"""
VIF analysis for the primary model.
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os

ABIDE_DIR = 'E:/ABIDE项目'
FC_FILE = os.path.join(ABIDE_DIR, 'timeseries', 'seed_connectivity.csv')
FD_FILE = os.path.join(ABIDE_DIR, 'results', 'subjects_with_corrected_fd.csv')

fc = pd.read_csv(FC_FILE, dtype={'SUB_ID': str})
fd_data = pd.read_csv(FD_FILE, dtype={'SUB_ID': str})

# Merge
df = fc.merge(fd_data[['SUB_ID', 'AGE_AT_SCAN', 'DX', 'func_mean_fd', 'SITE_ID']],
              on=['SUB_ID', 'SITE_ID', 'DX'], how='left')

df['DX_binary'] = (df['DX'] == 'ASD').astype(int)

# Check N
print(f'Total N with all data: {df[["PCC_mPFC_z", "DX_binary", "AGE_AT_SCAN", "func_mean_fd"]].dropna().shape[0]}')

# Prep model matrix
X = pd.get_dummies(df[['DX_binary', 'AGE_AT_SCAN', 'func_mean_fd', 'SITE_ID']],
                    columns=['SITE_ID'], drop_first=True)
X = sm.add_constant(X).astype(float)
y = df['PCC_mPFC_z']

mask = y.notna() & (X.notna().all(axis=1))
X_clean = X[mask]
y_clean = y[mask]

print(f'Model N: {len(X_clean)}')

model = sm.OLS(y_clean, X_clean).fit()

# VIF
vif_data = pd.DataFrame()
vif_data['variable'] = X_clean.columns
vif_data['VIF'] = [variance_inflation_factor(X_clean.values, i) for i in range(X_clean.shape[1])]
vif_data['tolerance'] = 1 / vif_data['VIF']

# Show top VIF
vif_sorted = vif_data.sort_values('VIF', ascending=False)
print('\nTop 15 VIF values:')
print(vif_sorted.head(15).to_string())

# Key predictors only
print(f'\nKey predictors:')
for var in ['const', 'DX_binary', 'AGE_AT_SCAN', 'func_mean_fd']:
    row = vif_data[vif_data['variable'] == var]
    if len(row) > 0:
        print(f'  {var}: VIF = {row["VIF"].values[0]:.2f}')

print(f'\nCondition number: {model.condition_number:.0f}')
print(f'DX_binary: β={model.params["DX_binary"]:.5f}, p={model.pvalues["DX_binary"]:.5f}')
