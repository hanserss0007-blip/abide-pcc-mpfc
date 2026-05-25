"""
Sensitivity analysis: ABIDE II participants with observed FD data only (N=76).
Addresses reviewer concern that site-mean FD imputation for 249/325 ABIDE II
participants may bias the null ABIDE II result.
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import os

ABIDE_DIR = 'E:/ABIDE项目'
FC_FILE = os.path.join(ABIDE_DIR, 'timeseries', 'seed_connectivity.csv')
REAL_FD_FILE = os.path.join(ABIDE_DIR, 'results', 'abide2_real_fd.csv')
OUT_FILE = os.path.join(ABIDE_DIR, 'results', 'abide2_realfd_sensitivity.csv')

# Load data
fc = pd.read_csv(FC_FILE, dtype={'SUB_ID': str})
real_fd = pd.read_csv(REAL_FD_FILE, dtype={'SUB_ID': str})

# Merge real FD values
fc = fc.merge(real_fd[['SUB_ID', 'real_mean_fd']], on='SUB_ID', how='left')

# Identify ABIDE II participants with real FD
abide2 = fc[fc['SITE_ID'].str.contains('ABIDEII', na=False)].copy()
abide2_real = abide2[abide2['real_mean_fd'].notna()].copy()
abide2_imputed = abide2[abide2['real_mean_fd'].isna()].copy()

print(f'ABIDE II total: {len(abide2)}')
print(f'  Real FD: {len(abide2_real)}')
print(f'  Imputed FD: {len(abide2_imputed)}')

# Check site distribution in real FD subset
site_dist = abide2_real.groupby(['SITE_ID', 'DX']).size().unstack(fill_value=0)
print('\nSite distribution (real FD):')
print(site_dist)

# OLS model on ABIDE II real FD subset
# PCC_mPFC_z ~ DX + age + real_mean_fd + site
keep_cols = ['SUB_ID', 'SITE_ID', 'DX', 'AGE_AT_SCAN', 'PCC_mPFC_z', 'real_mean_fd']
demo = pd.read_csv(os.path.join(ABIDE_DIR, 'results', 'demographics_table.csv'))
# Actually, AGE_AT_SCAN may not be in seed_connectivity.csv
# Let me check available columns
print(f'\nFC columns: {fc.columns.tolist()}')

# Check if AGE_AT_SCAN is in fc
if 'AGE_AT_SCAN' not in fc.columns:
    print('AGE_AT_SCAN not in fc, loading from combined_subjects...')
    demo_path = r'C:\Users\22803\Desktop\代码\ABIDE_CPAC_filt_noglobal\code\combined_subjects.csv'
    demo = pd.read_csv(demo_path, dtype={'SUB_ID': str})
    abide2_real = abide2_real.merge(demo[['SUB_ID', 'AGE_AT_SCAN']], on='SUB_ID', how='left')

# Prepare model data
model_data = abide2_real[['DX', 'AGE_AT_SCAN', 'real_mean_fd', 'SITE_ID', 'PCC_mPFC_z']].dropna()
print(f'\nModel N: {len(model_data)}')
print(f'  ASD: {(model_data["DX"]=="ASD").sum()}')
print(f'  TD: {(model_data["DX"]=="TD").sum()}')

# Encode DX
model_data = model_data.copy()
model_data['DX_binary'] = (model_data['DX'] == 'ASD').astype(int)

# Check number of unique sites
n_sites = model_data['SITE_ID'].nunique()
print(f'  Sites: {n_sites}')

# Run OLS
X = pd.get_dummies(model_data[['DX_binary', 'AGE_AT_SCAN', 'real_mean_fd', 'SITE_ID']],
                    columns=['SITE_ID'], drop_first=True)
X = sm.add_constant(X)
y = model_data['PCC_mPFC_z']

model = sm.OLS(y, X.astype(float)).fit()
print(f'\nPrimary model (ABIDE II real FD, N={len(model_data)}):')
print(model.summary())

# Extract key result
beta = model.params['DX_binary']
ci = model.conf_int().loc['DX_binary']
p = model.pvalues['DX_binary']

result = {
    'label': f'ABIDE II real FD N={len(model_data)}',
    'N': len(model_data),
    'ASD': int((model_data['DX_binary'] == 1).sum()),
    'TD': int((model_data['DX_binary'] == 0).sum()),
    'beta': round(beta, 5),
    'p': round(p, 5),
    'ci_lower': round(ci[0], 5),
    'ci_upper': round(ci[1], 5),
}
result_df = pd.DataFrame([result])
print(f'\nResult:\n{result_df.to_string()}')
result_df.to_csv(OUT_FILE, index=False)
print(f'\nSaved to {OUT_FILE}')
