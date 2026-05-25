"""
Generate 4 main figures for Autism Research manuscript.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os, warnings
warnings.filterwarnings('ignore')

ABIDE_DIR = 'E:/ABIDE项目'
OUT = os.path.join(ABIDE_DIR, 'results', 'manuscript_figures')
os.makedirs(OUT, exist_ok=True)

# ── Style ──
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'font.size': 8,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})
ASD_COLOR = '#D62728'
TD_COLOR = '#1F77B4'
ASD_ALPHA = 0.4
TD_ALPHA = 0.3


def fig1_group_boxplot():
    """Figure 1: PCC-mPFC Fisher Z by group (boxplot + scatter)."""
    fc = pd.read_csv(os.path.join(ABIDE_DIR, 'timeseries', 'seed_connectivity.csv'),
                     dtype={'SUB_ID': str})
    fc = fc[['SUB_ID', 'SITE_ID', 'DX', 'PCC_mPFC_z']].dropna()

    fig, ax = plt.subplots(figsize=(3.2, 4.2))

    groups = ['TD', 'ASD']
    positions = [1, 2]
    colors = [TD_COLOR, ASD_COLOR]

    for i, (grp, pos) in enumerate(zip(groups, positions)):
        vals = fc[fc['DX'] == grp]['PCC_mPFC_z'].values
        # Scatter
        np.random.seed(42 + i)
        jitter = np.random.normal(pos, 0.04, size=len(vals))
        ax.scatter(jitter, vals, s=4, alpha=0.15, color=colors[i], edgecolors='none', zorder=1)
        # Box
        bp = ax.boxplot(vals, positions=[pos], widths=0.5,
                        patch_artist=True, showfliers=False, zorder=3)
        bp['boxes'][0].set_facecolor(colors[i])
        bp['boxes'][0].set_alpha(0.5)
        bp['medians'][0].set_color('black')
        bp['medians'][0].set_linewidth(1.2)

    # Mean markers
    for grp, pos, c in zip(groups, positions, colors):
        mu = fc[fc['DX'] == grp]['PCC_mPFC_z'].mean()
        ax.scatter(pos, mu, marker='D', s=40, color=c, edgecolors='black',
                   linewidths=0.8, zorder=4, label=f'{grp} mean')

    ax.set_xticks(positions)
    ax.set_xticklabels(['TD\n(n=308)', 'ASD\n(n=291)'], fontsize=8)
    ax.set_ylabel('PCC-mPFC Functional Connectivity (Fisher Z)', fontsize=8)
    ax.set_title('A', loc='left', fontweight='bold', fontsize=11)

    # Stats annotation
    td_vals = fc[fc['DX'] == 'TD']['PCC_mPFC_z']
    asd_vals = fc[fc['DX'] == 'ASD']['PCC_mPFC_z']
    ax.text(1.5, ax.get_ylim()[1] + 0.02,
            f'β = −0.054, p = 0.030\nCohen\'s d = −0.14',
            ha='center', va='bottom', fontsize=7.5,
            bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'))

    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig1_group_comparison.png'))
    plt.close(fig)
    print('Fig1 done')


def fig2_sensitivity_coefficient():
    """Figure 2: Coefficient plot — beta with 95% CI across models."""
    rs = pd.read_csv(os.path.join(ABIDE_DIR, 'results', 'results_summary.csv'))
    # Filter PCC-mPFC rows with numeric CIs
    keep = ['Primary N=599', 'Coverage N=596', 'Balanced N=574',
            'ABIDE I N=274', 'ABIDE II N=325', 'Cov+Bal N=572']
    rows = rs[rs['label'].isin(keep)].copy()
    rows = rows.sort_values('beta')

    labels = [r.replace(' N=', '; n=') for r in rows['label']]
    betas = rows['beta'].values
    lo = rows['ci_lower'].values
    hi = rows['ci_upper'].values
    colors_sig = ['#D62728' if p < 0.05 else '#555555' for p in rows['p']]

    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    y = np.arange(len(rows))

    for i in range(len(rows)):
        ax.hlines(y[i], lo[i], hi[i], color=colors_sig[i], linewidth=1.5, zorder=1)
        ax.scatter(betas[i], y[i], color=colors_sig[i], s=40,
                   edgecolors='white', linewidths=0.5, zorder=2, clip_on=False)

    ax.axvline(0, color='black', linewidth=0.6, linestyle='-', zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel('β (95% CI)', fontsize=8.5)
    ax.set_title('B', loc='left', fontweight='bold', fontsize=11)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='y', length=0)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig2_sensitivity_coefficient.png'))
    plt.close(fig)
    print('Fig2 done')


def fig3_forest():
    """Figure 3: Forest plot of site-wise effects."""
    sw = pd.read_csv(os.path.join(ABIDE_DIR, 'results', 'site_wise_effects.csv'))
    sw = sw.sort_values('d')

    fig, ax = plt.subplots(figsize=(4.8, 4.5))
    y = np.arange(len(sw))

    for i, (_, r) in enumerate(sw.iterrows()):
        ax.hlines(i, r['ci_low'], r['ci_high'], color='gray', linewidth=1.2, zorder=1)
        c = ASD_COLOR if r['d'] < 0 else TD_COLOR
        ax.scatter(r['d'], i, color=c, s=35, edgecolors='white', linewidths=0.5, zorder=2)

    # RE meta line
    ax.axvline(-0.287, color='#D62728', linewidth=1.5, linestyle='--',
               alpha=0.7, zorder=0)
    ax.axvline(0, color='black', linewidth=0.6, linestyle='-', zorder=0)

    ax.set_yticks(y)
    ax.set_yticklabels(sw['Site'], fontsize=6.5)
    ax.set_xlabel("Cohen's d (95% CI)", fontsize=8.5)
    ax.set_title('C', loc='left', fontweight='bold', fontsize=11)

    ax.text(0.98, 0.98, 'RE: d = −0.287, p = 0.007, I² = 24.6%',
            transform=ax.transAxes, ha='right', va='top', fontsize=7,
            bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'))

    # Dataset annotation
    ylim = ax.get_ylim()
    ax.hlines(ylim[0] + 0.3, -0.02, -0.02, color='gray', linewidth=8, alpha=0.15)

    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='y', length=0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig3_forest_plot.png'))
    plt.close(fig)
    print('Fig3 done')


def fig4_age_interaction():
    """Figure 4: Age × DX interaction scatter."""
    ai = pd.read_csv(os.path.join(ABIDE_DIR, 'results', 'age_interaction.csv'))
    pm = ai[(ai['model'].str.lower().str.contains('age')) & (ai['term'] == 'dx_age')]
    inter_beta = pm['beta'].values[0]
    inter_p = pm['p'].values[0]

    fc = pd.read_csv(os.path.join(ABIDE_DIR, 'timeseries', 'seed_connectivity.csv'),
                     dtype={'SUB_ID': str}, usecols=['SUB_ID', 'SITE_ID', 'DX', 'PCC_mPFC_z'])
    demo = pd.read_csv(r'C:\Users\22803\Desktop\代码\ABIDE_CPAC_filt_noglobal\code\combined_subjects.csv',
                       dtype={'SUB_ID': str}, usecols=['SUB_ID', 'SITE_ID', 'DX', 'AGE_AT_SCAN'])
    df = fc.merge(demo, on=['SUB_ID', 'SITE_ID', 'DX'], how='left')

    fig, ax = plt.subplots(figsize=(4.2, 3.5))

    for grp, c, alp in [('TD', TD_COLOR, TD_ALPHA), ('ASD', ASD_COLOR, ASD_ALPHA)]:
        sub = df[df['DX'] == grp].dropna(subset=['PCC_mPFC_z', 'AGE_AT_SCAN'])
        ax.scatter(sub['AGE_AT_SCAN'], sub['PCC_mPFC_z'], s=6, alpha=alp,
                   color=c, edgecolors='none', zorder=1)
        # Regression
        x = sub['AGE_AT_SCAN'].values
        y = sub['PCC_mPFC_z'].values
        if len(x) > 5:
            m, b = np.polyfit(x, y, 1)
            x_sorted = np.sort(x)
            ax.plot(x_sorted, m * x_sorted + b, color=c, linewidth=1.5,
                    label=grp, zorder=2)

    ax.set_xlabel('Age (years)', fontsize=8.5)
    ax.set_ylabel('PCC-mPFC (Fisher Z)', fontsize=8.5)
    ax.set_title('D', loc='left', fontweight='bold', fontsize=11)

    ax.text(0.98, 0.05, f'Age × DX: β = {inter_beta:.4f}, p = {inter_p:.3f}',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=7.5,
            bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'))
    ax.legend(frameon=False, fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig4_age_interaction.png'))
    plt.close(fig)
    print('Fig4 done')


if __name__ == '__main__':
    fig1_group_boxplot()
    fig2_sensitivity_coefficient()
    fig3_forest()
    fig4_age_interaction()
    print(f'All figures saved to {OUT}')
