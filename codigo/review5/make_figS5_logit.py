# Supplementary Fig. S5 redrawn with the random-effects estimate and prediction interval pooled on the logit scale
# (review5_post.py), so that the prediction interval stays inside (0, 1). Same layout as codigo/review/figs/make_figs_b.py.
import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT = '/home/claude/figs_new/'; RV = '/home/claude/exp/out/review/'
plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#9a9a9a', 'xtick.color': '#52514e', 'ytick.color': '#52514e', 'savefig.bbox': 'tight', 'figure.dpi': 200})
C = ['#2a78d6', '#eb6834', '#1baf7a']
R = json.load(open(RV + 'review_results.json')); R.update(json.load(open(RV + 'review_results_b.json')))
L = R['loho']; per = sorted(L['per_centre'], key=lambda x: x['AUC'])
RM = json.load(open('/home/claude/exp/out/review5/post.json'))['loho_logit']['471']
fig, ax = plt.subplots(figsize=(5.6, 3.2))
for i, x in enumerate(per):
    ax.errorbar(x['AUC'], i, xerr=[[x['AUC'] - x['ci'][0]], [x['ci'][1] - x['AUC']]], fmt='s', color=C[0], ms=4 + 6 * np.sqrt(x['n'] / 100), capsize=2, lw=0.9)
ax.errorbar(L['pooled_AUC'], len(per), xerr=[[L['pooled_AUC'] - L['pooled_ci'][0]], [L['pooled_ci'][1] - L['pooled_AUC']]], fmt='o', color=C[1], ms=5, capsize=2, lw=0.9)
yk = len(per) + 1; ax.plot(RM['pred_int'], [yk, yk], color=C[2], lw=5, alpha=0.25, solid_capstyle='butt')
ax.errorbar(RM['mu'], yk, xerr=[[RM['mu'] - RM['ci_hk'][0]], [RM['ci_hk'][1] - RM['mu']]], fmt='D', color=C[2], ms=5, capsize=2, lw=0.9)
ax.set_yticks(range(len(per) + 2))
ax.set_yticklabels([f"{x['hospital']} (n = {x['n']})" for x in per] + ['Pooled out-of-facility predictions', f"Random effects on logit(AUC), REML + Hartung-Knapp\n(k = {RM['k']}; shaded: 95 % prediction interval)"])
ax.axvline(0.5, ls=':', color='#9a9a9a'); ax.set_xlabel('AUC (95 % CI), Random Forest trained on the other facilities'); ax.set_xlim(0.4, 1.0); ax.grid(axis='x', color='#eeeeea', lw=0.8)
plt.savefig(OUT + 'figS5_loho_forest.pdf'); plt.savefig(OUT + 'figS5_loho_forest.png'); plt.close(); print('ok', RM['mu'], RM['pred_int'])
