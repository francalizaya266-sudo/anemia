# Sixth review: the geometric set (337 distinct images) as the principal set.
# A  Discrimination, metrics at 0.5, Brier and Brier skill, calibration slope and calibration-in-the-large for the
#    classical models (averaged out-of-fold probabilities of review2_geo.py) and for every CNN run on the geometric set.
# B  Decision curves on the geometric and 471-image sets: net benefit of the model minus that of the best default strategy.
# C  (The stage-4 funnel is computed in codigo/review5/review5_post.py, key 'funnel' of resultados/review5/post.json,
#    which separates the 9,891 candidates from the 1,000 random control pairs.)
# D  Figures S4 (calibration) and S6 (decision curves) redrawn on the geometric set.
import json, glob, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression
REPO = '/home/claude/anemia/'; OUT = '/home/claude/exp/out/review6/'; FIG = '/home/claude/figs_new/'
R = {}
geo = json.load(open(REPO + 'resultados/review2/geo/review_geo.json')); oofg = json.load(open(REPO + 'resultados/review2/geo/oof_geo.json'))
mg = pd.read_csv(REPO + 'codigo/meta_groups_geo.csv').set_index('image_id')
yg = mg.loc[geo['images'], 'anemia'].values.astype(int)
def metr(y, p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4); d = (p >= 0.5).astype(int); prev = y.mean()
    brier = float(np.mean((p - y) ** 2)); lp = np.log(p / (1 - p))
    lr = LogisticRegression(C=1e6).fit(lp.reshape(-1, 1), y)
    return dict(AUC=float(roc_auc_score(y, p)), F1=float(f1_score(y, d)), Acc=float((d == y).mean()), Sens=float(d[y == 1].mean()), Spec=float(1 - d[y == 0].mean()),
                Brier=brier, BSS=float(1 - brier / (prev * (1 - prev))), cal_slope=float(lr.coef_[0][0]), cal_in_large=float(np.log(prev / (1 - prev)) - np.log(p.mean() / (1 - p.mean()))))
def nb(y, p, t): n = len(y); return ((p >= t) & (y == 1)).sum() / n - ((p >= t) & (y == 0)).sum() / n * t / (1 - t)
def dca(y, p):
    ts = np.round(np.arange(0.05, 0.951, 0.05), 2); prev = y.mean(); out = {}
    for t in ts:
        all_ = prev - (1 - prev) * t / (1 - t); out[str(t)] = dict(model=float(nb(y, p, t)), treat_all=float(all_), gain=float(nb(y, p, t) - max(all_, 0)))
    pos = [float(t) for t in ts if out[str(t)]['gain'] > 0.005]
    return dict(curve=out, gain_range=[min(pos), max(pos)] if pos else None)
# A classical models on the geometric set
R['classical_geo'] = {}
for k, e in geo['dedup'].items():
    p = np.array(oofg[k]['geo349']); m = metr(yg, p)
    R['classical_geo'][k] = dict(AUC_avg=e['AUC_avgprob'], AUC_avg_ci=e['AUC_avgprob_ci'], AUC_rep_mean=e['AUC_rep_mean'], AUC_rep_sd=e['AUC_rep_sd'], **{kk: m[kk] for kk in m if kk != 'AUC'})
# CNN runs on the geometric set (original recipe 'geo', tuned recipe 'tunedgeo')
R['cnn_geo'] = {}
for task in ('geo', 'tunedgeo'):
    for f in sorted(glob.glob(REPO + 'resultados/review2/gpu2/cnn_*_%s_s*.json' % task)):
        d = json.load(open(f)); m = d['res']['model']; s = d['res']['seed']
        ids = d['ids']; y = mg.loc[ids, 'anemia'].values.astype(int); assert (y == np.array(d['y'])).all()
        R['cnn_geo'].setdefault(f'{m}|{task}', {})[str(s)] = metr(y, np.array(d['oof']))
for k, v in R['cnn_geo'].items():
    a = [x['AUC'] for x in v.values()]; v['summary'] = dict(n_seeds=len(a), AUC_mean=float(np.mean(a)), AUC_sd=float(np.std(a, ddof=1)) if len(a) > 1 else None,
                                                             **{kk: float(np.mean([x[kk] for x in v.values() if isinstance(x, dict) and kk in x])) for kk in ('F1', 'Acc', 'Sens', 'Spec', 'Brier', 'BSS')})
# B decision curves
R['dca'] = {}
for k in ('RF (color)', 'XGBoost (color)'):
    R['dca']['geo|' + k] = dca(yg, np.array(oofg[k]['geo349']))
oof471 = json.load(open('/home/claude/exp/out/review/oof_review.json')); y471 = np.array(json.load(open('/home/claude/exp/out/classic_oof.json'))['y'])
for k in ('RF (color)', 'XGBoost (color)'):
    R['dca']['471|' + k] = dca(y471, np.array(oof471[k]['dedup471']))
json.dump(R, open(OUT + 'geo_primary.json', 'w'), indent=1)
# D figures
plt.rcParams.update({'pdf.fonttype': 42, 'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#9a9a9a', 'savefig.bbox': 'tight', 'figure.dpi': 200})
C = ['#2a78d6', '#eb6834']; M = {'RF (color)': 'Random Forest', 'XGBoost (color)': 'XGBoost'}
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d; return c - h, c + h
fig, axs = plt.subplots(1, 2, figsize=(6.3, 3.1), sharey=True)
for ax, k, col in zip(axs, ['RF (color)', 'XGBoost (color)'], C):
    p = np.array(oofg[k]['geo349']); q = np.quantile(p, np.linspace(0, 1, 11)); q[0] -= 1e-9
    for lo, hi in zip(q[:-1], q[1:]):
        m = (p > lo) & (p <= hi); n = m.sum(); kk = yg[m].sum(); w = wilson(kk, n); ax.errorbar(p[m].mean(), kk / n, yerr=[[kk / n - w[0]], [w[1] - kk / n]], fmt='o', color=col, ms=4, capsize=2, lw=0.9)
    ax.plot([0, 1], [0, 1], ls=':', color='#9a9a9a'); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xlabel('predicted probability of anaemia'); ax.set_aspect('equal')
    e = R['classical_geo'][k]; ax.set_title(f"{M[k]}: slope {e['cal_slope']:.2f}, CITL {e['cal_in_large']:.2f}", fontsize=8, loc='left')
axs[0].set_ylabel('observed fraction anaemic'); plt.savefig(FIG + 'figS4_calibration.pdf'); plt.savefig(FIG + 'figS4_calibration.png'); plt.close()
fig, ax = plt.subplots(figsize=(4.8, 3.3)); pts = np.linspace(0.02, 0.95, 94)
for k, col in zip(['RF (color)', 'XGBoost (color)'], C):
    p = np.array(oofg[k]['geo349']); ax.plot(pts, [nb(yg, p, t) for t in pts], color=col, lw=1.8, label=M[k] + ', geometric set')
    p = np.array(oof471[k]['dedup471']); ax.plot(pts, [nb(y471, p, t) for t in pts], color=col, lw=1.0, ls='--', label=M[k] + ', 471-image set')
prev = yg.mean(); ax.plot(pts, [prev - (1 - prev) * t / (1 - t) for t in pts], color='#9a9a9a', lw=1.2, label='treat all (geometric set)')
ax.axhline(0, color='#9a9a9a', lw=1.2, ls=':', label='treat none')
ax.set_ylim(-0.05, 0.6); ax.set_xlabel('threshold probability'); ax.set_ylabel('net benefit'); ax.legend(frameon=False, fontsize=6.5)
plt.savefig(FIG + 'figS6_decision_curve.pdf'); plt.savefig(FIG + 'figS6_decision_curve.png'); plt.close()
for k, v in R['classical_geo'].items(): print(k, {a: round(b, 3) for a, b in v.items() if isinstance(b, float)})
for k, v in R['cnn_geo'].items(): print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v['summary'].items()})
for k, v in R['dca'].items(): print(k, v['gain_range'], {t: round(v['curve'][t]['gain'], 3) for t in ('0.1', '0.2', '0.3', '0.5')})
