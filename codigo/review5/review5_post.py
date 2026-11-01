# Fifth review, statistics on the outputs of review5_cpu.py, review5_synthetic.py and the stored stage-4 verification.
import sys, json, glob; sys.path.insert(0, '/home/claude/exp')
import numpy as np, pandas as pd
from scipy import stats, optimize
from sklearn.metrics import roc_auc_score
from common import load_meta, dedup, SEED
REPO = '/home/claude/anemia/'; OUT = '/home/claude/exp/out/review5/'
R5 = json.load(open(OUT + 'review5.json')); SY = json.load(open(OUT + 'synthetic.json')); P = {}
def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]: j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1; i = j
    T2 = np.empty(N); T2[J] = T; return T2
def delong(y, p1, p2):
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]; m = len(pos); n = len(neg); preds = np.vstack([p1, p2])
    tx = np.array([_midrank(preds[r, pos]) for r in range(2)]); ty = np.array([_midrank(preds[r, neg]) for r in range(2)]); tz = np.array([_midrank(np.concatenate([preds[r, pos], preds[r, neg]])) for r in range(2)])
    aucs = tz[:, :m].sum(1) / m / n - (m + 1.0) / (2.0 * n); v01 = (tz[:, :m] - tx) / n; v10 = 1.0 - (tz[:, m:] - ty) / m
    S = np.cov(v01) / m + np.cov(v10) / n; L = np.array([1, -1]); z = (aucs[0] - aucs[1]) / np.sqrt(L @ S @ L)
    return float(aucs[0] - aucs[1]), float(z), float(2 * stats.norm.sf(abs(z)))
def paired_boot_diff(y, p1, p2, B=2000, seed=SEED):
    rng = np.random.default_rng(seed); n = len(y); d = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if len(np.unique(y[i])) < 2: continue
        d.append(roc_auc_score(y[i], p1[i]) - roc_auc_score(y[i], p2[i]))
    d = np.array(d); return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], float(2 * min((d <= 0).mean(), (d >= 0).mean()))
def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = np.zeros(m); run = 0
    for k, i in enumerate(o): run = max(run, (m - k) * ps[i]); adj[i] = min(1.0, run)
    return adj.tolist()
# ---------- 1. stage-4 near misses and the false-link bound ----------
mfull = pd.read_csv(REPO + 'codigo/meta_groups.csv'); mg = pd.read_csv(REPO + 'codigo/meta_groups_geo.csv').set_index('image_id')
reps = mfull.sort_values('image_id').drop_duplicates('grp').reset_index(drop=True); y4 = reps.anemia.values; g4 = mg.loc[reps.image_id, 'grp'].values
a = np.load(REPO + 'resultados/review2/geomverify.npy'); i, j = a[:, 0].astype(int), a[:, 1].astype(int); k, cov, ncc, s = a[:, 2], a[:, 3], a[:, 4], a[:, 6]
acc = (k >= 10) & (s >= 0.5) & (s <= 2) & (cov >= 0.3) & (ncc >= 0.7); cand = k >= 10; conc = y4[i] == y4[j]; same = g4[i] == g4[j]
rej = cand & ~acc; rejd = rej & ~same
def cp_upper(x, n): return float(stats.beta.ppf(0.95, x + 1, n - x)) if x < n else 1.0
nm = {}
for tag, m in (('rejected_all', rejd), ('rejected_inl20', rejd & (k >= 20)), ('rejected_inl30', rejd & (k >= 30)), ('random_low', ~cand)):
    nm[tag] = dict(n=int(m.sum()), concordance=float(conc[m].mean()), upper95=cp_upper(int(conc[m].sum()), int(m.sum())))
npairs = int(acc.sum() - 17); base = 1 - 0.05 ** (1 / npairs)
def bound(q): return float(base / (1 - q))
P['nearmiss'] = dict(n_cand=int(cand.sum()), n_acc=int(acc.sum()), n_rej=int(rej.sum()), rej_ncc_below_02=int((rej & (ncc < 0.2)).sum()), rej_ncc_02_07=int((rej & (ncc >= 0.2) & (ncc < 0.7)).sum()),
                     acc_ncc_min=float(ncc[acc].min()), groups=nm, prevalence_reps=float(y4.mean()), chance_concordance=float(y4.mean() ** 2 + (1 - y4.mean()) ** 2),
                     new_pairs=npairs, bound_q05=bound(0.5), bound_q_inl20=bound(nm['rejected_inl20']['concordance']), bound_q_inl20_upper=bound(nm['rejected_inl20']['upper95']))
# ---------- 2. synthetic ----------
P['synthetic'] = SY
# ---------- 3. matched single runs: DeLong between families ----------
d = load_meta(); dd = dedup(d); yd = dd.anemia.values
cnn = {}
for f in sorted(glob.glob(REPO + 'resultados/seeds/cnn_*_clean_s*.json')) + sorted(glob.glob('/home/claude/gpu_out/cnn_*_clean.json')):
    jj = json.load(open(f)); assert jj['y'] == yd.tolist(); name = jj['res']['model']; seed = int(jj['res'].get('seed', 42))
    o = jj['oof']; o = o['fp32'] if isinstance(o, dict) and 'fp32' in o else (list(o.values())[0] if isinstance(o, dict) else o)
    cnn[(name, seed)] = np.array(o, float)
old = json.load(open('/home/claude/exp/out/review/review_results_b.json'))['comparisons']
comps = []
for ref in ('RF (color)', 'XGBoost (color)'):
    for (name, seed), po in sorted(cnn.items(), key=lambda t: (t[0][1] != 42, t[0][0], t[0][1])):
        pr = np.array(R5['single']['471|%s|%d' % (ref, seed)]['oof'])
        dl, z, pv = delong(yd, pr, po); ci, pb = paired_boot_diff(yd, pr, po)
        comps.append(dict(ref=ref, other='%s seed %d' % (name, seed), kind='single runs, same folds', AUC_ref=float(roc_auc_score(yd, pr)), AUC_other=float(roc_auc_score(yd, po)), diff=dl, delong_z=z, delong_p=pv, boot_ci=ci, boot_p=pb))
    for c in old:
        if c['ref'] == ref and c['other'].endswith('(color)'):
            cc = dict(c); cc['kind'] = 'averaged probabilities'; comps.append(cc)
for c, h in zip(comps, holm(np.array([c['delong_p'] for c in comps]))): c['delong_p_holm'] = h
P['comparisons'] = comps
# ---------- 4. Nadeau-Bengio corrected resampled t test, RF vs tuned CNNs ----------
def nb(dvec, ratio=0.25):
    dvec = np.asarray(dvec); J = len(dvec); m = dvec.mean(); v = dvec.var(ddof=1); se = np.sqrt((1 / J + ratio) * v); t = m / se; tq = stats.t.ppf(0.975, J - 1)
    return dict(J=J, mean=float(m), ci=[float(m - tq * se), float(m + tq * se)], t=float(t), p=float(2 * stats.t.sf(abs(t), J - 1)), naive_ci=[float(m - tq * np.sqrt(v / J)), float(m + tq * np.sqrt(v / J))])
P['nadeau_bengio'] = {}
for setname, task in (('471', 'tuned'), ('geo', 'tunedgeo')):
    for model in ('EfficientNet-B0', 'MobileNetV2', 'MobileNetV3-Large'):
        dv = []
        for seed in (42, 1, 2):
            cj = json.load(open(REPO + 'resultados/review2/gpu2/cnn_%s_%s_s%d.json' % (model, task, seed)))
            rf = R5['single']['%s|RF (color)|%d' % (setname, seed)]
            assert len(cj['res']['per_fold_AUC']) == 5 == len(rf['per_fold'])
            dv += list(np.array(rf['per_fold']) - np.array(cj['res']['per_fold_AUC']))
        P['nadeau_bengio']['%s|%s' % (setname, model)] = nb(dv)
ps = [v['p'] for v in P['nadeau_bengio'].values()]
for (kk, v), h in zip(P['nadeau_bengio'].items(), holm(np.array(ps))): v['p_holm'] = h
# ---------- 5. LOHO on the logit scale ----------
def reml_hk_logit(per):
    A = np.array([p['AUC'] for p in per]); SE = np.array([p['se'] for p in per]); A = np.clip(A, 1e-3, 1 - 1e-3)
    yk = np.log(A / (1 - A)); vk = (SE / (A * (1 - A))) ** 2; kk = len(yk)
    def nll(t2):
        w = 1 / (vk + t2); mu = (w * yk).sum() / w.sum(); return 0.5 * (np.log(vk + t2).sum() + np.log(w.sum()) + (w * (yk - mu) ** 2).sum())
    t2 = float(optimize.minimize_scalar(nll, bounds=(0, 10), method='bounded').x); w = 1 / (vk + t2); mu = float((w * yk).sum() / w.sum())
    q = float((w * (yk - mu) ** 2).sum() / (kk - 1)); se_hk = np.sqrt(q / w.sum()); tq = stats.t.ppf(0.975, kk - 1); se_re = np.sqrt(1 / w.sum()); ph = stats.t.ppf(0.975, kk - 2) * np.sqrt(t2 + se_re ** 2)
    ex = lambda x: float(1 / (1 + np.exp(-x)))
    wf = 1 / vk; muf = (wf * yk).sum() / wf.sum(); Q = (wf * (yk - muf) ** 2).sum(); t2dl = max(0, (Q - (kk - 1)) / (wf.sum() - (wf ** 2).sum() / wf.sum())); wd = 1 / (vk + t2dl); mud = (wd * yk).sum() / wd.sum(); sed = np.sqrt(1 / wd.sum())
    return dict(k=kk, tau2_logit=t2, mu=ex(mu), ci_hk=[ex(mu - tq * se_hk), ex(mu + tq * se_hk)], pred_int=[ex(mu - ph), ex(mu + ph)], dl_mu=ex(mud), dl_ci=[ex(mud - 1.96 * sed), ex(mud + 1.96 * sed)], I2=float(max(0, (Q - (kk - 1)) / Q)) if Q > 0 else 0.0)
RB = json.load(open('/home/claude/exp/out/review/review_results_b.json')); R3 = json.load(open('/home/claude/exp/out/review3/review3.json'))
P['loho_logit'] = {'471': reml_hk_logit(RB['loho']['per_centre']), 'geo': reml_hk_logit(R3['loho_geo']['per_centre']), 'geo_singletons': reml_hk_logit(R3['loho_geo_singletons']['per_centre'])}
P['loho_pooled'] = {'471': [RB['loho']['pooled_AUC'], RB['loho']['pooled_ci']], 'geo': [R3['loho_geo']['pooled_AUC'], R3['loho_geo']['pooled_ci']], 'geo_singletons': [R3['loho_geo_singletons']['pooled_AUC'], R3['loho_geo_singletons']['pooled_ci'], R3['loho_geo_singletons']['n']]}
# ---------- 6. hue and SHAP (copied) ----------
for kk in [x for x in R5 if x.startswith('huerot|')] + ['hue', 'shap_grouped']: P[kk] = R5[kk]
P['rep_original'] = {'471|RF (color)': json.load(open('/home/claude/exp/out/review/review_results.json'))['dedup']['RF (color)']['AUC_per_rep'] if 'dedup' in json.load(open('/home/claude/exp/out/review/review_results.json')) else None}
json.dump(P, open(OUT + 'post.json', 'w'), indent=1, default=float)
print(json.dumps({k: v for k, v in P.items() if k not in ('comparisons', 'synthetic')}, indent=1, default=float)[:6000])
for c in comps: print(c['ref'][:3], c['other'], round(c['AUC_ref'], 3), round(c['AUC_other'], 3), round(c['diff'], 3), '%.2g' % c['delong_p'], '%.2g' % c['delong_p_holm'])
