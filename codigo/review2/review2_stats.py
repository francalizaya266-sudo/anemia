# Second internal review, part 1 (statistics only, no image processing):
# class-conditional duplication, clustered CIs/tests on the 708 rows (Obuchowski 1997 + group bootstrap),
# paired 471-vs-452 comparison on common images, strict-set operating points and PPV/NPV at 42-47 %,
# REML + Hartung-Knapp meta-analysis of LOHO, Brier skill score, Riley et al. 2021 sample size.
import sys, os, json, glob; sys.path.insert(0, '/home/claude/exp'); sys.path.insert(0, '/home/claude/anemia/codigo/review')
import numpy as np, pandas as pd
from scipy import stats, optimize
from sklearn.metrics import roc_auc_score, brier_score_loss
from common import load_meta, dedup
from review2_lib import obu_ci, obu_test, cluster_boot
from review_analyses_lib import auc_ci, paired_boot_diff
REPO = '/home/claude/anemia/'; OUT = '/home/claude/exp/out/review2/'; R = {}
d = load_meta(); dd = dedup(d); y = d.anemia.values
ms = pd.read_csv(REPO + 'codigo/meta_groups_strict.csv').set_index('image_id')
sg = ms.loc[d.image_id, 'grp'].values                      # strict group of each of the 708 rows (clusters)
# ---- A. class-conditional duplication ----
mfull = pd.read_csv(REPO + 'codigo/meta_groups.csv'); sz = mfull.groupby('grp').grp.transform('size'); dup = mfull[sz > 1]
cons = int(dup.groupby('grp').anemia.nunique().eq(1).sum()); G = dup.grp.nunique()
rng = np.random.default_rng(0); gi = dup.grp.values; lab = dup.anemia.values; labA = mfull.anemia.values; maskA = (sz > 1).values
def ncons(g, l): return int(pd.DataFrame({'g': g, 'l': l}).groupby('g').l.nunique().eq(1).sum())
null_dup = np.array([ncons(gi, rng.permutation(lab)) for _ in range(20000)])
null_all = np.array([ncons(gi, rng.permutation(labA)[maskA]) for _ in range(20000)])
hbr = dup.groupby('grp').hb.agg(lambda s: s.max() - s.min())
R['class_conditional'] = dict(groups=G, label_consistent=cons, null_within_dup_mean=float(null_dup.mean()), null_within_dup_max=int(null_dup.max()),
    p_within_dup=float((1 + (null_dup >= cons).sum()) / (1 + len(null_dup))), null_all_rows_mean=float(null_all.mean()), null_all_rows_max=int(null_all.max()),
    hb_range_median=float(hbr.median()), hb_range_max=float(hbr.max()), groups_hb_differ=int((hbr > 0).sum()),
    groups_multi_centre=int((dup.groupby('grp').hospital.nunique() > 1).sum()), max_centres=int(dup.groupby('grp').hospital.nunique().max()),
    hb_two_decimals=int((mfull.hb.astype(str).str.split('.').str[1].str.len() == 2).sum()))
print('A', R['class_conditional'])
# ---- B. clustered statistics on the 708 rows ----
o = json.load(open(REPO + 'resultados/review/oof_review.json')); os_ = json.load(open(REPO + 'resultados/review/oof_strict.json'))
R['clustered'] = {}
def clustered(name, pn, pg, cl, tag):
    an, cin, _ = obu_ci(y, pn, cl); ag, cig, _ = obu_ci(y, pg, cl); dlt, cid, z, p = obu_test(y, pn, pg, cl); cb = cluster_boot(y, np.vstack([pn, pg]), cl)
    e = dict(naive=an, naive_ci_obu=cin, naive_ci_boot=cb['auc_ci'][0], grouped=ag, grouped_ci_obu=cig, grouped_ci_boot=cb['auc_ci'][1], diff=dlt, diff_ci_obu=cid,
             diff_ci_boot=cb['diff_ci'], p_obu=p, z_obu=z, p_boot=cb['diff_p'])
    R['clustered'][f'{name}|{tag}'] = e; print('B', name, tag, {k: (round(v, 4) if isinstance(v, float) else [round(x, 4) for x in v]) for k, v in e.items()})
for m in ['LR (color)', 'SVM-RBF (color)', 'RF (color)', 'XGBoost (color)']:
    clustered(m, np.array(o[m]['naive708']), np.array(o[m]['grouped708']), sg, 'first')
    clustered(m, np.array(o[m]['naive708']), np.array(os_[m]['grouped_strict708']), sg, 'strict')
GN = ['/home/claude/anemia/gpu/out_review', '/home/claude/gpu_out', '/mnt/user-data/uploads/AnemIA-E2/Nueva carpeta/out']
for m in ['EfficientNet-B0', 'MobileNetV2', 'MobileNetV3-Small']:
    g = json.load(open(f'/home/claude/anemia/gpu/out_review/cnn_{m}_grouped.json'))
    nf = next(os.path.join(x, f'cnn_{m}_naive.json') for x in GN if os.path.exists(os.path.join(x, f'cnn_{m}_naive.json')))
    n = json.load(open(nf)); assert np.array_equal(np.array(n['y']), y) and np.array_equal(np.array(g['y']), y)
    clustered(m, np.array(n['oof']['fp32']), np.array(g['oof']['fp32']), sg, 'first')
    sn = json.load(open(f'/home/claude/anemia/gpu/out_strict/cnn_{m}_naive.json')); sgj = json.load(open(f'/home/claude/anemia/gpu/out_strict/cnn_{m}_grouped.json'))
    assert np.array_equal(np.array(sn['y']), y)
    clustered(m, np.array(sn['oof']['fp32']), np.array(sgj['oof']['fp32']), sg, 'strict')
# ---- C. paired 471 vs 452 on common images ----
st = json.load(open(REPO + 'resultados/review/review_strict.json')); simg = st['images']
common = sorted(set(dd.image_id) & set(simg)); i471 = dd.reset_index().set_index('image_id').loc[common, 'index'].values
i452 = pd.Series(range(len(simg)), index=simg).loc[common].values; yc = dd.anemia.values[i471]
R['paired_471_452'] = {'n_common': len(common)}
for m in ['RF (color)', 'XGBoost (color)']:
    a = np.array(o[m]['dedup471'])[i471]; b = np.array(os_[m]['strict452'])[i452]
    ci, p = paired_boot_diff(yc, a, b)
    R['paired_471_452'][m] = dict(auc_471_models=float(roc_auc_score(yc, a)), auc_452_models=float(roc_auc_score(yc, b)), diff=float(roc_auc_score(yc, a) - roc_auc_score(yc, b)), diff_ci=ci, p=p)
    print('C', m, R['paired_471_452'][m])
# ---- E. operating points and PPV/NPV ----
rr = json.load(open(REPO + 'resultados/review/review_results.json'))
def ppv(se, sp, pi): return se * pi / (se * pi + (1 - sp) * (1 - pi))
def npv(se, sp, pi): return sp * (1 - pi) / (sp * (1 - pi) + (1 - se) * pi)
R['screening'] = {}
for m in ['RF (color)', 'XGBoost (color)']:
    for tag, src in [('471', rr['dedup'][m]), ('452', st['dedup'][m])]:
        se, sp = src['nested_thr_sens_mean'], src['nested_thr_spec_mean']
        R['screening'][f'{m}|{tag}'] = dict(sens=se, spec=sp, **{f'PPV_{int(100*pi)}': ppv(se, sp, pi) for pi in (0.42, 0.47, 0.70)}, **{f'NPV_{int(100*pi)}': npv(se, sp, pi) for pi in (0.42, 0.47, 0.70)},
            **{f'test_neg_frac_{int(100*pi)}': (sp * (1 - pi) + (1 - se) * pi) for pi in (0.42, 0.47)}, **{f'missed_per_1000_{int(100*pi)}': 1000 * (1 - se) * pi for pi in (0.42, 0.47)})
        print('E', m, tag, {k: round(v, 3) for k, v in R['screening'][f'{m}|{tag}'].items()})
# ---- F. REML + Hartung-Knapp for LOHO ----
rb = json.load(open(REPO + 'resultados/review/review_results_b.json')); pc = rb['loho']['per_centre']
yk = np.array([c['AUC'] for c in pc]); vk = np.array([c['se'] ** 2 for c in pc]); k = len(yk)
def reml_negll(t2):
    w = 1 / (vk + t2); mu = (w * yk).sum() / w.sum()
    return 0.5 * (np.log(vk + t2).sum() + np.log(w.sum()) + (w * (yk - mu) ** 2).sum())
t2 = float(optimize.minimize_scalar(reml_negll, bounds=(0, 1), method='bounded').x)
w = 1 / (vk + t2); mu = float((w * yk).sum() / w.sum())
q = float((w * (yk - mu) ** 2).sum() / (k - 1)); se_hk = float(np.sqrt(q / w.sum())); tq = stats.t.ppf(0.975, k - 1)
se_re = float(np.sqrt(1 / w.sum())); pi_half = stats.t.ppf(0.975, k - 2) * np.sqrt(t2 + se_re ** 2)
R['loho_meta'] = dict(k=k, tau2_reml=t2, mu=mu, ci_hk=[mu - tq * se_hk, mu + tq * se_hk], pred_int=[mu - pi_half, min(1.0, mu + pi_half)], dl_mu=rb['loho']['random_effects_AUC'], dl_ci=rb['loho']['random_effects_ci'])
# centre ambiguity of the retained representatives
dmap = d.groupby('grp').hospital.nunique(); rep_amb = int(dd.grp.map(dmap).gt(1).sum())
R['loho_meta']['representatives_multi_centre'] = rep_amb; R['loho_meta']['n_dedup'] = len(dd)
print('F', R['loho_meta'])
# ---- L. Brier skill score ----
R['brier'] = {}
for m in ['RF (color)', 'XGBoost (color)', 'LR (color)', 'SVM-RBF (color)']:
    for tag, yy, pp in [('471', dd.anemia.values, np.array(o[m]['dedup471'])), ('452', np.array(st_y := None) if False else None, None)]:
        if pp is None: continue
        b = brier_score_loss(yy, pp); ref = yy.mean() * (1 - yy.mean()); R['brier'][f'{m}|{tag}'] = dict(brier=float(b), bss=float(1 - b / ref))
print('L', R['brier'])
# ---- M. Riley et al. 2021 minimum sample size for external validation (binary outcome) ----
def n_cstat(C, phi, width=0.10):
    # SE of C (Newcombe 2006 / Riley 2021 eq.) as a function of n; find n giving 95 % CI width <= width
    def se(n):
        n1 = n * phi; n0 = n * (1 - phi)
        return np.sqrt(C * (1 - C) * (1 + (n1 / 2 - 1) * (1 - C) / (2 - C) + (n0 / 2 - 1) * C / (1 + C)) / (n1 * n0))
    n = 50
    while 2 * 1.96 * se(n) > width: n += 1
    return n
def n_oe(phi, width=0.2):
    # O/E: SE(ln O/E) ~ sqrt((1-phi)/(n phi)); CI width on O/E scale ~ 0.2 -> SE(lnOE) = 0.051
    return int(np.ceil((1 - phi) / (phi * 0.051 ** 2)))
def n_slope(lp, phi, width=0.2, sims=1):
    # Riley 2021 closed form via Fisher information of (alpha, beta) for a perfectly calibrated model
    p = 1 / (1 + np.exp(-lp)); I_aa = np.mean(p * (1 - p)); I_ab = np.mean(lp * p * (1 - p)); I_bb = np.mean(lp ** 2 * p * (1 - p))
    var1 = I_aa / (I_aa * I_bb - I_ab ** 2); se_target = width / (2 * 1.96)
    return int(np.ceil(var1 / se_target ** 2))
prf = np.clip(np.array(o['RF (color)']['dedup471']), 1e-4, 1 - 1e-4); lp = np.log(prf / (1 - prf))
R['sample_size'] = {}
for phi in (0.42, 0.47):
    lpa = lp - np.log(dd.anemia.mean() / (1 - dd.anemia.mean())) + np.log(phi / (1 - phi))   # intercept updated to the target prevalence
    R['sample_size'][str(phi)] = dict(cstat_085=n_cstat(0.85, phi), cstat_080=n_cstat(0.80, phi), oe=n_oe(phi), slope=n_slope(lpa, phi))
print('M', R['sample_size'])
json.dump(R, open(OUT + 'review2_stats.json', 'w'), indent=1, default=float)
print('saved')
