# Third round of internal review (CPU analyses). Writes out/review3/review3.json; every section is skipped if already done.
#  B  two duplication patterns (composition by audit stage)
#  C  paired comparison 471-image models vs geometric-set models on the 337 common images
#  H  Riley et al. (2020) sample size for developing a new model, and a two-phase design
#  D  size and class-balance control: the 471-image set subsampled to 337 images (179 anaemic, 158 not) by removing, at random
#     within class, images that have no geometric duplicate in the set (near-duplicate pairs are kept)
#  E  Random Forest per-repetition out-of-fold predictions (471 and geometric sets) and the formal RF-versus-tuned-CNN contrast
#  G  eroded-mask pipeline evaluated on perturbed images (JPEG 75, 50 % rescaling, white balance)
#  F  leave-one-hospital-out on the geometric set and on its singletons, REML + Hartung-Knapp
import sys, os, json, time, glob, warnings; sys.path.insert(0, '/home/claude/exp'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, cv2
from scipy import stats, optimize
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from common import load_meta, dedup, read_rgb, mask_of, folds, SEED
from classic import color_features, MODELS, run_cv
from review_analyses_lib import auc_ci, paired_boot_diff, delong
# Speed: parallelise the grid search instead of the forest (identical results: the trees depend only on random_state)
import classic as _cl
_GS = GridSearchCV
def GridSearchCV(*a, **k): k['n_jobs'] = 4; return _GS(*a, **k)
_cl.GridSearchCV = GridSearchCV
for _k in ('RF (color)', 'XGBoost (color)'): MODELS[_k][0].set_params(n_jobs=1)
OUT3 = '/home/claude/exp/out/review3/'; REPO = '/home/claude/anemia/'; GPU2 = '/home/claude/exp/out/review2/gpu2/'
RJ = OUT3 + 'review3.json'; R = json.load(open(RJ)) if os.path.exists(RJ) else {}
def save(): json.dump(R, open(RJ, 'w'), indent=1, default=float)
T0 = time.time()
def log(*a): print('[%6.0fs]' % (time.time() - T0), *a, flush=True)
d = load_meta(); dd = dedup(d); yd = dd.anemia.values
Xa = pd.read_csv('/home/claude/exp/out/feat_all.csv'); Xd = pd.read_csv('/home/claude/exp/out/feat_dedup.csv'); assert len(Xd) == len(dd) == 471
geo = pd.read_csv(REPO + 'codigo/meta_groups_geo.csv'); gmap = geo.set_index('image_id').grp
# geometric set exactly as in review2_geo.py
s = geo.copy(); conf = s.groupby('grp').anemia.nunique(); s = s[~s.grp.isin(conf[conf > 1].index)].reset_index(drop=True)
assert (s.image_id.values == d.image_id.values).all()
ss = s.sort_values('image_id').drop_duplicates('grp'); gidx = ss.index.values; Xg = Xa.iloc[gidx].reset_index(drop=True); dg = ss.reset_index(drop=True); yg = dg.anemia.values
assert len(dg) == 337
oof = json.load(open(REPO + 'resultados/review/oof_review.json')); oofg = json.load(open(REPO + 'resultados/review2/geo/oof_geo.json'))
RG = json.load(open(REPO + 'resultados/review2/geo/review_geo.json')); assert RG['images'] == dg.image_id.tolist()
CM = ['LR (color)', 'SVM-RBF (color)', 'RF (color)', 'XGBoost (color)']
def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = np.zeros(m); run = 0
    for k, i in enumerate(o): run = max(run, (m - k) * ps[i]); adj[i] = min(1.0, run)
    return adj.tolist()
def reml_hk(a, se):
    yk = np.asarray(a, float); vk = np.asarray(se, float) ** 2; k = len(yk)
    def nll(t2):
        w = 1 / (vk + t2); mu = (w * yk).sum() / w.sum(); return 0.5 * (np.log(vk + t2).sum() + np.log(w.sum()) + (w * (yk - mu) ** 2).sum())
    t2 = float(optimize.minimize_scalar(nll, bounds=(0, 1), method='bounded').x); w = 1 / (vk + t2); mu = float((w * yk).sum() / w.sum())
    q = float((w * (yk - mu) ** 2).sum() / (k - 1)); se_hk = float(np.sqrt(q / w.sum())); tq = stats.t.ppf(0.975, k - 1)
    se_re = float(np.sqrt(1 / w.sum())); ph = stats.t.ppf(0.975, k - 2) * np.sqrt(t2 + se_re ** 2)
    return dict(k=k, tau2=t2, mu=mu, ci_hk=[mu - tq * se_hk, min(1.0, mu + tq * se_hk)], pred_int=[mu - ph, min(1.0, mu + ph)])

# ---------------- B. two duplication patterns ----------------
if 'patterns' not in R:
    m1 = pd.read_csv(REPO + 'codigo/meta_groups.csv'); m3 = pd.read_csv(REPO + 'codigo/meta_groups_strict.csv'); m4 = geo
    out = {}
    for tag, m in (('prespecified', m1), ('strict', m3), ('geometric', m4)):
        sz = m.groupby('grp').grp.transform('size'); dup = sz > 1
        out[tag] = dict(rows=int(len(m)), dup_rows=int(dup.sum()), dup_anaemic_pct=float(100 * m.anemia[dup].mean()), single_rows=int((~dup).sum()),
                        single_anaemic_pct=float(100 * m.anemia[~dup].mean()), groups=int(m.grp.nunique()),
                        extra_copies_anaemic=int(((m.groupby('grp').size() - 1) * m.groupby('grp').anemia.max()).sum()),
                        extra_copies=int((m.groupby('grp').size() - 1).sum()))
    s1 = m1.groupby('grp').grp.transform('size').values == 1; s4 = m4.groupby('grp').grp.transform('size').values > 1
    ab = s1 & s4                                                        # singletons of the pre-specified map absorbed by stages 3-4
    s3 = m3.groupby('grp').grp.transform('size').values > 1
    out['absorbed_stage34'] = dict(n=int(ab.sum()), non_anaemic=int((m1.anemia.values[ab] == 0).sum()), anaemic=int((m1.anemia.values[ab] == 1).sum()),
                                   by_stage3=int((s1 & s3).sum()), by_stage3_non=int(((s1 & s3) & (m1.anemia.values == 0)).sum()),
                                   base_non_pct_singletons_prespec=float(100 * (m1.anemia.values[s1] == 0).mean()))
    out['absorbed_stage34']['binom_p_vs_singleton_base'] = float(stats.binomtest(out['absorbed_stage34']['non_anaemic'], out['absorbed_stage34']['n'], out['absorbed_stage34']['base_non_pct_singletons_prespec'] / 100).pvalue)
    # by stage-4 pair: class of the 168 new pairs
    G = np.load(REPO + 'resultados/review2/geomverify.npy'); arr = np.load(REPO + 'resultados/review2/orb_pairs.npy'); ncand = int((arr[:, 3] >= 10).sum())
    sgr = m3.set_index('image_id').loc[m1.sort_values('image_id').drop_duplicates('grp').image_id, 'grp'].values
    gi, gj = G[:, 0].astype(int), G[:, 1].astype(int); okm = (np.arange(len(G)) < ncand) & (G[:, 2] >= 10) & (G[:, 4] >= 0.70) & (G[:, 6] >= 0.5) & (G[:, 6] <= 2.0) & (G[:, 3] >= 0.30)
    acc = np.c_[gi[okm], gj[okm], (sgr[gi] == sgr[gj])[okm]]; acc = np.c_[acc[:, :2], np.zeros((len(acc), 5)), acc[:, 2]]
    reps = m1.sort_values('image_id').drop_duplicates('grp').reset_index(drop=True)
    if acc is not None:
        new = acc[acc[:, 7] == 0]; la = reps.anemia.values[new[:, 0].astype(int)]
        out['stage4_new_pairs'] = dict(n=int(len(new)), anaemic_pairs=int(la.sum()), non_anaemic_pairs=int((la == 0).sum()), base_anaemic_pct_reps=float(100 * reps.anemia.mean()))
    # rows added to groups by each stage (identical / hash copies vs keypoint matches), by class
    g12 = m1.groupby('grp'); out['hash_copies_rows'] = dict(anaemic=int(m1.anemia[m1.groupby('grp').grp.transform('size') > 1].sum()), non=int((m1.anemia[m1.groupby('grp').grp.transform('size') > 1] == 0).sum()))
    R['patterns'] = out; save(); log('B', json.dumps(out)[:900])

# ---------------- C. paired comparison on the 337 common images ----------------
if 'paired_471_geo' not in R:
    common = sorted(set(dd.image_id) & set(dg.image_id)); i471 = pd.Series(range(len(dd)), index=dd.image_id).loc[common].values
    ig = pd.Series(range(len(dg)), index=dg.image_id).loc[common].values; yc = yd[i471]; assert (yc == yg[ig]).all()
    out = {'n_common': len(common), 'prev': float(yc.mean())}
    for m in CM:
        a = np.array(oof[m]['dedup471'])[i471]; b = np.array(oofg[m]['geo349'])[ig]; ci, p = paired_boot_diff(yc, a, b); dl = delong(yc, a, b)
        out[m] = dict(auc_471_models=float(roc_auc_score(yc, a)), auc_geo_models=float(roc_auc_score(yc, b)), diff=float(roc_auc_score(yc, a) - roc_auc_score(yc, b)), diff_ci=ci, p_boot=p, p_delong=float(dl[2]))
    R['paired_471_geo'] = out; save(); log('C', out)

# ---------------- H. sample size for developing a new model (Riley et al. 2020) ----------------
if 'riley2020' not in R:
    rng = np.random.default_rng(1)
    def r2cs_from_c(C, phi, n=400000):
        # binormal-equal-variance linear predictor: C = Phi(delta / sqrt(2)); calibrated risks via logistic fit on a large simulated sample
        delta = np.sqrt(2) * stats.norm.ppf(C); yy = rng.random(n) < phi; lp = rng.normal(0, 1, n) + delta * yy
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(C=1e6).fit(lp[:, None], yy); p = np.clip(lr.predict_proba(lp[:, None])[:, 1], 1e-12, 1 - 1e-12)
        ll = (yy * np.log(p) + (1 - yy) * np.log(1 - p)).sum(); ll0 = n * (phi * np.log(phi) + (1 - phi) * np.log(1 - phi))
        return float(1 - np.exp(-2 * (ll - ll0) / n)), float(roc_auc_score(yy, lp))
    out = {}
    for phi in (0.42, 0.47):
        mx = 1 - (phi ** phi * (1 - phi) ** (1 - phi)) ** 2
        for C in (0.75, 0.80):
            r2, cchk = r2cs_from_c(C, phi)
            for P in (10, 20):
                S1 = 0.9; n1 = P / ((S1 - 1) * np.log(1 - r2 / S1))
                S2 = r2 / (r2 + 0.05 * mx); n2 = P / ((S2 - 1) * np.log(1 - r2 / S2))
                n3 = (1.96 / 0.05) ** 2 * phi * (1 - phi); n = int(np.ceil(max(n1, n2, n3)))
                out[f'phi{phi}|C{C}|p{P}'] = dict(r2cs=r2, max_r2cs=mx, c_check=cchk, n_shrinkage=int(np.ceil(n1)), n_optimism=int(np.ceil(n2)), n_intercept=int(np.ceil(n3)), n=n, events=int(np.ceil(n * phi)), epp=float(n * phi / P))
    R['riley2020'] = out; save(); log('H', {k: v['n'] for k, v in out.items()})

# ---------------- D. size and class-balance control ----------------
gcount = pd.Series(gmap.loc[dd.image_id].values).map(pd.Series(gmap.loc[dd.image_id].values).value_counts()).values
single471 = gcount == 1
R.setdefault('size_control', {'design': 'remove images without a geometric duplicate in the 471-image set, at random within class, to 179 anaemic / 158 not'})
need_a = int(yd.sum() - yg.sum()); need_n = int((1 - yd).sum() - (1 - yg).sum())
R['size_control'].update(nsub_note='10 subsamples for XGBoost, 5 for Random Forest', n_single471=int(single471.sum()), single_anaemic=int((single471 & (yd == 1)).sum()), single_non=int((single471 & (yd == 0)).sum()), remove_anaemic=need_a, remove_non=need_n)
NSUB = 10
def subsample(k):
    rg = np.random.default_rng(1000 + k); ia = np.where(single471 & (yd == 1))[0]; inn = np.where(single471 & (yd == 0))[0]
    drop = np.concatenate([rg.choice(ia, need_a, replace=False), rg.choice(inn, need_n, replace=False)]); keep = np.setdiff1d(np.arange(len(yd)), drop)
    return keep
for m in ['XGBoost (color)', 'RF (color)']:
    key = m + '|aucs'; NSUB = 10 if m.startswith('XGB') else 5
    if len(R['size_control'].get(key, [])) >= NSUB and m in R['size_control']: continue
    aucs = R['size_control'].get(key, []); est, grid = MODELS[m]
    for k in range(len(aucs), NSUB):
        keep = subsample(k); assert len(keep) == 337 and yd[keep].sum() == yg.sum()
        P = run_cv(Xd.iloc[keep].reset_index(drop=True), yd[keep], None, m, est, grid, reps=1); aucs.append(float(roc_auc_score(yd[keep], P[0])))
        R['size_control'][key] = aucs; save(); log('D', m, k, round(aucs[-1], 3))
    R['size_control'][m] = dict(mean=float(np.mean(aucs)), sd=float(np.std(aucs, ddof=1)), min=float(np.min(aucs)), max=float(np.max(aucs)),
                                full471_rep_mean=float(json.load(open(REPO + 'resultados/review/review_results.json'))['dedup'][m]['AUC_rep_mean']),
                                geo_rep_mean=float(RG['dedup'][m]['AUC_rep_mean'])); save(); log('D', m, R['size_control'][m])

# ---------------- E. RF per repetition and RF versus tuned CNN ----------------
os.makedirs(OUT3, exist_ok=True)
est, grid = MODELS['RF (color)']
if not os.path.exists(OUT3 + 'rf_rep471.npy'):
    P = run_cv(Xd, yd, None, 'RF', est, grid, reps=5); np.save(OUT3 + 'rf_rep471.npy', P); log('E RF 471 per rep', [round(roc_auc_score(yd, p), 3) for p in P])
def cnn_oof(task, ids_ref, y_ref):
    out = {}
    for f in sorted(glob.glob(GPU2 + f'cnn_*_{task}_s*.json')):
        j = json.load(open(f)); mname = os.path.basename(f)[4:].split('_' + task + '_')[0]
        o = pd.Series(j['oof'], index=j['ids']).loc[ids_ref].values; assert (pd.Series(j['y'], index=j['ids']).loc[ids_ref].values == y_ref).all()
        out.setdefault(mname, []).append(o)
    return {k: np.array(v) for k, v in out.items()}
def contrast(Prf, C, y, cl, tag):
    res = {}; ps = []; psd = []
    for mname, Pc in sorted(C.items()):
        a_rf = np.mean([roc_auc_score(y, p) for p in Prf]); a_c = np.mean([roc_auc_score(y, p) for p in Pc])
        ids, inv = np.unique(cl, return_inverse=True); rows = [np.where(inv == g)[0] for g in range(len(ids))]; rng = np.random.default_rng(SEED); bd = []
        for _ in range(2000):
            idx = np.concatenate([rows[g] for g in rng.integers(0, len(ids), len(ids))])
            if len(np.unique(y[idx])) < 2: continue
            bd.append(np.mean([roc_auc_score(y[idx], p[idx]) for p in Prf]) - np.mean([roc_auc_score(y[idx], p[idx]) for p in Pc]))
        bd = np.array(bd); p = float(min(1.0, 2 * min((bd <= 0).mean(), (bd >= 0).mean()))); dl = delong(y, Prf.mean(0), Pc.mean(0))
        res[mname] = dict(seeds=int(len(Pc)), rf_rep_mean=float(a_rf), cnn_run_mean=float(a_c), diff=float(a_rf - a_c), diff_ci=[float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))], p_boot=p, p_delong_avg=float(dl[2]))
        ps.append(p); psd.append(float(dl[2]))
    for (k, v), h, hd in zip(sorted(res.items()), holm(ps), holm(psd)): v['p_boot_holm'] = h; v['p_delong_holm'] = hd
    log('E', tag, {k: (round(v['diff'], 3), [round(x, 3) for x in v['diff_ci']], '%.2g' % v['p_boot_holm']) for k, v in res.items()}); return res
Prf = np.load(OUT3 + 'rf_rep471.npy'); cl471 = gmap.loc[dd.image_id].values
R['rf_vs_tuned_471'] = contrast(Prf, cnn_oof('tuned', dd.image_id.tolist(), yd), yd, cl471, '471'); R['rf_rep471_aucs'] = [float(roc_auc_score(yd, p)) for p in Prf]; save()

# ---------------- G. eroded-mask pipeline on perturbed images ----------------
if 'eroded_perturbed' not in R:
    def feats_masked(im, m): im2 = im.copy(); im2[~m] = 0; return color_features(im2)
    def jpeg(im, q=75):
        ok, buf = cv2.imencode('.jpg', cv2.cvtColor(im, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q]); return cv2.cvtColor(cv2.imdecode(buf, 1), cv2.COLOR_BGR2RGB)
    def rescale(im, f=0.5):
        h, w = im.shape[:2]; s_ = cv2.resize(im, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA); return cv2.resize(s_, (w, h), interpolation=cv2.INTER_LINEAR)
    def wb(im, g=(1.05, 1.0, 0.95)):
        out = np.clip(im.astype(np.float32) * np.array(g)[None, None], 0, 255).astype(np.uint8); out[~mask_of(im)] = 0; return out
    def erode(im, px=2):
        m = mask_of(im).astype(np.uint8); m = cv2.erode(m, np.ones((2 * px + 1, 2 * px + 1), np.uint8)).astype(bool); return feats_masked(im, m)
    imgs = [read_rgb(p) for p in dd.image_path]; F0 = Xd
    FE = {'none': pd.read_csv('/home/claude/exp/out/review2/feat_dedup_erode2.csv')[F0.columns]}
    FO = {'jpeg75': pd.read_csv('/home/claude/exp/out/review2/feat_dedup_jpeg75.csv')[F0.columns], 'rescale50': pd.read_csv('/home/claude/exp/out/review2/feat_dedup_rescale50.csv')[F0.columns],
          'whitebalance5': pd.DataFrame([color_features(wb(im)) for im in imgs])[F0.columns]}
    for k, fn in (('jpeg75', jpeg), ('rescale50', rescale), ('whitebalance5', wb)): FE[k] = pd.DataFrame([erode(fn(im)) for im in imgs])[F0.columns]
    log('G features done')
    def icc31(a, b):
        Y = np.vstack([a, b]).T; n, k = Y.shape; gm = Y.mean(); ssr = k * ((Y.mean(1) - gm) ** 2).sum(); ssc = n * ((Y.mean(0) - gm) ** 2).sum(); sst = ((Y - gm) ** 2).sum(); sse = sst - ssr - ssc
        msr = ssr / (n - 1); mse = sse / ((n - 1) * (k - 1)); return (msr - mse) / (msr + (k - 1) * mse)
    out = {'icc': {}}
    for k in ('jpeg75', 'rescale50', 'whitebalance5'):
        ie = np.array([icc31(FE['none'][c].values, FE[k][c].values) for c in F0.columns]); io = np.array([icc31(F0[c].values, FO[k][c].values) for c in F0.columns])
        out['icc'][k] = dict(eroded_median=float(np.nanmedian(ie)), eroded_n_below_090=int((ie < 0.9).sum()), original_median=float(np.nanmedian(io)), original_n_below_090=int((io < 0.9).sum()))
    Po = {k: np.zeros(len(yd)) for k in ['clean', 'jpeg75', 'rescale50', 'whitebalance5']}; Pe = {k: np.zeros(len(yd)) for k in Po}
    for tr, te in folds(yd, None, seed=SEED):
        go = GridSearchCV(est, grid, scoring='roc_auc', cv=StratifiedKFold(5, shuffle=True, random_state=0), n_jobs=1).fit(F0.iloc[tr], yd[tr])
        ge = GridSearchCV(est, grid, scoring='roc_auc', cv=StratifiedKFold(5, shuffle=True, random_state=0), n_jobs=1).fit(FE['none'].iloc[tr], yd[tr])
        Po['clean'][te] = go.predict_proba(F0.iloc[te])[:, 1]; Pe['clean'][te] = ge.predict_proba(FE['none'].iloc[te])[:, 1]
        for k in ('jpeg75', 'rescale50', 'whitebalance5'):
            Po[k][te] = go.predict_proba(FO[k].iloc[te])[:, 1]; Pe[k][te] = ge.predict_proba(FE[k].iloc[te])[:, 1]
    out['auc_original_mask'] = {k: float(roc_auc_score(yd, v)) for k, v in Po.items()}; out['auc_eroded_mask'] = {k: float(roc_auc_score(yd, v)) for k, v in Pe.items()}
    out['drop_original'] = {k: out['auc_original_mask']['clean'] - out['auc_original_mask'][k] for k in Po if k != 'clean'}
    out['drop_eroded'] = {k: out['auc_eroded_mask']['clean'] - out['auc_eroded_mask'][k] for k in Pe if k != 'clean'}
    R['eroded_perturbed'] = out; save(); log('G', out)

# ---------------- F. LOHO on the geometric set and on its singletons ----------------
def loho(X, y, hosp):
    P = np.zeros(len(y))
    for h in np.unique(hosp):
        te = hosp == h; tr = ~te
        gs = GridSearchCV(est, grid, scoring='roc_auc', cv=StratifiedKFold(5, shuffle=True, random_state=0), n_jobs=1).fit(X[tr], y[tr]); P[te] = gs.predict_proba(X[te])[:, 1]
    per = []
    for h in np.unique(hosp):
        i = hosp == h
        if i.sum() >= 20 and 0 < y[i].sum() < i.sum():
            a = roc_auc_score(y[i], P[i]); ci = auc_ci(y[i], P[i]); per.append(dict(hospital=h, n=int(i.sum()), n_pos=int(y[i].sum()), AUC=float(a), ci=ci, se=(ci[1] - ci[0]) / 3.92))
    meta = reml_hk([p['AUC'] for p in per], [p['se'] for p in per])
    return dict(n=int(len(y)), centres=int(len(np.unique(hosp))), pooled_AUC=float(roc_auc_score(y, P)), pooled_ci=auc_ci(y, P), per_centre=per, range=[min(p['AUC'] for p in per), max(p['AUC'] for p in per)], **meta)
if 'loho_geo' not in R:
    R['loho_geo'] = loho(Xg, yg, dg.hospital.values); save(); log('F geo', {k: v for k, v in R['loho_geo'].items() if k != 'per_centre'})
if 'loho_geo_singletons' not in R:
    szg = s.groupby('grp').grp.transform('size').values == 1; Xs = Xa[szg].reset_index(drop=True); ys = s.anemia.values[szg]
    R['loho_geo_singletons'] = loho(Xs, ys, s.hospital.values[szg]); save(); log('F geo singletons', {k: v for k, v in R['loho_geo_singletons'].items() if k != 'per_centre'})

# ---------------- E2. RF per repetition on the geometric set (for the tuned-CNN contrast on that set) ----------------
if not os.path.exists(OUT3 + 'rf_repgeo.npy'):
    P = run_cv(Xg, yg, None, 'RF', est, grid, reps=5); np.save(OUT3 + 'rf_repgeo.npy', P); log('E RF geo per rep', [round(roc_auc_score(yg, p), 3) for p in P])
Pg = np.load(OUT3 + 'rf_repgeo.npy'); R['rf_repgeo_aucs'] = [float(roc_auc_score(yg, p)) for p in Pg]
Cg = cnn_oof('tunedgeo', dg.image_id.tolist(), yg)
if Cg: R['rf_vs_tuned_geo'] = contrast(Pg, Cg, yg, dg.grp.values, 'geo')
save(); log('done')
