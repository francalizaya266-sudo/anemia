# Fifth review (independent critique of the duplicate audit), CPU part.
#  A  Random Forest and XGBoost single runs on the fold seeds of the CNN runs (42, 1, 2), 471-image set; Random Forest on the
#     geometric set. Used for (i) DeLong tests between single runs on identical folds and (ii) the Nadeau-Bengio corrected
#     resampled t test for Random Forest against the tuned CNNs (per-fold AUCs on 3 x 5 matched folds).
#  B  Hue as a circular variable: the five hue statistics recomputed on a rotated hue axis (red moved away from the 0/360
#     wrap), Random Forest and XGBoost re-run with five repetitions (471 and geometric sets) and ICC under the white-balance shift.
#  C  Grouped SHAP importance (XGBoost, pre-specified features): mean |sum of SHAP values| per feature family.
import sys, json, time, glob; sys.path.insert(0, '/home/claude/exp')
import numpy as np, pandas as pd, cv2, warnings; warnings.filterwarnings('ignore')
from common import *
from classic import MODELS, color_features
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import classic as _cl
_GS = GridSearchCV
def GridSearchCV(*a, **k): k['n_jobs'] = 4; return _GS(*a, **k)
_cl.GridSearchCV = GridSearchCV
for _k in ('RF (color)', 'XGBoost (color)'): MODELS[_k][0].set_params(n_jobs=1)
REPO = '/home/claude/anemia/'; OUT = '/home/claude/exp/out/review5/'; RJ = OUT + 'review5.json'
T0 = time.time()
def log(*a): print('[%6.0fs]' % (time.time() - T0), *a, flush=True)
try: R = json.load(open(RJ))
except FileNotFoundError: R = {}
def save(): json.dump(R, open(RJ, 'w'), indent=1, default=float)
d = load_meta(); dd = dedup(d); yd = dd.anemia.values
Xa = pd.read_csv('/home/claude/exp/out/feat_all.csv'); Xd = pd.read_csv('/home/claude/exp/out/feat_dedup.csv'); assert len(Xd) == 471
geo = pd.read_csv(REPO + 'codigo/meta_groups_geo.csv')
s = geo.copy(); conf = s.groupby('grp').anemia.nunique(); s = s[~s.grp.isin(conf[conf > 1].index)].reset_index(drop=True)
assert (s.image_id.values == d.image_id.values).all()
ss = s.sort_values('image_id').drop_duplicates('grp'); gidx = ss.index.values; Xg = Xa.iloc[gidx].reset_index(drop=True); dg = ss.reset_index(drop=True); yg = dg.anemia.values
assert len(dg) == 337
def single_run(X, y, name, seed, inner=0):
    est, grid = MODELS[name]; P = np.zeros(len(y)); per = []
    for tr, te in folds(y, None, seed=seed):
        gs = GridSearchCV(est, grid, scoring='roc_auc', cv=StratifiedKFold(5, shuffle=True, random_state=inner)).fit(X.iloc[tr], y[tr])
        P[te] = gs.predict_proba(X.iloc[te])[:, 1]; per.append(float(roc_auc_score(y[te], P[te])))
    return P, per
# ---------------- A ----------------
if 'single' not in R: R['single'] = {}
for setname, X, y in (('471', Xd, yd), ('geo', Xg, yg)):
    for name in (('RF (color)', 'XGBoost (color)') if setname == '471' else ('RF (color)',)):
        for seed in (42, 1, 2):
            key = '%s|%s|%d' % (setname, name, seed)
            if key in R['single']: continue
            P, per = single_run(X, y, name, seed)
            R['single'][key] = dict(AUC=float(roc_auc_score(y, P)), per_fold=per, oof=P.tolist()); save(); log('A', key, round(R['single'][key]['AUC'], 4))
# ---------------- B ----------------
def hue_rot_feats(im):
    m = mask_of(im); px = im[m]
    if len(px) < 20: px = im.reshape(-1, 3)
    h = cv2.cvtColor(px[None].astype(np.uint8), cv2.COLOR_RGB2HSV)[0][:, 0].astype(np.float64)
    hr = (h + 90.0) % 180.0                         # OpenCV hue is 0-179 (degrees / 2); red moved from the wrap to the middle
    th = np.deg2rad(h * 2.0); C, S = np.cos(th).mean(), np.sin(th).mean(); Rl = np.hypot(C, S)
    p10, p50, p90 = np.percentile(hr, [10, 50, 90])
    return {'H_mean': hr.mean(), 'H_std': hr.std(), 'H_p10': p10, 'H_p50': p50, 'H_p90': p90,
            'circ_mean_deg': float(np.rad2deg(np.arctan2(S, C)) % 360), 'circ_sd_deg': float(np.rad2deg(np.sqrt(-2 * np.log(max(Rl, 1e-12)))))}
def wb(im, g=(1.05, 1.0, 0.95)): return np.clip(im.astype(np.float32) * np.array(g, np.float32), 0, 255).astype(np.uint8)
def icc31(a, b):
    x = np.c_[a, b]; n, k = x.shape; gm = x.mean(); ssr = k * ((x.mean(1) - gm) ** 2).sum(); ssc = n * ((x.mean(0) - gm) ** 2).sum(); sst = ((x - gm) ** 2).sum()
    msr = ssr / (n - 1); mse = (sst - ssr - ssc) / ((n - 1) * (k - 1)); return float((msr - mse) / (msr + (k - 1) * mse))
if 'hue' not in R:
    imgs = [read_rgb(p) for p in dd.image_path]
    HR = pd.DataFrame([hue_rot_feats(im) for im in imgs]); HW = pd.DataFrame([hue_rot_feats(wb(im)) for im in imgs])
    F0 = Xd; FW = pd.DataFrame([color_features(wb(im)) for im in imgs])[F0.columns]
    hcols = ['H_mean', 'H_std', 'H_p10', 'H_p50', 'H_p90']
    wrap = []
    for im in imgs:
        m = mask_of(im); h = cv2.cvtColor(im[m][None].astype(np.uint8), cv2.COLOR_RGB2HSV)[0][:, 0]
        wrap.append(bool(((h <= 10).mean() > 0.05) and ((h >= 170).mean() > 0.05)))
    R['hue'] = dict(n_wrap_images=int(np.sum(wrap)), n=len(imgs),
        icc_wb_linear={c: icc31(F0[c].values, FW[c].values) for c in hcols}, icc_wb_rotated={c: icc31(HR[c].values, HW[c].values) for c in hcols},
        icc_wb_circ_sd=icc31(HR['circ_sd_deg'].values, HW['circ_sd_deg'].values))
    Xr = Xd.copy()
    for c in hcols: Xr[c] = HR[c].values
    Xr.to_csv(OUT + 'feat_dedup_huerot.csv', index=False); save(); log('B icc', R['hue'])
Xr = pd.read_csv(OUT + 'feat_dedup_huerot.csv')
gpos = pd.Index(dd.image_id).get_indexer(dg.image_id); assert (gpos >= 0).all()
Xrg = Xr.iloc[gpos].reset_index(drop=True)
for setname, X, y, models in (('471', Xr, yd, ('RF (color)', 'XGBoost (color)')), ('geo', Xrg, yg, ('RF (color)',))):
    for name in models:
        key = 'huerot|%s|%s' % (setname, name)
        if key in R: continue
        P = np.zeros((5, len(y))); per = []
        for r in range(5):
            est, grid = MODELS[name]
            for tr, te in folds(y, None, seed=SEED + r):
                gs = GridSearchCV(est, grid, scoring='roc_auc', cv=StratifiedKFold(5, shuffle=True, random_state=r)).fit(X.iloc[tr], y[tr]); P[r, te] = gs.predict_proba(X.iloc[te])[:, 1]
            per.append(float(roc_auc_score(y, P[r])))
        R[key] = dict(per_rep=per, mean=float(np.mean(per)), sd=float(np.std(per, ddof=1)), avg=float(roc_auc_score(y, P.mean(0)))); save(); log('B', key, R[key])
# ---------------- C ----------------
if 'shap_grouped' not in R:
    import shap
    est, grid = MODELS['XGBoost (color)']
    gs = _GS(est, grid, scoring='roc_auc', cv=StratifiedKFold(5, shuffle=True, random_state=0)).fit(Xd, yd)
    sv = shap.TreeExplainer(gs.best_estimator_).shap_values(Xd)
    fam = {}
    for j, c in enumerate(Xd.columns):
        base = c.split('_')[0] if c not in ('a_over_L', 'R_over_G', 'frac_high_a') else 'ratios'
        fam.setdefault(base, []).append(j)
    broad = {'luminance (R, G, B, V, L*)': ['R', 'G', 'B', 'V', 'L'], 'red-green balance (rg, gg, EI, R/G, a*/L*)': ['rg', 'gg', 'EI', 'ratios'],
             'CIELAB colour (a*, b*)': ['a', 'b'], 'hue and saturation (H, S)': ['H', 'S']}
    out = {k: float(np.abs(sv[:, idx].sum(1)).mean()) for k, idx in fam.items()}
    outb = {k: float(np.abs(sv[:, sum((fam[f] for f in v), [])].sum(1)).mean()) for k, v in broad.items()}
    R['shap_grouped'] = dict(family=dict(sorted(out.items(), key=lambda t: -t[1])), broad=dict(sorted(outb.items(), key=lambda t: -t[1])),
                             top_single=pd.Series(np.abs(sv).mean(0), index=Xd.columns).sort_values(ascending=False).head(12).round(4).to_dict(),
                             corr_EI_RG=float(np.corrcoef(Xd['EI_mean'], Xd['R_over_G'])[0, 1]), corr_EI_rg=float(np.corrcoef(Xd['EI_mean'], Xd['rg_mean'])[0, 1]))
    save(); log('C', R['shap_grouped'])
log('done')
