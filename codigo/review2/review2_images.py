# Second internal review, part 2 (image-based, CPU): Grad-CAM ceiling at the 7x7 resolution, mask area by class,
# pHash Hamming-distance histogram, feature stability (rescaling, JPEG, mask erosion), RF with eroded masks,
# LOHO restricted to singleton images.
import sys, os, json, time; sys.path.insert(0, '/home/claude/exp')
import numpy as np, pandas as pd, cv2, imagehash
from PIL import Image
from scipy import stats
from sklearn.metrics import roc_auc_score
from common import load_meta, dedup, read_rgb, letterbox, mask_of, folds, SEED
from classic import color_features, MODELS, run_cv
OUT = '/home/claude/exp/out/review2/'; R = json.load(open(OUT + 'review2_images.json')) if os.path.exists(OUT + 'review2_images.json') else {}
def save(): json.dump(R, open(OUT + 'review2_images.json', 'w'), indent=1, default=float)
d = load_meta(); dd = dedup(d); y = dd.anemia.values; S = 224
def tight(im, s=S):
    m = mask_of(im); ys, xs = np.where(m)
    return letterbox(im[ys.min():ys.max() + 1, xs.min():xs.max() + 1], s) if len(ys) >= 10 else letterbox(im, s)
imgs = [read_rgb(p) for p in dd.image_path]
X = np.stack([tight(im) for im in imgs]); masks = np.array([mask_of(x) for x in X]); area = masks.mean((1, 2))
# ---- G. Grad-CAM ceiling: best achievable in-mask ratio for a non-negative 7x7 map upsampled bilinearly ----
if 'gradcam_ceiling' not in R:
    B = []
    for i in range(49):
        g = np.zeros((7, 7), np.float32); g.flat[i] = 1; B.append(cv2.resize(g, (S, S)))
    B = np.array(B)                                   # 49 x 224 x 224 (same cv2.resize as gradcam_review.py)
    Bs = B.reshape(49, -1); tot = Bs.sum(1)
    ceil = np.array([((Bs @ masks[i].ravel().astype(np.float32)) / tot).max() / area[i] for i in range(len(y))])
    # a map uniform over the cells that touch the mask (a realistic 'perfect' Grad-CAM)
    R['gradcam_ceiling'] = dict(mean=float(ceil.mean()), median=float(np.median(ceil)), p10=float(np.percentile(ceil, 10)), p90=float(np.percentile(ceil, 90)),
                                mask_area_mean=float(area.mean()))
    np.save(OUT + 'gradcam_ceiling.npy', ceil); save(); print('G', R['gradcam_ceiling'], flush=True)
# ---- H. mask area by class ----
if 'mask_area' not in R:
    a_nat = np.array([mask_of(im).sum() for im in imgs]); a_box = area
    R['mask_area'] = dict(native_px_median_anaemic=float(np.median(a_nat[y == 1])), native_px_median_non=float(np.median(a_nat[y == 0])),
                          native_auc=float(roc_auc_score(y, -a_nat)), letterbox_frac_median_anaemic=float(np.median(a_box[y == 1])), letterbox_frac_median_non=float(np.median(a_box[y == 0])),
                          letterbox_auc=float(roc_auc_score(y, -a_box)), mw_p=float(stats.mannwhitneyu(a_nat[y == 1], a_nat[y == 0]).pvalue))
    save(); print('H', R['mask_area'], flush=True)
# ---- I. pHash Hamming distances (image + horizontal mirror, as in stages 1-2) between all 710 files ----
if 'hamming' not in R:
    mfull = pd.read_csv('/home/claude/anemia/codigo/meta_groups.csv')
    H = []; Hm = []
    for p in mfull.image_path:
        im = Image.fromarray(read_rgb(p)); H.append(imagehash.phash(im, hash_size=16)); Hm.append(imagehash.phash(im.transpose(Image.FLIP_LEFT_RIGHT), hash_size=16))
    hv = np.array([np.array(h.hash, bool).ravel() for h in H]); hm = np.array([np.array(h.hash, bool).ravel() for h in Hm])
    D = np.minimum((hv[:, None, :] != hv[None]).sum(2), (hv[:, None, :] != hm[None]).sum(2))
    iu = np.triu_indices(len(hv), 1); dist = D[iu]; shv = np.asarray(mfull.sha, dtype=object); same_sha = (shv[:, None] == shv[None])[iu]
    dd_ = dist[~same_sha]
    hist = np.bincount(dd_, minlength=257)
    R['hamming'] = dict(n_pairs=int(len(dd_)), hist_0_80=hist[:81].tolist(), quantiles=np.percentile(dd_, [0.01, 0.1, 1, 5, 50]).tolist(), n_le20=int((dd_ <= 20).sum()), n_le10=int((dd_ <= 10).sum()), n_21_25=int(((dd_ > 20) & (dd_ <= 25)).sum()),
                        n_le30=int((dd_ <= 30).sum()), min_gap_note='pairs of non-identical files only')
    save(); print('I', {k: v for k, v in R['hamming'].items() if k != 'hist_0_80'}, R['hamming']['hist_0_80'][:81], flush=True)
# ---- K. feature stability ----
def icc31(a, b):
    Y = np.vstack([a, b]).T; n, k = Y.shape; gm = Y.mean(); ssr = k * ((Y.mean(1) - gm) ** 2).sum(); ssc = n * ((Y.mean(0) - gm) ** 2).sum(); sst = ((Y - gm) ** 2).sum(); sse = sst - ssr - ssc
    msr = ssr / (n - 1); mse = sse / ((n - 1) * (k - 1)); return (msr - mse) / (msr + (k - 1) * mse)
def feats_masked(im, m):
    im2 = im.copy(); im2[~m] = 0; return color_features(im2)
if 'stability' not in R:
    F0 = pd.DataFrame([color_features(im) for im in imgs]); res = {}
    def jpeg(im, q=75):
        ok, buf = cv2.imencode('.jpg', cv2.cvtColor(im, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q]); return cv2.cvtColor(cv2.imdecode(buf, 1), cv2.COLOR_BGR2RGB)
    def rescale(im, f=0.5):
        h, w = im.shape[:2]; s = cv2.resize(im, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA); return cv2.resize(s, (w, h), interpolation=cv2.INTER_LINEAR)
    def erode(im, px=2):
        m = mask_of(im).astype(np.uint8); m = cv2.erode(m, np.ones((2 * px + 1, 2 * px + 1), np.uint8)).astype(bool); return feats_masked(im, m)
    def wb(im, g=(1.05, 1.0, 0.95)):
        out = np.clip(im.astype(np.float32) * np.array(g)[None, None], 0, 255).astype(np.uint8); out[~mask_of(im)] = 0; return out
    variants = {'jpeg75': lambda im: color_features(jpeg(im)), 'rescale50': lambda im: color_features(rescale(im)), 'erode2': lambda im: erode(im, 2),
                'erode3': lambda im: erode(im, 3), 'whitebalance5': lambda im: color_features(wb(im))}
    Fv = {}
    for k, fn in variants.items():
        Fv[k] = pd.DataFrame([fn(im) for im in imgs]); ic = np.array([icc31(F0[c].values, Fv[k][c].values) for c in F0.columns])
        res[k] = dict(icc_median=float(np.nanmedian(ic)), icc_min=float(np.nanmin(ic)), n_below_090=int((ic < 0.9).sum()), n_below_075=int((ic < 0.75).sum()),
                      worst=[[c, float(v)] for c, v in sorted(zip(F0.columns, ic), key=lambda t: t[1])[:5]])
        print('K', k, res[k], flush=True)
    R['stability'] = res; save()
    for k in ('erode2', 'jpeg75', 'rescale50'):
        Fv[k].to_csv(OUT + f'feat_dedup_{k}.csv', index=False)
# RF AUC with eroded-mask features and on perturbed images (model trained on original features, 1 repetition, nested)
if 'rf_perturbed' not in R:
    est, grid = MODELS['RF (color)']; F0 = pd.read_csv('/home/claude/exp/out/feat_dedup.csv'); out = {}
    for k in ('erode2',):
        Fk = pd.read_csv(OUT + f'feat_dedup_{k}.csv')[F0.columns]; P = run_cv(Fk, y, None, 'RF', est, grid, reps=1); out[k + '_retrained_AUC'] = float(roc_auc_score(y, P[0]))
    # train on original, test on perturbed (same folds)
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    Fp = {k: pd.read_csv(OUT + f'feat_dedup_{k}.csv')[F0.columns] for k in ('jpeg75', 'rescale50', 'erode2')}; Pt = {k: np.zeros(len(y)) for k in Fp}; P0 = np.zeros(len(y))
    for tr, te in folds(y, None, seed=SEED):
        gs = GridSearchCV(est, grid, scoring='roc_auc', cv=StratifiedKFold(5, shuffle=True, random_state=0), n_jobs=1).fit(F0.iloc[tr], y[tr])
        P0[te] = gs.predict_proba(F0.iloc[te])[:, 1]
        for k in Fp: Pt[k][te] = gs.predict_proba(Fp[k].iloc[te])[:, 1]
    out['orig_AUC_rep1'] = float(roc_auc_score(y, P0)); out.update({k + '_test_only_AUC': float(roc_auc_score(y, Pt[k])) for k in Pt})
    R['rf_perturbed'] = out; save(); print('K2', out, flush=True)
# ---- F2. LOHO on singleton images only (centre unambiguous) ----
if 'loho_singletons' not in R:
    sz = d.groupby('grp').grp.transform('size').values; ds = d[sz == 1].reset_index(drop=True); Fs = pd.DataFrame([color_features(read_rgb(p)) for p in ds.image_path])
    ys = ds.anemia.values; hosp = ds.hospital.values; est, grid = MODELS['RF (color)']; P = np.zeros(len(ys))
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    for h in np.unique(hosp):
        te = hosp == h; tr = ~te
        gs = GridSearchCV(est, grid, scoring='roc_auc', cv=StratifiedKFold(5, shuffle=True, random_state=0), n_jobs=1).fit(Fs[tr], ys[tr]); P[te] = gs.predict_proba(Fs[te])[:, 1]
    per = []
    for h in np.unique(hosp):
        te = hosp == h
        if te.sum() >= 20 and len(np.unique(ys[te])) == 2: per.append(dict(hospital=h, n=int(te.sum()), n_pos=int(ys[te].sum()), AUC=float(roc_auc_score(ys[te], P[te]))))
    rng = np.random.default_rng(SEED); bo = []
    for _ in range(2000):
        i = rng.integers(0, len(ys), len(ys))
        if len(np.unique(ys[i])) == 2: bo.append(roc_auc_score(ys[i], P[i]))
    R['loho_singletons'] = dict(n=int(len(ys)), centres=int(len(np.unique(hosp))), pooled_AUC=float(roc_auc_score(ys, P)), pooled_ci=[float(np.percentile(bo, 2.5)), float(np.percentile(bo, 97.5))],
                                per_centre=per, range=[min(p['AUC'] for p in per), max(p['AUC'] for p in per)])
    save(); print('F2', {k: v for k, v in R['loho_singletons'].items() if k != 'per_centre'}, flush=True)
print('done')
