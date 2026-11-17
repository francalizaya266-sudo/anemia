"""Grad-CAM of the predicted class and tissue occlusion on the GEOMETRIC set (337 images, one per geometric group).
Answers the first observation of the course review: the analysis of gradcam_review2.py was run on the 471-image set,
which still holds rotated copies and second photographs of the same eye, so a held-out image could have another image
of the same eye in its training folds. Here every image is the only representative of its geometric group
(meta_groups_geo.csv), so the stratified folds cannot split an eye between training and test.

For one model, one recipe and one seed (SEED environment variable) the script retrains the network fold by fold and,
for every held-out image, computes exactly what gradcam_review2.py computed:
  * Grad-CAM for the anaemia logit and for the PREDICTED class (+logit if p >= 0.5, -logit otherwise);
  * the in-mask ratio of both maps (mass inside the conjunctiva mask / mask area; 1 = chance), whether the rectified
    map is identically zero, and the raw 7 x 7 rectified grids;
  * predictions on two occluded versions of the image: 'silhouette' (mask filled with the mean tissue colour of the
    training fold: shape only) and 'meanfill' (mask filled with the image's own mean colour: shape + mean colour);
  * the ceiling of the in-mask ratio that a non-negative 7 x 7 map can reach for that image.
Recipes (the two recipes of the performance table, same code):
  comun     train_torch.train: letter-boxed input, BatchNorm frozen, fine-tuning learning rate 5e-5.
  ajustada  review2_cnn.train_tuned: tissue-filling input, BatchNorm trainable, fine-tuning learning rate chosen per
            training fold among 5e-5, 2e-4 and 5e-4 on the inner validation split.
Writes out/gradcam_geo/gradcam_geo_<model>_<recipe>_s<seed>.json and .npz and skips the run if the JSON already
exists, so the batch can be resumed.   Uso: python gradcam_geo.py ajustada EfficientNet-B0"""
import sys, os, json, time, numpy as np, torch, cv2
from scipy.stats import wilcoxon
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import train_torch as T
import review2_cnn as R2            # load(), stretch(), train_tuned() and the smoke-test switch (R2_SMOKE)
OUTG = os.path.join(R2.HERE, 'out', 'gradcam_geo'); os.makedirs(OUTG, exist_ok=True)

def cam_fn(m):
    feat = m.b.features if hasattr(m.b, 'features') else m.b.layer4
    acts = {}; grads = {}
    h1 = feat.register_forward_hook(lambda mod, i, o: acts.__setitem__('a', o))
    h2 = feat.register_full_backward_hook(lambda mod, gi, go: grads.__setitem__('g', go[0]))
    def gradcam(x, sign):
        xb = torch.from_numpy(x[None]).permute(0, 3, 1, 2).float().to(T.DEV).requires_grad_(True)
        m.eval(); out = m(xb); m.zero_grad(); (sign * out[0, 0]).backward()
        a = acts['a'][0]; g = grads['g'][0]; w = g.mean((1, 2), keepdim=True)
        grid = torch.relu((a * w).sum(0)).detach().float().cpu().numpy()
        cam = cv2.resize(grid, (T.S, T.S)); cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, grid, float(torch.sigmoid(out[0, 0]))
    return gradcam, (h1, h2)

def med(v): return float(np.median(v)) if len(v) else None
def frac(v): return float((v > 1).mean()) if len(v) else None

def stats_of(ratio, zero, ps):
    rng = np.random.default_rng(SEED); boots = [ratio[rng.integers(0, len(ratio), len(ratio))].mean() for _ in range(2000)]
    w = wilcoxon(ratio - 1, alternative='greater'); neg = ps < 0.5
    return {'ratio_mean': float(ratio.mean()), 'ratio_ci': [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], 'ratio_median': float(np.median(ratio)),
            'frac_ratio_gt1': float((ratio > 1).mean()), 'wilcoxon_p_greater_than_1': float(w.pvalue), 'n_zero_maps': int(zero.sum()), 'n_ratio0': int((ratio == 0).sum()),
            'median_pred_pos': med(ratio[~neg]), 'median_pred_neg': med(ratio[neg]), 'frac_gt1_pred_pos': frac(ratio[~neg]), 'frac_gt1_pred_neg': frac(ratio[neg]),
            'n_pred_pos': int((~neg).sum()), 'n_pred_neg': int(neg.sum())}

def ceiling(masks, area):
    """Best in-mask ratio a non-negative 7 x 7 map can reach after the same bilinear up-sampling (as in review2_images.py)."""
    B = []
    for i in range(49):
        g = np.zeros((7, 7), np.float32); g.flat[i] = 1; B.append(cv2.resize(g, (T.S, T.S)))
    Bs = np.array(B).reshape(49, -1); tot = Bs.sum(1)
    return np.array([((Bs @ masks[i].ravel().astype(np.float32)) / tot).max() / area[i] for i in range(len(masks))])

if __name__ == '__main__':
    recipe, name = sys.argv[1], sys.argv[2]
    assert recipe in ('comun', 'ajustada') and name in T.ARCH, 'Uso: python gradcam_geo.py comun|ajustada MODELO'
    tag = f'gradcam_geo_{name}_{recipe}_s{SEED}'
    if os.path.exists(os.path.join(OUTG, tag + '.json')): print('ya existe', tag); sys.exit(0)
    d = R2.load('meta_groups_geo.csv')
    assert d.grp.is_unique and d.groupby('grp').anemia.nunique().max() == 1      # one image per geometric group
    if R2.SMOKE: d = d.groupby('anemia').head(30).reset_index(drop=True)
    y = d.anemia.values; ids = d.image_id.values; n = len(y)
    prep = R2.stretch if recipe == 'ajustada' else T.tight
    X = np.stack([prep(read_rgb(p), T.S) for p in d.image_path])
    masks = np.array([mask_of(x) for x in X]); area = masks.mean((1, 2)); ceil = ceiling(masks, area)
    print(tag, 'n =', n, 'device =', T.DEV, flush=True)
    t0 = time.time(); fold_of = np.full(n, -1); lrs = []; per_fold = []
    ra = np.full(n, np.nan); rc = np.full(n, np.nan); za = np.zeros(n, bool); zc = np.zeros(n, bool); ps = np.full(n, np.nan)
    p_sil = np.full(n, np.nan); p_mf = np.full(n, np.nan); grids_a = None; grids_c = None
    for k, (tr, te) in enumerate(folds(y, None, seed=SEED)):
        torch.manual_seed(SEED); np.random.seed(SEED); fold_of[te] = k
        if recipe == 'ajustada':
            m, lr, _ = R2.train_tuned(name, X[tr], y[tr]); lrs.append(float(lr))
        else:
            m = T.train(name, X[tr], y[tr])
        gc, hooks = cam_fn(m)
        tissue_mean = np.concatenate([X[i][masks[i]] for i in tr]).mean(0)
        Xs = X[te].copy(); Xm = X[te].copy()
        for j, i in enumerate(te):
            Xs[j][masks[i]] = tissue_mean.astype(np.uint8); Xm[j][masks[i]] = X[i][masks[i]].mean(0).astype(np.uint8)
        p_sil[te] = T.predict(m, Xs); p_mf[te] = T.predict(m, Xm)
        for i in te:
            c, g, p = gc(X[i], 1.0); ps[i] = p
            if grids_a is None: grids_a = np.zeros((n,) + g.shape, np.float32); grids_c = np.zeros((n,) + g.shape, np.float32)
            grids_a[i] = g; ra[i] = (c * masks[i]).sum() / (c.sum() + 1e-8) / area[i]; za[i] = g.max() <= 0
            c2, g2, _ = gc(X[i], 1.0 if p >= 0.5 else -1.0)
            grids_c[i] = g2; rc[i] = (c2 * masks[i]).sum() / (c2.sum() + 1e-8) / area[i]; zc[i] = g2.max() <= 0
        for h in hooks: h.remove()
        per_fold.append(float(roc_auc_score(y[te], ps[te])))
        print(f'{tag} fold {k} AUC {per_fold[-1]:.3f} ratio anaemia-logit {np.mean(ra[te]):.2f} predicted-class {np.mean(rc[te]):.2f} {time.time() - t0:.0f}s', flush=True)
    res = {'model': name, 'recipe': recipe, 'seed': SEED, 'set': 'geometric', 'n': int(n), 'AUC_oof_fp32': float(roc_auc_score(y, ps)), 'per_fold_AUC': per_fold,
           'AUC_silhouette': float(roc_auc_score(y, p_sil)), 'AUC_meanfill': float(roc_auc_score(y, p_mf)), 'AUC_mask_area': float(roc_auc_score(y, -area)),
           'mask_area_mean': float(area.mean()), 'ceiling_median': float(np.median(ceil)), 'ceiling_p10': float(np.percentile(ceil, 10)), 'ceiling_p90': float(np.percentile(ceil, 90)),
           'anaemia_logit': stats_of(ra, za, ps), 'predicted_class': stats_of(rc, zc, ps), 'chosen_lr_per_fold': lrs, 'train_time_s': time.time() - t0,
           'torch': torch.__version__, 'device': T.DEV}
    np.savez_compressed(os.path.join(OUTG, tag + '.npz'), ratio_a=ra, ratio_c=rc, zero_a=za, zero_c=zc, grid_a=grids_a, grid_c=grids_c,
                        p=ps, p_sil=p_sil, p_meanfill=p_mf, y=y, ids=ids, fold=fold_of, area=area, ceiling=ceil)
    json.dump(res, open(os.path.join(OUTG, tag + '.json'), 'w'), indent=1, default=float)
    print(json.dumps(res, default=float), flush=True)
    print('LISTO', tag, flush=True)
