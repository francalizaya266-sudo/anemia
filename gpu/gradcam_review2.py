"""Grad-CAM for the predicted class and tissue/background occlusion (second internal review).
Retrains EfficientNet-B0 and MobileNetV2 fold by fold (seed-42 folds, same recipe as train_torch.py) and, for every
held-out image of the 471-image set, computes
  * Grad-CAM for the anaemia logit (as in gradcam_review.py) and for the PREDICTED class (+logit if p >= 0.5,
    -logit otherwise), so that maps of images predicted non-anaemic are no longer zero by construction;
  * the in-mask ratio of both maps (mass inside the conjunctiva mask / mask area; 1 = chance) and whether the
    rectified map is identically zero;
  * the raw 7 x 7 rectified grids of both maps (for re-plotting Fig. 6 exactly);
  * predictions on two occluded versions of the image: 'silhouette' (mask filled with the mean tissue colour of the
    training fold: shape only) and 'meanfill' (mask filled with the image's own mean colour: shape + mean colour).
Writes out/gradcam_review2.json and out/gradcam_review2_<model>.npz.   Uso: python gradcam_review2.py"""
import sys, os, json, time, numpy as np, torch, cv2
from scipy.stats import wilcoxon
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import train_torch as T

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

def stats_of(ratio, zero, ps):
    rng = np.random.default_rng(SEED); boots = [ratio[rng.integers(0, len(ratio), len(ratio))].mean() for _ in range(2000)]
    w = wilcoxon(ratio - 1, alternative='greater'); neg = ps < 0.5
    return {'ratio_mean': float(ratio.mean()), 'ratio_ci': [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], 'ratio_median': float(np.median(ratio)),
            'frac_ratio_gt1': float((ratio > 1).mean()), 'wilcoxon_p_greater_than_1': float(w.pvalue), 'n_zero_maps': int(zero.sum()), 'n_ratio0': int((ratio == 0).sum()),
            'median_pred_pos': float(np.median(ratio[~neg])), 'median_pred_neg': float(np.median(ratio[neg])), 'frac_gt1_pred_pos': float((ratio[~neg] > 1).mean()),
            'frac_gt1_pred_neg': float((ratio[neg] > 1).mean()), 'n_pred_pos': int((~neg).sum()), 'n_pred_neg': int(neg.sum())}

d = dedup(load_meta()); y = d.anemia.values; hb = d.hb.values; ids = d.image_id.values
X = np.stack([T.tight(read_rgb(p), T.S) for p in d.image_path])
masks = np.array([mask_of(x) for x in X]); area = masks.mean((1, 2))
summary = {}
for name in ['EfficientNet-B0', 'MobileNetV2']:
    t0 = time.time(); n = len(y)
    ra = np.full(n, np.nan); rc = np.full(n, np.nan); za = np.zeros(n, bool); zc = np.zeros(n, bool); ps = np.full(n, np.nan)
    p_sil = np.full(n, np.nan); p_mf = np.full(n, np.nan); grids_a = None; grids_c = None
    for k, (tr, te) in enumerate(folds(y, None, seed=SEED)):
        m = T.train(name, X[tr], y[tr]); gc, hooks = cam_fn(m)
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
        print(f'{name} fold {k} AUC {roc_auc_score(y[te], ps[te]):.3f} ratio anaemia-logit {np.mean(ra[te]):.2f} predicted-class {np.mean(rc[te]):.2f} {time.time() - t0:.0f}s', flush=True)
    summary[name] = {'n': int(n), 'AUC_oof_fp32': float(roc_auc_score(y, ps)), 'AUC_silhouette': float(roc_auc_score(y, p_sil)), 'AUC_meanfill': float(roc_auc_score(y, p_mf)),
                     'mask_area_mean': float(area.mean()), 'anaemia_logit': stats_of(ra, za, ps), 'predicted_class': stats_of(rc, zc, ps), 'train_time_s': time.time() - t0}
    np.savez_compressed(os.path.join(T.OUT, f'gradcam_review2_{name}.npz'), ratio_a=ra, ratio_c=rc, zero_a=za, zero_c=zc, grid_a=grids_a, grid_c=grids_c,
                        p=ps, p_sil=p_sil, p_meanfill=p_mf, y=y, hb=hb, ids=ids)
    print(json.dumps(summary[name]), flush=True)
json.dump({'summary': summary}, open(os.path.join(T.OUT, 'gradcam_review2.json'), 'w'), indent=1, default=float)
print('LISTO gradcam2', flush=True)
