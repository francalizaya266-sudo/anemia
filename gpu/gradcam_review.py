"""Grad-CAM over the five folds (peer-review request): retrains EfficientNet-B0 and MobileNetV2 fold by fold
(seed 42 folds, same recipe as train_torch.py), computes for every held-out image the fraction of Grad-CAM
mass inside the conjunctiva mask divided by the mask area (ratio; 1 = chance), and reports the mean ratio with
a bootstrap 95 % CI, the Wilcoxon signed-rank test against 1 and the fraction of images with ratio > 1.
Also renders the publication figure (TrueType fonts, no internal title, viridis overlay, mask contour,
per-image labels) and writes the per-example values.  Uso: python gradcam_review.py"""
import sys, os, json, time, numpy as np, torch, cv2
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import train_torch as T
plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.family': 'DejaVu Sans', 'font.size': 7, 'savefig.bbox': 'tight', 'figure.dpi': 200})

def cam_fn(m):
    feat = m.b.features if hasattr(m.b, 'features') else m.b.layer4
    acts = {}; grads = {}
    h1 = feat.register_forward_hook(lambda mod, i, o: acts.__setitem__('a', o))
    h2 = feat.register_full_backward_hook(lambda mod, gi, go: grads.__setitem__('g', go[0]))
    def gradcam(x):
        xb = torch.from_numpy(x[None]).permute(0, 3, 1, 2).float().to(T.DEV).requires_grad_(True)
        m.eval(); out = m(xb); m.zero_grad(); out[0, 0].backward()
        a = acts['a'][0]; g = grads['g'][0]; w = g.mean((1, 2), keepdim=True)
        cam = torch.relu((a * w).sum(0)).detach().float().cpu().numpy(); cam = cv2.resize(cam, (T.S, T.S)); cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, float(torch.sigmoid(out[0, 0]))
    return gradcam, (h1, h2)

d = dedup(load_meta()); y = d.anemia.values; hb = d.hb.values; ids = d.image_id.values
X = np.stack([T.tight(read_rgb(p), T.S) for p in d.image_path])
masks = np.array([mask_of(x) for x in X]); area = masks.mean((1, 2))
summary = {}; examples = {}; cams_keep = {}
for name in ['EfficientNet-B0', 'MobileNetV2']:
    t0 = time.time(); ratio = np.full(len(y), np.nan); inside = np.full(len(y), np.nan); ps = np.full(len(y), np.nan); cams = np.zeros((len(y), T.S, T.S), np.float16)
    for k, (tr, te) in enumerate(folds(y, None, seed=SEED)):
        m = T.train(name, X[tr], y[tr]); gc, hooks = cam_fn(m)
        for i in te:
            c, p = gc(X[i]); cams[i] = c; ps[i] = p
            inside[i] = (c * masks[i]).sum() / (c.sum() + 1e-8); ratio[i] = inside[i] / area[i]
        for h in hooks: h.remove()
        print(f'{name} fold {k} AUC {roc_auc_score(y[te], ps[te]):.3f} mean ratio {np.nanmean(ratio[te]):.2f} {time.time() - t0:.0f}s', flush=True)
    rng = np.random.default_rng(SEED); boots = [ratio[rng.integers(0, len(y), len(y))].mean() for _ in range(2000)]
    w = wilcoxon(ratio - 1, alternative='greater')
    summary[name] = {'n': int(len(y)), 'AUC_oof_fp32': float(roc_auc_score(y, ps)), 'mask_area_mean': float(area.mean()), 'cam_mass_inside_mask_mean': float(inside.mean()),
                     'ratio_mean': float(ratio.mean()), 'ratio_ci': [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], 'ratio_median': float(np.median(ratio)),
                     'frac_ratio_gt1': float((ratio > 1).mean()), 'wilcoxon_stat': float(w.statistic), 'wilcoxon_p_greater_than_1': float(w.pvalue),
                     'ratio_mean_fold5_only': float(ratio[folds(y, None, seed=SEED)[-1][1]].mean()), 'train_time_s': time.time() - t0}
    np.savez_compressed(os.path.join(T.OUT, f'gradcam_review_{name}.npz'), ratio=ratio, inside=inside, p=ps, y=y, hb=hb, ids=ids)
    order = np.argsort(-np.abs(ps - 0.5))
    tp = [i for i in order if y[i] == 1 and ps[i] >= 0.5][:3]; tn = [i for i in order if y[i] == 0 and ps[i] < 0.5][:3]; err = [i for i in order if (ps[i] >= 0.5) != y[i]][:2]
    sel = tp + tn + err; examples[name] = [{'image_id': str(ids[i]), 'hb_gL': float(hb[i] * 10), 'label': int(y[i]), 'p': float(ps[i]), 'ratio': float(ratio[i]), 'kind': ['TP'] * 3 + ['TN'] * 3 + ['error'] * 2} for i in sel]
    for j, e in enumerate(examples[name]): e['kind'] = e['kind'][j]
    cams_keep[name] = (sel, cams)
    print(json.dumps(summary[name]), flush=True)
json.dump({'summary': summary, 'examples': examples}, open(os.path.join(T.OUT, 'gradcam_review.json'), 'w'), indent=1, default=float)
# ---- figure ----
fig = plt.figure(figsize=(6.5, 4.9)); gs = fig.add_gridspec(2, 1, hspace=0.30)
for pi, name in enumerate(['EfficientNet-B0', 'MobileNetV2']):
    sel, cams = cams_keep[name]; sub = gs[pi].subgridspec(2, 8, wspace=0.05, hspace=0.05)
    for k, i in enumerate(sel):
        a = fig.add_subplot(sub[0, k]); a.imshow(X[i]); a.set_xticks([]); a.set_yticks([])
        e = examples[name][k]; a.set_title(f"{e['kind']}  Hb {e['hb_gL']:.0f}\np {e['p']:.2f}  ratio {e['ratio']:.1f}", fontsize=6, pad=2)
        b = fig.add_subplot(sub[1, k]); b.imshow(X[i]); b.imshow(cams[i].astype(np.float32), cmap='viridis', alpha=0.55, vmin=0, vmax=1)
        b.contour(masks[i].astype(float), levels=[0.5], colors='white', linewidths=0.5); b.set_xticks([]); b.set_yticks([])
        for s in list(a.spines.values()) + list(b.spines.values()): s.set_visible(False)
        if k == 0: a.set_ylabel('ROI', fontsize=7); b.set_ylabel('Grad-CAM', fontsize=7)
    fig.text(0.125, 0.905 if pi == 0 else 0.475, f"({'ab'[pi]}) {name}: mean in-mask ratio {summary[name]['ratio_mean']:.2f} (95 % CI {summary[name]['ratio_ci'][0]:.2f}-{summary[name]['ratio_ci'][1]:.2f}), {100*summary[name]['frac_ratio_gt1']:.0f} % of images above chance", fontsize=7.5, ha='left', va='bottom')
sm = plt.cm.ScalarMappable(cmap='viridis'); sm.set_array([0, 1]); cb = fig.colorbar(sm, ax=fig.axes, fraction=0.015, pad=0.01); cb.set_label('Grad-CAM (normalised)', fontsize=7); cb.ax.tick_params(labelsize=6)
plt.savefig(os.path.join(T.OUT, 'Fig6_gradcam.pdf')); plt.savefig(os.path.join(T.OUT, 'Fig6_gradcam.png'))
print('LISTO gradcam', flush=True)
