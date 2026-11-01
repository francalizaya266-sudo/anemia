r"""Integrate the laptop GPU results of the peer-review runs.
usage: python integrate_gpu.py <gpu_out_dir> [strict_out_dir]
 gpu_out_dir : copy of "Nueva carpeta\out" with cnn_<model>_grouped.json and gradcam_review.json
 strict_out_dir : copy of "Nueva carpeta\strict\out" (optional)
writes /home/claude/exp/out/review/gpu_review.json
"""
import sys, os, json, glob, numpy as np
sys.path.insert(0, '/home/claude/exp')
from review_analyses_lib import paired_boot_diff, delong, auc_ci
from sklearn.metrics import roc_auc_score
G = sys.argv[1]; S = sys.argv[2] if len(sys.argv) > 2 else None
OUTF = '/home/claude/exp/out/review/gpu_review.json'
MODELS = ['EfficientNet-B0', 'MobileNetV2', 'MobileNetV3-Small', 'MobileNetV3-Large', 'ShuffleNetV2', 'ResNet50', 'EfficientNetV2-S']
NAIVE_DIRS = [G, '/home/claude/gpu_out', '/mnt/user-data/uploads/AnemIA-E2/Nueva carpeta/out']
R = {'grouped': {}, 'gradcam': {}, 'strict': {}}
def load(p): return json.load(open(p))
for m in MODELS:
    f = os.path.join(G, f'cnn_{m}_grouped.json')
    if not os.path.exists(f): print('missing', f); continue
    d = load(f); y = np.array(d['y']); pg = np.array(d['oof']['fp32'])
    e = {'AUC': float(d['res']['fp32']['AUC']), 'ci': [float(d['res']['fp32_ci']['AUC'][0]), float(d['res']['fp32_ci']['AUC'][1])] if isinstance(d['res']['fp32_ci'], dict) else list(d['res']['fp32_ci']),
         'n': int(d['res']['n']), 'groups': int(len(set(d['grp']))), 'train_time_s': float(d['res']['train_time_s'])}
    assert abs(e['AUC'] - roc_auc_score(y, pg)) < 1e-6
    nf = next((os.path.join(nd, f'cnn_{m}_naive.json') for nd in NAIVE_DIRS if os.path.exists(os.path.join(nd, f'cnn_{m}_naive.json'))), None)
    if nf:
        dn = load(nf); yn = np.array(dn['y']); pn = np.array(dn['oof']['fp32'])
        if dn['res'].get('device') is None and '/exp/out/' in nf: print('skip CPU naive', nf); nf = None
    if nf and np.array_equal(yn, y):
        ci, pb = paired_boot_diff(y, pn, pg); dd, z, p = delong(y, pn, pg)
        e.update({'naive_AUC': float(roc_auc_score(y, pn)), 'naive_file': nf, 'naive_minus_grouped': float(roc_auc_score(y, pn) - e['AUC']), 'naive_minus_grouped_ci': ci, 'boot_p': pb, 'delong_p': p, 'delong_z': z})
    elif nf: print('row mismatch naive/grouped for', m)
    R['grouped'][m] = e; print(m, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in e.items() if k != 'naive_file'}))
def gc_extra(r, p):
    """Extra Grad-CAM statistics from the per-image arrays: split by predicted class, and the degenerate
    all-zero maps (rectified Grad-CAM with no positive gradient for the anaemia logit; ratio recorded as 0)."""
    z0 = r == 0; nz = ~z0; rng = np.random.default_rng(0)
    boots = [r[nz][rng.integers(0, nz.sum(), nz.sum())].mean() for _ in range(2000)]
    return {'frac_lt1_pred_neg': float((r[p < 0.5] < 1).mean()), 'frac_lt1_pred_pos': float((r[p >= 0.5] < 1).mean()), 'n_pred_neg': int((p < 0.5).sum()), 'n_pred_pos': int((p >= 0.5).sum()),
            'median_pred_pos': float(np.median(r[p >= 0.5])), 'median_pred_neg': float(np.median(r[p < 0.5])),
            'n_zero_maps': int(z0.sum()), 'n_zero_maps_pred_neg': int((z0 & (p < 0.5)).sum()), 'n': int(len(r)),
            'ratio_mean_excl0': float(r[nz].mean()), 'ratio_ci_excl0': [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            'ratio_median_excl0': float(np.median(r[nz])), 'frac_ratio_gt1_excl0': float((r[nz] > 1).mean()), 'frac_lt1_pred_neg_excl0': float((r[nz & (p < 0.5)] < 1).mean())}

gf = os.path.join(G, 'gradcam_review.json')
if os.path.exists(gf):
    gj = load(gf); R['gradcam'] = gj['summary']; R['gradcam_examples'] = gj['examples']
    for m in R['gradcam']:
        zf = os.path.join(G, f'gradcam_review_{m}.npz')
        if os.path.exists(zf):
            z = np.load(zf); r = z['ratio']; p = z['p']
            R['gradcam'][m].update(gc_extra(r, p))
    for m, s in gj['summary'].items(): print('gradcam', m, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()})
if S and os.path.isdir(S):
    for f in sorted(glob.glob(os.path.join(S, 'cnn_*.json'))):
        d = load(f); r = d['res']; key = os.path.basename(f)[4:-5]
        fp = r['fp32'] if isinstance(r['fp32'], dict) else {'AUC': r['fp32']}
        ci = r['fp32_ci']['AUC'] if isinstance(r['fp32_ci'], dict) else r['fp32_ci']
        R['strict'][key] = {'AUC': float(fp['AUC']), 'ci': [float(ci[0]), float(ci[1])], 'n': int(r['n']), 'proto': r.get('proto'), 'seed': r.get('seed'), **{k: float(fp[k]) for k in ('F1', 'Acc', 'Sens', 'Spec', 'Brier') if k in fp}}
        print('strict', key, round(R['strict'][key]['AUC'], 4))
    for m in MODELS:
        fn_, fg_ = os.path.join(S, f'cnn_{m}_naive.json'), os.path.join(S, f'cnn_{m}_grouped.json')
        if os.path.exists(fn_) and os.path.exists(fg_):
            dn, dg = load(fn_), load(fg_); y = np.array(dn['y']); pn = np.array(dn['oof']['fp32']); pg = np.array(dg['oof']['fp32'])
            assert np.array_equal(y, np.array(dg['y']))
            ci_, pb = paired_boot_diff(y, pn, pg); dd, z, p = delong(y, pn, pg)
            R['strict'][m + '_grouped'].update({'naive_minus_grouped_strict': float(roc_auc_score(y, pn) - roc_auc_score(y, pg)), 'naive_minus_grouped_strict_ci': ci_, 'boot_p': pb, 'delong_p': p})
            print('strict leak', m, round(R['strict'][m + '_grouped']['naive_minus_grouped_strict'], 4), ci_, 'p', p)
    sg = os.path.join(S, 'gradcam_review.json')
    if os.path.exists(sg):
        R['strict_gradcam'] = load(sg)['summary']
        for m in R['strict_gradcam']:
            zf = os.path.join(S, f'gradcam_review_{m}.npz')
            if os.path.exists(zf): z = np.load(zf); R['strict_gradcam'][m].update(gc_extra(z['ratio'], z['p']))
json.dump(R, open(OUTF, 'w'), indent=1); print('written', OUTF)
