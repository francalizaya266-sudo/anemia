"""Second internal review: CNN experiments run on the laptop GPU.
Tasks (each writes its own JSON under out/review2/ and is skipped if that file already exists, so the batch can be resumed):
  int8fix  MODEL        original recipe, seed-42 folds of the 471-image set; every fold model is exported to ONNX twice:
                        (i) as in the manuscript (input normalisation inside the graph, all layers quantised) and
                        (ii) with the normalisation moved outside the graph and the first convolution and the classifier
                        kept in floating point; both are statically quantised to INT8 (QDQ, per channel, 100 calibration
                        images of the training fold) and scored on the held-out fold.
  tuned    MODEL        tuned recipe on the 471-image set: tissue-filling input (bounding box of the mask resized to
                        224 x 224 without letter-boxing, so the conjunctiva fills the frame), BatchNorm trainable, and
                        the fine-tuning learning rate chosen per training fold among 5e-5, 2e-4 and 5e-4 on the inner
                        15 % validation split (the test fold is never used); per-epoch validation curves are saved.
  geo      MODEL        original recipe on the geometric set (duplicate map meta_groups_geo.csv: hash stages plus the
                        SIFT/RANSAC + photometric verification stage).
  tunedgeo MODEL        tuned recipe (as in 'tuned') on the geometric set.
The seed comes from the SEED environment variable, as in train_torch.py.   Uso: python review2_cnn.py tuned EfficientNet-B0"""
import sys, os, json, time, copy
import numpy as np, pandas as pd, torch, torch.nn as nn, cv2
from sklearn.model_selection import train_test_split
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import train_torch as T
HERE = os.path.dirname(os.path.abspath(__file__)); OUT2 = os.path.join(HERE, 'out', 'review2'); os.makedirs(OUT2, exist_ok=True)

def load(meta):
    d = pd.read_csv(os.path.join(HERE, meta)); conf = d.groupby('grp').anemia.nunique(); d = d[~d.grp.isin(conf[conf > 1].index)]
    return d.sort_values('image_id').drop_duplicates('grp').reset_index(drop=True)

def stretch(im, s=T.S):
    m = mask_of(im); ys, xs = np.where(m)
    if len(ys) < 10: return letterbox(im, s)
    return cv2.resize(im[ys.min():ys.max() + 1, xs.min():xs.max() + 1], (s, s), interpolation=cv2.INTER_AREA)

def fit_curve(m, params, lr, epochs, Xt, yt, Xv, yv, w, patience=8, bs=32):
    """Same loop as train_torch.fit_phase, but also returns the per-epoch validation AUC."""
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4); best = (-1, None, 0); curve = []
    crit = nn.BCEWithLogitsLoss(reduction='none'); scaler = torch.amp.GradScaler('cuda', enabled=T.DEV == 'cuda')
    for ep in range(epochs):
        m.train(); idx = np.random.permutation(len(Xt))
        for i in range(0, len(idx), bs):
            j = idx[i:i + bs]; xb = torch.from_numpy(Xt[j]).permute(0, 3, 1, 2).float().to(T.DEV); yb = torch.from_numpy(yt[j]).float().to(T.DEV)
            xb = T.augment(xb)
            with torch.autocast('cuda', enabled=T.DEV == 'cuda'):
                loss = (crit(m(xb).squeeze(1), yb) * torch.where(yb == 1, w[1], w[0])).mean()
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        auc = roc_auc_score(yv, T.predict(m, Xv)); curve.append(float(auc))
        if auc > best[0]: best = (auc, copy.deepcopy(m.state_dict()), ep)
        elif ep - best[2] >= patience: break
    m.load_state_dict(best[1]); return m, best[0], curve

SMOKE = bool(os.environ.get('R2_SMOKE'))           # quick functional test on a CPU: 60 images, one epoch per phase
E1, E2 = (1, 1) if SMOKE else (10, 40)
if SMOKE:
    import torchvision as tv                         # no pretrained download in the smoke test
    _fn = {'MobileNetV2': 'mobilenet_v2', 'MobileNetV3-Small': 'mobilenet_v3_small', 'MobileNetV3-Large': 'mobilenet_v3_large', 'EfficientNet-B0': 'efficientnet_b0',
           'EfficientNetV2-S': 'efficientnet_v2_s', 'ResNet50': 'resnet50', 'ShuffleNetV2': 'shufflenet_v2_x1_0'}
    T.ARCH = {k: (lambda f=f: getattr(tv.models, f)(weights=None)) for k, f in _fn.items()}
    _fp = T.fit_phase
    T.fit_phase = lambda m, params, lr, epochs, *a, **k: _fp(m, params, lr, 1, *a, **k)

def train_tuned(name, Xtr, ytr, lrs=(5e-5, 2e-4, 5e-4)):
    Xt, Xv, yt, yv = train_test_split(Xtr, ytr, test_size=0.15, stratify=ytr, random_state=SEED)
    w = {0: len(yt) / (2 * (yt == 0).sum()), 1: len(yt) / (2 * (yt == 1).sum())}
    best = None; info = {}
    for lr in lrs:
        torch.manual_seed(SEED); np.random.seed(SEED)
        m = T.Net(name).to(T.DEV)
        for p in m.parameters(): p.requires_grad = False
        for p in T.head_params(m): p.requires_grad = True
        m, _, c1 = fit_curve(m, T.head_params(m), 1e-3, E1, Xt, yt, Xv, yv, w, patience=5)
        for p in m.parameters(): p.requires_grad = True                       # BatchNorm trainable in the tuned recipe
        m, va, c2 = fit_curve(m, list(m.parameters()), lr, E2, Xt, yt, Xv, yv, w, patience=8)
        info[str(lr)] = {'val_auc': float(va), 'curve_head': c1, 'curve_finetune': c2}
        if best is None or va > best[0]: best = (va, lr, copy.deepcopy(m.state_dict()))
    m = T.Net(name).to(T.DEV); m.load_state_dict(best[2]); return m, best[1], info

class NoNorm(nn.Module):
    def __init__(self, net): super().__init__(); self.b = net.b
    def forward(self, x): return self.b(x)

def norm(X): return ((X.astype(np.float32) / 255.0 - T.MEAN.numpy().reshape(1, 1, 1, 3)) / T.STD.numpy().reshape(1, 1, 1, 3)).astype(np.float32)

def int8_variants(m, name, calib_u8, k):
    import onnx
    from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader, QuantFormat
    from onnxruntime.quantization.shape_inference import quant_pre_process
    class R(CalibrationDataReader):
        def __init__(s, X): s.it = iter([{'img': X[i:i + 1].transpose(0, 3, 1, 2).astype(np.float32)} for i in range(len(X))])
        def get_next(s): return next(s.it, None)
    out = {}
    # (i) manuscript configuration
    p32 = os.path.join(OUT2, f'tmp_{name}_f{k}_fp32.onnx'); T.export_onnx(m, p32); quant_pre_process(p32, p32)
    p8 = os.path.join(OUT2, f'tmp_{name}_f{k}_int8.onnx')
    quantize_static(p32, p8, R(calib_u8), quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True)
    out['int8_as_configured'] = (p8, False)
    # (ii) normalisation outside the graph, first conv and classifier in float
    mm = copy.deepcopy(NoNorm(m)).cpu().eval(); q32 = os.path.join(OUT2, f'tmp_{name}_f{k}_nonorm_fp32.onnx')
    torch.onnx.export(mm, torch.zeros(1, 3, T.S, T.S), q32, input_names=['img'], output_names=['logit'], opset_version=17, dynamo=False, dynamic_axes={'img': {0: 'n'}, 'logit': {0: 'n'}})
    quant_pre_process(q32, q32); g = onnx.load(q32).graph
    convs = [n.name for n in g.node if n.op_type == 'Conv']; heads = [n.name for n in g.node if n.op_type in ('Gemm', 'MatMul')]
    q8 = os.path.join(OUT2, f'tmp_{name}_f{k}_nonorm_int8.onnx')
    quantize_static(q32, q8, R(norm(calib_u8)), quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True,
                    nodes_to_exclude=convs[:1] + heads[-1:])
    out['int8_norm_outside_first_last_float'] = (q8, True)
    q8b = os.path.join(OUT2, f'tmp_{name}_f{k}_nonorm_int8_all.onnx')
    quantize_static(q32, q8b, R(norm(calib_u8)), quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True)
    out['int8_norm_outside_all_layers'] = (q8b, True)
    return out

def save(fn, obj): json.dump(obj, open(os.path.join(OUT2, fn), 'w'), default=float)

if __name__ == '__main__':
    task, name = sys.argv[1], sys.argv[2]; sfx = f'_s{SEED}'
    fn = f'cnn_{name}_{task}{sfx}.json'
    if os.path.exists(os.path.join(OUT2, fn)): print('ya existe', fn); sys.exit(0)
    d = load('meta_groups_geo.csv' if task in ('geo', 'tunedgeo') else 'meta_groups.csv')
    if SMOKE: d = d.groupby('anemia').head(30).reset_index(drop=True)
    y = d.anemia.values; ids = d.image_id.tolist()
    prep = stretch if task in ('tuned', 'tunedgeo') else T.tight
    X = np.stack([prep(read_rgb(p), T.S) for p in d.image_path])
    print(task, name, 'seed', SEED, 'n =', len(y), 'device =', T.DEV, flush=True)
    t0 = time.time(); oof = np.zeros(len(y)); extra = {}; per_fold = []
    for k, (tr, te) in enumerate(folds(y, None, seed=SEED)):
        torch.manual_seed(SEED); np.random.seed(SEED)
        if task in ('tuned', 'tunedgeo'):
            m, lr, info = train_tuned(name, X[tr], y[tr]); extra[f'fold{k}'] = {'chosen_lr': lr, 'candidates': info}
        else:
            m = T.train(name, X[tr], y[tr])
        oof[te] = T.predict(m, X[te]); fa = roc_auc_score(y[te], oof[te]); per_fold.append(float(fa))
        if task == 'int8fix':
            for key, (path, pre) in int8_variants(m, name, X[tr][:100], k).items():
                p = T.ort_predict(path, norm(X[te]) if pre else X[te]); extra.setdefault(key, np.zeros(len(y)))[te] = p
                extra.setdefault(key + '_MB', []).append(os.path.getsize(path) / 2 ** 20)
                extra.setdefault(key + '_lat_ms', []).append(T.ort_latency(path, norm(X[te]) if pre else X[te], n=50) if k == 0 else None)
        print(f'{task} {name} s{SEED} fold {k} AUC {fa:.3f} {time.time() - t0:.0f}s', flush=True)
    res = {'task': task, 'model': name, 'seed': SEED, 'n': int(len(y)), 'AUC_fp32': float(roc_auc_score(y, oof)), 'per_fold_AUC': per_fold, 'train_time_s': time.time() - t0}
    for key in list(extra):
        if isinstance(extra[key], np.ndarray): res['AUC_' + key] = float(roc_auc_score(y, extra[key])); extra[key] = extra[key].tolist()
    print(json.dumps({k: v for k, v in res.items()}, default=float), flush=True)
    save(fn, {'res': res, 'y': y.tolist(), 'ids': ids, 'oof': oof.tolist(), 'extra': extra})
    for f in os.listdir(OUT2):
        if f.startswith('tmp_'): os.remove(os.path.join(OUT2, f))
