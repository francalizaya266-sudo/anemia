"""CNN ligeras para anemia desde conjuntiva (CP-AnemiC) - PyTorch/CUDA.
Mismo protocolo que la version TensorFlow: 471 imagenes unicas, CV estratificada 5 pliegues,
15 % del pliegue de entrenamiento como validacion interna para parada temprana, IC bootstrap.
Exporta ONNX FP32 / FP16 / INT8 dinamico / INT8 estatico y mide latencia con onnxruntime (1 hilo).
Uso:  python train_torch.py MobileNetV2 clean   |   python train_torch.py MobileNetV2 naive
"""
import sys, os, json, time, copy
import numpy as np, torch, torch.nn as nn, torchvision as tv
from sklearn.model_selection import train_test_split
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out'); os.makedirs(OUT, exist_ok=True)
S = 224; DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(SEED); np.random.seed(SEED)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

ARCH = {
    'MobileNetV2': lambda: tv.models.mobilenet_v2(weights='DEFAULT'),
    'MobileNetV3-Small': lambda: tv.models.mobilenet_v3_small(weights='DEFAULT'),
    'MobileNetV3-Large': lambda: tv.models.mobilenet_v3_large(weights='DEFAULT'),
    'EfficientNet-B0': lambda: tv.models.efficientnet_b0(weights='DEFAULT'),
    'EfficientNetV2-S': lambda: tv.models.efficientnet_v2_s(weights='DEFAULT'),
    'ResNet50': lambda: tv.models.resnet50(weights='DEFAULT'),
    'ShuffleNetV2': lambda: tv.models.shufflenet_v2_x1_0(weights='DEFAULT'),
}

def tight(im, s=S):
    m = mask_of(im); ys, xs = np.where(m)
    if len(ys) < 10: return letterbox(im, s)
    return letterbox(im[ys.min():ys.max() + 1, xs.min():xs.max() + 1], s)

class Net(nn.Module):
    def __init__(self, name):
        super().__init__(); b = ARCH[name]()
        if hasattr(b, 'classifier'):
            last = [m for m in b.classifier.modules() if isinstance(m, nn.Linear)][0]; nf = last.in_features  # primera Linear: entrada del clasificador (MobileNetV3 tiene dos)
            b.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(nf, 1))
        else:
            nf = b.fc.in_features; b.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(nf, 1))
        self.b = b
    def forward(self, x):  # x: uint8-range float NCHW 0..255
        x = (x / 255.0 - MEAN.to(x.device)) / STD.to(x.device)
        return self.b(x)

def head_params(m): return list((m.b.classifier if hasattr(m.b, 'classifier') else m.b.fc).parameters())

def augment(xb):  # geometric only: flips, small rotation/zoom/shift (no colour jitter: colour IS the signal)
    if np.random.rand() < 0.5: xb = xb.flip(3)
    if np.random.rand() < 0.5: xb = xb.flip(2)
    ang = (torch.rand(xb.size(0), device=xb.device) - 0.5) * 2 * np.deg2rad(18)
    sc = 1 + (torch.rand(xb.size(0), device=xb.device) - 0.5) * 0.2
    tx = (torch.rand(xb.size(0), device=xb.device) - 0.5) * 0.1; ty = (torch.rand(xb.size(0), device=xb.device) - 0.5) * 0.1
    th = torch.stack([torch.stack([torch.cos(ang) * sc, -torch.sin(ang) * sc, tx], 1), torch.stack([torch.sin(ang) * sc, torch.cos(ang) * sc, ty], 1)], 1)
    grid = nn.functional.affine_grid(th, xb.shape, align_corners=False)
    return nn.functional.grid_sample(xb, grid, padding_mode='zeros', align_corners=False)

def predict(m, X, bs=64):
    m.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs]).permute(0, 3, 1, 2).float().to(DEV)
            out.append(torch.sigmoid(m(xb)).squeeze(1).cpu().numpy())
    return np.concatenate(out)

def fit_phase(m, params, lr, epochs, Xt, yt, Xv, yv, w, patience=8, bs=32):
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4); best = (-1, None, 0)
    crit = nn.BCEWithLogitsLoss(reduction='none'); scaler = torch.amp.GradScaler('cuda', enabled=DEV == 'cuda')
    for ep in range(epochs):
        m.train(); idx = np.random.permutation(len(Xt))
        for i in range(0, len(idx), bs):
            j = idx[i:i + bs]; xb = torch.from_numpy(Xt[j]).permute(0, 3, 1, 2).float().to(DEV); yb = torch.from_numpy(yt[j]).float().to(DEV)
            xb = augment(xb)
            with torch.autocast('cuda', enabled=DEV == 'cuda'):
                loss = (crit(m(xb).squeeze(1), yb) * torch.where(yb == 1, w[1], w[0])).mean()
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        auc = roc_auc_score(yv, predict(m, Xv))
        if auc > best[0]: best = (auc, copy.deepcopy(m.state_dict()), ep)
        elif ep - best[2] >= patience: break
    m.load_state_dict(best[1]); return m

def train(name, Xtr, ytr):
    Xt, Xv, yt, yv = train_test_split(Xtr, ytr, test_size=0.15, stratify=ytr, random_state=SEED)
    w = {0: len(yt) / (2 * (yt == 0).sum()), 1: len(yt) / (2 * (yt == 1).sum())}
    m = Net(name).to(DEV)
    for p in m.parameters(): p.requires_grad = False
    for p in head_params(m): p.requires_grad = True
    m = fit_phase(m, head_params(m), 1e-3, 10, Xt, yt, Xv, yv, w, patience=5)        # fase 1: solo cabeza
    for mod in m.modules():
        for p in mod.parameters(recurse=False): p.requires_grad = not isinstance(mod, nn.BatchNorm2d)  # BN congelada
    m = fit_phase(m, [p for p in m.parameters() if p.requires_grad], 5e-5, 40, Xt, yt, Xv, yv, w, patience=8)  # fase 2: ajuste fino
    return m

# ---------- exportacion y medicion en borde (ONNX Runtime) ----------
def export_onnx(m, path):
    m = copy.deepcopy(m).cpu().eval()
    torch.onnx.export(m, torch.zeros(1, 3, S, S), path, input_names=['img'], output_names=['logit'], opset_version=17, dynamo=False, dynamic_axes={'img': {0: 'n'}, 'logit': {0: 'n'}})
    return path

def make_variants(m, name, calib):
    import onnx, onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType, CalibrationDataReader, QuantFormat
    from onnxruntime.quantization.shape_inference import quant_pre_process
    from onnxconverter_common import float16
    paths = {}
    p32 = export_onnx(m, os.path.join(OUT, f'{name}_fp32.onnx')); quant_pre_process(p32, p32); paths['fp32'] = p32
    p16 = os.path.join(OUT, f'{name}_fp16.onnx'); onnx.save(float16.convert_float_to_float16(onnx.load(p32), keep_io_types=True), p16); paths['fp16'] = p16
    pdy = os.path.join(OUT, f'{name}_int8dyn.onnx'); quantize_dynamic(p32, pdy, weight_type=QuantType.QInt8); paths['dyn'] = pdy
    class R(CalibrationDataReader):
        def __init__(s): s.it = iter([{'img': calib[i:i + 1].transpose(0, 3, 1, 2).astype(np.float32)} for i in range(len(calib))])
        def get_next(s): return next(s.it, None)
    p8 = os.path.join(OUT, f'{name}_int8.onnx')
    try:
        quantize_static(p32, p8, R(), quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True); paths['int8'] = p8
    except Exception as e: print('int8 static failed:', e)
    return paths

def ort_sess(path, threads):
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.inter_op_num_threads = 1
    return ort.InferenceSession(path, so, providers=['CPUExecutionProvider'])

def ort_predict(path, X):
    s = ort_sess(path, 4); out = []
    for i in range(0, len(X), 32):
        out.append(s.run(None, {'img': X[i:i + 32].transpose(0, 3, 1, 2).astype(np.float32)})[0][:, 0])
    return 1 / (1 + np.exp(-np.concatenate(out)))

def ort_latency(path, X, n=100):
    s = ort_sess(path, 1); x = X[0:1].transpose(0, 3, 1, 2).astype(np.float32)
    for _ in range(10): s.run(None, {'img': x})
    t = []
    for _ in range(n):
        t0 = time.perf_counter(); s.run(None, {'img': x}); t.append((time.perf_counter() - t0) * 1000)
    return float(np.median(t))

if __name__ == '__main__':
    name = sys.argv[1]; proto = sys.argv[2] if len(sys.argv) > 2 else 'clean'; sfx = '' if SEED == 42 else f'_s{SEED}'
    nf = int(os.environ.get('NFOLDS', '5'))
    d = load_meta(); dd = dedup(d) if proto == 'clean' else d
    X = np.stack([tight(read_rgb(p), S) for p in dd.image_path]); y = dd.anemia.values
    print(name, proto, 'seed', SEED, 'n =', len(y), 'device =', DEV, torch.cuda.get_device_name(0) if DEV == 'cuda' else '', flush=True)
    Q = ['fp32', 'fp16', 'dyn', 'int8']; P = {q: np.zeros(len(y)) for q in Q}; have = {q: np.zeros(len(y), bool) for q in Q}
    t0 = time.time(); last = None
    for k, (tr, te) in enumerate(folds(y, None, seed=SEED)[:nf]):
        m = train(name, X[tr], y[tr]); P['fp32'][te] = predict(m, X[te]); have['fp32'][te] = True
        if proto == 'clean' and SEED == 42:
            paths = make_variants(m, name, X[tr][:100])
            for q in Q[1:]:
                if q in paths: P[q][te] = ort_predict(paths[q], X[te]); have[q][te] = True
        print(f'{name} {proto} fold {k} AUC {roc_auc_score(y[te], P["fp32"][te]):.3f} {time.time() - t0:.0f}s', flush=True)
        last = m
    res = {'model': name, 'proto': proto, 'n': int(len(y)), 'train_time_s': time.time() - t0, 'params_M': sum(p.numel() for p in last.parameters()) / 1e6, 'device': DEV, 'runtime': 'onnxruntime-cpu-1thread', 'seed': SEED}
    for q in (Q if (proto == 'clean' and SEED == 42) else ['fp32']):
        if have[q].all(): res[q] = metrics(y, P[q]); res[q + '_ci'] = boot_ci(y, P[q])
    if proto == 'clean' and SEED == 42:
        paths = make_variants(last, name, X[:100])
        for q, p in paths.items(): res[q + '_MB'] = os.path.getsize(p) / 2 ** 20; res[q + '_lat_ms'] = ort_latency(p, X)
        torch.save(last.state_dict(), os.path.join(OUT, f'{name}_lastfold{sfx}.pt'))
    json.dump({'res': res, 'y': y.tolist(), 'oof': {k: v.tolist() for k, v in P.items()}}, open(os.path.join(OUT, f'cnn_{name}_{proto}{sfx}.json'), 'w'), default=float)
    print(json.dumps(res, default=float), flush=True)
