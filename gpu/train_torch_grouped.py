"""Grouped protocol for the CNNs (peer-review request C3): the same 708 rows as the naive protocol, but every
duplicate group (column grp of meta_groups.csv) is confined to one fold (StratifiedGroupKFold), and the
internal 15 % validation split also respects groups.  Uso: python train_torch_grouped.py EfficientNet-B0
Escribe out/cnn_<modelo>_grouped.json con el mismo formato que los demas (solo FP32)."""
import sys, os, json, time, numpy as np, torch, torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import train_torch as T

def train_grouped(name, Xtr, ytr, gtr):
    # ~15 % internal validation split that keeps duplicate groups together
    tr_i, va_i = next(StratifiedGroupKFold(7, shuffle=True, random_state=SEED).split(np.zeros(len(ytr)), ytr, gtr))
    Xt, Xv, yt, yv = Xtr[tr_i], Xtr[va_i], ytr[tr_i], ytr[va_i]
    w = {0: len(yt) / (2 * (yt == 0).sum()), 1: len(yt) / (2 * (yt == 1).sum())}
    m = T.Net(name).to(T.DEV)
    for p in m.parameters(): p.requires_grad = False
    for p in T.head_params(m): p.requires_grad = True
    m = T.fit_phase(m, T.head_params(m), 1e-3, 10, Xt, yt, Xv, yv, w, patience=5)
    for mod in m.modules():
        for p in mod.parameters(recurse=False): p.requires_grad = not isinstance(mod, nn.BatchNorm2d)
    m = T.fit_phase(m, [p for p in m.parameters() if p.requires_grad], 5e-5, 40, Xt, yt, Xv, yv, w, patience=8)
    return m

if __name__ == '__main__':
    name = sys.argv[1]
    d = load_meta(); g = d.grp.values; y = d.anemia.values
    X = np.stack([T.tight(read_rgb(p), T.S) for p in d.image_path])
    print(name, 'grouped seed', SEED, 'n =', len(y), 'groups =', len(np.unique(g)), 'device =', T.DEV, flush=True)
    P = np.zeros(len(y)); t0 = time.time()
    for k, (tr, te) in enumerate(folds(y, g, seed=SEED)):
        assert len(set(g[tr]) & set(g[te])) == 0
        m = train_grouped(name, X[tr], y[tr], g[tr]); P[te] = T.predict(m, X[te])
        print(f'{name} grouped fold {k} n_test {len(te)} AUC {roc_auc_score(y[te], P[te]):.3f} {time.time() - t0:.0f}s', flush=True)
    res = {'model': name, 'proto': 'grouped', 'n': int(len(y)), 'train_time_s': time.time() - t0, 'device': T.DEV, 'seed': SEED,
           'fp32': metrics(y, P), 'fp32_ci': boot_ci(y, P)}
    json.dump({'res': res, 'y': y.tolist(), 'grp': g.tolist(), 'oof': {'fp32': P.tolist()}}, open(os.path.join(T.OUT, f'cnn_{name}_grouped.json'), 'w'), default=float)
    print(json.dumps(res, default=float), flush=True)
