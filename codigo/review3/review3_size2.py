# Third round, size control (design 2): the 471-image set subsampled to 337 images with the class balance of the geometric set
# by removing images at random within class (near-duplicate pairs may be broken). Five subsamples, one repetition each,
# for XGBoost and (added in the fourth round) Random Forest.
import sys, os, json; sys.path.insert(0, '/home/claude/exp')
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from common import load_meta, dedup
from classic import MODELS, run_cv
# Speed: parallelise the grid search instead of the forest (identical results: the trees depend only on random_state)
import classic as _cl
from sklearn.model_selection import GridSearchCV as _GS
def _gs4(*a, **k): k['n_jobs'] = 4; return _GS(*a, **k)
_cl.GridSearchCV = _gs4
MODELS['RF (color)'][0].set_params(n_jobs=1)
d = load_meta(); dd = dedup(d); yd = dd.anemia.values; Xd = pd.read_csv('/home/claude/exp/out/feat_dedup.csv')
RJ = '/home/claude/exp/out/review3/size2.json'; R = json.load(open(RJ)) if os.path.exists(RJ) else {}
na, nn = int(yd.sum() - 179), int((1 - yd).sum() - 158); out = R.setdefault('size_control2', {'design': 'random removal within class to 179/158'})
for m in ['XGBoost (color)', 'RF (color)']:
    est, grid = MODELS[m]; aucs = out.get(m + '|aucs', [])
    for k in range(len(aucs), 5):
        rg = np.random.default_rng(2000 + k); drop = np.concatenate([rg.choice(np.where(yd == 1)[0], na, replace=False), rg.choice(np.where(yd == 0)[0], nn, replace=False)])
        keep = np.setdiff1d(np.arange(len(yd)), drop); P = run_cv(Xd.iloc[keep].reset_index(drop=True), yd[keep], None, m, est, grid, reps=1)
        aucs.append(float(roc_auc_score(yd[keep], P[0]))); out[m + '|aucs'] = aucs
        R["size_control2"] = out; json.dump(R, open(RJ, 'w'), indent=1); print(m, k, aucs[-1], flush=True)
    out[m] = dict(mean=float(np.mean(aucs)), sd=float(np.std(aucs, ddof=1)))
R["size_control2"] = out; json.dump(R, open(RJ, 'w'), indent=1); print('done', flush=True)
