# Sixth review: the "luminance-only" ablation of review_analyses_b.py selected columns by prefix, so R/G (prefix "R")
# entered the luminance set (26 features). Rerun with the 25 luminance statistics (R, G, B, V, L* x 5) only.
import sys; sys.path.insert(0, '/home/claude/exp')
import numpy as np, pandas as pd, json, warnings; warnings.filterwarnings('ignore')
from common import *
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
d = load_meta(); dd = dedup(d); Xd = pd.read_csv('/home/claude/exp/out/feat_dedup.csv'); yd = dd.anemia.values; assert len(Xd) == 471
lum = [c for c in Xd.columns if c.split('_')[0] in ('L', 'V', 'R', 'G', 'B') and c != 'R_over_G']; assert len(lum) == 25
rf = RandomForestClassifier(500, class_weight='balanced', random_state=SEED, n_jobs=4); grid = {'min_samples_leaf': [1, 3, 5], 'max_features': ['sqrt', 0.3]}
P = np.zeros((3, len(yd)))
for r in range(3):
    for tr, te in folds(yd, None, seed=SEED + r):
        gs = GridSearchCV(rf, grid, scoring='roc_auc', cv=StratifiedKFold(5, shuffle=True, random_state=r), n_jobs=1).fit(Xd[lum].iloc[tr], yd[tr]); P[r, te] = gs.predict_proba(Xd[lum].iloc[te])[:, 1]
p = P.mean(0); rng = np.random.default_rng(SEED); a = []
for _ in range(2000):
    i = rng.integers(0, len(yd), len(yd)); a.append(roc_auc_score(yd[i], p[i]))
out = dict(n_feat=len(lum), AUC=float(roc_auc_score(yd, p)), ci=[float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))], per_rep=[float(roc_auc_score(yd, P[r])) for r in range(3)])
json.dump(out, open('/home/claude/exp/out/review6/luminance25.json', 'w'), indent=1); print(out)
