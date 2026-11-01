# Third round, addendum: AUC of the run-averaged probabilities (Random Forest: five repetitions; tuned CNNs: three seeds),
# the estimand of the DeLong test in review3_cpu.py (rf_vs_tuned_*: p_delong_avg), next to the mean single-run AUC.
import sys, json, glob; sys.path.insert(0, '/home/claude/exp')
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from common import load_meta, dedup
OUT3 = '/home/claude/exp/out/review3/'; G2 = '/home/claude/exp/out/review2/gpu2/'
d = load_meta(); dd = dedup(d); yd = dd.anemia.values
g = pd.read_csv('/home/claude/anemia/codigo/meta_groups_geo.csv'); conf = g.groupby('grp').anemia.nunique()
g = g[~g.grp.isin(conf[conf > 1].index)].sort_values('image_id').drop_duplicates('grp').reset_index(drop=True); yg = g.anemia.values
def oof(task, ids):
    out = {}
    for f in sorted(glob.glob(G2 + f'cnn_*_{task}_s*.json')):
        j = json.load(open(f)); m = f.split('/')[-1][4:].split('_' + task + '_')[0]
        out.setdefault(m, []).append(pd.Series(j['oof'], index=j['ids']).loc[ids].values)
    return out
R = json.load(open(OUT3 + 'review3.json')); E = {}
for tag, task, P, ids, y in (('471', 'tuned', np.load(OUT3 + 'rf_rep471.npy'), dd.image_id.tolist(), yd), ('geo', 'tunedgeo', np.load(OUT3 + 'rf_repgeo.npy'), g.image_id.tolist(), yg)):
    e = {'RF (color)': dict(runs=int(len(P)), single_mean=float(np.mean([roc_auc_score(y, p) for p in P])), averaged=float(roc_auc_score(y, P.mean(0))))}
    for m, L in oof(task, ids).items():
        e[m] = dict(runs=len(L), single_mean=float(np.mean([roc_auc_score(y, p) for p in L])), averaged=float(roc_auc_score(y, np.mean(L, 0))))
    E[tag] = e
R['ensemble_tuned'] = E; json.dump(R, open(OUT3 + 'review3.json', 'w'), indent=1, default=float)
print(json.dumps(E, indent=1))
