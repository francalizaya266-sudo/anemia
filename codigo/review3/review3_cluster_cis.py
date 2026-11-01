# Third round: cluster-aware confidence intervals for every AUC in Table 2 and Fig. 3 (cluster bootstrap, 2 000 resamples,
# clusters = groups of the geometric duplicate map), and Obuchowski tests of naive minus grouped. Writes out/review3/cluster_cis.json.
import sys, os, json; sys.path.insert(0, '/home/claude/exp')
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from common import load_meta, dedup
from review2_lib import obu_ci, obu_test, cluster_boot
REPO = '/home/claude/anemia/'; GN = '/mnt/user-data/uploads/AnemIA-E2/Nueva carpeta/out/'
d = load_meta(); y = d.anemia.values; dd = dedup(d); yd = dd.anemia.values
gmap = pd.read_csv(REPO + 'codigo/meta_groups_geo.csv').set_index('image_id').grp
cl708 = gmap.loc[d.image_id].values; cl471 = gmap.loc[dd.image_id].values
o = json.load(open(REPO + 'resultados/review/oof_review.json')); os_ = json.load(open(REPO + 'resultados/review/oof_strict.json')); og = json.load(open(REPO + 'resultados/review2/geo/oof_geo.json'))
R = {'clusters_708': int(len(np.unique(cl708))), 'clusters_471': int(len(np.unique(cl471)))}
def ci(yy, p, cl): return cluster_boot(yy, p[None], cl)['auc_ci'][0]
def entry(yy, p, cl): return dict(auc=float(roc_auc_score(yy, p)), ci=ci(yy, p, cl), ci_obu=[float(x) for x in obu_ci(yy, p, cl)[1]])
def diff(pn, pg):
    dlt, cid, z, p = obu_test(y, pn, pg, cl708); cb = cluster_boot(y, np.vstack([pn, pg]), cl708)
    return dict(diff=float(dlt), ci_boot=cb['diff_ci'], ci_obu=[float(x) for x in cid], p_obu=float(p))
for m in ['LR (color)', 'SVM-RBF (color)', 'RF (color)', 'XGBoost (color)']:
    pn = np.array(o[m]['naive708']); pg = np.array(o[m]['grouped708']); ps = np.array(os_[m]['grouped_strict708']); pq = np.array(og[m]['grouped_geo708']); pd_ = np.array(o[m]['dedup471'])
    R[m] = dict(naive=entry(y, pn, cl708), grouped=entry(y, pg, cl708), grouped_strict=entry(y, ps, cl708), grouped_geo=entry(y, pq, cl708), dedup=entry(yd, pd_, cl471),
                d_first=diff(pn, pg), d_strict=diff(pn, ps), d_geo=diff(pn, pq)); print(m, R[m]['naive']['ci'], R[m]['d_first'], flush=True)
NAIVE = {'EfficientNet-B0': GN + 'cnn_EfficientNet-B0_naive.json', 'MobileNetV2': GN + 'cnn_MobileNetV2_naive.json', 'MobileNetV3-Small': REPO + 'gpu/out_review/cnn_MobileNetV3-Small_naive.json'}
for m in ['EfficientNet-B0', 'MobileNetV2', 'MobileNetV3-Small', 'EfficientNetV2-S', 'ResNet50', 'ShuffleNetV2', 'MobileNetV3-Large']:
    e = {}
    g = json.load(open(REPO + f'gpu/out_review/cnn_{m}_grouped.json')); assert np.array_equal(np.array(g['y']), y); pg = np.array(g['oof']['fp32']); e['grouped'] = entry(y, pg, cl708)
    c = json.load(open(GN + f'cnn_{m}_clean.json')); assert np.array_equal(np.array(c['y']), yd); e['dedup'] = entry(yd, np.array(c['oof']['fp32']), cl471)
    if m in NAIVE:
        n = json.load(open(NAIVE[m])); assert np.array_equal(np.array(n['y']), y); pn = np.array(n['oof']['fp32']); e['naive'] = entry(y, pn, cl708); e['d_first'] = diff(pn, pg)
        sj = REPO + f'gpu/out_strict/cnn_{m}_grouped.json'; snj = REPO + f'gpu/out_strict/cnn_{m}_naive.json'
        if os.path.exists(sj):
            sg = json.load(open(sj)); sn = json.load(open(snj)); psg = np.array(sg['oof']['fp32']); psn = np.array(sn['oof']['fp32'])
            e['grouped_strict'] = entry(y, psg, cl708); e['naive_strictrun'] = entry(y, psn, cl708); dlt, cid, z, p = obu_test(y, psn, psg, cl708); cb = cluster_boot(y, np.vstack([psn, psg]), cl708)
            e['d_strict'] = dict(diff=float(dlt), ci_boot=cb['diff_ci'], ci_obu=[float(x) for x in cid], p_obu=float(p))
    R[m] = e; print(m, {k: (round(v['auc'], 3), [round(x, 3) for x in v['ci']]) for k, v in e.items() if 'auc' in v}, flush=True)
json.dump(R, open('/home/claude/exp/out/review3/cluster_cis.json', 'w'), indent=1, default=float); print('saved')
