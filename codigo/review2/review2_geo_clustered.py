# Cluster-aware naive-minus-grouped tests on the 708 rows with the GEOMETRIC groups as clusters (the coarsest, most complete duplicate map):
# Obuchowski (1997) clustered-ROC variance and a 2 000-sample cluster bootstrap, for the grouped protocol with the pre-specified, strict
# and geometric maps (colour-feature models) and the pre-specified and strict maps (CNNs). Writes out/review2/clustered_geo.json.
import sys, os, json; sys.path.insert(0, '/home/claude/exp'); sys.path.insert(0, '/home/claude/anemia/codigo/review')
import numpy as np, pandas as pd
from common import load_meta
from review2_lib import obu_ci, obu_test, cluster_boot
REPO = '/home/claude/anemia/'; OUT = '/home/claude/exp/out/review2/'
d = load_meta(); y = d.anemia.values
cl = pd.read_csv(REPO + 'codigo/meta_groups_geo.csv').set_index('image_id').loc[d.image_id, 'grp'].values
o = json.load(open(REPO + 'resultados/review/oof_review.json')); os_ = json.load(open(REPO + 'resultados/review/oof_strict.json'))
og = json.load(open(OUT + 'geo/oof_geo.json'))
R = {'clusters': int(len(np.unique(cl))), 'rows': int(len(y))}
def clustered(name, pn, pg, tag):
    an, cin, _ = obu_ci(y, pn, cl); ag, cig, _ = obu_ci(y, pg, cl); dlt, cid, z, p = obu_test(y, pn, pg, cl); cb = cluster_boot(y, np.vstack([pn, pg]), cl)
    R[f'{name}|{tag}'] = dict(naive=an, naive_ci_obu=cin, grouped=ag, grouped_ci_obu=cig, diff=dlt, diff_ci_obu=cid, diff_ci_boot=cb['diff_ci'], p_obu=p, z_obu=z, p_boot=cb['diff_p'])
    print(name, tag, round(dlt, 3), [round(x, 3) for x in cid], [round(x, 3) for x in cb['diff_ci']], '%.2g' % p, flush=True)
for m in ['LR (color)', 'SVM-RBF (color)', 'RF (color)', 'XGBoost (color)']:
    pn = np.array(o[m]['naive708'])
    clustered(m, pn, np.array(o[m]['grouped708']), 'first')
    clustered(m, pn, np.array(os_[m]['grouped_strict708']), 'strict')
    if 'grouped_geo708' in og.get(m, {}): clustered(m, pn, np.array(og[m]['grouped_geo708']), 'geo')
GN = ['/home/claude/anemia/gpu/out_review', '/home/claude/gpu_out', '/mnt/user-data/uploads/AnemIA-E2/Nueva carpeta/out']
for m in ['EfficientNet-B0', 'MobileNetV2', 'MobileNetV3-Small']:
    g = json.load(open(f'/home/claude/anemia/gpu/out_review/cnn_{m}_grouped.json'))
    nf = next(os.path.join(x, f'cnn_{m}_naive.json') for x in GN if os.path.exists(os.path.join(x, f'cnn_{m}_naive.json')))
    n = json.load(open(nf)); assert np.array_equal(np.array(n['y']), y)
    clustered(m, np.array(n['oof']['fp32']), np.array(g['oof']['fp32']), 'first')
    sn = json.load(open(f'/home/claude/anemia/gpu/out_strict/cnn_{m}_naive.json')); sgj = json.load(open(f'/home/claude/anemia/gpu/out_strict/cnn_{m}_grouped.json'))
    clustered(m, np.array(sn['oof']['fp32']), np.array(sgj['oof']['fp32']), 'strict')
json.dump(R, open(OUT + 'clustered_geo.json', 'w'), indent=1, default=float); print('saved')
