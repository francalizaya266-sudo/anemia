# Third round: sensitivity of the stage-4 map to its acceptance thresholds (inliers 8 / 10 / 15, NCC 0.60 / 0.70 / 0.80;
# overlap >= 0.30 and scale 0.5-2 kept), with the label concordance of the new pairs. Writes out/review3/thresholds.json.
import numpy as np, pandas as pd, json
REPO = '/home/claude/anemia/'; G0 = np.load(REPO + 'resultados/review2/geomverify.npy'); arr = np.load(REPO + 'resultados/review2/orb_pairs.npy')
GL = np.load('/home/claude/exp/out/review3/geomverify_lo.npy')
ncand = int((arr[:, 3] >= 10).sum()); cand0 = np.arange(len(G0)) < ncand          # first rows of geomverify.npy are the >=10 candidates
G = np.vstack([G0[cand0], GL])                                                 # every pair with >= 8 inliers in the SIFT search, verified
orbk = pd.Series(arr[:, 3], index=pd.MultiIndex.from_arrays([arr[:, 0].astype(int), arr[:, 1].astype(int)]))
k_orb = orbk.loc[list(zip(G[:, 0].astype(int), G[:, 1].astype(int)))].values
m1 = pd.read_csv(REPO + 'codigo/meta_groups.csv'); ms = pd.read_csv(REPO + 'codigo/meta_groups_strict.csv')
reps = m1.sort_values('image_id').drop_duplicates('grp').reset_index(drop=True); sgr = ms.set_index('image_id').loc[reps.image_id, 'grp'].values; la = reps.anemia.values
i = G[:, 0].astype(int); j = G[:, 1].astype(int); inl = G[:, 2]; cov = G[:, 3]; ncc = G[:, 4]; sc = G[:, 6]; pos = sgr[i] == sgr[j]
sg = ms.set_index('image_id').grp
def build(kmin, nmin):
    acc = (k_orb >= kmin) & (inl >= kmin) & (ncc >= nmin) & (sc >= 0.5) & (sc <= 2.0) & (cov >= 0.30); new = acc & ~pos; a, b = i[new], j[new]
    par = {}
    def f(x):
        par.setdefault(x, x)
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    for g in ms.grp.unique(): f(g)
    for p, q in zip(a, b):
        ra, rb = f(sg[reps.image_id[p]]), f(sg[reps.image_id[q]])
        if ra != rb: par[ra] = rb
    geo = ms.copy(); geo['grp'] = pd.factorize(geo.grp.map(f))[0]
    conf = geo.groupby('grp').anemia.nunique(); d = geo[~geo.grp.isin(conf[conf > 1].index)]; dd = d.sort_values('image_id').drop_duplicates('grp')
    return dict(inliers=kmin, ncc=nmin, accepted=int(acc.sum()), new_pairs=int(new.sum()), concordant=int((la[a] == la[b]).sum()), distinct=int(geo.grp.nunique()),
                classification_set=int(len(dd)), anaemic=int(dd.anemia.sum()), label_conflict_groups=int((conf > 1).sum()))
R = [build(k, n) for k in (8, 10, 15) for n in (0.60, 0.70, 0.80)]
for r in R: print(r)
# precision bound from label concordance: a false link carries the same label with probability ~0.5 (null concordance 84/168)
base = [r for r in R if r['inliers'] == 10 and r['ncc'] == 0.70][0]; n = base['new_pairs']; disc = n - base['concordant']
from scipy import stats, optimize
q = 0.5; f = optimize.brentq(lambda f: stats.binom.cdf(disc, n, f * q) - 0.05, 1e-9, 1 - 1e-9) if disc < n else 1.0
json.dump(dict(grid=R, precision_bound=dict(new_pairs=n, discordant=disc, null_discordance=q, false_link_upper95=f)), open('/home/claude/exp/out/review3/thresholds.json', 'w'), indent=1); print('false-link upper bound', f)
