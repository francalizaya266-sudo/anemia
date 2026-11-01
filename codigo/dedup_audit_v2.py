"""Third stage of the CP-AnemiC duplicate audit (post hoc).

Recomputes the 256-bit perceptual hash (pHash, 16 x 16 DCT) of every image under the eight transforms of the
dihedral group (identity, three rotations, and the mirror of each), links images whose hashes are <= 20 bits apart
under any transform, adds pairs whose standardised colour-feature vectors coincide, merges the links with the
first-stage duplicate map (SHA-256 + mirror-only pHash) and writes the strict duplicate map.

Usage: python dedup_audit_v2.py <images_dir> <meta_groups.csv> <feat_all.csv> <out_dir>
Outputs: meta_groups_strict.csv (same rows and columns as meta_groups.csv, grp = strict component id),
         dihedral_links.json (new links with their Hamming distance), strict_pairs.json (merged representative pairs).
"""
import sys, os, json, itertools, numpy as np, pandas as pd, imagehash
from PIL import Image
from sklearn.preprocessing import StandardScaler
IMG, META, FEAT, OUT = sys.argv[1:5]; os.makedirs(OUT, exist_ok=True)
THR = 20
T = {'id': lambda im: im, 'rot90': lambda im: im.transpose(Image.ROTATE_90), 'rot180': lambda im: im.transpose(Image.ROTATE_180),
     'rot270': lambda im: im.transpose(Image.ROTATE_270)}
T.update({'mirror+' + k: (lambda f: (lambda im: f(im.transpose(Image.FLIP_LEFT_RIGHT))))(f) for k, f in list(T.items())})
m = pd.read_csv(META)
H = {name: np.array([imagehash.phash(f(Image.open(os.path.join(IMG, p)).convert('RGB')), hash_size=16).hash.flatten() for p in m.image_path]) for name, f in T.items()}
base = H['id'].astype(np.uint8)
dist = np.full((len(m), len(m)), 999, int); best = np.full((len(m), len(m)), '', object)
for name, Hk in H.items():
    d = (base[:, None, :] != Hk[None, :, :].astype(np.uint8)).sum(-1)
    upd = d < dist; best[upd] = name; dist = np.minimum(dist, d)
np.save(os.path.join(OUT, 'dm_dihedral.npy'), dist)
# feature identity (pairs with coincident standardised feature vectors); feat_all rows follow meta rows minus label-conflict groups
conf = m.groupby('grp').anemia.nunique(); keep = ~m.grp.isin(conf[conf > 1].index)
X = pd.read_csv(FEAT); assert len(X) == keep.sum()
Z = StandardScaler().fit_transform(X.values); idx = np.where(keep)[0]
D2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1); fi, fj = np.where(np.triu(D2 < 1e-6, 1))
feat_pairs = [(int(idx[a]), int(idx[b])) for a, b in zip(fi, fj) if m.grp[idx[a]] != m.grp[idx[b]]]  # pairs not already in one first-stage group
# union-find over first-stage groups, dihedral links and feature pairs
parent = list(range(len(m)))
def find(a):
    while parent[a] != a: parent[a] = parent[parent[a]]; a = parent[a]
    return a
def union(a, b): parent[find(a)] = find(b)
for g, rows in m.groupby('grp').groups.items():
    rows = list(rows)
    for r in rows[1:]: union(rows[0], r)
old_grp = m.grp.values; links = []
ii, jj = np.where(np.triu(dist <= THR, 1))
for a, b in zip(ii, jj):
    if old_grp[a] != old_grp[b]: links.append([m.image_id[a], m.image_id[b], int(dist[a, b]), str(best[a, b])])
    union(a, b)
for a, b in feat_pairs: union(a, b)
roots = [find(i) for i in range(len(m))]; ids = {r: k for k, r in enumerate(dict.fromkeys(roots))}
s = m.copy(); s['grp'] = [ids[r] for r in roots]; s.to_csv(os.path.join(OUT, 'meta_groups_strict.csv'), index=False)
json.dump({'threshold_bits': THR, 'new_links': links, 'feature_identical_pairs': [[m.image_id[a], m.image_id[b]] for a, b in feat_pairs]}, open(os.path.join(OUT, 'dihedral_links.json'), 'w'), indent=1)
print('rows', len(m), 'first-stage components', m.grp.nunique(), 'strict components', s.grp.nunique(), 'new links', len(links), 'feature-identical pairs', len(feat_pairs))
