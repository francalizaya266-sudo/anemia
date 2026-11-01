"""Duplicate audit of CP-AnemiC: SHA-256 (byte-identical) + perceptual hash (pHash-16, with horizontal mirror,
Hamming <= 20) + connected components -> one group id per unique image.  Output: meta_groups.csv
Usage: python dedup_audit.py <DATA_DIR with metadata.csv and images/> <out.csv>"""
import sys, os, hashlib, numpy as np, pandas as pd, imagehash
from PIL import Image
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix
D = sys.argv[1] if len(sys.argv) > 1 else '/home/claude/data/cp-anemic/'; OUT = sys.argv[2] if len(sys.argv) > 2 else 'meta_groups.csv'
T = 20  # Hamming threshold on 256-bit pHash
m = pd.read_csv(os.path.join(D, 'metadata.csv'))
sha, H, Hm = [], [], []
for p in m.image_path:
    fp = os.path.join(D, p); sha.append(hashlib.sha256(open(fp, 'rb').read()).hexdigest())
    im = Image.open(fp).convert('RGB')
    H.append(imagehash.phash(im, hash_size=16).hash.flatten()); Hm.append(imagehash.phash(im.transpose(Image.FLIP_LEFT_RIGHT), hash_size=16).hash.flatten())
H = np.array(H, bool); Hm = np.array(Hm, bool); n = len(m)
dm = np.minimum((H[:, None, :] ^ H[None, :, :]).sum(2), (H[:, None, :] ^ Hm[None, :, :]).sum(2))  # mirror-invariant distance
sha_arr = np.array(sha); same_sha = sha_arr[:, None] == sha_arr[None, :]
adj = (dm <= T) | same_sha; np.fill_diagonal(adj, False)
ii, jj = np.where(adj); ncomp, lab = connected_components(coo_matrix((np.ones(len(ii)), (ii, jj)), shape=(n, n)), directed=False)
out = m[['image_id', 'image_path', 'hb', 'anemia', 'severity', 'hospital', 'age_months', 'gender']].copy(); out['sha'] = sha; out['grp'] = lab
out.to_csv(OUT, index=False)
print(f'rows {n}  sha-identical groups {sum(1 for s, c in pd.Series(sha).value_counts().items() if c > 1)}  unique groups {ncomp}  '
      f'label-conflict groups {int((out.groupby("grp").anemia.nunique() > 1).sum())}')
