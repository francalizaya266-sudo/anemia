# Second internal review, part 3b: photometric verification of SIFT+RANSAC candidate pairs.
# For every pair with >= 10 RANSAC inliers, re-estimate the similarity transform (plain and mirrored), warp image A onto
# image B and compute (i) the fraction of the smaller conjunctiva covered by the overlap and (ii) the zero-mean normalised
# cross-correlation (NCC) of the CLAHE green channel inside the overlap. A true copy (re-crop, rescale, mirror) gives a
# high NCC over a large overlap; a chance keypoint configuration does not.
import sys, json, time; sys.path.insert(0, '/home/claude/exp')
import numpy as np, pandas as pd, cv2
from multiprocessing import Pool
from common import read_rgb, mask_of
OUT = '/home/claude/exp/out/review2/'
mfull = pd.read_csv('/home/claude/anemia/codigo/meta_groups.csv')
reps = mfull.sort_values('image_id').drop_duplicates('grp').reset_index(drop=True)
sift = cv2.SIFT_create(nfeatures=800, contrastThreshold=0.01); clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)); bf = cv2.BFMatcher(cv2.NORM_L2)
def prep(path, flip=False):
    im = read_rgb(path)
    if flip: im = im[:, ::-1].copy()
    im = cv2.resize(im, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    m = mask_of(im).astype(np.uint8); me = cv2.erode(m, np.ones((5, 5), np.uint8))
    g = clahe.apply(np.ascontiguousarray(im[:, :, 1])); kp, des = sift.detectAndCompute(g, me * 255)
    pts = np.array([k.pt for k in kp], np.float32) if kp else np.zeros((0, 2), np.float32)
    return dict(g=g.astype(np.float32), m=me, pts=pts, des=des)
def match(a, b):
    if a['des'] is None or b['des'] is None or len(a['des']) < 8 or len(b['des']) < 8: return None, 0
    mm = bf.knnMatch(a['des'], b['des'], k=2); good = [m for m, *n in mm if n and m.distance < 0.75 * n[0].distance]
    if len(good) < 6: return None, 0
    M, inl = cv2.estimateAffinePartial2D(a['pts'][[g.queryIdx for g in good]], b['pts'][[g.trainIdx for g in good]], method=cv2.RANSAC, ransacReprojThreshold=4.0, maxIters=2000, confidence=0.995)
    return M, int(inl.sum()) if inl is not None else 0
def verify(a, b, M):
    h, w = b['g'].shape
    ga = cv2.warpAffine(a['g'], M, (w, h)); ma = cv2.warpAffine(a['m'], M, (w, h), flags=cv2.INTER_NEAREST)
    ov = (ma > 0) & (b['m'] > 0); n = ov.sum()
    cover = n / max(1, min(a['m'].sum() * abs(np.linalg.det(M[:, :2])), b['m'].sum()))
    if n < 200: return float(cover), 0.0
    x = ga[ov] - ga[ov].mean(); y = b['g'][ov] - b['g'][ov].mean()
    return float(cover), float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum() + 1e-9))
P = None
def init(paths):
    global P; P = paths
def work(pair):
    i, j = pair; A = prep(P[i]); Af = prep(P[i], True); B = prep(P[j]); best = (0, 0.0, 0.0, 0, 1.0)
    for a, fl in ((A, 0), (Af, 1)):
        M, k = match(a, B)
        if M is None: continue
        cov, ncc = verify(a, B, M); s = float(np.sqrt(abs(np.linalg.det(M[:, :2]))))
        if ncc > best[2] or (best[2] == 0 and k > best[0]): best = (k, cov, ncc, fl, s)
    return (i, j) + best
if __name__ == '__main__':
    t0 = time.time(); arr = np.load(OUT + 'orb_pairs.npy'); cand = arr[arr[:, 3] >= 10][:, :2].astype(int)
    rng = np.random.default_rng(42); allp = arr[:, :2].astype(int); low = allp[rng.choice(np.where(arr[:, 3] < 10)[0], 1000, replace=False)]
    todo = [tuple(p) for p in np.vstack([cand, low])]; print('pairs to verify', len(todo), flush=True)
    with Pool(4, initializer=init, initargs=(reps.image_path.tolist(),)) as pool: res = pool.map(work, todo, chunksize=50)
    res = np.array(res, dtype=float); np.save(OUT + 'geomverify.npy', res); print('done %.0fs' % (time.time() - t0), flush=True)
