# Second internal review, part 3: geometric near-duplicate search over all pairs of unique images.
# SIFT keypoints (2x upscaled, CLAHE-enhanced green channel) inside the eroded conjunctiva mask, Lowe ratio test, RANSAC similarity transform (rotation, scale,
# translation); mirrored copies handled by also matching the flipped image. Positive controls: the 19 pairs merged by
# the third audit stage and the pHash-linked non-identical files inside first-stage groups.
import sys, os, json, time, itertools; sys.path.insert(0, '/home/claude/exp')
import numpy as np, pandas as pd, cv2
from multiprocessing import Pool
from common import read_rgb, mask_of
OUT = '/home/claude/exp/out/review2/'
mfull = pd.read_csv('/home/claude/anemia/codigo/meta_groups.csv'); ms = pd.read_csv('/home/claude/anemia/codigo/meta_groups_strict.csv').set_index('image_id')
reps = mfull.sort_values('image_id').drop_duplicates('grp').reset_index(drop=True)          # 472 first-stage representatives
sift = cv2.SIFT_create(nfeatures=800, contrastThreshold=0.01)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
def describe(path, flip=False):
    im = read_rgb(path)
    if flip: im = im[:, ::-1].copy()
    im = cv2.resize(im, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    m = mask_of(im).astype(np.uint8); m = cv2.erode(m, np.ones((5, 5), np.uint8))
    g = clahe.apply(np.ascontiguousarray(im[:, :, 1])); kp, des = sift.detectAndCompute(g, m * 255)
    return (np.array([k.pt for k in kp], np.float32) if kp else np.zeros((0, 2), np.float32)), des
def load_all(paths):
    return [describe(p) for p in paths], [describe(p, True) for p in paths]
bf = cv2.BFMatcher(cv2.NORM_L2)
def score(a, b):
    (pa, da), (pb, db) = a, b
    if da is None or db is None or len(da) < 8 or len(db) < 8: return 0, 0
    mm = bf.knnMatch(da, db, k=2); good = [m for m, *n in mm if n and m.distance < 0.75 * n[0].distance]
    if len(good) < 6: return len(good), 0
    src = pa[[g.queryIdx for g in good]]; dst = pb[[g.trainIdx for g in good]]
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0, maxIters=2000, confidence=0.995)
    return len(good), int(inl.sum()) if inl is not None else 0
D = Df = None
def init(paths):
    global D, Df; D, Df = load_all(paths)
def work(pair):
    i, j = pair; g1, i1 = score(D[i], D[j]); g2, i2 = score(D[i], Df[j]); return i, j, max(g1, g2), max(i1, i2), len(D[i][0]), len(D[j][0])
if __name__ == '__main__':
    t0 = time.time(); paths = reps.image_path.tolist(); n = len(paths)
    pairs = list(itertools.combinations(range(n), 2))
    with Pool(4, initializer=init, initargs=(paths,)) as pool:
        res = pool.map(work, pairs, chunksize=500)
    arr = np.array(res); np.save(OUT + 'orb_pairs.npy', arr); print('pairs', len(arr), 'time %.0fs' % (time.time() - t0), flush=True)
    # positive controls 1: representatives merged by stage 3
    sgr = ms.loc[reps.image_id, 'grp'].values; merged = sgr[:, None] == sgr[None]
    pos1 = np.array([merged[i, j] for i, j in arr[:, :2].astype(int)])
    # positive controls 2: non-identical files linked inside first-stage groups (rep vs member with different SHA)
    init(paths); ctrl2 = []
    for g, sub in mfull.groupby('grp'):
        if sub.sha.nunique() < 2: continue
        r = reps[reps.grp == g].iloc[0]; others = sub[sub.sha != r.sha].drop_duplicates('sha')
        a = describe(r.image_path); af = None
        for _, o in others.iterrows():
            b = describe(o.image_path); bfp = describe(o.image_path, True)
            ctrl2.append(max(score(a, b)[1], score(a, bfp)[1]))
    inl = arr[:, 3]; neg = inl[~pos1]
    out = dict(n_images=n, n_pairs=int(len(arr)), stage3_pairs=int(pos1.sum()), stage3_inliers=sorted(inl[pos1].astype(int).tolist()),
               within_group_nonidentical_pairs=len(ctrl2), within_group_inliers_quantiles=np.percentile(ctrl2, [0, 10, 50, 90, 100]).tolist() if ctrl2 else [],
               other_pairs_inlier_quantiles=np.percentile(neg, [50, 90, 99, 99.9, 99.99, 100]).tolist(), other_pairs_ge10=int((neg >= 10).sum()), other_pairs_ge15=int((neg >= 15).sum()),
               other_pairs_ge20=int((neg >= 20).sum()), other_pairs_ge30=int((neg >= 30).sum()))
    top = np.argsort(-inl * (~pos1))[:24]
    out['top_other_pairs'] = [dict(a=reps.image_id[int(arr[k, 0])], b=reps.image_id[int(arr[k, 1])], good=int(arr[k, 2]), inliers=int(arr[k, 3])) for k in top]
    json.dump(out, open(OUT + 'orb_results.json', 'w'), indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if k != 'top_other_pairs'}, default=float)); print(out['top_other_pairs'][:12])
