# Fifth review: validation of the duplicate detector with synthetic copies of known identity.
# Each representative of the geometric set is transformed in ways that real copies can show (mirror, rotation, rescaling,
# JPEG recompression, re-cropping, re-segmentation, a combination), and every audit stage is applied to the pair
# (original, transformed copy): stage 2 = pHash-16 of image and mirror, Hamming <= 20; stage 3 = pHash under the 8 dihedral
# transforms; stage 4 = SIFT + RANSAC + photometric verification with the thresholds of the article. Recall = share of
# synthetic copies linked to their original. Spurious links = accepted stage-4 links between a synthetic copy and 5 random
# images of other groups.
import sys, json, time; sys.path.insert(0, '/home/claude/exp')
import numpy as np, pandas as pd, cv2, imagehash
from PIL import Image
from multiprocessing import Pool
from common import read_rgb, mask_of
OUT = '/home/claude/exp/out/review5/'
mg = pd.read_csv('/home/claude/anemia/codigo/meta_groups_geo.csv')
conf = mg.groupby('grp').anemia.nunique(); mg = mg[~mg.grp.isin(conf[conf > 1].index)]
reps = mg.sort_values('image_id').drop_duplicates('grp').reset_index(drop=True)
sift = cv2.SIFT_create(nfeatures=800, contrastThreshold=0.01); clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)); bf = cv2.BFMatcher(cv2.NORM_L2)

def rot(im, ang):
    h, w = im.shape[:2]; M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0); c, s = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * s + w * c), int(h * c + w * s); M[0, 2] += nw / 2 - w / 2; M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(im, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
def jpeg(im, q):
    ok, b = cv2.imencode('.jpg', im[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, q]); return cv2.imdecode(b, cv2.IMREAD_COLOR)[:, :, ::-1].copy()
def crop(im, f, rng):
    h, w = im.shape[:2]; nh, nw = max(8, int(round(h * f))), max(8, int(round(w * f)))
    y = rng.integers(0, h - nh + 1); x = rng.integers(0, w - nw + 1); return im[y:y + nh, x:x + nw].copy()
def reseg(im, px):
    m = mask_of(im).astype(np.uint8); me = cv2.erode(m, np.ones((2 * px + 1, 2 * px + 1), np.uint8)); o = im.copy(); o[me == 0] = 0; return o
def rs(im, f): return cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)
TRANSFORMS = {
    'mirror': lambda im, r: im[:, ::-1].copy(),
    'rotate 90': lambda im, r: np.ascontiguousarray(np.rot90(im, 1)),
    'rotate 180': lambda im, r: np.ascontiguousarray(np.rot90(im, 2)),
    'rotate 10 deg': lambda im, r: rot(im, 10),
    'rescale 0.6': lambda im, r: rs(im, 0.6),
    'rescale 1.5': lambda im, r: rs(im, 1.5),
    'JPEG q60': lambda im, r: jpeg(im, 60),
    'crop 85 %': lambda im, r: crop(im, 0.85, r),
    'crop 70 %': lambda im, r: crop(im, 0.70, r),
    're-segment (erode 3 px)': lambda im, r: reseg(im, 3),
    'combined (crop 85 %, x0.8, 5 deg, JPEG q75)': lambda im, r: jpeg(rot(rs(crop(im, 0.85, r), 0.8), 5), 75),
}
def ph(im):
    p = Image.fromarray(im); return np.array([imagehash.phash(t, hash_size=16).hash.flatten() for t in
        (p, p.transpose(Image.FLIP_LEFT_RIGHT), p.rotate(90, expand=True), p.rotate(180, expand=True), p.rotate(270, expand=True),
         p.transpose(Image.FLIP_LEFT_RIGHT).rotate(90, expand=True), p.transpose(Image.FLIP_LEFT_RIGHT).rotate(180, expand=True),
         p.transpose(Image.FLIP_LEFT_RIGHT).rotate(270, expand=True))], bool)
def prep(im, flip=False):
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
def stage4(imA, imB):
    A, Af, B = prep(imA), prep(imA, True), prep(imB); best = (0, 0.0, 0.0, 1.0)
    for a in (A, Af):
        M, k = match(a, B)
        if M is None: continue
        cov, ncc = verify(a, B, M); s = float(np.sqrt(abs(np.linalg.det(M[:, :2]))))
        if ncc > best[2] or (best[2] == 0 and k > best[0]): best = (k, cov, ncc, s)
    k, cov, ncc, s = best
    return bool(k >= 10 and 0.5 <= s <= 2 and cov >= 0.3 and ncc >= 0.70), best
def work(i):
    rng = np.random.default_rng(1000 + i); im = read_rgb(reps.image_path[i]); H0 = ph(im); out = []
    others = rng.choice(np.delete(np.arange(len(reps)), i), 5, replace=False)
    for name, f in TRANSFORMS.items():
        t = f(im, rng); Ht = ph(t)
        d2 = min((Ht[0] ^ H0[0]).sum(), (Ht[0] ^ H0[1]).sum()); d3 = min((Ht[0] ^ h).sum() for h in H0)
        acc, best = stage4(im, t)
        spur = sum(stage4(read_rgb(reps.image_path[o]), t)[0] for o in others)
        out.append(dict(i=int(i), transform=name, s2=bool(d2 <= 20), s3=bool(d3 <= 20), s4=acc, d2=int(d2), d3=int(d3),
                        inl=int(best[0]), cov=best[1], ncc=best[2], scale=best[3], spurious=int(spur)))
    return out
if __name__ == '__main__':
    t0 = time.time()
    with Pool(4) as pool: res = sum(pool.map(work, range(len(reps)), chunksize=4), [])
    df = pd.DataFrame(res); df.to_csv(OUT + 'synthetic_pairs.csv', index=False)
    summ = []
    for name in TRANSFORMS:
        s = df[df['transform'] == name]
        summ.append(dict(transform=name, n=int(len(s)), stage2=float(s.s2.mean()), stage3=float((s.s2 | s.s3).mean()),
                         stage4=float(s.s4.mean()), any_stage=float((s.s2 | s.s3 | s.s4).mean()), spurious=int(s.spurious.sum()), spurious_pairs=int(5 * len(s))))
    R = dict(n_images=int(len(reps)), summary=summ, total_spurious=int(df.spurious.sum()), total_spurious_pairs=int(5 * len(df)), seconds=time.time() - t0)
    json.dump(R, open(OUT + 'synthetic.json', 'w'), indent=1); print(json.dumps(R, indent=1))
