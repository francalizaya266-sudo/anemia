# Observacion 1 del docente, parte en CPU: clasificador que solo ve la forma de la mascara y area de la mascara,
# recalculados en el conjunto geometrico (D-337). Se repite tambien en D-471 para comprobar que el codigo reproduce
# la cifra publicada (0.627, IC 0.574-0.677). Mismo protocolo que codigo/review/review_analyses_b.py (seccion 5):
# Random Forest de 500 arboles, validacion cruzada anidada 5 x 5, 3 repeticiones, AUC de la media de las repeticiones.
# Uso: python review7_shape.py <repo> <carpeta del conjunto de datos> <carpeta de salida>
import sys, os, json
import numpy as np, pandas as pd, cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score

REPO, DATA, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
SEED = 42
os.makedirs(OUT, exist_ok=True)


def read_rgb(rel):
    im = cv2.imread(os.path.join(DATA, rel), cv2.IMREAD_UNCHANGED)
    if im.ndim == 3 and im.shape[2] == 4:
        im = im[:, :, :3]
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def mask_of(im, t=12):
    return im.max(axis=2) > t


def load(meta):
    d = pd.read_csv(os.path.join(REPO, 'codigo', meta))
    conf = d.groupby('grp').anemia.nunique()
    d = d[~d.grp.isin(conf[conf > 1].index)]
    return d.sort_values('image_id').drop_duplicates('grp').reset_index(drop=True)


def shape_feats(im):
    m = mask_of(im).astype(np.uint8); h, w = m.shape; f = {}
    f['area_frac'] = m.mean(); ys_, xs_ = np.nonzero(m)
    if len(xs_) < 5:
        return {k: 0.0 for k in ['area_frac', 'bbox_w', 'bbox_h', 'aspect', 'extent', 'solidity', 'perimeter_n', 'circularity', 'cx', 'cy', 'angle', 'ell_ratio'] + ['hu%d' % i for i in range(7)]}
    x0, x1, y0, y1 = xs_.min(), xs_.max(), ys_.min(), ys_.max(); f['bbox_w'] = (x1 - x0 + 1) / w; f['bbox_h'] = (y1 - y0 + 1) / h; f['aspect'] = (x1 - x0 + 1) / (y1 - y0 + 1)
    f['extent'] = m.sum() / ((x1 - x0 + 1) * (y1 - y0 + 1))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); c = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(c); f['solidity'] = cv2.contourArea(c) / max(cv2.contourArea(hull), 1); per = cv2.arcLength(c, True); f['perimeter_n'] = per / np.sqrt(h * w)
    f['circularity'] = 4 * np.pi * cv2.contourArea(c) / max(per ** 2, 1); f['cx'] = xs_.mean() / w; f['cy'] = ys_.mean() / h
    if len(c) >= 5:
        (ex, ey), (ma, mi), ang = cv2.fitEllipse(c); f['angle'] = ang; f['ell_ratio'] = mi / max(ma, 1e-6)
    else:
        f['angle'] = 0; f['ell_ratio'] = 1
    hu = cv2.HuMoments(cv2.moments(m)).flatten()
    for i, v in enumerate(hu):
        f['hu%d' % i] = float(-np.sign(v) * np.log10(abs(v) + 1e-12))
    return f


def run(X, y, est, grid, reps):
    P = np.zeros((reps, len(y)))
    for r in range(reps):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED + r).split(np.zeros(len(y)), y):
            gs = GridSearchCV(est, grid, scoring='roc_auc', cv=StratifiedKFold(5, shuffle=True, random_state=r), n_jobs=1).fit(X.iloc[tr], y[tr])
            P[r, te] = gs.predict_proba(X.iloc[te])[:, 1]
    return P


def auc_ci(y, p, B=2000, seed=SEED):
    rng = np.random.default_rng(seed); n = len(y); a = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if len(np.unique(y[i])) < 2:
            continue
        a.append(roc_auc_score(y[i], p[i]))
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


R = {}
rf = RandomForestClassifier(500, class_weight='balanced', random_state=SEED, n_jobs=4)
grid = {'min_samples_leaf': [1, 3, 5], 'max_features': ['sqrt', 0.3]}
for label, meta in (('D-471', 'meta_groups.csv'), ('D-337', 'meta_groups_geo.csv')):
    d = load(meta); y = d.anemia.values
    imgs = [read_rgb(p) for p in d.image_path]
    S = pd.DataFrame([shape_feats(im) for im in imgs])
    P = run(S, y, rf, grid, reps=3); p = P.mean(0)
    a_nat = np.array([mask_of(im).sum() for im in imgs])
    R[label] = dict(n=int(len(y)), n_anaemic=int(y.sum()), n_feat=int(S.shape[1]),
                    shape_only_RF_AUC=float(roc_auc_score(y, p)), shape_only_RF_ci=auc_ci(y, p),
                    shape_only_RF_per_rep=[float(roc_auc_score(y, P[r])) for r in range(len(P))],
                    mask_area_native_AUC=float(roc_auc_score(y, -a_nat)), mask_area_native_ci=auc_ci(y, -a_nat.astype(float)))
    pd.DataFrame(dict(image_id=d.image_id, y=y, p_shape=p)).to_csv(os.path.join(OUT, 'shape_only_oof_%s.csv' % label), index=False)
    print(label, json.dumps(R[label]), flush=True)
json.dump(R, open(os.path.join(OUT, 'review7_shape.json'), 'w'), indent=1)
