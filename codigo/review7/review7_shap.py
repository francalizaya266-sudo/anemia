# Observacion 1 del docente, SHAP: repite el analisis SHAP de XGBoost en el conjunto geometrico (D-337) y lo compara
# con el publicado en D-471. Mismo procedimiento que codigo/xai_classic.py y codigo/review/review_analyses_b.py
# (seccion 8): hiperparametros elegidos por busqueda en rejilla con 5 particiones (random_state=0) sobre todo el
# conjunto, modelo reajustado con todas las imagenes y TreeExplainer sobre esas mismas imagenes. SHAP describe aqui
# el modelo ajustado, no su generalizacion. En D-471 se comprueba que el codigo reproduce las cifras publicadas.
# Uso: python review7_shap.py <repo> <carpeta del conjunto de datos> <carpeta de salida>
import sys, os, json
import numpy as np, pandas as pd, cv2, shap
from scipy.stats import spearmanr
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier

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


def color_features(im):                                  # identica a codigo/classic.py
    m = mask_of(im); px = im[m].astype(np.float32)
    if len(px) < 20: px = im.reshape(-1, 3).astype(np.float32)
    f = {}
    hsv = cv2.cvtColor(px[None].astype(np.uint8), cv2.COLOR_RGB2HSV)[0].astype(np.float32)
    lab = cv2.cvtColor((px[None] / 255).astype(np.float32), cv2.COLOR_RGB2LAB)[0]
    chans = {'R': px[:, 0], 'G': px[:, 1], 'B': px[:, 2], 'H': hsv[:, 0], 'S': hsv[:, 1], 'V': hsv[:, 2], 'L': lab[:, 0], 'a': lab[:, 1], 'b': lab[:, 2]}
    eps = 1.0
    chans['EI'] = np.log(px[:, 0] + eps) - np.log(px[:, 1] + eps)
    chans['rg'] = px[:, 0] / (px.sum(1) + eps)
    chans['gg'] = px[:, 1] / (px.sum(1) + eps)
    for k, v in chans.items():
        f[k + '_mean'] = v.mean(); f[k + '_std'] = v.std()
        f[k + '_p10'], f[k + '_p50'], f[k + '_p90'] = np.percentile(v, [10, 50, 90])
    f['a_over_L'] = chans['a'].mean() / (chans['L'].mean() + eps)
    f['R_over_G'] = px[:, 0].mean() / (px[:, 1].mean() + eps)
    f['frac_high_a'] = (chans['a'] > 20).mean()
    return f


def load(meta):
    d = pd.read_csv(os.path.join(REPO, 'codigo', meta))
    conf = d.groupby('grp').anemia.nunique()
    d = d[~d.grp.isin(conf[conf > 1].index)]
    return d.sort_values('image_id').drop_duplicates('grp').reset_index(drop=True)


FAMILIES = {'brightness': {'R', 'G', 'B', 'V', 'L'}, 'red_green_balance': {'rg', 'gg', 'EI', 'a_over_L', 'R_over_G', 'frac_high_a'},
            'cielab_a_b': {'a', 'b'}, 'hue_saturation': {'H', 'S'}}


def channel(c):
    return c if c in ('a_over_L', 'R_over_G', 'frac_high_a') else c.rsplit('_', 1)[0]


def analyse(meta):
    d = load(meta); y = d.anemia.values
    X = pd.DataFrame([color_features(read_rgb(p)) for p in d.image_path])
    est = XGBClassifier(n_estimators=300, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', n_jobs=4, random_state=SEED)
    grid = {'max_depth': [2, 3, 4], 'min_child_weight': [1, 5]}
    gs = GridSearchCV(est, grid, scoring='roc_auc', cv=StratifiedKFold(5, shuffle=True, random_state=0)).fit(X, y)
    m = gs.best_estimator_
    sv = shap.TreeExplainer(m).shap_values(X)
    imp = pd.Series(np.abs(sv).mean(0), index=X.columns)
    fam = {k: float(np.abs(sv[:, [j for j, c in enumerate(X.columns) if channel(c) in names]].sum(1)).mean()) for k, names in FAMILIES.items()}
    # direccion del efecto: correlacion de Spearman entre el valor de la caracteristica y su valor SHAP
    direction = {c: float(spearmanr(X[c], sv[:, j]).statistic) for j, c in enumerate(X.columns)}
    return dict(n=int(len(y)), best_params=gs.best_params_, inner_cv_auc=float(gs.best_score_), imp=imp, fam=fam, direction=direction, sv=sv, X=X)


A = analyse('meta_groups.csv'); B = analyse('meta_groups_geo.csv')
topA = list(A['imp'].sort_values(ascending=False).index); topB = list(B['imp'].sort_values(ascending=False).index)
published_top6 = ['G_p50', 'gg_p90', 'a_std', 'a_p10', 'rg_p90', 'rg_std']
R = {'published_top6_D471': published_top6, 'reproduced_top6_D471': topA[:6], 'reproduces_published': topA[:6] == published_top6}
for label, Z, top in (('D-471', A, topA), ('D-337', B, topB)):
    R[label] = dict(n=Z['n'], best_params=Z['best_params'], inner_cv_auc=Z['inner_cv_auc'],
                    top12={c: round(float(Z['imp'][c]), 4) for c in top[:12]}, family_mean_abs_sum=Z['fam'],
                    direction_top12={c: round(Z['direction'][c], 3) for c in top[:12]})
R['comparison'] = dict(
    spearman_mean_abs_shap_63_features=float(spearmanr(A['imp'].values, B['imp'][A['imp'].index].values).statistic),
    top6_in_common=sorted(set(topA[:6]) & set(topB[:6])), top12_in_common=sorted(set(topA[:12]) & set(topB[:12])),
    rank_in_D337_of_D471_top6={c: topB.index(c) + 1 for c in topA[:6]},
    direction_D471_vs_D337_for_D471_top12={c: [round(A['direction'][c], 3), round(B['direction'][c], 3)] for c in topA[:12]},
    family_rank_D471=sorted(A['fam'], key=A['fam'].get, reverse=True), family_rank_D337=sorted(B['fam'], key=B['fam'].get, reverse=True))
np.savez(os.path.join(OUT, 'shap_xgb_D337.npz'), sv=B['sv'], X=B['X'].values, cols=np.array(B['X'].columns))
json.dump(R, open(os.path.join(OUT, 'review7_shap.json'), 'w'), indent=1)
print(json.dumps(R, indent=1))
