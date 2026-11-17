# Calificador en pantalla para el kit ampliado de inspeccion ciega (observacion 2 del docente).
# Rehace la seleccion de kit_v2.py con la misma semilla y escribe un unico archivo HTML que funciona sin internet:
# los mismos 356 pares, con los mismos codigos P001-P356, en el mismo orden y con la misma imagen a cada lado que
# el PDF. El archivo no lleva nombres de archivo, etiquetas, Hb, establecimiento ni el tipo de par.
# Comprueba que la clave que reconstruye coincide con solo_coordinador/CLAVE_no_mostrar_a_evaluadores.csv.
# Uso: python kit_v2_html.py <repo> <carpeta del conjunto de datos> <carpeta del kit> <plantilla.html> [archivo de salida]
# El archivo que produce contiene imagenes del conjunto: es de uso interno y no va al repositorio.
import sys, os, json, base64, hashlib
import numpy as np, pandas as pd, cv2

REPO, DATA, KIT, TEMPLATE = sys.argv[1:5]
OUT = sys.argv[5] if len(sys.argv) > 5 else os.path.join(KIT, 'calificador_pares_v2.html')
SEED, N_REJ, N_RANDOM = 20261117, 168, 20


def png_data_uri(rel):
    im = cv2.imread(os.path.join(DATA, rel), cv2.IMREAD_UNCHANGED)
    if im.ndim == 3 and im.shape[2] == 4:
        im = im[:, :, :3]  # igual que el PDF: se descarta el canal alfa
    ok, buf = cv2.imencode('.png', im, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    assert ok
    return 'data:image/png;base64,' + base64.b64encode(buf.tobytes()).decode('ascii'), im.shape[1], im.shape[0]


# --- seleccion: copia literal de kit_v2.py ---
mfull = pd.read_csv(os.path.join(REPO, 'codigo/meta_groups.csv'))
ms = pd.read_csv(os.path.join(REPO, 'codigo/meta_groups_strict.csv')).set_index('image_id')
mg = pd.read_csv(os.path.join(REPO, 'codigo/meta_groups_geo.csv')).set_index('image_id')
reps = mfull.sort_values('image_id').drop_duplicates('grp').reset_index(drop=True)
sg = ms.loc[reps.image_id, 'grp'].values
gg = mg.loc[reps.image_id, 'grp'].values
a = np.load(os.path.join(REPO, 'resultados/review2/geomverify.npy'))
i, j = a[:, 0].astype(int), a[:, 1].astype(int)
k, cov, ncc, s = a[:, 2], a[:, 3], a[:, 4], a[:, 6]
acc = (k >= 10) & (s >= 0.5) & (s <= 2) & (cov >= 0.3) & (ncc >= 0.7)
stage3 = sg[i] == sg[j]
samegeo = gg[i] == gg[j]
n_cand = int((np.load(os.path.join(REPO, 'resultados/review2/orb_pairs.npy'))[:, 3] >= 10).sum())
is_cand = np.arange(len(a)) < n_cand
new = np.where(acc & ~stage3)[0]
assert len(new) == 168 and samegeo[new].all()
rej_pool = np.where(is_cand & ~acc & ~stage3 & ~samegeo)[0]
rej = rej_pool[np.argsort(-ncc[rej_pool], kind='stable')][:N_REJ]
rng = np.random.default_rng(SEED)
low_pool = np.where(~is_cand & ~acc & ~samegeo)[0]
low = rng.choice(low_pool, N_RANDOM, replace=False)
pick = list(new) + list(rej) + list(low)
kind = ['accepted'] * len(new) + ['rejected_lookalike'] * len(rej) + ['random'] * len(low)
order = rng.permutation(len(pick))

# --- mismos codigos y mismo lado que el PDF ---
rows, shown = [], []
for idx in order:
    p = pick[idx]
    code = 'P%03d' % (len(rows) + 1)
    left, right = reps.image_path[i[p]], reps.image_path[j[p]]
    if rng.random() < 0.5:  # el PDF intercambia los lados con esta misma tirada
        left, right = right, left
    shown.append((code, left, right))
    rows.append(dict(code=code, kind=kind[idx], a=reps.image_id[i[p]], b=reps.image_id[j[p]]))

key = pd.read_csv(os.path.join(KIT, 'solo_coordinador', 'CLAVE_no_mostrar_a_evaluadores.csv'))
mine = pd.DataFrame(rows)
assert len(mine) == len(key) == 356
assert (mine[['code', 'kind', 'a', 'b']].values == key[['code', 'kind', 'a', 'b']].values).all(), 'la clave no coincide'

# --- imagenes: cada una se guarda una vez, con un indice barajado que no dice nada del archivo ---
paths = sorted({p for _, l, r in shown for p in (l, r)})
perm = np.random.default_rng(SEED + 1).permutation(len(paths))
slot = {p: int(perm[n]) for n, p in enumerate(paths)}
images = [None] * len(paths)
for p in paths:
    uri, w, h = png_data_uri(p)
    images[slot[p]] = [uri, w, h]
pairs = [[code, slot[l], slot[r]] for code, l, r in shown]

html = open(TEMPLATE, encoding='utf-8').read()
assert html.count('/*__KIT_DATA__*/') == 1
# nada en la parte legible de los datos puede delatar el archivo, la etiqueta o el tipo de par
legible = json.dumps(pairs)
for banned in ('Image', 'accepted', 'rejected', 'random', 'anemi', 'ncc', 'png"'):
    assert banned not in legible, banned
assert all(len(p) == 3 and isinstance(p[1], int) and isinstance(p[2], int) for p in pairs)
payload = json.dumps(dict(kit='v2', pairs=pairs, images=images), separators=(',', ':'))
with open(OUT, 'w', encoding='utf-8') as f:
    f.write(html.replace('/*__KIT_DATA__*/', 'window.KIT=' + payload + ';'))
sha = hashlib.sha256(open(OUT, 'rb').read()).hexdigest()
print('pares: %d; imagenes distintas: %d; archivo: %.1f MB; sha256: %s' % (len(pairs), len(images), os.path.getsize(OUT) / 1e6, sha))
