# Kit ampliado de inspeccion ciega de la etapa 4 (respuesta a la observacion 2 del docente).
# 356 pares: los 168 enlaces nuevos que el detector acepto, los 168 candidatos rechazados con mayor NCC
# (los mas cercanos a la aceptacion) y 20 pares al azar como control. Se muestran mezclados, con codigo P001-P356,
# sin nombre de archivo, etiqueta, Hb ni establecimiento. La clave va a un archivo aparte que los evaluadores no ven.
# Uso: python kit_v2.py <repo> <carpeta del conjunto de datos> <carpeta de salida>
import sys, os
import numpy as np, pandas as pd, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

REPO, DATA, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
SEED, N_REJ, N_RANDOM, PER_PAGE = 20261117, 168, 20, 4
os.makedirs(os.path.join(OUT, 'solo_coordinador'), exist_ok=True)


def read_rgb(rel):
    im = cv2.imread(os.path.join(DATA, rel), cv2.IMREAD_UNCHANGED)
    if im.ndim == 3 and im.shape[2] == 4:
        im = im[:, :, :3]
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


mfull = pd.read_csv(os.path.join(REPO, 'codigo/meta_groups.csv'))
ms = pd.read_csv(os.path.join(REPO, 'codigo/meta_groups_strict.csv')).set_index('image_id')
mg = pd.read_csv(os.path.join(REPO, 'codigo/meta_groups_geo.csv')).set_index('image_id')
# los 472 representantes del mapa preespecificado, en el mismo orden que uso la etapa 4
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
is_cand = np.arange(len(a)) < n_cand  # las ultimas 1000 filas son controles al azar

new = np.where(acc & ~stage3)[0]
assert len(new) == 168 and samegeo[new].all()
# rechazados mas cercanos a la aceptacion: candidatos de la busqueda de puntos clave que la verificacion rechazo,
# cuyas dos imagenes quedaron en grupos geometricos distintos, ordenados por NCC descendente
rej_pool = np.where(is_cand & ~acc & ~stage3 & ~samegeo)[0]
rej = rej_pool[np.argsort(-ncc[rej_pool], kind='stable')][:N_REJ]
rng = np.random.default_rng(SEED)
low_pool = np.where(~is_cand & ~acc & ~samegeo)[0]
low = rng.choice(low_pool, N_RANDOM, replace=False)

pick = list(new) + list(rej) + list(low)
kind = ['accepted'] * len(new) + ['rejected_lookalike'] * len(rej) + ['random'] * len(low)
assert len(set(pick)) == len(pick) == 168 + N_REJ + N_RANDOM
order = rng.permutation(len(pick))
rows = []
with PdfPages(os.path.join(OUT, 'pares_inspeccion_ciega_v2.pdf')) as pdf:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.08, 0.90, 'Inspeccion ciega de pares (etapa 4), kit ampliado', fontsize=16, weight='bold')
    fig.text(0.08, 0.70,
             'Para cada par P001-P%03d decida, sin consultar a nadie ni ningun otro archivo:\n\n'
             '  1 = las dos imagenes muestran el mismo ojo (la misma foto recortada, reescalada,\n'
             '       espejada o rotada, o una segunda toma del mismo ojo)\n'
             '  0 = son ojos distintos\n'
             '  9 = no puedo decidir\n\n'
             'Anote la respuesta en la columna "respuesta" de su hoja\n'
             '(respuestas_evaluador_A.csv o respuestas_evaluador_B.csv).\n\n'
             'Los pares estan en orden aleatorio. No se muestra el nombre del archivo, la etiqueta\n'
             'de anemia, la hemoglobina ni el establecimiento. Mire los vasos, los pliegues y el\n'
             'contorno del tejido; una imagen puede estar espejada, rotada o recortada respecto\n'
             'de la otra. Trabaje solo, sin comentar los pares con el otro evaluador.' % len(pick),
             fontsize=11, va='top')
    pdf.savefig(fig)
    plt.close(fig)
    for page in range(0, len(order), PER_PAGE):
        fig, axes = plt.subplots(PER_PAGE, 2, figsize=(8.27, 11.69))
        chunk = order[page:page + PER_PAGE]
        for r, idx in enumerate(chunk):
            p = pick[idx]
            code = 'P%03d' % (len(rows) + 1)
            A = read_rgb(reps.image_path[i[p]])
            B = read_rgb(reps.image_path[j[p]])
            if rng.random() < 0.5:
                A, B = B, A
            for c, im in enumerate((A, B)):
                ax = axes[r, c]
                ax.imshow(im)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_facecolor('k')
            axes[r, 0].set_ylabel(code, fontsize=14, rotation=0, labelpad=32, weight='bold')
            rows.append(dict(code=code, kind=kind[idx], a=reps.image_id[i[p]], b=reps.image_id[j[p]], inliers=int(k[p]),
                             ncc=round(float(ncc[p]), 3), overlap=round(float(cov[p]), 3), scale=round(float(s[p]), 3)))
        for r in range(len(chunk), PER_PAGE):
            for c in range(2):
                axes[r, c].axis('off')
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

key = pd.DataFrame(rows)
key.to_csv(os.path.join(OUT, 'solo_coordinador', 'CLAVE_no_mostrar_a_evaluadores.csv'), index=False)
for r in ('A', 'B'):
    pd.DataFrame(dict(code=key.code, respuesta='')).to_csv(os.path.join(OUT, 'respuestas_evaluador_%s.csv' % r), index=False)
print(key.kind.value_counts().to_dict())
print('NCC de los rechazados: %.3f a %.3f; con NCC >= 0.70: %d' % (ncc[rej].min(), ncc[rej].max(), int((ncc[rej] >= 0.7).sum())))
