# Blinded two-rater inspection kit for the stage-4 links: 70 pairs (30 accepted new links, 30 rejected look-alike candidates,
# 10 random pairs), shuffled and coded P01-P70, shown without file names, labels or clinical data. The key is written to a
# separate file that the raters must not see. kappa_inspection.py scores two answer sheets against each other and the key.
import sys; sys.path.insert(0, '/home/claude/exp')
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from common import read_rgb
REPO = '/home/claude/anemia/'; OUT = '/home/claude/exp/out/review5/kit/'
mfull = pd.read_csv(REPO + 'codigo/meta_groups.csv'); ms = pd.read_csv(REPO + 'codigo/meta_groups_strict.csv').set_index('image_id')
reps = mfull.sort_values('image_id').drop_duplicates('grp').reset_index(drop=True); sg = ms.loc[reps.image_id, 'grp'].values
a = np.load(REPO + 'resultados/review2/geomverify.npy'); i, j = a[:, 0].astype(int), a[:, 1].astype(int); k, cov, ncc, s = a[:, 2], a[:, 3], a[:, 4], a[:, 6]
acc = (k >= 10) & (s >= 0.5) & (s <= 2) & (cov >= 0.3) & (ncc >= 0.7); stage3 = sg[i] == sg[j]
n_orb = int((np.load(REPO + 'resultados/review2/orb_pairs.npy')[:, 3] >= 10).sum()); is_cand = np.arange(len(a)) < n_orb  # last 1000 rows: random controls
rng = np.random.default_rng(2026)
new = np.where(acc & ~stage3)[0]; low = np.where(~is_cand)[0]
# rejected look-alikes: every rejected candidate (>= 10 inliers, different stage-3 group) that came closest to acceptance
# (NCC >= 0.2 or a non-degenerate scale), topped up with the rejected candidates that had the most inliers
rej_all = np.where(is_cand & (k >= 10) & ~acc & ~stage3)[0]
near = rej_all[(ncc[rej_all] >= 0.2) | (s[rej_all] > 0.1)]
rest = np.setdiff1d(rej_all, near); rest = rest[np.argsort(-k[rest])]
rej = np.concatenate([near, rest])[:30]
pick_new = list(rng.choice(new, 30, replace=False)); pick_rej = list(rej); pick_low = list(rng.choice(low, 10, replace=False))
pick = pick_new + pick_rej + pick_low
kind = ['accepted'] * len(pick_new) + ['rejected_lookalike'] * len(pick_rej) + ['random'] * len(pick_low)
assert len(kind) == len(pick) == 70
order = rng.permutation(len(pick)); rows = []
with PdfPages(OUT + 'pares_inspeccion_ciega.pdf') as pdf:
    fig = plt.figure(figsize=(8.27, 11.69)); fig.text(0.08, 0.9, 'Inspeccion ciega de pares (etapa 4)', fontsize=16, weight='bold')
    fig.text(0.08, 0.84, 'Para cada par P01-P70 decida, sin consultar a nadie ni ningun otro archivo:\n  1 = las dos imagenes muestran el mismo ojo (misma foto recortada, reescalada,\n       espejada o una segunda toma del mismo ojo)\n  0 = son ojos distintos\n  9 = no puedo decidir\nAnote la respuesta en la columna de su hoja (respuestas_evaluador_X.csv).\nLos pares estan en orden aleatorio y no llevan etiqueta, Hb ni nombre de archivo.', fontsize=10, va='top', family='monospace')
    pdf.savefig(fig); plt.close(fig)
    for page in range(0, len(order), 4):
        fig, axes = plt.subplots(4, 2, figsize=(8.27, 11.69))
        for r, idx in enumerate(order[page:page + 4]):
            p = pick[idx]; code = 'P%02d' % (len(rows) + 1)
            A = read_rgb(reps.image_path[i[p]]); B = read_rgb(reps.image_path[j[p]])
            if rng.random() < 0.5: A, B = B, A
            for c, im in enumerate((A, B)):
                ax = axes[r, c]; ax.imshow(im); ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor('k')
            axes[r, 0].set_ylabel(code, fontsize=14, rotation=0, labelpad=28, weight='bold')
            rows.append(dict(code=code, kind=kind[idx], a=reps.image_id[i[p]], b=reps.image_id[j[p]], inliers=int(k[p]), ncc=round(float(ncc[p]), 3), overlap=round(float(cov[p]), 3), scale=round(float(s[p]), 3)))
        for r in range(len(order[page:page + 4]), 4):
            for c in range(2): axes[r, c].axis('off')
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
key = pd.DataFrame(rows); key.to_csv(OUT + 'CLAVE_no_mostrar_a_evaluadores.csv', index=False)
for r in ('A', 'B'): pd.DataFrame(dict(code=key.code, respuesta='')).to_csv(OUT + 'respuestas_evaluador_%s.csv' % r, index=False)
print(key.kind.value_counts().to_dict())
