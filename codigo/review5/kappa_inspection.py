# Scores the blinded inspection: Cohen's kappa between the two raters, and each rater against the detector decision.
# Usage: python kappa_inspection.py respuestas_evaluador_A.csv respuestas_evaluador_B.csv CLAVE_no_mostrar_a_evaluadores.csv
import sys, pandas as pd
from sklearn.metrics import cohen_kappa_score
a, b, key = (pd.read_csv(f) for f in sys.argv[1:4])
d = key.merge(a.rename(columns={'respuesta': 'A'}), on='code').merge(b.rename(columns={'respuesta': 'B'}), on='code')
d['det'] = (d.kind == 'accepted').astype(int)
both = d[(d.A != 9) & (d.B != 9)]
print('pairs', len(d), 'unsure A/B', int((d.A == 9).sum()), int((d.B == 9).sum()))
print('kappa A vs B: %.3f (n = %d)' % (cohen_kappa_score(both.A, both.B), len(both)))
for r in ('A', 'B'):
    x = d[d[r] != 9]; print('rater %s vs detector: kappa %.3f, agreement %.1f %%' % (r, cohen_kappa_score(x[r], x.det), 100 * (x[r] == x.det).mean()))
print(pd.crosstab(d.kind, [d.A, d.B]))
