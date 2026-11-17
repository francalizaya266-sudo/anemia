# Califica la inspeccion ciega del kit ampliado (etapa 4).
# Reporta: el kappa de Cohen entre los dos evaluadores (con IC 95 % por bootstrap), la proporcion de pares juzgados
# "mismo ojo" en cada tipo de par (aceptados, rechazados parecidos, al azar) y el acuerdo de cada evaluador con el detector.
# Uso: python kappa_inspection_v2.py <hoja de A>.csv <hoja de B>.csv CLAVE_no_mostrar_a_evaluadores.csv
# Sirve con las hojas llenadas a mano (columnas code, respuesta) y con las que descarga el calificador en pantalla.
import sys
import numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score

a, b, key = (pd.read_csv(f) for f in sys.argv[1:4])
# las hojas del calificador en pantalla traen ademas el evaluador, el inicio, el fin y los minutos de trabajo
for letra, hoja in (('A', a), ('B', b)):
    if 'evaluador' in hoja.columns:
        quien = set(hoja.evaluador.dropna().astype(str))
        if quien != {letra}:
            sys.exit('La hoja que se paso como %s dice ser del evaluador %s. Revise el orden de los archivos.' % (letra, ', '.join(sorted(quien))))
    if 'minutos' in hoja.columns:
        print('hoja %s: inicio %s, fin %s, %d minutos de trabajo' % (letra, hoja.inicio.iloc[0], hoja.fin.iloc[0], int(hoja.minutos.max())))
a, b = a[['code', 'respuesta']], b[['code', 'respuesta']]
d = key.merge(a.rename(columns={'respuesta': 'A'}), on='code', validate='one_to_one').merge(b.rename(columns={'respuesta': 'B'}), on='code', validate='one_to_one')
falta = d[d.A.isna() | d.B.isna()]
if len(falta):
    sys.exit('Faltan respuestas en %d pares (por ejemplo %s). Complete las dos hojas antes de calificar.' % (len(falta), ', '.join(falta.code.head(5))))
for r in ('A', 'B'):
    d[r] = d[r].astype(int)
    malas = d[~d[r].isin([0, 1, 9])]
    if len(malas):
        sys.exit('El evaluador %s tiene respuestas distintas de 0, 1 o 9 (por ejemplo en %s).' % (r, ', '.join(malas.code.head(5))))
d['det'] = (d.kind == 'accepted').astype(int)
both = d[(d.A != 9) & (d.B != 9)]
print('pares: %d; "no puedo decidir": A = %d, B = %d' % (len(d), int((d.A == 9).sum()), int((d.B == 9).sum())))

kap = cohen_kappa_score(both.A, both.B)
rng = np.random.default_rng(2026)
A, B = both.A.values, both.B.values
boots = []
for _ in range(2000):
    idx = rng.integers(0, len(both), len(both))
    if len(np.unique(np.r_[A[idx], B[idx]])) > 1:
        boots.append(cohen_kappa_score(A[idx], B[idx]))
print('kappa de Cohen entre A y B: %.3f (IC 95 %% %.3f a %.3f; n = %d pares decididos por ambos)'
      % (kap, np.percentile(boots, 2.5), np.percentile(boots, 97.5), len(both)))
print('acuerdo bruto entre A y B: %.1f %%' % (100 * (both.A == both.B).mean()))

print('\nProporcion juzgada "mismo ojo" por tipo de par (entre los pares que el evaluador decidio):')
for kind, g in d.groupby('kind'):
    linea = '  %-20s n = %3d' % (kind, len(g))
    for r in ('A', 'B'):
        x = g[g[r] != 9]
        linea += ' | %s: %d de %d (%.1f %%)' % (r, int((x[r] == 1).sum()), len(x), 100 * (x[r] == 1).mean() if len(x) else float('nan'))
    x = g[(g.A != 9) & (g.B != 9)]
    linea += ' | ambos: %d de %d' % (int(((x.A == 1) & (x.B == 1)).sum()), len(x))
    print(linea)

print('\nAcuerdo de cada evaluador con el detector (aceptado = mismo ojo):')
for r in ('A', 'B'):
    x = d[d[r] != 9]
    print('  evaluador %s: kappa %.3f, acuerdo %.1f %% (n = %d)' % (r, cohen_kappa_score(x[r], x.det), 100 * (x[r] == x.det).mean(), len(x)))
print('\nTabla cruzada (tipo de par por respuestas de A y de B):')
print(pd.crosstab(d.kind, [d.A, d.B]))
