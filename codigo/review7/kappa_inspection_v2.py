# Califica la inspeccion ciega del kit ampliado (etapa 4) con dos o mas hojas de respuestas.
# Reporta: el kappa de Cohen entre cada par de evaluadores (IC 95 % por bootstrap), el kappa de Fleiss si hay tres o mas,
# la proporcion de pares juzgados "mismo ojo" en cada tipo de par (aceptados, rechazados parecidos, al azar),
# el acuerdo de cada evaluador con el detector y cuantos evaluadores juzgaron "mismo ojo" cada par.
# Uso: python kappa_inspection_v2.py <hoja 1>.csv <hoja 2>.csv [<hoja 3>.csv ...] CLAVE_no_mostrar_a_evaluadores.csv [--json salida.json]
# Los evaluadores se nombran E1, E2, ... en el orden en que se pasan las hojas. Sirve con las hojas llenadas a mano
# (columnas code, respuesta) y con las que descarga el calificador en pantalla (traen ademas letra, inicio, fin y minutos).
import sys, json, itertools
import numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score

args = sys.argv[1:]
out_json = None
if '--json' in args:
    k = args.index('--json')
    out_json = args[k + 1]
    del args[k:k + 2]
if len(args) < 3:
    sys.exit('Uso: python kappa_inspection_v2.py <hoja 1>.csv <hoja 2>.csv [...] CLAVE.csv [--json salida.json]')
key = pd.read_csv(args[-1])
names = ['E%d' % (n + 1) for n in range(len(args) - 1)]
d = key.copy()
info = {}
for name, f in zip(names, args[:-1]):
    h = pd.read_csv(f)
    info[name] = dict(archivo=f.replace('\\', '/').split('/')[-1])
    if 'minutos' in h.columns:
        info[name].update(letra=str(h.evaluador.iloc[0]), inicio=str(h.inicio.iloc[0]), fin=str(h.fin.iloc[0]), minutos=int(h.minutos.max()))
    d = d.merge(h[['code', 'respuesta']].rename(columns={'respuesta': name}), on='code', validate='one_to_one')
    print('%s: %s' % (name, ', '.join('%s %s' % kv for kv in info[name].items())))
assert len(d) == len(key), 'las hojas no tienen los mismos pares que la clave'
falta = d[d[names].isna().any(axis=1)]
if len(falta):
    sys.exit('Faltan respuestas en %d pares (por ejemplo %s). Complete las hojas antes de calificar.' % (len(falta), ', '.join(falta.code.head(5))))
for r in names:
    d[r] = d[r].astype(int)
    malas = d[~d[r].isin([0, 1, 9])]
    if len(malas):
        sys.exit('El evaluador %s tiene respuestas distintas de 0, 1 o 9 (por ejemplo en %s).' % (r, ', '.join(malas.code.head(5))))
d['det'] = (d.kind == 'accepted').astype(int)
res = dict(pares=int(len(d)), evaluadores=info, por_tipo={}, kappa_entre_evaluadores={}, acuerdo_con_detector={})
print('pares: %d; "no puedo decidir": %s' % (len(d), ', '.join('%s = %d' % (r, int((d[r] == 9).sum())) for r in names)))


def boot_kappa(x, y, seed=2026, n=2000):
    x, y = np.asarray(x), np.asarray(y)
    rng = np.random.default_rng(seed)
    b = []
    for _ in range(n):
        idx = rng.integers(0, len(x), len(x))
        if len(np.unique(np.r_[x[idx], y[idx]])) > 1:
            b.append(cohen_kappa_score(x[idx], y[idx]))
    return float(cohen_kappa_score(x, y)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def fleiss(M):
    # M: una fila por par, una columna por categoria, con el numero de evaluadores que eligio cada categoria
    M = np.asarray(M, float)
    n = M.sum(1)[0]
    p = M.sum(0) / M.sum()
    P = ((M ** 2).sum(1) - n) / (n * (n - 1))
    Pe = (p ** 2).sum()
    return float((P.mean() - Pe) / (1 - Pe))


print('\nKappa de Cohen entre evaluadores (pares que ambos decidieron):')
for a, b in itertools.combinations(names, 2):
    x = d[(d[a] != 9) & (d[b] != 9)]
    kap, lo, hi = boot_kappa(x[a], x[b])
    ac = float((x[a] == x[b]).mean())
    res['kappa_entre_evaluadores']['%s-%s' % (a, b)] = dict(kappa=round(kap, 3), ic95=[round(lo, 3), round(hi, 3)], acuerdo=round(ac, 3), n=int(len(x)))
    print('  %s y %s: kappa %.3f (IC 95 %% %.3f a %.3f), acuerdo bruto %.1f %%, n = %d' % (a, b, kap, lo, hi, 100 * ac, len(x)))
if len(names) >= 3:
    x = d[(d[names] != 9).all(axis=1)]
    M = np.c_[(x[names] == 0).sum(axis=1), (x[names] == 1).sum(axis=1)]
    kf = fleiss(M)
    rng = np.random.default_rng(2026)
    b = [fleiss(M[rng.integers(0, len(M), len(M))]) for _ in range(2000)]
    res['kappa_fleiss'] = dict(kappa=round(kf, 3), ic95=[round(float(np.percentile(b, 2.5)), 3), round(float(np.percentile(b, 97.5)), 3)], n=int(len(x)))
    print('  kappa de Fleiss entre los %d: %.3f (IC 95 %% %.3f a %.3f), n = %d pares que todos decidieron' % (len(names), kf, np.percentile(b, 2.5), np.percentile(b, 97.5), len(x)))

print('\nPares juzgados "mismo ojo" por tipo de par (mismo ojo / ojos distintos / no puedo decidir):')
for kind, g in d.groupby('kind'):
    res['por_tipo'][kind] = dict(n=int(len(g)))
    linea = '  %-20s n = %3d' % (kind, len(g))
    for r in names:
        c = [int((g[r] == v).sum()) for v in (1, 0, 9)]
        dec = c[0] + c[1]
        res['por_tipo'][kind][r] = dict(mismo=c[0], distintos=c[1], no_decide=c[2], pct_mismo_entre_decididos=round(100 * c[0] / dec, 1) if dec else None)
        linea += ' | %s: %d / %d / %d (%.1f %%)' % (r, c[0], c[1], c[2], 100 * c[0] / dec if dec else float('nan'))
    print(linea)

print('\nCuantos evaluadores juzgaron "mismo ojo" cada par (columnas) y cuantos "ojos distintos":')
d['n_mismo'] = (d[names] == 1).sum(axis=1)
d['n_distintos'] = (d[names] == 0).sum(axis=1)
t1 = pd.crosstab(d.kind, d.n_mismo)
t0 = pd.crosstab(d.kind, d.n_distintos)
print(t1)
print(t0)
res['evaluadores_que_dicen_mismo_ojo'] = {k: {str(c): int(v) for c, v in row.items()} for k, row in t1.iterrows()}
res['evaluadores_que_dicen_ojos_distintos'] = {k: {str(c): int(v) for c, v in row.items()} for k, row in t0.iterrows()}
may = len(names) // 2 + 1
acc = d[d.kind == 'accepted']
n_false = int((acc.n_distintos >= may).sum())
res['aceptados_que_la_mayoria_juzga_distintos'] = dict(n=n_false, de=int(len(acc)),
                                                       cota_superior_95_si_cero=round(100 * (1 - 0.025 ** (1 / len(acc))), 2) if n_false == 0 else None)
print('pares aceptados que la mayoria (%d de %d) juzgo "ojos distintos": %d de %d' % (may, len(names), n_false, len(acc)))

print('\nAcuerdo de cada evaluador con el detector (aceptado = mismo ojo):')
for r in names:
    x = d[d[r] != 9]
    kap, lo, hi = boot_kappa(x[r], x.det)
    ac = float((x[r] == x.det).mean())
    res['acuerdo_con_detector'][r] = dict(kappa=round(kap, 3), ic95=[round(lo, 3), round(hi, 3)], acuerdo=round(ac, 3), n=int(len(x)))
    print('  %s: kappa %.3f (IC 95 %% %.3f a %.3f), acuerdo %.1f %% (n = %d)' % (r, kap, lo, hi, 100 * ac, len(x)))
if len(names) == 2:
    print('\nTabla cruzada (tipo de par por respuestas de E1 y de E2):')
    print(pd.crosstab(d.kind, [d.E1, d.E2]))
if out_json:
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print('\nresumen guardado en', out_json)
