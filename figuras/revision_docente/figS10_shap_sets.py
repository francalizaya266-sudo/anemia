# Supplementary Figure S10. SHAP of XGBoost on colour features, fitted on D-471 and on D-337 (teacher's observation 1).
# Input: resultados/review7/review7_shap.json (aggregate importances only; written by codigo/review7/review7_shap.py).
import json, sys, numpy as np
from matplotlib.lines import Line2D
from figstyle import *
REPO = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '')
R = json.load(open(REPO + 'resultados/review7/review7_shap.json'))
A, B = R['D-471'], R['D-337']
CH = {'rg': tr('red chromaticity', 'cromaticidad roja'), 'gg': tr('green chromaticity', 'cromaticidad verde'), 'EI': tr('erythema index', 'índice de eritema'),
      'a': 'a*', 'b': 'b*', 'L': 'L*', 'R': 'R', 'G': 'G', 'B': 'B', 'H': 'H', 'S': 'S', 'V': 'V'}
ST = {'mean': tr('mean', 'media'), 'std': tr('SD', 'DE'), 'p10': 'P10', 'p50': tr('median', 'mediana'), 'p90': 'P90'}
def name(c):
    if c in ('a_over_L', 'R_over_G', 'frac_high_a'): return {'a_over_L': 'a*/L*', 'R_over_G': 'R/G', 'frac_high_a': tr('fraction a* > 20', 'fracción a* > 20')}[c]
    ch, st = c.rsplit('_', 1); return CH[ch] + ', ' + ST[st]
feats = sorted(set(list(A['top12']) + list(B['top12'])), key=lambda c: -B['mean_abs_shap_all'][c]); n = len(feats)
RH = 0.212; Hh = 0.62 + RH * n + 0.52; fig = figure(Hh); c = canvas(fig)
panel(fig, 0.0, 0.03, 'a', tr('Mean |SHAP| of the most influential features', 'Media de |SHAP| de las características más influyentes'))
xl, xr, yt = 1.62, 3.30, 0.62
ax = axes(fig, xl, yt, xr - xl, RH * n); ax.set_xlim(0, 0.36); ax.set_xticks([0, 0.1, 0.2, 0.3]); ax.set_ylim(RH * n, 0); ax.set_yticks([]); ax.spines['left'].set_visible(False); ax.grid(axis='x')
c.text(xr + 0.14, yt - 0.10, tr('direction$^{*}$', 'dirección$^{*}$'), fontsize=FS_XS, color=GREY, va='center_baseline')
for i, f in enumerate(feats):
    yy = (i + 0.5) * RH; a, b = A['mean_abs_shap_all'][f], B['mean_abs_shap_all'][f]
    ax.plot([a, b], [yy, yy], color=LIGHT, lw=0.9, zorder=2)
    ax.plot([a], [yy], ls='', marker='o', mfc='white', mec=MID, mew=0.9, ms=4.2, zorder=3); ax.plot([b], [yy], ls='', marker='o', mfc=GREEN, mec=GREEN, ms=4.6, zorder=4)
    c.text(xl - 0.08, yt + yy, name(f), ha='right', va='center_baseline', fontsize=FS_S)
    da, db = A['direction_all'][f], B['direction_all'][f]
    c.text(xr + 0.14, yt + yy, ('%+.2f / %+.2f' % (da, db)).replace('-', '−'), fontsize=FS_XS, color=GREY, va='center_baseline')
ax.set_xlabel(tr('mean |SHAP| (log-odds)', 'media de |SHAP| (log-odds)'))
fig.legend(handles=[Line2D([], [], ls='', marker='o', mfc=GREEN, mec=GREEN, ms=4.6, label='D-337'), Line2D([], [], ls='', marker='o', mfc='white', mec=MID, mew=0.9, ms=4.2, label='D-471')],
           loc='upper left', bbox_to_anchor=((xl - 0.05) / W, 1 - 0.27 / Hh), ncol=2, handletextpad=0.2, columnspacing=1.0, borderaxespad=0, borderpad=0)
# ---------------------------------------------------------------- b
x0 = 4.28
panel(fig, x0 - 0.05, 0.03, 'b', tr('By family of features', 'Por familia de características'))
FAM = [('brightness', tr('Brightness\n(R, G, B, V, L*)', 'Brillo\n(R, G, B, V, L*)'), 25), ('red_green_balance', tr('Red–green\nbalance', 'Balance\nrojo–verde'), 13),
       ('cielab_a_b', 'CIELAB\na*, b*', 10), ('hue_saturation', tr('Hue and\nsaturation', 'Tono y\nsaturación'), 15)]
RB = 0.46; bx = axes(fig, x0 + 0.84, yt, W - (x0 + 0.84) - 0.26, RB * 4); bx.set_xlim(0, 1.05); bx.set_xticks([0, 0.5, 1.0]); bx.set_ylim(RB * 4, 0); bx.set_yticks([]); bx.spines['left'].set_visible(False); bx.grid(axis='x')
for i, (k, lab, nf) in enumerate(FAM):
    yy = (i + 0.5) * RB
    bx.barh(yy - 0.085, B['family_mean_abs_sum'][k], height=0.15, color=GREEN, zorder=3); bx.barh(yy + 0.085, A['family_mean_abs_sum'][k], height=0.15, color='white', ec=MID, lw=0.8, zorder=3)
    c.text(x0 + 0.76, yt + yy, lab, ha='right', va='center', fontsize=FS_XS, linespacing=1.15)
bx.set_xlabel(tr('mean |sum of SHAP|', 'media de |suma de SHAP|'))
c.text(x0 - 0.05, yt + RB * 4 + 0.62, tr('$^{*}$Spearman correlation between the value\nof the feature and its SHAP value (D-471 /\nD-337): negative, a lower value pushes the\nprediction towards anaemia.',
                                       '$^{*}$Correlación de Spearman entre el\nvalor de la característica y su valor\nSHAP (D-471 / D-337): si es negativa,\nun valor más bajo empuja la predicción\nhacia la anemia.'),
       fontsize=FS_XS, color=GREY, va='top', linespacing=1.25)
save(fig, 'FigS10_shap_sets')
