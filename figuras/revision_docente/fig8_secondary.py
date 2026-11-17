# Figure 8. Secondary analyses on D-337 (the 337 distinct images of the geometric map), conditional on the labels.
# (a) single-run AUC of every model on D-337 and, for comparison, on D-471; (b) tissue occlusion; (c) Grad-CAM in-mask ratio.
# Inputs: resultados/review6/geo_primary.json (models on D-337) and resultados/review7/gradcam_geo_*.json (12 runs:
# EfficientNet-B0 and MobileNetV2, common and tuned recipes, seeds 42, 1 and 2, folds of D-337), review7_shape.json.
import json, glob, sys, numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from figstyle import *
REPO = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '')
G = json.load(open(REPO + 'resultados/review6/geo_primary.json'))
SH = json.load(open(REPO + 'resultados/review7/review7_shape.json'))['D-337']
runs = [json.load(open(f)) for f in sorted(glob.glob(REPO + 'resultados/review7/gradcam_geo_*.json'))]; assert len(runs) == 12
# single-run AUC on D-471 (mean, SD): Supplementary Tables S17a and S13
D471 = {'RF (color)': (0.857, 0.010), 'XGBoost (color)': (0.845, 0.010), 'SVM-RBF (color)': (0.795, 0.011), 'LR (color)': (0.672, 0.015),
        'EfficientNetV2-S|geo': (0.765, 0.028), 'EfficientNet-B0|geo': (0.732, 0.014), 'ResNet50|geo': (0.730, 0.064), 'MobileNetV2|geo': (0.653, 0.032), 'MobileNetV3-Small|geo': (0.536, 0.025),
        'EfficientNet-B0|tunedgeo': (0.794, 0.006), 'MobileNetV2|tunedgeo': (0.776, 0.007), 'MobileNetV3-Large|tunedgeo': (0.749, 0.039)}
groups = [(tr('Colour features', 'Características de color'), GREEN, [('RF (color)', 'Random Forest'), ('XGBoost (color)', 'XGBoost'), ('SVM-RBF (color)', 'SVM-RBF'), ('LR (color)', tr('Logistic regression', 'Regresión logística'))]),
          (tr('CNNs, common recipe', 'CNN, receta común'), PURPLE, [(k + '|geo', k) for k in ('EfficientNetV2-S', 'EfficientNet-B0', 'ResNet50', 'MobileNetV2', 'MobileNetV3-Small')]),
          (tr('CNNs, tuned recipe', 'CNN, receta ajustada'), PURPLE, [(k + '|tunedgeo', k) for k in ('EfficientNet-B0', 'MobileNetV2', 'MobileNetV3-Large')])]
def geo(k):
    if k in G['classical_geo']: e = G['classical_geo'][k]; return e['AUC_rep_mean'], e['AUC_rep_sd']
    e = G['cnn_geo'][k]['summary']; return e['AUC_mean'], e['AUC_sd']
assert abs(geo('RF (color)')[0] - 0.741) < 6e-4 and abs(geo('EfficientNet-B0|tunedgeo')[0] - 0.678) < 6e-4 and abs(geo('EfficientNetV2-S|geo')[0] - 0.620) < 6e-4

Hh = 4.50; fig = figure(Hh); c = canvas(fig)
# ---------------------------------------------------------------- a
panel(fig, 0.0, 0.03, 'a', tr('AUC of single runs (mean ± SD)', 'AUC de corridas únicas (media ± DE)'))
RH = 0.232; gap = 0.26; xl, xr, yt = 1.30, 3.05, 0.62
n = sum(len(g[2]) for g in groups); ax = axes(fig, xl, yt, xr - xl, RH * n + gap * 2)
ax.set_xlim(0.48, 0.90); ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9]); ax.set_ylim(RH * n + gap * 2, 0); ax.set_yticks([]); ax.spines['left'].set_visible(False); ax.grid(axis='x')
ax.axvline(0.5, color=MID, lw=0.7, ls=(0, (3, 2))); ax.set_xlabel('AUC')
y = 0.0
for gi, (gname, col, items) in enumerate(groups):
    c.text(0.02, yt + y - 0.075, gname, fontsize=FS_XS, fontweight='bold', va='center_baseline')
    for k, lab in items:
        yy = y + RH / 2; m, sd = geo(k); m1, s1 = D471[k]
        ax.plot([m1 - s1, m1 + s1], [yy, yy], color=LIGHT, lw=0.9, solid_capstyle='butt', zorder=2); ax.plot([m1], [yy], ls='', marker='o', mfc='white', mec=MID, mew=0.9, ms=4.2, zorder=3)
        ax.plot([m - sd, m + sd], [yy, yy], color=col, lw=1.1, solid_capstyle='butt', zorder=4); ax.plot([m], [yy], ls='', marker='o', mfc=col, mec=col, ms=4.6, zorder=5)
        c.text(xl - 0.08, yt + yy, lab, ha='right', va='center_baseline', fontsize=FS_S)
        y += RH
    y += gap
fig.legend(handles=[Line2D([], [], ls='', marker='o', mfc=INK, mec=INK, ms=4.6, label='D-337'), Line2D([], [], ls='', marker='o', mfc='white', mec=MID, mew=0.9, ms=4.2, label='D-471')],
           loc='upper left', bbox_to_anchor=((xl - 0.05) / W, 1 - 0.27 / Hh), ncol=2, handletextpad=0.2, columnspacing=1.0, borderaxespad=0, borderpad=0)

# ---------------------------------------------------------------- b, c
cfg = [('EfficientNet-B0', 'ajustada'), ('EfficientNet-B0', 'comun'), ('MobileNetV2', 'ajustada'), ('MobileNetV2', 'comun')]
RN = {'ajustada': tr('tuned', 'ajustada'), 'comun': tr('common', 'común')}
def sel(mo, re_): return [r for r in runs if r['model'] == mo and r['recipe'] == re_]
x0 = 3.42; xl2, xr2 = 4.58, 6.20
panel(fig, x0, 0.03, 'b', tr('AUC with the tissue occluded', 'AUC con el tejido ocluido'))
RH2 = 0.29; yb = 0.90; b = axes(fig, xl2, yb, xr2 - xl2, RH2 * 4)
b.set_xlim(0.46, 0.74); b.set_xticks([0.5, 0.6, 0.7]); b.set_ylim(RH2 * 4, 0); b.set_yticks([]); b.spines['left'].set_visible(False); b.grid(axis='x')
b.axvspan(SH['shape_only_RF_ci'][0], SH['shape_only_RF_ci'][1], color=PALE, lw=0, zorder=0); b.axvline(SH['shape_only_RF_AUC'], color=MID, lw=0.8, zorder=1); b.axvline(0.5, color=MID, lw=0.7, ls=(0, (3, 2)))
ST = [('AUC_oof_fp32', 'o', PURPLE, PURPLE, tr('intact image', 'imagen intacta')), ('AUC_silhouette', 's', 'white', CHAR, tr('silhouette only', 'solo la silueta')), ('AUC_meanfill', '^', 'white', CHAR, tr('silhouette and mean colour', 'silueta y color medio'))]
for i, (mo, re_) in enumerate(cfg):
    rr = sel(mo, re_); assert len(rr) == 3
    for j, (key, mk, mfc, mec, lab) in enumerate(ST):
        v = np.array([r[key] for r in rr]); yy = (i + 0.5) * RH2 + (j - 1) * 0.085
        b.plot([v.min(), v.max()], [yy, yy], color=mec, lw=0.8, solid_capstyle='butt', zorder=3); b.plot([v.mean()], [yy], ls='', marker=mk, mfc=mfc, mec=mec, mew=0.9, ms=4.4, zorder=4)
    c.text(xl2 - 0.08, yb + (i + 0.5) * RH2 - 0.055, mo, ha='right', va='center_baseline', fontsize=FS_XS); c.text(xl2 - 0.08, yb + (i + 0.5) * RH2 + 0.065, RN[re_], ha='right', va='center_baseline', fontsize=FS_XS, color=GREY)
b.set_xlabel(tr('AUC (mean and range, 3 seeds)', 'AUC (media y rango, 3 semillas)'), labelpad=2)
b.text(SH['shape_only_RF_AUC'], -0.035, tr('mask shape alone (95% CI)', 'solo la forma de la máscara (IC 95 %)'), fontsize=FS_XS - 0.5, color=GREY, va='bottom', ha='center', clip_on=False)
fig.legend(handles=[Line2D([], [], ls='', marker=mk, mfc=mfc, mec=mec, mew=0.9, ms=4.4, label=lab) for key, mk, mfc, mec, lab in ST], loc='upper left',
           bbox_to_anchor=((x0 + 0.17) / W, 1 - 0.27 / Hh), ncol=2, handletextpad=0.1, columnspacing=0.8, labelspacing=0.25, borderaxespad=0, borderpad=0, fontsize=FS_XS)

yc = 2.62
panel(fig, x0, yc, 'c', tr('Grad-CAM attention inside the mask', 'Atención de Grad-CAM dentro de la máscara'))
d = axes(fig, xl2, yc + 0.34, xr2 - xl2, RH2 * 4)
d.set_xlim(0, 7.4); d.set_xticks([1, 2, 3, 4, 5, 6, 7]); d.set_ylim(RH2 * 4, 0); d.set_yticks([]); d.spines['left'].set_visible(False); d.grid(axis='x'); d.axvline(1, color=MID, lw=0.7, ls=(0, (3, 2)))
for i, (mo, re_) in enumerate(cfg):
    rr = sel(mo, re_); yy = (i + 0.5) * RH2
    d.add_patch(Rectangle((rr[0]['ceiling_p10'], yy - 0.075), rr[0]['ceiling_p90'] - rr[0]['ceiling_p10'], 0.15, fc=LIGHT, ec='none', zorder=2)); d.plot([rr[0]['ceiling_median']] * 2, [yy - 0.075, yy + 0.075], color=GREY, lw=1.0, zorder=3)
    v = np.array([r['predicted_class']['ratio_mean'] for r in rr]); d.plot([v.min(), v.max()], [yy, yy], color=PURPLE, lw=0.8, zorder=4); d.plot([v.mean()], [yy], ls='', marker='o', mfc=PURPLE, mec=PURPLE, ms=4.4, zorder=5)
    c.text(xl2 - 0.08, yc + 0.34 + yy - 0.055, mo, ha='right', va='center_baseline', fontsize=FS_XS); c.text(xl2 - 0.08, yc + 0.34 + yy + 0.065, RN[re_], ha='right', va='center_baseline', fontsize=FS_XS, color=GREY)
d.set_xlabel(tr('Attention ratio (1 = chance)', 'Razón de atención (1 = azar)'), labelpad=2)
r0 = sel(*cfg[0])[0]; d.text(r0['ceiling_p90'] + 0.15, 0.5 * RH2, tr('ceiling', 'techo'), fontsize=FS_XS - 0.5, color=GREY, ha='left', va='center_baseline')
save(fig, 'Fig8_secondary')
