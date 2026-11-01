# Fig. 2 (duplicate audit flow), second revision: four stages, two columns (stage | what it showed); box heights from the
# measured text, with a check that every text block lies inside its box.
import json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
OUT = '/home/claude/figs_new/'
G = json.load(open('/home/claude/exp/out/review2/geo_map_summary.json')); C = json.load(open('/home/claude/exp/out/review2/review2_stats.json'))['class_conditional']
plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.family': 'DejaVu Sans', 'figure.dpi': 200})
GREY = '#52514e'; INK = '#1f1e1d'; W = 6.6; H = 9.5; PAD = 0.1; GAP = 0.26
fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')
fig.canvas.draw(); rend = fig.canvas.get_renderer(); inv = ax.transData.inverted()
def ext(t): return inv.transform(t.get_window_extent(rend).get_points())
checks = []
def box(x, ytop, w, title, body, fc, fs=7.0):
    t1 = ax.text(x + PAD, ytop - PAD, title, ha='left', va='top', fontsize=fs + 0.6, fontweight='bold', color=INK)
    b1 = ext(t1); t2 = ax.text(x + PAD, b1[0, 1] - 0.06, body, ha='left', va='top', fontsize=fs, color=INK, linespacing=1.38)
    h = ytop - ext(t2)[0, 1] + PAD
    p = FancyBboxPatch((x, ytop - h), w, h, boxstyle='round,pad=0,rounding_size=0.05', fc=fc, ec='#8a8a8a', lw=0.8, zorder=0); ax.add_patch(p)
    checks.append([p, (t1, t2)]); return ytop - h
def row(y, left, right, fcl, fcr='#ffffff'):
    yl = box(LX, y, LW, *left, fcl); yr = box(RX, y, RW, *right, fcr) if right else yl
    yb = min(yl, yr)
    for c in checks[-(2 if right else 1):]: c[0].set_y(yb); c[0].set_height(y - yb)
    if right: ax.add_patch(FancyArrowPatch((LX + LW, (y + yb) / 2), (RX, (y + yb) / 2), arrowstyle='-|>', mutation_scale=8, color=GREY, lw=0.8))
    return yb
def down(y): ax.add_patch(FancyArrowPatch((LX + LW / 2, y), (LX + LW / 2, y - GAP), arrowstyle='-|>', mutation_scale=9, color=GREY, lw=0.9)); return y - GAP
LX = 0.1; LW = 3.55; RX = 3.9; RW = W - RX - 0.1; y = H - 0.1
Y, B, Gn = '#fdf6e3', '#f3f6fb', '#e6f3ec'
y = row(y, ('CP-AnemiC version 1', '710 rows: 424 anaemic (60 %), 286 non-anaemic'), None, Y); y = down(y)
y = row(y, ('Stage 1: SHA-256 file digests', '498 distinct files\n303 rows in 91 groups of byte-identical files'),
        ('Metadata inside the groups', 'Hb differs in 90 of the 91 byte-identical groups;\nthe largest holds 10 copies with 9 Hb values'), B); y = down(y)
y = row(y, ('Stage 2: perceptual hash (image and mirror)', '256-bit pHash, Hamming distance of at most 20 bits\n26 further files linked, in 13 groups'),
        ('Largest connected component', '11 rows, 11 Hb values, 9 facilities\n(10 byte-identical copies + 1 mirrored copy)'), B); y = down(y)
y = row(y, ('472 unique images (pre-specified map)', '386 singleton rows + 86 multi-row groups (324 rows)\n1 group with conflicting labels excluded\nClassification set: 471 images (229 anaemic, 242 not)\nSingleton set for Hb regression: 386 images'),
        ('Class dependence', 'Duplicated rows 81 %% anaemic, singletons 42 %%\n%d of 86 groups carry one label;\npermutation null %.1f (maximum %d), p < 0.0001' % (85, C['null_all_rows_mean'], C['null_all_rows_max'])), Gn); y = down(y)
y = row(y, ('Stage 3 (post hoc): pHash under 8 rotations and\nreflections + identical colour features', '19 pairs of groups merged (rotated copies)\n453 unique images; strict set: 452 (218 anaemic, 234 not)'),
        ('Rotated copies', 'All 19 pairs agree on the anaemia label\nbut not on Hb'), Y); y = down(y)
y = row(y, ('Stage 4 (post hoc): SIFT keypoints + RANSAC on all\n111,156 pairs, photometric verification', '%d new pairs of images of the same eye\n(re-cropped, rescaled or mirrored)\n%d distinct images (%d singletons + %d groups)\nGeometric set: %d images (%d anaemic, %d not)' % (G['n_new_pairs'], G['geo_unique'], G['singletons'], G['multi'], G['n_class'], G['n_anaemic'], G['n_non'])),
        ('Images of the same eye', 'All %d pairs agree on the anaemia label\n(permutation null %.0f, p < 0.0001);\n%d differ in facility and %d in Hb' % (G['n_new_pairs'], G['null_mean'], G['hosp_differ'], G['hb_differ'])), Y)
fig.canvas.draw()
for p, ts in checks:
    x0, y0, w, h = p.get_x(), p.get_y(), p.get_width(), p.get_height()
    for t in ts:
        b = ext(t); assert b[0, 0] >= x0 - 1e-6 and b[1, 0] <= x0 + w + 1e-6 and b[0, 1] >= y0 - 1e-6 and b[1, 1] <= y0 + h + 1e-6, (t.get_text()[:40], b, (x0, y0, w, h))
used = H - y + 0.1; ax.set_ylim(H - used, H); fig.set_size_inches(W, used)
for ext_ in ('pdf', 'png'): fig.savefig(OUT + 'Fig2_dedup_flow.' + ext_)
print('ok height %.2f' % used)
