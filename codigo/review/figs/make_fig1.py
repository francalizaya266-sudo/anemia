# Fig. 1 (study overview), redrawn as a vertical flow with wide boxes. Box heights are computed from the measured
# text, and every text block is checked to lie inside its box before saving.
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
OUT = '/home/claude/figs_new/'
plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.family': 'DejaVu Sans', 'figure.dpi': 200})
GREY = '#52514e'; INK = '#1f1e1d'; W = 6.3; PAD = 0.11; GAP = 0.24
fig = plt.figure(figsize=(W, 9)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, 9); ax.axis('off')
fig.canvas.draw(); rend = fig.canvas.get_renderer(); inv = ax.transData.inverted()
def ext(t):
    b = inv.transform(t.get_window_extent(rend).get_points()); return b
checks = []
def box(x, ytop, w, title, body, fc):
    t1 = ax.text(x + PAD, ytop - PAD, title, ha='left', va='top', fontsize=8, fontweight='bold', color=INK)
    b1 = ext(t1); t2 = ax.text(x + PAD, b1[0, 1] - 0.07, body, ha='left', va='top', fontsize=7.2, color=INK, linespacing=1.4)
    b2 = ext(t2); h = ytop - b2[0, 1] + PAD
    ax.add_patch(FancyBboxPatch((x, ytop - h), w, h, boxstyle='round,pad=0,rounding_size=0.06', fc=fc, ec='#8a8a8a', lw=0.8, zorder=0))
    checks.append(((x, ytop - h, w, h), (t1, t2))); return ytop - h
def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=9, color=GREY, lw=0.9, shrinkA=0, shrinkB=0))
L = 0.12; FW = W - 0.24; y = 8.9
y = box(L, y, FW, 'Data: CP-AnemiC version 1',
        '710 rows: conjunctival regions of interest on a black background, photographed with a tablet camera\n'
        '(Samsung Galaxy Tab A7, ambient light) and segmented by the dataset authors; laboratory Hb, age, sex\n'
        'and facility for children aged 6-59 months at ten health facilities in Ghana', '#fdf6e3')
arrow(W / 2, y, W / 2, y - GAP); y -= GAP
y = box(L, y, FW, 'Duplicate audit',
        'Stage 1: SHA-256 file digests (498 distinct files)\n'
        'Stage 2: perceptual hash of each image and its mirror, Hamming distance of at most 20 bits\n'
        '            → 472 unique images; 471 for classification\n'
        'Stage 3 (post hoc): perceptual hash under the 8 rotations and reflections, feature identity\n'
        '            → 453 unique images; strict set of 452 for classification\n'
        'Stage 4 (post hoc): SIFT keypoints + RANSAC on every pair, photometric verification\n'
        '            → 338 distinct images (images of the same eye merged); geometric set of 337\n'
        'Permutation tests: are duplicates confined to one anaemia class? Composition by stage', '#fdf6e3')
arrow(W / 2, y, W / 2, y - GAP); y -= GAP
y = box(L, y, FW, 'Three cross-validation protocols on identical folds',
        'Naive: 708 rows, duplicates spread at random across folds, as in prior work\n'
        'Grouped (post hoc): the same 708 rows, every duplicate group kept inside one fold\n'
        'Deduplicated: 471 unique images (pre-specified); strict (452) and geometric (337) sets post hoc', '#fdf6e3')
cw = (FW - 0.24) / 2; xl = L; xr = L + cw + 0.24
arrow(W / 2 - 0.5, y, xl + cw / 2, y - GAP); arrow(W / 2 + 0.5, y, xr + cw / 2, y - GAP); y -= GAP
yl = box(xl, y, cw, 'Colour features',
         '63 statistics inside the mask (RGB, HSV,\nL*a*b*, erythema index, chromaticities)\n'
         'Logistic regression, SVM-RBF,\nRandom Forest, XGBoost\n(nested grid search)', '#e6f3ec')
yr = box(xr, y, cw, 'Images',
         'ROI cropped and letter-boxed to 224 × 224,\ngeometric augmentation only\n'
         'Seven ImageNet-pretrained CNNs fine-tuned\nwith one common recipe\nTuned recipe (mask bounding box, trainable\nBatchNorm, learning rate chosen per fold)\nfor EfficientNet-B0, MobileNetV2 and\nMobileNetV3-Large', '#eaf0fb')
# equalise the two column boxes
yb = min(yl, yr)
for (bx, by, bw, bh), ts in checks[-2:]:
    pass
for p in ax.patches[-2:]: p.set_y(yb); p.set_height(y - yb)
checks[-2] = ((xl, yb, cw, y - yb), checks[-2][1]); checks[-1] = ((xr, yb, cw, y - yb), checks[-1][1])
y = yb
arrow(xl + cw / 2, y, W / 2 - 0.5, y - GAP); arrow(xr + cw / 2, y, W / 2 + 0.5, y - GAP); y -= GAP
y = box(L, y, FW, 'Evaluation',
        'AUC with bootstrap CI; DeLong and paired-bootstrap tests (Holm); cluster-aware tests on the 708 rows\n'
        'Calibration; screening thresholds fixed inside the training folds; PPV and NPV at local prevalence\n'
        'Leave-one-hospital-out validation; sex and age subgroups\n'
        'Sensitivity analyses: singletons only, random representative, strict set, shape-only model,\n'
        'brightness ablation, feature stability, 1-NN memorisation check\n'
        'Explainability: SHAP for the colour models; Grad-CAM and tissue occlusion for the CNNs\n'
        'Supplement: ONNX export in FP32, FP16 and INT8 with size and CPU latency', '#ffffff')
fig.canvas.draw(); bad = []
for (x, yy, w, hh), ts in checks:
    for t in ts:
        b = ext(t)
        if not (b[0, 0] >= x + 0.05 and b[1, 0] <= x + w - 0.05 and b[0, 1] >= yy + 0.03 and b[1, 1] <= yy + hh - 0.03): bad.append((t.get_text()[:50], b.round(2).tolist(), (x, yy, w, hh)))
assert not bad, bad
ax.set_ylim(y - 0.08, 9); fig.set_size_inches(W, (9 - y + 0.08))
plt.savefig(OUT + 'Fig1_overview.pdf'); plt.savefig(OUT + 'Fig1_overview.png', dpi=200); print('ok height', round(9 - y + 0.08, 2))
