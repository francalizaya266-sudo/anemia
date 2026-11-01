# Fig 6 (second revision): Grad-CAM for the predicted class, re-plotted from the exact 7x7 grids saved by gradcam_review2.py
# (laptop GPU run), with the in-mask ratio distribution against the 7x7 ceiling. Examples are not hand-picked: for each model
# and each correctly classified class, the images at the 20th, 40th, 60th and 80th percentiles of the predicted-class ratio.
import sys, json, numpy as np, cv2, matplotlib
sys.path.insert(0, '/home/claude/exp')
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import pandas as pd
from common import read_rgb, letterbox, mask_of, load_meta, dedup
plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.size':7,'font.family':'DejaVu Sans','savefig.bbox':'tight','figure.dpi':200,
                     'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#9a9a9a','xtick.color':'#52514e','ytick.color':'#52514e'})
G = '/home/claude/exp/out/review2/gpu/'; OUT = '/home/claude/figs_new/'
d = dedup(load_meta()); path = dict(zip(d.image_id, d.image_path))
def tight(im, s=224):
    m = mask_of(im); ys, xs = np.where(m)
    return letterbox(im[ys.min():ys.max()+1, xs.min():xs.max()+1], s)
ceil = np.load('/home/claude/exp/out/review2/gradcam_ceiling.npy')
Z = {n: np.load(G + f'gradcam_review2_{n}.npz', allow_pickle=True) for n in ('EfficientNet-B0', 'MobileNetV2')}
S = json.load(open(G + 'gradcam_review2.json'))['summary']
fig = plt.figure(figsize=(6.3, 7.4))
outer = fig.add_gridspec(3, 1, height_ratios=[2.0, 2.0, 1.25], hspace=0.30)
left = [outer[0], outer[1]]
for pi, (name, z) in enumerate(Z.items()):
    y = z['y'].astype(int); p = z['p']; r = z['ratio_c']; ids = z['ids']; pred = (p >= 0.5).astype(int)
    sub = left[pi].subgridspec(2, 4, wspace=0.04, hspace=0.22)
    for row, cls in enumerate((1, 0)):
        idx = np.where((y == cls) & (pred == cls))[0]; o = idx[np.argsort(r[idx])]
        pick = [o[int(q * (len(o) - 1))] for q in (0.2, 0.4, 0.6, 0.8)]
        for c, i in enumerate(pick):
            im = tight(read_rgb(path[str(ids[i])])); m = mask_of(im)
            cam = cv2.resize(z['grid_c'][i].astype(np.float32), (224, 224)); cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            ax = fig.add_subplot(sub[row, c]); ax.imshow(im); ax.imshow(cam, cmap='inferno', alpha=0.45, vmin=0, vmax=1)
            ax.contour(m.astype(float), levels=[0.5], colors='white', linewidths=0.5)
            ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]
            ax.set_title(f"{'TP' if cls else 'TN'}  p = {p[i]:.2f}  ratio {r[i]:.1f}", fontsize=5.8, pad=1.5)
    fig.text(0.125, left[pi].get_position(fig).y1 + 0.025, f"({'ab'[pi]}) {name}: Grad-CAM of the predicted class; top row true positives, bottom row true negatives", fontsize=7)
ax = fig.add_subplot(outer[2])
data = [Z['EfficientNet-B0']['ratio_a'], Z['EfficientNet-B0']['ratio_c'], Z['MobileNetV2']['ratio_a'], Z['MobileNetV2']['ratio_c'], ceil]
labs = ['EfficientNet-B0, anaemia logit', 'EfficientNet-B0, predicted class', 'MobileNetV2, anaemia logit', 'MobileNetV2, predicted class', 'Ceiling: best possible 7 x 7 map']
cols = ['#9fc3ee', '#2a78d6', '#f6b89e', '#eb6834', '#b5b5b5']
data = data[::-1]; labs = labs[::-1]; cols = cols[::-1]
bp = ax.boxplot(data, orientation='horizontal', widths=0.6, patch_artist=True, showfliers=False, medianprops=dict(color='black', lw=1))
for b, c in zip(bp['boxes'], cols): b.set_facecolor(c); b.set_edgecolor('#52514e'); b.set_linewidth(0.6)
for k in ('whiskers', 'caps'): [l.set_color('#52514e') or l.set_linewidth(0.6) for l in bp[k]]
ax.axvline(1, ls=':', color='#7a7a7a', lw=0.9); ax.text(1.06, 5.62, 'chance (ratio 1)', fontsize=5.8, color='#7a7a7a', ha='left', va='center')
ax.set_yticks(range(1, 6)); ax.set_yticklabels(labs, fontsize=6.3); ax.set_xlabel('In-mask ratio: Grad-CAM mass inside the mask divided by the mask area', fontsize=6.5)
ax.set_xlim(0, 8.2); ax.set_ylim(0.4, 5.9); ax.set_title('(c) In-mask ratio over the 471 images (boxes: quartiles; whiskers: 1.5 IQR)', fontsize=7, loc='left')
for ext in ('pdf', 'png'): plt.savefig(OUT + 'Fig6_gradcam.' + ext)
print({n: (round(float(np.median(z['ratio_c'])), 2), round(float(np.median(z['ratio_a'])), 2)) for n, z in Z.items()}, 'ceiling median', round(float(np.median(ceil)), 2))
