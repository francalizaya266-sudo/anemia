"""Grad-CAM de la CNN entrenada (ultimo pliegue) sobre su pliegue de prueba + metrica cuantitativa:
fraccion de la masa del mapa dentro de la conjuntiva frente al azar.  Uso: python gradcam_torch.py MobileNetV2"""
import sys, os, json, numpy as np, torch, torch.nn as nn, cv2
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from train_torch import Net, tight, OUT, S, DEV
name = sys.argv[1]
m = Net(name); m.load_state_dict(torch.load(os.path.join(OUT, f'{name}_lastfold.pt'), map_location='cpu')); m.eval()
feat = m.b.features if hasattr(m.b, 'features') else m.b.layer4   # ultimo bloque convolucional
acts = {}; grads = {}
feat.register_forward_hook(lambda mod, i, o: acts.__setitem__('a', o))
feat.register_full_backward_hook(lambda mod, gi, go: grads.__setitem__('g', go[0]))
def gradcam(x):
    xb = torch.from_numpy(x[None]).permute(0, 3, 1, 2).float().requires_grad_(True)
    out = m(xb); m.zero_grad(); out[0, 0].backward()
    a = acts['a'][0]; g = grads['g'][0]; w = g.mean((1, 2), keepdim=True)
    cam = torch.relu((a * w).sum(0)).detach().numpy(); cam = cv2.resize(cam, (S, S)); cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam, float(torch.sigmoid(out[0, 0]))
d = dedup(load_meta()); y = d.anemia.values; tr, te = folds(y, None, seed=SEED)[-1]
X = np.stack([tight(read_rgb(p), S) for p in d.image_path[te]]); yt = y[te]; hb = d.hb.values[te]
cams, ps = zip(*[gradcam(x) for x in X]); cams = np.array(cams); ps = np.array(ps)
masks = np.array([mask_of(x) for x in X]); inside = (cams * masks).sum((1, 2)) / (cams.sum((1, 2)) + 1e-8); area = masks.mean((1, 2))
res = {'model': name, 'n_test': int(len(te)), 'cam_mass_inside_mask_mean': float(inside.mean()), 'mask_area_mean': float(area.mean()), 'ratio_vs_chance': float((inside / area).mean())}
order = np.argsort(-np.abs(ps - 0.5))
tp = [i for i in order if yt[i] == 1 and ps[i] >= 0.5][:3]; tn = [i for i in order if yt[i] == 0 and ps[i] < 0.5][:3]; err = [i for i in order if (ps[i] >= 0.5) != yt[i]][:2]; sel = tp + tn + err
fig, ax = plt.subplots(2, len(sel), figsize=(2.2 * len(sel), 4.6))
for k, i in enumerate(sel):
    ax[0, k].imshow(X[i]); ax[0, k].axis('off'); ax[0, k].set_title(f"Hb {hb[i]:.1f} · y={yt[i]} · p={ps[i]:.2f}", fontsize=8)
    heat = cv2.applyColorMap((cams[i] * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
    ax[1, k].imshow((0.55 * X[i] + 0.45 * heat).astype(np.uint8)); ax[1, k].axis('off')
plt.suptitle(f'Grad-CAM · {name} · pliegue de prueba (no visto en entrenamiento)', fontsize=10); plt.tight_layout()
plt.savefig(os.path.join(OUT, f'gradcam_{name}.png'), dpi=150); json.dump(res, open(os.path.join(OUT, f'gradcam_{name}.json'), 'w'), indent=1)
print(json.dumps(res, indent=1))
