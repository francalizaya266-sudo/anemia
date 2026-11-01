# Interim Grad-CAM figure: re-assemble the existing panels (laptop GPU run) without internal titles, with legible row labels.
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.size':8,'font.family':'DejaVu Sans','savefig.bbox':'tight','figure.dpi':200})
def tiles(path):
    im=np.array(Image.open(path).convert('RGB')); dark=(im.sum(2)<600)  # tiles have black/blue backgrounds; page is white
    cols=np.where(dark.mean(0)>0.5)[0]; rows=np.where(dark.mean(1)>0.3)[0]
    def runs(idx):
        out=[]; s=idx[0]; p=idx[0]
        for i in idx[1:]:
            if i!=p+1: out.append((s,p)); s=i
            p=i
        out.append((s,p)); return out
    cr=runs(cols); rr=runs(rows)
    cr=[c for c in cr if c[1]-c[0]>100]
    c0,c1=cr[0]; blk=dark[:,c0:c1+1].mean(1); rows=np.where(blk>0.95)[0]; rr=[r for r in runs(rows) if r[1]-r[0]>100]
    side=c1-c0+1
    if len(rr)==1: r0,r1=rr[0]; rr=[(r0,r0+side-1),(r1-side+1,r1)]
    return im,cr,rr
fig=plt.figure(figsize=(6.5,4.2))
gs=fig.add_gridspec(2,1,hspace=0.18)
for pi,(name,lab) in enumerate([('EfficientNet-B0','(a) EfficientNet-B0'),('MobileNetV2','(b) MobileNetV2')]):
    im,cr,rr=tiles(f'/home/claude/anemia/resultados/gradcam_{name}.png')
    assert len(cr)==8 and len(rr)==2, (len(cr),len(rr))
    sub=gs[pi].subgridspec(2,8,wspace=0.04,hspace=0.04)
    for r,(r0,r1) in enumerate(rr):
        for c,(c0,c1) in enumerate(cr):
            ax=fig.add_subplot(sub[r,c]); ax.imshow(im[r0:r1+1,c0:c1+1]); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_visible(False)
            if c==0: ax.set_ylabel('ROI' if r==0 else 'Grad-CAM',fontsize=7)
            if r==0: ax.set_title(['TP','TP','TP','TN','TN','TN','error','error'][c],fontsize=7,pad=2)
    fig.text(0.125,0.90 if pi==0 else 0.47,lab,fontsize=8,ha='left',va='bottom')
plt.savefig('/home/claude/figs_new/Fig6_gradcam.pdf'); plt.savefig('/home/claude/figs_new/Fig6_gradcam.png'); print('ok')
