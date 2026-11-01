# Figures that do not depend on the new review analyses: overview (Fig 1), dedup flow (Fig 2), ROC (Fig 4), size vs AUC (Fig 7)
import json, csv, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
OUT='/home/claude/figs_new/'
plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.size':8,'font.family':'DejaVu Sans','axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,
 'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#9a9a9a','xtick.color':'#52514e','ytick.color':'#52514e','savefig.bbox':'tight','figure.dpi':200})
C=['#2a78d6','#eb6834','#1baf7a','#eda100','#e87ba4','#008300','#6250d6','#e34948']
GREY='#52514e'

def box(ax,x,y,w,h,text,fc='#f3f6fb',ec='#8a8a8a',fs=7.2,bold=False,align='center'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.004,rounding_size=0.012',fc=fc,ec=ec,lw=0.8))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs,fontweight='bold' if bold else 'normal',linespacing=1.25)
def arrow(ax,x0,y0,x1,y1):
    ax.add_patch(FancyArrowPatch((x0,y0),(x1,y1),arrowstyle='-|>',mutation_scale=8,color=GREY,lw=0.9,shrinkA=0,shrinkB=0))

# ---------------- Fig 1: overview of the study pipeline ----------------
fig,ax=plt.subplots(figsize=(6.3,3.0)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
box(ax,0.01,0.62,0.17,0.30,'Conjunctiva photograph\n(tablet camera, ambient\nlight) and laboratory Hb\nCP-AnemiC v1, 710 rows',fc='#fdf6e3')
box(ax,0.21,0.62,0.17,0.30,'Duplicate audit\nSHA-256 + pHash (mirror):\n472 unique; feature-space check\n+ dihedral pHash: 452 strict',fc='#fdf6e3')
box(ax,0.41,0.62,0.17,0.30,'Three protocols\nnaive (708 rows)\ngrouped (708, groups kept\ntogether), deduplicated (471)',fc='#fdf6e3')
arrow(ax,0.18,0.77,0.21,0.77); arrow(ax,0.38,0.77,0.41,0.77)
box(ax,0.63,0.70,0.36,0.22,'Segmented ROI on black background\n(released by the dataset authors)',fc='#eeeeea')
arrow(ax,0.58,0.77,0.63,0.80)
box(ax,0.63,0.36,0.17,0.26,'63 colour statistics\nwithin the mask\n(RGB, HSV, L*a*b*, EI,\nchromaticities)',fc='#e6f3ec')
box(ax,0.82,0.36,0.17,0.26,'224 x 224 letter-boxed\nROI, geometric\naugmentation only',fc='#eaf0fb')
arrow(ax,0.715,0.70,0.715,0.62); arrow(ax,0.905,0.70,0.905,0.62)
box(ax,0.63,0.06,0.17,0.24,'LR, SVM-RBF,\nRandom Forest, XGBoost\n(nested grid search)\nSHAP',fc='#e6f3ec')
box(ax,0.82,0.06,0.17,0.24,'7 ImageNet-pretrained\nlightweight CNNs\n(fine-tuned)\nGrad-CAM',fc='#eaf0fb')
arrow(ax,0.715,0.36,0.715,0.30); arrow(ax,0.905,0.36,0.905,0.30)
box(ax,0.01,0.06,0.57,0.48,'Evaluation on identical folds\n\n'
    'AUC with bootstrap CI; DeLong and paired-bootstrap tests (Holm)\n'
    'per-repetition AUC; nested screening thresholds (sensitivity 0.90)\n'
    'calibration slope and intercept; PPV / NPV at 43 % and 70 % prevalence\n'
    'leave-one-hospital-out (per centre, random effects); sex and age subgroups\n'
    'sensitivity analyses: singletons only, random representative, shape-only,\n'
    'brightness-invariant features, 1-NN memorisation check\n'
    'ONNX export: FP32 / FP16 / INT8 size and single-thread CPU latency',fc='#ffffff',fs=6.8)
arrow(ax,0.63,0.18,0.58,0.18)
plt.close()  # superseded by make_fig1.py (vertical layout with fit checks)

# ---------------- Fig 2: duplicate-audit flow (vertical) ----------------
fig,ax=plt.subplots(figsize=(6.7,6.4)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
L=0.02; W=0.50; cx=L+W/2; FS=6.9
SH=0.16  # vertical shift to make room for the post hoc strict stage
_box=box
def box(ax,x,y,w,h,text,**kw): return _box(ax,x,y*(1-SH)+SH,w,h*(1-SH),text,**kw)
_arrow=arrow
def arrow(ax,x0,y0,x1,y1): return _arrow(ax,x0,y0*(1-SH)+SH,x1,y1*(1-SH)+SH)
box(ax,L,0.88,W,0.10,'CP-AnemiC version 1: 710 rows\n424 anaemic (60 %) / 286 non-anaemic',fc='#fdf6e3',fs=FS)
box(ax,L,0.70,W,0.12,'SHA-256 digests: 498 distinct files\n303 rows in 91 groups of byte-identical files',fc='#f3f6fb',fs=FS)
box(ax,L,0.52,W,0.12,'Perceptual hash (16 x 16 DCT, image and mirror)\nHamming distance <= 20 bits links\n26 further files into 13 groups',fc='#f3f6fb',fs=FS)
box(ax,L,0.34,W,0.12,'472 unique images (connected components)\n= 386 singleton rows\n+ 86 multi-row groups holding 324 rows',fc='#e6f3ec',fs=FS)
box(ax,L,0.16,W,0.12,'Classification set: 471 images\n229 anaemic (49 %) / 242 non-anaemic\n(one representative per group, first by file name)',fc='#e6f3ec',fs=FS)
box(ax,L,0.01,W,0.09,'Hb regression and singleton-only\nsensitivity analysis: 386 singleton images',fc='#e6f3ec',fs=FS)
for y0,y1 in [(0.88,0.82),(0.70,0.64),(0.52,0.46),(0.34,0.28),(0.16,0.10)]: arrow(ax,cx,y0,cx,y1)
R=0.58; RW=0.41; FS2=6.6
box(ax,R,0.72,RW,0.08,'Within byte-identical groups\nHb differs in 90 of 91 groups',fc='#ffffff',fs=FS2)
box(ax,R,0.53,RW,0.10,'Largest component: 11 rows, 11 Hb values,\n9 facilities (10 byte-identical copies\n+ 1 mirrored copy)',fc='#ffffff',fs=FS2)
box(ax,R,0.32,RW,0.16,'Class composition\nrows in duplicate groups: 263/324 anaemic (81 %)\nsingleton rows: 161/386 anaemic (42 %)\nwithin the 86 groups Hb differs in 85,\nage in 77, facility in 82, sex in 61',fc='#ffffff',fs=FS2)
box(ax,R,0.185,RW,0.07,'Excluded: 1 group (2 rows)\nwith conflicting anaemia labels',fc='#fbeaea',fs=FS2)
for yy in (0.76,0.58,0.40,0.22): arrow(ax,L+W,yy,R,yy)
# post hoc third stage (strict map), drawn in the reserved bottom band
box=_box; arrow=_arrow
box(ax,L,0.01,W,0.125,'Post hoc third stage: pHash under the 8 dihedral\ntransforms (rotations, mirror) + identical features\n111 further links, 19 pairs of groups merged\n453 unique = 370 singletons + 83 groups (340 rows)',fc='#fdf6e3',fs=FS2)
box(ax,R,0.01,RW,0.125,'Strict set (sensitivity analysis):\n452 images, 218 anaemic (48 %),\n234 non-anaemic;\nstrict singleton set: 370 images',fc='#e6f3ec',fs=FS2)
arrow(ax,L+W,0.0725,R,0.0725)
arrow(ax,cx,SH+0.01*(1-SH),cx,0.135)
plt.close()  # superseded by make_fig2.py (four-stage audit flow)

# ---------------- Fig 4: ROC curves ----------------
def roc(y,p):
    o=np.argsort(-p); y=y[o]; p=p[o]; P=y.sum(); N=len(y)-P
    tps=np.cumsum(y); fps=np.cumsum(1-y)
    # keep last index of tied scores
    keep=np.r_[np.diff(p)!=0, True]
    tpr=np.r_[0,tps[keep]/P]; fpr=np.r_[0,fps[keep]/N]; return fpr,tpr
def auc(y,p):
    fpr,tpr=roc(y,p); return np.trapz(tpr,fpr)
def band(y,p,B=500,seed=0):
    rng=np.random.default_rng(seed); g=np.linspace(0,1,101); T=[]
    for _ in range(B):
        i=rng.integers(0,len(y),len(y))
        if len(np.unique(y[i]))<2: continue
        f,t=roc(y[i],p[i]); T.append(np.interp(g,f,t))
    T=np.array(T); return g,np.percentile(T,2.5,0),np.percentile(T,97.5,0)
cl=json.load(open('/home/claude/exp/out/classic_oof.json')); y=np.array(cl['y'])
def gpu(name):
    j=json.load(open(f'/home/claude/gpu_out/cnn_{name}_clean.json')); return np.array(j['y']),np.array(j['oof']['fp32'])
series=[('Random Forest, colour features',y,np.array(cl['oof']['RF (color)']),'0.871 (0.837--0.903)'),
        ('XGBoost, colour features',y,np.array(cl['oof']['XGBoost (color)']),'0.863 (0.830--0.895)'),
        ('EfficientNet-B0 (seed 42)',*gpu('EfficientNet-B0'),'0.734 (0.693--0.779)'),('MobileNetV2 (seed 42)',*gpu('MobileNetV2'),'0.664 (0.612--0.712)')]
fig,ax=plt.subplots(figsize=(4.3,4.9))
for k,(lab,yy,p,ci) in enumerate(series):
    f,t=roc(yy,p); a=auc(yy,p); ax.plot(f,t,color=C[k],lw=1.8,label=f'{lab}: AUC {ci.replace("--","-")}')
    g,lo,hi=band(yy,p); ax.fill_between(g,lo,hi,color=C[k],alpha=0.10,lw=0)
ax.plot([0,1],[0,1],ls=':',color='#9a9a9a',lw=1); ax.text(0.62,0.57,'chance (AUC 0.5)',rotation=38,fontsize=6.5,color='#7a7a7a',ha='center')
ax.set_xlabel('1 - specificity (false-positive rate)'); ax.set_ylabel('Sensitivity (true-positive rate)'); ax.set_xlim(0,1); ax.set_ylim(0,1)
ax.set_xticks(np.linspace(0,1,6)); ax.set_yticks(np.linspace(0,1,6)); ax.set_aspect('equal')
ax.legend(frameon=False,loc='upper center',bbox_to_anchor=(0.5,-0.13),title='Model: AUC (95 % bootstrap CI)',title_fontsize=7,alignment='left')
plt.savefig(OUT+'Fig4_roc.pdf'); plt.savefig(OUT+'Fig4_roc.png'); plt.close()
print('AUCs', [round(auc(s[1],s[2]),3) for s in series])

# ---------------- Fig 7: AUC vs model size ----------------
rows=list(csv.DictReader(open('/home/claude/anemia/resultados/cnn_gpu_results.csv')))
pts=[]
for r in rows:
    if r['variant'] in ('fp32','fp16'): pts.append((r['model'],r['variant'],float(r['MB']),float(r['AUC']),float(r['lat_ms'])))
classical=[('XGBoost',0.40,0.863,6.1),('Random Forest',3.84,0.871,29.8),('SVM-RBF',0.16,0.808,6.3),('Logistic regression',0.01,0.679,7.9)]
fig,ax=plt.subplots(figsize=(6.3,4.6))
def s_of(lat): return 12+lat*1.6
for name,mb,a,lat in classical:
    ax.scatter(mb,a,s=s_of(lat),marker='s',color=C[0],alpha=0.9,edgecolor='white',lw=0.6,zorder=3)
for name,var,mb,a,lat in pts:
    if var=='fp32': ax.scatter(mb,a,s=s_of(lat),marker='o',color=C[1],alpha=0.9,edgecolor='white',lw=0.6,zorder=3)
    else: ax.scatter(mb,a,s=s_of(lat),marker='^',facecolor='none',edgecolor=C[1],lw=1.0,zorder=3)
lab_off={'XGBoost':(0.40,0.863,0.9,0.905),'Random Forest':(3.84,0.871,5.5,0.905),'SVM-RBF':(0.16,0.808,0.06,0.845),'Logistic regression':(0.01,0.679,0.012,0.72),
 'EfficientNet-B0':(15.3,0.734,3.0,0.775),'EfficientNetV2-S':(76.9,0.734,120,0.775),'ResNet50':(89.6,0.687,160,0.66),'MobileNetV2':(8.5,0.664,1.1,0.63),
 'ShuffleNetV2':(4.9,0.561,0.35,0.585),'MobileNetV3-Small':(3.6,0.556,0.5,0.52),'MobileNetV3-Large':(11.4,0.513,25,0.49)}
for k,(x,yv,tx,ty) in lab_off.items():
    ax.annotate(k,(x,yv),xytext=(tx,ty),fontsize=6.8,color=GREY,ha='center',va='center',arrowprops=dict(arrowstyle='-',color='#b5b5b5',lw=0.6,shrinkB=3))
ax.set_xscale('log'); ax.set_xlabel('Model file size (MB, log scale)'); ax.set_ylabel('AUC, deduplicated protocol (471 images)'); ax.set_ylim(0.44,0.93); ax.set_xlim(0.006,400)
ax.axhline(0.5,ls=':',color='#9a9a9a',lw=1); ax.text(380,0.505,'chance (AUC 0.5)',fontsize=6.5,color='#7a7a7a',va='bottom',ha='right')
ax.grid(color='#eeeeea',lw=0.8,zorder=0)
h1=ax.scatter([],[],marker='s',color=C[0],s=30,label='Colour features + classical model')
h2=ax.scatter([],[],marker='o',color=C[1],s=30,label='CNN, FP32')
h3=ax.scatter([],[],marker='^',facecolor='none',edgecolor=C[1],s=30,label='CNN, FP16')
leg1=ax.legend(handles=[h1,h2,h3],frameon=False,loc='upper left',bbox_to_anchor=(0.0,-0.14),title='Model type',title_fontsize=7,alignment='left'); ax.add_artist(leg1)
hs=[ax.scatter([],[],s=s_of(l),color='#9a9a9a',alpha=0.6,label=f'{l} ms') for l in (5,20,50)]
ax.legend(handles=hs,frameon=False,loc='upper right',bbox_to_anchor=(1.0,-0.14),title='Marker area: latency, one CPU thread',title_fontsize=7,ncol=3,columnspacing=1.6,borderpad=0.8,alignment='left')
leg2=ax.get_legend()
for ext in ('pdf','png'): plt.savefig(OUT+'Fig7_size_vs_auc.'+ext,bbox_extra_artists=[leg1,leg2])
plt.close()
print('done')
