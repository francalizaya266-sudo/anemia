import sys; sys.path.insert(0,'/home/claude/exp')
import json, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
O='/home/claude/exp/out/'; G='/home/claude/gpu_out/'; F='/mnt/project-files/anemia/figuras/'
import os; os.makedirs(F,exist_ok=True)
C=['#2a78d6','#eb6834','#1baf7a','#eda100','#e87ba4','#008300','#6250d6','#e34948']
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#9a9a9a','axes.labelcolor':'#0b0b0b','xtick.color':'#52514e','ytick.color':'#52514e','figure.dpi':150})

cl=json.load(open(O+'classic_oof.json')); y=np.array(cl['y'])
def gpu_oof(name):
    j=json.load(open(G+f'cnn_{name}_clean.json')); return np.array(j['y']), np.array(j['oof']['fp32'])

# ---- Figure 2: ROC curves with bootstrap band ----
def roc_band(y,p,B=500,seed=0):
    rng=np.random.default_rng(seed); grid=np.linspace(0,1,101); T=[]
    for _ in range(B):
        i=rng.integers(0,len(y),len(y));
        if len(np.unique(y[i]))<2: continue
        fpr,tpr,_=roc_curve(y[i],p[i]); T.append(np.interp(grid,fpr,tpr))
    T=np.array(T); return grid,np.percentile(T,2.5,0),np.percentile(T,97.5,0)
fig,ax=plt.subplots(figsize=(4.6,4.4))
series=[('Random Forest (colour)',y,np.array(cl['oof']['RF (color)'])),('XGBoost (colour)',y,np.array(cl['oof']['XGBoost (color)'])),
        ('EfficientNet-B0',*gpu_oof('EfficientNet-B0')),('MobileNetV2',*gpu_oof('MobileNetV2'))]
for k,(lab,yy,p) in enumerate(series):
    fpr,tpr,_=roc_curve(yy,p); a=roc_auc_score(yy,p)
    ax.plot(fpr,tpr,color=C[k],lw=2,label=f'{lab}  AUC {a:.2f}')
    g,lo,hi=roc_band(yy,p); ax.fill_between(g,lo,hi,color=C[k],alpha=0.10,lw=0)
ax.plot([0,1],[0,1],ls=':',color='#9a9a9a',lw=1)
ax.set_xlabel('1 − specificity'); ax.set_ylabel('Sensitivity'); ax.set_xlim(0,1); ax.set_ylim(0,1.0)
ax.legend(frameon=False,loc='lower right',fontsize=8); ax.set_title('ROC, deduplicated protocol (n = 471), 95 % bootstrap bands',fontsize=9,loc='left')
plt.tight_layout(); plt.savefig(F+'fig2_roc.png'); plt.savefig(F+'fig2_roc.pdf'); plt.close()

# ---- Figure 3: leakage inflation (dumbbell naive vs clean) ----
rows=[('Random Forest (colour)',0.871,0.951),('XGBoost (colour)',0.863,0.938),('SVM-RBF (colour)',0.808,0.882),('Logistic regression (colour)',0.679,0.716),
      ('EfficientNet-B0',0.734,0.855),('MobileNetV2',0.664,0.827),('MobileNetV3-Small',0.556,0.606)]
rows=sorted(rows,key=lambda r:r[1])
fig,ax=plt.subplots(figsize=(5.2,3.4))
for i,(lab,c,n) in enumerate(rows):
    ax.plot([c,n],[i,i],color='#c9c8c3',lw=2,zorder=1)
    ax.scatter([c],[i],color=C[0],s=40,zorder=2); ax.scatter([n],[i],color=C[1],s=40,zorder=2)
    ax.text(n+0.012,i,f'+{n-c:.2f}',va='center',fontsize=8,color='#52514e')
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows]); ax.set_xlim(0.5,1.03); ax.set_xlabel('AUC')
ax.scatter([],[],color=C[0],label='Deduplicated (471 unique images)'); ax.scatter([],[],color=C[1],label='Naive, with duplicates (708 rows)')
ax.legend(frameon=False,loc='upper left',fontsize=8); ax.set_title('Inflation of AUC by duplicate leakage',fontsize=9,loc='left'); ax.grid(axis='x',color='#eeeeea',lw=0.8)
plt.tight_layout(); plt.savefig(F+'fig3_leakage.png'); plt.savefig(F+'fig3_leakage.pdf'); plt.close()

# ---- Figure 4: model size vs AUC (edge feasibility) ----
g=pd.read_csv(O+'cnn_gpu_results.csv'); g=g[g.variant.isin(['fp32','fp16'])]
pts=[('XGBoost (colour)',0.40,0.863,6.1,'classical'),('Random Forest (colour)',3.84,0.871,29.8,'classical'),('SVM-RBF (colour)',0.16,0.808,6.3,'classical'),('Logistic regression (colour)',0.01,0.679,7.9,'classical')]
for _,r in g.iterrows(): pts.append((f"{r.model} {r.variant.upper()}",r.MB,r.AUC,r.lat_ms,'cnn'))
fig,ax=plt.subplots(figsize=(5.6,3.8))
for lab,mb,auc,lat,kind in pts:
    col=C[0] if kind=='classical' else C[1]; ax.scatter(mb,auc,s=18+lat*1.2,color=col,alpha=0.85,edgecolor='white',lw=0.8,zorder=2)
    off={'EfficientNet-B0 FP32':(-62,8),'EfficientNetV2-S FP32':(6,-10),'ResNet50 FP32':(6,4),'MobileNetV2 FP32':(6,4),'ShuffleNetV2 FP32':(6,6),'MobileNetV3-Small FP32':(6,-9),'MobileNetV3-Large FP32':(6,-9),'Random Forest':(6,3),'XGBoost':(6,3),'SVM-RBF':(6,3),'Logistic regression':(6,3)}
    k=lab.replace(' (colour)','')
    if k in off: ax.annotate(k,(mb,auc),xytext=off[k],textcoords='offset points',fontsize=7,color='#52514e')
ax.set_xscale('log'); ax.set_xlabel('Model size (MB, log scale)'); ax.set_ylabel('AUC (deduplicated protocol)'); ax.set_ylim(0.45,0.92)
ax.axhline(0.5,ls=':',color='#9a9a9a',lw=1); ax.grid(color='#eeeeea',lw=0.8,zorder=0)
ax.scatter([],[],color=C[0],label='Colour features + classical'); ax.scatter([],[],color=C[1],label='CNN (FP32 and FP16)')
ax.legend(frameon=False,loc='lower left',fontsize=8); ax.set_title('Accuracy vs. footprint; marker area ∝ latency (1 CPU thread)',fontsize=9,loc='left')
plt.tight_layout(); plt.savefig(F+'fig4_size_vs_auc.png'); plt.savefig(F+'fig4_size_vs_auc.pdf'); plt.close()

# ---- Figure 1: data flow (deduplication) ----
fig,ax=plt.subplots(figsize=(8.2,2.4)); ax.axis('off')
boxes=[(0.01,'CP-AnemiC\n710 rows\n424 anaemic / 286 non'),(0.26,'SHA-256 audit\n91 groups of\nbyte-identical files\n(303 rows)'),
       (0.51,'Perceptual hash\n(pHash-16 + mirror)\n+21 near-duplicate pairs\n472 unique images'),(0.76,'Analysis set\n471 images\n(1 label-conflict group out)\n229 anaemic / 242 non')]
for k,(x,t) in enumerate(boxes):
    ax.add_patch(plt.Rectangle((x,0.2),0.22,0.62,fc='#f3f6fb' if k<3 else '#e6f3ec',ec='#9a9a9a',lw=0.8))
    ax.text(x+0.11,0.51,t,ha='center',va='center',fontsize=7.5)
    if k<3: ax.annotate('',xy=(x+0.25,0.51),xytext=(x+0.22,0.51),arrowprops=dict(arrowstyle='->',color='#52514e',lw=1))
ax.text(0.5,0.03,'Duplicated rows carried different Hb, age and hospital; 81 % were labelled anaemic vs 42 % among unique rows.',ha='center',fontsize=7.5,color='#52514e')
plt.savefig(F+'fig1_dedup_flow.png',bbox_inches='tight'); plt.savefig(F+'fig1_dedup_flow.pdf',bbox_inches='tight'); plt.close()
print('figures written', os.listdir(F))
