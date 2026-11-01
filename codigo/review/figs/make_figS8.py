# Figures that do not depend on the new review analyses: overview (Fig 1), dedup flow (Fig 2), ROC (Fig 4), size vs AUC (Fig 7)
import json, csv, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
OUT='/home/claude/figs_new/'
plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.size':8,'font.family':'DejaVu Sans','axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,
 'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#9a9a9a','xtick.color':'#52514e','ytick.color':'#52514e','savefig.bbox':'tight','figure.dpi':200})
C=['#2a78d6','#eb6834','#1baf7a','#eda100','#e87ba4','#008300','#6250d6','#e34948']
GREY='#52514e'

# ---------------- Fig 7: AUC vs model size ----------------
rows=list(csv.DictReader(open('/home/claude/anemia/resultados/cnn_gpu_results.csv')))
pts=[]
for r in rows:
    if r['variant'] in ('fp32','fp16'): pts.append((r['model'],r['variant'],float(r['MB']),float(r['AUC']),float(r['lat_ms'])))
RR=json.load(open('/home/claude/anemia/resultados/review/review_results.json'))['dedup']
_rp={'XGBoost':'XGBoost (color)','Random Forest':'RF (color)','SVM-RBF':'SVM-RBF (color)','Logistic regression':'LR (color)'}
classical=[(n,mb,RR[_rp[n]]['AUC_rep_mean'],lat) for n,mb,_,lat in [('XGBoost',0.40,0.863,6.1),('Random Forest',3.84,0.871,29.8),('SVM-RBF',0.16,0.808,6.3),('Logistic regression',0.01,0.679,7.9)]]
print([(n,round(a,3)) for n,_,a,_ in classical])
_ca={n:a for n,_,a,_ in classical}
fig,ax=plt.subplots(figsize=(6.3,4.6))
def s_of(lat): return 12+lat*1.6
for name,mb,a,lat in classical:
    ax.scatter(mb,a,s=s_of(lat),marker='s',color=C[0],alpha=0.9,edgecolor='white',lw=0.6,zorder=3)
for name,var,mb,a,lat in pts:
    if var=='fp32': ax.scatter(mb,a,s=s_of(lat),marker='o',color=C[1],alpha=0.9,edgecolor='white',lw=0.6,zorder=3)
    else: ax.scatter(mb,a,s=s_of(lat),marker='^',facecolor='none',edgecolor=C[1],lw=1.0,zorder=3)
lab_off={'XGBoost':(0.40,_ca['XGBoost'],0.9,0.905),'Random Forest':(3.84,_ca['Random Forest'],5.5,0.905),'SVM-RBF':(0.16,_ca['SVM-RBF'],0.06,0.845),'Logistic regression':(0.01,_ca['Logistic regression'],0.012,0.72),
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
for ext in ('pdf','png'): plt.savefig(OUT+'figS8_size_vs_auc.'+ext,bbox_extra_artists=[leg1,leg2])
plt.close()
print('done')
