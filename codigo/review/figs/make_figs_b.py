# Figures that depend on the review analyses: Fig 3 (three protocols), Fig 5 (SHAP), S3 (feature space), S4 (calibration), S5 (LOHO forest), S6 (decision curve)
import json, csv, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
OUT='/home/claude/figs_new/'; RV='/home/claude/exp/out/review/'
plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,
 'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#9a9a9a','xtick.color':'#52514e','ytick.color':'#52514e','savefig.bbox':'tight','figure.dpi':200})
C=['#2a78d6','#eb6834','#1baf7a','#eda100','#e87ba4','#008300','#6250d6','#e34948']; GREY='#52514e'
R=json.load(open(RV+'review_results.json'))
import os as _os
if _os.path.exists(RV+'review_results_b.json'): R.update(json.load(open(RV+'review_results_b.json')))
M={'LR (color)':'Logistic regression, colour','SVM-RBF (color)':'SVM-RBF, colour','RF (color)':'Random Forest, colour','XGBoost (color)':'XGBoost, colour'}
cls={r['model']:r for r in csv.DictReader(open('/home/claude/anemia/resultados/classic_results.csv'))}
cls['XGBoost (color)']=cls.get('XGBoost (color)') or cls['XGB (color)']
import re
def pci(s): a=re.findall(r'\d+\.\d+',s); return (float(a[0]),float(a[1]))

# ---------------- Fig 3: three protocols ----------------
P=R['protocols']
import os as _o
RS=json.load(open('/home/claude/exp/out/review_strict/review_strict.json')) if _o.path.exists('/home/claude/exp/out/review_strict/review_strict.json') else {}
GS=RS.get('grouped_strict',{})
GEOF='/home/claude/exp/out/review2/geo/review_geo.json'
GQ=(json.load(open(GEOF)).get('grouped_strict',{}) if _o.path.exists(GEOF) else {}); GQ={k:v for k,v in GQ.items() if 'naive_minus_grouped_strict' in v}
rows=[]
for k in M:
    rows.append((M[k],P[k]['naive708_AUC'],P[k]['naive708_ci'],P[k]['grouped708_AUC'],P[k]['grouped708_ci'],float(cls[k]['AUC']),pci(cls[k]['AUC_ci'])))
cnn={'EfficientNet-B0':(0.8553,(0.8272,0.8826),0.7341,(0.6925,0.7788)),'MobileNetV2':(0.8265,(0.7960,0.8549),0.6644,(0.6123,0.7124)),'MobileNetV3-Small':(0.6058,(0.5629,0.6446),0.5562,(0.5044,0.6064))}
GRF='/home/claude/exp/out/review/gpu_review.json'
_GR=json.load(open(GRF)) if _o.path.exists(GRF) else {}; GG=_GR.get('grouped',{}); GSC={m[:-8]:v for m,v in _GR.get('strict',{}).items() if m.endswith('_grouped') and 'naive_minus_grouped_strict' in v}
for k,(na,nci,ca,cci) in cnn.items():
    if k in GG: rows.append((k,na,nci,GG[k]['AUC'],tuple(GG[k]['ci']),ca,cci))
    else: rows.append((k,na,nci,None,None,ca,cci))
# classical models (sorted by leak, largest at the top) above the CNNs (sorted by leak when the grouped run exists, else by naive - dedup.)
cl=sorted([r for r in rows if r[0] in M.values()],key=lambda r: r[1]-r[3]); cn=sorted([r for r in rows if r[0] in cnn],key=lambda r: (r[1]-r[3]) if r[3] is not None else (r[1]-r[5]))
rows=cn+cl
fig,ax=plt.subplots(figsize=(6.3,5.0))
ax.axhline(len(cn)-0.5,color='#d9d8d3',lw=0.8,ls=(0,(3,3)),zorder=0)
for i,(lab,na,nci,ga,gci,ca,cci) in enumerate(rows):
    pass  # (range bar removed: four markers per row)
    ax.errorbar(na,i+0.24,xerr=[[na-nci[0]],[nci[1]-na]],fmt='o',color=C[1],ms=4.5,capsize=2,lw=0.9,zorder=3)
    if ga is not None: ax.errorbar(ga,i+0.12,xerr=[[ga-gci[0]],[gci[1]-ga]],fmt='D',color=C[3],ms=4,capsize=2,lw=0.9,zorder=3)
    kk=[k for k in M if M[k]==lab]
    if kk and kk[0] in GS:
        gs_=GS[kk[0]]['grouped_strict_AUC']; gsc=GS[kk[0]]['ci']
        ax.errorbar(gs_,i,xerr=[[gs_-gsc[0]],[gsc[1]-gs_]],fmt='D',mfc='white',mec=C[3],color=C[3],ms=4,capsize=2,lw=0.9,zorder=3)
    if kk and kk[0] in GQ:
        gq=GQ[kk[0]]['grouped_strict_AUC']; gqc=GQ[kk[0]]['ci']
        ax.errorbar(gq,i-0.12,xerr=[[gq-gqc[0]],[gqc[1]-gq]],fmt='^',mfc='white',mec='#9a5b00',color='#9a5b00',ms=4.5,capsize=2,lw=0.9,zorder=3)
    if lab in GSC:
        gs_=GSC[lab]['AUC']; gsc=GSC[lab]['ci']
        ax.errorbar(gs_,i,xerr=[[gs_-gsc[0]],[gsc[1]-gs_]],fmt='D',mfc='white',mec=C[3],color=C[3],ms=4,capsize=2,lw=0.9,zorder=3)
    ax.errorbar(ca,i-0.24,xerr=[[ca-cci[0]],[cci[1]-ca]],fmt='s',color=C[0],ms=4.5,capsize=2,lw=0.9,zorder=3)
    kk=[k for k in M if M[k]==lab]
    _st=(GS[kk[0]]['naive_minus_grouped_strict'] if kk and kk[0] in GS and 'naive_minus_grouped_strict' in GS[kk[0]] else (GSC[lab]['naive_minus_grouped_strict'] if lab in GSC else None))
    _gq=GQ[kk[0]]['naive_minus_grouped_strict'] if kk and kk[0] in GQ else None
    txt=(f'leak {na-ga:.3f}'+(f' / {_st:.3f}' if _st is not None else '')+(f' / {_gq:.3f}' if _gq is not None else '')) if ga is not None else f'naive $-$ dedup. {na-ca:+.2f}'
    ax.text(1.005,i,txt.replace('+',''),va='center',fontsize=6.5,color=GREY)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows]); ax.set_xlim(0.45,1.0); ax.set_xlabel('AUC (95 % bootstrap CI)')
ax.grid(axis='x',color='#eeeeea',lw=0.8,zorder=0)
h=[plt.Line2D([],[],marker='o',color=C[1],ls='',ms=5,label='Naive: 708 rows, random row-level folds'),
   plt.Line2D([],[],marker='D',color=C[3],ls='',ms=4.5,label='Grouped: same 708 rows, duplicate groups in one fold'),
   plt.Line2D([],[],marker='D',mfc='white',mec=C[3],color=C[3],ls='',ms=4.5,label='Grouped with the strict duplicate map (third audit stage'+('' if GSC else '; colour-feature models')+')'),
   *([plt.Line2D([],[],marker='^',mfc='white',mec='#9a5b00',color='#9a5b00',ls='',ms=5,label='Grouped with the geometric duplicate map (fourth audit stage; colour-feature models)')] if GQ else []),
   plt.Line2D([],[],marker='s',color=C[0],ls='',ms=5,label='Deduplicated: 471 unique images')]
ax.legend(handles=h,frameon=False,loc='upper center',bbox_to_anchor=(0.42,-0.15),ncol=1,handletextpad=0.4,borderaxespad=0,fontsize=6.8)
ax.text(0.455,len(rows)-0.55,'Colour-feature models',fontsize=6.5,color=GREY,va='center'); ax.text(0.455,len(cn)-0.85,'CNNs',fontsize=6.5,color=GREY,va='center')
ax.set_xlim(0.45,1.16); ax.set_xticks(np.arange(0.5,1.01,0.1))
plt.savefig(OUT+'Fig3_leakage.pdf'); plt.savefig(OUT+'Fig3_leakage.png'); plt.close()

# ---------------- Fig 5: SHAP beeswarm (XGBoost) ----------------
z=np.load(RV+'shap_xgb.npz',allow_pickle=True); sv=z['sv']; X=z['X']; cols=list(z['cols'])
CH={'R':'R','G':'G','B':'B','H':'hue','S':'saturation','V':'V','L':'L*','a':'a*','b':'b*','EI':'EI','rg':'rg','gg':'gg'}
ST={'mean':'mean','std':'SD','p10':'P10','p50':'P50','p90':'P90'}
def lab(f):
    if f=='a_over_L': return 'a*/L*'
    if f=='R_over_G': return 'R/G'
    if f=='frac_high_a': return 'fraction a* > 20'
    c,s=f.rsplit('_',1); return f'{CH[c]} {ST[s]}'
imp=np.abs(sv).mean(0); top=np.argsort(-imp)[:12][::-1]
fig,ax=plt.subplots(figsize=(5.4,4.0)); rng=np.random.default_rng(0)
cmap=plt.get_cmap('coolwarm')
for i,j in enumerate(top):
    v=sv[:,j]; x=X[:,j]; r=(np.argsort(np.argsort(x))/(len(x)-1))
    # simple beeswarm: jitter proportional to local density
    order=np.argsort(v); yj=np.zeros(len(v)); bins=np.linspace(v.min(),v.max(),40); idx=np.digitize(v,bins)
    for b in np.unique(idx):
        m=idx==b; n=m.sum(); yj[m]=(np.arange(n)-(n-1)/2)*min(0.36/max(n,1)*2,0.08)
    ax.scatter(v,i+yj,c=cmap(r),s=5,alpha=0.8,lw=0)
ax.set_yticks(range(len(top))); ax.set_yticklabels([lab(cols[j]) for j in top]); ax.axvline(0,color='#9a9a9a',lw=0.8)
ax.set_xlabel('SHAP value (impact on the log-odds of anaemia)')
sm=plt.cm.ScalarMappable(cmap=cmap); sm.set_array([0,1]); cb=plt.colorbar(sm,ax=ax,ticks=[0,1],pad=0.10,aspect=30,shrink=0.9); cb.ax.set_yticklabels(['low','high']); cb.set_label('feature value',fontsize=7)
for i,j in enumerate(top): ax.text(ax.get_xlim()[1]*0.98 if False else sv[:,top].max()*1.02,i,f'{imp[j]:.3f}',fontsize=6,color=GREY,va='center')
ax.text(sv[:,top].max()*1.02,len(top)-0.4,'mean |SHAP|',fontsize=6,color=GREY,va='bottom')
plt.savefig(OUT+'Fig5_shap.pdf'); plt.savefig(OUT+'Fig5_shap.png'); plt.close()
print('shap top', [cols[j] for j in top[::-1]])

# ---------------- S3: feature-space audit ----------------
def readcsv(p):
    rows=list(csv.reader(open(p))); h=rows[0]; A=np.array([[float(x) for x in r] for r in rows[1:]]); return h,A
h,Xd=readcsv('/home/claude/exp/out/feat_dedup.csv'); h2,Xa=readcsv('/home/claude/exp/out/feat_all.csv')
mu=Xd.mean(0); sd=Xd.std(0); sd[sd==0]=1; Zd=(Xd-mu)/sd; Za=(Xa-mu)/sd
D2=((Zd[:,None,:]-Zd[None,:,:])**2).sum(-1); np.fill_diagonal(D2,np.inf); nnd=np.sqrt(D2.min(1)); nni=D2.argmin(1)
meta=list(csv.DictReader(open('/home/claude/sr/meta_groups.csv')))
# reproduce load_meta(): drop label-conflicting groups
from collections import defaultdict
g2l=defaultdict(set)
for r in meta: g2l[r['grp']].add(r['anemia'])
meta=[r for r in meta if len(g2l[r['grp']])==1]; assert len(meta)==len(Xa)==708
grp=defaultdict(list)
for i,r in enumerate(meta): grp[r['grp']].append(i)
ref=[]
for g,idx in grp.items():
    shas=[meta[i]['sha'] for i in idx]
    if len(set(shas))>1:
        for a in range(len(idx)):
            for b in range(a+1,len(idx)):
                if shas[a]!=shas[b]: ref.append(float(np.linalg.norm(Za[idx[a]]-Za[idx[b]])))
# dedup ids in the order of feat_dedup (sorted by image_id, first per grp)
dd=sorted(meta,key=lambda r:r['image_id']); seen=set(); ddl=[]
for r in dd:
    if r['grp'] not in seen: seen.add(r['grp']); ddl.append(r)
assert len(ddl)==len(Xd)==471
# strict set: representatives of the strict map (same rows as feat_all), standardised in their own space
sm=list(csv.DictReader(open('/home/claude/sr/meta_groups_strict.csv'))); sg2l=defaultdict(set)
for r in sm: sg2l[r['grp']].add(r['anemia'])
sm=[r for r in sm if len(sg2l[r['grp']])==1]; assert [r['image_id'] for r in sm]==[r['image_id'] for r in meta]
seen=set(); sidx=[]
for i in sorted(range(len(sm)),key=lambda i: sm[i]['image_id']):
    if sm[i]['grp'] not in seen: seen.add(sm[i]['grp']); sidx.append(i)
assert len(sidx)==452
Xs=Xa[sidx]; Zs=(Xs-Xs.mean(0))/np.where(Xs.std(0)==0,1,Xs.std(0)); DS=((Zs[:,None,:]-Zs[None,:,:])**2).sum(-1); np.fill_diagonal(DS,np.inf); nnd_s=np.sqrt(DS.min(1))
import json as _json
SP={(p['a'],p['b']):p for p in _json.load(open('/home/claude/exp/out/review/strict_pairs.json'))}
def tr_of(a,b):
    p=SP.get((a,b)) or SP.get((b,a)); return p['transform'].replace('rot','rot ').replace('mirror+','mirror + ') if p else ''
fig=plt.figure(figsize=(6.3,4.6)); gs=fig.add_gridspec(2,1,height_ratios=[1.1,1.3],hspace=0.45)
ax=fig.add_subplot(gs[0])
bins=np.linspace(0,max(nnd.max(),max(ref))*1.05,45)
ax.hist(nnd,bins=bins,color=C[0],alpha=0.85,label=f'nearest neighbour, pre-specified set of 471 images (median {np.median(nnd):.2f}; {int((nnd<1e-3).sum())} images at distance 0)')
ax.hist(nnd_s,bins=bins,histtype='step',color=C[2],lw=1.4,label=f'nearest neighbour, strict set of 452 images (minimum {nnd_s.min():.2f})')
ax.hist(ref,bins=bins,color=C[1],alpha=0.75,label=f'pHash-linked non-identical copies, {len(ref)} pairs (median {np.median(ref):.2f})')
ax.set_xlabel('Euclidean distance in standardised 63-feature space'); ax.set_ylabel('count'); ax.legend(frameon=False,loc='upper right',fontsize=6.5); ax.set_title('(a)',loc='left',fontsize=8)
closest=[]; seenp=set()
for i in np.argsort(nnd):
    key=frozenset((int(i),int(nni[i])))
    if key in seenp: continue
    seenp.add(key); closest.append(int(i))
    if len(closest)==6: break
sub=gs[1].subgridspec(2,6,wspace=0.05,hspace=0.25)
for k,i in enumerate(closest):
    for r,ii in enumerate([i,nni[i]]):
        a=fig.add_subplot(sub[r,k]); im=Image.open('/home/claude/data/cp-anemic/'+ddl[ii]['image_path']).convert('RGB'); a.imshow(im); a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values(): s.set_visible(False)
        a.set_title(f"{ddl[ii]['image_id'].replace('Image_','#')}, Hb {float(ddl[ii]['hb']):.1f}"+(f"\n{tr_of(ddl[i]['image_id'],ddl[nni[i]]['image_id'])}" if r==1 else ''),fontsize=6,pad=2)
        if k==0: a.set_ylabel('image A' if r==0 else 'nearest\nimage B',fontsize=6,rotation=0,ha='right',va='center',labelpad=4)
    if True: fig.text(0,0,'')
pos=fig.axes[1].get_position(); fig.text(0.02,pos.y1+0.05,'(b)  closest pairs in feature space (Hb in g/dL)',fontsize=8)
plt.savefig(OUT+'figS3_feature_space.pdf'); plt.savefig(OUT+'figS3_feature_space.png'); plt.close()
print('S3 median nnd',np.median(nnd),'ref median',np.median(ref),'n below max ref',(nnd<=max(ref)).sum())

# ---------------- S4 calibration, S6 decision curve ----------------
oof=json.load(open(RV+'oof_review.json')); y=np.array(json.load(open('/home/claude/exp/out/classic_oof.json'))['y'])
def wilson(k,n,z=1.96):
    if n==0: return (np.nan,np.nan)
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return c-h,c+h
fig,axs=plt.subplots(1,2,figsize=(6.3,3.1),sharey=True)
for ax,k,col in zip(axs,['RF (color)','XGBoost (color)'],[C[0],C[1]]):
    p=np.array(oof[k]['dedup471']); q=np.quantile(p,np.linspace(0,1,11)); q[0]-=1e-9
    for lo,hi in zip(q[:-1],q[1:]):
        m=(p>lo)&(p<=hi); n=m.sum(); kk=y[m].sum(); w=wilson(kk,n); ax.errorbar(p[m].mean(),kk/n,yerr=[[kk/n-w[0]],[w[1]-kk/n]],fmt='o',color=col,ms=4,capsize=2,lw=0.9)
    ax.plot([0,1],[0,1],ls=':',color='#9a9a9a'); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_xlabel('predicted probability'); ax.set_aspect('equal')
    e=R['dedup'][k]; ax.set_title(f"{M[k].replace(', colour','')}: slope {e['cal_slope']:.2f}, CITL {e['cal_in_large']:+.2f}".replace('+',''),fontsize=8,loc='left')
axs[0].set_ylabel('observed fraction anaemic')
plt.savefig(OUT+'figS4_calibration.pdf'); plt.savefig(OUT+'figS4_calibration.png'); plt.close()
fig,ax=plt.subplots(figsize=(4.8,3.3)); pts=np.linspace(0.02,0.95,94); n=len(y); prev=y.mean()
for k,col in [('RF (color)',C[0]),('XGBoost (color)',C[1])]:
    p=np.array(oof[k]['dedup471']); nb=[((p>=t)&(y==1)).sum()/n-((p>=t)&(y==0)).sum()/n*t/(1-t) for t in pts]; ax.plot(pts,nb,color=col,lw=1.8,label=M[k])
ax.plot(pts,[prev-(1-prev)*t/(1-t) for t in pts],color='#9a9a9a',lw=1.2,label='treat all'); ax.axhline(0,color='#9a9a9a',lw=1.2,ls='--',label='treat none')
ax.set_ylim(-0.05,0.55); ax.set_xlabel('threshold probability'); ax.set_ylabel('net benefit'); ax.legend(frameon=False)
plt.savefig(OUT+'figS6_decision_curve.pdf'); plt.savefig(OUT+'figS6_decision_curve.png'); plt.close()

# ---------------- S5 LOHO forest ----------------
L=R['loho']; per=sorted(L['per_centre'],key=lambda x:x['AUC'])
fig,ax=plt.subplots(figsize=(5.6,3.2))
for i,x in enumerate(per):
    ax.errorbar(x['AUC'],i,xerr=[[x['AUC']-x['ci'][0]],[x['ci'][1]-x['AUC']]],fmt='s',color=C[0],ms=4+6*np.sqrt(x['n']/100),capsize=2,lw=0.9)
ax.errorbar(L['pooled_AUC'],len(per),xerr=[[L['pooled_AUC']-L['pooled_ci'][0]],[L['pooled_ci'][1]-L['pooled_AUC']]],fmt='o',color=C[1],ms=5,capsize=2,lw=0.9)
RM=json.load(open('/home/claude/exp/out/review2/review2_stats.json'))['loho_meta']
yk=len(per)+1; ax.plot(RM['pred_int'],[yk,yk],color=C[2],lw=5,alpha=0.25,solid_capstyle='butt')
ax.errorbar(RM['mu'],yk,xerr=[[RM['mu']-RM['ci_hk'][0]],[RM['ci_hk'][1]-RM['mu']]],fmt='D',color=C[2],ms=5,capsize=2,lw=0.9)
ax.set_yticks(range(len(per)+2)); ax.set_yticklabels([f"{x['hospital']} (n = {x['n']})" for x in per]+['Pooled out-of-facility predictions',f"Random effects, REML + Hartung-Knapp\n(k = {RM['k']}; shaded: 95 % prediction interval)"])
ax.axvline(0.5,ls=':',color='#9a9a9a'); ax.set_xlabel('AUC (95 % CI), Random Forest trained on the other facilities'); ax.set_xlim(0.4,1.0); ax.grid(axis='x',color='#eeeeea',lw=0.8)
plt.savefig(OUT+'figS5_loho_forest.pdf'); plt.savefig(OUT+'figS5_loho_forest.png'); plt.close()
print('done')
