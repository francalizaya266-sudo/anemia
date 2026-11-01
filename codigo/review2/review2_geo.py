# Second internal review: sensitivity analysis on the GEOMETRIC set (hash stages + SIFT/RANSAC + photometric verification; 349 images).
# Post hoc stage added after the exhaustive geometric search found re-cropped, rescaled and mirrored copies missed by hashing.
import sys; sys.path.insert(0,'/home/claude/exp')
import numpy as np, pandas as pd, json, time, warnings, os; warnings.filterwarnings('ignore')
import common
from common import SEED, folds, metrics
from classic import MODELS
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedGroupKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from review_analyses_lib import thr_for_sens, auc_ci, paired_boot_diff, delong
OUT='/home/claude/exp/out/review2/geo/'; os.makedirs(OUT,exist_ok=True)
T0=time.time()
def log(*a): print('[%6.0fs]'%(time.time()-T0),*a,flush=True)
old=pd.read_csv('/home/claude/sr/meta_groups.csv'); s=pd.read_csv('/home/claude/anemia/codigo/meta_groups_geo.csv')
assert (old.image_id.values==s.image_id.values).all()
# drop label-conflicting group (same pair Image_188/Image_310 in both maps)
conf=s.groupby('grp').anemia.nunique(); s=s[~s.grp.isin(conf[conf>1].index)].reset_index(drop=True)
oc=old.groupby('grp').anemia.nunique(); old=old[~old.grp.isin(oc[oc>1].index)].reset_index(drop=True)
assert (old.image_id.values==s.image_id.values).all() and len(s)==708
Xa=pd.read_csv('/home/claude/exp/out/feat_all.csv'); assert len(Xa)==708
ya=s.anemia.values; gs_=s.grp.values
ss=s.sort_values('image_id').drop_duplicates('grp'); idx=ss.index.values; Xd=Xa.iloc[idx].reset_index(drop=True); dd=ss.reset_index(drop=True); yd=dd.anemia.values
assert len(dd)==json.load(open("/home/claude/exp/out/review2/geo_map_summary.json"))["n_class"], len(dd)
R=json.load(open(OUT+'review_geo.json')) if os.path.exists(OUT+'review_geo.json') else {}
R.update(n_strict=int(len(dd)),n_anaemic=int(yd.sum()),n_non=int((1-yd).sum()),images=dd.image_id.tolist())
oof=json.load(open(OUT+'oof_geo.json')) if os.path.exists(OUT+'oof_geo.json') else {}
def save(): json.dump(R,open(OUT+'review_geo.json','w'),indent=1,default=float)
def run(X,y,groups,est,grid,reps,inner_groups=False,nested_thr=False,seed0=SEED):
    P=np.zeros((reps,len(y))); T=np.full((reps,len(y)),np.nan)
    for r in range(reps):
        for tr,te in folds(y,groups,seed=seed0+r):
            if inner_groups:
                cv=StratifiedGroupKFold(5,shuffle=True,random_state=r); gs=GridSearchCV(est,grid,scoring='roc_auc',cv=cv,n_jobs=1).fit(X.iloc[tr],y[tr],groups=groups[tr])
            else:
                cv=StratifiedKFold(5,shuffle=True,random_state=r); gs=GridSearchCV(est,grid,scoring='roc_auc',cv=cv,n_jobs=1).fit(X.iloc[tr],y[tr])
            P[r,te]=gs.predict_proba(X.iloc[te])[:,1]
            if nested_thr:
                pi=cross_val_predict(gs.best_estimator_,X.iloc[tr],y[tr],cv=StratifiedKFold(5,shuffle=True,random_state=100+r),method='predict_proba')[:,1]
                T[r,te]=thr_for_sens(y[tr],pi,0.90)
    return P,T
# ---- 1. feature space on strict set: NN distances and 1-NN / 5-NN AUC
log('1. feature space strict')
Z=StandardScaler().fit_transform(Xd.values); nn=NearestNeighbors(n_neighbors=2).fit(Z); dist,ind=nn.kneighbors(Z)
d1=dist[:,1]; R['nn_dist_quantiles']={q:float(np.percentile(d1,q)) for q in (0,1,5,25,50,75,95,100)}
R['nn_label_agreement']=float((yd[ind[:,1]]==yd).mean())
for k in (1,5):
    P=np.zeros(len(yd))
    for tr,te in folds(yd,None,seed=SEED):
        sc=StandardScaler().fit(Xd.iloc[tr]); knn=KNeighborsClassifier(k).fit(sc.transform(Xd.iloc[tr]),yd[tr]); P[te]=knn.predict_proba(sc.transform(Xd.iloc[te]))[:,1]
    R['AUC_%dNN'%k]=roc_auc_score(yd,P); R['AUC_%dNN_ci'%k]=auc_ci(yd,P)
log('NN quantiles',R['nn_dist_quantiles'],'1NN %.3f 5NN %.3f agree %.3f'%(R['AUC_1NN'],R['AUC_5NN'],R['nn_label_agreement']))
save()
# ---- 2. strict dedup CV (same recipe as the 471 analysis)
log('2. strict dedup CV')
R.setdefault('dedup',{})
for name,(est,grid) in MODELS.items():
    if name in R['dedup'] and 'geo349' in oof.get(name,{}): log('skip',name); continue
    reps=5; nested=name.startswith(('RF','XGB'))
    P,T=run(Xd,yd,None,est,grid,reps=reps,nested_thr=nested)
    p=P.mean(0); per=[roc_auc_score(yd,P[r]) for r in range(reps)]
    e=dict(AUC_avgprob=roc_auc_score(yd,p),AUC_avgprob_ci=auc_ci(yd,p),AUC_per_rep=per,AUC_rep_mean=float(np.mean(per)),AUC_rep_sd=float(np.std(per,ddof=1)),metrics05=metrics(yd,p))
    if nested:
        dec=(P>=T).astype(int); sens=[((dec[r]==1)&(yd==1)).sum()/(yd==1).sum() for r in range(reps)]; spec=[((dec[r]==0)&(yd==0)).sum()/(yd==0).sum() for r in range(reps)]
        e['nested_thr_sens_mean']=float(np.mean(sens)); e['nested_thr_spec_mean']=float(np.mean(spec))
    for sx in ('M','F'):
        i=(dd.gender.values==sx); e['AUC_sex_'+sx]=roc_auc_score(yd[i],p[i]); e['n_sex_'+sx]=int(i.sum())
    R['dedup'][name]=e; oof.setdefault(name,{})['geo349']=p.tolist()
    log(name,'strict AUC %.3f [%.3f,%.3f] per-rep %.3f±%.3f'%(e['AUC_avgprob'],*e['AUC_avgprob_ci'],e['AUC_rep_mean'],e['AUC_rep_sd']),e.get('nested_thr_sens_mean'),e.get('nested_thr_spec_mean'))
    save(); json.dump(oof,open(OUT+'oof_geo.json','w'))
# ---- 3. grouped protocol on the 708 rows with STRICT groups (naive OOF reused from the main run)
log('3. grouped-strict protocol 708')
R.setdefault('grouped_strict',{})
for name,(est,grid) in MODELS.items():
    if 'grouped_geo708' in oof.get(name,{}): log('skip grouped',name); continue
    Pg,_=run(Xa,ya,gs_,est,grid,reps=3,inner_groups=True); pg=Pg.mean(0)
    R['grouped_strict'][name]=dict(grouped_strict_AUC=roc_auc_score(ya,pg),ci=auc_ci(ya,pg),per_rep=[roc_auc_score(ya,Pg[r]) for r in range(3)])
    oof.setdefault(name,{})['grouped_geo708']=pg.tolist(); log(name,'grouped-strict AUC %.3f'%R['grouped_strict'][name]['grouped_strict_AUC'])
    save(); json.dump(oof,open(OUT+'oof_geo.json','w'))
# differences against the naive protocol (same 708 rows): wait for the naive out-of-fold predictions of review_analyses.py
MAIN='/home/claude/exp/out/review/oof_review.json'
for _ in range(360):
    oofm=json.load(open(MAIN)) if os.path.exists(MAIN) else {}
    if all('naive708' in oofm.get(k,{}) for k in MODELS): break
    time.sleep(30)
else: log('naive OOF not available; differences not computed')
for name in MODELS:
    if 'naive708' not in oofm.get(name,{}): continue
    pn=np.array(oofm[name]['naive708']); pg=np.array(oof[name]['grouped_geo708']); ci,pv=paired_boot_diff(ya,pn,pg); dl=delong(ya,pn,pg)
    R['grouped_strict'][name].update(naive_minus_grouped_strict=float(roc_auc_score(ya,pn)-roc_auc_score(ya,pg)),ci_diff=ci,p_boot=pv,delong_p=dl[2])
    log(name,'naive - grouped-strict %.3f'%R['grouped_strict'][name]['naive_minus_grouped_strict'],ci)
save()
log('done')
