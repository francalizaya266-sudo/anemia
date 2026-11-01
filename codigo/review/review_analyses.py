# Additional analyses requested in peer review (CPU only): grouped protocol, feature-space duplicate audit,
# 1-NN baseline, singleton-only and random-representative sensitivity, shape-only baseline, brightness-invariant
# ablation, per-repetition AUCs, nested operating thresholds, calibration, PPV/NPV, sex subgroups, LOHO per centre,
# DeLong / paired-bootstrap tests against the CNN out-of-fold predictions, and SHAP values for re-plotting.
import sys; sys.path.insert(0,'/home/claude/exp')
import numpy as np, pandas as pd, json, time, warnings, os, glob, cv2; warnings.filterwarnings('ignore')
from common import *
from classic import MODELS
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedGroupKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from scipy import stats
OUT='/home/claude/exp/out/review/'; os.makedirs(OUT,exist_ok=True)
T0=time.time()
def log(*a):
    print('[%6.0fs]'%(time.time()-T0),*a,flush=True)
d=load_meta(); dd=dedup(d)
Xa=pd.read_csv('/home/claude/exp/out/feat_all.csv'); Xd=pd.read_csv('/home/claude/exp/out/feat_dedup.csv')
assert len(Xa)==len(d)==708 and len(Xd)==len(dd)==471
ya=d.anemia.values; yd=dd.anemia.values; ga=d.grp.values
R=json.load(open(OUT+'review_results.json')) if os.path.exists(OUT+'review_results.json') else {}
oof_store=json.load(open(OUT+'oof_review.json')) if os.path.exists(OUT+'oof_review.json') else {}
def save(): json.dump(R,open(OUT+'review_results.json','w'),indent=1,default=float); json.dump(oof_store,open(OUT+'oof_review.json','w'))

def thr_for_sens(y,p,target=0.90):
    # largest threshold whose sensitivity >= target
    ts=np.unique(p)[::-1]
    for t in ts:
        if ((p>=t)&(y==1)).sum()/(y==1).sum()>=target: return float(t)
    return float(ts[-1])

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

def auc_ci(y,p,B=2000,seed=SEED):
    rng=np.random.default_rng(seed); n=len(y); a=[]
    for _ in range(B):
        i=rng.integers(0,n,n)
        if len(np.unique(y[i]))<2: continue
        a.append(roc_auc_score(y[i],p[i]))
    return [float(np.percentile(a,2.5)),float(np.percentile(a,97.5))]

def paired_boot_diff(y,p1,p2,B=2000,seed=SEED):
    rng=np.random.default_rng(seed); n=len(y); dlist=[]
    for _ in range(B):
        i=rng.integers(0,n,n)
        if len(np.unique(y[i]))<2: continue
        dlist.append(roc_auc_score(y[i],p1[i])-roc_auc_score(y[i],p2[i]))
    dlist=np.array(dlist); return [float(np.percentile(dlist,2.5)),float(np.percentile(dlist,97.5))], float(2*min((dlist<=0).mean(),(dlist>=0).mean()))

# ---- DeLong test (Sun & Xu 2014) ----
def _midrank(x):
    J=np.argsort(x); Z=x[J]; N=len(x); T=np.zeros(N); i=0
    while i<N:
        j=i
        while j<N and Z[j]==Z[i]: j+=1
        T[i:j]=0.5*(i+j-1)+1; i=j
    T2=np.empty(N); T2[J]=T; return T2
def delong(y,p1,p2):
    pos=np.where(y==1)[0]; neg=np.where(y==0)[0]; m=len(pos); n=len(neg)
    preds=np.vstack([p1,p2]); k=2
    tx=np.array([_midrank(preds[r,pos]) for r in range(k)]); ty=np.array([_midrank(preds[r,neg]) for r in range(k)]); tz=np.array([_midrank(np.concatenate([preds[r,pos],preds[r,neg]])) for r in range(k)])  # positives first (Sun & Xu 2014)
    aucs=tz[:,:m].sum(1)/m/n-(m+1.0)/(2.0*n)
    v01=(tz[:,:m]-tx)/n; v10=1.0-(tz[:,m:]-ty)/m
    S=np.cov(v01)/m+np.cov(v10)/n
    L=np.array([1,-1]); z=(aucs[0]-aucs[1])/np.sqrt(L@S@L); pval=2*stats.norm.sf(abs(z))
    return float(aucs[0]-aucs[1]), float(z), float(pval)

# =====================================================================
# 1. Three protocols on the same models: naive 708, grouped 708, deduplicated 471
# =====================================================================
log('1. protocols')
R.setdefault('protocols',{})
for name,(est,grid) in MODELS.items():
    if name in oof_store and 'naive708' in oof_store[name] and 'grouped708' in oof_store[name] and name in R.get('protocols',{}): log('skip',name); continue
    Pn,_=run(Xa,ya,None,est,grid,reps=3)
    Pg,_=run(Xa,ya,ga,est,grid,reps=3,inner_groups=True)
    pn=Pn.mean(0); pg=Pg.mean(0)
    ci,pv=paired_boot_diff(ya,pn,pg)
    dl=delong(ya,pn,pg)
    R['protocols'][name]=dict(naive708_AUC=roc_auc_score(ya,pn),naive708_ci=auc_ci(ya,pn),naive708_rep=[roc_auc_score(ya,Pn[r]) for r in range(3)],
        grouped708_AUC=roc_auc_score(ya,pg),grouped708_ci=auc_ci(ya,pg),grouped708_rep=[roc_auc_score(ya,Pg[r]) for r in range(3)],
        naive_minus_grouped=roc_auc_score(ya,pn)-roc_auc_score(ya,pg),naive_minus_grouped_ci=ci,naive_minus_grouped_p_boot=pv,delong_z=dl[1],delong_p=dl[2],
        grouped708_metrics=metrics(ya,pg))
    oof_store.setdefault(name,{}).update(naive708=pn.tolist(),grouped708=pg.tolist()); save()
    log(name,'naive %.3f grouped %.3f diff %.3f ci %s p=%.4f'%(roc_auc_score(ya,pn),roc_auc_score(ya,pg),R['protocols'][name]['naive_minus_grouped'],ci,pv))
save()

# =====================================================================
# 2. Deduplicated set: per-repetition AUC, nested thresholds, calibration, PPV/NPV, sex subgroups
# =====================================================================
log('2. dedup rerun with nested thresholds')
R.setdefault('dedup',{})
for name,(est,grid) in MODELS.items():
    if name in R['dedup'] and 'dedup471' in oof_store.get(name,{}): log('skip dedup',name); continue
    reps=5; nested = name.startswith(('RF','XGB'))
    P,T=run(Xd,yd,None,est,grid,reps=reps,nested_thr=nested)
    p=P.mean(0); per=[roc_auc_score(yd,P[r]) for r in range(reps)]
    e=dict(AUC_avgprob=roc_auc_score(yd,p),AUC_avgprob_ci=auc_ci(yd,p),AUC_per_rep=per,AUC_rep_mean=float(np.mean(per)),AUC_rep_sd=float(np.std(per,ddof=1)))
    # calibration slope/intercept (logistic recalibration on logit)
    lp=np.log(np.clip(p,1e-4,1-1e-4)/(1-np.clip(p,1e-4,1-1e-4)))
    lr=LogisticRegression(C=1e6).fit(lp.reshape(-1,1),yd); e['cal_slope']=float(lr.coef_[0][0]); e['cal_intercept_at_mean']=float(lr.intercept_[0]+lr.coef_[0][0]*lp.mean())
    lr0=LogisticRegression(C=1e6,fit_intercept=True).fit(np.zeros((len(yd),1)),yd)  # placeholder
    # calibration-in-the-large: intercept with slope fixed at 1
    e['cal_in_large']=float(np.log(yd.mean()/(1-yd.mean()))-np.log(p.mean()/(1-p.mean())))
    if nested:
        # nested threshold: per repetition each image has its fold's threshold; decision = p_r >= T_r, pooled
        dec=(P>=T).astype(int); sens=[((dec[r]==1)&(yd==1)).sum()/(yd==1).sum() for r in range(reps)]; spec=[((dec[r]==0)&(yd==0)).sum()/(yd==0).sum() for r in range(reps)]
        e['nested_thr_sens_mean']=float(np.mean(sens)); e['nested_thr_spec_mean']=float(np.mean(spec)); e['nested_thr_sens_per_rep']=sens; e['nested_thr_spec_per_rep']=spec
        e['nested_thr_values_median']=float(np.nanmedian(T))
        # post hoc threshold on pooled predictions for comparison
        t=thr_for_sens(yd,p,0.90); e['posthoc_thr']=t; e['posthoc_sens']=float(((p>=t)&(yd==1)).sum()/(yd==1).sum()); e['posthoc_spec']=float(((p<t)&(yd==0)).sum()/(yd==0).sum())
        for prev in (0.43,0.70):
            se,sp=e['nested_thr_sens_mean'],e['nested_thr_spec_mean']
            e['PPV_prev%d'%int(prev*100)]=se*prev/(se*prev+(1-sp)*(1-prev)); e['NPV_prev%d'%int(prev*100)]=sp*(1-prev)/(sp*(1-prev)+(1-se)*prev)
    # sex subgroups
    for sx in ('M','F'):
        i=(dd.gender.values==sx); e['AUC_sex_'+sx]=roc_auc_score(yd[i],p[i]); e['AUC_sex_%s_ci'%sx]=auc_ci(yd[i],p[i]); e['n_sex_'+sx]=int(i.sum())
    R['dedup'][name]=e; oof_store.setdefault(name,{})['dedup471']=p.tolist(); save()
    log(name,'dedup AUC %.3f per-rep %.3f±%.3f'%(e['AUC_avgprob'],e['AUC_rep_mean'],e['AUC_rep_sd']), 'nested sens/spec' , e.get('nested_thr_sens_mean'), e.get('nested_thr_spec_mean'))
save()

# =====================================================================
# 3. Singleton-only and random-representative sensitivity (RF, XGB)
# =====================================================================
log('3. singleton / random representative')
R.setdefault('singleton',{})
sz=d.groupby('grp').grp.transform('size').values; single=sz==1
Xs=Xa[single].reset_index(drop=True); ys=ya[single]
for name in ['RF (color)','XGBoost (color)']:
    if name+'_random_representative' in R['singleton']: log('skip singleton',name); continue
    est,grid=MODELS[name]
    P,_=run(Xs,ys,None,est,grid,reps=3); p=P.mean(0)
    R['singleton'][name]=dict(n=int(single.sum()),prev=float(ys.mean()),AUC=roc_auc_score(ys,p),ci=auc_ci(ys,p),per_rep=[roc_auc_score(ys,P[r]) for r in range(3)])
    log(name,'singletons AUC %.3f %s'%(R['singleton'][name]['AUC'],R['singleton'][name]['ci']))
    rr=[]
    for k in range(3):
        rng=np.random.default_rng(k); idx=d.groupby('grp').apply(lambda g: rng.choice(g.index)).values
        Xr=Xa.iloc[idx].reset_index(drop=True); yr=ya[idx]
        P,_=run(Xr,yr,None,est,grid,reps=1,seed0=SEED+k); rr.append(roc_auc_score(yr,P[0]))
    R['singleton'][name+'_random_representative']=dict(AUCs=rr,mean=float(np.mean(rr)))
    log(name,'random representative AUCs',rr); save()
save(); log('DONE_MAIN (sections 4-8 run in review_analyses_b.py)'); sys.exit(0)

# =====================================================================
# 4. Feature-space duplicate audit and 1-NN baseline
# =====================================================================
log('4. feature-space audit')
sc=StandardScaler().fit(Xd); Zd=sc.transform(Xd)
nn=NearestNeighbors(n_neighbors=2).fit(Zd); dist,ind=nn.kneighbors(Zd); nnd=dist[:,1]
# reference: distances between distinct-digest members of the same pHash group (true near-duplicates) in feat_all space
Za=sc.transform(Xa); ref=[]
for g,sub in d.groupby('grp'):
    if sub.sha.nunique()>1:
        idx=sub.index.values; shas=sub.sha.values
        for i in range(len(idx)):
            for j in range(i+1,len(idx)):
                if shas[i]!=shas[j]: ref.append(float(np.linalg.norm(Za[idx[i]]-Za[idx[j]])))
closest=np.argsort(nnd)[:10]
pairs=[dict(a=dd.image_id[i],b=dd.image_id[ind[i,1]],dist=float(nnd[i]),hb_a=float(dd.hb[i]),hb_b=float(dd.hb[ind[i,1]]),label_a=int(yd[i]),label_b=int(yd[ind[i,1]]),hosp_a=dd.hospital[i],hosp_b=dd.hospital[ind[i,1]]) for i in closest]
R['feature_space']=dict(nn_dist_quantiles={q:float(np.percentile(nnd,q)) for q in (0,1,5,25,50,75,95,100)},n_true_near_dup_pairs=len(ref),
    true_near_dup_dist_quantiles={q:float(np.percentile(ref,q)) for q in (0,25,50,75,100)} if ref else None,
    n_dedup_pairs_below_max_true_dup=int((nnd<=max(ref)).sum()) if ref else None, n_dedup_pairs_below_median_true_dup=int((nnd<=np.median(ref)).sum()) if ref else None,
    closest_pairs=pairs, nn_label_agreement=float((yd==yd[ind[:,1]]).mean()))
# 1-NN and 5-NN baselines (5x5 CV)
for k in (1,5):
    P=np.zeros((5,len(yd)))
    for r in range(5):
        for tr,te in folds(yd,None,seed=SEED+r):
            m=make_pipeline(StandardScaler(),KNeighborsClassifier(k)).fit(Xd.iloc[tr],yd[tr]); P[r,te]=m.predict_proba(Xd.iloc[te])[:,1]
    p=P.mean(0); R['feature_space']['%dNN_AUC'%k]=roc_auc_score(yd,p); R['feature_space']['%dNN_ci'%k]=auc_ci(yd,p); R['feature_space']['%dNN_acc'%k]=float(((p>=0.5)==yd).mean())
log('feature space', R['feature_space']['nn_dist_quantiles'], 'true dup', R['feature_space']['true_near_dup_dist_quantiles'], '1NN', R['feature_space']['1NN_AUC'])
save()

# =====================================================================
# 5. Shape-only baseline (mask geometry) and brightness-invariant ablation
# =====================================================================
log('5. shape-only + ablations')
def shape_feats(im):
    m=mask_of(im).astype(np.uint8); h,w=m.shape; f={}
    f['area_frac']=m.mean(); ys_,xs_=np.nonzero(m)
    if len(xs_)<5: return {k:0.0 for k in ['area_frac','bbox_w','bbox_h','aspect','extent','solidity','perimeter_n','circularity','cx','cy','angle','ell_ratio']+['hu%d'%i for i in range(7)]}
    x0,x1,y0,y1=xs_.min(),xs_.max(),ys_.min(),ys_.max(); f['bbox_w']=(x1-x0+1)/w; f['bbox_h']=(y1-y0+1)/h; f['aspect']=(x1-x0+1)/(y1-y0+1)
    f['extent']=m.sum()/((x1-x0+1)*(y1-y0+1))
    cnts,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); c=max(cnts,key=cv2.contourArea)
    hull=cv2.convexHull(c); f['solidity']=cv2.contourArea(c)/max(cv2.contourArea(hull),1); per=cv2.arcLength(c,True); f['perimeter_n']=per/np.sqrt(h*w)
    f['circularity']=4*np.pi*cv2.contourArea(c)/max(per**2,1); f['cx']=xs_.mean()/w; f['cy']=ys_.mean()/h
    if len(c)>=5:
        (ex,ey),(ma,mi),ang=cv2.fitEllipse(c); f['angle']=ang; f['ell_ratio']=mi/max(ma,1e-6)
    else: f['angle']=0; f['ell_ratio']=1
    hu=cv2.HuMoments(cv2.moments(m)).flatten()
    for i,v in enumerate(hu): f['hu%d'%i]=float(-np.sign(v)*np.log10(abs(v)+1e-12))
    return f
S=pd.DataFrame([shape_feats(read_rgb(p)) for p in dd.image_path]); S.to_csv(OUT+'shape_feats_dedup.csv',index=False)
R['ablation']={}
rf=RandomForestClassifier(500,class_weight='balanced',random_state=SEED,n_jobs=4); grid={'min_samples_leaf':[1,3,5],'max_features':['sqrt',0.3]}
P,_=run(S,yd,None,rf,grid,reps=3); p=P.mean(0); R['ablation']['shape_only_RF']=dict(n_feat=S.shape[1],AUC=roc_auc_score(yd,p),ci=auc_ci(yd,p)); log('shape-only RF AUC %.3f'%roc_auc_score(yd,p))
inv=[c for c in Xd.columns if c.split('_')[0] in ('rg','gg','H','S','EI') or c in ('R_over_G',)]
P,_=run(Xd[inv],yd,None,rf,grid,reps=3); p=P.mean(0); R['ablation']['scale_invariant_RF']=dict(features=inv,n_feat=len(inv),AUC=roc_auc_score(yd,p),ci=auc_ci(yd,p)); log('scale-invariant RF AUC %.3f (%d feats)'%(roc_auc_score(yd,p),len(inv)))
inv2=inv+[c for c in Xd.columns if c.split('_')[0] in ('a','b')]+['a_over_L','frac_high_a']
P,_=run(Xd[inv2],yd,None,rf,grid,reps=3); p=P.mean(0); R['ablation']['chromatic_RF']=dict(n_feat=len(inv2),AUC=roc_auc_score(yd,p),ci=auc_ci(yd,p)); log('chromatic (no L/V/RGB means) RF AUC %.3f (%d feats)'%(roc_auc_score(yd,p),len(inv2)))
lum=[c for c in Xd.columns if c.split('_')[0] in ('L','V','R','G','B')]
P,_=run(Xd[lum],yd,None,rf,grid,reps=3); p=P.mean(0); R['ablation']['luminance_only_RF']=dict(n_feat=len(lum),AUC=roc_auc_score(yd,p),ci=auc_ci(yd,p)); log('luminance-only RF AUC %.3f (%d feats)'%(roc_auc_score(yd,p),len(lum)))
save()

# =====================================================================
# 6. LOHO per centre with CI (RF, nested grid per fold)
# =====================================================================
log('6. LOHO')
hosp=dd.hospital.values; p=np.zeros(len(yd)); rf=RandomForestClassifier(500,class_weight='balanced',random_state=SEED,n_jobs=4)
for hsp in np.unique(hosp):
    te=np.where(hosp==hsp)[0]; tr=np.where(hosp!=hsp)[0]
    gs=GridSearchCV(rf,grid,scoring='roc_auc',cv=StratifiedKFold(5,shuffle=True,random_state=0)).fit(Xd.iloc[tr],yd[tr]); p[te]=gs.predict_proba(Xd.iloc[te])[:,1]
per=[]
for hsp in np.unique(hosp):
    i=hosp==hsp; n=int(i.sum()); npos=int(yd[i].sum())
    if n>=20 and 0<npos<n:
        a=roc_auc_score(yd[i],p[i]); ci=auc_ci(yd[i],p[i]); se=(ci[1]-ci[0])/3.92
        per.append(dict(hospital=hsp,n=n,n_pos=npos,AUC=a,ci=ci,se=se))
# DerSimonian-Laird random-effects pooled AUC
a=np.array([x['AUC'] for x in per]); v=np.array([x['se']**2 for x in per]); w=1/v; mu_fe=(w*a).sum()/w.sum(); Q=(w*(a-mu_fe)**2).sum(); dfq=len(a)-1
tau2=max(0,(Q-dfq)/(w.sum()-(w**2).sum()/w.sum())); wr=1/(v+tau2); mu_re=(wr*a).sum()/wr.sum(); se_re=np.sqrt(1/wr.sum()); I2=max(0,(Q-dfq)/Q) if Q>0 else 0
R['loho']=dict(pooled_AUC=roc_auc_score(yd,p),pooled_ci=auc_ci(yd,p),per_centre=per,random_effects_AUC=float(mu_re),random_effects_ci=[float(mu_re-1.96*se_re),float(mu_re+1.96*se_re)],tau2=float(tau2),I2=float(I2),Q=float(Q),k=len(per))
log('LOHO pooled %.3f RE %.3f [%.3f,%.3f] I2 %.2f'%(R['loho']['pooled_AUC'],mu_re,*R['loho']['random_effects_ci'],I2))
save()

# =====================================================================
# 7. Formal comparisons: DeLong + paired bootstrap, Holm-adjusted
# =====================================================================
log('7. DeLong')
cl=json.load(open('/home/claude/exp/out/classic_oof.json')); assert cl['y']==yd.tolist()
cnn={}
for f in sorted(glob.glob('/home/claude/anemia/resultados/seeds/cnn_*_clean_s*.json')):
    j=json.load(open(f)); assert j['y']==yd.tolist(); name=j['res']['model']; seed=j['res']['seed']
    o=j['oof']; o=o['fp32'] if isinstance(o,dict) and 'fp32' in o else (list(o.values())[0] if isinstance(o,dict) else o)
    cnn['%s seed %s'%(name,seed)]=np.array(o,float)
comps=[]
for ref in ['RF (color)','XGBoost (color)']:
    pr=np.array(cl['oof'][ref])
    for other,po in list(cnn.items())+[(k,np.array(v)) for k,v in cl['oof'].items() if k!=ref and (ref=='RF (color)' or k not in ('RF (color)',))]:
        dlt,z,pv=delong(yd,pr,po); ci,pb=paired_boot_diff(yd,pr,po)
        comps.append(dict(ref=ref,other=other,AUC_ref=roc_auc_score(yd,pr),AUC_other=roc_auc_score(yd,po),diff=dlt,delong_z=z,delong_p=pv,boot_ci=ci,boot_p=pb))
# Holm
ps=np.array([c['delong_p'] for c in comps]); order=np.argsort(ps); m=len(ps); adj=np.empty(m)
running=0
for rank,idx in enumerate(order):
    running=max(running,(m-rank)*ps[idx]); adj[idx]=min(1,running)
for c,a_ in zip(comps,adj): c['delong_p_holm']=float(a_)
R['comparisons']=comps
for c in comps: log('%s vs %s: diff %.3f DeLong p=%.2g Holm %.2g boot CI %s'%(c['ref'],c['other'],c['diff'],c['delong_p'],c['delong_p_holm'],np.round(c['boot_ci'],3)))
save()

# =====================================================================
# 8. SHAP values for re-plotting (XGBoost best params on all 471)
# =====================================================================
log('8. SHAP')
import shap
xa=json.load(open('/home/claude/anemia/resultados/xai_classic.json'))
est,grid=MODELS['XGBoost (color)']; bp=xa['XGBoost (color)']; est.set_params(**bp); est.fit(Xd,yd)
sv=shap.TreeExplainer(est).shap_values(Xd); np.savez(OUT+'shap_xgb.npz',sv=sv,X=Xd.values,cols=np.array(Xd.columns))
imp=pd.Series(np.abs(sv).mean(0),index=Xd.columns).sort_values(ascending=False); R['shap_top12']=imp.head(12).round(4).to_dict(); R['shap_svm_top']=xa.get('SVM-RBF (color)_top')
log('top SHAP', list(imp.head(8).index)); save(); log('DONE')
