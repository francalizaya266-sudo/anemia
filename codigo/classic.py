import sys; sys.path.insert(0,'/home/claude/exp')
import numpy as np, pandas as pd, cv2, time, pickle, json, warnings; warnings.filterwarnings('ignore')
from common import *
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier

def color_features(im):
    m=mask_of(im); px=im[m].astype(np.float32)
    if len(px)<20: px=im.reshape(-1,3).astype(np.float32)
    f={}
    hsv=cv2.cvtColor(px[None].astype(np.uint8),cv2.COLOR_RGB2HSV)[0].astype(np.float32)
    lab=cv2.cvtColor((px[None]/255).astype(np.float32),cv2.COLOR_RGB2LAB)[0]
    chans={'R':px[:,0],'G':px[:,1],'B':px[:,2],'H':hsv[:,0],'S':hsv[:,1],'V':hsv[:,2],'L':lab[:,0],'a':lab[:,1],'b':lab[:,2]}
    eps=1.0
    chans['EI']=np.log(px[:,0]+eps)-np.log(px[:,1]+eps)          # erythema index
    chans['rg']=px[:,0]/(px.sum(1)+eps)                           # normalized red chromaticity
    chans['gg']=px[:,1]/(px.sum(1)+eps)
    for k,v in chans.items():
        f[k+'_mean']=v.mean(); f[k+'_std']=v.std()
        f[k+'_p10'],f[k+'_p50'],f[k+'_p90']=np.percentile(v,[10,50,90])
    f['a_over_L']=chans['a'].mean()/(chans['L'].mean()+eps)
    f['R_over_G']=px[:,0].mean()/(px[:,1].mean()+eps)
    f['frac_high_a']=(chans['a']>20).mean()   # fraction of strongly red pixels
    return f

def featurize(d):
    return pd.DataFrame([color_features(read_rgb(p)) for p in d.image_path])

MODELS={
 'LR (color)':(make_pipeline(StandardScaler(),LogisticRegression(max_iter=5000,class_weight='balanced')),{'logisticregression__C':[0.01,0.1,1,10]}),
 'SVM-RBF (color)':(make_pipeline(StandardScaler(),SVC(probability=True,class_weight='balanced',random_state=SEED)),{'svc__C':[0.3,1,3,10],'svc__gamma':['scale',0.01]}),
 'RF (color)':(RandomForestClassifier(500,class_weight='balanced',random_state=SEED,n_jobs=4),{'min_samples_leaf':[1,3,5],'max_features':['sqrt',0.3]}),
 'XGBoost (color)':(XGBClassifier(n_estimators=300,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,eval_metric='logloss',n_jobs=4,random_state=SEED),{'max_depth':[2,3,4],'min_child_weight':[1,5]}),
}

def run_cv(X,y,groups,name,est,grid,reps=5):
    P=np.zeros((reps,len(y)))
    for r in range(reps):
        for tr,te in folds(y,groups,seed=SEED+r):
            gs=GridSearchCV(est,grid,scoring='roc_auc',cv=StratifiedKFold(5,shuffle=True,random_state=r),n_jobs=1)  # nested: inner CV on train fold only
            gs.fit(X.iloc[tr],y[tr]); P[r,te]=gs.predict_proba(X.iloc[te])[:,1]
    return P

if __name__=='__main__':
    d=load_meta(); dd=dedup(d)
    print('all',len(d),'dedup',len(dd),dd.anemia.value_counts().to_dict())
    Xa=featurize(d); Xd=featurize(dd)
    Xa.to_csv('/home/claude/exp/out/feat_all.csv',index=False); Xd.to_csv('/home/claude/exp/out/feat_dedup.csv',index=False)
    ya=d.anemia.values; yd=dd.anemia.values
    rows=[]; oof={}
    for name,(est,grid) in MODELS.items():
        # A) naive protocol, as in prior literature: random image-level CV on 710 incl. duplicates
        Pn=run_cv(Xa,ya,None,name,est,grid,reps=1); mn=metrics(ya,Pn.mean(0))
        # B) leakage-free protocol: deduplicated, group-aware, repeated 5x5 nested CV
        P=run_cv(Xd,yd,None,name,est,grid,reps=5); p=P.mean(0); m=metrics(yd,p); ci=boot_ci(yd,p)
        aucs=[metrics(yd,P[r])['AUC'] for r in range(P.shape[0])]
        oof[name]=p.tolist()
        rows.append(dict(model=name,naive_AUC=mn['AUC'],naive_Acc=mn['Acc'],**m,**{k+'_ci':v for k,v in ci.items()},AUC_rep_sd=np.std(aucs)))
        print(name,'naive AUC %.3f acc %.3f | clean AUC %.3f [%.3f-%.3f] F1 %.3f Acc %.3f'%(mn['AUC'],mn['Acc'],m['AUC'],*ci['AUC'],m['F1'],m['Acc']),flush=True)
    pd.DataFrame(rows).to_csv('/home/claude/exp/out/classic_results.csv',index=False)
    json.dump({'y':yd.tolist(),'oof':oof},open('/home/claude/exp/out/classic_oof.json','w'))
