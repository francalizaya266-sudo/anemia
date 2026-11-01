import sys; sys.path.insert(0,'/home/claude/exp')
import numpy as np, pandas as pd, shap, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from common import *
from classic import MODELS, featurize, color_features
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import time, pickle, os
d=dedup(load_meta()); y=d.anemia.values
X=pd.read_csv('/home/claude/exp/out/feat_dedup.csv')
out={}
for name in ['XGBoost (color)','SVM-RBF (color)']:
    est,grid=MODELS[name]
    gs=GridSearchCV(est,grid,scoring='roc_auc',cv=StratifiedKFold(5,shuffle=True,random_state=0)).fit(X,y)
    m=gs.best_estimator_; out[name]=gs.best_params_
    pickle.dump(m,open(f'/home/claude/exp/out/{name.split()[0]}.pkl','wb'))
    # edge feasibility for classic: feature extraction + predict, single image
    im=read_rgb(d.image_path[0]); t=[]
    for _ in range(50):
        t0=time.perf_counter(); f=pd.DataFrame([color_features(im)]); m.predict_proba(f); t.append((time.perf_counter()-t0)*1000)
    out[name+'_lat_ms']=float(np.median(t)); out[name+'_MB']=os.path.getsize(f'/home/claude/exp/out/{name.split()[0]}.pkl')/2**20
    if name.startswith('XGB'):
        sv=shap.TreeExplainer(m).shap_values(X)
    else:
        bg=shap.kmeans(X,20); ex=shap.KernelExplainer(lambda z:m.predict_proba(pd.DataFrame(z,columns=X.columns))[:,1],bg)
        idx=np.random.default_rng(0).choice(len(X),120,replace=False); sv=ex.shap_values(X.iloc[idx],nsamples=300); Xs=X.iloc[idx]
    Xs = X if name.startswith('XGB') else Xs
    imp=pd.Series(np.abs(sv).mean(0),index=X.columns).sort_values(ascending=False)
    out[name+'_top']=imp.head(10).round(4).to_dict()
    plt.figure(); shap.summary_plot(sv,Xs,max_display=12,show=False); plt.title(f'SHAP – {name}'); plt.tight_layout()
    plt.savefig(f'/home/claude/exp/out/shap_{name.split()[0]}.png',dpi=160); plt.close()
json.dump(out,open('/home/claude/exp/out/xai_classic.json','w'),indent=1,default=str); print(json.dumps(out,indent=1,default=str))
