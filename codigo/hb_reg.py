# Secondary analysis: Hb regression only on images that appear once (metadata linkage plausible)
import sys; sys.path.insert(0,'/home/claude/exp')
import numpy as np, pandas as pd, json
from common import *
from classic import featurize
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.ensemble import RandomForestRegressor
d=load_meta(); n=d.groupby('grp').size(); s=d[d.grp.map(n)==1].reset_index(drop=True)
X=featurize(s); y=s.hb.values; res={'n':len(s),'hb_mean':y.mean(),'hb_sd':y.std()}
for name,est,grid in [('SVR',make_pipeline(StandardScaler(),SVR()),{'svr__C':[0.3,1,3,10],'svr__epsilon':[0.1,0.5]}),
                      ('RF',RandomForestRegressor(500,random_state=SEED,n_jobs=4),{'min_samples_leaf':[3,10]})]:
    P=[]
    for r in range(3):
        P.append(cross_val_predict(GridSearchCV(est,grid,cv=KFold(5,shuffle=True,random_state=r),scoring='neg_mean_absolute_error'),X,y,cv=KFold(5,shuffle=True,random_state=SEED+r)))
    p=np.mean(P,0); err=np.abs(p-y); rng=np.random.default_rng(0)
    bs=[np.mean(err[rng.integers(0,len(y),len(y))]) for _ in range(2000)]
    diff=p-y
    res[name]=dict(MAE=err.mean(),MAE_ci=np.percentile(bs,[2.5,97.5]).tolist(),r=np.corrcoef(p,y)[0,1],bias=diff.mean(),LoA=[diff.mean()-1.96*diff.std(),diff.mean()+1.96*diff.std()],
                   MAE_baseline_mean=np.abs(y-y.mean()).mean())
json.dump(res,open('/home/claude/exp/out/hb_reg.json','w'),indent=1,default=float); print(json.dumps(res,indent=1,default=float))
