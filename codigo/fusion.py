# Hybrid model: frozen CNN embedding (PCA) + colour features -> classical classifier, same leakage-free protocol
import sys, os; sys.path.insert(0,'/home/claude/exp'); os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import numpy as np, pandas as pd, json, time, pickle, tensorflow as tf
from common import *
from cnn import tight
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier
A=tf.keras.applications
d=dedup(load_meta()); y=d.anemia.values
C=pd.read_csv('/home/claude/exp/out/feat_dedup.csv')
X=np.stack([tight(read_rgb(p),224) for p in d.image_path]).astype(np.float32)
base=A.EfficientNetB0(include_top=False,weights='imagenet',input_shape=(224,224,3),pooling='avg')
t0=time.perf_counter(); E=base.predict(X,batch_size=32,verbose=0); emb_ms=(time.perf_counter()-t0)*1000/len(X)
E=pd.DataFrame(E,columns=[f'e{i}' for i in range(E.shape[1])]); XX=pd.concat([C,E],axis=1)
ct=ColumnTransformer([('col',StandardScaler(),list(C.columns)),('emb',Pipeline([('s',StandardScaler()),('p',PCA(32,random_state=SEED))]),list(E.columns))])
models={'Fusión EffNet-B0 + color (LR)':(Pipeline([('ct',ct),('clf',LogisticRegression(max_iter=5000,class_weight='balanced'))]),{'clf__C':[0.01,0.03,0.1,0.3]}),
        'Fusión EffNet-B0 + color (XGB)':(Pipeline([('ct',ct),('clf',XGBClassifier(n_estimators=300,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,eval_metric='logloss',n_jobs=4,random_state=SEED))]),{'clf__max_depth':[2,3,4]}),
        'Embedding EffNet-B0 solo (LR)':(Pipeline([('s',StandardScaler()),('p',PCA(32,random_state=SEED)),('clf',LogisticRegression(max_iter=5000,class_weight='balanced'))]),{'clf__C':[0.01,0.03,0.1,0.3]})}
rows=[]; oof={}
for name,(est,grid) in models.items():
    Xin = E if 'solo' in name else XX
    P=np.zeros((3,len(y)))
    for r in range(3):
        for tr,te in folds(y,None,seed=SEED+r):
            gs=GridSearchCV(est,grid,scoring='roc_auc',cv=StratifiedKFold(5,shuffle=True,random_state=r)).fit(Xin.iloc[tr],y[tr]); P[r,te]=gs.predict_proba(Xin.iloc[te])[:,1]
    p=P.mean(0); m=metrics(y,p); ci=boot_ci(y,p); oof[name]=p.tolist()
    rows.append(dict(model=name,**m,**{k+'_ci':v for k,v in ci.items()},emb_ms=emb_ms))
    print(name,'AUC %.3f [%.3f-%.3f] F1 %.3f Acc %.3f'%(m['AUC'],*ci['AUC'],m['F1'],m['Acc']),flush=True)
pd.DataFrame(rows).to_csv('/home/claude/exp/out/fusion_results.csv',index=False)
json.dump({'y':y.tolist(),'oof':oof,'emb_ms':emb_ms},open('/home/claude/exp/out/fusion_oof.json','w'))
