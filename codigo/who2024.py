# Sensitivity of the benchmark to the WHO 2024 cut-offs (105 g/L for 6-23 months, 110 g/L for 24-59 months)
import sys; sys.path.insert(0,'/home/claude/exp')
import numpy as np, pandas as pd, json, warnings; warnings.filterwarnings('ignore')
from common import *
from classic import MODELS, run_cv
d=dedup(load_meta()); X=pd.read_csv('/home/claude/exp/out/feat_dedup.csv')
y11=d.anemia.values
y24=np.where(d.age_months<24, d.hb<10.5, d.hb<11.0).astype(int)
res={'n':len(d),'anemia_hb11':int(y11.sum()),'anemia_who2024':int(y24.sum()),'reclassified':int((y11!=y24).sum()),
     'reclassified_6_23m':int(((y11!=y24)&(d.age_months<24)).sum()),'n_6_23m':int((d.age_months<24).sum())}
# (a) model trained on Hb<11 labels, scored against WHO 2024 labels (what happens if the deployed model meets the new rule)
oof=json.load(open('/home/claude/exp/out/classic_oof.json'))
for m in ['RF (color)','XGBoost (color)']:
    p=np.array(oof['oof'][m]); res[m+' | trained Hb<11, scored WHO2024']=dict(metrics(y24,p),AUC_ci=boot_ci(y24,p)['AUC'])
# (b) retrained with WHO 2024 labels, same nested repeated CV
for m in ['RF (color)','XGBoost (color)']:
    est,grid=MODELS[m]; P=run_cv(X,y24,None,m,est,grid,reps=5); p=P.mean(0)
    res[m+' | retrained WHO2024']=dict(metrics(y24,p),AUC_ci=boot_ci(y24,p)['AUC'])
    print(m,res[m+' | retrained WHO2024'],flush=True)
# (c) per age-group performance of the Hb<11 model
for m in ['RF (color)']:
    p=np.array(oof['oof'][m])
    for lab,mask in [('6-23m',d.age_months.values<24),('24-59m',d.age_months.values>=24)]:
        res[m+f' | {lab}']=dict(n=int(mask.sum()),AUC=roc_auc_score(y11[mask],p[mask]))
json.dump(res,open('/home/claude/exp/out/who2024.json','w'),indent=1,default=float); print(json.dumps(res,indent=1,default=float))
