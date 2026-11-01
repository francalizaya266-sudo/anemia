import json, glob, pandas as pd, numpy as np, os
O='/home/claude/exp/out/'
rows=[]
def ci(c): return f"{c[0]:.2f}–{c[1]:.2f}"
cl=pd.read_csv(O+'classic_results.csv'); xai=json.load(open(O+'xai_classic.json'))
lat={'XGBoost (color)':6.1,'SVM-RBF (color)':6.3,'RF (color)':None,'LR (color)':None}
mb={'XGBoost (color)':xai['XGBoost (color)_MB'],'SVM-RBF (color)':xai['SVM-RBF (color)_MB']}
for _,r in cl.iterrows():
    rows.append(dict(Modelo=r.model,Protocolo='limpio (471)',AUC=r.AUC,AUC_IC=ci(eval(r.AUC_ci)),F1=r.F1,Acc=r.Acc,Sens=r.Sens,Spec=r.Spec,Brier=r.Brier,Lat_ms=lat.get(r.model),Tam_MB=mb.get(r.model),AUC_ingenuo=r.naive_AUC))
if os.path.exists(O+'fusion_results.csv'):
    fu=pd.read_csv(O+'fusion_results.csv')
    for _,r in fu.iterrows():
        rows.append(dict(Modelo=r.model,Protocolo='limpio (471)',AUC=r.AUC,AUC_IC=ci(eval(r.AUC_ci)),F1=r.F1,Acc=r.Acc,Sens=r.Sens,Spec=r.Spec,Brier=r.Brier,Lat_ms=r.emb_ms+6,Tam_MB=None,AUC_ingenuo=None))
for f in sorted(glob.glob(O+'cnn_*_clean.json')):
    j=json.load(open(f)); r=j['res']
    for q,lab in [('fp32','FP32'),('fp16','FP16'),('dyn','INT8 dinámico'),('int8','INT8 completo')]:
        if q not in r: continue
        m=r[q]; rows.append(dict(Modelo=f"{r['model']} ({lab})",Protocolo='limpio (471)',AUC=m['AUC'],AUC_IC=ci(r[q+'_ci']['AUC']),F1=m['F1'],Acc=m['Acc'],Sens=m['Sens'],Spec=m['Spec'],Brier=m['Brier'],Lat_ms=r.get(q+'_lat_ms'),Tam_MB=r.get(q+'_MB'),AUC_ingenuo=None))
for f in sorted(glob.glob(O+'cnn_*_naive.json')):
    j=json.load(open(f)); r=j['res']; m=r['fp32']
    rows.append(dict(Modelo=f"{r['model']} (FP32)",Protocolo='ingenuo (708)',AUC=m['AUC'],AUC_IC=ci(r['fp32_ci']['AUC']),F1=m['F1'],Acc=m['Acc'],Sens=m['Sens'],Spec=m['Spec'],Brier=m['Brier'],Lat_ms=None,Tam_MB=None,AUC_ingenuo=None))
T=pd.DataFrame(rows); T.to_csv(O+'tabla_comparativa.csv',index=False)
pd.set_option('display.width',250); print(T.round(3).to_string(index=False))
