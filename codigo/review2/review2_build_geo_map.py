# Builds the geometric duplicate map (codigo/meta_groups_geo.csv) from the stage-4 search (review2_orb.py) and its photometric
# verification (review2_geomverify.py). Accept rule: >= 10 RANSAC inliers, NCC >= 0.70, overlap >= 0.30, scale 0.5-2.
# Run from /home/claude/exp (paths relative to out/review2/); writes out/review2/geo_map_summary.json.
import numpy as np, pandas as pd, json
G=np.load('out/review2/geomverify.npy'); arr=np.load('out/review2/orb_pairs.npy'); ncand=int((arr[:,3]>=10).sum())
mfull=pd.read_csv('/home/claude/anemia/codigo/meta_groups.csv'); ms=pd.read_csv('/home/claude/anemia/codigo/meta_groups_strict.csv')
reps=mfull.sort_values('image_id').drop_duplicates('grp').reset_index(drop=True); sgr=ms.set_index('image_id').loc[reps.image_id,'grp'].values
i=G[:,0].astype(int); j=G[:,1].astype(int); inl=G[:,2]; cov=G[:,3]; ncc=G[:,4]; fl=G[:,5]; sc=G[:,6]
cand=np.arange(len(G))<ncand; pos=sgr[i]==sgr[j]
acc=cand&(inl>=10)&(ncc>=0.70)&(sc>=0.5)&(sc<=2.0)&(cov>=0.30)
np.save('/tmp/geo_accept.npy',np.c_[i[acc],j[acc],inl[acc],cov[acc],ncc[acc],fl[acc],sc[acc],pos[acc]])
new=acc&~pos; a=i[new]; b=j[new]; la=reps.anemia.values
print('accepted',acc.sum(),'stage3',(acc&pos).sum(),'new',new.sum(),'concordant',(la[a]==la[b]).sum())
sg=ms.set_index('image_id').grp; par={}
def f(x):
    par.setdefault(x,x)
    while par[x]!=x: par[x]=par[par[x]]; x=par[x]
    return x
for g in ms.grp.unique(): f(g)
for p,q in zip(a,b):
    ra,rb=f(sg[reps.image_id[p]]),f(sg[reps.image_id[q]])
    if ra!=rb: par[ra]=rb
geo=ms.copy(); geo['grp']=pd.factorize(geo.grp.map(f))[0]
conf=geo.groupby('grp').anemia.nunique(); d=geo[~geo.grp.isin(conf[conf>1].index)]; dd=d.sort_values('image_id').drop_duplicates('grp')
sizes=geo.groupby('grp').size(); dup=geo[geo.grp.isin(sizes[sizes>1].index)]; sing=geo[geo.grp.isin(sizes[sizes==1].index)]
rng=np.random.default_rng(42); obs=(la[a]==la[b]).sum(); null=np.array([(lambda p_: (p_[a]==p_[b]).sum())(rng.permutation(la)) for _ in range(20000)])
S=dict(n_new_pairs=int(new.sum()),concordant=int(obs),null_mean=float(null.mean()),null_max=int(null.max()),p=float((1+(null>=obs).sum())/20001),
  flips=int(fl[new].sum()),rescaled=int((np.abs(sc[new]-1)>0.03).sum()),hosp_differ=int((reps.hospital.values[a]!=reps.hospital.values[b]).sum()),hb_differ=int((reps.hb.values[a]!=reps.hb.values[b]).sum()),
  geo_unique=int(geo.grp.nunique()),label_conflicts=int((conf>1).sum()),n_class=int(len(dd)),n_anaemic=int(dd.anemia.sum()),n_non=int((1-dd.anemia).sum()),singletons=int((sizes==1).sum()),multi=int((sizes>1).sum()),mrows=int(sizes[sizes>1].sum()),
  dup_anaemic_pct=float(100*dup.anemia.mean()),single_anaemic_pct=float(100*sing.anemia.mean()),largest=int(sizes.max()),criteria='inliers>=10, NCC>=0.70, overlap>=0.30, scale 0.5-2')
print(S); json.dump(S,open('out/review2/geo_map_summary.json','w'),indent=1)
geo.to_csv('/home/claude/anemia/codigo/meta_groups_geo.csv',index=False)
