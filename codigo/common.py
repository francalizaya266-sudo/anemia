import numpy as np, pandas as pd, cv2
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, recall_score, brier_score_loss
D='/home/claude/data/cp-anemic/'
SEED=42

def load_meta():
    d=pd.read_csv('/home/claude/sr/meta_groups.csv')
    conf=d.groupby('grp').anemia.nunique(); d=d[~d.grp.isin(conf[conf>1].index)]  # drop label-conflicting group
    return d.reset_index(drop=True)

def dedup(d):
    # one representative per near-duplicate group (first by image id)
    return d.sort_values('image_id').drop_duplicates('grp').reset_index(drop=True)

def read_rgb(p):
    im=cv2.imread(D+p, cv2.IMREAD_UNCHANGED)
    if im.ndim==3 and im.shape[2]==4: im=im[:,:,:3]
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

def letterbox(im, s=224):
    h,w=im.shape[:2]; r=s/max(h,w); nh,nw=max(1,round(h*r)),max(1,round(w*r))
    im=cv2.resize(im,(nw,nh),interpolation=cv2.INTER_AREA)
    out=np.zeros((s,s,3),np.uint8); y=(s-nh)//2; x=(s-nw)//2; out[y:y+nh,x:x+nw]=im; return out

def mask_of(im, t=12):
    return im.max(axis=2)>t

def folds(y, groups=None, k=5, seed=SEED):
    if groups is None:
        return list(StratifiedKFold(k,shuffle=True,random_state=seed).split(np.zeros(len(y)),y))
    return list(StratifiedGroupKFold(k,shuffle=True,random_state=seed).split(np.zeros(len(y)),y,groups))

def metrics(y,p,t=0.5):
    yh=(p>=t).astype(int)
    return dict(AUC=roc_auc_score(y,p),F1=f1_score(y,yh),Acc=accuracy_score(y,yh),
                Sens=recall_score(y,yh),Spec=recall_score(1-y,1-yh),Brier=brier_score_loss(y,p))

def boot_ci(y,p,B=2000,seed=SEED,t=0.5):
    rng=np.random.default_rng(seed); n=len(y); res={k:[] for k in metrics(y,p,t)}
    for _ in range(B):
        i=rng.integers(0,n,n)
        if len(np.unique(y[i]))<2: continue
        for k,v in metrics(y[i],p[i],t).items(): res[k].append(v)
    return {k:(np.percentile(v,2.5),np.percentile(v,97.5)) for k,v in res.items()}
