import numpy as np
from sklearn.metrics import roc_auc_score
from scipy import stats
SEED=42
def thr_for_sens(y,p,target=0.90):
    # largest threshold whose sensitivity >= target
    ts=np.unique(p)[::-1]
    for t in ts:
        if ((p>=t)&(y==1)).sum()/(y==1).sum()>=target: return float(t)
    return float(ts[-1])

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
    tx=np.array([_midrank(preds[r,pos]) for r in range(k)]); ty=np.array([_midrank(preds[r,neg]) for r in range(k)]); tz=np.array([_midrank(preds[r]) for r in range(k)])
    aucs=tz[:,:m].sum(1)/m/n-(m+1.0)/(2.0*n)
    v01=(tz[:,:m]-tx)/n; v10=1.0-(tz[:,m:]-ty)/m
    S=np.cov(v01)/m+np.cov(v10)/n
    L=np.array([1,-1]); z=(aucs[0]-aucs[1])/np.sqrt(L@S@L); pval=2*stats.norm.sf(abs(z))
    return float(aucs[0]-aucs[1]), float(z), float(pval)
