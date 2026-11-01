# Contact sheets of candidate pairs of the stage-4 search for visual inspection.
import sys; sys.path.insert(0,'/home/claude/exp')
import numpy as np, pandas as pd, cv2, json
from common import read_rgb
mfull=pd.read_csv('/home/claude/anemia/codigo/meta_groups.csv'); ms=pd.read_csv('/home/claude/anemia/codigo/meta_groups_strict.csv').set_index('image_id')
reps=mfull.sort_values('image_id').drop_duplicates('grp').reset_index(drop=True)
arr=np.load('/home/claude/exp/out/review2/orb_pairs.npy')
sgr=ms.loc[reps.image_id,'grp'].values
pos1=sgr[arr[:,0].astype(int)]==sgr[arr[:,1].astype(int)]
inl=arr[:,3]; ratio=inl/np.minimum(arr[:,4],arr[:,5]).clip(1)
order=np.argsort(-inl*(~pos1))
print(reps.columns.tolist())
def tile(k):
    i,j=int(arr[k,0]),int(arr[k,1]); ims=[]
    for t in (i,j):
        im=read_rgb(reps.image_path[t]); h=120; im=cv2.resize(im,(int(im.shape[1]*h/im.shape[0]),h)); ims.append(im)
    w=sum(x.shape[1] for x in ims)+6; c=np.full((150,max(w,300),3),255,np.uint8); x=0
    for im in ims: c[0:120,x:x+im.shape[1]]=im; x+=im.shape[1]+6
    a,b=reps.iloc[i],reps.iloc[j]
    txt=f"{a.image_id[6:]}({int(a.anemia)},{a.hospital[:4]},hb{a.hb}) {b.image_id[6:]}({int(b.anemia)},{b.hospital[:4]},hb{b.hb}) in={int(inl[k])} r={ratio[k]:.2f}"
    cv2.putText(c,txt,(2,142),cv2.FONT_HERSHEY_SIMPLEX,0.36,(0,0,0),1); return c
def grid(ks,fn,cols=4):
    ts=[tile(k) for k in ks]; W=max(t.shape[1] for t in ts)
    ts=[np.pad(t,((0,0),(0,W-t.shape[1]),(0,0)),constant_values=255) for t in ts]
    rows=[np.hstack(ts[r:r+cols]+[np.full_like(ts[0],255)]*(cols-len(ts[r:r+cols]))) for r in range(0,len(ts),cols)]
    cv2.imwrite(fn,cv2.cvtColor(np.vstack(rows),cv2.COLOR_RGB2BGR))
grid(order[:24],'/tmp/top24.png')
neg=np.where(~pos1)[0]
for lo,hi,name in [(50,120,'b50'),(25,50,'b25'),(15,25,'b15')]:
    sel=neg[(inl[neg]>=lo)&(inl[neg]<hi)]; rng=np.random.default_rng(0); sel=rng.choice(sel,min(16,len(sel)),replace=False)
    grid(sel,f'/tmp/{name}.png'); print(name,((inl[neg]>=lo)&(inl[neg]<hi)).sum())
# stage3 controls
grid(np.where(pos1)[0][np.argsort(inl[pos1])][:12],'/tmp/stage3.png')
def tile2(k,h=200):
    i,j=int(arr[k,0]),int(arr[k,1]); ims=[]
    for t in (i,j):
        im=read_rgb(reps.image_path[t]); im=cv2.resize(im,(int(im.shape[1]*h/im.shape[0]),h)); ims.append(im)
    W=ims[0].shape[1]+ims[1].shape[1]+10; c=np.full((h+24,W,3),255,np.uint8); c[:h,:ims[0].shape[1]]=ims[0]; c[:h,ims[0].shape[1]+10:]=ims[1]
    a,b=reps.iloc[i],reps.iloc[j]
    cv2.putText(c,f"{a.image_id[6:]}(a{int(a.anemia)},{a.hospital[:5]},{a.hb},{a.age_months}m) | {b.image_id[6:]}(a{int(b.anemia)},{b.hospital[:5]},{b.hb},{b.age_months}m) in={int(inl[k])}",(2,h+17),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
    return c
for n0 in (0,6):
    ts=[tile2(k) for k in order[n0:n0+6]]; W=max(t.shape[1] for t in ts)
    ts=[np.pad(t,((0,0),(0,W-t.shape[1]),(0,0)),constant_values=255) for t in ts]
    cv2.imwrite(f'/tmp/topz{n0}.png',cv2.cvtColor(np.vstack(ts),cv2.COLOR_RGB2BGR))
