# Supplementary Fig. S7: examples of the fourth audit stage (SIFT + RANSAC + photometric verification).
# (a) accepted new pairs at evenly spaced quantiles of the NCC (not hand-picked); (b) rejected candidates closest to the thresholds.
import sys; sys.path.insert(0,'/home/claude/exp')
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from common import read_rgb
plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.size':6,'font.family':'DejaVu Sans','savefig.bbox':'tight','figure.dpi':200})
R2='/home/claude/exp/out/review2/'
G=np.load(R2+'geomverify.npy'); arr=np.load(R2+'orb_pairs.npy'); ncand=int((arr[:,3]>=10).sum())
mf=pd.read_csv('/home/claude/anemia/codigo/meta_groups.csv'); ms=pd.read_csv('/home/claude/anemia/codigo/meta_groups_strict.csv').set_index('image_id')
reps=mf.sort_values('image_id').drop_duplicates('grp').reset_index(drop=True); sg=ms.loc[reps.image_id,'grp'].values
i=G[:,0].astype(int); j=G[:,1].astype(int); inl=G[:,2]; cov=G[:,3]; ncc=G[:,4]; fl=G[:,5]; sc=G[:,6]
cand=np.arange(len(G))<ncand; pos=sg[i]==sg[j]
acc=cand&(inl>=10)&(ncc>=0.70)&(sc>=0.5)&(sc<=2)&(cov>=0.30)
new=np.where(acc&~pos)[0]; new=new[np.argsort(ncc[new])]; pick=[new[int(q*(len(new)-1))] for q in np.linspace(0,1,12)]
rej=np.where(cand&~acc&~pos&(sc>=0.5)&(sc<=2)&(inl>=6))[0]; rej=rej[np.argsort(-ncc[rej])][:4]
def show(ax,k,title):
    a=read_rgb(reps.image_path[i[k]]); b=read_rgb(reps.image_path[j[k]]); h=120
    import cv2
    a=cv2.resize(a,(int(a.shape[1]*h/a.shape[0]),h)); b=cv2.resize(b,(int(b.shape[1]*h/b.shape[0]),h))
    c=np.full((h,a.shape[1]+b.shape[1]+8,3),255,np.uint8); c[:,:a.shape[1]]=a; c[:,a.shape[1]+8:]=b
    ax.imshow(c); ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]; ax.set_title(title,fontsize=5.6,pad=1.5)
fig=plt.figure(figsize=(6.6,4.3)); gs=fig.add_gridspec(4,4,hspace=1.05,wspace=0.06,top=0.88,bottom=0.02)
for n,k in enumerate(pick):
    a,b=reps.iloc[i[k]],reps.iloc[j[k]]
    show(fig.add_subplot(gs[n//4,n%4]),k,f"#{a.image_id[6:]} vs #{b.image_id[6:]}{' (mirror)' if fl[k] else ''}\nNCC {ncc[k]:.2f}, {int(inl[k])} inliers, scale {sc[k]:.2f}\nHb {a.hb:.1f} vs {b.hb:.1f}; {'same' if a.hospital==b.hospital else 'different'} facility")
for n,k in enumerate(rej):
    a,b=reps.iloc[i[k]],reps.iloc[j[k]]
    show(fig.add_subplot(gs[3,n%4]),k,f"#{a.image_id[6:]} vs #{b.image_id[6:]} (rejected)\nNCC {ncc[k]:.2f}, {int(inl[k])} inliers, overlap {cov[k]:.2f}")
fig.text(0.125,0.935,'(a) Accepted new pairs at evenly spaced quantiles of the NCC (lowest top left)',fontsize=7)
fig.text(0.125,0.215,'(b) Rejected candidates with the highest NCC (scale 0.5-2, at least 6 inliers after verification)',fontsize=7)
for ext in ('pdf','png'): plt.savefig('/home/claude/figs_new/figS7_geometric_pairs.'+ext)
print(len(new),len(rej))
