# Rebuild Fig 6 from the GPU-rendered PDF: recover each Grad-CAM (7x7 grid, bilinear) from the composited panels,
# verify against the per-image ratios computed on the laptop, and re-render with legible labels.
import json, os, numpy as np, cv2, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
S=224; D='/home/claude/data/cp-anemic/'; IM='/tmp/fig6img/'; OUT='/home/claude/figs_new/'
plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.family':'DejaVu Sans','font.size':7,'savefig.bbox':'tight','figure.dpi':200})
def letterbox(im, s=224):
    h,w=im.shape[:2]; r=s/max(h,w); nh,nw=max(1,round(h*r)),max(1,round(w*r))
    im=cv2.resize(im,(nw,nh),interpolation=cv2.INTER_AREA); out=np.zeros((s,s,3),np.uint8); y=(s-nh)//2; x=(s-nw)//2; out[y:y+nh,x:x+nw]=im; return out
def mask_of(im,t=12): return im.max(axis=2)>t
def tight(im,s=S):
    m=mask_of(im); ys,xs=np.where(m)
    if len(ys)<10: return letterbox(im,s)
    return letterbox(im[ys.min():ys.max()+1, xs.min():xs.max()+1], s)
def read_rgb(p):
    im=cv2.imread(D+p, cv2.IMREAD_UNCHANGED)
    if im.ndim==3 and im.shape[2]==4: im=im[:,:,:3]
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
meta=pd.read_csv('/home/claude/anemia/codigo/meta_groups.csv').set_index('image_id')
G=json.load(open('/home/claude/anemia/gpu/out_review/gradcam_review.json')); SUM=G['summary']; EX=G['examples']
vir=plt.get_cmap('viridis')(np.linspace(0,1,256))[:,:3]
def vir_inv(rgb):  # nearest colormap entry -> value in [0,1]
    d=((rgb[...,None,:]-vir[None,None,:,:])**2).sum(-1); return np.argmin(d,axis=-1)/255.0
_BC={}
def basis(h, w, n=7):
    if (h,w) not in _BC:
        B=[]
        for i in range(n*n):
            g=np.zeros((n,n),np.float32); g.flat[i]=1; B.append(cv2.resize(g,(w,h),interpolation=cv2.INTER_LINEAR).ravel())
        _BC[(h,w)]=np.array(B).T
    return _BC[(h,w)]
ALPHA=0.55*0.55  # imshow(alpha=0.55) on an RGBA colormap image composited by Agg: the alpha is applied twice
def fit_cam(mi,k,scheme):
    name=['EfficientNet-B0','MobileNetV2'][mi]; e=EX[name][k]; roi224=tight(read_rgb(meta.loc[e['image_id'],'image_path']))
    top=cv2.cvtColor(cv2.imread(IM+f'im-{mi*16+2*k:03d}.png'),cv2.COLOR_BGR2RGB).astype(np.float32)/255
    comp=cv2.cvtColor(cv2.imread(IM+f'im-{mi*16+2*k+1:03d}.png'),cv2.COLOR_BGR2RGB).astype(np.float32)/255
    h,w=comp.shape[:2]; top=cv2.resize(top,(w,h))
    roi_small=cv2.resize(roi224.astype(np.float32)/255,(w,h),interpolation=cv2.INTER_AREA); corr=np.corrcoef(top.ravel(),roi_small.ravel())[0,1]
    virrgb=np.clip((comp-(1-ALPHA)*top)/ALPHA,0,1); cam_obs=vir_inv(virrgb)
    if cam_obs.max()<0.1 and e['ratio']==0:  # identically-zero (rectified) map: nothing to reconstruct
        return roi224,np.zeros((S,S),np.float32),mask_of(roi224),e,0.0,float(np.corrcoef(top.ravel(),roi_small.ravel())[0,1])
    fg=(top.max(2)>=0.05).astype(np.uint8); band=cv2.dilate(fg,np.ones((5,5),np.uint8))-cv2.erode(fg,np.ones((3,3),np.uint8))
    wts={'bg':np.where(fg==0,1.0,0.0),'in05':np.where(band>0,0.2,np.where(fg>0,0.5,1.0)),'in1':np.ones(fg.shape)}[scheme].ravel()
    Bm=basis(h,w); coef,*_=np.linalg.lstsq(Bm*wts[:,None],cam_obs.ravel()*wts,rcond=None)
    grid=coef.reshape(7,7); cam=cv2.resize(grid.astype(np.float32),(S,S),interpolation=cv2.INTER_LINEAR); cam=np.clip(cam,0,None); cam=(cam-cam.min())/(cam.max()-cam.min()+1e-8)
    m=mask_of(roi224); ratio=(cam*m).sum()/(cam.sum()+1e-8)/m.mean()
    return roi224,cam,m,e,float(ratio),float(corr)
allres={scheme:{(mi,k):fit_cam(mi,k,scheme) for mi in (0,1) for k in range(8)} for scheme in ['bg','in05','in1']}
res={}; chosen={}
for key in allres['in1']:
    sc=min(allres,key=lambda s_: abs(allres[s_][key][4]-allres[s_][key][3]['ratio'])); res[key]=allres[sc][key]; chosen[key]=sc
err=float(np.mean([abs(v[4]-v[3]['ratio']) for v in res.values()])); scheme='per-panel'
print('per-panel choice: mean abs ratio error %.3f max %.3f'%(err,max(abs(v[4]-v[3]['ratio']) for v in res.values())))
for (mi,k),v in sorted(res.items()): print(['EfficientNet-B0','MobileNetV2'][mi],k,v[3]['kind'],v[3]['image_id'],'true %.2f recon %.2f (%s)'%(v[3]['ratio'],v[4],chosen[(mi,k)]))
cams={n:[res[(mi,k)][:4] for k in range(8)] for mi,n in enumerate(['EfficientNet-B0','MobileNetV2'])}
json.dump({'alpha':ALPHA,'scheme':scheme,'mean_abs_ratio_error':err,'per_panel':[{'model':['EfficientNet-B0','MobileNetV2'][mi],'k':k,'image_id':v[3]['image_id'],'ratio_true':v[3]['ratio'],'ratio_recon':v[4]} for (mi,k),v in sorted(res.items())]},open(OUT+'gc/fig6_recon_check.json','w'),indent=1)
# ---------------- render ----------------
fig=plt.figure(figsize=(6.7,5.0)); gs=fig.add_gridspec(2,1,hspace=0.42,left=0.06,right=0.90,top=0.90,bottom=0.02)
for pi,name in enumerate(['EfficientNet-B0','MobileNetV2']):
    sub=gs[pi].subgridspec(2,8,wspace=0.06,hspace=0.04)
    for k,(roi,cam,m,e) in enumerate(cams[name]):
        a=fig.add_subplot(sub[0,k]); a.imshow(roi); a.set_xticks([]); a.set_yticks([])
        a.set_title(f"{e['kind']} · Hb {e['hb_gL']:.0f}\np {e['p']:.2f} · "+("flat map" if cam.max()==0 else f"ratio {e['ratio']:.1f}"),fontsize=5.6,pad=2.2,linespacing=1.15)
        b=fig.add_subplot(sub[1,k]); b.imshow(roi); b.imshow(cam,cmap='viridis',alpha=0.55,vmin=0,vmax=1)
        b.contour(m.astype(float),levels=[0.5],colors='white',linewidths=0.5); b.set_xticks([]); b.set_yticks([])
        for sp in list(a.spines.values())+list(b.spines.values()): sp.set_visible(False)
        if k==0: a.set_ylabel('ROI',fontsize=6.5); b.set_ylabel('Grad-CAM',fontsize=6.5)
    s=SUM[name]
    fig.text(0.06,0.945 if pi==0 else 0.465,f"({'ab'[pi]}) {name}: mean in-mask ratio {s['ratio_mean']:.2f} (95 % CI {s['ratio_ci'][0]:.2f}–{s['ratio_ci'][1]:.2f}); ratio > 1 in {100*s['frac_ratio_gt1']:.0f} % of the 471 images",fontsize=7,ha='left',va='bottom')
sm=plt.cm.ScalarMappable(cmap='viridis'); sm.set_array([0,1]); cax=fig.add_axes([0.915,0.25,0.012,0.45]); cb=fig.colorbar(sm,cax=cax); cb.set_label('Grad-CAM (normalised)',fontsize=6.5); cb.ax.tick_params(labelsize=6)
plt.savefig(OUT+'Fig6_gradcam.pdf'); plt.savefig(OUT+'Fig6_gradcam.png',dpi=200); print('saved')
