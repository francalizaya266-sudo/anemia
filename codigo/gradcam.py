import sys, os; sys.path.insert(0,'/home/claude/exp'); os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import numpy as np, json, cv2, tensorflow as tf
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from common import *
K=tf.keras
name=sys.argv[1]
m=K.models.load_model(f'/home/claude/exp/out/{name}_lastfold.keras',compile=False,safe_mode=False)
# locate backbone (functional submodel) and its last 4D conv output
base=[l for l in m.layers if isinstance(l,K.Model) and l.name!='aug'][0]
conv=[l for l in base.layers if len(l.output.shape)==4][-1]
inner=K.Model(base.input,[conv.output,base.output])
head=[l for l in m.layers if isinstance(l,K.layers.Dense)][0]
pre=[l for l in m.layers if l.name not in('aug',) and not isinstance(l,(K.Model,K.layers.Dense,K.layers.Dropout,K.layers.InputLayer))]

def forward(x):
    for l in pre: x=l(x)
    conv_out,feat=inner(x,training=False)
    return conv_out,head(feat)

def gradcam(x):
    x=tf.convert_to_tensor(x[None])
    with tf.GradientTape() as t:
        t.watch(x); c,p=forward(x); s=p[:,0]
    g=t.gradient(s,c)[0]; w=tf.reduce_mean(g,axis=(0,1)); cam=tf.nn.relu(tf.reduce_sum(c[0]*w,-1)).numpy()
    cam=cv2.resize(cam,(224,224)); cam=(cam-cam.min())/(cam.max()-cam.min()+1e-8); return cam,float(p[0,0])

d=dedup(load_meta()); y=d.anemia.values
# use the last fold's test indices so the examples are unseen by this model
tr,te=folds(y,None,seed=SEED)[-1]
from cnn import tight
X=np.stack([tight(read_rgb(p),224) for p in d.image_path[te]]).astype(np.float32); yt=y[te]; hb=d.hb.values[te]
cams=[];ps=[]
for x in X: c,p=gradcam(x); cams.append(c); ps.append(p)
cams=np.array(cams); ps=np.array(ps)
# quantitative: fraction of CAM mass inside the conjunctiva mask
masks=np.array([mask_of(x.astype(np.uint8)) for x in X])
inside=(cams*masks).sum((1,2))/(cams.sum((1,2))+1e-8); area=masks.mean((1,2))
res={'model':name,'n_test':len(te),'cam_mass_inside_mask_mean':float(inside.mean()),'mask_area_mean':float(area.mean()),'ratio_vs_chance':float((inside/area).mean())}
# figure: 4 TP, 4 TN (most confident correct) + 2 errors
order=np.argsort(-np.abs(ps-0.5))
tp=[i for i in order if yt[i]==1 and ps[i]>=0.5][:3]; tn=[i for i in order if yt[i]==0 and ps[i]<0.5][:3]
err=[i for i in order if (ps[i]>=0.5)!=yt[i]][:2]; sel=tp+tn+err
fig,ax=plt.subplots(2,len(sel),figsize=(2.2*len(sel),4.6))
for k,i in enumerate(sel):
    im=X[i].astype(np.uint8); ax[0,k].imshow(im); ax[0,k].axis('off')
    ax[0,k].set_title(f"Hb {hb[i]:.1f} · y={yt[i]} · p={ps[i]:.2f}",fontsize=8)
    heat=cv2.applyColorMap((cams[i]*255).astype(np.uint8),cv2.COLORMAP_JET)[:,:,::-1]
    ax[1,k].imshow((0.55*im+0.45*heat).astype(np.uint8)); ax[1,k].axis('off')
ax[0,0].set_ylabel('imagen'); ax[1,0].set_ylabel('Grad-CAM')
plt.suptitle(f'Grad-CAM · {name} · pliegue de prueba (no visto en entrenamiento)',fontsize=10); plt.tight_layout()
plt.savefig(f'/home/claude/exp/out/gradcam_{name}.png',dpi=150); json.dump(res,open(f'/home/claude/exp/out/gradcam_{name}.json','w'),indent=1)
print(json.dumps(res,indent=1))
