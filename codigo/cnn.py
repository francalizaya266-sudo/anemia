import sys, os; sys.path.insert(0,'/home/claude/exp'); os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import numpy as np, pandas as pd, json, time
import tensorflow as tf
from ai_edge_litert.interpreter import Interpreter, OpResolverType
from common import *
from sklearn.model_selection import train_test_split
K=tf.keras; A=K.applications
tf.random.set_seed(SEED); np.random.seed(SEED)
ARCH={'MobileNetV3-Small':(A.MobileNetV3Small,None),'MobileNetV3-Large':(A.MobileNetV3Large,None),
      'MobileNetV2':(A.MobileNetV2,A.mobilenet_v2.preprocess_input),'EfficientNet-B0':(A.EfficientNetB0,None),
      'EfficientNetV2-B0':(A.EfficientNetV2B0,None),'ResNet50':(A.ResNet50,A.resnet50.preprocess_input)}
S=224
def build(name):
    f,pre=ARCH[name]
    base=f(include_top=False,weights='imagenet',input_shape=(S,S,3),pooling='avg')
    inp=K.Input((S,S,3)); x=inp
    # geometric augmentation only: brightness/contrast jitter would destroy the colour signal that defines pallor
    aug=K.Sequential([K.layers.RandomFlip('horizontal_and_vertical'),K.layers.RandomRotation(0.05,fill_mode='constant'),
                      K.layers.RandomZoom(0.1,fill_mode='constant'),K.layers.RandomTranslation(0.05,0.05,fill_mode='constant')],name='aug')
    x=aug(x)
    if pre is not None: x=K.layers.Lambda(pre,name='pre')(x) if False else pre(x)
    x=base(x,training=False); x=K.layers.Dropout(0.3)(x); out=K.layers.Dense(1,activation='sigmoid')(x)
    return K.Model(inp,out),base

def tight(im,s=S):
    m=mask_of(im); ys,xs=np.where(m)
    if len(ys)<10: return letterbox(im,s)
    im=im[ys.min():ys.max()+1, xs.min():xs.max()+1]; return letterbox(im,s)
def train(name,Xtr,ytr,epochs1=10,epochs2=40,verbose=0):
    Xt,Xv,yt,yv=train_test_split(Xtr,ytr,test_size=0.15,stratify=ytr,random_state=SEED)
    w={0:len(yt)/(2*(yt==0).sum()),1:len(yt)/(2*(yt==1).sum())}
    m,base=build(name); base.trainable=False
    es=lambda:[K.callbacks.EarlyStopping('val_auc',mode='max',patience=8,restore_best_weights=True)]
    m.compile(K.optimizers.Adam(1e-3),'binary_crossentropy',metrics=[K.metrics.AUC(name='auc')])
    m.fit(Xt,yt,validation_data=(Xv,yv),epochs=epochs1,batch_size=32,class_weight=w,callbacks=es(),verbose=verbose)
    base.trainable=True; n=len(base.layers)
    for i,l in enumerate(base.layers):
        l.trainable = not isinstance(l,K.layers.BatchNormalization)   # full fine-tune, BN frozen
    m.compile(K.optimizers.Adam(5e-5),'binary_crossentropy',metrics=[K.metrics.AUC(name='auc')])
    m.fit(Xt,yt,validation_data=(Xv,yv),epochs=epochs2,batch_size=32,class_weight=w,callbacks=es(),verbose=verbose)
    return m

def to_tflite(m,mode,calib=None):
    # strip augmentation for export: rebuild inference graph
    inp=K.Input((S,S,3)); x=inp
    for l in m.layers[1:]:
        if l.name=='aug': continue
        x=l(x)
    inf=K.Model(inp,x)
    c=tf.lite.TFLiteConverter.from_keras_model(inf)
    if mode=='fp16': c.optimizations=[tf.lite.Optimize.DEFAULT]; c.target_spec.supported_types=[tf.float16]
    if mode=='dyn': c.optimizations=[tf.lite.Optimize.DEFAULT]
    if mode=='int8':
        c.optimizations=[tf.lite.Optimize.DEFAULT]
        c.representative_dataset=lambda:([calib[i:i+1].astype(np.float32)] for i in range(len(calib)))
    return c.convert()

XNN={}
def interp(b,threads):
    try:
        it=Interpreter(model_content=b,num_threads=threads); it.allocate_tensors(); XNN[len(b)]=True
    except RuntimeError:
        it=Interpreter(model_content=b,num_threads=threads,experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES); it.allocate_tensors(); XNN[len(b)]=False
    return it
def tfl_predict(b,X):
    it=interp(b,4)
    i=it.get_input_details()[0]; o=it.get_output_details()[0]; out=[]
    for x in X:
        it.set_tensor(i['index'],x[None].astype(np.float32)); it.invoke(); out.append(float(it.get_tensor(o['index'])[0,0]))
    return np.array(out)

def tfl_latency(b,X,n=100):
    it=interp(b,1)
    i=it.get_input_details()[0]
    x=X[0:1].astype(np.float32)
    for _ in range(10): it.set_tensor(i['index'],x); it.invoke()
    t=[]
    for _ in range(n):
        t0=time.perf_counter(); it.set_tensor(i['index'],x); it.invoke(); t.append((time.perf_counter()-t0)*1000)
    return float(np.median(t))

if __name__=='__main__':
    name=sys.argv[1]; proto=sys.argv[2] if len(sys.argv)>2 else 'clean'
    d=load_meta(); dd=dedup(d) if proto=='clean' else d
    X=np.stack([tight(read_rgb(p),S) for p in dd.image_path]).astype(np.float32); y=dd.anemia.values
    Q=['fp32','fp16','dyn','int8']; P={q:np.zeros(len(y)) for q in Q}
    t0=time.time(); last=None
    nf=int(os.environ.get('NFOLDS','5'))
    for k,(tr,te) in enumerate(folds(y,None,seed=SEED)[:nf]):
        m=train(name,X[tr],y[tr]); P['fp32'][te]=m.predict(X[te],verbose=0)[:,0]
        if proto=='clean':
            for q in Q[1:]: P[q][te]=tfl_predict(to_tflite(m,q,X[tr][:100]),X[te])
        print(name,proto,'fold',k,'AUC %.3f'%roc_auc_score(y[te],P['fp32'][te]),'%.0fs'%(time.time()-t0),flush=True)
        last=m
    res={'model':name,'proto':proto,'n':len(y),'train_time_s':time.time()-t0,'params_M':last.count_params()/1e6}
    for q in (Q if proto=='clean' else ['fp32']):
        res[q]=metrics(y,P[q]); res[q+'_ci']=boot_ci(y,P[q])
    if proto=='clean':
        for q in Q:
            b=to_tflite(last,q,X[:100]); res[q+'_MB']=len(b)/2**20; res[q+'_lat_ms']=tfl_latency(b,X); res[q+'_xnnpack']=XNN.get(len(b))
            open(f'/home/claude/exp/out/{name}_{q}.tflite','wb').write(b)
        last.save(f'/home/claude/exp/out/{name}_lastfold.keras')
    json.dump({'res':res,'y':y.tolist(),'oof':{k:v.tolist() for k,v in P.items()}},open(f'/home/claude/exp/out/cnn_{name}_{proto}.json','w'),default=float)
    print(json.dumps(res,default=float),flush=True)
