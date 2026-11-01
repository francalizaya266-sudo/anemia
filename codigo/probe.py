import sys, os; sys.path.insert(0,'/home/claude/exp'); os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import numpy as np, json, cv2, tensorflow as tf
from common import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, StratifiedKFold
A=tf.keras.applications
d=dedup(load_meta()); y=d.anemia.values
X=np.stack([letterbox(read_rgb(p),224) for p in d.image_path]).astype(np.float32)
def tight(im):
    m=mask_of(im); ys,xs=np.where(m); im=im[ys.min():ys.max()+1, xs.min():xs.max()+1]
    return cv2.resize(im,(224,224),interpolation=cv2.INTER_AREA)
Xt=np.stack([tight(read_rgb(p)) for p in d.image_path]).astype(np.float32)
out={}
for nm,f,pre in [('MobileNetV3Small',A.MobileNetV3Small,lambda x:x),('MobileNetV2',A.MobileNetV2,A.mobilenet_v2.preprocess_input),('EfficientNetB0',A.EfficientNetB0,lambda x:x),('ResNet50',A.ResNet50,A.resnet50.preprocess_input)]:
    base=f(include_top=False,weights='imagenet',input_shape=(224,224,3),pooling='avg')
    for tag,XX in [('letterbox',X),('tight',Xt)]:
        E=base.predict(pre(XX.copy()),batch_size=32,verbose=0)
        clf=make_pipeline(StandardScaler(),LogisticRegression(C=0.05,max_iter=3000,class_weight='balanced'))
        p=cross_val_predict(clf,E,y,cv=StratifiedKFold(5,shuffle=True,random_state=SEED),method='predict_proba')[:,1]
        out[f'{nm}_{tag}']=metrics(y,p)['AUC']; print(nm,tag,'linear-probe AUC %.3f'%out[f'{nm}_{tag}'],flush=True)
json.dump(out,open('/home/claude/exp/out/probe.json','w'),indent=1)
