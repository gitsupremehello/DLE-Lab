import kagglehub

path = kagglehub.model_download("tensorflow/ssd-mobilenet-v2/tensorFlow2/ssd-mobilenet-v2")

print("Path to model files:", path)

import tensorflow as tf
import tensorflow_hub as hub
import cv2,matplotlib.pyplot as plt
#r"C:\Users\admin\.cache\kagglehub\models\tensorflow\ssd-mobilenet-v2\tensorFlow2\ssd-mobilenet-v2\1"
model_path=r"C:\Users\yaswanth\.cache\kagglehub\models\tensorflow\ssd-mobilenet-v2\tensorFlow2\ssd-mobilenet-v2\1"
detector=hub.load(model_path)

def detect_objects(image_path):
    img=cv2.imread(image_path)
    img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_resized=tf.image.resize(img_rgb,(320,320))
    img_resized=tf.cast(img_resized,tf.uint8)
    result=detector(tf.expand_dims(img_resized,0))
    result={k:v.numpy() for k,v in result.items()}
    objs=[]
    h,w=img.shape[:2]
    for b,s,c in zip(result["detection_boxes"][0],result["detection_scores"][0],result["detection_classes"][0]):
        if s>0.5:
            y1,x1,y2,x2=b
            x1,y1,x2,y2=int(x1*w),int(y1*h),int(x2*w),int(y2*h)
            objs.append({'class_id':int(c),'score':float(s),'box':[x1,y1,x2,y2]})
            cv2.rectangle(img_rgb,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(img_rgb,f"ID:{int(c)} {s:.2f}",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
    return img_rgb,objs

image_path="sample.jpeg"
img_with_boxes,detected=detect_objects(image_path)
plt.imshow(img_with_boxes)
plt.axis("off")
plt.show()
print(detected)
