#-*- coding: UTF-8 -*-
import cv2 as cv
import face_recognition
import numpy as np
import time
import pickle

name = ['1','2','3','4'] #预测的标签样本
model_path='models/20180408-102900' #face_net的路径
mtcnn = face_recognition.Facedetection() #导入mtcnn
face_net=face_recognition.facenetEmbedding(model_path)
with open('face_svm.pkl','rb') as infile:
    (classifymodel1, class_names1) = pickle.load(infile)
with open('face_xg.pkl', 'rb') as infile:
    (classifymodel2, class_names2) = pickle.load(infile)
'''
以下三个函数为人脸的裁剪和图像归一化处理
'''
def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y
def crop_image(image, box):
    crop_img= image[box[1]:box[3], box[0]:box[2]]
    return crop_img
def get_crop_images(image, boxes, resize_height=0, resize_width=0, whiten=False):
    crops=[]
    for box in boxes:
        crop_img=crop_image(image, box)
        if resize_height > 0 and resize_width > 0:
            crop_img = cv.resize(crop_img, (resize_width, resize_height))
        if whiten:
            crop_img = prewhiten(crop_img)
        crops.append(crop_img)
    crops=np.stack(crops)
    return crops


#视频处理
def RTrecognization():
    video = cv.VideoCapture(1)
    if(not video.isOpened()):
        exit(-1)
    video.set(3, 480)
    video.set(4, 640)
    while True:
        ret, frame = video.read()
        if not ret:
            continue
        img_rgb = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
        bounding_box, points = mtcnn.detect_face(img_rgb)
        if(len(bounding_box)>=1 and len(bounding_box[0])>1):
            bounding_box = bounding_box[:,0:4].astype(int)
            try:
                faces = get_crop_images(img_rgb,bounding_box, 160,160, True)
                cv.rectangle(frame,(bounding_box[0][0],bounding_box[0][1]),
                    (bounding_box[0][2],bounding_box[0][3]),(0,255,0),2,8,0)
                pred_emb = face_net.get_embedding(faces)
                #print(pred_emb[0])
                predictions1 = classifymodel1.predict_proba(np.array(pred_emb[0]).reshape(1,512))
                predictions2=classifymodel2.predict_proba(np.array(pred_emb[0]).reshape(1,512))
                predictions=(predictions1+predictions2)/2
                print(predictions)
                best_class_indices = np.argmax(predictions, axis=1)[0] #最大的索引
                perhaps = predictions[0][best_class_indices] * 100
                if(perhaps>65.0):
                    text = name[best_class_indices] + ' {:.2f} % '.format(perhaps)
                else:
                    text = 'Unknown'
                frame = cv.putText(frame, text,(bounding_box[0][0]-5,bounding_box[0][1]-5),
                    cv.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
            except:
                pass
        key = cv.waitKey(20) & 0xFF
        cv.imshow('carme',frame)
        if(key==ord('q')):
            break
    video.release()
    cv.destroyAllWindows()

#主函数
if __name__ == "__main__":
    RTrecognization()