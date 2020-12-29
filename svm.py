import cv2
from sklearn.svm import SVC
from xgboost import XGBClassifier
import numpy as np
import h5py
import pickle
from sklearn.model_selection import cross_val_score

'''
svm训练
x_train  储存到h5文件里 只读取就行
y_train  为 0-4的标签只预测编号即可
'''

file=h5py.File('face.h5','r')
x_train = file['X_train'][:]
file.close()

y_train=[]
k = 0
for i in range(0,4):
	for j in range(1,105):
		y_train.append(i)

y_train = np.array(y_train)
for x in x_train:
	x = x.reshape(1,512)
print(x_train.shape)

svc = SVC(kernel='linear',probability=True)
xg =XGBClassifier(max_depth=2, learning_rate=1, n_estimators=1000,
                   silent=True, objective='binary:logistic')
svc_model=svc.fit(x_train,y_train)
xg_model=xg.fit(x_train,y_train)

with open("face_svm.pkl", 'wb') as outfile:
    pickle.dump((svc_model,5), outfile)
with open("face_xg.pkl", 'wb') as outfile:
    pickle.dump((xg_model,5), outfile)