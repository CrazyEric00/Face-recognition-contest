import numpy as np
from utils import image_processing , file_processing
import face_recognition
import cv2
import os
import h5py

'''
将分割好的人脸转为1*512的face_net编码
'''

def create_embedding(model_path, emb_face_dir):
	face_net = face_recognition.facenetEmbedding(model_path)
	image_list,names_list=file_processing.gen_files_labels(emb_face_dir,postfix='jpg')
	images= image_processing.get_images(image_list,160,160,whiten=True)
	compare_emb = face_net.get_embedding(images)
	h5file = h5py.File('face.h5','w')
	h5file['X_train']=compare_emb
	h5file.close()
	file_processing.write_data('name.txt', image_list, model='w')

if __name__ == '__main__':
	model_path = 'models/20180408-102900'
	emb_face_dir = './dataset/emb_face'
	create_embedding(model_path,emb_face_dir)
