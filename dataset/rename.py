#coding=gbk
import os
import sys
def rename():
    path="images/face4"
    name="face4_"
    startNumber=1
    fileType=".jpg"
    print("����������"+name+str(startNumber)+fileType+"�������ļ���")
    count=0
    filelist=os.listdir(path)
    for files in filelist:
        Olddir=os.path.join(path,files)
        if os.path.isdir(Olddir):
            continue
        Newdir=os.path.join(path,name+str(count+int(startNumber))+fileType)
        os.rename(Olddir,Newdir)
        count+=1
    print("һ���޸���"+str(count)+"���ļ�")

rename()