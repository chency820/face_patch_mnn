import cv2
import time
import ntpath
import sys, os
import pickle
import FaceProcessUtil as fpu

#txt path
txtPath = './txt/CK+_106_single_309.txt'

#replace your data path of CK+
data_root_path = 'D:/chenchuyang/learning/FNN/fera/cohn-kanade-images/cohn-kanade-images/'

#total list of path and label
imglist = [[],[],[],[],[],[],[],[],[],[]] #ten fold
imglist_neu = [[],[],[],[],[],[],[],[],[],[]]
labellist = [[],[],[],[],[],[],[],[],[],[]] #ten fold

list_txt = open(txtPath,'r')
content = list_txt.readlines()

#append img to one list
for i, line in enumerate(content):
    line = line.replace('\n','')
    line = line.split('\t')
    group = int(line[0])
    path = line[1]
    path_neu = path[:-12] + '00000001.png'
    label = line[2]
    imglist[group].append(data_root_path+path)
    imglist_neu[group].append(data_root_path+path_neu)
    labellist[group].append(label)

total = 0
print('10', imglist_neu[0])
print('20', imglist[0])
print('30', labellist[0])
for i ,e in enumerate(labellist):
   # print(len(e))
    total = total+len(e)
print("Total training images:%d"%(total))
#-----------------prepare imglist end -----------------------------

count=0

gc = 0

feature_group_of_subject=[]

tm1=time.time()

for i in range(len(imglist)):
    ckplus={}
    ckplus_label=[]
    ckplus_img = []
    ckplus_img_neu = []
    imagelist = imglist[i]
    imagelist_neu = imglist_neu[i]
    lablist = labellist[i]
    for j,v in enumerate(imagelist):
        count = count+1
        label = int(lablist[j])
        #-----map label-----
        if label == 7:
            label = 2
        #-----map label-----
        label = label-1
        print("\n> Prepare image                                         %f%%"%(count*100/total))

        image_path = v
        print('1', image_path)
        flag, img=fpu.calibrateImge(image_path)
        if flag:
                imgr = fpu.getLandMarkFeatures_and_ImgPatches(img, True, False, True)
        else:
            print('Unexpected case while calibrating for:'+str(image_path))
            exit(1)

        if imgr[1]:
            gc = gc + 1
            img = imgr[0]

            img2 = img
            print("Get Geometry>>>>>>>>>>>>>>")
            ckplus_label.append(label)
            ckplus_img.append(img2)
        else:
            print('No feature detected:' + image_path)
            exit(1)

    for j, v in enumerate(imagelist_neu):
        image_path_neu = v
        print('2', image_path_neu)
        flag, img_neu = fpu.calibrateImge(image_path_neu)
        if flag:
            imgr_neu = fpu.getLandMarkFeatures_and_ImgPatches(img_neu, True, False, True)
        else:
            print('Unexpected case while calibrating for:' + str(image_path_neu))
            exit(1)

        if imgr_neu[1]:
            img_neu = imgr_neu[0]
            print("Get Geometry>>>>>>>>>>>>>>")
            ckplus_img_neu.append(img_neu)
        else:
            print('No feature detected:'+image_path_neu)
            exit(1)

    ckplus['labels']=ckplus_label
    ckplus['img'] = ckplus_img
    ckplus['img_neu'] = ckplus_img_neu
    feature_group_of_subject.append(ckplus)

filenametosave='./pkl/kdef_pair_106.pkl'

with open(filenametosave,'wb') as fin:
    pickle.dump(feature_group_of_subject,fin,4)

tm2=time.time()
dtm = tm2-tm1
print('Total images: %d\tGet: %d'%(count, gc))
print("Total time comsuming: %fs for %d images"%(dtm, count))