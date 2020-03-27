# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 11:06:37 2019

@author: prophet lin
"""
#%%
from matplotlib import pyplot as plt
import numpy as np
import cv2
import os

# def get_good_num(img1, img2):
#     h, w = img1.shape[:2]      #获取图像的高和宽
#     img2 = cv2.resize(img2,(img1.shape[1],img1.shape[0]))
#     img1 = img1[:,100:]
#     img2 = img2[:,100:]
#
#     sift = cv2.xfeatures2d.SIFT_create()
#
#     matches = flann.knnMatch(des1,des2,k=2)
#
#     good = []
#     good_p1 = []
#     good_p2 = []
#     dis = []
#     delta_vector = []
#     for m,n in matches:
#         if m.distance < 0.7*n.distance:
#             delta_p = [int(kp2[m.trainIdx].pt[0] - kp1[m.queryIdx].pt[0]),
#                        int(kp2[m.trainIdx].pt[1] - kp1[m.queryIdx].pt[1])]
#             delta_p_dis = delta_p[0]**2+delta_p[1]**2
#             if delta_p_dis < 100000:
#                 good.append([m])
#                 dis.append(delta_p_dis)
#                 delta_vector.append(delta_p)
#                 good_p1.append([int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1])])
#                 good_p2.append([int(kp2[m.trainIdx].pt[0]),int(kp2[m.trainIdx].pt[1])])
#
#     return len(good)


#%%
filePath = r'C:\Users\prophet lin\Documents\code\resource\dc'
dirList = os.listdir(filePath)


ori = []
lable  =[]
sim_lable = []
count = 0

for dir in dirList:
    print(dir)
    for file in os.listdir(os.path.join(filePath,dir)):
        print(":"+file)
        #img = cv2.imread(os.path.join(filePath,dir,file))
        #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #ori.append(img)
        ori.append(os.path.join(filePath,dir,file))
        lable.append(dir+"-"+file)
        sim_lable.append(dir)
        count = count + 1

#%%
rate = []
record = []
for i in range(len(sim_lable)):
    res=[]
    cur_sim_label = sim_lable[i]
    cur_label = lable[i]
    cur_img = cv2.imread(ori[i])
    max_num = 0
    target_label = -1
    for j in range(count):
        print('ing:',j)
        com_sim_label = sim_lable[j]
        com_label = lable[j]
        com_img = cv2.imread(ori[j])
        if cur_label != com_label:
            SITF_num = get_good_num(cur_img, com_img)
            res.append([int(SITF_num),com_label])
            if SITF_num > max_num:
                max_num = SITF_num
                target_label = com_sim_label
    record.append(res)
    rate.append(target_label == cur_sim_label)



