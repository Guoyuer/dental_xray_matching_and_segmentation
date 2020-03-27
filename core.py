import heapq
import numpy as np
import os
import pickle
from utility import *
import pandas as pd
import time


#############每次调用核心函数都需要的全局变量################
####常量
# filePath是最原始的那个数据集地址
filePath = r'C:\同步文件夹——mi\my_dachuang\original_dataset-archive'
dirList = os.listdir(filePath)
img_paths = []
labels = []
persons = []
for dir in dirList:
    for file in os.listdir(os.path.join(filePath, dir)):
        img_paths.append(os.path.join(filePath, dir, file))
        labels.append(dir + "-" + file)
        persons.append(dir)

# databasePath是数据库文件的地址
databasePath = 'database_200feature.pkl'
# 读入数据库
with open(databasePath, 'rb') as f:
    database = pickle.load(f)
# flann
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
# sift
sift = cv2.xfeatures2d.SIFT_create(200)


#############核心函数定义################
def query(path: str) -> list:
    '''

    :param path: 图像的路径
    :return: 一个列表，列表里有1~5个dict，每个dict格式如下：

    '''
    img = imread(path)
    similarity_vector = pd.DataFrame(np.zeros(shape=(1, len(labels)), dtype=np.int))
    similarity_vector.columns = labels
    kp1, des1 = sift.detectAndCompute(img, None)
    # 遍历数据库进行匹配
    for label in labels:
        kp_list2 = database[label][0]
        kp2 = list2kp(kp_list2)
        des2 = database[label][1]
        matches = flann.knnMatch(des1, des2, k=2)
        good = []
        good_p1 = []
        good_p2 = []
        dis = []
        delta_vector = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                delta_p = [int(kp2[m.trainIdx].pt[0] - kp1[m.queryIdx].pt[0]),
                           int(kp2[m.trainIdx].pt[1] - kp1[m.queryIdx].pt[1])]
                delta_p_dis = delta_p[0] ** 2 + delta_p[1] ** 2
                if delta_p_dis < 100000:
                    good.append([m])
                    dis.append(delta_p_dis)
                    delta_vector.append(delta_p)
                    good_p1.append([int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1])])
                    good_p2.append([int(kp2[m.trainIdx].pt[0]), int(kp2[m.trainIdx].pt[1])])
        similarity_vector[label] = len(good)

    K = 3
    temp = np.array(similarity_vector)
    temp = temp.squeeze()
    index = heapq.nlargest(K, range(len(temp)), temp.take)
    res = []
    for i in index:
        t = {'path': img_paths[i], 'label': labels[i], 'similarity': similarity_vector[labels[i]].values[0]}
        res.append(t)
    return res

# 核心函数的使用与返回
# path = r"C:\同步文件夹——mi\my_dachuang\original_dataset-archive\1\20121211.jpg"
# print(query(path))
