from matplotlib import pyplot as plt
import numpy as np
import cv2
import os
import time
import pickle
import gzip
from utility import *
# def gdump(obj, filename):
#     file = gzip.GzipFile(filename, 'wb')
#     pickle.dump(obj, file, -1)
#     file.close()


# def gload(filename):
#     file = gzip.GzipFile(filename, 'rb')
#     res = pickle.load(file, -1)
#     file.close()
#     return res


filePath = r'C:\同步文件夹——mi\my_dachuang\original_dataset-archive'
dirList = os.listdir(filePath)
img_paths = []
lables = []
for dir in dirList:
    print(dir)
    for file in os.listdir(os.path.join(filePath, dir)):
        print(":" + file)
        # img = cv2.imread(os.path.join(filePath,dir,file))
        # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # ori.append(img)
        img_paths.append(os.path.join(filePath, dir, file))
        lables.append(dir + "-" + file)

sift = cv2.xfeatures2d.SIFT_create(200)


database = dict()

for i in range(len(img_paths)):
    print(lables[i])
    img = imread(img_paths[i])
    img = img[:, 100:]
    kp, des = sift.detectAndCompute(img, None)
    kp = kp2list(kp)
    database[lables[i]] = (kp, des)

with open('database_200feature.pkl', 'wb') as f:
    pickle.dump(database, f)