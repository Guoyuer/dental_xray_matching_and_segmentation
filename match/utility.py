import cv2
import numpy as np


# 以下都是核心函数需要用的辅助函数

def imread(filePath):
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)
    if len(cv_img.shape) == 3:  # 是三通道图则转成单通道
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    cv_img = cv_img[:, 100:]
    return cv_img


def kp2list(kps):
    l = []
    for kp in kps:
        l.append((kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id))
    return l


def list2kp(l):
    kps = []
    for point in l:
        kps.append(cv2.KeyPoint(x=point[0][0], y=point[0][1], _size=point[1], _angle=point[2], _response=point[3],
                                _octave=point[4], _class_id=point[5])
                   )
    return kps
