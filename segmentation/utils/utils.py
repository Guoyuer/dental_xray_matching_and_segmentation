import random
import numpy as np
import sys
import cv2
from matplotlib import pyplot as plt
import torch


def get_square(img, pos):
    """Extract a left or a right square from ndarray shape : (C, H, W))"""
    h = img.shape[1]  # h是高度
    if pos == 0:
        return img[:, :, :h]  # 截取左边的正方形，边长==高度
    else:
        return img[:, :, -h:]  # 截取右边的正方形，边长==高度.-h先定位，然后：右边没有，就代表着取完


def split_img_into_squares(img):
    return get_square(img, 0), get_square(img, 1)


def hwc_to_chw(img):
    # print(img.shape)

    return np.transpose(img[0], axes=[2, 0, 1])


def resize_crop_tranpose(pilimg, scale=0.5, final_height=None, masks=False):
    # print(pilimg.size)
    w = pilimg.size[0]
    h = pilimg.size[1]
    newW = int(w * scale)
    newH = int(h * scale)

    if not final_height:
        diff = 0
    else:
        diff = newH - final_height

    img = pilimg.resize((newW, newH))
    img = img.crop((0, diff // 2, newW, newH - diff // 2))
    if masks:
        img = np.array(img, dtype=np.uint8)
        img = np.array((np.uint8(img > 0)) * 255)

    else:
        img = np.array(img, dtype=np.uint8)
    # one=np.ones(shape=[1])
    # img = img[:,np.newaxis]
    img = np.expand_dims(img, axis=2)
    # print(img.shape)
    # img = np.column_stack((img,one))
    # chw
    img = img.transpose((2, 0, 1))
    return img  # 尚未正方形


def tensor_to_pics(tensor, width):
    width = round(width)
    all_pred = np.empty(shape=(0, width))
    for i in range(tensor.shape[0]):
        temp = tensor[i:i + 1]
        if i % 2 == 0:
            leftpred = temp.squeeze().detach().cpu().numpy()
        else:
            rightpred = temp.squeeze().detach().cpu().numpy()
            onesquare = merge_square(leftpred, rightpred, width)
            all_pred = np.append(all_pred, onesquare, axis=0)
    return all_pred


def show_img_gt_pred(img, gt, pred):
    newImg = np.concatenate((img, gt, pred), axis=1)
    plt.imshow(newImg, cmap='gray')
    plt.show()


def split_train_val(dataset, val_num=3):
    length = len(dataset)
    if val_num >= length:
        print("val_num is too big!")
        exit(1)
    random.shuffle(dataset)
    return {'train': dataset[:-val_num], 'val': dataset[-val_num:]}


def normalize(x: np.ndarray):
    x = x.astype(np.uint8)
    return x / 255


def merge_square(img1, img2, full_w):
    h = img1.shape[0]

    new = np.zeros((h, full_w), np.float32)
    new[:, :full_w // 2 + 1] = img1[:, :full_w // 2 + 1]  # 左边
    new[:, full_w // 2 + 1:] = img2[:, -(full_w // 2 - 1):]  # 右边
    # 也就是说，重叠部分是从左右小square各取一半

    return new


# credits to https://stackoverflow.com/users/6076729/manuel-lagunas
def rle_encode(mask_image):
    pixels = mask_image.flatten()
    # We avoid issues with '1' at the start or end (at the corners of
    # the original image) by setting those pixels to '0' explicitly.
    # We do not expect these to be non-zero for an accurate mask,
    # so this should not harm the score.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs
