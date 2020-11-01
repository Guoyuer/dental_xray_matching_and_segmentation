#
# load.py : utils on generators / lists of ids to transform from strings to
#           cropped images and masks

import os
import torchvision.transforms.functional as TF
import torchvision.transforms as tfs
import random
import numpy as np
from PIL import Image

from .utils import resize_crop_tranpose, get_square, normalize


def transforms_with_augm(image, mask):
    # CHW-->HWC
    image = image.transpose((1, 2, 0))
    mask = mask.transpose((1, 2, 0))

    image = tfs.ToPILImage('L')(image)
    mask = tfs.ToPILImage('L')(mask)
    # 先进行旋转

    angle = random.randint(-45, 45)
    image = TF.rotate(image, angle)
    mask = TF.rotate(mask, angle)
    # 再对图像进行颜色变换
    image = tfs.ColorJitter(0.5, 0.5, 0.5)(image)
    image = np.array(image, dtype=np.float32)
    mask = np.array(mask, dtype=np.float32)
    image /= 255
    mask /= 255
    # 增加channel，转到CHW
    image = image[np.newaxis,:]
    mask = mask[np.newaxis,:]

    return image, mask

def transforms_without_augm(image, mask):
    image = np.array(image, dtype=np.float32)
    mask = np.array(mask, dtype=np.float32)
    image /= 255
    mask /= 255
    return image,mask

def get_ids(dir):
    """Returns a list of the ids in the directory，这套id是img和mask共享的"""
    return [f[:-4] for f in os.listdir(dir)]


# def split_ids(ids, n=2):
#     """Split each id in n, creating n tuples (id, k) for each id"""
#
#     return [(id, i) for id in ids for i in range(n)]


def crop_all_images(ids, dir, suffix, scale, masks=False):
    """From a list of tuples, returns the correct cropped img"""
    l = []

    for id in ids:
        raw_img = Image.open(dir + id + suffix)
        if masks:
            im = resize_crop_tranpose(raw_img, scale=scale, masks=True)
        else:
            im = resize_crop_tranpose(raw_img, scale=scale, masks=False)
        left_square = get_square(im, 0)
        right_square = get_square(im, 1)
        l.append(left_square)
        l.append(right_square)

    return l


def get_imgs_and_masks(ids, dir_img, dir_mask, scale, augm):
    """Return all the couples (img, mask)"""
    # ids是image和mask共享的
    imgs = crop_all_images(ids, dir_img, '.png', scale)
    masks = crop_all_images(ids, dir_mask, '_mask.png', scale, masks=True)

    new_imgs = []
    new_masks = []
    for i in range(len(imgs)):
        if augm:
            a, b = transforms_with_augm(imgs[i], masks[i])
        else:
            a, b = transforms_without_augm(imgs[i], masks[i])
        new_imgs.append(a)
        new_masks.append(b)

    return np.array(new_imgs), np.array(new_masks)


def get_full_img_and_mask(id, dir_img, dir_mask):
    im = Image.open(dir_img + id + '.jpg')
    mask = Image.open(dir_mask + id + '_mask.gif')
    temp_im = np.array(im)
    # temp_im = [temp_im:, np.newaxis]
    temp_mask = np.array(mask)
    # temp_mask = [temp_mask:, np.newaxis]
    return np.array(temp_im), np.array(temp_mask)
