import os

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from eval import eval_net
from postprocess import post_process
from unet import UNet
from utils import plot_img_pred
from utils import resize_crop_tranpose, normalize, split_img_into_squares, merge_square, dense_crf


def predict_img(net,
                full_img,
                scale_factor=1,
                use_dense_crf=True,
                use_gpu=True,
                post_proc=False):
    net.eval()
    img_height = full_img.size[1]
    img_width = full_img.size[0]

    img = resize_crop_tranpose(full_img, scale=scale_factor)
    img = normalize(img)

    left_square, right_square = split_img_into_squares(img)

    X_left = torch.from_numpy(left_square).unsqueeze(0)
    X_right = torch.from_numpy(right_square).unsqueeze(0)
    X_left = X_left.float()
    X_right = X_right.float()
    if use_gpu:
        X_left = X_left.cuda()
        X_right = X_right.cuda()

    with torch.no_grad():
        output_left = net(X_left)
        output_right = net(X_right)

        left_probs = output_left.squeeze(0)
        right_probs = output_right.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                # reseize的作用是，神经网络输入的图像是压缩过的，输出的图像大小不变，也是压缩过的，因此需要放大。
                transforms.Resize(img_height, interpolation=3),
                transforms.ToTensor()
            ]
        )

        left_probs = tf(left_probs.cpu())
        right_probs = tf(right_probs.cpu())

        left_mask_np = left_probs.squeeze().cpu().numpy()
        right_mask_np = right_probs.squeeze().cpu().numpy()

    full_mask = merge_square(left_mask_np, right_mask_np, img_width)

    if use_dense_crf:
        full_mask = dense_crf(np.array(full_img).astype(np.uint8), full_mask)

    if post_proc:
        full_mask *= 255
        full_mask = full_mask.astype('uint8')
        full_mask = post_process(full_mask)
        full_mask = full_mask.astype('float32')
        full_mask /= 255
    return full_mask


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == "__main__":
    tot_coeff = []
    '''参数开始'''
    model_path = r"H:\SophomoreYearII\my_dachuang\segmentation\final_model.pth"
    gpu = True
    viz = False  # 可视化
    save = True
    crf = False
    post_proc = False
    '''参数结束'''


    net = UNet(n_channels=1, n_classes=1)

    print("Loading model {}".format(model_path))
    checkpoint = torch.load(model_path)
    if gpu:
        print("Using CUDA version of the net, prepare your GPU !")
        net.cuda()
        net.load_state_dict(checkpoint['net'])
    else:
        net.cpu()
        net.load_state_dict(checkpoint['net'])
        print("Using CPU version of the net, this may be very slow")

    print("Model loaded !")
    file_list = []
    path = r'H:\SophomoreYearII\my_dachuang\segmentation\testSet'
    for root, dirnames, filenames in os.walk(path):
        for file_name in filenames:
            if file_name.endswith('.jpg'):
                file_list.append(os.path.join(root, file_name))

    for i, file_name in enumerate(file_list):
        # path = os.path.abspath(os.path.join(outputPath, name))
        img = cv2.imread(file_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, dsize=(882, 460))
        img = img[:, 32:]

        cv2.imwrite(file_name, img)
        #     预测一张前先预处理一张
        temp = file_name.split('\\')[-1]
        print(f"\nPredicting image {temp} ...")
        img = Image.open(file_name)
        if img.size[0] < img.size[1]:
            print("Error: image height larger than the width")
        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=1,
                           use_dense_crf=crf,
                           use_gpu=gpu,
                           post_proc=post_proc)
        trans = transforms.Compose([
            transforms.ToTensor()
        ])
        img1 = trans(img)
        img1 = img1.unsqueeze(0)
        img1 = img1.cuda()
        mask1 = trans(mask)
        mask1 = mask1.unsqueeze(0)
        mask1 = mask1.cuda()
        dice_coeff = eval_net(net, img1, mask1, '', False, True)
        print(dice_coeff)
        tot_coeff.append(dice_coeff)
        mask = np.array(mask > 0.5, dtype=np.float)
        temp = file_name.split('\\')
        temp[-1] = temp[-1].split('.')[0] + '_mask.png'
        mask_name = '\\'.join(temp)

        if viz:
            print("Visualizing results for image {}, close to continue ...".format(file_name))
            plot_img_pred(img, mask)

        if save:
            result = mask_to_image(mask)

            # fileName = "{}_{}.png".format(model_path.split('/')[-1].split('.')[0], fn.split('/')[-1].split('.')[0])
            # fileName = './testSet/' + fileName
            result.save(mask_name)

            print("Mask saved to {}".format(mask_name))

    tot_coeff = np.array(tot_coeff)
    np.savetxt('tot_coeff',tot_coeff)
