import sys
import random
import math
from optparse import OptionParser
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim

from segmentation import dice_loss
import matplotlib.pyplot as plt
from torchvision import transforms as tsf
from .eval import eval_net
from .unet import UNet
from .utils import get_ids, split_train_val, get_imgs_and_masks
import os

print(os.path.join(os.path.abspath('.'), './data/train/'))


def train_net(net, iters_per_epoch=10,
              epochs=1,
              batch_size=1,
              lr=0.001,
              display_step=10,
              save_epoch=10,
              train_gpu=True,
              eval_gpu=True,
              img_scale=0.8,
              model=None,
              val_num=3,
              viz=True
              ):
    if train_gpu:
        net.cuda()
        cudnn.benchmark = True  # faster convolutions, but more memory

    optimizer = optim.Adam(net.parameters(),
                           lr=lr,
                           weight_decay=0.0001)
    start_epoch = 0
    iters = 0
    if model:
        checkpoint = torch.load(model)
        net.load_state_dict(checkpoint['net'])
        # optimizer.load_state_dict(checkpoint['optimizer']) 不能错误地又读入之前的optimizer!
        start_epoch = checkpoint['epoch'] + 1
        iters = checkpoint['iters']
        print('Model loaded from {}'.format(model))
    else:
        print("Training started anew!")
    dir_img = r'./data/train/'
    dir_mask = r'./data/train_masks/'
    dir_checkpoint = './checkpoints/'

    ids = get_ids(dir_img)
    # print('ids1:',tuple(ids))
    # ids = split_ids(ids)#每个id现在都变成了两个，一个和0组成元组表示左边正方形，一个和1组成元组表示右边正方形

    iddataset = split_train_val(ids, val_num=val_num)
    N_train = len(iddataset['train'])

    print(f'''
Starting training:
    Epochs: {epochs}
    Batch size: {batch_size}
    Learning rate: {lr}
    Training size: {N_train}
    Validation size: {len(iddataset['val'])}
    Display_step: {display_step}
    Start_epoch: {start_epoch}
    Iters_per_epoch: {iters_per_epoch}
    Save_epoch: {save_epoch}
    Train_gpu: {train_gpu},
    Eval_gpu:{eval_gpu}
    ''')



    # 所有数据全部读入内存且处理完毕
    global_step = start_epoch * iters

    for epoch in range(start_epoch, start_epoch + epochs):
        # 每个epoch都要重新这么读入一下，因为split_train_val有随机性，这样子可以让每个epoch内的顺序随机
        # 都含左右两个方块
        iddataset = split_train_val(ids, val_num=val_num)

        img_train, mask_train = get_imgs_and_masks(iddataset['train'], dir_img, dir_mask, img_scale, augm=False)
        img_val, mask_val = get_imgs_and_masks(iddataset['val'], dir_img, dir_mask, img_scale, augm=False)

        img_train = torch.from_numpy(img_train)
        mask_train = torch.from_numpy(mask_train)

        img_val = torch.from_numpy(img_val)
        mask_val = torch.from_numpy(mask_val)

        if train_gpu:
            img_train = img_train.cuda()
            mask_train = mask_train.cuda()
            img_val = img_val.cuda()
            mask_val = mask_val.cuda()

        print('Starting epoch {}/{}.'.format(epoch + 1, start_epoch + epochs))
        net.train()
        epoch_loss = 0

        for i in range(iters_per_epoch):
            # 从内存中取出batch，起始位置随机
            begin_index = random.randint(0, img_train.shape[0] - batch_size)
            batch = img_train[begin_index:begin_index + batch_size]
            masks_pred = net(batch)
            masks_probs_flat = masks_pred.view(-1)
            true_mask = mask_train[begin_index:begin_index + batch_size]
            true_masks_flat = true_mask.view(-1)
            # 已经加了sigmoid了！
            criterion = nn.BCELoss()
            loss = criterion(masks_probs_flat, true_masks_flat)
            epoch_loss += loss.item()
            if global_step % display_step == 0:
                print('iter:{0} --- loss: {1:.6f}'.format(global_step, loss.item()))
            global_step += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch finished ! Loss: {}'.format(epoch_loss / (i + 1)))
        # 验证+保存
        # epoch % save_epoch == 0
        if 1:
            state = {'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'iters': iters}
            print('Checkpoint {} saved !'.format(epoch + 1))
            # validation前后必须释放显存
            torch.save(state,
                       dir_checkpoint + 'CP{}.pth'.format(epoch + 1))
            torch.cuda.empty_cache()
            val_dice = eval_net(net, img_val, mask_val, title=f'CP{epoch + 1}.pth', viz=viz, gpu=eval_gpu)
            torch.cuda.empty_cache()
            print('Validation Dice Coeff: {}'.format(val_dice))

        random.shuffle(iddataset['train'])


if __name__ == '__main__':
    model_path = r"./checkpoints/CP38.pth"
    net = UNet(n_channels=1, n_classes=1)

    try:
        train_net(net=net,
                  iters_per_epoch=200,
                  epochs=30,
                  batch_size=3,
                  lr=0.000001,
                  display_step=50,
                  save_epoch=1,  # 隔几个epoch保存一下模型
                  train_gpu=True,
                  eval_gpu=True,
                  img_scale=1,
                  model=model_path,
                  val_num=60,
                  viz=True
                  )
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
