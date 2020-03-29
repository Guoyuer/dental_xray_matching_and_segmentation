import torch
import torch.nn.functional as F
from utils import plot_img_pred_gt
from dice_loss import dice_coeff
import numpy as np
from sklearn.metrics import roc_curve, auc


def eval_net(net, imgs, true_masks, title, viz, gpu):
    """Evaluation without the densecrf with the dice coefficient"""
    tot0, tot1, tot2, tot3 = 0, 0, 0, 0
    net.eval()
    # all_mask_pred = torch.empty(size=(0, 1, imgs.shape[3], imgs.shape[3])).cuda()
    # 避免再占用显存，转移至CPU
    # model的转移是原地的，而tensor则不是
    if not gpu:
        torch.set_num_threads(11)
        imgs = imgs.cpu()
        true_masks = true_masks.cpu()
        net.cpu()
    # all_mask_pred = all_mask_pred.cpu()
    # 避免超显存，一个正方形一个正方形得来
    for i in range(0, imgs.shape[0]):
        img = imgs[i].unsqueeze(0)
        mask_pred0 = net(img)
        mask_pred1 = (mask_pred0 > 0.2).float()
        mask_pred2 = (mask_pred0 > 0.5).float()
        mask_pred3 = (mask_pred0 > 0.8).float()
        # all_mask_pred = torch.cat((all_mask_pred, mask_pred))
        true_mask = true_masks[i].unsqueeze(0)
        tot0 += dice_coeff(mask_pred0, true_mask).item()
        tot1 += dice_coeff(mask_pred1, true_mask).item()
        tot2 += dice_coeff(mask_pred2, true_mask).item()
        tot3 += dice_coeff(mask_pred3, true_mask).item()
        total = np.array([tot0, tot1, tot2, tot3]) / imgs.shape[0]

    dsp = f'''dice_coeff = {total[0]:.4f} (no threshold)'''
    if viz:
        plot_img_pred_gt(img.cpu()[0][0], mask_pred0.detach().cpu()[0][0], true_mask.cpu()[0][0], title=title,
                         dsp=dsp)
        # 记得将net放回gpu
    if gpu:
        net.cuda()
    return total
