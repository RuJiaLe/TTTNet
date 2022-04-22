import torch
from utils.material import SSIM, IOU, S_Loss
import torch.nn as nn

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# Loss
bce_loss = nn.BCELoss(size_average=True)
ssim_loss = SSIM(window_size=11, size_average=True)
iou_loss = IOU(size_average=True)
s_loss = S_Loss()


def Loss(predict, target):
    bce_out = bce_loss(predict, target)
    ssim_out = 1 - ssim_loss(predict, target)
    iou_out = iou_loss(predict, target)
    s_out = s_loss(predict, target)

    loss = bce_out + ssim_out + iou_out + s_out

    return loss, bce_out, ssim_out, iou_out, s_out


def multi_loss(out1, out2, gts):

    all_loss = []

    for j in range(len(gts)):

        frame_loss = torch.tensor(0.0).to(device)

        for i in range(len(out1)):
            loss1 = Loss(out1[i][j], gts[j])
            loss2 = Loss(out2[i][j], gts[j])
            frame_loss += (loss1[0] + loss2[0])

        all_loss.append(frame_loss)

    return all_loss
