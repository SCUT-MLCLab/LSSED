# -*- coding: utf-8 -*- 
# @Describe : 
# @Time : 2022/07/04 15:47 
# @Author : zpx 
# @File : utils.py
import numpy as np
import torch
import torch.nn as nn
import torchvision
from ignite.metrics import Metric
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce
from medpy import metric

'''
Paper: TransUNet
Reference: https://github.com/Beckschen/TransUNet/blob/main/utils.py
'''


def calculate_metric_percase(pred_, gt_):
    # (W,H) (W,H)
    pred = pred_.cpu().detach().numpy()
    gt = gt_.cpu().detach().numpy()
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0
    else:
        return 0, 0


def plot_gt_predict(engine):
    with torch.no_grad():
        y_pred, y = engine.state.output[0][0], engine.state.output[1]
        y_pred = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        label = torch.cat([y.unsqueeze(1), y_pred.unsqueeze(1)], dim=0).repeat(1, 3, 1, 1)
        label = torchvision.utils.make_grid(label, normalize=False, padding=5, nrow=y_pred.shape[0], pad_value=0)
    return label


def logValidAndTest(engine, tag, evaluator, dataloader, tb_logger, progressBar):
    evaluator.run(dataloader)
    metrics = evaluator.state.metrics
    prediction_gt_images = plot_gt_predict(evaluator)
    tb_logger.writer.add_image('{}/prediction_gt_images'.format(tag), prediction_gt_images, engine.state.epoch)
    progressBar.log_message(f"{tag} Results - Epoch: {engine.state.epoch} Avg Dice: {metrics['Dice'][0]:.2f} ")


class HD95Metric(Metric):
    def __init__(self, output_transform=lambda x: x, device="cpu"):
        self.sum = 0
        self.len = 0
        super().__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self.sum = 0
        self.len = 0
        super().reset()

    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output[0].detach(), output[1].detach()
        self.len += y_pred.shape[0]
        y_pred = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        for i in range(y_pred.shape[0]):
            _, temp = calculate_metric_percase(y_pred[i] == 1, y[i] == 1)
            self.sum += temp

    @sync_all_reduce("_num_examples", "_num_correct:SUM")
    def compute(self):
        return self.sum / self.len


'''
Paper: TransUNet
Reference: https://github.com/Beckschen/TransUNet/blob/main/utils.py
'''


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


# LSS loss function
class LSS_LOSS(nn.Module):
    def __init__(self, H: int = 30, Q: float = 0.01, gamma: float = 0.1, net=None,
                 device: torch.device = torch.device('cuda')):
        super().__init__()
        self.H = H
        self.Q = Q
        self.device = device
        self.gamma = gamma
        self.net = net

    def forward(self, data, target):
        inputs, pred = data
        ssm = []
        for i in range(self.H):
            delta = torch.empty(inputs.shape, dtype=torch.float32).uniform_(-self.Q, self.Q).to(self.device)
            ssm.append(torch.mean(torch.square(pred - self.net(inputs + delta))))
        ssm = torch.stack(ssm, dim=0)
        total_loss = self.gamma * torch.mean(torch.sqrt(torch.mean(ssm, dim=0)))
        return total_loss


if __name__ == '__main__':
    from network import LRTNet

    lgem_loss = LSS_LOSS(H=2, )
    test = torch.ones((1, 3, 224, 224)).cuda()
    net = LRTNet().cuda()
    lgem_loss = LSS_LOSS(H=2, net=net.seg_head)
    tmp = net(test)
    print(lgem_loss(tmp[1], tmp[0]))

    # dice = DiceCoefficient()
    # print(dice(test, test))
