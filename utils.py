#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@author toby
'''
import numpy as np
import torch.nn.functional as F
from skimage import data, io, filters, transform
from scipy import stats
from matplotlib.patches import Ellipse, Circle
import math
import smtplib
from email.mime.text import MIMEText
import torch

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc

def cross_entropy2d(input, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    log_p = F.log_softmax(input)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss

def bceloss3d(input, target):
    # input: (n, c, h, w), target: (n, c,  h, w)
    return F.binary_cross_entropy_with_logits(input, target)


def show_img(img):
    io.imshow(img)
    io.show()

def rotate(x, y, c, angel, cos):
    if cos == True:
        return (x - c[0]) * math.cos(angel) - \
               (y - c[1]) * math.sin(angel) + c[0]
    else:
        return (x - c[0]) * math.sin(angel) + \
               (y - c[1]) * math.cos(angel) + c[1]

def accuracy_score(output, lable):
    # TODO
    # output: (n, c, w, h), lable: (n, c, i, j)，都是Variable类型
    # outpu是输出的w*h大小的二维高斯分布（方差为5px），c代表16种类别，n代表mini_batch的大小。
    # lable中i和j代表正确的point。你写一个计算正确率的函数，ouput中的高斯分布的中心就是预测的point的点
    # 如果（i, j）和 （预测的i，预测的j）之间欧式距离在5内，则算正确。
    pass

def G_P(h, w, mean, conv):
    """Gaouse x, y

    Args:
        h,w: image h and w
        mean: center []
        conv, Varience

    Math: G(u,v) = \frac{1}{2\pi \sigma^2} e^{-(u^2 + v^2)/(2 \sigma^2)}
    """
    position_heatmap = np.zeros((len(mean), h, w), dtype=np.float32)
    for i in range(len(mean)):
        array = [(k, j) for k in range(h) for j in range(w)]
        p = stats.multivariate_normal.pdf(array, [mean[i][1], mean[i][0]], conv)
        position_heatmap[i] = np.array(p).reshape(h ,w)

    return position_heatmap

def binnary_heatmap(center, h, w, k, r):
    """Produce heatmap
    Args:
        center: center bout N part
        k: k feature
        r: radius
    Math: first we must get inscribed circle
    """
    heatmap = np.zeros((k, h, w), dtype=np.float32)
    heatmap[0] = np.ones((h, w), dtype=np.float32)
    for i in range(1, k):
        position = center[i]
        sub_heatmap = np.zeros((h, w), dtype=np.float32)
        left = int(round(position[0])) - r
        if left <= 0:
            left = 0

        right = int(round(position[0])) + r
        if right >= w:
            right = w

        bottom = int(round(position[1])) - r
        if bottom <= 0:
            bottom = 0

        top = int(round(position[1])) + r
        if top >= h:
            top = h

        for ii in range(left, right):
            for jj in range(bottom, top):
                if(math.sqrt(math.pow(ii - position[0], 2) + math.pow(jj - position[1], 2))) <= r:
                    sub_heatmap[jj, ii] = 1
                    if i != 0:
                        heatmap[0, jj, ii] = 0
        heatmap[i] = sub_heatmap
    return heatmap

def sendToToby():
    SMTPserver = 'smtp.163.com'
    sender = 'julie19970618@163.com'
    password = "584014930"

    message = 'train success.'
    msg = MIMEText(message)

    msg['Subject'] = 'Hello ClearLove'
    msg['From'] = sender
    msg['To'] = "julie334600@nwsuaf.edu.cn"

    mailserver = smtplib.SMTP(SMTPserver, 25)
    mailserver.login(sender, password)
    mailserver.sendmail(sender, [msg['To']], msg.as_string())
    mailserver.quit()
    print('send email success')

#
def save_model(model, name):
    torch.save(model.state_dict(), '{}.pkl'.format(name))