#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@author toby
'''
from config import app
import argparse
from subnets import PartDetectionNet, RegresstionLocationNet
from utils import bceloss3d, save_model, show_img
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from dataload import  transformed_dataset, transformed_test_dataset, test_data_length
from torch.autograd import  Variable
import torch
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
from resnetfcn import ResNetFCN
from best_model import HourGlassNet
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
# pdn = PartDetectionNet()
# reg = RegresstionLocationNet()

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--outf', default='generate/', help='generate heatmap picture')
parser.add_argument('--model', type=int, default=1,  help='choose model')
parser.add_argument('--train', type=int, default=1,  help='train which model')
parser.add_argument('--batchsize', type=int, default=2,  help='train which model')
parser.add_argument('--deviceamount', type=int ,default=0, help='cuda dvevice')
parser.add_argument('--mode', type=int ,default=0, help='0:train,1:test')
parser.add_argument('--imagesize', type=int ,default=256, help='0:train,1:test')

opt = parser.parse_args()
opt.deviceamount = torch.cuda.device_count()
print(opt.deviceamount)
print(opt)

cudnn.benchmark = True
def accuracy(output, target):
    # realize by toby's gf
    target = target[:, : 16]
    output = output[:, : 16]
    output = output.view(output.size(0), output.size(1), -1)
    _, predict = torch.max(output, 2)
    print(output[0, 2])
    accuracy_static = np.zeros((15))
    total = 0
    for i in range(predict.size(0)):
        for j in range(predict.size(1) - 1):
            position = predict[i, j]
            y = math.floor(position/256)
            x = position - (y) * 256
            if math.sqrt(math.pow(target[i, j, 1] - y, 2) + math.pow(target[i, j, 0] - x, 2)) <= 10:
                accuracy_static[j] += 1
                total += 1

    return total, accuracy_static

def annotation_img(img, output):
    output = output[:, : 16]
    output = output.view(output.size(0), output.size(1), -1)
    _, predict = torch.max(output, 2)
    annotation_coordinate = np.zeros((15, 2))
    for i in range(predict.size(0)):
        for j in range(predict.size(1)):
            position = predict[i, j].data + 1
            y = math.floor(position / opt.imagesize) - 1
            x = position - (y + 1) * opt.imagesize - 1


def trainer(pdn, reg):
    """"Common trainer for every epoch
    """
    dataloader = DataLoader(transformed_dataset, batch_size=opt.batchsize, shuffle=True, num_workers=6)
    j = 0
    mode = "detector"
    use_cuda = app["use_cuda"]
    epochs = app["train"]["epochs"]

    if use_cuda:
        pdn = pdn.cuda()
        reg = reg.cuda()

    optimizer1 = optim.Adam(pdn.parameters(), lr=1e-4, weight_decay=0.0005)
    exp_lr_scheduler1 = lr_scheduler.StepLR(optimizer1, step_size=3000, gamma=0.1)
    optimizer2 = optim.Adam(reg.parameters(), lr=1e-4, weight_decay=0.0005)
    exp_lr_scheduler2 = lr_scheduler.StepLR(optimizer1, step_size=3000, gamma=0.1)

    for i in range(epochs["total"]):
        print("epoch: {}".format(i))
        for num, data in enumerate(dataloader, 0):
            image, target = data["image"], data["landmarks"]
            # show_img(image[0].numpy().transpose(1, 2, 0))
            # show_img(target[0][0][0].numpy())

            if use_cuda:
                image = Variable(image).cuda()
            else:
                image = Variable(image)

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            output = pdn(image)

            if mode != "regresstion":
                if use_cuda:
                    heatmap = Variable(target[0].cuda())
                else:
                    heatmap = Variable(target[0])

                loss = bceloss3d(output, heatmap)
                loss.backward()
                optimizer1.step()
                print("part loss: ",loss)
                # if i % 10 ==0 and i != 0:
                #     print(output[0].size())
                #     vutils.save_image(output[0].data,
                #                       '%s/samples_iteration_%03d.png' % (opt.outf, i),
                #                       normalize=True)

            if mode != "detector" and opt.train == 2:
                if use_cuda:
                    G = Variable(target[1].cuda())
                else:
                    G = Variable(target[0])
                optimizer1.zero_grad()
                output = torch.cat((output, image), 1)
                output = reg(output.detach())
                loss = F.mse_loss(output[:, 1:], G[:, 1:])
                loss.backward()
                optimizer2.step()
                print("r losss: ", loss)

        if i - j == epochs["together"]:
            if mode == "together":
                mode = "detector"
                j = i
        elif i - j == epochs["detector"]:
            trian_mode = app["train"]["model"]
            mode = trian_mode[trian_mode.index(mode) + 1]
            j = i

        if i % 50 == 0 and i != 0:
            save_model(pdn, "{}{}pdn".format(opt.outf, i))
            save_model(reg, "{}{}reg".format(opt.outf, i))

if opt.mode == 0:
    if opt.model == 1:
        fcn = PartDetectionNet()
        rnet = HourGlassNet()
    else:
        fcn = PartDetectionNet()
        rnet = RegresstionLocationNet()

    trainer(fcn, rnet)

# test the specific dataset
else:
    if opt.model == 1:
        fcn = PartDetectionNet()
        rnet = HourGlassNet()
    else:
        fcn = PartDetectionNet()
        rnet = RegresstionLocationNet()

    fcn.load_state_dict(torch.load('{}{}'.format(opt.outf, '200pdn.pkl')))
    rnet.load_state_dict(torch.load('{}{}'.format(opt.outf, '200reg.pkl')))
    fcn = fcn.cuda()
    rnet = rnet.cuda()
    #init test dataset
    dataloader = iter(DataLoader(transformed_test_dataset, batch_size=8, shuffle=False, num_workers=3))
    total = 0
    part = np.zeros((15))
    while True:
        try:
            data = dataloader.next()
        except:
            break
        image, target = data["image"], data["landmarks"]
        image = Variable(image.cuda())
        output = fcn(image)
        output = torch.cat((output, image), 1)
        output = rnet(output)
        t, p = accuracy(output.data, target)
        total += t
        part += p


    print("total accuracy: {}".format(total / test_data_length))
    print("part accuracy: {}".format(part / (test_data_length * 15)))


