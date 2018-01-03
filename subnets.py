#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@author toby
@decribe two net (part detection, parts regression)
'''
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models
from skimage import data, io, filters, transform
from fcn32s import FCN32s
from fcn8s import FCN8s
from torchvision import models
from fcn32s import FCN32s
from utils import *
from config import app

use_cuda = torch.cuda.is_available()

class PartDetectionNet(nn.Module):

    def __init__(self):
        super(PartDetectionNet, self).__init__()
        # load pretrained model
        vgg16 = models.vgg16(pretrained=True)
        self.fcn = FCN32s(n_class=app["dataload"]["N"])
        self.fcn.copy_params_from_vgg16(vgg16)

    def forward(self, input):
        x = self.fcn(input)
        return x

class RegresstionLocationNet(nn.Module):

    def __init__(self):
        super(RegresstionLocationNet, self).__init__()
        layers = []
        layers += [nn.Conv2d(app["dataload"]["N"]+3, 64, 9)]
        layers += [nn.Conv2d(64, 64, 13)]
        layers += [nn.Conv2d(64, 128, 13)]
        layers += [nn.Conv2d(128, 256, 15)]
        layers += [nn.Conv2d(256, 512, 1)]
        layers += [nn.Conv2d(512, 512, 1)]
        layers += [nn.Conv2d(512, 16, 1)]
        layers += [nn.ConvTranspose2d(16, 16, 8, 4)]
        self.features = nn.Sequential(*layers)

    def forward(self, input):
        return self.features(input)[:, :, 31:31 + input.size()[2], 31:31 + input.size()[3]].contiguous()




# net1 = PartDetectionNet()
# net2 = RegresstionLocationNet()
#
# input = Variable(torch.zeros(1, 3, 380, 380))
# print(net1(input).size())
