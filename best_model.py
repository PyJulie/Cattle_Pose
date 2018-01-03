#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@author toby
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
from utils import *
from config import app


class Bottleneck(nn.Module):
    """Residual learning

    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class HourGlassNet(nn.Module):
    def __init__(self):
        super(HourGlassNet, self).__init__()
        self.inplanes = 64

        self.d1 = nn.Sequential(*[
            nn.Conv2d(19, 64, 7, 2, padding=3),
            nn.MaxPool2d(2, 2)
        ])

        self.d2 = self._make_layer(Bottleneck, 64, 3)

        self.d3 = nn.Sequential(*[
            nn.MaxPool2d(2, 2),
            self._make_layer(Bottleneck, 64, 3)
        ])

        self.d4 = self._make_layer(Bottleneck, 64, 3)

        self.d5 = nn.ConvTranspose2d(256, 256, 2, 2)


        self.d6 = nn.Conv2d(256, 512, 1, 1)

        self.d66 = nn.Conv2d(512, 256, 1, 1)

        self.d7 = nn.Conv2d(256, 16, 1)

        self.d8 = nn.ConvTranspose2d(16, 16, 8, 8)

    def forward(self, input):
        x = self.d1(input)
        x = self.d2(x)

        t1 = self.d3(x)
        f1 =  self.d4(t1)

        t2 = self.d3(t1)
        f2 = self.d4(t2)

        t3 = self.d3(t2)
        f3 = self.d4(t3)

        t4 = self.d3(t3)
        f4 = self.d4(t4)

        output = self.d5(self.d3(t4)) + f4
        output = self.d5(output) + f3
        output = self.d5(output) + f2
        output = self.d5(output) + f1

        output = self.d6(output)
        output = self.d66(output)
        output = self.d7(output)

        output = self.d8(output)
        return output

    def make_bottlenecks(self, inplanes, planes):
        bott = []
        for i in range(3):
            bott.append(Bottleneck(inplanes, planes))
        return bott

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
