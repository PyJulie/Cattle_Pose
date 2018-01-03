#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@author toby
'''
from __future__ import print_function, division
import torch
import os
from skimage import io, transform
import numpy as np
import glob
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as torch_transform, utils
import random
from torch.autograd import Variable
from config import app
import json
import math
from utils import binnary_heatmap, G_P, rotate
import warnings
from PIL import Image


warnings.filterwarnings("ignore")

dataload_config = app["dataload"]

handle = open('data/train/tensor.lable', 'r+')
body = handle.read()
in_json = json.loads(body)
dataload_config["data_length"] = len(in_json)

def find_files(path): return glob.glob(path)

def get_img_name(idx):
    return "name"

def read_img(idx):
    """Read image from vid, and formate lable

    """
    png = Image.open("{}{}.jpg".format(app["dataload"]["root_dir"], idx))
    png.load()  # required for png.split()
    background = Image.new("RGB", png.size, (255, 255, 255))
    background.paste(png, mask=png.split()[3])  # 3 is the alpha channel
    lable = np.array(in_json[idx][2])
    return (np.array(background), lable)

for i in range(dataload_config["data_length"]):
    try:
        read_img(i)
    except:
        print(i)