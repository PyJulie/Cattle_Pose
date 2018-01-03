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

# test dataset
handle = open('data/test/tensor.lable', 'r+')
body = handle.read()
test_in_json = json.loads(body)
test_data_length = len(test_in_json)
handle.close()

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

def read_test_img(idx):
    """Read image from vid, and formate lable

    """
    png = Image.open("{}{}.jpg".format('data/test/', idx))
    png.load()  # required for png.split()
    background = Image.new("RGB", png.size, (255, 255, 255))
    background.paste(png, mask=png.split()[3])  # 3 is the alpha channel
    lable = np.array(test_in_json[idx][2])
    return (np.array(background), lable)

class Dataset(Dataset):
    """Read Data from movie and lable

    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = "train"

    def __len__(self):
        if self.mode == "train":
            return dataload_config["data_length"]
        else:
            return test_data_length

    def __getitem__(self, idx):
        if self.mode == "train":
            sample = read_img(idx)
        else:
            sample =read_test_img(idx)
        sample = {"image": sample[0], "landmarks": sample[1]}
        if self.transform:
            sample = self.transform(sample)
        return sample



class Rescale(object):
    def __init__(self, out_size):
        assert isinstance(out_size, (int, tuple))
        self.out_size = out_size
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        h, w = image.shape[:2]
        if isinstance(self.out_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.out_size
        img = transform.resize(image, (new_h, new_w))
        landmarks = landmarks * [new_w / w, new_h / h]
        return {'image': img, 'landmarks': landmarks}

class RandomCrop(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}

class Rotation(object):
    def __call__(self, sample):
        angle = np.random.randint(-20, 20)
        image, landmarks = sample['image'], sample['landmarks']
        image = transform.rotate(image, angle)
        landmarks[0] = rotate(landmarks[0], landmarks[1], [image.shape[1] / 2 - 0.5,
                              image.shape[0] / 2 - 0.5], angle, True)
        landmarks[1] = rotate(landmarks[0], landmarks[1], [image.shape[1] / 2 - 0.5,
                                                           image.shape[0] / 2 - 0.5], angle, False)

        return {'image': image, 'landmarks': landmarks}


class ToTensor(object):

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        shape = image.shape
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image)
        image = image.type_as(torch.FloatTensor())
        return {'image': image,
                'landmarks': (torch.FloatTensor(binnary_heatmap(landmarks, shape[0], shape[1],
                                                               len(app["nummeric_level"]),
                                                                dataload_config['heatmap'])),

                              torch.FloatTensor(G_P(shape[0], shape[1], landmarks,
                                                   dataload_config["standard_deviation"])),
                              landmarks)}
class TestToTensor(object):

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        shape = image.shape
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image)
        image = image.type_as(torch.FloatTensor())
        return {'image': image,
                'landmarks': (torch.FloatTensor(binnary_heatmap(landmarks, shape[0], shape[1],
                                                                len(app["nummeric_level"]),
                                                                dataload_config['heatmap'])),

                              torch.FloatTensor(G_P(shape[0], shape[1], landmarks,
                                                    dataload_config["standard_deviation"])),
                              landmarks)}

transformed_dataset = Dataset(root_dir=app["dataload"]["root_dir"],
                                           transform=torch_transform.Compose([
                                               Rescale((dataload_config["h"], dataload_config["w"])),
                                               RandomCrop((dataload_config["new_h"], dataload_config["new_w"])),
                                               Rotation(),
                                               ToTensor()
                                           ]))

transformed_test_dataset = Dataset(root_dir=app["dataload"]["root_dir"],
                                           transform=torch_transform.Compose([
                                               Rescale((dataload_config["h"], dataload_config["w"])),
                                               RandomCrop((dataload_config["new_h"], dataload_config["new_w"])),
                                               TestToTensor()
                                           ]))
transformed_test_dataset.mode = "test"
