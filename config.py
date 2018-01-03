#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@author toby
'''
import torch
import json


app = {
    "version": 1.0,
    "train": {
        "epochs": {
            "epoch_size": 20,
            "total": 1000,
            "detector": 20,
            "regresstion": 20,
            "together": 10
        },
        "model": ["detector", "regresstion", "together"]
    },
    "dataload": {
        "w": 270,
        "h": 270,
        "new_w": 256,
        "new_h": 256,
        "root_dir": "data/train/",
        "N": 16,
        "G_H": 82,
        "G_W": 82,
        "data_length": 1000,
        "standard_deviation": 5,
        "heatmap": 5
    },
    "nummeric_level": [
        "background", "body",  "head", "neck",
        "left shoulder", "right shoulder",
        "left elbow", "right elbow",
        "left wrist", "right wrist",
        "left hip", "right hip",
        "left knee", "right knee",
        "left ankle", "right ankle"
    ],
    "use_cuda": torch.cuda.is_available()
}