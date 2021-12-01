#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :optim_map.py
@Description  :
@Date         :2021/11/17 13:50:35
@Author       :Arctic Little Pig
@Version      :1.0
'''

import torch.nn as nn
import torch.optim as optim

NAME_TO_LOSS_FUNC_MAP = {
    "SelfDefined": None,
    "CrossEntropy": nn.CrossEntropyLoss()
}


def get_optimizer(model, config):
    optim_name = config.optim_name
    if optim_name == "Adam":
        print("---> Adam optimizer is used for Training...")
        optimizer = optim.Adam(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    elif config.SGD is True:
        print("---> SGD optimizer is used for Training...")
        optimizer = optim.SGD(model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
                              momentum=config.momentum)
    elif config.Adadelta is True:
        print("---> Adadelta optimizer is used for Training...")
        optimizer = optim.Adadelta(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    return optimizer


def get_loss_function(loss_function):
    return NAME_TO_LOSS_FUNC_MAP[loss_function]
