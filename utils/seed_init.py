#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :seed_init.py
@Description  :
@Date         :2021/10/17 16:51:06
@Author       :Arctic Little Pig
@Version      :1.0
'''

import os
import random

import numpy as np
import torch


def init_seed(seed: int, cuda: bool, gpus: str) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if cuda:
        print("using GPU to train.")

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus

        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
