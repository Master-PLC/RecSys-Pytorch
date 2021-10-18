#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :seed_init.py
@Description  :
@Date         :2021/10/17 16:51:06
@Author       :Arctic Little Pig
@Version      :1.0
'''

import random

import numpy as np
import torch

SEED = 1024


def init_seed():
    random.seed(SEED)
    np.random.seed(SEED)

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
