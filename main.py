#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :main.py
@Description  :
@Date         :2021/10/17 16:36:35
@Author       :Arctic Little Pig
@Version      :1.0
'''

import argparse

from utils.seed_init import init_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pytorch Practice of Recommendation Systems.")
    parser.add_argument('--datafile_path', type=str, default="./data",
                        help="The folder where the data files are located.")
    parser.add_argument('--init_seed', type=bool, default=True,
                        help="Whether to initialize the seed(default: 1024).")
    config = parser.parse_args()

    if config.init_seed:
        init_seed()
    
