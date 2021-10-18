#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :preprocess.py
@Description  :
@Date         :2021/10/18 19:38:48
@Author       :Arctic Little Pig
@Version      :1.0
'''

import os

from .utils import preprocess_movielens

NAME_TO_FUNC = {
    "ml-1m": preprocess_movielens
}
LIMIT_FOR_DATASET = {
    "ml-1m": 1,
}


def preprocess(raw_data_dir, processed_data_dir, dataset):
    preprocess_func = NAME_TO_FUNC[dataset]
    filter_size = LIMIT_FOR_DATASET[dataset]

    preprocess_func(raw_data_dir, filter_size, processed_data_dir)


if __name__ == "__main__":
    data_path = "."
    dataset = "ml-1m"
    preprocess(data_path, dataset)
