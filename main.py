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
import os

from data.preprocess import preprocess
from utils.seed_init import init_seed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pytorch Practice of Recommendation Systems.")
    parser.add_argument('--init_seed', type=bool, default=True,
                        help="Whether to initialize the seed(default: 1024).")
    parser.add_argument('--datafile_dir', type=str, default="./data",
                        help="The folder where the data files are located.")
    parser.add_argument('--dataset', type=str, default="ml-1m",
                        help="Name of the dataset to be used.")
    parser.add_argument('--raw_data', type=str, default="raw",
                        help="The folder where th raw data is located.")
    parser.add_argument('--processed_data', type=str, default="processed",
                        help="The folder where th processed data is located.")
    parser.add_argument('--preprocess', type=bool, default=True,
                        help="Whether to preprocess the data.")
    parser.add_argument('--embedding_dim', type=int,
                        default=128, help="Dimension of embedding.")
    parser.add_argument('--item_count', type=int,
                        default=1000, help="Number of items.")
    parser.add_argument('--hidden_size', type=int,
                        default=512, help="Dimension of hidden layer.")
    parser.add_argument('--category_num', type=int,
                        default=2, help="Number of category.")
    parser.add_argument('--topic_num', type=int,
                        default=10, help="Number of topics.")
    parser.add_argument('--neg_num', type=int, default=10,
                        help="Number of negative samples.")
    parser.add_argument('--cpt_feat', type=int, default=1, help="")
    parser.add_argument('--model_type', type=str, default='SINE',
                        help='Name of model for recommendation system.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate of learning algorithm.')
    parser.add_argument('--alpha', type=float, default=0.0, help='')
    parser.add_argument('--batch_size', type=int,
                        default=128, help='Size of mini-batch.')
    parser.add_argument('--maxlen', type=int, default=20, help='Max length.')
    parser.add_argument('--epoch', type=int, default=30,
                        help='Number of epoches for training.')
    parser.add_argument('--patience', type=int, default=5, help="")
    parser.add_argument('--coef', default=None)
    parser.add_argument('--test_iter', type=int, default=50,
                        help="Iterations of test.")
    parser.add_argument('--user_norm', type=int, default=0,
                        help="Norm of user data.")
    parser.add_argument('--item_norm', type=int, default=0,
                        help="Norm of item data.")
    parser.add_argument('--cate_norm', type=int, default=0,
                        help="Norm of category.")
    parser.add_argument('--n_head', type=int, default=1,
                        help="Number of heads for attention.")
    config = parser.parse_args()

    if config.init_seed:
        init_seed()

    if not config.preprocess:
        data_dir = os.path.join(config.datafile_dir, config.dataset)
        raw_data_dir = os.path.join(data_dir, config.raw_data)
        processed_data_dir = os.path.join(data_dir, config.processed_data)

        preprocess(raw_data_dir, processed_data_dir, config.dataset)
