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

from torch.utils.data import DataLoader

from config.config import Configurable
from data.preprocess import preprocess
from data.utils import check_dir
from iterator.data_iterator import DataIterator
from utils.model_map import get_model_class, get_model_prefix
from utils.seed_init import init_seed


def data_preprocess():
    """
    Description::对原始数据集进行前处理，生成训练/验证/测试样本
    """

    if not config.preprocess:
        preprocess(config.raw_data_dir,
                   config.processed_data_dir, config.dataset)


def update_arguments():
    check_dir(config.save_dir)
    config.save_dir = os.path.join(config.save_dir, config.dataset)
    check_dir(config.save_dir)
    config.save_dir = os.path.join(config.save_dir, config.model_name.lower())
    check_dir(config.save_dir)

    if config.dataset == 'taobao':
        config.item_count = 1708531
        config.test_iter = 1000
    if config.dataset == 'ml1m':
        config.category_num = 2
        config.topic_num = 10
        config.item_count = 3706
        config.test_iter = 500
    elif config.dataset == 'book':
        config.item_count = 367983
        config.test_iter = 1000

    best_model_prefix = get_model_prefix(config)
    config.save_path = os.path.join(
        config.save_dir, best_model_prefix+config.suffix)


def load_data():
    train_dataset = DataIterator(config.train_fp, shuffle=config.shuffle)
    valid_dataset = DataIterator(config.valid_fp)
    test_dataset = DataIterator(config.test_fp)

    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size,
                              num_workers=config.num_workers, pin_memory=config.pin_memory, drop_last=False)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=config.eval_batch_size,
                              num_workers=config.num_workers, pin_memory=config.pin_memory, drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.eval_batch_size,
                             num_workers=config.num_workers, pin_memory=config.pin_memory, drop_last=True)

    return train_loader, valid_loader, test_loader


def load_model():
    model = None
    model_name = config.model_name
    print(f"loading {model_name} model......")
    model_class = get_model_class(model_name)
    model = model_class(config)
    print(model)

    if config.cuda:
        # if torch.cuda.device_count() > 1:
        #    model = nn.DataParallel(model)
        model = model.cuda()

    return model


def start_train(model, train_loader, valid_loader, test_loader):
    # print("\n cpu_count \n", mu.cpu_count())
    # torch.set_num_threads(config.num_threads)
    model.train()
    # if os.path.exists("./Test_Result.txt"):
    #     os.remove("./Test_Result.txt")
    print(f"{config.model_name} training start......")


def main():
    """
    Description::主函数
    """

    data_preprocess()
    update_arguments()
    train_loader, valid_loader, test_loader = load_data()
    model = load_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pytorch Practice of Recommendation Systems.")
    parser.add_argument(
        "--config_file", default="./config/configurations.cfg", help="Parameter file address.")
    config = parser.parse_args()

    config = Configurable(config_file=config.config_file)
    if config.init_seed:
        init_seed(config.seed, config.cuda, config.gpus)

    main()

    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu
