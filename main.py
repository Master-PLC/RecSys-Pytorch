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
import time

import torch
import ujson
from torch.utils.data import DataLoader

from config.config import Configurable
from data.preprocess import preprocess
from data.utils import check_dir
from iterator.data_iterator import DataIterator
from metrics.metrics_rs import evaluate_full
from utils.common_utils import build_unigram_noise
from utils.model_map import get_model_class, get_model_prefix
from utils.seed_init import init_seed
from utils.tool_map import get_loss_function, get_optimizer


def data_preprocess():
    """
    Description::
    
    :param :
    
    :returns :
    
    Usage::
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
    
    if config.nce:
        with open(config.noise_fp, "r") as f:
            item_count = ujson.load(f)
        freq = list(item_count.values())
        noise = build_unigram_noise(torch.FloatTensor(freq))
        config.noise = noise


def load_data():
    train_dataset = DataIterator(config.train_fp, shuffle=config.shuffle)
    valid_dataset = DataIterator(config.valid_fp, train_flag=False)
    test_dataset = DataIterator(config.test_fp, train_flag=False)

    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size,
                              num_workers=config.num_workers, pin_memory=config.pin_memory, drop_last=False)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=config.eval_batch_size,
                              num_workers=config.num_workers, pin_memory=config.pin_memory, drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.eval_batch_size,
                             num_workers=config.num_workers, pin_memory=config.pin_memory, drop_last=True)
    
    return train_loader, valid_loader, test_loader


def load_model(sample_dist=None):
    model = None
    model_name = config.model_name
    print(f'---> Experiment model - {model_name}: ')
    model_class = get_model_class(model_name)

    model = model_class(config)
    print(model)

    if config.cuda:
        # if torch.cuda.device_count() > 1:
        #    model = nn.DataParallel(model)
        model.cuda()

    return model


def train(model, train_loader, valid_loader, test_loader):
    # print("\n cpu_count \n", mu.cpu_count())
    # torch.set_num_threads(config.num_threads)

    print("---> Start training...")
    best_metric = 0
    best_metric_ndcg = 0
    best_epoch = 0

    model.train()
    optimizer = get_optimizer(model, config)
    loss_function = get_loss_function(config.loss_function)

    for epoch in range(config.epochs):
        torch.cuda.empty_cache()

        print(f'---> Epoch {epoch}/{config.epochs}')
        step = 0
        
        trials = 0
        iter = 0
        loss_iter = 0.0
        start_time = time.time()
        for i, (hist_item, nbr_mask, i_ids) in enumerate(train_loader):
            if config.cuda:
                hist_item = hist_item.cuda()
                nbr_mask = nbr_mask.cuda()
                i_ids = i_ids.cuda()

            optimizer.zero_grad()
            if loss_function is not None:
                output = model(i_ids, hist_item, nbr_mask)
                loss = loss_function(output, i_ids)
            else:
                loss = model(i_ids, hist_item, nbr_mask)
            # loss = loss.requires_grad_()
            loss_iter += loss.item()
            if step % config.log_inr == 0:
                print(
                    f'---> Epoch {epoch}/{config.epochs} at step {step} loss {loss.item()}')

            loss.backward()
            # optimizer.step()
            step += 1
            break

        metrics = evaluate_full(None, test_loader, model, config.embedding_dim)
        for k in range(len(config.topk)):
            print(f'!!!! Test result epoch {epoch} topk={config.topk[k]} hitrate={metrics["hitrate"][k]:.4f} ndcg={metrics["ndcg"][k]:.4f}')
        metrics = evaluate_full(
            None, valid_loader, model, config.embedding_dim)
        for k in range(len(config.topk)):
            print(f'!!!! Validate result topk={config.topk[k]} hitrate={metrics["hitrate"][k]:.4f} ndcg={metrics["ndcg"][k]:.4f}')

        if 'hitrate' in metrics:
            hitrate = metrics['hitrate'][0]
            # ndcg = metrics['ndcg'][0]
            hitrate2 = metrics['ndcg'][1]
            if hitrate >= best_metric and hitrate2 >= best_metric_ndcg:
                best_metric = hitrate
                best_metric_ndcg = hitrate2
                # best_metric_ndcg = ndcg
                torch.save(model.state_dict(), config.save_path)
                trials = 0
                best_epoch = epoch
                print(
                    f'---> Current best valid hitrate={best_metric:.4f} ndcg={best_metric_ndcg:.4f}')
            else:
                trials += 1
                if trials > config.patience:
                    break

        test_time = time.time()
        print(
            f"time interval for one epoch: {(test_time-start_time)/60.0:.4f} min")
        break
    pass
    print(f'---> Best epoch is {best_epoch}')


def test():
    best_model_path = config.save_path
    model = load_model()
    model.load_state_dict(torch.load(best_model_path))
    print('---> Start testing...')

    test_dataset = DataIterator(config.test_fp, train_flag=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.eval_batch_size,
                             num_workers=config.num_workers, pin_memory=config.pin_memory, drop_last=True)

    metrics = evaluate_full(None, test_loader, model, config.embedding_dim)
    for k in range(len(config.topk)):
        print(f'!!!! Test result topk={config.topk[k]} hitrate={metrics["hitrate"][k]:.4f} ndcg={metrics["ndcg"][k]:.4f}')


def main():
    """
    Description::主函数
    """

    data_preprocess()
    update_arguments()
    # print(config.noise)
    # train_loader, valid_loader, test_loader = load_data()
    # model = load_model()
    # train(model, train_loader, valid_loader, test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pytorch Practice of Recommendation Algorithms.")
    parser.add_argument(
        "--config_file", default="./config/configurations.cfg", help="Parameter file address.")
    config = parser.parse_args()

    config = Configurable(config_file=config.config_file)
    if config.init_seed:
        init_seed(config.seed, config.cuda, config.gpus)

    main()
