#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File         :config.py
@Description  :
@Date         :2021/10/20 22:31:55
@Author       :Arctic Little Pig
@Version      :1.0
"""

import ast
import os
from configparser import ConfigParser


class MyConfigParser(ConfigParser):
    def __init__(self, defaults=None, inline_comment_prefixes="#"):
        super(MyConfigParser, self).__init__(defaults=defaults,
                                             inline_comment_prefixes=inline_comment_prefixes)

    def optionxform(self, optionstr):
        return optionstr


class Configurable(MyConfigParser):
    def __init__(self, config_file):
        config = MyConfigParser()
        config.read(config_file)
        self._config = config
        self.config_file = config_file

        print('---> Experiment configuration: ')
        self._config_show = config.getboolean("Data", "config_show")
        self.sections = config.sections()
        if self._config_show:
            for section in config.sections():
                print(f"[{section}]")
                for k, v in config.items(section):
                    print(f"{k}: {v}")
                print("")

        config.write(open(config_file, "w"))

    # Section: Seed
    @property
    def init_seed(self):
        return self._config.getboolean("Seed", "init_seed")

    @property
    def seed(self):
        return self._config.getint("Seed", "seed")

    # Section: Data
    @property
    def config_show(self):
        return self._config_show

    @property
    def preprocess(self):
        return self._config.getboolean("Data", "preprocess")

    @property
    def data_dir(self):
        return self._config.get("Data", "data_dir")

    @property
    def dataset(self):
        return self._config.get("Data", "dataset")

    @property
    def raw_data_dir(self):
        _raw_data_dir = self._config.get("Data", "raw_data_dir")
        return os.path.join(self.data_dir, self.dataset, _raw_data_dir)

    @property
    def processed_data_dir(self):
        _processed_data_dir = self._config.get("Data", "processed_data_dir")
        return os.path.join(self.data_dir, self.dataset, _processed_data_dir)

    @property
    def train_fp(self):
        train_fn = self._config.get("Data", "train_filename")
        return os.path.join(self.processed_data_dir, train_fn)

    @property
    def valid_fp(self):
        valid_fn = self._config.get("Data", "valid_filename")
        return os.path.join(self.processed_data_dir, valid_fn)

    @property
    def test_fp(self):
        test_fn = self._config.get("Data", "test_filename")
        return os.path.join(self.processed_data_dir, test_fn)

    @property
    def noise_fp(self):
        noise_fn = self._config.get("Data", "noise_filename")
        return os.path.join(self.processed_data_dir, noise_fn)

    @property
    def shuffle(self):
        return self._config.getboolean("Data", "shuffle")

    @property
    def save_img(self):
        return self._config.getboolean("Data", "save_img")

    # Section: Model
    @property
    def model_name(self):
        return self._config.get("Model", "model_name")

    @property
    def embedding_dim(self):
        return self._config.getint("Model", "embedding_dim")

    @property
    def item_count(self):
        return self._config.getint("Model", "item_count")

    @item_count.setter
    def item_count(self, value):
        self._config.set("Model", "item_count", value)

    @property
    def hidden_size(self):
        return self._config.getint("Model", "hidden_size")

    @property
    def category_num(self):
        return self._config.getint("Model", "category_num")

    @category_num.setter
    def category_num(self, value):
        self._config.set("Model", "category_num", value)

    @property
    def topic_num(self):
        return self._config.getint("Model", "topic_num")

    @topic_num.setter
    def topic_num(self, value):
        self._config.set("Model", "topic_num", value)

    @property
    def neg_num(self):
        return self._config.getint("Model", "neg_num")

    @property
    def cpt_feat(self):
        return self._config.getint("Model", "cpt_feat")

    @property
    def alpha(self):
        return self._config.getfloat("Model", "alpha")

    @property
    def maxlen(self):
        return self._config.getint("Model", "maxlen")

    @property
    def patience(self):
        return self._config.getint("Model", "patience:")

    @property
    def coef(self):
        value = self._config.get("Model", "coef")
        if value == "None" or value == "none":
            return None
        else:
            return value

    @property
    def user_norm(self):
        return self._config.getfloat("Model", "user_norm")

    @property
    def item_norm(self):
        return self._config.getfloat("Model", "item_norm")

    @property
    def cate_norm(self):
        return self._config.getfloat("Model", "cate_norm")

    @property
    def noise_norm(self):
        return self._config.getfloat("Model", "noise_norm")

    @property
    def n_head(self):
        return self._config.getint("Model", "n_head")

    @property
    def share_emb(self):
        return self._config.getboolean("Model", "share_emb")

    @property
    def flag(self):
        return self._config.get("Model", "flag")

    @property
    def topk(self):
        _str = self._config.get("Model", "topk")
        return ast.literal_eval(_str)

    # Section: Optimizer
    @property
    def optim_name(self):
        return self._config.get("Optimizer", "optim_name")

    @property
    def lr(self):
        return self._config.getfloat("Optimizer", "learning_rate")

    @property
    def momentum(self):
        return self._config.getfloat("Optimizer", "momentum")

    @property
    def weight_decay(self):
        return self._config.getfloat("Optimizer", "weight_decay")

    # Section: Loss
    @property
    def nce(self):
        return self._config.getboolean("Loss", "nce")

    @property
    def loss_type(self):
        return self._config.get("Loss", "loss_type")

    @property
    def loss_function(self):
        return self._config.get("Loss", "loss_function")

    # Section: Train
    @property
    def num_workers(self):
        return self._config.getint("Train", "num_workers")

    @property
    def num_threads(self):
        return self._config.getint("Train", "num_threads")

    @property
    def pin_memory(self):
        return self._config.getboolean("Train", "pin_memory")

    @property
    def cuda(self):
        return self._config.getboolean("Train", "cuda")

    @property
    def gpus(self):
        return self._config.get("Train", "gpus")

    @property
    def batch_size(self):
        return self._config.getint("Train", "batch_size")

    @property
    def eval_batch_size(self):
        return self._config.getint("Train", "eval_batch_size")

    @property
    def seq_len(self):
        return self._config.getint("Train", "seq_len")

    @property
    def epochs(self):
        return self._config.getint("Train", "epochs")

    @property
    def log_inr(self):
        return self._config.getint("Train", "log_interval")

    @property
    def test_inr(self):
        return self._config.getint("Train", "test_interval")

    @property
    def test_iter(self):
        return self._config.getint("Train", "test_iter")

    @test_iter.setter
    def test_iter(self, value):
        self._config.set("Train", "test_iter", value)

    @property
    def epoch_inr(self):
        return self._config.getint("Train", "epoch_interval")

    # Section: Save
    @property
    def save_dir(self):
        return self._config.get("Save", "save_dir")

    @save_dir.setter
    def save_dir(self,  value):
        self._config.set("Save", "save_dir", str(value))

    @property
    def save_inr(self):
        return self._config.getint("Save", "save_interval")

    @property
    def suffix(self):
        return self._config.get("Save", "suffix")
