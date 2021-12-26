#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :model_map.py
@Description  :
@Date         :2021/10/21 14:59:41
@Author       :Arctic Little Pig
@Version      :1.0
'''

from models.sine import SINE

NAME_TO_MODEL_MAP = {
    "SINE": SINE
}


def get_model_class(model_name: str) -> object:
    return NAME_TO_MODEL_MAP[model_name]


def get_model_prefix(config):
    if config.model_name == "SINE":
        best_model_prefix = f"{config.dataset}_{config.model_name.lower()}_topic{config.topic_num} \
                            _cept{config.category_num}_len{config.maxlen}_neg{config.neg_num} \
                            _unorm{config.user_norm}_inorm{config.item_norm}_catnorm{config.cate_norm} \
                            _head{config.n_head}_alpha{config.alpha}"

    return best_model_prefix
