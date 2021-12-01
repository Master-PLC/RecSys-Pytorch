#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :common_utils.py
@Description  :
@Date         :2021/11/19 18:24:14
@Author       :Arctic Little Pig
@Version      :1.0
'''


def build_unigram_noise(freq):
    """build the unigram noise from a list of frequency
    Parameters:
        freq: a tensor of #occurrences of the corresponding index
    Return:
        unigram_noise: a torch.Tensor with size ntokens,
        elements indicate the probability distribution
    """
    total = freq.sum()
    noise = freq / total
    assert abs(noise.sum() - 1) < 0.001

    return noise
