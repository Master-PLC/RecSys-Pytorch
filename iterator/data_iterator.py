#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :data_iterator.py
@Description  :
@Date         :2021/10/21 18:58:53
@Author       :Arctic Little Pig
@Version      :1.0
'''

import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# def collate_func(batch):
#     # print(batch)
#     # print(len(batch))
#     # print(input_split)

#     input = [item[0] for item in batch]
#     label = [item[1] for item in batch]
#     # print(input, label)
#     input = np.array(input)
#     label = np.array(label)
#     # label = torch.tensor(label)

#     return input, label


class DataIterator(Dataset):
    def __init__(self, source, shuffle=False, maxlen=20, train_flag=True, seq_length=1):
        self.shuffle = shuffle
        self.maxlen = maxlen
        self.train_flag = train_flag
        self.seq_len = seq_length
        self.read(source)

    def __len__(self):
        return self.edge_count

    def __getitem__(self, idx):
        # if self.train_flag:

        if idx + self.seq_len > self.__len__():
            hist_item = torch.zeros(self.seq_len, self.maxlen)
            nbr_mask = torch.zeros(self.seq_len, self.maxlen)
            item_id = torch.zeros(self.seq_len, 1)

            hist_item[:self.__len__() -
                      idx] = torch.from_numpy(self.hist_item_list[idx:])
            nbr_mask[:self.__len__() -
                     idx] = torch.from_numpy(self.hist_mask_list[idx:])
            item_id[:self.__len__()-idx] = torch.from_numpy(self.item_id_list[idx:])
        else:
            hist_item = torch.from_numpy(
                self.hist_item_list[idx:idx+self.seq_len])
            nbr_mask = torch.from_numpy(
                self.hist_mask_list[idx:idx+self.seq_len])
            item_id = torch.from_numpy(self.item_id_list[idx:idx+self.seq_len])

        return hist_item, nbr_mask, item_id

    def read(self, source):
        self.graph = {}
        conts = pd.read_table(source, sep=",", encoding='utf-8',
                              names=["user_id", "item_id", "time_stamp"])
        self.line_cnt = len(conts)
        self.users = conts.user_id.unique()
        self.total_user = len(self.users)
        self.items = conts.item_id.unique()
        for user in self.users:
            self.graph[user] = conts[conts.user_id.isin(
                [user])].values.tolist()

        del conts

        self.edges = []
        for user_id, value in self.graph.items():
            self.edges.extend(value[3:])
        # self.edges = torch.Tensor(self.edges)
        self.edge_count = len(self.edges)

        self._getdata()

    def _shuffle(self):
        np.random.shuffle(self.edges)

    def _getdata(self):
        if self.shuffle:
            self._shuffle()

        item_time_list = []
        if self.train_flag:
            user_id_list, item_id_list, item_time_list = zip(*self.edges)
        else:
            user_id_list = self.users[:self.total_user]
            item_id_list = [self.graph[user_][-1][1] for user_ in user_id_list]
        self.item_id_list = np.array(item_id_list)

        hist_item_list = []
        hist_mask_list = []
        for i, user_id in enumerate(user_id_list):
            item_list = self.graph[user_id]
            item_ = [_item[1] for _item in item_list]
            if self.train_flag:
                try:
                    k = item_list.index(
                        [user_id_list[i], self.item_id_list[i], item_time_list[i]])
                except ValueError:
                    print(f"({user_id_list[i]},{self.item_id_list[i]},{item_time_list[i]})")
                    print(item_list)
            else:
                k = item_list.index(item_list[-1])

            if k >= self.maxlen:
                hist_item_list.append(item_[k-self.maxlen: k])
                hist_mask_list.append([1.0] * self.maxlen)
            else:
                hist_item_list.append(item_[:k] + [0] * (self.maxlen - k))
                hist_mask_list.append(
                    [1.0] * k + [0.0] * (self.maxlen - k))

        self.hist_item_list = np.array(hist_item_list)
        self.hist_mask_list = np.array(hist_mask_list)
