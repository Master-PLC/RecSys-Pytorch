#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :read_utils.py
@Description  :
@Date         :2021/10/18 12:15:53
@Author       :Arctic Little Pig
@Version      :1.0
'''

import os

import pandas as pd
import ujson
from numpy import exp

DATA_TYPE = ["train", "valid", "test"]


def check_dir(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)


def preprocess_movielens(raw_data_dir, filter_size, processed_data_dir):
    df_ratings, item_count, num_users = read_from_movielens(raw_data_dir)
    item_count, item_map, item_count_map = filter_items(item_count, filter_size)
    # print(item_count)
    # print(item_count_map)
    df_ratings, user_map = filter_users(
        df_ratings, item_map, num_users, filter_size)

    check_dir(processed_data_dir)
    user_map_path = os.path.join(processed_data_dir, "user_map.json")
    export_map(user_map_path, user_map)
    item_map_path = os.path.join(processed_data_dir, "item_map.json")
    export_map(item_map_path, item_map)
    item_count_map_path = os.path.join(processed_data_dir, "item_count_map.json")
    export_map(item_count_map_path, item_count_map)
    # print(item_map)

    for i, data_type in enumerate(DATA_TYPE):
        file_path = os.path.join(processed_data_dir, data_type+".txt")
        print(f"start to prepare {data_type}.txt.")
        ix_num = export_data(file_path, df_ratings,
                             user_map, item_map, max_time=2-i)
        print(f'total interactions for {data_type}={ix_num}.')
        # break


def read_from_movielens(raw_data_dir):
    filename = "ratings.dat"
    file_path = os.path.join(raw_data_dir, filename)

    df_ratings = pd.read_table(file_path, names=[
                               'user_id', 'item_id', 'rating', 'timestamp'], sep='::', encoding='utf-8', engine='python')

    item_count = df_ratings.item_id.value_counts()
    num_users = df_ratings.user_id.nunique()

    return df_ratings, item_count, num_users


def filter_items(item_count, filter_size):
    item_total = len(item_count[item_count >= filter_size])
    item_count = item_count[:item_total]
    item_map = dict(
        zip(item_count.index, list(range(item_total))))
    item_count_map = dict(
        zip(list(range(item_total)), item_count.values.tolist()))
    return item_count, item_map, item_count_map


def filter_users(df_ratings, item_map, num_users, filter_size):
    for user in range(num_users):
        index = 0
        user_seq = df_ratings[df_ratings.user_id.isin([user+1])]
        for item in user_seq.item_id:
            # print(item)
            if item in item_map:
                index += 1
        if index < filter_size:
            df_ratings = df_ratings[~df_ratings.user_id.isin([user+1])]

    num_users = df_ratings.user_id.nunique()
    # print(num_users)
    user_map = dict(zip(df_ratings.user_id.unique()-1, list(range(num_users))))

    return df_ratings, user_map


def export_map(filename, map_dict):
    with open(filename, 'w') as f:
        ujson.dump(map_dict, f)


def export_data(filename, user_seq, user_map, item_map, max_time=2):
    user_list = user_seq.user_id.unique()
    total_data = 0
    with open(filename, 'w') as f:
        for user in user_list:
            if user-1 not in user_map:
                continue

            item_list = user_seq[user_seq.user_id.isin([user])]
            item_list = item_list.loc[:, ["item_id", "timestamp"]]
            item_list = item_list[item_list.item_id.isin(item_map.keys())]
            item_list = item_list.sort_values(by="timestamp")
            item_list["timestamp"] = list(range(1, len(item_list)+1))
            # print(item_list)
            if max_time == 2:
                item_list = item_list.iloc[0:-2, :]
            elif max_time == 1:
                item_list = item_list.iloc[0:-1, :]
                if len(item_list) > 100:
                    item_list = item_list.iloc[-100:, :]
            else:
                item_list = item_list
                if len(item_list) > 100:
                    item_list = item_list.iloc[-100:, :]
            # print(item_list)

            for index, row in item_list.iterrows():
                item, timestamp = row.to_list()
                f.write(f'{user_map[user-1]},{item_map[item]},{timestamp}\n')
                total_data += 1
            # break

    return total_data


if __name__ == "__main__":
    raw_data_dir = "ml-1m/raw"
    filter_size = 1
    processed_data_dir = "ml-1m/processed"
    preprocess_movielens(raw_data_dir, filter_size, processed_data_dir)
