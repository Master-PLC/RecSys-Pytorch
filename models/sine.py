#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :sine.py
@Description  :
@Date         :2021/10/18 20:29:21
@Author       :Arctic Little Pig
@Version      :1.0
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class SINE(nn.Module):
    def __init__(self, config, n_mid, embedding_dim, hidden_size, batch_size, seq_len,
                 topic_num, category_num, alpha, neg_num, cpt_feat,
                 user_norm, cate_norm, n_head,
                 share_emb=True, flag="DNN", item_norm=0):
        super(SINE, self).__init__()
        self.n_size = config.item_count
        self.dim = config.embedding_dim
        self.hidden_units = config.hidden_size
        self.batch_size = config.batch_size
        self.hist_max = config.maxlen
        self.num_topic = config.topic_num
        self.category_num = config.category_num
        self.alpha_para = config.alpha
        self.neg_num = config.neg_num
        if config.cpt_feat == 1:
            self.cpt_feat = True
        else:
            self.cpt_feat = False
        self.user_norm = config.user_norm
        self.item_norm = config.item_norm
        self.cate_norm = config.cate_norm
        self.num_heads = config.n_head
        self.share_emb = config.share_emb
        self.model_flag = config.flag
        self.temperature = 0.07
        # self.temperature = 0.1

        self.item_input_lookup = Variable(torch.randn(
            n_mid, embedding_dim), requires_grad=True)
        self.item_input_lookup_var = Variable(torch.zeros(n_mid))
        self.position_embedding = Variable(
            torch.randn(1, self.hist_max, embedding_dim), requires_grad=True)
        if self.share_emb:
            self.item_output_lookup = self.item_input_lookup
            self.item_output_lookup_var = self.item_input_lookup_var
        else:
            self.item_output_lookup = Variable(torch.randn(
                n_mid, embedding_dim), requires_grad=True)
            self.item_output_lookup_var = Variable(torch.zeros(n_mid))

        self.item_output_emb = self.output_item2()

        self.topic_embed = Variable(torch.randn(
            self.num_topic, self.dim), requires_grad=True)

    def forward(self, i_ids, item, nbr_mask):
        self.i_ids = i_ids
        self.item = item
        self.nbr_mask = nbr_mask

        emb = torch.index_select(
            self.item_input_lookup, 0, self.item.reshape([-1]))
        self.item_emb = emb.reshape([-1, self.hist_max, self.dim])
        self.mask_length = self.nbr_mask.sum(-1).type(dtype=torch.LongTensor)

        self.seq_multi = self.sequence_encode_cpt(self.item_emb, nbr_mask)
        self.user_eb = self.labeled_attention(self.seq_multi)

        pass

    def output_item2(self):
        if self.item_norm:
            item_emb = F.normalize(self.item_output_lookup, dim=-1, p=2)
            return item_emb
        else:
            return self.item_output_lookup

    def sequence_encode_cpt(self, items, nbr_mask):
        item_emb_input = items.reshape([-1, self.dim])
        self.cate_dist = self.seq_cate_dist(item_emb_input).reshape(
            [-1, self.hist_max, self.category_num]).permute([0, 2, 1])
        item_list_emb = item_emb_input.reshape([-1, self.hist_max, self.dim])
        item_list_add_pos = item_list_emb + self.position_embedding.repeat([
            item_list_emb.shape[0], 1, 1])

        fc1 = nn.Sequential(
            nn.Linear(item_list_add_pos.shape[1], self.hidden_units),
            nn.Tanh()
        )
        item_hidden = fc1(item_list_add_pos)
        fc2 = nn.Linear(item_hidden.shape[1],
                        self.num_heads * self.category_num)
        item_att_w = fc2(item_hidden)

        # [batch_size, category_num*num_head, hist_max]
        item_att_w = item_att_w.permute([0, 2, 1])

        # [batch_size, category_num, num_head, hist_max]
        item_att_w = item_att_w.reshape(
            [-1, self.category_num, self.num_heads, self.hist_max])

        category_mask_tile = self.cate_dist.unsqueeze(2).repeat(
            [1, 1, self.num_heads, 1])  # [batch_size, category_num, num_head, hist_max]
        # paddings = tf.ones_like(category_mask_tile) * (-2 ** 32 + 1)
        seq_att_w = torch.mul(item_att_w, category_mask_tile).reshape(
            [-1, self.category_num * self.num_heads, self.hist_max])

        atten_mask = nbr_mask.unsqueeze(1).reshape(
            [1, self.category_num * self.num_heads, 1])

        paddings = torch.ones_like(atten_mask) * (-2 ** 32 + 1)

        seq_att_w = torch.where(torch.eq(atten_mask, 0), paddings, seq_att_w)
        seq_att_w = seq_att_w.reshape(
            [-1, self.category_num, self.num_heads, self.hist_max])

        seq_att_w = F.softmax(seq_att_w)

        # here use item_list_emb or item_list_add_pos, that is a question
        item_emb = torch.matmul(seq_att_w, item_list_emb.unsqueeze(1).repeat(
            [1, self.category_num, 1, 1]))  # [batch_size, category_num, num_head, dim]

        # [batch_size, category_num, dim]
        category_embedding_mat = item_emb.reshape(
            [-1, self.num_heads, self.dim])
        if self.num_heads != 1:
            mu = torch.mean(category_embedding_mat, dim=1)  # [N,H,D]->[N,D]
            maha = nn.Linear(mu.shape[1], self.dim)
            mu = maha(mu)
            # (H,D)x(D,1) = [N,H,1]
            wg = torch.matmul(category_embedding_mat, mu.unsqueeze(-1))
            wg = F.softmax(wg, dim=1)  # [N,H,1]

            # seq = tf.reduce_mean(category_embedding_mat * wg, axis=1)  # [N,H,D]->[N,D]
            seq = category_embedding_mat * wg
            seq = seq.sum(1)  # [N,H,D]->[N,D]
        else:
            seq = category_embedding_mat
        self.category_embedding_mat = seq
        seq = seq.reshape([-1, self.category_num, self.dim])

        return seq

    def seq_cate_dist(self, input_seq):
        #     input_seq [-1, dim]
        top_logit, top_index = self.topic_select(input_seq)
        topic_embed = torch.index_select(self.topic_embed, 0, top_index)
        self.batch_tpt_emb = torch.index_select(
            self.topic_embed, 0, top_index)  # [-1, cate_num, dim]
        self.batch_tpt_emb = self.batch_tpt_emb * \
            top_logit.unsqueeze(2).repeat([1, 1, self.dim])
        norm_seq = F.normalize(
            input_seq, dim=1, p=2).unsqueeze(-1)  # [-1, dim, 1]
        cores = F.normalize(topic_embed, dim=-1, p=2)  # [-1, cate_num, dim]
        cores_t = cores.unsqueeze(1).repeat([1, self.hist_max, 1, 1]).reshape(
            [-1, self.category_num, self.dim])
        cate_logits = torch.matmul(cores_t, norm_seq).reshape(
            [-1, self.category_num]) / self.temperature  # [-1, cate_num]
        cate_dist = F.softmax(cate_logits, dim=-1)

        return cate_dist

    def topic_select(self, input_seq):
        seq = input_seq.reshape([-1, self.hist_max, self.dim])
        seq_emb = self.seq_aggre(seq, self.nbr_mask)
        if self.cate_norm:
            seq_emb = F.normalize(seq_emb, dim=-1, p=2)
            topic_emb = F.normalize(self.topic_embed, dim=-1, p=2)
            topic_logit = torch.matmul(seq_emb, topic_emb.transpose(-1, -2))
        else:
            # [batch_size, topic_num]
            topic_logit = torch.matmul(
                seq_emb, self.topic_embed.transpose(-1, -2))
        # two [batch_size, categorty_num] tensors
        top_logits, top_index = torch.topk(topic_logit, self.category_num)
        top_logits = F.sigmoid(top_logits)

        return top_logits, top_index

    def seq_aggre(self, item_list_emb, nbr_mask):
        num_aggre = 1
        item_list_add_pos = item_list_emb + \
            self.position_embedding.repeat([item_list_emb.shape[0], 1, 1])

        item_hidden_layer = nn.Sequential(
            nn.Linear(item_list_add_pos.shape[1], self.hidden_units),
            nn.Tanh()
        )
        item_hidden = item_hidden_layer(item_list_add_pos)
        item_att_w_layer = nn.Linear(item_hidden.shape[1], num_aggre)
        item_att_w = item_att_w_layer(item_hidden)
        item_att_w = item_att_w.t([0, 2, 1])

        atten_mask = nbr_mask.unsqueeze(1).repeat([1, num_aggre, 1])

        paddings = torch.ones_like(atten_mask) * (-2 ** 32 + 1)

        item_att_w = torch.where(torch.eq(atten_mask, 0), paddings, item_att_w)

        item_att_w = F.softmax(item_att_w)

        item_emb = torch.matmul(item_att_w, item_list_emb)

        item_emb = item_emb.reshape([-1, self.dim])

        return item_emb

    def labeled_attention(self, seq):
        # item_emb = tf.reshape(self.cate_dist, [-1, self.hist_max, self.category_num])
        item_emb = self.cate_dist.permute([0, 2, 1])
        item_emb = torch.matmul(item_emb, self.batch_tpt_emb)

        if self.cpt_feat:
            item_emb = item_emb + \
                self.item_emb.reshape([-1, self.hist_max, self.dim])
        target_item = self.sequence_encode_concept(
            item_emb, self.nbr_mask)  # [N,  D]

        mu_seq = torch.mean(seq, dim=1)  # [N,H,D] -> [N,D]
        target_label = torch.cat([mu_seq, target_item], dim=1)

        maha_cpt2 = nn.Linear(target_label.shape[1], self.dim)
        mu = maha_cpt2(target_label)

        wg = torch.matmul(seq, mu.unsqueeze(-1))  # (H,D)x(D,1)
        wg = F.softmax(wg, dim=1)

        user_emb = torch.sum(seq * wg, dim=1)  # [N,H,D]->[N,D]
        if self.user_norm:
            user_emb = F.normalize(user_emb, dim=-1, p=2)
        
        return user_emb

    def sequence_encode_concept(self, item_emb, nbr_mask):

        item_list_emb = item_emb.reshape([-1, self.hist_max, self.dim])

        item_list_add_pos = item_list_emb + self.position_embedding.repeat([
            item_list_emb.shape[0], 1, 1])

        item_hidden_layer = nn.Sequential(
            nn.Linear(item_list_add_pos.shape[1], self.hidden_units),
            nn.Tanh()
        )
        item_hidden = item_hidden_layer(item_list_add_pos)
        item_att_w_layer = nn.Linear(item_hidden.shape[1], self.num_heads)
        item_att_w = item_att_w_layer(item_hidden)
        item_att_w = item_att_w.permute([0, 2, 1])

        atten_mask = nbr_mask.unsqueeze(1).repeat([1, self.num_heads, 1])

        paddings = torch.ones_like(atten_mask) * (-2 ** 32 + 1)

        item_att_w = torch.where(torch.eq(atten_mask, 0), paddings, item_att_w)

        item_att_w = F.softmax(item_att_w)

        item_emb = torch.matmul(item_att_w, item_list_emb)

        seq = item_emb.reshape([-1, self.num_heads, self.dim])
        if self.num_heads != 1:
            mu = torch.mean(seq, dim=1)
            maha_cpt = nn.Linear(mu.shape[1], self.dim)
            mu = maha_cpt(mu)
            wg = torch.matmul(seq, mu.unsqueeze(-1))
            wg = F.softmax(wg, dim=1)
            seq = torch.mean(seq * wg, dim=1)
        else:
            seq = seq.reshape([-1, self.dim])

        return seq
