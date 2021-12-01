#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :sine.py
@Description  :
@Date         :2021/10/18 20:29:21
@Author       :Arctic Little Pig
@Version      :1.0
'''

from copy import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from metrics.nce_loss import NCELoss


class SINE(nn.Module):
    def __init__(self, config):
        super(SINE, self).__init__()
        self.config = config
        self.model_flag = config.flag
        self.reg = False
        self.user_eb = None
        self.batch_size = config.batch_size
        self.n_size = config.item_count
        self.hist_max = config.maxlen
        self.dim = config.embedding_dim
        self.share_emb = config.share_emb

        self.num_topic = config.topic_num
        self.category_num = config.category_num
        self.hidden_units = config.hidden_size
        self.alpha_para = config.alpha
        self.temperature = 0.07
        # self.temperature = 0.1
        self.user_norm = config.user_norm
        self.item_norm = config.item_norm
        self.cate_norm = config.cate_norm
        self.neg_num = config.neg_num
        if config.cpt_feat == 1:
            self.cpt_feat = True
        else:
            self.cpt_feat = False

        self.num_heads = config.n_head
        self.num_aggre = 1

        # Item embedding Layer
        # After pytorch-v1.0, Variable and Tensor are the same
        # Shape: [item_num, embed_dim] = [3706, 128] (for ml-1m data)
        self.item_input_embedding = nn.Embedding([self.n_size, self.dim])
        self.item_input_lookup_var = nn.Embedding(self.n_size, 1)
        # self.item_input_lookup = torch.randn(
        #     [self.n_size, self.dim], requires_grad=True)
        # Tensor defaults to having no gradient information
        # Shape: [item_num] = [3706] (for ml-1m data)
        # self.item_input_lookup_var = torch.zeros(self.n_size)
        # Shape: [1, maxlen, embed_dim] = [1, 20, 128] (for ml-1m data)
        self.position_embedding = torch.randn(
            [1, self.hist_max, self.dim], requires_grad=True)

        if self.share_emb:
            self.item_output_embedding = self.item_input_embedding
            self.item_output_lookup_var = self.item_input_lookup_var
            # self.item_output_lookup = self.item_input_lookup
            # self.item_output_lookup_var = self.item_input_lookup_var
        else:
            self.item_output_embedding = nn.Embedding(self.n_size, self.dim)
            self.item_output_lookup_var = nn.Embedding(self.n_size, 1)
            # self.item_output_lookup = torch.randn(
            #     [self.n_size, self.dim], requires_grad=True)
            # self.item_output_lookup_var = torch.zeros(self.n_size)

        self.item_output_emb = self.output_item2()

        # Topic embedding layer
        # Shape: [topic_num, embed_dim] = [10, 128] (for ml-1m data)
        self.topic_embedding = nn.Embedding(self.num_topic, self.dim)
        # self.topic_embed = torch.randn(
        #     [self.num_topic, self.dim], requires_grad=True)

        # Self attention for aggregation: hidden layer
        self.item_hidden_layer_aggre = nn.Sequential(
            nn.Linear(self.dim, self.hidden_units),
            nn.Tanh()
        )
        self.item_att_w_layer_aggre = nn.Linear(
            self.hidden_units, self.num_aggre)
        self.item_att_w_softmax_aggre = nn.Softmax(dim=-1)
        self.top_logits_sigmoid_aggre = nn.Sigmoid()
        self.cate_dist_softmax_aggre = nn.Softmax(dim=-1)

        # Self attention: hidden layer
        self.fc1 = nn.Sequential(
            nn.Linear(self.dim, self.hidden_units),
            nn.Tanh()
        )
        self.fc2 = nn.Linear(
            self.hidden_units, self.num_heads*self.category_num)
        self.seq_att_w_softmax = nn.Softmax(dim=-1)

        # Linear layer for multi-heads
        self.maha = nn.Linear(self.dim, self.dim)
        self.wg_softmax = nn.Softmax(dim=1)

        # Self attention for concept: hidden layer
        self.item_hidden_layer_cpt = nn.Sequential(
            nn.Linear(self.dim, self.hidden_units),
            nn.Tanh()
        )
        self.item_att_w_layer_cpt = nn.Linear(
            self.hidden_units, self.num_heads)
        self.item_att_w_softmax_cpt = nn.Softmax(dim=-1)
        self.maha_cpt = nn.Linear(self.dim, self.dim)
        self.wg_softmax_cpt = nn.Softmax(dim=1)

        # Self attention for concept2: hidden layer
        self.maha_cpt2 = nn.Linear(2*self.dim, self.dim)
        self.wg_softmax_cpt2 = nn.Softmax(dim=1)

        self.criterion = NCELoss(config.noise, emb=self.output_item2(), bias=self.item_output_lookup_var,
                                 num_sampled=self.neg_num*self.batch_size, noise_norm=config.noise_norm, reduction='none', loss_type=config.loss_type)

    def forward(self, i_ids, item, nbr_mask):
        # Shape: [batch_size, 1] = [128, 1]
        self.i_ids = i_ids
        # Shape: [batch_size, maxlen] = [128, 20]
        self.item = item
        # Shape: [batch_size, maxlen] = [128, 20]
        self.nbr_mask = nbr_mask

        # Shape: [batch_size, maxlen, embed_dim] = [128, 20, 128]
        # emb = torch.index_select(
        #     self.item_input_lookup, 0, self.item.reshape([-1]))
        # self.item_emb = emb.reshape([-1, self.hist_max, self.dim])
        self.item_emb = self.item_input_embedding(self.item)
        # Shape: [batch_size] = [128]
        self.mask_length = self.nbr_mask.sum(-1).type(dtype=torch.LongTensor)
        # Shape: [batch_size, category_num, embed_dim] = [128, 2, 128]
        self.seq_multi = self.sequence_encode_cpt(self.item_emb, nbr_mask)
        # Shape: [batch_size, embed_dim] = [128, 128]
        self.user_eb = self.labeled_attention(self.seq_multi)

        loss = self._xent_loss_weight(self.user_eb, self.seq_multi)

        pass

    def output_item2(self):
        if self.item_norm:
            item_emb = copy(self.item_output_embedding)
            item_emb.max_norm = 1.0

            return item_emb
        else:
            return self.item_output_embedding

    def sequence_encode_cpt(self, items, nbr_mask):
        # Items shape: [batch_size*maxlen, embed_dim] = [128*20, 128]
        # Nbr_mask shape: [batch_size, maxlen] = [128, 20]
        item_emb_input = items.reshape([-1, self.dim])
        # Shape: [batch_size*num_aggre*maxlen, categorty_num] = [128*20, 2]
        self.cate_dist = self.seq_cate_dist(item_emb_input)
        # Shape: [batch_size*num_aggre, maxlen, categorty_num] = [128, 20, 2]
        self.cate_dist = self.cate_dist.reshape(
            [-1, self.hist_max, self.category_num])
        # Shape: [batch_size*num_aggre, categorty_num, maxlen] = [128, 2, 20]
        self.cate_dist = self.cate_dist.permute([0, 2, 1])
        # Shape: [batch_size, maxlen, embed_dim] = [128, 20, 128]
        item_list_emb = item_emb_input.reshape([-1, self.hist_max, self.dim])
        # Shape: [batch_size, maxlen, embed_dim] = [128, 20, 128]
        item_list_add_pos = item_list_emb + self.position_embedding.repeat([
            item_list_emb.shape[0], 1, 1])
        # Shape: [batch_size, maxlen, hidden_units] = [128, 20, 512]
        item_hidden = self.fc1(item_list_add_pos)
        # Shape: [batch_size, maxlen, num_heads*category_num] = [128, 20, 1*2]
        item_att_w = self.fc2(item_hidden)

        # Shape: [batch_size, num_head*category_num, maxlen] = [128, 2, 20]
        item_att_w = item_att_w.permute([0, 2, 1])

        # Shape: [batch_size, category_num, num_head, maxlen] = [128, 2, 1, 20]
        item_att_w = item_att_w.reshape(
            [-1, self.category_num, self.num_heads, self.hist_max])

        # Shape: [batch_size*num_aggre, categorty_num,, 1*num_heads, maxlen] = [128, 2, 1*1, 20]
        category_mask_tile = self.cate_dist.unsqueeze(2).repeat(
            [1, 1, self.num_heads, 1])
        # paddings = tf.ones_like(category_mask_tile) * (-2 ** 32 + 1)
        # Shape: [batch_size, category_num, num_head, maxlen] = [128, 2, 1, 20]
        seq_att_w = torch.mul(item_att_w, category_mask_tile)
        # Shape: [batch_size, category_num*num_head, maxlen] = [128, 2*1, 20]
        seq_att_w = seq_att_w.reshape(
            [-1, self.category_num * self.num_heads, self.hist_max])

        # Shape: [batch_size, 1, maxlen] = [128, 1, 20]
        atten_mask = nbr_mask.unsqueeze(1)
        # Shape: [batch_size, category_num*num_head, maxlen] = [128, 2*1, 20]
        atten_mask = atten_mask.repeat(
            [1, self.category_num * self.num_heads, 1])
        # Shape: [batch_size, category_num*num_head, maxlen] = [128, 2*1, 20]
        paddings = torch.ones_like(atten_mask) * (-2**32 + 1)
        # Shape: [batch_size, category_num*num_head, maxlen] = [128, 2*1, 20]
        seq_att_w = torch.where(torch.eq(atten_mask, 0), paddings, seq_att_w)
        # Shape: [batch_size, category_num, num_head, maxlen] = [128, 2, 1, 20]
        seq_att_w = seq_att_w.reshape(
            [-1, self.category_num, self.num_heads, self.hist_max])
        # Shape: [batch_size, category_num, num_head, maxlen] = [128, 2, 1, 20]
        seq_att_w = self.seq_att_w_softmax(seq_att_w)

        # here use item_list_emb or item_list_add_pos, that is a question
        # [128, 2, 1, 20] matmul [128, 2, 20, 128]
        # Shape: [batch_size, category_num, num_heads, embed_dim] = [128, 2, 1, 128]
        item_emb = torch.matmul(seq_att_w, item_list_emb.unsqueeze(
            1).repeat([1, self.category_num, 1, 1]))

        # Shape: [batch_size*category_num, num_heads, embed_dim] = [128*2, 1, 128]
        category_embedding_mat = item_emb.reshape(
            [-1, self.num_heads, self.dim])

        if self.num_heads != 1:
            # Shape: [batch_size*category_num, embed_dim] = [128*2, 128]
            mu = category_embedding_mat.mean(dim=1)
            # Shape: [batch_size*category_num, embed_dim] = [128*2, 128]
            mu = self.maha(mu)
            # [128*2, num_heads, 128] matmul [128*2, 128, 1]
            # Shape: [batch_size*category_num, num_heads, 1] = [128*2, num_heads, 1]
            wg = torch.matmul(category_embedding_mat, mu.unsqueeze(-1))
            # Shape: [batch_size*category_num, num_heads, 1] = [128*2, num_heads, 1]
            wg = self.wg_softmax(wg)

            # seq = tf.reduce_mean(category_embedding_mat * wg, axis=1)  # [N,H,D]->[N,D]
            # Shape: [batch_size*category_num, num_heads, embed_dim] = [128*2, num_heads, 128]
            seq = category_embedding_mat * wg
            # Shape: [batch_size*category_num, embed_dim] = [128*2, 128]
            seq = seq.sum(dim=1)
        else:
            seq = category_embedding_mat
        self.category_embedding_mat = seq
        # Shape: [batch_size, category_num, embed_dim] = [128, 2, 128]
        seq = seq.reshape([-1, self.category_num, self.dim])

        return seq

    def seq_cate_dist(self, input_seq):
        # Input_seq shape: [batch_size*maxlen, embed_dim] = [128*20, 128]
        # Shape: [batch_size*num_aggre, categorty_num] = [128, 2]
        top_logit, top_index = self.topic_select(input_seq)
        # Shape: [batch_size*num_aggre, categorty_num, 1] = [128, 2, 1]
        top_logit = top_logit.unsqueeze(2)
        # Shape: [batch_size*num_aggre, categorty_num, embed_dim] = [128, 2, 128]
        topic_embed = self.topic_embedding(top_index)
        # Shape: [batch_size*num_aggre, categorty_num, embed_dim] = [128, 2, 128]
        self.batch_tpt_emb = self.topic_embedding(top_index)
        # Corresponding fractional product
        # Shape: [batch_size*num_aggre, categorty_num, embed_dim] = [128, 2, 128]
        self.batch_tpt_emb = self.batch_tpt_emb * \
            top_logit.repeat([1, 1, self.dim])
        # Shape: [batch_size*maxlen, embed_dim, 1] = [128*20, 128, 1]
        norm_seq = F.normalize(input_seq, p=2, dim=1).unsqueeze(-1)
        # Shape: [batch_size*num_aggre, categorty_num, embed_dim] = [128, 2, 128]
        cores = F.normalize(topic_embed, p=2, dim=-1)
        # Shape: [batch_size*num_aggre, 1*maxlen, categorty_num, embed_dim] = [128, 20, 2, 128]
        cores_t = cores.unsqueeze(1).repeat([1, self.hist_max, 1, 1])
        # Shape: [batch_size*num_aggre*maxlen, categorty_num, embed_dim] = [128*20, 2, 128]
        cores_t = cores_t.reshape([-1, self.category_num, self.dim])
        # Shape: [batch_size*num_aggre*maxlen, categorty_num, 1] = [128*20, 2, 1]
        cate_logits = torch.matmul(cores_t, norm_seq)
        # Shape: [batch_size*num_aggre*maxlen, categorty_num] = [128*20, 2]
        cate_logits = cate_logits.reshape([-1, self.category_num])
        # Shape: [batch_size*num_aggre*maxlen, categorty_num] = [128*20, 2]
        cate_logits = cate_logits / self.temperature
        cate_dist = self.cate_dist_softmax_aggre(cate_logits)

        return cate_dist

    def topic_select(self, input_seq):
        # Input_seq shape: [batch_size*maxlen, embed_dim] = [128*20, 128]
        # Shape: [batch_size, maxlen, embed_dim] = [128, 20, 128]
        seq = input_seq.reshape([-1, self.hist_max, self.dim])
        # Shape: [batch_size*num_aggre, embed_dim] = [128*1, 128]
        seq_emb = self.seq_aggre(seq, self.nbr_mask)
        # Shape: [num_topic, embed_dim] = [10, 128]
        topic_emb = self.topic_embedding.weight
        if self.cate_norm:
            # Shape: [batch_size*num_aggre, embed_dim] = [128*1, 128]
            seq_emb = F.normalize(seq_emb, p=2, dim=-1)
            # Shape: [num_topic, embed_dim] = [10, 128]
            topic_emb = F.normalize(topic_emb, p=2, dim=-1)
            # Shape: [batch_size*num_aggre, num_topic] = [128*1, 10]
            topic_logit = torch.matmul(seq_emb, topic_emb.transpose(-1, -2))
        else:
            # Shape: [batch_size*num_aggre, num_topic] = [128*1, 10]
            topic_logit = torch.matmul(
                seq_emb, topic_emb.transpose(-1, -2))
        # Shape: [batch_size*num_aggre, categorty_num] = [128, 2]
        top_logits, top_index = torch.topk(topic_logit, self.category_num)
        top_logits = self.top_logits_sigmoid_aggre(top_logits)

        return top_logits, top_index

    def seq_aggre(self, item_list_emb, nbr_mask):
        # Item_list_emb shape: [batch_size, maxlen, embed_dim] = [128, 20, 128]
        # Nbr_mask shape: [batch_size, maxlen] = [128, 20]
        # num_aggre = 1
        # Shape: [batch_size, maxlen, embed_dim] = [128, 20, 128]
        item_list_add_pos = item_list_emb + \
            self.position_embedding.repeat(item_list_emb.shape[0], 1, 1)

        # Shape: [batch_size, maxlen, hidden_units] = [128, 20, 512]
        item_hidden = self.item_hidden_layer_aggre(item_list_add_pos)
        # Shape: [batch_size, maxlen, num_aggre] = [128, 20, 1]
        item_att_w = self.item_att_w_layer_aggre(item_hidden)
        # Shape: [batch_size, num_aggre, maxlen] = [128, 1, 20]
        item_att_w = item_att_w.t([0, 2, 1])

        # Shape: [batch_size, 1, maxlen] = [128, 1, 20]
        atten_mask = nbr_mask.unsqueeze(1)
        # Shape: [batch_size, num_aggre, maxlen] = [128, 1, 20]
        atten_mask = atten_mask.repeat([1, self.num_aggre, 1])
        # Shape: [batch_size, num_aggre, maxlen] = [128, 1, 20]
        paddings = torch.ones_like(atten_mask) * (-2**32 + 1)

        # Places that meet the condition are reserved from paddings,
        # and places that don't meet the condition are reserved from item_att_w
        item_att_w = torch.where(torch.eq(atten_mask, 0), paddings, item_att_w)
        item_att_w = self.item_att_w_softmax_aggre(item_att_w)

        # Item_att_w shape: [batch_size, num_aggre, maxlen] = [128, 1, 20]
        # Item_list_emb shape: [batch_size, maxlen, embed_dim] = [128, 20, 128]
        # Shape: [batch_size, num_aggre, embed_dim] = [128, 1, 128]
        item_emb = torch.matmul(item_att_w, item_list_emb)
        # Shape: [batch_size*num_aggre, embed_dim] = [128*1, 128]
        item_emb = item_emb.reshape([-1, self.dim])

        return item_emb

    def labeled_attention(self, seq):
        # Seq shape: [batch_size, category_num, embed_dim] = [128, 2, 128]
        # item_emb = tf.reshape(self.cate_dist, [-1, self.hist_max, self.category_num])
        # Shape: [batch_size*num_aggre, maxlen, categorty_num] = [128, 20, 2]
        item_emb = self.cate_dist.permute([0, 2, 1])
        # [128, 20, 2] matmul [128, 2, 128]
        # Shape: [batch_size, maxlen, embed_dim] = [128, 20, 128]
        item_emb = torch.matmul(item_emb, self.batch_tpt_emb)

        if self.cpt_feat:
            # Shape: [batch_size, maxlen, embed_dim] = [128, 20, 128]
            item_emb = item_emb + \
                self.item_emb.reshape([-1, self.hist_max, self.dim])
        # Shape: [batch_size, embed_dim] = [128, 128]
        target_item = self.sequence_encode_concept(
            item_emb, self.nbr_mask)

        # Shape: [batch_size, embed_dim] = [128, 128]
        mu_seq = torch.mean(seq, dim=1)
        # Shape: [batch_size, embed_dim+embed_dim] = [128, 256]
        target_label = torch.cat([mu_seq, target_item], dim=1)

        # Shape: [batch_size, embed_dim] = [128, 128]
        mu = self.maha_cpt2(target_label)

        # [128, 2, 128] matmul [128, 128, 1]
        # Shape: [batch_size, category_num, 1] = [128, 2, 1]
        wg = torch.matmul(seq, mu.unsqueeze(-1))
        wg = F.softmax(wg, dim=1)

        # Shape: [batch_size, category_num, embed_dim] = [128, 2, 128]
        user_emb = seq * wg
        # Shape: [batch_size, embed_dim] = [128, 128]
        user_emb = user_emb.sum(dim=1)
        if self.user_norm:
            user_emb = F.normalize(user_emb, p=2, dim=-1)

        return user_emb

    def sequence_encode_concept(self, item_emb, nbr_mask):
        # Item_emb shape: [batch_size, maxlen, embed_dim] = [128, 20, 128]
        # Nbr_mask shape: [batch_size, maxlen] = [128, 20]
        # Shape: [batch_size, maxlen, embed_dim] = [128, 20, 128]
        item_list_emb = item_emb.reshape([-1, self.hist_max, self.dim])

        # Shape: [batch_size, maxlen, embed_dim] = [128, 20, 128]
        item_list_add_pos = item_list_emb + self.position_embedding.repeat([
            item_list_emb.shape[0], 1, 1])

        # Shape: [batch_size, maxlen, hidden_units] = [128, 20, 512]
        item_hidden = self.item_hidden_layer_cpt(item_list_add_pos)
        # Shape: [batch_size, maxlen, num_heads] = [128, 20, 1]
        item_att_w = self.item_att_w_layer_cpt(item_hidden)
        # Shape: [batch_size, num_heads, maxlen] = [128, 1, 20]
        item_att_w = item_att_w.permute([0, 2, 1])

        # Shape: [batch_size, num_heads, maxlen] = [128, 1, 20]
        atten_mask = nbr_mask.unsqueeze(1).repeat([1, self.num_heads, 1])

        paddings = torch.ones_like(atten_mask) * (-2**32 + 1)

        item_att_w = torch.where(torch.eq(atten_mask, 0), paddings, item_att_w)
        # Shape: [batch_size, num_heads, maxlen] = [128, 1, 20]
        item_att_w = self.item_att_w_softmax_cpt(item_att_w)
        # [128, 1, 20] matmul [128, 20, 128]
        # Shape: [batch_size, num_heads, embed_dim] = [128, 1, 128]
        item_emb = torch.matmul(item_att_w, item_list_emb)
        # Shape: [batch_size, num_heads, embed_dim] = [128, 1, 128]
        seq = item_emb.reshape([-1, self.num_heads, self.dim])
        if self.num_heads != 1:
            # Shape: [batch_size, embed_dim] = [128, 128]
            mu = seq.mean(dim=1)
            mu = self.maha_cpt(mu)
            # [128, num_heads, 128] matmul [128, 128, 1]
            # Shape: [batch_size, num_heads, 1] = [128, num_heads, 1]
            wg = torch.matmul(seq, mu.unsqueeze(-1))
            wg = self.wg_softmax_cpt2(wg)
            # Shape: [batch_size, num_heads, embed_dim] = [128, num_heads, 128]
            seq = seq * wg
            # Shape: [batch_size, embed_dim] = [128, 128]
            seq = seq.mean(dim=1)
        else:
            # Shape: [batch_size, embed_dim] = [128, 128]
            seq = seq.reshape([-1, self.dim])

        return seq

    def _xent_loss_weight(self, user, seq_multi):
        emb_dim = self.dim
        loss = self.criterion(self.i_ids.reshape(
            [-1, 1]), user.reshape([-1, emb_dim]))

        # Shape: [batch_size] = [128]
        regs = self.calculate_interest_loss(seq_multi)

        self.loss = tf.reduce_mean(loss)
        self.reg_loss = self.alpha_para * tf.reduce_mean(regs)
        loss = self.loss + self.reg_loss

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.lr).minimize(loss)

        return loss

    def calculate_interest_loss(self, user_interest):
        # User_interest shape: [batch_size, category_num, embed_dim] = [128, 2, 128]
        norm_interests = F.normalize(user_interest, p=2, dim=-1)
        # batch_size, category_num, embed_dim
        dim0, dim1, dim2 = user_interest.shape

        # Shape: [category_num//2, batch_size] = [1, 128]
        interests_losses = []
        for i in range(1, (dim1 + 1) // 2):
            # Shape: [batch_size, category_num, embed_dim] = [128, 2, 128]
            roll_interests = torch.cat(
                [norm_interests[:, i:, :], norm_interests[:, 0:i, :]], dim=1)
            # compute pair-wise interests similarity.
            # Shape: [batch_size*category_num, embed_dim] = [128*2, 128]
            interests_radial_diffs = torch.mul(norm_interests.reshape(
                [dim0*dim1, dim2]), roll_interests.reshape([dim0*dim1, dim2]))
            # Shape: [batch_size*category_num] = [128*2]
            interests_loss = interests_radial_diffs.sum(dim=-1)
            # Shape: [batch_size, category_num] = [128, 2]
            interests_loss = interests_loss.reshape([dim0, dim1])
            # Shape: [batch_size] = [128]
            interests_loss = interests_loss.sum(dim=-1)
            interests_losses.append(interests_loss)

        if dim1 % 2 == 0:
            # Size: category_num // 2 = 1
            half_dim1 = dim1 // 2
            interests_part1 = norm_interests[:, :half_dim1, :]
            interests_part2 = norm_interests[:, half_dim1:, :]
            # Shape: [batch_size*(category_num//2), embed_dim] = [128*1, 128]
            interests_radial_diffs = torch.mul(interests_part1.reshape(
                [dim0*half_dim1, dim2]), interests_part2.reshape([dim0*half_dim1, dim2]))
            # Shape: [batch_size*(category_num//2)] = [128*1]
            interests_loss = interests_radial_diffs.sum(dim=-1)
            # Shape: [batch_size, category_num//2] = [128, 1]
            interests_loss = interests_loss.reshape([dim0, half_dim1])
            # Shape: [batch_size] = [128]
            interests_loss = interests_loss.sum(dim=-1)
            interests_losses.append(interests_loss)

        interests_losses = torch.vstack(interests_losses)

        # NOTE(reed): the original interests_loss lay in [0, 2], so the
        # combination_size didn't divide 2 to normalize interests_loss into
        # [0, 1]
        self._interests_length = None
        if self._interests_length is not None:
            combination = self._interests_length * (self._interests_length - 1)
            combination_size = combination.to(torch.float32)
        else:
            combination_size = dim1 * (dim1 - 1)
        # Shape: [batch_size] = [128]
        interests_loss = 0.5 + interests_losses.sum(dim=0) / combination_size

        return interests_loss
