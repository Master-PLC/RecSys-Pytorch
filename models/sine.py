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
    def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, seq_len,
                 topic_num, category_num, alpha, neg_num, cpt_feat,
                 user_norm, cate_norm, n_head,
                 share_emb=True, flag="DNN", item_norm=0):
        super(SINE, self).__init__()
        self.n_size = n_mid
        self.dim = embedding_dim
        self.hidden_units = hidden_size
        self.batch_size = batch_size
        self.hist_max = seq_len
        self.num_topic = topic_num
        self.category_num = category_num
        self.alpha_para = alpha
        self.neg_num = neg_num
        if cpt_feat == 1:
            self.cpt_feat = True
        else:
            self.cpt_feat = False
        self.user_norm = user_norm
        self.item_norm = item_norm
        self.cate_norm = cate_norm
        self.num_heads = n_head
        self.share_emb = share_emb
        self.model_flag = flag
        self.item_norm = item_norm
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
        
        self.  

    def forward(self, i_ids, item, nbr_mask):
        self.i_ids = i_ids
        self.item = item
        self.nbr_mask = nbr_mask

        emb = torch.index_select(self.item_input_lookup, 0, self.item.reshape([-1]))
        self.item_emb = emb.reshape([-1, self.hist_max, self.dim])
        self.mask_length = self.nbr_mask.sum(-1).type(dtype=torch.LongTensor)

        seq_multi = self.sequence_encode_cpt(self.item_emb, nbr_mask)
        user_eb = self.labeled_attention(seq_multi)
        self._xent_loss_weight(self.user_eb, self.seq_multi)

        pass

    def output_item2(self):
        if self.item_norm:
            item_emb = F.normalize(self.item_output_lookup, dim=-1, p=2)
            return item_emb
        else:
            return self.item_output_lookup

    def sequence_encode_cpt(self, items, nbr_mask):
        item_emb_input = items.reshape([-1, self.dim])
        self.cate_dist = tf.transpose(tf.reshape(self.seq_cate_dist(
            item_emb_input), [-1, self.hist_max, self.category_num]), [0, 2, 1])
        item_list_emb = tf.reshape(
            item_emb_input, [-1, self.hist_max, self.dim])
        item_list_add_pos = item_list_emb + \
            tf.tile(self.position_embedding, [
                    tf.shape(item_list_emb)[0], 1, 1])

        with tf.variable_scope("self_atten", reuse=tf.AUTO_REUSE) as scope:
            item_hidden = tf.layers.dense(
                item_list_add_pos, self.hidden_units, activation=tf.nn.tanh, name='fc1')
            item_att_w = tf.layers.dense(
                item_hidden, self.num_heads * self.category_num, activation=None, name='fc2')

            # [batch_size, category_num*num_head, hist_max]
            item_att_w = tf.transpose(item_att_w, [0, 2, 1])

            # [batch_size, category_num, num_head, hist_max]
            item_att_w = tf.reshape(
                item_att_w, [-1, self.category_num, self.num_heads, self.hist_max])

            category_mask_tile = tf.tile(tf.expand_dims(self.cate_dist, axis=2), [
                                         1, 1, self.num_heads, 1])  # [batch_size, category_num, num_head, hist_max]
            # paddings = tf.ones_like(category_mask_tile) * (-2 ** 32 + 1)
            seq_att_w = tf.reshape(tf.multiply(
                item_att_w, category_mask_tile), [-1, self.category_num * self.num_heads, self.hist_max])

            atten_mask = tf.tile(tf.expand_dims(nbr_mask, axis=1), [
                                 1, self.category_num * self.num_heads, 1])

            paddings = tf.ones_like(atten_mask) * (-2 ** 32 + 1)

            seq_att_w = tf.where(tf.equal(atten_mask, 0), paddings, seq_att_w)
            seq_att_w = tf.reshape(
                seq_att_w, [-1, self.category_num, self.num_heads, self.hist_max])

            seq_att_w = tf.nn.softmax(seq_att_w)

            # here use item_list_emb or item_list_add_pos, that is a question
            item_emb = tf.matmul(seq_att_w, tf.tile(tf.expand_dims(item_list_emb, axis=1), [
                                 1, self.category_num, 1, 1]))  # [batch_size, category_num, num_head, dim]

            # [batch_size, category_num, dim]
            category_embedding_mat = tf.reshape(
                item_emb, [-1, self.num_heads, self.dim])
            if self.num_heads != 1:
                mu = tf.reduce_mean(category_embedding_mat,
                                    axis=1)  # [N,H,D]->[N,D]
                mu = tf.layers.dense(mu, self.dim, name='maha')
                wg = tf.matmul(category_embedding_mat, tf.expand_dims(
                    mu, axis=-1))  # (H,D)x(D,1) = [N,H,1]
                wg = tf.nn.softmax(wg, dim=1)  # [N,H,1]

                # seq = tf.reduce_mean(category_embedding_mat * wg, axis=1)  # [N,H,D]->[N,D]
                seq = tf.reduce_sum(category_embedding_mat *
                                    wg, axis=1)  # [N,H,D]->[N,D]
            else:
                seq = category_embedding_mat
            self.category_embedding_mat = seq
            seq = tf.reshape(seq, [-1, self.category_num, self.dim])

        return seq

    def seq_cate_dist(self, input_seq):
        #     input_seq [-1, dim]
        top_logit, top_index = self.topic_select(input_seq)
        topic_embed = tf.nn.embedding_lookup(self.topic_embed, top_index)
        self.batch_tpt_emb = tf.nn.embedding_lookup(
            self.topic_embed, top_index)  # [-1, cate_num, dim]
        self.batch_tpt_emb = self.batch_tpt_emb * \
            tf.tile(tf.expand_dims(top_logit, axis=2), [1, 1, self.dim])
        norm_seq = tf.expand_dims(tf.nn.l2_normalize(
            input_seq, dim=1), axis=-1)  # [-1, dim, 1]
        cores = tf.nn.l2_normalize(topic_embed, dim=-1)  # [-1, cate_num, dim]
        cores_t = tf.reshape(tf.tile(tf.expand_dims(cores, axis=1), [
                             1, self.hist_max, 1, 1]), [-1, self.category_num, self.dim])
        cate_logits = tf.reshape(tf.matmul(
            cores_t, norm_seq), [-1, self.category_num]) / self.temperature  # [-1, cate_num]
        cate_dist = tf.nn.softmax(cate_logits, dim=-1)
        return cate_dist

    def topic_select(self, input_seq):
        seq = input_seq.reshape([-1, self.hist_max, self.dim])
        seq_emb = self.seq_aggre(seq, self.nbr_mask)
        if self.cate_norm:
            seq_emb = tf.nn.l2_normalize(seq_emb, dim=-1)
            topic_emb = tf.nn.l2_normalize(self.topic_embed, dim=-1)
            topic_logit = tf.matmul(seq_emb, topic_emb, transpose_b=True)
        else:
            # [batch_size, topic_num]
            topic_logit = tf.matmul(
                seq_emb, self.topic_embed, transpose_b=True)
        # two [batch_size, categorty_num] tensors
        top_logits, top_index = tf.nn.top_k(topic_logit, self.category_num)
        top_logits = tf.sigmoid(top_logits)
        return top_logits, top_index

    def seq_aggre(self, item_list_emb, nbr_mask):
        num_aggre = 1
        item_list_add_pos = item_list_emb + torch.repeat(self.position_embedding, [item_list_emb.shape[0], 1, 1])

        item_hidden = tf.layers.dense(
            item_list_add_pos, self.hidden_units, activation=tf.nn.tanh)
        item_att_w = tf.layers.dense(
            item_hidden, num_aggre, activation=None)
        item_att_w = tf.transpose(item_att_w, [0, 2, 1])

        atten_mask = tf.tile(tf.expand_dims(
            nbr_mask, axis=1), [1, num_aggre, 1])

        paddings = tf.ones_like(atten_mask) * (-2 ** 32 + 1)

        item_att_w = tf.where(tf.equal(atten_mask, 0),
                                paddings, item_att_w)

        item_att_w = tf.nn.softmax(item_att_w)

        item_emb = tf.matmul(item_att_w, item_list_emb)

        item_emb = tf.reshape(item_emb, [-1, self.dim])

        return item_emb

    def sequence_encode_concept(self, item_emb, nbr_mask):

        item_list_emb = tf.reshape(item_emb, [-1, self.hist_max, self.dim])

        item_list_add_pos = item_list_emb + \
            tf.tile(self.position_embedding, [
                    tf.shape(item_list_emb)[0], 1, 1])

        with tf.variable_scope("self_atten_cpt", reuse=tf.AUTO_REUSE) as scope:
            item_hidden = tf.layers.dense(
                item_list_add_pos, self.hidden_units, activation=tf.nn.tanh)
            item_att_w = tf.layers.dense(
                item_hidden, self.num_heads, activation=None)
            item_att_w = tf.transpose(item_att_w, [0, 2, 1])

            atten_mask = tf.tile(tf.expand_dims(
                nbr_mask, axis=1), [1, self.num_heads, 1])

            paddings = tf.ones_like(atten_mask) * (-2 ** 32 + 1)

            item_att_w = tf.where(tf.equal(atten_mask, 0),
                                  paddings, item_att_w)

            item_att_w = tf.nn.softmax(item_att_w)

            item_emb = tf.matmul(item_att_w, item_list_emb)

            seq = tf.reshape(item_emb, [-1, self.num_heads, self.dim])
            if self.num_heads != 1:
                mu = tf.reduce_mean(seq, axis=1)
                mu = tf.layers.dense(mu, self.dim, name='maha_cpt')
                wg = tf.matmul(seq, tf.expand_dims(mu, axis=-1))
                wg = tf.nn.softmax(wg, dim=1)
                seq = tf.reduce_mean(seq * wg, axis=1)
            else:
                seq = tf.reshape(seq, [-1, self.dim])
        return seq

    def labeled_attention(self, seq):
        # item_emb = tf.reshape(self.cate_dist, [-1, self.hist_max, self.category_num])
        item_emb = tf.transpose(self.cate_dist, [0, 2, 1])
        item_emb = tf.matmul(item_emb, self.batch_tpt_emb)

        if self.cpt_feat:
            item_emb = item_emb + \
                tf.reshape(self.item_emb, [-1, self.hist_max, self.dim])
        target_item = self.sequence_encode_concept(
            item_emb, self.nbr_mask)  # [N,  D]

        mu_seq = tf.reduce_mean(seq, axis=1)  # [N,H,D] -> [N,D]
        target_label = tf.concat([mu_seq, target_item], axis=1)

        mu = tf.layers.dense(target_label, self.dim,
                             name='maha_cpt2', reuse=tf.AUTO_REUSE)

        wg = tf.matmul(seq, tf.expand_dims(mu, axis=-1))  # (H,D)x(D,1)
        wg = tf.nn.softmax(wg, dim=1)

        user_emb = tf.reduce_sum(seq * wg, axis=1)  # [N,H,D]->[N,D]
        if self.user_norm:
            user_emb = tf.nn.l2_normalize(user_emb, dim=-1)
        return user_emb
