#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :nce_loss.py
@Description  :
@Date         :2021/11/19 15:57:09
@Author       :Arctic Little Pig
@Version      :1.0
'''

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .alias_multinomial import AliasMultinomial

# A backoff probability to stabilize log operation
BACKOFF_PROB = 1e-10


class NCELoss(nn.Module):
    """Noise Contrastive Estimation

    NCE is to eliminate the computational cost of softmax
    normalization.

    There are 3 loss modes in this NCELoss module:
        - nce: enable the NCE approximation
        - sampled: enabled sampled softmax approximation
        - full: use the original cross entropy as default loss
    They can be switched by directly setting `nce.loss_type = 'nce'`.

    Ref:
        X.Chen etal Recurrent neural network language
        model training with noise contrastive estimation
        for speech recognition
        https://core.ac.uk/download/pdf/42338485.pdf

    Attributes:
        noise: the distribution of noise
        num_sampled: $\frac{#noises}{#real data samples}$ (k in paper)
        noise_norm: the normalization term (lnZ in paper), can be heuristically
        determined by the number of classes, plz refer to the code.
        reduction: reduce methods, same with pytorch's loss framework, 'none',
        'elementwise_mean' and 'sum' are supported.
        loss_type: loss type of this module, currently 'full', 'sampled', 'nce'
        are supported

    Shape:
        - noise: :math:`(V)` where `V = vocabulary size`
        - target: :math:`(B, N)`
        - loss: a scalar loss by default, :math:`(B, N)` if `reduction='none'`

    Input:
        target: the supervised training label.
        args&kwargs: extra arguments passed to underlying index module

    Return:
        loss: if `reduction='sum' or 'elementwise_mean'` the scalar NCELoss ready for backward,
        else the loss matrix for every individual targets.
    """

    def __init__(self, noise, emb, bias, num_sampled=100, noise_norm=-1, reduction='ElementWiseMean', per_word=False, loss_type='nce'):
        super(NCELoss, self).__init__()

        # Re-norm the given noise frequency list and compensate words with
        # extremely low prob for numeric stability
        # Shape: [n_size]
        probs = noise / noise.sum()
        probs = probs.clamp(min=BACKOFF_PROB)
        # Shape: [n_size]
        renormed_probs = probs / probs.sum()

        self.register_buffer('logprob_noise', renormed_probs.log())
        self.alias = AliasMultinomial(renormed_probs)

        self.emb = emb
        self.bias = bias
        self.reset_parameters()

        self.num_sampled = num_sampled
        if noise_norm == -1:
            self.noise_norm = math.log(noise.numel())
        else:
            self.noise_norm = noise_norm
        self.reduction = reduction
        self.per_word = per_word
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.loss_type = loss_type

    def reset_parameters(self):
        stdv = 1./math.sqrt(self.emb.embedding_dim)
        self.emb.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            # initialize the bias with unigram instead of uniform
            self.bias.weight.data = torch.unsqueeze(
                self.logprob_noise + self.noise_norm, 1)

    def forward(self, target, input, training, *args, **kwargs):
        """compute the loss with output and the desired target

        The `forward` is the same among all NCELoss submodules, it
        takes care of generating noises and calculating the loss
        given target and noise scores.
        """
        # target shape: [batch_size, 1] = [128, 1]
        # input shape: [batch_size, embed_dim] = [128, 128]
        batch_size, max_len = target.shape

        if self.loss_type != 'FullLoss':
            # Shape: [batch_size, 1, num_sampled] = [128, 1, 100]
            noise_samples = self.get_noise(batch_size, max_len)

            # Shape: [batch_size, 1, num_sampled] = [128, 1, 100]
            logit_noise_in_noise = self.logprob_noise[noise_samples.data.view(
                -1)].view_as(noise_samples)
            # Shape: [batch_size, 1] = [128, 1]
            logit_target_in_noise = self.logprob_noise[target.data.view(
                -1)].view_as(target)

            # Shape: [batch_size, 1] = [128, 1]
            # Shape: [batch_size, 1, num_sampled] = [128, 1, 100]
            logit_target_in_model, logit_noise_in_model = self._get_logit(
                target, noise_samples, input, *args, **kwargs)

            if self.loss_type == 'NCELoss':
                if training:
                    # Shape: [batch_size, 1]
                    loss = self.nce_loss(
                        logit_target_in_model, logit_noise_in_model,
                        logit_noise_in_noise, logit_target_in_noise,
                    )
                else:
                    # directly output the approximated posterior
                    # Shape: [batch_size, 1] = [128, 1]
                    loss = - logit_target_in_model
            elif self.loss_type == 'SampledSoftmax':
                # Shape: [batch_size, 1] = [128, 1]
                loss = self.sampled_softmax_loss(
                    logit_target_in_model, logit_noise_in_model,
                    logit_noise_in_noise, logit_target_in_noise,
                )
            # NOTE: The mix mode is still under investigation
            elif self.loss_type == 'Mix' and training:
                loss = 0.5 * self.nce_loss(
                    logit_target_in_model, logit_noise_in_model,
                    logit_noise_in_noise, logit_target_in_noise,
                )
                loss += 0.5 * self.sampled_softmax_loss(
                    logit_target_in_model, logit_noise_in_model,
                    logit_noise_in_noise, logit_target_in_noise,
                )
            else:
                current_stage = 'training' if training else 'inference'
                raise NotImplementedError(
                    f'loss type {self.loss_type} not implemented at {current_stage}')
        else:
            # Fallback into conventional cross entropy
            # Shape: [batch_size, 1] = [128, 1]
            loss = self.ce_loss(target, input, *args, **kwargs)

        if self.reduction == 'ElementWiseMean':
            return loss.mean()
        elif self.reduction == 'Sum':
            return loss.sum()
        else:
            return loss

    def get_noise(self, batch_size, max_len):
        """Generate noise samples from noise distribution"""
        # Shape: [batch_size, 1, num_sampled] = [128, 1, 100]
        noise_size = [batch_size, max_len, self.num_sampled]
        if self.per_word:
            noise_samples = self.alias.draw(*noise_size)
        else:
            noise_samples = self.alias.draw(
                1, 1, self.num_sampled).expand(*noise_size)
        # Shape: [batch_size, 1, num_sampled] = [128, 1, 100]
        noise_samples = noise_samples.contiguous()
        print(noise_samples)

        return noise_samples

    def _get_logit(self, target_idx, noise_idx, input, *args, **kwargs):
        """Get the logits of NCE estimated probability for target and noise

        Both NCE and sampled softmax Loss are unchanged when the probabilities are scaled
        evenly, here we subtract the maximum value as in softmax, for numeric stability.

        Shape:
            - Target_idx: :math:`(N)`
            - Noise_idx: :math:`(N, N_r)` where `N_r = noise ratio`
        """
        # target_idx shape: [batch_size, 1] = [128, 1]
        # noise_idx shape: [batch_size, 1, num_sampled] = [128, 1, 100]
        # input shape: [batch_size, embed_dim] = [128, 128]

        # Shape: [batch_size, 1, 1] = [128, 1, 1]
        # Shape: [batch_size, 1, num_sampled] = [128, 1, 100]
        target_logit, noise_logit = self.get_score(
            target_idx, noise_idx, input, *args, **kwargs)

        target_logit = target_logit.sub(self.noise_norm)
        noise_logit = noise_logit.sub(self.noise_norm)

        return target_logit, noise_logit

    def get_score(self, target_idx, noise_idx, input, *args, **kwargs):
        """Get the target and noise score

        Usually logits are used as score.
        This method should be override by inherit classes

        Returns:
            - target_score: real valued score for each target index
            - noise_score: real valued score for each noise index
        """
        # target_idx shape: [batch_size, 1] = [128, 1]
        # noise_idx shape: [batch_size, 1, num_sampled] = [128, 1, 100]
        # input shape: [batch_size, embed_dim] = [128, 128]

        # Shape: [batch_size, 1] = [128, 1]
        # Shape: [batch_size, 1, num_sampled] = [128, 1, 100]
        if self.per_word:
            return self._compute_sampled_logit(target_idx, noise_idx, input)
        else:
            return self._compute_sampled_logit_batched(target_idx, noise_idx, input)

    def _compute_sampled_logit(self, target_idx, noise_idx, input):
        """compute the logits of given indices based on input vector

        Args:
            - target_idx: :math:`B, L, 1`
            - noise_idx: :math:`B, L, N_r` target_idx and noise_idx are
            concatenated into one single index matrix for performance
            - input: :math:`(B, L, E)` where `E = vector dimension`

        Returns:
            - target_score: :math:`(B, L)` the computed logits of target_idx
            - noise_score: :math:`(B, L, N_r)` the computed logits of noise_idx
        """
        # target_idx shape: [batch_size, 1] = [128, 1]
        # noise_idx shape: [batch_size, 1, num_sampled] = [128, 1, 100]
        # input shape: [batch_size, embed_dim] = [128, 128]

        # the size will be used to pack the output of indexlinear
        # Shape: [batch_size, 1] = [128, 1]
        original_size = target_idx.size()

        # flatten the following matrix
        # Shape: [batch_size, 1, embed_dim] = [128, 1, 128]
        input = input.contiguous().view(-1, 1, input.size(-1))
        # Shape: [batch_size, 1] = [128, 1]
        target_idx = target_idx.view(-1).unsqueeze(-1)
        # Shape: [batch_size, num_sampled] = [128, 100]
        noise_idx = noise_idx.view(-1, noise_idx.size(-1))
        # Shape: [batch_size, num_sampled+1] = [128, 101]
        indices = torch.cat([target_idx, noise_idx], dim=-1)

        # the pytorch's [] operator can't BP correctly with redundant indices
        # before version 0.2.0
        # [] operator is much slower than index_select in pytorch-0.4.0

        # index_select is faster than pure embedding look-up which is weird
        # 20it/s vs. 14 it/s

        # Shape: [batch_size, num_sampled+1, embed_dim] = [128, 101, 128]
        target_batch = self.emb(indices)
        # target_batch = self.emb.weight.index_select(
        #     0, indices.view(-1)).view(*indices.size(), -1)
        # Shape: [batch_size, num_sampled+1] = [128, 101]
        bias = self.bias(indices).squeeze(2)
        # bias = self.bias.weight.index_select(
        #     0, indices.view(-1)).view_as(indices)
        # the element-wise multiplication is automatically broadcasted
        # [128, 1, 128] * [128, 101, 128] = [128, 101, 128]
        # Shape: [batch_size, num_sampled+1] = [128, 101]
        logits = torch.sum(input * target_batch, dim=2) + bias
        # Shape: [batch_size, 1, num_sampled+1] = [128, 1, 101]
        logits = logits.view(*original_size, -1)

        target_score, noise_score = logits[:, :, 0], logits[:, :, 1:]

        # Shape: [batch_size, 1, 1] = [128, 1, 1]
        # Shape: [batch_size, 1, num_sampled] = [128, 1, 100]
        return target_score.squeeze(1), noise_score

    def _compute_sampled_logit_batched(self, target_idx, noise_idx, input):
        """compute the logits of given indices based on input vector

        A batched version, it speeds up computation and puts less burden on
        sampling methods.

        Args:
            - target_idx: :math:`B, L, 1` flatten to `(N)` where `N=BXL`
            - noise_idx: :math:`B, L, N_r`, noises at the dim along B and L
            should be the same, flatten to `N_r`
            - input: :math:`(B, L, E)` where `E = vector dimension`

        Returns:
            - target_score: :math:`(B, L)` the computed logits of target_idx
            - noise_score: :math:`(B, L, N_r)` the computed logits of noise_idx
        """
        # target_idx shape: [batch_size, 1] = [128, 1]
        # noise_idx shape: [batch_size, 1, num_sampled] = [128, 1, 100]
        # input shape: [batch_size, embed_dim] = [128, 128]

        # Shape: [batch_size, 1] = [128, 1]
        original_size = target_idx.size()

        # flatten the following matrix
        # Shape: [batch_size, embed_dim] = [128, 128]
        input = input.contiguous().view(-1, input.size(-1))
        # Shape: [batch_size] = [128]
        target_idx = target_idx.view(-1)
        # Shape: [num_sampled] = [100]
        noise_idx = noise_idx[0, 0].view(-1)

        # Shape: [batch_size, embed_dim] = [128, 128]
        target_batch = self.emb(target_idx)
        # target_bias = self.bias.index_select(0, target_idx)  # N
        # Shape: [batch_size] = [128]
        target_bias = self.bias(target_idx).squeeze(1)
        # [128, 128] * [128, 128] = [128, 128]
        # Shape: [batch_size] = [128]
        target_score = torch.sum(input * target_batch, dim=1) + target_bias

        # Shape: [num_sampled, embed_dim] = [100, 128]
        noise_batch = self.emb(noise_idx)
        # noise_bias = self.bias.index_select(0, noise_idx).unsqueeze(0)
        # Shape: [num_sampled, 1] = [100, 1]
        noise_bias = self.bias(noise_idx)
        # [128, 128] * [128, 100]
        # Shape: [batch_size, num_sampled] = [128, 100]
        noise_score = torch.matmul(
            input, noise_batch.t()) + noise_bias.t()

        # Shape: [batch_size, 1, 1] = [128, 1, 1]
        # Shape: [batch_size, 1, num_sampled] = [128, 1, 100]
        return target_score.view(original_size), noise_score.view(*original_size, -1)

    def ce_loss(self, target_idx, input, *args, **kwargs):
        """Get the conventional CrossEntropyLoss

        The returned loss should be of the same size of `target`

        Args:
            - target_idx: batched target index
            - args, kwargs: any arbitrary input if needed by sub-class

        Returns:
            - loss: the estimated loss for each target
        """
        # target_idx shape: [batch_size, 1] = [128, 1]
        # input shape: [batch_size, embed_dim] = [128, 128]

        # Shape: [batch_size, vocab_size] = [128, 3702]
        score = F.linear(input, self.emb.weight, self.bias.weight.squeeze(1))

        # ce input shape: [128, 3702], [128]
        # Shape: [batch_size, 1] = [128, 1]
        loss = self.ce(score.view(-1, score.size(-1)),
                       target_idx.view(-1)).view_as(target_idx)

        return loss

    def nce_loss(self, logit_target_in_model, logit_noise_in_model, logit_noise_in_noise, logit_target_in_noise):
        """Compute the classification loss given all four probabilities

        Args:
            - logit_target_in_model: logit of target words given by the model (RNN)
            - logit_noise_in_model: logit of noise words given by the model
            - logit_noise_in_noise: logit of noise words given by the noise distribution
            - logit_target_in_noise: logit of target words given by the noise distribution

        Returns:
            - loss: a mis-classification loss for every single case
        """
        # logit_target_in_model shape: [batch_size, 1] = [128, 1]
        # logit_noise_in_model shape: [batch_size, 1, num_sampled] = [128, 1, 100]
        # logit_noise_in_noise shape: [batch_size, 1, num_sampled] = [128, 1, 100]
        # logit_target_in_noise shape: [batch_size, 1] = [128, 1]

        # NOTE: prob <= 1 is not guaranteed
        # Shape: [batch_size, 1, num_sampled+1] = [128, 1, 101]
        logit_model = torch.cat(
            [logit_target_in_model.unsqueeze(2), logit_noise_in_model], dim=2)
        # Shape: [batch_size, 1, num_sampled+1] = [128, 1, 101]
        logit_noise = torch.cat(
            [logit_target_in_noise.unsqueeze(2), logit_noise_in_noise], dim=2)

        # predicted probability of the word comes from true data distribution
        # The posterior can be computed as following
        # p_true = logit_model.exp() / (logit_model.exp() + self.num_sampled * logit_noise.exp())
        # For numeric stability we compute the logits of true label and
        # directly use bce_with_logits.
        # Ref https://pytorch.org/docs/stable/nn.html?highlight=bce#torch.nn.BCEWithLogitsLoss

        # Shape: [batch_size, 1, num_sampled+1] = [128, 1, 101]
        logit_true = logit_model - logit_noise - math.log(self.num_sampled)
        # Shape: [batch_size, 1, num_sampled+1] = [128, 1, 101]
        label = torch.zeros_like(logit_model)
        label[:, :, 0] = 1
        # Shape: [batch_size, 1] = [128, 1]
        loss = self.bce_with_logits(logit_true, label).sum(dim=2)
        return loss

    def sampled_softmax_loss(self, logit_target_in_model, logit_noise_in_model, logit_noise_in_noise, logit_target_in_noise):
        """Compute the sampled softmax loss based on the tensorflow's impl"""
        # logit_target_in_model shape: [batch_size, 1] = [128, 1]
        # logit_noise_in_model shape: [batch_size, 1, num_sampled] = [128, 1, 100]
        # logit_noise_in_noise shape: [batch_size, 1, num_sampled] = [128, 1, 100]
        # logit_target_in_noise shape: [batch_size, 1] = [128, 1]

        # Shape: [batch_size, 1, num_sampled+1] = [128, 1, 101]
        logits = torch.cat(
            [logit_target_in_model.unsqueeze(2), logit_noise_in_model], dim=2)
        # Shape: [batch_size, 1, num_sampled+1] = [128, 1, 101]
        q_logits = torch.cat(
            [logit_target_in_noise.unsqueeze(2), logit_noise_in_noise], dim=2)
        # subtract Q for correction of biased sampling
        # Shape: [batch_size, 1, num_sampled+1] = [128, 1, 101]
        logits = logits - q_logits
        # Shape: [batch_size, 1] = [128, 1]
        labels = torch.zeros_like(logits.narrow(2, 0, 1)).squeeze(2).long()

        # ce input shape: [128, 101], [128]
        # Shape: [batch_size, 1] = [128, 1]
        loss = self.ce(logits.view(-1, logits.size(-1)),
                       labels.view(-1)).view_as(labels)

        return loss
