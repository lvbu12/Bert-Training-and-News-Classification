# -*- coding: utf-8 -*-
import torch.nn as nn
import model_utils


class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """
    def __init__(self, num_embeddings, embedding_dim, padding_idx=0, left_pad=False):
        super().__init__(num_embeddings, embedding_dim, padding_idx)

        self.left_pad = left_pad

    def forward(self, input, incremental_state=None, timestep=None):
        """Input is expected to be of size [bsz x seqlen]"""
        if incremental_state is not None:
            assert timestep is not None
            bsz, seq_len = input.size()
            positions = (timestep.int() + 1).long().expand(bsz, 1) if timestep is not None else seq_len
        else:
            positions = model_utils.make_positions(input.data, self.padding_idx, self.left_pad)

        return super().forward(positions)

    def max_positions(self):
        return self.num_embeddings - self.padding_idx + 1