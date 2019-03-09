# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from learned_positional_embedding import LearnedPositionalEmbedding
from multihead_attention import MultiheadAttention
import math


class Para_cls_model(nn.Module):

    def __init__(self, args, bert_net):
        super().__init__()

        self.name = "Para_cls_model"
        self.encoder = bert_net.encoder
        self.para_cls_fn = Linear(args.embed_dim, args.para_cls_size)

    def forward(self, src_tokens, src_lengths, sent_idx):
        
        encoder_out_dict = self.encoder(src_tokens, src_lengths, sent_idx)
        encoder_out = encoder_out_dict['encoder_out'].transpose(0, 1)   # B x T x C
        para_cls_out = encoder_out[:, 0]
        para_cls_out = self.para_cls_fn(para_cls_out)

        return para_cls_out

class Bert(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.name = "Bert"
        self.embed_tokens = Embedding(args.vocab_size, args.embed_dim, padding_idx=args.padding_idx)
        self.sent_idx_tokens = Embedding(args.sent_type_num, args.embed_dim, padding_idx=args.padding_idx)
        self.encoder = TransformerEncoder(args, self.embed_tokens, self.sent_idx_tokens)
        self.mask_fc = Linear(args.embed_dim, args.vocab_size)
        if args.share_embed:
            self.mask_fc.weight = self.embed_tokens.weight
        self.sent_cls_fc = Linear(args.embed_dim, args.sent_cls_size)

    def forward(self, src_tokens, src_lengths, sent_idx, mask_index_t):

        encoder_out_dict = self.encoder(src_tokens, src_lengths, sent_idx)
        # T x B x C
        encoder_out = encoder_out_dict['encoder_out'].transpose(0, 1)   # B x T x C
        # B x T
        encoder_padding_mask = encoder_out_dict['encoder_padding_mask']
        # mask_index_t -> (batch_size, int(0.15 * maxlen))
        batch_size, maxlen, channels = encoder_out.size()
        mask_out = [torch.index_select(encoder_out[i], 0, mask_index_t[i]).unsqueeze(0) for i in range(batch_size)]
        mask_out = torch.cat(mask_out, dim=0)       # (batch_size, 77, embed_dim)

        mask_out = self.mask_fc(mask_out)           # (batch_size , 77, vocab_size)

        sent_cls_out = encoder_out[:, 0]
        sent_cls_out = self.sent_cls_fc(sent_cls_out)

        return mask_out, sent_cls_out


class TransformerEncoder(nn.Module):
    """
    Transformer encoder consisting of *args.encoder_layers* layers.
    Each layer is a :class: `TransformerEncoderLayer`.
    Args:
        args (argparse.Namespace): parsed command-line arguments.
        dictionary (~fairseq.data.Dictionary): encoding dictionary.
        embed_tokens (torch.nn.Embedding): input embedding.
    """
    def __init__(self, args, embed_tokens, sent_idx_tokens):
        super().__init__()

        self.dropout = args.dropout
        self.embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(self.embed_dim)
        self.sent_idx_tokens = sent_idx_tokens

        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, self.embed_dim, self.padding_idx,
            learned=args.encoder_learned_pos,
        ) if not args.no_tokens_positional_embeddings else None

        self.layers = nn.ModuleList([])
        self.layers.extend([TransformerEncoderLayer(args) for i in range(args.encoder_layers)])
        self.normalize = args.encoder_normalize_before
        if self.normalize:
            self.layer_norm = LayerNorm(self.embed_dim)

    def forward(self, src_tokens, src_lengths, sent_idx):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape `(batch, src_len)`.
            src_lengths (LongTensor): lengths of each source sentence of shape `(batch)`.

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of shape `(src_len, batch, embed_dim)`.
                - **encoder_padding_mask** (ByteTensor): the positions of padding elements of shape `(batch, src_len)`.
        """
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(src_tokens)
        if self.embed_positions is not None:
            x += self.embed_positions(src_tokens)
        if self.sent_idx_tokens is not None:
            x += self.sent_idx_tokens(sent_idx)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Batch x Time x Channel -> Time x Batch x Channel
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)

        if self.normalize:
            x = self.layer_norm(x)

        return {
            'encoder_out': x,       # T x B x C
            'encoder_padding_mask': encoder_padding_mask,       # B x T
        }


class TransformerEncoderLayer(nn.Module):
    """
    Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is postprocessed with:
    `dropout -> add residual -> layernorm`. In the tensor2tensor code they suggest that
    learning is more robust when preprocessing each layer with layernorm and postprocessing
    with: `dropout -> add residual`. We default to the approach in the paper, but the
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments.
    """
    def __init__(self, args):
        super().__init__()

        self.embed_dim = args.encoder_embed_dim
        self.self_attn = MultiheadAttention(self.embed_dim, args.encoder_attention_heads,
                                            dropout=args.attention_dropout)
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = Linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.fc2 = Linear(args.encoder_ffn_embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for i in range(2)])

    def forward(self, x, encoder_padding_mask):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`.
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape `(batch, src_len)`
                where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`.
        """
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        x = self.self_attn(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)

        return x

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)

    return m


def LayerNorm(embedding_dim):
    m = nn.LayerNorm(embedding_dim)
    return m


def Embedding(num_embeddings, embedding_dim, padding_idx=0):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)

    return m


def PositionalEmbedding(num_embeddings, embedding_dim, padding_idx=0, left_pad=False, learned=False):
    m = LearnedPositionalEmbedding(num_embeddings + padding_idx + 1, embedding_dim, padding_idx, left_pad)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)

    return m
