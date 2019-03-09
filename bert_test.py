# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import time_since, show_model_size, get_chars
from loaders.idx_test_loader import TestData, collate_fn
from config import Config
from modules.model import Bert
from mask_cross_entropy_loss import mask_cross_entropy_loss
import argparse
import os
import sys
import time

parser = argparse.ArgumentParser(description='add config path')
parser.add_argument('-config_path', type=str, default='Configs/bert.json')

args = parser.parse_args()

cfg = Config.from_json_file(args.config_path)
for key, val in cfg.__dict__.items():
    print('{} = {}'.format(key, val))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test(net, words_t, words_num_t, sent_t, mask_index_t, mask_num_t):
    net.eval()
    with torch.no_grad():
        mask_out, sent_cls_out = net(words_t, words_num_t, sent_t, mask_index_t)

        mask_pred = mask_out.argmax(dim=-1)
        sent_cls_pred = sent_cls_out.argmax(dim=-1)

        return mask_pred, sent_cls_pred

char2idx, idx2char = get_chars(cfg.chars_path)
test_data = TestData(cfg.test_path)

net = Bert(cfg).to(device)
print(net)
show_model_size(net)

try:
    model_path = os.path.abspath(cfg.load_model_path)
    net.load_state_dict(torch.load(os.path.join(model_path, '%s_%.8f_lr_%d_embeddim_%.2f_dropout_%d_layers.pt' % (
        net.name, cfg.lr, cfg.embed_dim, cfg.dropout, cfg.encoder_layers))))
    print('load pre-train model succeed.')
except Exception as e:
    print(e)
    print('load pre-train model failed.')


test_loader = DataLoader(test_data, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

with open(cfg.pred_path, 'w', encoding='utf-8') as f:
    for words_t, words_num_t, sent_t, mask_index_t, mask_num_t in test_loader:
        words_t, words_num_t, sent_t = words_t.to(device), words_num_t.to(device), sent_t.to(device)
        mask_index_t, mask_num_t, = mask_index_t.to(device), mask_num_t.to(device)

        mask_pred, sent_cls_pred = test(net, words_t, words_num_t, sent_t, mask_index_t, mask_num_t)
        batch_size = words_t.size(0)
        maxlen = words_t.size(1)
        mask_len = mask_pred.size(1)

        for i in range(batch_size):
            for j in range(maxlen):
                if words_t[i, j].item() == char2idx['PAD']:
                    continue
                f.write(idx2char[words_t[i, j].item()])
            f.write('\t')
            mask_pred_chars = [idx2char[mask_pred[i, j].item()] for j in range(mask_num_t[i].item())]
            f.write(' '.join(mask_pred_chars) + '\t')
            f.write(str(sent_cls_pred[i].item()) + '\n')

