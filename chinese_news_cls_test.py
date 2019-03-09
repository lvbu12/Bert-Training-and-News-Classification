# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from utils import time_since, show_model_size, get_chars
from Chinese_News_Classification.loaders.test_loader import TestData, collate_fn
from config import Config
from modules.model import Bert, Para_cls_model
import argparse
import os
import sys
import time

parser = argparse.ArgumentParser(description='add config path')
parser.add_argument('-config_path', type=str, default='Configs/para_cls.json')

args = parser.parse_args()

cfg = Config.from_json_file(args.config_path)
for key, val in cfg.__dict__.items():
    print('{} = {}'.format(key, val))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test(net, words_t, words_num_t, sent_t):
    net.eval()
    with torch.no_grad():
        para_cls_out = net(words_t, words_num_t, sent_t)


        para_cls_pred = para_cls_out.argmax(dim=-1)

        return para_cls_out


char2idx, idx2char = get_chars(cfg.chars_path)
label2idx, idx2label = get_chars(cfg.para_cls_labels_path)

test_data = TestData(cfg.test_path, char2idx)

bert_net = Bert(cfg).to(device)

para_cls_net = Para_cls_model(cfg, bert_net).to(device)
print(para_cls_net)
show_model_size(para_cls_net)

try:
    model_path = os.path.abspath(cfg.para_cls_load_model_path)
    para_cls_net.load_state_dict(torch.load(os.path.join(model_path, '%s_%.8f_lr_%d_embeddim_%.2f_dropout_%d_layers.pt' % (
        para_cls_net.name, cfg.lr, cfg.embed_dim, cfg.dropout, cfg.encoder_layers))))
    print('load pre-train {} succeed.'.format(para_cls_net.name))
except Exception as e:
    print(e)
    print('load pre-train {} failed.'.format(para_cls_net.name))


test_loader = DataLoader(test_data, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

with open(cfg.pred_path, 'w', encoding='utf-8') as f:

    for words_t, words_num_t, sent_t in test_loader:
        words_t, words_num_t, sent_t = words_t.to(device), words_num_t.to(device), sent_t.to(device)

        pred = test(para_cls_net, words_t, words_num_t, sent_t)
        batch_size, maxlen = words_t.size()
        for i in range(batch_size):
            for j in range(maxlen):
                if words_t[i, j].item() == idx2char['PAD']:
                    continue
                f.write(idx2char[words_t[i, j].item()])
            f.write('\t' + str(pred[i].item()) + '\n')
