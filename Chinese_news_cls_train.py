# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from utils import time_since, show_model_size, get_chars
from Chinese_News_Classification.loaders.train_loader import TrainData, collate_fn
from Chinese_News_Classification.loaders.valid_loader import ValidData
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


def train(net, words_t, words_num_t, sent_t, label_t, loss_fn, opt):

    net.train()
    net.zero_grad()

    para_cls_out = net(words_t, words_num_t, sent_t)
    loss = loss_fn(para_cls_out, label_t)

    loss.backward()

    opt.step()

    return loss.item()


def valid(net, words_t, words_num_t, sent_t, label_t, loss_fn):
    net.eval()
    with torch.no_grad():
        para_cls_out = net(words_t, words_num_t, sent_t)

        loss = loss_fn(para_cls_out, label_t)

        para_cls_pred = para_cls_out.argmax(dim=-1)
        para_cls_acc = para_cls_pred.eq(label_t).float().sum() / words_t.size(0)

        return loss.item(), para_cls_acc.item()


char2idx, idx2char = get_chars(cfg.chars_path)
label2idx, idx2label = get_chars(cfg.para_cls_labels_path)
train_data = TrainData(cfg.train_path, char2idx, label2idx)
valid_data = ValidData(cfg.valid_path, char2idx, label2idx)

bert_net = Bert(cfg).to(device)

#try:
#    model_path = os.path.abspath(cfg.bert_load_model_path)
#    bert_net.load_state_dict(torch.load(os.path.join(model_path, '%s_%.8f_lr_%d_embeddim_%.2f_dropout_%d_layers.pt' % (
#        bert_net.name, cfg.lr, cfg.embed_dim, cfg.dropout, cfg.encoder_layers))))
#    print('load pre-train {} succeed.'.format(bert_net.name))
#except Exception as e:
#    print(e)
#    print('load pre-train {} failed.'.format(bert_net.name))

para_cls_net = Para_cls_model(cfg, bert_net).to(device)
print(para_cls_net)
show_model_size(para_cls_net)

try:
    model_path = os.path.abspath(cfg.para_cls_load_model_path)
    para_cls_net.load_state_dict(torch.load(os.path.join(model_path, '%s_%.8f_lr_%d_embeddim_%.2f_dropout_%d_layers.pt' % (
        para_cls_net.name, cfg.lr, cfg.embed_dim, cfg.dropout, cfg.encoder_layers))))
    opt = optim.Adam(para_cls_net.parameters(), lr=cfg.cur_lr)
    print('load pre-train {} succeed.'.format(para_cls_net.name))
except Exception as e:
    print(e)
    opt = optim.Adam(para_cls_net.parameters(), lr=cfg.lr)
    print('load pre-train {} failed.'.format(para_cls_net.name))

para_cls_loss_fn = nn.CrossEntropyLoss()
lr_sche = lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.9, patience=50, verbose=True, min_lr=1e-6)

best_acc = cfg.best_acc
print('init acc = {}'.format(best_acc))
sd = para_cls_net.state_dict()

train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_data, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

for epoch in range(cfg.epochs):
    start = time.time()
    train_loss = 0.0
    for i,(words_t, words_num_t, sent_t, label_t) in enumerate(train_loader):
        words_t, words_num_t, sent_t, label_t = words_t.to(device), words_num_t.to(device), sent_t.to(device), label_t.to(device)

        loss = train(para_cls_net, words_t, words_num_t, sent_t, label_t, para_cls_loss_fn, opt)
        train_loss += loss

        if (i+1) % cfg.print_every == 0:
            lr_sche.step(train_loss)
            print('Epoch %d, %s, (%d -- %d %%), train loss %.3f' % (
                epoch, time_since(start, (i + 1) / len(train_loader)), i + 1, (i + 1) * 100 / len(train_loader),
                train_loss / cfg.print_every))
            train_loss = 0.0

        if (i+1) % cfg.valid_every == 0:
            valid_loss,  valid_acc = 0., 0.
            for words_t, words_num_t, sent_t, label_t in valid_loader:
                words_t, words_num_t, sent_t, label_t = words_t.to(device), words_num_t.to(device), sent_t.to(device), label_t.to(device)

                valid_l, valid_a = valid(para_cls_net, words_t, words_num_t, sent_t, label_t, para_cls_loss_fn)
                valid_loss += valid_l
                valid_acc += valid_a

            print('Epoch %d, step %d, valid loss %.3f, valid_acc %.3f' % (epoch, i + 1, valid_loss / len(valid_loader), valid_acc / len(valid_loader)))

            if best_acc < (valid_acc / len(valid_loader)):
                best_acc = (valid_acc / len(valid_loader))
                sd = para_cls_net.state_dict()
                print('best acc = {}'.format(best_acc))
                model_path = os.path.abspath(cfg.para_cls_save_model_path)
                torch.save(sd, os.path.join(model_path, '%s_%.8f_lr_%d_embeddim_%.2f_dropout_%d_layers.pt' % (
                    para_cls_net.name, cfg.lr, cfg.embed_dim, cfg.dropout, cfg.encoder_layers
                )))
