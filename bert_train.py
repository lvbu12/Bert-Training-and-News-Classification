# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from utils import time_since, show_model_size, get_chars
from loaders.idx_train_loader import TrainData, collate_fn
from loaders.idx_valid_loader import ValidData
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


def train(net, words_t, words_num_t, sent_t, mask_index_t, mask_num_t,
          mask_label_t, label_t, mask_loss_fn, sent_cls_loss_fn, opt):
    net.train()
    net.zero_grad()

    mask_out, sent_cls_out = net(words_t, words_num_t, sent_t, mask_index_t)
    mask_loss, _ = mask_loss_fn(mask_out, mask_label_t, mask_num_t)
    sent_cls_loss = sent_cls_loss_fn(sent_cls_out, label_t)

    total_loss = (mask_loss + sent_cls_loss * words_t.size(0)) / (mask_num_t.float().sum() + words_t.size(0))
    total_loss.backward()
    #mask_loss.backward(retain_graph=True)
    #sent_cls_loss.backward()

    opt.step()

    return total_loss.item()


def valid(net, words_t, words_num_t, sent_t, mask_index_t, mask_num_t,
          mask_label_t, label_t, mask_loss_fn, sent_cls_loss_fn):
    net.eval()
    with torch.no_grad():
        mask_out, sent_cls_out = net(words_t, words_num_t, sent_t, mask_index_t)

        mask_loss, mask_acc = mask_loss_fn(mask_out, mask_label_t, mask_num_t)
        mask_loss = mask_loss / mask_num_t.float().sum()
        sent_cls_loss = sent_cls_loss_fn(sent_cls_out, label_t)
        total_loss = mask_loss + sent_cls_loss

        sent_cls_pred = sent_cls_out.argmax(dim=-1)
        sent_cls_acc = sent_cls_pred.eq(label_t).float().sum() / words_t.size(0)

        return total_loss.item(), mask_loss.item(), mask_acc.item(), sent_cls_loss.item(), sent_cls_acc.item()


char2idx, idx2char = get_chars(cfg.chars_path)
train_data = TrainData(cfg.train_path)
valid_data = ValidData(cfg.valid_path)

net = Bert(cfg).to(device)
print(net)
show_model_size(net)

try:
    model_path = os.path.abspath(cfg.load_model_path)
    net.load_state_dict(torch.load(os.path.join(model_path, '%s_%.8f_lr_%d_embeddim_%.2f_dropout_%d_layers.pt' % (
        net.name, cfg.lr, cfg.embed_dim, cfg.dropout, cfg.encoder_layers))))
    opt = optim.Adam(net.parameters(), lr=cfg.cur_lr)
    print('load pre-train model succeed.')
except Exception as e:
    print(e)
    opt = optim.Adam(net.parameters(), lr=cfg.lr)
    print('load pre-train model failed.')

sent_cls_loss_fn = nn.CrossEntropyLoss()
lr_sche = lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.9, patience=50, verbose=True, min_lr=1e-6)

best_acc = cfg.best_acc
print('init acc = {}'.format(best_acc))
sd = net.state_dict()

train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_data, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

for epoch in range(cfg.epochs):
    start = time.time()
    train_loss = 0.0
    for i,(words_t, words_num_t, sent_t, mask_index_t, mask_num_t, mask_label_t, label_t) in enumerate(train_loader):
        words_t,words_num_t,sent_t = words_t.to(device),words_num_t.to(device),sent_t.to(device)
        mask_index_t,mask_num_t, = mask_index_t.to(device),mask_num_t.to(device)
        mask_label_t,label_t = mask_label_t.to(device),label_t.to(device)

        loss = train(net, words_t, words_num_t, sent_t, mask_index_t, mask_num_t,
                     mask_label_t, label_t, mask_cross_entropy_loss, sent_cls_loss_fn, opt)
        train_loss += loss

        if (i+1) % cfg.print_every == 0:
            lr_sche.step(train_loss)
            print('Epoch %d, %s, (%d -- %d %%), train loss %.3f' % (
                epoch, time_since(start, (i + 1) / len(train_loader)), i + 1, (i + 1) * 100 / len(train_loader),
                train_loss / cfg.print_every))
            train_loss = 0.0

        if (i+1) % cfg.valid_every == 0:
            valid_loss, valid_mask_loss, valid_sent_cls_loss, valid_mask_acc, valid_sent_cls_acc = 0., 0., 0., 0., 0.
            for words_t, words_num_t, sent_t, mask_index_t, mask_num_t, mask_label_t, label_t in valid_loader:
                words_t, words_num_t,sent_t = words_t.to(device), words_num_t.to(device), sent_t.to(device)
                mask_index_t, mask_num_t, = mask_index_t.to(device), mask_num_t.to(device)
                mask_label_t, label_t = mask_label_t.to(device), label_t.to(device)

                total_loss, mask_loss, mask_acc, sent_cls_loss, sent_cls_acc = valid(
                    net, words_t, words_num_t, sent_t, mask_index_t, mask_num_t, mask_label_t, label_t,
                    mask_cross_entropy_loss, sent_cls_loss_fn)
                valid_loss += total_loss
                valid_mask_loss += mask_loss
                valid_mask_acc += mask_acc
                valid_sent_cls_loss += sent_cls_loss
                valid_sent_cls_acc += sent_cls_acc

            print('Epoch %d, step %d, valid loss %.3f, valid_mask_loss %.3f, valid_mask_acc %.3f, valid_sent_cls_loss %.3f, valid_sent_cls_acc %.3f' % (
                epoch, i + 1, valid_loss / len(valid_loader), valid_mask_loss / len(valid_loader),
                valid_mask_acc / len(valid_loader), valid_sent_cls_loss / len(valid_loader),
                valid_sent_cls_acc / len(valid_loader)))

            valid_acc = (valid_mask_acc / len(valid_loader) + valid_sent_cls_acc / len(valid_loader)) / 2
            if best_acc < valid_acc:
                best_acc = valid_acc
                sd = net.state_dict()
                print('best acc = {}'.format(best_acc))
                model_path = os.path.abspath(cfg.save_model_path)
                torch.save(sd, os.path.join(model_path, '%s_%.8f_lr_%d_embeddim_%.2f_dropout_%d_layers.pt' % (
                    net.name, cfg.lr, cfg.embed_dim, cfg.dropout, cfg.encoder_layers
                )))
