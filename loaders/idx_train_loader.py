# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


class TrainData(Dataset):

    def __init__(self, filepath):

        self.data = open(filepath, 'r', encoding='utf-8').readlines()

    def __getitem__(self, index):
        
        words_idx, sent_idx, mask_index, mask_label_idx, label_idx = self.data[index][:-1].split('\t')
        
        words_idx = [int(idx) for idx in words_idx.split(' ')]
        sent_idx = [int(idx) for idx in sent_idx.split(' ')]
        mask_index = [int(idx) for idx in mask_index.split(' ')]
        mask_label_idx = [int(idx) for idx in mask_label_idx.split(' ')]
        label_idx = int(label_idx)

        assert len(words_idx) == len(sent_idx)
        assert len(mask_index) == len(mask_label_idx)

        return words_idx, sent_idx, mask_index, mask_label_idx, label_idx
        
    def __len__(self):
        return len(self.data)


def collate_fn(data):

    words_idx = [d[0] for d in data]
    sent_idx = [d[1] for d in data]
    mask_index = [d[2] for d in data]
    mask_label_idx = [d[3] for d in data]
    label_idx = [d[4] for d in data]

    words_num = [len(idx) for idx in words_idx]
    mask_num = [len(idx) for idx in mask_index]

    words_ts = [torch.tensor(idx, dtype=torch.long) for idx in words_idx]
    sent_ts = [torch.tensor(idx, dtype=torch.long) for idx in sent_idx]
    mask_index_ts = [torch.tensor(idx, dtype=torch.long) for idx in mask_index]
    mask_label_ts = [torch.tensor(idx, dtype=torch.long) for idx in mask_label_idx]

    words_ts = [F.pad(t, pad=(0, 512-t.size(0))).view(1, -1) for t in words_ts]
    sent_ts = [F.pad(t, pad=(0, 512-t.size(0))).view(1, -1) for t in sent_ts]
    mask_index_ts = [F.pad(t, pad=(0, 77-t.size(0))).view(1, -1) for t in mask_index_ts]
    mask_label_ts = [F.pad(t, pad=(0, 77-t.size(0))).view(1, -1) for t in mask_label_ts]

    words_t = torch.cat(words_ts, dim=0)
    sent_t = torch.cat(sent_ts, dim=0)
    mask_index_t = torch.cat(mask_index_ts, dim=0)
    mask_label_t = torch.cat(mask_label_ts, dim=0)

    words_num_t = torch.tensor(words_num, dtype=torch.long)
    mask_num_t = torch.tensor(mask_num, dtype=torch.long)
    label_t = torch.tensor(label_idx, dtype=torch.long)

    return words_t, words_num_t, sent_t, mask_index_t, mask_num_t, mask_label_t, label_t


if __name__ == "__main__":

    data = TrainData('../idx_data/demo_train.txt')
#    words_idx, sent_idx, mask_index, mask_label_idx, label_idx = data[0]
#    print(words_idx)
#    print(sent_idx)
#    print(mask_index)
#    print(mask_label_idx)
#    print(label_idx)
#    words = [idx2char[idx] for idx in words_idx]
#    print(words)
#    mask_label = [idx2char[idx] for idx in mask_label_idx]
#    print(mask_label)
#    print(len(words_idx), len(sent_idx), len(mask_index), len(mask_label_idx))
#
    print('size of data = ', len(data))
    print(data[1])
#    maxlen = 0
#    for i in range(len(data)):
#        if i % 1000 == 0:
#            print(i)
#        words_idx, sent_idx, mask_index, mask_label_idx, label_idx = data[i]
#        if(maxlen < len(words_idx)):
#            maxlen = len(words_idx)
#    print('maxlen of lin = ', maxlen)
    loader = DataLoader(data, batch_size=4, shuffle=False, collate_fn=collate_fn)
    for words_t, words_num_t, sent_t, mask_index_t, mask_num_t, mask_label_t, label_t in loader:
        print(words_t)
        print(words_num_t)
        print(sent_t)
        print(mask_index_t)
        print(mask_num_t)
        print(mask_label_t)
        print(label_t)
        print(words_t.size(), words_num_t.size(), sent_t.size(), mask_index_t.size(), mask_num_t.size(), mask_label_t.size(), label_t.size())
        break
