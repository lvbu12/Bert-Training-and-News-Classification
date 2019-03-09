# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from utils import get_chars, get_labels, flatten


class TestData(Dataset):

    def __init__(self, filepath, char2idx):

        self.data = open(filepath, 'r', encoding='utf-8').readlines()
        self.char2idx = char2idx

    def __getitem__(self, index):
        
        para = self.data[index][:-1].split('\t')[0]

        words_idx = [self.char2idx['[CLS]']] + [self.char2idx[c] if c in self.char2idx else self.char2idx['UNK'] for c in para] + [self.char2idx['[SEP]']]
        sent_idx = [1 for _ in words_idx]

        return words_idx, sent_idx
        
    def __len__(self):
        return len(self.data)


def collate_fn(data):

    words_idx = [d[0] for d in data]
    sent_idx = [d[1] for d in data]

    words_num = [len(idx) for idx in words_idx]

    words_ts = [torch.tensor(idx, dtype=torch.long) for idx in words_idx]
    sent_ts = [torch.tensor(idx, dtype=torch.long) for idx in sent_idx]

    words_ts = [F.pad(t, pad=(0, 512-t.size(0))).view(1, -1) for t in words_ts]
    sent_ts = [F.pad(t, pad=(0, 512-t.size(0))).view(1, -1) for t in sent_ts]

    words_t = torch.cat(words_ts, dim=0)
    sent_t = torch.cat(sent_ts, dim=0)

    words_num_t = torch.tensor(words_num, dtype=torch.long)

    return words_t, words_num_t, sent_t


if __name__ == "__main__":

    char2idx, idx2char = get_chars('../../corpus/chars.lst')
    label2idx, idx2label = get_chars('../../corpus/labels.lst')
    data = TestData('../data/test.txt', char2idx)
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
#    print(data[1])
#    para_idx, para_label = data[1]
#    para_1 = [idx2char[idx] for idx in para_idx]
#    print(para_1)
#    maxlen = 0
#    for i in range(len(data)):
#        if i % 1000 == 0:
#            print(i)
#        words_idx, sent_idx, mask_index, mask_label_idx, label_idx = data[i]
#        if(maxlen < len(words_idx)):
#            maxlen = len(words_idx)
#    print('maxlen of lin = ', maxlen)
    loader = DataLoader(data, batch_size=4, shuffle=False, collate_fn=collate_fn)
    for words_t, words_num_t, sent_t in loader:
        print(words_t)
        print(words_num_t)
        print(sent_t)
        print(words_t.size(), words_num_t.size(), sent_t.size())
        break
