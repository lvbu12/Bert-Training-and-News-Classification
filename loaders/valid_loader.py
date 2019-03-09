# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from ../utils import get_chars, get_labels, flatten
import random
import collections


class ValidData(Dataset):

    def __init__(self, filepath, char2idx):

        self.data = open(filepath, 'r', encoding='utf-8').readlines()
        self.char2idx = char2idx
        self.chars = [key for key in self.char2idx]

    def __getitem__(self, index):
        
        random.seed(index)
        one, two, label = self.data[index][:-1].split('\t')
        one_idx = [self.char2idx[c] if c in self.char2idx else self.char2idx['UNK'] for c in one]
        two_idx = [self.char2idx[c] if c in self.char2idx else self.char2idx['UNK'] for c in two]
        words_idx = [self.char2idx['[CLS]']] + one_idx + [self.char2idx['[SEP]']] + two_idx + [self.char2idx['[SEP]']]
        sent_idx = [1] + [1 for _ in one_idx] + [1] + [2 for _ in two_idx] + [2]
        i = 0
        mask_num = int(0.15 * len(words_idx))
        shuf_idx = list(range(len(words_idx)))
        random.shuffle(shuf_idx)
        tmp_set = set()
        mask_dict = collections.OrderedDict()
        while len(mask_dict) < mask_num:
            if words_idx[shuf_idx[i]] == self.char2idx['[CLS]'] or words_idx[shuf_idx[i]] == self.char2idx['[SEP]']:
                i += 1
                continue
            if shuf_idx[i] in tmp_set:
                i += 1
                continue
            tmp_set.add(shuf_idx[i])
            mask_dict[shuf_idx[i]] = words_idx[shuf_idx[i]]
            if random.random() < 0.8:
                words_idx[shuf_idx[i]] = self.char2idx['[MASK]']
            else:
                if random.random() < 0.5:
                    pass
                else:
                    words_idx[shuf_idx[i]] = self.char2idx[random.choice(self.chars)]
            i += 1

        label_idx = int(label)
        
        mask_index = sorted(mask_dict)
        mask_label_idx = [mask_dict[k] for k in mask_index]

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

    char2idx, idx2char = get_chars('../corpus/chars.lst')
    data = ValidData('../data/valid.txt', char2idx)
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
