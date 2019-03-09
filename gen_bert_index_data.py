# -*- coding: utf-8 -*-
from utils import get_chars, get_labels, flatten
import random
import collections
import os
import sys


class Dataset(object):

    def __init__(self, filepath, char2idx):

        self.data = open(filepath, 'r', encoding='utf-8').readlines()
        self.char2idx = char2idx
        self.chars = [key for key in self.char2idx]

    def _gen_item(self, index):
        
        one, two, label = self.data[index][:-1].split('\t')
        one_idx = [self.char2idx[c] for c in one]
        two_idx = [self.char2idx[c] for c in two]
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

        mask_index = sorted(mask_dict)
        mask_label_idx = [mask_dict[k] for k in mask_index]

        assert len(words_idx) == len(sent_idx)
        assert len(mask_index) == len(mask_label_idx)

        #return words_idx, sent_idx, mask_index, mask_label_idx, label_idx
        res = ' '.join([str(idx) for idx in words_idx]) + '\t'
        res += ' '.join([str(idx) for idx in sent_idx]) + '\t'
        res += ' '.join([str(idx) for idx in mask_index]) + '\t'
        res += ' '.join([str(idx) for idx in mask_label_idx]) + '\t'
        res += label + '\n'
        
        return res
        
    def __len__(self):
        return len(self.data)

    def gen_bert_data(self, output_file):

        with open(output_file, 'w', encoding='utf-8') as f:
            for i in range(len(self.data)):
                f.write(self._gen_item(i))


if __name__ == "__main__":

    if len(sys.argv) != 4:
        print('Using: python %s chars_vocab_path raw_text_path output_idx_data_path')
        sys.exit(1)

    char2idx, idx2char = get_chars(sys.argv[1])
    data = Dataset(sys.argv[2], char2idx)
    data.gen_bert_data(sys.argv[3])
    #print(data._gen_item(0))
    pass
