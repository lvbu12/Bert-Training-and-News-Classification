# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F


def sequence_mask(sequence, maxlen=None):
    device = sequence.device
    
    if maxlen is None:
        maxlen = sequence.data.max()
    batch_size = sequence.size(0)
    seq_range = torch.arange(0, maxlen).long().to(device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, maxlen)
    seq_len_expand = sequence.unsqueeze(1).expand_as(seq_range_expand)

    return seq_range_expand < seq_len_expand


def mask_cross_entropy_loss(out, label, label_len):

    """
    Args:
        out (FloatTensor): shape (batch_size, maxlen, output_size)
        label (LongTensor): shape (batch_size, maxlen)
        label_len (LongTensor): shape (batch)
    """
    pred = out.argmax(dim=-1)
    acc = pred.eq(label).float()
    out_flat = out.contiguous().view(-1, out.size(-1))
    #print('size of out_flat = ', out_flat.size())
    log_probs_flat = F.log_softmax(out_flat, dim=-1)
    #print('size of log_probs_flat = ', log_probs_flat.size())
    label_flat = label.contiguous().view(-1, 1)
    #print('size of label_flat = ', label_flat.size())
    loss_flat = - torch.gather(log_probs_flat, dim=1, index=label_flat)
    #print('size of loss flat = ', loss_flat.size())
    loss = loss_flat.view(*label.size())
    #print('size of loss = ', loss.size())
    mask = sequence_mask(label_len, maxlen=label.size(1))
    #print('size of mask = ', mask.size())
    loss = loss * mask.float()
    #print('size of loss = ', loss.size())
    #loss = loss.sum() / label_len.float().sum()
    loss = loss.sum()
    #print('size of loss = ', loss.size())
    acc = acc * mask.float()
    acc = acc.sum() / label_len.float().sum()

    return loss, acc



if __name__ == "__main__":

    out = torch.randn(3, 4, 100)
    label = torch.tensor([[10, 20, 0, 0], [12, 35, 17, 0], [42, 0, 0, 0]], dtype=torch.long)
    a = torch.tensor([2, 3, 1], dtype=torch.long)
    print(mask_cross_entropy_loss(out, label, a))
