import torch
from torch import Tensor


def same_value_mask(seq1: Tensor, seq2: Tensor):
    s1 = list(seq1.size())
    s2 = list(seq2.size())
    s1.append( seq2.size(-1))
    s2.insert(-1, seq1.size(-1))
    cmp = seq1.unsqueeze(-1).expand(s1) == seq2.unsqueeze(-2).expand(s2)
    return cmp.to(torch.float)

