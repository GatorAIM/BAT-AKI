import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingLoss(nn.Module):
    def __init__(self, label_smoothing, vocab_size, ignore_index=0):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction='batchmean')
        self.confidence = 1.0 - label_smoothing
        self.smoothing = label_smoothing
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        pred = F.log_softmax(pred, dim=-1)
        true_dist = torch.full_like(pred, self.smoothing / (self.vocab_size - 2))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        true_dist[:, self.ignore_index] = 0
        true_dist.masked_fill_(target.unsqueeze(1) == self.ignore_index, 0)
        return self.criterion(pred, true_dist)