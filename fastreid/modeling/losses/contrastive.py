import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

class ViewContrastiveLoss(nn.Module):
    def __init__(self, num_instance=4, T=1.0):
        super(ViewContrastiveLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.num_instance = num_instance
        self.T = T

    def forward(self, q, k, label, cfg=None):
        batchSize = q.shape[0]
        N = q.size(0)
        mat_sim = torch.matmul(q, k.transpose(0, 1))
        mat_eq = label.expand(N, N).eq(label.expand(N, N).t()).float()
        # batch hard
        hard_p, hard_n, hard_p_indice, hard_n_indice = self.batch_hard(mat_sim, mat_eq, True)
        l_pos = hard_p.view(batchSize, 1)
        mat_ne = label.expand(N, N).ne(label.expand(N, N).t())
        # positives = torch.masked_select(mat_sim, mat_eq).view(-1, 1)
        negatives = torch.masked_select(mat_sim, mat_ne).view(batchSize, -1)
        out = torch.cat((l_pos, negatives), dim=1) / self.T
        # out = torch.cat((l_pos, l_neg, negatives), dim=1) / self.T
        targets = torch.zeros([batchSize]).cuda(cfg.MODEL.DEVICE).long()
        triple_dist = F.log_softmax(out, dim=1)
        triple_dist_ref = torch.zeros_like(triple_dist).scatter_(1, targets.unsqueeze(1), 1)
        # triple_dist_ref = torch.zeros_like(triple_dist).scatter_(1, targets.unsqueeze(1), 1)*l + torch.zeros_like(triple_dist).scatter_(1, targets.unsqueeze(1)+1, 1) * (1-l)
        loss = (- triple_dist_ref * triple_dist).mean(0).sum()
        return loss

    def batch_hard(self, mat_sim, mat_eq, indice=False):
        sorted_mat_sim, positive_indices = torch.sort(mat_sim + (9999999.) * (1 - mat_eq), dim=1,
                                                           descending=False)
        hard_p = sorted_mat_sim[:, 0]
        hard_p_indice = positive_indices[:, 0]
        sorted_mat_distance, negative_indices = torch.sort(mat_sim + (-9999999.) * (mat_eq), dim=1,
                                                           descending=True)
        hard_n = sorted_mat_distance[:, 0]
        hard_n_indice = negative_indices[:, 0]
        if (indice):
            return hard_p, hard_n, hard_p_indice, hard_n_indice
        return hard_p, hard_n

