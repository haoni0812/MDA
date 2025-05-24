from cmath import isnan
import torch
import torch.nn as nn
from torch.nn import functional as F
class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets, all_posvid=None, soft_label=False, soft_weight=0.1, soft_lambda=0.2):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        all_posvid = torch.cat(all_posvid, dim=1)
        soft_targets = []
        for i in range(all_posvid.size(0)):
            s_id, s_num = torch.unique(all_posvid[i,:], return_counts=True)
            sum_num = s_num.sum()
            temp = torch.zeros(inputs.size(1)).cuda().scatter_(0, s_id, (soft_lambda/sum_num)*s_num)
            soft_targets.append(temp)
            
        soft_targets = torch.stack(soft_targets, dim=0)

        
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        if soft_label:
            soft_targets = (1 - soft_lambda) * targets + soft_targets
            targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
            loss = (- targets * log_probs).mean(0).sum()*(1 - soft_weight) + \
                (- soft_targets * log_probs).mean(0).sum()*soft_weight
            # if torch.isnan(loss).item():
            #     print("====nan!!!====\n{}\n{}".format((- targets * log_probs).mean(0).sum(), (- soft_targets * log_probs).mean(0).sum()))
                
        else:
            targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
            loss = (- targets * log_probs).mean(0).sum()
        return loss

class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()