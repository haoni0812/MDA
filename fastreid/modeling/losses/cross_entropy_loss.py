# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
import torch.nn.functional as F

from fastreid.utils.events import get_event_storage



def cross_entropy_loss(pred_class_logits, gt_classes, eps, alpha=0.2, test_time=False, lamda=None):
    num_classes = pred_class_logits.size(1)
    if test_time:
        #生成概率均等的标签
        log_probs = F.log_softmax(pred_class_logits, dim=1)
        targets = torch.ones_like(log_probs)
        targets *= 1 / (num_classes )

        loss = (-targets * log_probs).sum(dim=1)
        with torch.no_grad():
            non_zero_cnt = max(loss.nonzero(as_tuple=False).size(0), 1)

        loss = loss.sum() / non_zero_cnt

        return loss        

    eps = 0
    if eps >= 0:
        smooth_param = eps
    else:
        # Adaptive label smooth regularization
        soft_label = F.softmax(pred_class_logits, dim=1)
        smooth_param = alpha * soft_label[torch.arange(soft_label.size(0)), gt_classes].unsqueeze(1)

    log_probs = F.log_softmax(pred_class_logits, dim=1)
    with torch.no_grad():
        targets = torch.ones_like(log_probs)
        targets *= smooth_param / (num_classes - 1)
        targets.scatter_(1, gt_classes.data.unsqueeze(1), (1 - smooth_param))
    # import ipdb; ipdb.set_trace()
    if lamda:
        half_batch_size = targets.size(0) // 2
        source_input = targets[:half_batch_size]
        target_input = targets[half_batch_size:]
        mixed_input = lamda * source_input + (1 - lamda) * target_input
        targets = torch.cat([mixed_input, targets[half_batch_size:]], dim=0)
        loss = (-targets[:half_batch_size] * log_probs[:half_batch_size]).sum(dim=1)
        
    else:
        loss = (-targets * log_probs).sum(dim=1)

    """
    # confidence penalty
    conf_penalty = 0.3
    probs = F.softmax(pred_class_logits, dim=1)
    entropy = torch.sum(-probs * log_probs, dim=1)
    loss = torch.clamp_min(loss - conf_penalty * entropy, min=0.)
    """

    with torch.no_grad():
        non_zero_cnt = max(loss.nonzero(as_tuple=False).size(0), 1)

    loss = loss.sum() / non_zero_cnt

    return loss
