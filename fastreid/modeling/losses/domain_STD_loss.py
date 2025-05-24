# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
import torch.nn.functional as F
from .utils import concat_all_gather, euclidean_dist, normalize


def domain_STD_loss(embedding, domain_labels, norm_feat, std_type, log_scale):

    # eps=1e-05
    if norm_feat: embedding = normalize(embedding, axis=-1)

    if 'all' in std_type:
        torch.mean(torch.std(embedding, 0))  # 0.05
        if std_type == 'all':
            # loss_all = torch.std(embedding, unbiased=False)
            loss_all = torch.std(embedding)
        elif std_type == 'all_channel':
            # loss_all = torch.mean(torch.std(embedding, 0, unbiased=False))
            loss_all = torch.mean(torch.std(embedding, 0))
        if log_scale:
            # loss_all = -torch.log(loss_all + 1e-12)
            loss_all = -torch.log(loss_all)
        else:
            loss_all = -loss_all
    else:
        unique_label = torch.unique(domain_labels)
        embedding_all = []
        for i, x in enumerate(unique_label):
            if i == 0:
                embedding_all = embedding[x == domain_labels].unsqueeze(0)
            else:
                embedding_all = torch.cat((embedding_all, embedding[x == domain_labels].unsqueeze(0)), dim=0)
        num_domain = len(embedding_all)
        loss_all = []
        for i in range(num_domain):
            feat = embedding_all[i]

            # torch.mean(torch.std(feat, 0)) # 0.05
            if std_type == 'domain':
                # loss = torch.std(feat, unbiased=False)
                loss = torch.std(feat)
            elif std_type == 'domain_channel':
                # loss = torch.mean(torch.std(feat, 0, unbiased=False))
                loss = torch.mean(torch.std(feat, 0))

            if log_scale:
                # loss = -torch.log(loss + 1e-12)
                loss = -torch.log(loss)
            else:
                loss = -loss
            loss_all.append(loss)
        loss_all = torch.mean(torch.stack(loss_all))
    # print("std:{},norm:{},log:{},result:{}".format(std_type, norm_feat, log_scale, loss_all))

    return loss_all
