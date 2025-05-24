# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
import torch.nn.functional as F
from .utils import concat_all_gather, euclidean_dist, normalize



def domain_MMD_loss(embedding, domain_labels, norm_feat, norm_flag, kernel_mul, kernel_num, fix_sigma):

    if norm_feat: embedding = normalize(embedding, axis=-1)

    unique_label = torch.unique(domain_labels)
    embedding_all = []
    for i, x in enumerate(unique_label):
        if i == 0:
            import ipdb; ipdb.set_trace()
            embedding_all = embedding[x == domain_labels].unsqueeze(0)
            
        else:
            embedding_all = torch.cat((embedding_all, embedding[x == domain_labels].unsqueeze(0)), dim=0)
    num_domain = len(embedding_all)
    all_set = list()
    for x in range(num_domain):
        for y in range(num_domain):
            if x != y and x < y:
                all_set.append((x, y))
    loss_all = []
    for i in range(len(all_set)):
        num_source = int(embedding_all[all_set[i][0]].size()[0])
        num_target = int(embedding_all[all_set[i][1]].size()[0])
        num_total = num_source + num_target
        source = embedding_all[all_set[i][0]]
        target = embedding_all[all_set[i][1]]
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(num_total, num_total, int(total.size(1)))
        total1 = total.unsqueeze(1).expand(num_total, num_total, int(total.size(1)))
        if norm_flag == 'l1norm':
            dist = (torch.abs(total0 - total1)).sum(2)
        else:
            dist = ((total0 - total1) ** 2).sum(2)

        if fix_sigma > 0:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(dist.data) / (num_total ** 2 - num_total)

        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-dist / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        kernels = sum(kernel_val)
        XX = kernels[:num_source, :num_source]
        YY = kernels[num_source:, num_source:]
        XY = kernels[:num_source, num_source:]
        YX = kernels[num_source:, :num_source]
        loss_all.append(torch.mean(XX + YY - XY -YX))


    return torch.mean(torch.stack(loss_all))
