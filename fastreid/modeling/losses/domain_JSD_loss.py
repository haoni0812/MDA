# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
import torch.nn.functional as F
from .utils import concat_all_gather, euclidean_dist, normalize


def domain_JSD_loss(embedding, domain_labels, norm_feat):

    if norm_feat: embedding = normalize(embedding, axis=-1)

    unique_label = torch.unique(domain_labels)
    embedding_all = []
    for i, x in enumerate(unique_label):
        if i == 0:
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




    return 0.0
