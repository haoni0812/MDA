# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
import torch.nn.functional as F
from .utils import concat_all_gather, euclidean_dist, normalize, cosine_dist, cosine_sim

def domain_TEST_loss(embedding, domain_labels, norm_feat, type):

    type = 'euclidean'# 'cosine', 'euclidean'
    # eps=1e-05
    #if norm_feat: embedding = normalize(embedding, axis=-1)
    unique_label = torch.unique(domain_labels)
    embedding_all = list()
    for i, x in enumerate(unique_label):
        embedding_all.append(embedding[x == domain_labels])
    num_domain = len(embedding_all)
    loss_all = []
    for i in range(num_domain):
        feat = embedding_all[i]
        center_feat = torch.mean(feat, 0)
        if type == 'euclidean':
            dist_ap = euclidean_dist(center_feat.view(1, -1), feat)
            for j in range(num_domain):
                if j == i: continue
                center_feat_j = torch.mean(embedding_all[j], 0)
                dist_an = euclidean_dist(center_feat_j.view(1, -1), feat)
                y = dist_an.new().resize_as_(dist_an).fill_(1)
                loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=0.1)
                loss_all.append(loss)
        elif type == 'cosine':
            loss = torch.mean(cosine_dist(center_feat.view(1, -1), feat))
            loss_all.append(-loss)
        elif type == 'cosine_sim':
            loss = torch.mean(cosine_sim(center_feat.view(1, -1), feat))
            loss_all.append(loss)

    loss_all = torch.mean(torch.stack(loss_all))

    return loss_all


def domain_SCT_loss(embedding, domain_labels, norm_feat, type):

    # type = 'cosine' # 'cosine', 'euclidean'
    # eps=1e-05
    if norm_feat: embedding = normalize(embedding, axis=-1)
    unique_label = torch.unique(domain_labels)
    embedding_all = list()
    for i, x in enumerate(unique_label):
        embedding_all.append(embedding[x == domain_labels])
    num_domain = len(embedding_all)
    loss_all = []
    for i in range(num_domain):
        feat = embedding_all[i]
        center_feat = torch.mean(feat, 0)
        if type == 'euclidean':
            loss = torch.mean(euclidean_dist(center_feat.view(1, -1), feat))
            loss_all.append(-loss)
        elif type == 'cosine':
            loss = torch.mean(cosine_dist(center_feat.view(1, -1), feat))
            loss_all.append(-loss)
        elif type == 'cosine_sim':
            loss = torch.mean(cosine_sim(center_feat.view(1, -1), feat))
            loss_all.append(loss)

    loss_all = torch.mean(torch.stack(loss_all))

    return loss_all

