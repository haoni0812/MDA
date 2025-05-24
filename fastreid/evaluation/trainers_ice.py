from __future__ import print_function, absolute_import
import time
import numpy as np
import collections

import torch
import torch.nn as nn
from torch.nn import functional as F
from fastreid.modeling.losses import CrossEntropyLabelSmooth, ViewContrastiveLoss
from .meters import AverageMeter
#from .utils.meters import AverageMeter
#from .evaluation_metrics import accuracy


class ImageTrainer(object):
    def __init__(self, model_1, model_1_ema, num_cluster=500, alpha=0.999, num_instance=4, tau_c=0.5, tau_v=0.09,
                 scale_kl=2.0):
        super(ImageTrainer, self).__init__()
        self.model_1 = model_1
        self.model_1_ema = model_1_ema
        self.alpha = alpha

        self.tau_c = tau_c
        self.tau_v = tau_v
        self.scale_kl = scale_kl

        self.ccloss = CrossEntropyLabelSmooth(num_cluster)
        self.vcloss = ViewContrastiveLoss(num_instance=num_instance, T=tau_v)
        self.kl = nn.KLDivLoss(reduction='batchmean')
        self.crosscam_epoch = 0
        self.beta = 0.07
        self.bg_knn = 50

        self.mse = nn.MSELoss(reduction='sum')

    def train(self, epoch, data_loader_target,
              optimizer, print_freq=1, train_iters=200, centers=None, intra_id_labels=None, intra_id_features=None,
              cams=None, all_pseudo_label=None, opt=None, cfg=None):
        self.model_1.train()
        self.model_1_ema.train()
        #centers = centers.cuda(cfg.MODEL.DEVICE)
        # outliers = outliers.cuda(cfg.MODEL.DEVICE)

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses_ccl = AverageMeter()
        losses_vcl = AverageMeter()
        losses_cam = AverageMeter()
        losses_kl = AverageMeter()
        precisions = AverageMeter()

        self.all_img_cams = torch.tensor(cams).cuda(cfg.MODEL.DEVICE)
        self.unique_cams = torch.unique(self.all_img_cams)
        # print(self.unique_cams)

        self.all_pseudo_label = torch.tensor(all_pseudo_label).cuda(cfg.MODEL.DEVICE)
        self.init_intra_id_feat = intra_id_features
        # print(len(self.init_intra_id_feat))

        # initialize proxy memory
        self.percam_memory = []
        self.memory_class_mapper = []
        self.concate_intra_class = []
        for cc in self.unique_cams:
            percam_ind = torch.nonzero(self.all_img_cams == cc).squeeze(-1)
            uniq_class = torch.unique(self.all_pseudo_label[percam_ind])
            uniq_class = uniq_class[uniq_class >= 0]
            self.concate_intra_class.append(uniq_class)
            cls_mapper = {int(uniq_class[j]): j for j in range(len(uniq_class))}
            self.memory_class_mapper.append(cls_mapper)  # from pseudo label to index under each camera

            if len(self.init_intra_id_feat) > 0:
                # print('initializing ID memory from updated embedding features...')
                proto_memory = self.init_intra_id_feat[cc]
                proto_memory = proto_memory.cuda(cfg.MODEL.DEVICE)
                self.percam_memory.append(proto_memory.detach())
        self.concate_intra_class = torch.cat(self.concate_intra_class)

        if epoch >= self.crosscam_epoch:
            percam_tempV = []
            for ii in self.unique_cams:
                percam_tempV.append(self.percam_memory[ii].detach().clone())
            percam_tempV = torch.cat(percam_tempV, dim=0).cuda(cfg.MODEL.DEVICE)

        end = time.time()
        for i in range(train_iters):
            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)
            # process inputs
            
            inputs_1, inputs_weak, targets, inputs_2, cids, b = self._parse_data_MDA(target_inputs, cfg=cfg)
            #b, c, h, w = inputs_1.size()

            # ids for ShuffleBN
            shuffle_ids, reverse_ids = self.get_shuffle_ids(b, cfg=cfg)
            
            f_out_t1 = self.model_1(inputs_1, opt=opt)
            f_out_t1           = f_out_t1["outputs"]["pooled_features"]
            p_out_t1 = torch.matmul(f_out_t1, centers.transpose(1, 0)) / self.tau_c

            f_out_t2 = self.model_1(inputs_2, opt=opt)
            f_out_t2           = f_out_t2["outputs"]["pooled_features"]

            loss_cam = torch.tensor([0.]).cuda(cfg.MODEL.DEVICE)
            for cc in torch.unique(cids):
                # print(cc)
                inds = torch.nonzero(cids == cc).squeeze(-1)
                percam_targets = targets[inds]
                # print(percam_targets)
                percam_feat = f_out_t1[inds]

                # # intra-camera loss
                # mapped_targets = [self.memory_class_mapper[cc][int(k)] for k in percam_targets]
                # mapped_targets = torch.tensor(mapped_targets).to(torch.device('cuda'))
                # # percam_inputs = ExemplarMemory.apply(percam_feat, mapped_targets, self.percam_memory[cc], self.alpha)
                # percam_inputs = torch.matmul(F.normalize(percam_feat), F.normalize(self.percam_memory[cc].t()))
                # percam_inputs /= self.beta  # similarity score before softmax
                # loss_cam += F.cross_entropy(percam_inputs, mapped_targets)

                # cross-camera loss
                if epoch >= self.crosscam_epoch:
                    associate_loss = 0
                    # target_inputs = percam_feat.mm(percam_tempV.t().clone())
                    target_inputs = torch.matmul(F.normalize(percam_feat), F.normalize(percam_tempV.t().clone()))
                    temp_sims = target_inputs.detach().clone()
                    target_inputs /= self.beta

                    for k in range(len(percam_feat)):
                        ori_asso_ind = torch.nonzero(self.concate_intra_class == percam_targets[k]).squeeze(-1)
                        temp_sims[k, ori_asso_ind] = -10000.0  # mask out positive
                        sel_ind = torch.sort(temp_sims[k])[1][-self.bg_knn:]
                        concated_input = torch.cat((target_inputs[k, ori_asso_ind], target_inputs[k, sel_ind]), dim=0)
                        concated_target = torch.zeros((len(concated_input)), dtype=concated_input.dtype).cuda(cfg.MODEL.DEVICE)
                        #import ipdb; ipdb.set_trace()
                        concated_target[0:len(ori_asso_ind)] = 1.0 / len(ori_asso_ind)
                        associate_loss += -1 * (
                                F.log_softmax(concated_input.unsqueeze(0), dim=1) * concated_target.unsqueeze(
                            0)).sum()
                    loss_cam += 0.5 * associate_loss / len(percam_feat)

            with torch.no_grad():
                inputs_1['images'] = inputs_1['images'][shuffle_ids]
                f_out_t1_ema = self.model_1_ema(inputs_1, opt=opt)
                f_out_t1_ema = f_out_t1_ema["outputs"]["pooled_features"]
                f_out_t1_ema = f_out_t1_ema[reverse_ids]

                inputs_2['images'] = inputs_2['images'][shuffle_ids]
                f_out_t2_ema = self.model_1_ema(inputs_2, opt=opt)
                f_out_t2_ema = f_out_t2_ema["outputs"]["pooled_features"]
                f_out_t2_ema = f_out_t2_ema[reverse_ids]

                inputs_weak['images'] = inputs_weak['images'][shuffle_ids]
                f_out_weak_ema = self.model_1_ema(inputs_weak, opt=opt)
                f_out_weak_ema = f_out_weak_ema["outputs"]["pooled_features"]
                f_out_weak_ema = f_out_weak_ema[reverse_ids]

            loss_ccl = self.ccloss(p_out_t1, targets)
            loss_vcl = self.vcloss(F.normalize(f_out_t1), F.normalize(f_out_t2_ema), targets, cfg=cfg)

            loss_kl = self.kl(F.softmax(
                torch.matmul(F.normalize(f_out_t1), F.normalize(f_out_t2_ema).transpose(1, 0)) / self.scale_kl,
                dim=1).log(),
                              F.softmax(torch.matmul(F.normalize(f_out_weak_ema),
                                                     F.normalize(f_out_weak_ema).transpose(1, 0)) / self.scale_kl,
                                        dim=1)) * 10

            loss = loss_kl #loss_vcl + loss_ccl  # + loss_cam 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self._update_ema_variables(self.model_1, self.model_1_ema, self.alpha, epoch * len(data_loader_target) + i)

            #prec_1, = accuracy(p_out_t1.data, targets.data)

            losses_ccl.update(loss_ccl.item())
            losses_cam.update(loss_cam.item())
            losses_vcl.update(loss_vcl.item())
            losses_kl.update(loss_kl.item())
            #precisions.update(prec_1[0])


            # print log #
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ccl {:.3f}\t'
                      'Loss_hard_instance {:.3f}\t'
                      'Loss_cam {:.3f}\t'
                      'Loss_kl {:.3f}\t'
                      'Prec {:.2%}\t'
                      .format(epoch, i + 1, len(data_loader_target),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ccl.avg,
                              losses_vcl.avg,
                              losses_cam.avg,
                              losses_kl.avg,
                              100))

    def _update_ema_variables(self, model, ema_model, alpha, global_step):
        # alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)



    def _parse_data_MDA(self, inputs, cfg=None):
        imgs_1, imgs_2, img_mutual, pids, cids = inputs
        inputs_1 = imgs_1.cuda(cfg.MODEL.DEVICE)
        inputs_2 = imgs_2.cuda(cfg.MODEL.DEVICE)
        img_mutual = img_mutual.cuda(cfg.MODEL.DEVICE)
        targets = pids.cuda(cfg.MODEL.DEVICE)
        cids = cids.cuda(cfg.MODEL.DEVICE)
        b, c, h, w = inputs_1.size()

        inputs_1= {
            "images": inputs_1,
            "targets": targets,
            "camid": cids,
            "img_path": targets,
            # "others": others
            "others": {"domains": cids}
        }
        inputs_2= {
            "images": inputs_2,
            "targets": targets,
            "camid": cids,
            "img_path": targets,
            # "others": others
            "others": {"domains": cids}
        }
        inputs_mutual= {
            "images": img_mutual,
            "targets": targets,
            "camid": cids,
            "img_path": targets,
            # "others": others
            "others": {"domains": cids}
        }

        return inputs_1, inputs_2, targets, inputs_mutual, cids, b



    def get_shuffle_ids(self, bsz, cfg=None):
        """generate shuffle ids for ShuffleBN"""
        forward_inds = torch.randperm(bsz).long().cuda(cfg.MODEL.DEVICE)
        backward_inds = torch.zeros(bsz).long().cuda(cfg.MODEL.DEVICE)
        value = torch.arange(bsz).long().cuda(cfg.MODEL.DEVICE)
        backward_inds.index_copy_(0, forward_inds, value)
        return forward_inds, backward_inds
