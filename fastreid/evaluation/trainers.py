from __future__ import print_function, absolute_import
import time
from .meters import AverageMeter
from fastreid.modeling.losses import *

class ClusterContrastTrainer(object):
    def __init__(self, encoder, memory=None, num_dataset=1000, cfg=None):
        super(ClusterContrastTrainer, self).__init__()
        self.encoder = encoder
        self.memory = memory
        self.agent = PatchMemory(momentum=0.1, num=1)
        self.criterion = Pedal(scale=0.02, k=1).cuda(cfg.MODEL.DEVICE)


    def train(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400, opt=None, cfg=None):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()


        losses = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs = data_loader.next()
            data_time.update(time.time() - end)

            # process inputs
            images, img_path, labels, indexes = self._parse_data(inputs, cfg=cfg)

            # forward
            outs = self._forward(inputs, opt=opt)
            # print("f_out shape: {}".format(f_out.shape))
            # compute loss with the hybrid memory
            # loss = self.memory(f_out, indexes)
            loss_names = ['contrstive_loss']
            loss_dict = {}
            outputs           = outs["outputs"]
            domain_labels     = outs['domains']
            pooled_features   = outputs['pooled_features']
            bn_features       = outputs['bn_features']
            cls_outputs       = ['cls_outputs']
            pred_class_logits = outputs["pred_class_logits"]
            gt_labels         = outs["targets"]
            if "TripletLoss" in loss_names:
                loss_dict['loss_triplet'] = triplet_loss(
                    pooled_features if cfg.MODEL.LOSSES.TRI.FEAT_ORDER == 'before' else bn_features,
                    labels,
                    cfg.MODEL.LOSSES.TRI.MARGIN,
                    cfg.MODEL.LOSSES.TRI.NORM_FEAT,
                    cfg.MODEL.LOSSES.TRI.HARD_MINING,
                    cfg.MODEL.LOSSES.TRI.DIST_TYPE,
                    cfg.MODEL.LOSSES.TRI.LOSS_TYPE,
                    domain_labels,
                    cfg.MODEL.LOSSES.TRI.NEW_POS,
                    cfg.MODEL.LOSSES.TRI.NEW_NEG,
                )
            if 'contrstive_loss' in loss_names: 
                loss_dict['contrstive_loss']  = self.memory(bn_features, labels, cfg, prototype=pred_class_logits, batch=i, domain_labels=domain_labels ) 
            if 'pedal_loss' in loss_names:
                patch_agent, position = self.agent.get_soft_label(img_path, pooled_features, vid=labels, camid=domain_labels, cfg=cfg)
                loss_dict['pedal_loss'] , all_posvid = self.criterion(pooled_features, patch_agent, position, self.agent, vid=labels, camid=domain_labels, cfg=cfg)
                
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})'
                       .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg))

    def _parse_data(self, inputs, cfg=None):
        imgs, img_path, pids, _, indexes = inputs
        return imgs.cuda(cfg.MODEL.DEVICE), img_path, pids.cuda(cfg.MODEL.DEVICE), indexes.cuda(cfg.MODEL.DEVICE)

    def _forward(self, inputs, opt=None):
        #import ipdb; ipdb.set_trace() 
        inputs_dict = {
            "images": inputs[0],
            "targets": inputs[2],
            "camid": inputs[3],
            "img_path": inputs[1],
            # "others": others
            "others": {"domains": inputs[3]}
        }
        return self.encoder(inputs_dict, opt=opt)

