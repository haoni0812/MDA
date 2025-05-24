# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import datetime
import logging
import time
from contextlib import contextmanager
from matplotlib.pyplot import axis
import torch.nn.functional as F
import torch
import numpy as np
from sklearn.cluster import DBSCAN
from fastreid.utils.logger import log_every_n_seconds
from models import *
import copy
import random
#test time update
from clustercontrast.utils.faiss_rerank import compute_jaccard_distance
from fastreid.modeling.losses import *
from fastreid.evaluation.infomap_cluster import get_dist_nbr, cluster_by_infomap
import collections
from fastreid.modeling.cm import ClusterMemory
from fastreid.evaluation.trainers import ClusterContrastTrainer
from fastreid.evaluation.trainers_ice import ImageTrainer
from fastreid.evaluation.data import IterLoader
from fastreid.evaluation.data import transforms as T
from fastreid.evaluation.data.sampler import RandomMultipleGallerySampler, RandomMultipleGallerySamplerNoCam, MoreCameraSampler
from fastreid.evaluation.data.preprocessor import Preprocessor, Preprocessor_mutual
from collections import OrderedDict
from collections import defaultdict
from torch.utils.data import DataLoader 
from fastreid.modeling.ops import meta_linear
from fastreid.utils.weight_init import weights_init_kaiming, weights_init_classifier
from fastreid.modeling.losses.center_loss import centerLoss


class DatasetEvaluator:
    """
    Base class for a dataset evaluator.
    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.
    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def preprocess_inputs(self, inputs):
        pass

    def process(self, inputs, outputs):
        """
        Process an input/output pair.
        Args:
            inputs: the inputs that's used to call the model.
            outputs: the return value of `model(input)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.
        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:
                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass



def _parse_data(inputs):
    imgs, pids, camids, path = inputs
    return imgs.cuda(), pids, camids


def flip_tensor_lr(img):
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().cuda()
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def _forward(model, inputs):
    with torch.no_grad():
        feature = model(inputs)
    if isinstance(feature, tuple) or isinstance(feature, list):
        output = []
        for x in feature:
            if isinstance(x, tuple) or isinstance(x, list):
                output.append([item.cpu() for item in x])
            else:
                output.append(x.cpu())
        return output
    else:
        return feature.cpu()

def inference_on_dataset(model, data_loader, evaluator, model_opt=None, cfg=None, pda_model=None, use_adain=True, optimizer_dic=None, pda_scheduler=None, opt=None):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    The model will be used in eval mode.
    Args:
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run. Use
            :class:`DatasetEvaluators([])` if you only want to benchmark, but
            don't want to do any evaluation.
    Returns:
        The return value of `evaluator.evaluate()`
    """
    logger = logging.getLogger(__name__)
    # if opt == None:
    #     logger.info("Start inference on {} images".format(len(data_loader.dataset)))

    total = len(data_loader['dataset'])  # inference data loader must have a fixed length
    model_ema = copy.deepcopy(model)
    evaluator.reset()


    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_compute_time = 0
    training_mode = model.training
    #test time update
    cluster_fine_tune = False
    neg_fine_tune = False
    pda_train = False
    if cluster_fine_tune:

        # Optimizer
        params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
        optimizer = torch.optim.Adam(params, lr=0.00035, weight_decay=5e-4)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

        # Trainer
        num_dataset = len(data_loader['dataset'].img_items)
        trainer = ClusterContrastTrainer(model, num_dataset=num_dataset, cfg=cfg)\
        
        
        with torch.no_grad():
            print("start test-time updating")
            print('==> Create pseudo labels for unlabeled data')
            cluster_loader = data_loader['update']
            dataset = data_loader['dataset']
            _, pids, camids = zip(*sorted(dataset.img_items))
            features, _ = extract_features(model, cluster_loader, print_freq=20)
            features = torch.cat([features[f].unsqueeze(0) for f, _, _ in sorted(dataset.img_items)], 0)
            use_infomap = False
            if use_infomap:
            ############################    infio map    ###################################
                features_array = F.normalize(features, dim=1).cpu().numpy()
                feat_dists, feat_nbrs = get_dist_nbr(features=features_array, k=15, knn_method='faiss-gpu')
                del features_array

                s = time.time()
                pseudo_labels = cluster_by_infomap(feat_nbrs, feat_dists, min_sim=0.6, cluster_num=4)
                pseudo_labels = pseudo_labels.astype(np.intp)

                print('cluster cost time: {}'.format(time.time() - s))
                num_cluster = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)        
            #################################################################################

            ############################    DBSCAN     ######################################
            else:
                rerank_dist = compute_jaccard_distance(features, k1=30, k2=6)
                use_DBSCAN = True
                if use_DBSCAN:
                    # DBSCAN cluster
                    eps = 0.6
                    print('Clustering criterion: eps: {:.3f}'.format(eps))
                    cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)

                # select & cluster images as training set of this epochs
                pseudo_labels = cluster.fit_predict(rerank_dist)
                # ipdb.set_trace() 
                pseudo_labels = generate_pseudo_labels(pseudo_labels, F.normalize(features, dim=1).cuda(), pids, camids)

                num_cluster = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
                print(num_cluster)

            print("epoch: {} \n pseudo_labels: {}".format(1, pseudo_labels.tolist()[:100]))
            ############################DBSCAN###############################################

        # generate new dataset and calculate cluster centers
        @torch.no_grad() 
        def generate_cluster_features(labels, features):
            centers = collections.defaultdict(list)

            for i, label in enumerate(labels):
                if label == -1:
                    continue
                centers[labels[i]].append(features[i])

            centers = [
                torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
            ]

            centers = torch.stack(centers, dim=0)
            return centers

        cluster_features = generate_cluster_features(pseudo_labels, features)

        del cluster_loader , features

        # Create hybrid memory
        memory = ClusterMemory(2048, num_cluster, temp=0.05,
                            momentum=0.2, use_hard=True).cuda(cfg.MODEL.DEVICE)

        memory.features = F.normalize(cluster_features, dim=1).cuda(cfg.MODEL.DEVICE)
        trainer.memory = memory
        pseudo_labeled_dataset = []

        for i, ((fname, pid, cid), label) in enumerate(zip(sorted(dataset.img_items), pseudo_labels)):
            if label != -1:
                pseudo_labeled_dataset.append((fname, label.item(), cid))
        

        print('==> Statistics for epoch {}: {} clusters'.format(1, num_cluster))

        # change pseudo labels

        
        #import pdb; pdb.set_trace()
        train_loader = get_train_loader(cfg, dataset, 256, 128,
                                        128, 4, 4, 200,
                                        trainset=pseudo_labeled_dataset, no_cam=False) ##Note pseudo_labeled_dataset shuold be uesd here   


        for epoch in range(4):        
            train_loader.new_epoch()
            #import pdb; pdb.set_trace()
            trainer.train(epoch, train_loader, optimizer,
                        print_freq=1, train_iters=len(train_loader), opt=model_opt, cfg=cfg)
            lr_scheduler.step()


    if neg_fine_tune:
        model.train()
        pda_optimizer = optimizer_dic["pda"]
        main_optimizer = optimizer_dic["main"]
        num_batch = 0
        
        tta_loader = data_loader["tta"]
        pid_mapping = data_loader["pid_mapping"]
        in_feat = cfg.MODEL.HEADS.IN_FEAT
        # import ipdb; ipdb.set_trace()
        test_numclass =  max(pid_mapping.values())  +  1
        classifier_tta = meta_linear(in_feat, test_numclass, bias=False).cuda(cfg.MODEL.DEVICE)
        classifier_tta = classifier_tta.cuda(cfg.MODEL.DEVICE)
        classifier_tta.apply(weights_init_classifier)
        #import ipdb; ipdb.set_trace()
        LP_FT = True
        if not LP_FT:
            for key, value in classifier_tta.named_parameters():
                main_optimizer.add_param_group({"name": key, "params": [value], "lr": 0.00035, "freeze": False})
        # classfier optimizer
        else:
            params = [{"params": [value]} for _, value in classifier_tta.named_parameters() if value.requires_grad]
            cls_optimizer = torch.optim.Adam(params, lr=0.00035, weight_decay=5e-4)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(cls_optimizer, step_size=1, gamma=0.1)
        alpha = 0.5
        beta_dist = torch.distributions.beta.Beta(alpha, alpha)
        for epoch in range(40): 
            temp_loader = iter(tta_loader)
            for idx, inputs in enumerate(temp_loader):
                # if inputs['images'].size(0) < 2:
                #         import ipdb; ipdb.set_trace()
                #         continue
                mapping_pids = torch.tensor([pid_mapping.get(old_label, old_label) for old_label in inputs["targets"]])
                inputs["targets"] = mapping_pids 
                batch_size = inputs['images'].size(0)
                mixed_st, lambda_st = mix_camera(inputs['images'], beta_dist)
                mix_up = False
                if mix_up ==True:
                    inputs['images'] = torch.cat([mixed_st, inputs['images'][batch_size // 2:]], dim=0)
                outs= model(inputs, model_opt)
                outs['lambda_st'] = lambda_st
                # import ipdb; ipdb.set_trace()
                main_loss = main_forward(outs, cfg, classifier_tta)
                print(main_loss)
                if LP_FT:
                    if epoch < 20:
                        pda_backward(main_loss, cls_optimizer)
                        print("LP training epoch-----", epoch)
                    else:
                        pda_backward(main_loss, main_optimizer)
                        print("FT training epoch-----", epoch)
                else:
                    pda_backward(main_loss, main_optimizer)
                # import pdb; pdb.set_trace()
                num_batch += 1
                print(num_batch, "the number of batch")
    model.eval()
    """ if is_CFS == True:
        return model """
    print("finish test-time updating")



    if use_adain:
        training_mode = model.training
        pda_optimizer = optimizer_dic["pda"]
        main_optimizer = optimizer_dic["main"]
        # collect bn information in a single camera
        network_bns = [x for x in list(model.modules()) if
                        isinstance(x, torch.nn.BatchNorm2d) or isinstance(x, torch.nn.BatchNorm1d)]
        
        # save all BN parameters
        bn_backup = []
        for bn in network_bns:
            bn_backup.append((bn.momentum, bn.running_mean, bn.running_var, bn.num_batches_tracked))
        
        for simple_loader, all_loader in zip(data_loader['simple'], data_loader['all']):
            for bn in network_bns:
                bn.momentum = 1
                bn.running_mean = torch.zeros(bn.running_mean.size()).float().to(cfg.MODEL.DEVICE)
                bn.running_var = torch.ones(bn.running_var.size()).float().to(cfg.MODEL.DEVICE)
                bn.num_batches_tracked = torch.tensor(0).to(cfg.MODEL.DEVICE).long()

            model.train()
            with torch.no_grad():
                for batch_idx, inputs in enumerate(simple_loader):
                    if inputs['images'].size(0) < 2: 
                        continue
                    outs= model(inputs, model_opt)

                    if pda_train:
                        print("start pda training")
                        loss = pda_forward(pda_model, outs['outputs']["bn_features"])
                        pda_backward(loss['loss'], pda_optimizer)
            model.eval()

            for idx, inputs in enumerate(all_loader):
                if idx == num_warmup:
                    start_time = time.perf_counter()
                    total_compute_time = 0
                start_compute_time = time.perf_counter()
                with torch.no_grad():
                    outputs = model(inputs, model_opt)
                    #pda_output = pda_model(outputs)
                    total_compute_time += time.perf_counter() - start_compute_time
                    evaluator.process(inputs, outputs * 0.6) #   +  pda_output[0] * 0.4)
                idx += 1
                iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
                seconds_per_batch = total_compute_time / iters_after_start
                if idx >= num_warmup * 2 or seconds_per_batch > 30:
                    total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                    eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                    log_every_n_seconds(
                        logging.INFO,
                        "Inference done {}/{}. {:.4f} s / batch. ETA={}".format(
                            idx + 1, total, seconds_per_batch, str(eta)
                        ),
                        n=30,
                    )

        # Restoring the state of the model
        model.train(training_mode)
        # Recovery of BN layer parameters
        for idx, bn in enumerate(network_bns):
            bn.momentum = bn_backup[idx][0]
            bn.running_mean = bn_backup[idx][1]
            bn.running_var = bn_backup[idx][2]
            bn.num_batches_tracked = bn_backup[idx][3]

    else:
        with inference_context(model), torch.no_grad():
            for idx, inputs in enumerate(data_loader):
                if idx == num_warmup:
                    start_time = time.perf_counter()
                    total_compute_time = 0

                start_compute_time = time.perf_counter()
                outputs = model(inputs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                total_compute_time += time.perf_counter() - start_compute_time
                evaluator.process(inputs, outputs)

                idx += 1
                iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
                seconds_per_batch = total_compute_time / iters_after_start
                if idx >= num_warmup * 2 or seconds_per_batch > 30:
                    total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                    eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                    log_every_n_seconds(
                        logging.INFO,
                        "Inference done {}/{}. {:.4f} s / batch. ETA={}".format(
                            idx + 1, total, seconds_per_batch, str(eta)
                        ),
                        n=30,
                    )

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    # NOTE this format is parsed by grep

    if opt == None:
        logger.info(
            "Total inference time: {} ({:.6f} s / batch per device)".format(
                total_time_str, total_time / (total - num_warmup)
            )
        )
        logger.info(
            "Total inference pure compute time: {} ({:.6f} s / batch per device)".format(
                total_compute_time_str, total_compute_time / (total - num_warmup)
            )
        )
    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results

def original_inference_on_dataset(model, data_loader, evaluator, opt=None):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    The model will be used in eval mode.
    Args:
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run. Use
            :class:`DatasetEvaluators([])` if you only want to benchmark, but
            don't want to do any evaluation.
    Returns:
        The return value of `evaluator.evaluate()`
    """
    logger = logging.getLogger(__name__)
    # if opt == None:
    #     logger.info("Start inference on {} images".format(len(data_loader.dataset)))

    total = len(data_loader)  # inference data loader must have a fixed length
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_compute_time = 0
    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0

            start_compute_time = time.perf_counter()
            outputs = model(inputs)
            # if torch.cuda.is_available():
            #     torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            evaluator.process(inputs, outputs)

            idx += 1
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_batch = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_batch > 30:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Inference done {}/{}. {:.4f} s / batch. ETA={}".format(
                        idx + 1, total, seconds_per_batch, str(eta)
                    ),
                    n=30,
                )

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    # NOTE this format is parsed by grep

    if opt == None:
        logger.info(
            "Total inference time: {} ({:.6f} s / batch per device)".format(
                total_time_str, total_time / (total - num_warmup)
            )
        )
        logger.info(
     
            "Total inference pure compute time: {} ({:.6f} s / batch per device)".format(
                total_compute_time_str, total_compute_time / (total - num_warmup)
            )
        )
    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results

@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.
    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)

   #####################################################################
    # forward
    #####################################################################
def pda_forward(pda_model=None, meta_results=None):
    #pda forward(first order)
    feas_pda = meta_results.detach()
    feas_pda.requires_grad = True
    results = pda_model(feas_pda, labels = None)
    batchsize = 48
    pda_loss = pda_model.loss_function(*results,
                                M_N = batchsize / 2236,
                                optimizer_idx=None,
                                batch_idx = None)
    return pda_loss
#####################################################################
# main forward
#####################################################################
def main_forward(outs=None, cfg=None, classifier_tta=None):
    # main forward
    # finetune in test st
    loss_names = ["CrossEntropyLoss"]
    loss_dict  = {}
    outputs           = outs["outputs"]
    domain_labels     = outs['domains']
    pooled_features   = outputs['pooled_features']
    bn_features       = outputs['bn_features']
    cls_outputs       = outputs['cls_outputs']
    gt_labels         = outs["targets"]
    if "DomainSCTLoss" in loss_names:
            loss_dict["loss_sct"] = domain_SCT_loss(
                pooled_features if cfg.MODEL.LOSSES.SCT.FEAT_ORDER == 'before' else bn_features,
                domain_labels,
                cfg.MODEL.LOSSES.SCT.NORM,
                cfg.MODEL.LOSSES.SCT.TYPE,
            )
    if "DomainTESTLoss" in loss_names:
            loss_dict["loss_test"] = domain_TEST_loss(
                pooled_features if cfg.MODEL.LOSSES.SCT.FEAT_ORDER == 'before' else bn_features,
                domain_labels,
                cfg.MODEL.LOSSES.SCT.NORM,
                cfg.MODEL.LOSSES.SCT.TYPE,
            )
    if "CrossEntropyLoss" in loss_names:
            # cls_outputs_tta = classifier_tta(F.normalize(bn_features))
            cls_outputs_tta = classifier_tta(bn_features)
            #import ipdb; ipdb.set_trace()
            mix_up = False
            if mix_up == True:
                loss_dict['loss_cls'] = cross_entropy_loss(
                    F.normalize(cls_outputs_tta),
                    gt_labels,
                    cfg.MODEL.LOSSES.CE.EPSILON,
                    cfg.MODEL.LOSSES.CE.ALPHA,
                    test_time = False,
                    lamda = outs['lambda_st']
                )
            else:
                loss_dict['loss_cls'] = cross_entropy_loss(
                    F.normalize(cls_outputs_tta),
                    gt_labels,
                    cfg.MODEL.LOSSES.CE.EPSILON,
                    cfg.MODEL.LOSSES.CE.ALPHA,
                    test_time = False
                )
    if 'CenterLoss' in loss_names:
        loss_dict['loss_center'] = 5e-3 * centerLoss(
            pooled_features,
            gt_labels
        )
    if "TripletLoss" in loss_names:
                loss_dict['loss_triplet'] = triplet_loss(
                pooled_features if cfg.MODEL.LOSSES.TRI.FEAT_ORDER == 'before' else bn_features,
                gt_labels,
                cfg.MODEL.LOSSES.TRI.MARGIN,
                cfg.MODEL.LOSSES.TRI.NORM_FEAT,
                cfg.MODEL.LOSSES.TRI.HARD_MINING,
                cfg.MODEL.LOSSES.TRI.DIST_TYPE,
                cfg.MODEL.LOSSES.TRI.LOSS_TYPE,
                domain_labels,
                cfg.MODEL.LOSSES.TRI.NEW_POS,
                cfg.MODEL.LOSSES.TRI.NEW_NEG,
            )     
    losses = sum(loss_dict.values())
    return losses


#####################################################################
# backward
#####################################################################
def pda_backward(losses, optimizer, retain_graph = False):
    if (losses != None) and (optimizer != None):
        optimizer.zero_grad()
        losses.backward(retain_graph = retain_graph)
        optimizer.step()

def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray

def extract_cnn_feature(model, inputs):
    #inputs = to_torch(inputs['images']).cuda()
    outputs  = model(inputs)
   
    outputs = outputs.data.cpu()
    return outputs

def extract_features(model, data_loader, print_freq=50):
    model.eval()

    features = OrderedDict()
    labels = OrderedDict()

    with torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            pids  = inputs["targets"]
            fnames= inputs["img_path"]
            outputs = extract_cnn_feature(model, inputs)
            for fname, output, pid in zip(fnames, outputs, pids):
                features[fname] = output
                labels[fname] = pid

            if (idx + 1) % print_freq == 0:
                print('Extract Features: [{}/{}]\t'
                      .format(idx + 1, len(data_loader)))
        """ for i, (imgs, fnames, pids, _, _) in enumerate(data_loader):
            outputs = extract_cnn_feature(model, imgs)
            for fname, output, pid in zip(fnames, outputs, pids):
                features[fname] = output
                labels[fname] = pid


            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}/{}]\t'
                      .format(i + 1, len(data_loader))) """

    return features, labels

def get_train_loader(args, dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None, no_cam=False):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        if no_cam:
            sampler = RandomMultipleGallerySamplerNoCam(train_set, num_instances)
        else:
            sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None

    train_loader = IterLoader(
        DataLoader(Preprocessor(train_set, root="./", transform=train_transformer),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)
    #import ipdb; ipdb.set_trace()
    return train_loader

# generate new dataset and calculate cluster centers
def generate_pseudo_labels(cluster_id, inputFeat, pids, camids):
    # cluster_id、pids 和 camids 分别为伪标签、真实标签和摄像头标签的列表
    # 假设 cluster_id 和 pids 分别为伪标签和真实标签的列表
    # 第一步：将cluster_id中除了-1以外的标签替换为pids对应的标签

    avg_num_real_labels, cluster_pid_proportions, avg_proportion, avg_count, avg_camid_diversity, camid_diversities = calculate_label_metrics(cluster_id, pids, camids)

    ori = cluster_id
    non_garbage_indices = np.where(cluster_id != -1)[0]
    cluster_id[non_garbage_indices] = np.array(pids)[non_garbage_indices]
    num_cluster = len(set(cluster_id)) - (1 if -1 in cluster_id else 0)
    print("before mapping:", num_cluster)
    # 第二步：将 cluster_id 中非 -1 的标签映射到 0 到 N-1，N 为非 -1 标签的数量
    unique_labels = np.unique(cluster_id[non_garbage_indices])
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    cluster_id[non_garbage_indices] = np.array([label_mapping[label] for label in cluster_id[non_garbage_indices]])
    
    

    num_cluster = len(set(cluster_id)) - (1 if -1 in cluster_id else 0)
    print("after mapping:", num_cluster)
    ipdb.set_trace()
    return cluster_id


def cal_dist(qFeat, gFeat):
    m, n = qFeat.size(0), gFeat.size(0)
    x = qFeat.view(m, -1)
    y = gFeat.view(n, -1)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
             torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m.addmm_(1, -2, x, y.t())
    return dist_m





def calculate_label_metrics(cluster_id, pids, camids):
    # 剔除标签为 -1 的样本
    valid_indices = np.where(cluster_id != -1)[0]
    valid_cluster_id = cluster_id[valid_indices]
    valid_pids = np.array(pids)[valid_indices]
    valid_camids = np.array(camids)[valid_indices]

    # 创建一个以伪标签为键，真实标签和camid列表为值的字典
    cluster_to_labels = defaultdict(list)
    for c_id, pid, camid in zip(valid_cluster_id, valid_pids, valid_camids):
        cluster_to_labels[c_id].append((pid, camid))
    #ipdb.set_trace()

    # 计算每个伪标签对应的真实标签的数量
    num_real_labels_per_cluster = [len(set(pid[0] for pid in pids)) for pids in cluster_to_labels.values()]

    # 计算平均每个伪标签对应的真实标签数量
    avg_num_real_labels = np.mean(num_real_labels_per_cluster)

    # 统计每个真实标签的总样本量
    total_pids_counts = defaultdict(int)
    for pid in pids:
        total_pids_counts[pid] += 1

    # 计算每个真实标签的样本量占其对应真实标签所有样本的比例和数量
    cluster_pid_proportions = defaultdict(dict)
    for cluster_id, labels_list in cluster_to_labels.items():
        for pid, camid in labels_list:
            proportion = np.sum(valid_pids[valid_cluster_id == cluster_id] == pid) / total_pids_counts[pid]
            cluster_pid_proportions[cluster_id][pid] = (proportion, np.sum(valid_pids[valid_cluster_id == cluster_id] == pid))

    # 计算所有真实标签的平均比例和数量
    avg_proportions = []
    avg_counts = []
    for proportions in cluster_pid_proportions.values():
        for proportion, count in proportions.values():
            avg_proportions.append(proportion)
            avg_counts.append(count)

    avg_proportion = np.mean(avg_proportions)
    avg_count = np.mean(avg_counts)

    # 计算每个伪标签对应的camid的多样性（信息熵）
    camid_diversities = {}
    for cluster_id, labels_list in cluster_to_labels.items():
        camids_per_cluster = [label[1] for label in labels_list]
        camid_counts = defaultdict(int)
        for camid in camids_per_cluster:
            camid_counts[camid] += 1

        total_camids = len(camids_per_cluster)
        entropy = -sum((count / total_camids) * np.log2(count / total_camids) for count in camid_counts.values())
        camid_diversities[cluster_id] = entropy

    # 计算所有伪标签的平均camid多样性
    avg_camid_diversity = np.mean(list(camid_diversities.values()))
    ipdb.set_trace()

    return avg_num_real_labels, cluster_pid_proportions, avg_proportion, avg_count, avg_camid_diversity, camid_diversities


def mix_camera(inputs, beta_dist):
    half_batch_size = inputs.size(0) // 2
    source_input = inputs[:half_batch_size]
    target_input = inputs[half_batch_size:]

    lambd = beta_dist.sample().item()
    mixed_input = lambd * source_input + (1 - lambd) * target_input
    return mixed_input, lambd




