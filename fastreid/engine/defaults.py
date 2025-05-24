# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
This file contains components with some default boilerplate logic user may need
in training / testing. They will not work for everyone, but many users may find them useful.
The behavior of functions/classes in this file is subject to change,
since they are meant to represent the "common default behavior" people need in their projects.
"""

import argparse
import logging
import os
import sys
from collections import OrderedDict
import numpy
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import numpy as np
import random

import copy
from torch.nn.parallel import DistributedDataParallel

from fastreid.data import build_reid_test_loader, build_reid_train_loader, build_my_reid_test_loader
from fastreid.evaluation import (DatasetEvaluator, ReidEvaluator,
                                 inference_on_dataset, inference_on_dataset, print_csv_format)
from fastreid.modeling.meta_arch import build_model
from fastreid.solver import build_lr_scheduler, build_optimizer
from fastreid.utils import comm
from fastreid.utils.env import seed_all_rng
from fastreid.utils.checkpoint import Checkpointer
from fastreid.utils.collect_env import collect_env_info
from fastreid.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter
from fastreid.utils.file_io import PathManager
from fastreid.utils.logger import setup_logger
import torch.optim as optim
from torch.autograd import Variable
from . import hooks
from .train_loop import SimpleTrainer
import pandas as pd
from fastreid.modeling.losses.utils import euclidean_dist, normalize, cosine_dist

from sklearn.manifold import TSNE
import seaborn as sns
# import logging
# logger = logging.getLogger(__name__)

from models import BaseVAE
from models import *

__all__ = ["default_argument_parser", "default_setup", "DefaultPredictor", "DefaultTrainer"]
def default_argument_parser():
    """
    Create a parser with some common arguments used by fastreid users.
    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(description="fastreid Training")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="whether to attempt to resume from the checkpoint directory",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--tsne-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--dist-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--domain-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument("--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)")
    parser.add_argument("--num-pth", type=int, default=0, help="number of pth")

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:{}".format(port))
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser
def default_setup(cfg, args, tmp = ''):
    """
    Perform some basic common setups at the beginning of a job, including:
    1. Set up the detectron2 logger
    2. Log basic information about environment, cmdline arguments, and config
    3. Backup the config to the output directory
    Args:
        cfg (CfgNode): the full config to be used
        args (argparse.NameSpace): the command line arguments to be logged
    """


    output_dir = cfg.OUTPUT_DIR
    if comm.is_main_process() and output_dir:
        PathManager.mkdirs(output_dir)


    rank = comm.get_rank()
    setup_logger(output_dir, distributed_rank=rank, name="fvcore", tmp=tmp)
    logger = setup_logger(output_dir, distributed_rank=rank)

    logger.info("Rank of current process: {}. World size: {}".format(rank, comm.get_world_size()))
    logger.info("Environment info:\n" + collect_env_info())

    logger.info("Command line arguments: " + str(args))
    if hasattr(args, "config_file") and args.config_file != "":
        logger.info(
            "Contents of args.config_file={}:\n{}".format(
                args.config_file, PathManager.open(args.config_file, "r").read()
            )
        )

    logger.info("Running with full config:\n{}".format(cfg))
    if comm.is_main_process() and output_dir:
        # Note: some of our scripts may expect the existence of
        # config.yaml in output directory
        path = os.path.join(output_dir, "config.yaml")
        with PathManager.open(path, "w") as f:
            f.write(cfg.dump())
        logger.info("Full config saved to {}".format(os.path.abspath(path)))

    # make sure each worker has a different, yet deterministic seed if specified

    if not cfg.META.SOLVER.MANUAL_SEED_FLAG:
        seed_all_rng()

    # cudnn benchmark has large overhead. It shouldn't be used considering the small size of
    # typical validation set.
    if not (hasattr(args, "eval_only") and args.eval_only):
        torch.backends.cudnn.benchmark = cfg.CUDNN_BENCHMARK
class DefaultPredictor:
    """
    Create a simple end-to-end predictor with the given config.
    The predictor takes an BGR image, resizes it to the specified resolution,
    runs the model and produces a dict of predictions.
    This predictor takes care of model loading and input preprocessing for you.
    If you'd like to do anything more fancy, please refer to its source code
    as examples to build and use the model manually.
    Attributes:
    Examples:
    .. code-block:: python
        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """

    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.cfg.defrost()
        self.cfg.MODEL.BACKBONE.PRETRAIN = False
        self.model = build_model(self.cfg)
        self.model.eval()
        Checkpointer(self.model).load(cfg.MODEL.WEIGHTS)

    def __call__(self, image):
        """
        Args:
            image (torch.tensor): an image tensor of shape (B, C, H, W).
        Returns:
            predictions (torch.tensor): the output features of the model
        """
        inputs = {"images": image}
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            predictions = self.model(inputs)
            # Normalize feature to compute cosine distance
            features = F.normalize(predictions)
            features = F.normalize(features).cpu().data
            return features
class DefaultTrainer(SimpleTrainer):
    def __init__(self, cfg):
        logger = logging.getLogger("fastreid")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for fastreid
            setup_logger()


        data_loader, data_loader_add, cfg = self.build_train_loader(cfg) # Build datasets
        cfg = self.auto_scale_hyperparams(cfg, data_loader) # auto scale hyperparameters
        model = self.build_model(cfg) # our model initialization
        pda_model = vae_models['PDA'](2048,64,None,device=cfg.MODEL.DEVICE)
        pda_model = pda_model.cuda(cfg.MODEL.DEVICE)
        # optimizer for base model
        optimizer_main = self.build_optimizer(cfg, model,
                                              solver_opt = cfg.SOLVER.OPT,
                                              momentum = cfg.SOLVER.MOMENTUM,
                                              flag = 'main') # params, lr, momentum, ..
        # optimizer for pda model
        optimizer_pda = self.build_optimizer(cfg, pda_model,
                                              solver_opt = cfg.SOLVER.OPT,
                                              momentum = cfg.SOLVER.MOMENTUM,
                                              flag = 'pda') # params, lr, momentum, ..       
        if 'BIN_gate' in cfg.MODEL.NORM.TYPE_BACKBONE: # optimizer for balancing parameter
            optimizer_norm = self.build_optimizer(cfg, model,
                                                  solver_opt=cfg.SOLVER.OPT_NORM,
                                                  momentum=cfg.SOLVER.MOMENTUM_NORM,
                                                  flag = 'norm') # params, lr, momentum, ..
        else:
            optimizer_norm = None

        # torch.cuda.empty_cache()
        meta_param = dict()
        if cfg.META.DATA.NAMES != "": # parameters for meta-learning (refer to fastreid/engine/defaults.py)

            meta_param['synth_data'] = cfg.META.DATA.SYNTH_FLAG
            meta_param['synth_method'] = cfg.META.DATA.SYNTH_METHOD
            meta_param['num_domain'] = cfg.META.DATA.NUM_DOMAINS
            meta_param['whole'] = cfg.META.DATA.WHOLE

            meta_param['meta_compute_layer'] = cfg.META.MODEL.META_COMPUTE_LAYER
            meta_param['meta_update_layer'] = cfg.META.MODEL.META_UPDATE_LAYER
            meta_param['meta_all_params'] = cfg.META.MODEL.ALL_PARAMS

            meta_param['iter_init_inner'] = cfg.META.SOLVER.INIT.INNER_LOOP
            meta_param['iter_init_inner_first'] = cfg.META.SOLVER.INIT.FIRST_INNER_LOOP
            meta_param['iter_init_outer'] = cfg.META.SOLVER.INIT.OUTER_LOOP

            meta_param['update_ratio'] = cfg.META.SOLVER.LR_FACTOR.META
            # meta_param['update_ratio'] = cfg.META.SOLVER.LR_FACTOR.GATE_CYCLIC_RATIO
            # meta_param['update_ratio'] = cfg.META.SOLVER.LR_FACTOR.GATE_CYCLIC_PERIOD_PER_EPOCH
            meta_param['iters_per_epoch'] = cfg.SOLVER.ITERS_PER_EPOCH


            meta_param['iter_mtrain'] = cfg.META.SOLVER.MTRAIN.INNER_LOOP
            meta_param['shuffle_domain'] = cfg.META.SOLVER.MTRAIN.SHUFFLE_DOMAIN
            meta_param['use_second_order'] = cfg.META.SOLVER.MTRAIN.SECOND_ORDER
            meta_param['num_mtrain'] = cfg.META.SOLVER.MTRAIN.NUM_DOMAIN
            meta_param['freeze_gradient_meta'] = cfg.META.SOLVER.MTRAIN.FREEZE_GRAD_META
            meta_param['allow_unused'] = cfg.META.SOLVER.MTRAIN.ALLOW_UNUSED
            meta_param['zero_grad'] = cfg.META.SOLVER.MTRAIN.BEFORE_ZERO_GRAD
            meta_param['type_running_stats_init'] = cfg.META.SOLVER.INIT.TYPE_RUNNING_STATS
            meta_param['type_running_stats_mtrain'] = cfg.META.SOLVER.MTRAIN.TYPE_RUNNING_STATS
            meta_param['type_running_stats_mtest'] = cfg.META.SOLVER.MTEST.TYPE_RUNNING_STATS
            meta_param['auto_grad_outside'] = cfg.META.SOLVER.AUTO_GRAD_OUTSIDE
            meta_param['inner_clamp'] = cfg.META.SOLVER.INNER_CLAMP
            meta_param['synth_grad'] = cfg.META.SOLVER.SYNTH_GRAD
            meta_param['constant_grad'] = cfg.META.SOLVER.CONSTANT_GRAD
            meta_param['random_scale_grad'] = cfg.META.SOLVER.RANDOM_SCALE_GRAD
            meta_param['print_grad'] = cfg.META.SOLVER.PRINT_GRAD
            meta_param['one_loss_for_iter'] = cfg.META.SOLVER.ONE_LOSS_FOR_ITER
            meta_param['one_loss_order'] = cfg.META.SOLVER.ONE_LOSS_ORDER

            if cfg.META.SOLVER.MTEST.ONLY_ONE_DOMAIN:
                meta_param['num_mtest'] = 1
            else:
                meta_param['num_mtest'] = meta_param['num_domain']\
                                                       - meta_param['num_mtrain']

            meta_param['sync'] = cfg.META.SOLVER.SYNC
            meta_param['detail_mode'] = cfg.META.SOLVER.DETAIL_MODE
            meta_param['stop_gradient'] = cfg.META.SOLVER.STOP_GRADIENT
            meta_param['flag_manual_zero_grad'] = cfg.META.SOLVER.MANUAL_ZERO_GRAD
            meta_param['flag_manual_memory_empty'] = cfg.META.SOLVER.MANUAL_MEMORY_EMPTY



            meta_param['main_zero_grad'] = cfg.META.NEW_SOLVER.MAIN_ZERO_GRAD
            meta_param['norm_zero_grad'] = cfg.META.NEW_SOLVER.NORM_ZERO_GRAD
            meta_param['momentum_init_grad'] = cfg.META.NEW_SOLVER.MOMENTUM_INIT_GRAD

            meta_param['write_period_param'] = cfg.SOLVER.WRITE_PERIOD_PARAM


            meta_param['loss_combined'] = cfg.META.LOSS.COMBINED
            meta_param['loss_weight'] = cfg.META.LOSS.WEIGHT
            meta_param['loss_name_mtrain'] = cfg.META.LOSS.MTRAIN_NAME
            meta_param['loss_name_mtest'] = cfg.META.LOSS.MTEST_NAME

            logger.info('-' * 30)
            logger.info('Meta-learning paramters')
            logger.info('-' * 30)
            for name, val in meta_param.items():
                logger.info('[M_param] {}: {}'.format(name, val))
            logger.info('-' * 30)

            meta_param['update_cyclic_ratio'] = cfg.META.SOLVER.LR_FACTOR.META_CYCLIC_RATIO
            meta_param['update_cyclic_period'] = cfg.META.SOLVER.LR_FACTOR.META_CYCLIC_PERIOD_PER_EPOCH
            meta_param['update_cyclic_new'] = cfg.META.SOLVER.LR_FACTOR.META_CYCLIC_NEW

            if meta_param['update_ratio'] == 0.0:
                if meta_param['update_cyclic_new']:
                    meta_param['update_cyclic_up_ratio'] = cfg.META.SOLVER.LR_FACTOR.META_CYCLIC_UP_RATIO
                    meta_param['update_cyclic_middle_lr'] = cfg.META.SOLVER.LR_FACTOR.META_CYCLIC_MIDDLE_LR

                    one_period = int(meta_param['iters_per_epoch'] / meta_param['update_cyclic_period'])
                    num_step_up = int(one_period * meta_param['update_cyclic_up_ratio'])
                    num_step_down = one_period - num_step_up
                    if num_step_up <= 0:
                        num_step_up = 1
                        num_step_down = one_period - 1
                    if num_step_down <= 0:
                        num_step_up = one_period - 1
                        num_step_down = 1

                    self.cyclic_optimizer = optim.SGD([Variable(torch.zeros(1), requires_grad=False)], lr=0.1)
                    self.cyclic_scheduler = torch.optim.lr_scheduler.CyclicLR(
                        optimizer=self.cyclic_optimizer,
                        base_lr= meta_param['update_cyclic_middle_lr'] / meta_param['update_cyclic_ratio'],
                        max_lr= meta_param['update_cyclic_middle_lr'] * meta_param['update_cyclic_ratio'],
                        step_size_up = num_step_up,
                        step_size_down = num_step_down,
                    )



        if comm.get_world_size() > 1:
            # ref to https://github.com/pytorch/pytorch/issues/22049 to set `find_unused_parameters=True`
            # for part of the parameters is not updated.
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

        super().__init__(cfg, model, pda_model, data_loader, data_loader_add, optimizer_main, optimizer_pda, optimizer_norm, meta_param)
        # scheduler for base model
    
        self.scheduler_main = self.build_lr_scheduler(
            optimizer = optimizer_main,
            scheduler_method = cfg.SOLVER.SCHED,
            warmup_factor = cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
            milestones=cfg.SOLVER.STEPS,
            gamma=cfg.SOLVER.GAMMA,
            max_iters=cfg.SOLVER.MAX_ITER,
            delay_iters=cfg.SOLVER.DELAY_ITERS,
            eta_min_lr=cfg.SOLVER.ETA_MIN_LR,
        )

        # scheduler for balancing parameters
        if optimizer_norm is not None:
            if cfg.SOLVER.NORM_SCHEDULER == 'same':
                self.scheduler_norm = self.build_lr_scheduler(
                    optimizer = optimizer_norm,
                    scheduler_method = cfg.SOLVER.SCHED,
                    warmup_factor = cfg.SOLVER.WARMUP_FACTOR,
                    warmup_iters=cfg.SOLVER.WARMUP_ITERS,
                    warmup_method=cfg.SOLVER.WARMUP_METHOD,
                    milestones=cfg.SOLVER.STEPS,
                    gamma=cfg.SOLVER.GAMMA,
                    max_iters=cfg.SOLVER.MAX_ITER,
                    delay_iters=cfg.SOLVER.DELAY_ITERS,
                    eta_min_lr=cfg.SOLVER.ETA_MIN_LR,
                )
            elif cfg.SOLVER.NORM_SCHEDULER == 'no_warm':
                self.scheduler_norm = self.build_lr_scheduler(
                    optimizer = optimizer_norm,
                    scheduler_method = cfg.SOLVER.SCHED,
                    warmup_factor = cfg.SOLVER.WARMUP_FACTOR,
                    warmup_iters= 0,
                    warmup_method=cfg.SOLVER.WARMUP_METHOD,
                    milestones=cfg.SOLVER.STEPS,
                    gamma=cfg.SOLVER.GAMMA,
                    max_iters=cfg.SOLVER.MAX_ITER,
                    delay_iters=cfg.SOLVER.DELAY_ITERS,
                    eta_min_lr=cfg.SOLVER.ETA_MIN_LR,
                )
            elif cfg.SOLVER.NORM_SCHEDULER == 'equal':
                self.scheduler_norm = self.build_lr_scheduler(
                    optimizer = optimizer_norm,
                    scheduler_method = cfg.SOLVER.SCHED,
                    warmup_factor = cfg.SOLVER.WARMUP_FACTOR,
                    warmup_iters= 0,
                    warmup_method=cfg.SOLVER.WARMUP_METHOD,
                    milestones=[100000000,1000000000],
                    gamma=1.0,
                    max_iters=cfg.SOLVER.MAX_ITER,
                    delay_iters=cfg.SOLVER.DELAY_ITERS,
                    eta_min_lr=cfg.SOLVER.ETA_MIN_LR,
                )
            elif cfg.SOLVER.NORM_SCHEDULER == 'cyclic':
                self.scheduler_norm = torch.optim.lr_scheduler.CyclicLR(
                    optimizer = optimizer_norm,
                    base_lr = cfg.SOLVER.CYCLIC_MIN_LR,
                    max_lr = cfg.SOLVER.CYCLIC_MAX_LR,
                    step_size_up = int(cfg.SOLVER.ITERS_PER_EPOCH / cfg.SOLVER.CYCLIC_PERIOD_PER_EPOCH / 2.0),
                    step_size_down = int(cfg.SOLVER.ITERS_PER_EPOCH / cfg.SOLVER.CYCLIC_PERIOD_PER_EPOCH / 2.0),
                )
        else:
            self.scheduler_norm = None

        if optimizer_norm is None:
            self.checkpointer = Checkpointer(
                model,
                cfg.OUTPUT_DIR,
                save_to_disk=comm.is_main_process(),
                optimizer_main=optimizer_main,
                scheduler_main=self.scheduler_main,
            )

        else:
            self.checkpointer = Checkpointer(
                model,
                cfg.OUTPUT_DIR,
                save_to_disk=comm.is_main_process(),
                optimizer_main=optimizer_main,
                scheduler_main=self.scheduler_main,
                optimizer_norm=optimizer_norm,
                scheduler_norm=self.scheduler_norm,
            )

        self.start_iter = 0
        if cfg.SOLVER.SWA.ENABLED:
            self.max_iter = cfg.SOLVER.MAX_ITER + cfg.SOLVER.SWA.ITER
        else:
            self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg
        self.register_hooks(self.build_hooks())



    def resume_or_load(self, resume=True):
        """
        If `resume==True`, and last checkpoint exists, resume from it.
        Otherwise, load a model specified by the config.
        Args:
            resume (bool): whether to do resume or not
        """
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration (or iter zero if there's no checkpoint).
        #assert False
        checkpoint = self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)


        # Reinitialize dataloader iter because when we update dataset person identity dict
        # to resume training, DataLoader won't update this dictionary when using multiprocess
        # because of the function scope.
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration (or iter zero if there's no checkpoint).
    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.
        Returns:
            list[HookBase]:
        """
        logger = logging.getLogger(__name__)
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN
        cfg.DATASETS.NAMES = tuple([cfg.TEST.PRECISE_BN.DATASET])  # set dataset name for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer_main,
                              self.scheduler_main,
                              self.optimizer_norm,
                              self.scheduler_norm),
        ]

        if cfg.SOLVER.SWA.ENABLED:
            ret.append(
                hooks.SWA(
                    cfg.SOLVER.MAX_ITER,
                    cfg.SOLVER.SWA.PERIOD,
                    cfg.SOLVER.SWA.LR_FACTOR,
                    cfg.SOLVER.SWA.ETA_MIN_LR,
                    cfg.SOLVER.SWA.LR_SCHED,
                )
            )

        if cfg.TEST.PRECISE_BN.ENABLED and hooks.get_bn_modules(self.model):
            logger.info("Prepare precise BN dataset")
            ret.append(hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            ))

        if cfg.MODEL.FREEZE_LAYERS != [''] and cfg.SOLVER.FREEZE_ITERS > 0:
            freeze_layers = ",".join(cfg.MODEL.FREEZE_LAYERS)
            logger.info(f'Freeze layer group "{freeze_layers}" training for {cfg.SOLVER.FREEZE_ITERS:d} iterations')
            ret.append(hooks.FreezeLayer(
                self.model,
                self.optimizer_main,
                self.optimizer_norm,
                cfg.MODEL.FREEZE_LAYERS,
                cfg.SOLVER.FREEZE_ITERS,
            ))
        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))
            # ret.append(hooks.PeriodicCheckpointer(self.checkpointer, 1))

        def test_and_save_results():
            if comm.is_main_process():
                self._last_eval_results = self.test(self.cfg, self.model, use_adain=True, pda_model=self.PDA_model, use_pda=True, evaluators=None)
                return self._last_eval_results
            else:
                return None

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), cfg.SOLVER.WRITE_PERIOD))

        return ret

        # IterationTimer: compute processing time each epoch
        # LRScheduler: step LR scheduler and summarize the LR
        # PeriodicCheckpointer: fastreid/utils/checkpoint.py, save checkpoint
        # EvalHook
        # PeriodicWriter: engine/defaults.py -> build_writers, fastreid/uitls/events.py
    def build_writers(self):
        """
        Build a list of writers to be used. By default it contains
        writers that write metrics to the screen,
        a json file, and a tensorboard event file respectively.
        If you'd like a different list of writers, you can overwrite it in
        your trainer.
        Returns:
            list[EventWriter]: a list of :class:`EventWriter` objects.
        It is now implemented by:
        .. code-block:: python
            return [
                CommonMetricPrinter(self.max_iter),
                JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
                TensorboardXWriter(self.cfg.OUTPUT_DIR),
            ]
        """
        # Assume the default print/log frequency.
        return [
            # It may not always print what you want to see, since it prints "common" metrics only.
            CommonMetricPrinter(self.max_iter),
            JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(self.cfg.OUTPUT_DIR),
        ]
    def train(self):
        """
        Run training.
        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        super().train(self.start_iter, self.max_iter)
        return 0
        if comm.is_main_process():
            assert hasattr(
                self, "_last_eval_results"
            ), "No evaluation results obtained during training!"
            # verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:
        It now calls :func:`fastreid.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        model = build_model(cfg)
        # logger = logging.getLogger(__name__)
        # logger.info("Model:\n{}".format(model))
        return model
    @classmethod
    def build_optimizer(cls, cfg, model, solver_opt, momentum, flag = None):
        """
        Returns:
            torch.optim.Optimizer:
        It now calls :func:`fastreid.solver.build_optimizer`.
        Overwrite it if you'd like a different optimizer.
        """
        return build_optimizer(cfg, model, solver_opt, momentum, flag)
    @classmethod
    def build_lr_scheduler(cls, optimizer, scheduler_method, warmup_factor,
                           warmup_iters, warmup_method, milestones,
                           gamma, max_iters, delay_iters, eta_min_lr):
        """
        It now calls :func:`fastreid.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(optimizer,
                                  scheduler_method,
                                  warmup_factor,
                                  warmup_iters,
                                  warmup_method,
                                  milestones,
                                  gamma,
                                  max_iters,
                                  delay_iters,
                                  eta_min_lr)
    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable
        It now calls :func:`fastreid.data.build_detection_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        logger = logging.getLogger(__name__)
        logger.info("Prepare training set")
        return build_reid_train_loader(cfg)
    @classmethod
    def build_test_loader(cls, cfg, dataset_name, opt=None, flag_test=True, use_adain=True):
        """
        Returns:
            iterable
        It now calls :func:`fastreid.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        if use_adain:
            return build_my_reid_test_loader(cfg, dataset_name, opt, flag_test)
        else:
            return build_reid_test_loader(cfg, dataset_name, opt, flag_test)
    @classmethod
    def build_evaluator(cls, cfg, num_query, output_dir=None):
        return ReidEvaluator(cfg, num_query, output_dir)


    @classmethod
    def domain_distance(cls, cfg, model):
        import warnings
        warnings.filterwarnings("ignore")

        def dataname_update(dataset_name):
            if 'VIPER' in dataset_name:  # 316+316 images (316 IDs) -> 316 IDs x 2 images
                dataset_name_local = 'DG_VIPeR'
                sub_name = 'split_{}'.format(dataset_name.split('_')[-1])
                flag_test = True
            elif 'PRID' in dataset_name:  # 100+649 images (649 IDs) -> 100 IDs x 2 images
                dataset_name_local = 'DG_PRID'
                sub_name = int(dataset_name.split('_')[-1])
                flag_test = True
            elif 'GRID' in dataset_name:  # 125+1025 images (900 IDs) -> 125 IDs x 2?3? images
                dataset_name_local = 'DG_GRID'
                sub_name = int(dataset_name.split('_')[-1])
                flag_test = True
            elif 'iLIDS' in dataset_name:  # 60+60 images (60 IDs) -> 60 IDs x 2 images
                dataset_name_local = 'DG_iLIDS'
                sub_name = int(dataset_name.split('_')[-1])
                flag_test = True

            elif 'CUHK02' in dataset_name:  # 7264 images (1816 IDs) -> A IDs x 4 images
                dataset_name_local = 'DG_CUHK02'
                sub_name = None
                flag_test = False
            elif 'CUHK03_detected' in dataset_name:  # 14097 images (1467 IDs) -> A IDs x 9~10 images
                dataset_name_local = 'DG_CUHK03_detected'
                sub_name = None
                flag_test = False
            elif 'Market1501' in dataset_name:  # 29419 images (1501 IDs) -> A IDs x 19~20 images
                dataset_name_local = 'DG_Market1501'
                sub_name = None
                flag_test = False
            elif 'DukeMTMC' in dataset_name:  # 36411 images (1812 IDs) -> A IDs x 20 images
                dataset_name_local = 'DG_DukeMTMC'
                sub_name = None
                flag_test = False
            elif 'CUHK_SYSU' in dataset_name:  # 34574 images (11934 IDs) -> A IDs x 3 images
                dataset_name_local = 'DG_CUHK_SYSU'
                sub_name = None
                flag_test = False
            return dataset_name_local, sub_name, flag_test

        def plot_hist(save_path, pos_dist, neg_dist, width_gap, pos_color, neg_color, metric, dpi, fig_size1, fig_size2, kde = True, bin = 20, both=False):
            fig = plt.figure(figsize=(fig_size1, fig_size2), dpi=dpi)
            ax = fig.add_subplot(111)

            if metric == "cos":
                plt.xlim(0.3, 1.2)
            else:
                plt.xlim(40, 80)
                # plt.xlim(int(min(min(pos_dist), min(neg_dist))/width_gap), int(max(max(pos_dist), max(neg_dist))*width_gap))

            if kde:
                sns.kdeplot(pos_dist, shade=True, ax=ax, color=pos_color, label='Intra-class')
                sns.kdeplot(neg_dist, shade=True, ax=ax, color=neg_color, label='Inter-class')
            else:
                if both:
                    sns.distplot(pos_dist, bins=bin, hist=True, norm_hist=True, kde=True, ax=ax, color=pos_color,
                                 label='Intra-domain')
                    sns.distplot(neg_dist, bins=bin, hist=True, norm_hist=True, kde=True, ax=ax, color=neg_color,
                                 label='Inter-domain')
                else:
                    sns.distplot(pos_dist, bins=bin, hist=True, norm_hist=True, kde=False, ax=ax, color=pos_color,
                                 label='Intra-domain')
                    sns.distplot(neg_dist, bins=bin, hist=True, norm_hist=True, kde=False, ax=ax, color=neg_color,
                                 label='Inter-domain')
            if metric == "cos":
                ax.set_xlabel('Cosine distance')
            else:
                ax.set_xlabel('Euclidean distance')
            ax.set_ylabel('Frequency')
            plt.legend(loc='upper left')
            # plt.show()
            fig.savefig(save_path)
            plt.close('all')


        all_dataset_row = ["GRID", "VIPER", "PRID", "iLIDS"]
        all_dataset_col = ["GRID", "VIPER", "PRID", "iLIDS", "CUHK02", "CUHK03_detected", "Market1501", "DukeMTMC", "CUHK_SYSU"]
        index_col = [1,1,1,1,0,0,0,0,0]

        viper_style = 'a'
        test_set = 1

        for i, name in enumerate(all_dataset_row):
            if ('GRID' in name) or ('PRID' in name) or ('iLIDS' in name):
                all_dataset_row[i] = all_dataset_row[i] + '_' + str(test_set)
            if 'VIPER' in name:
                all_dataset_row[i] = all_dataset_row[i] + '_' + str(test_set) + viper_style

        for i, name in enumerate(all_dataset_col):
            if ('GRID' in name) or ('PRID' in name) or ('iLIDS' in name):
                all_dataset_col[i] = all_dataset_col[i] + '_' + str(test_set)
            if 'VIPER' in name:
                all_dataset_col[i] = all_dataset_col[i] + '_' + str(test_set) + viper_style

        if not os.path.isdir(os.path.join(cfg.OUTPUT_DIR, 'hist_domain')):
            os.mkdir(os.path.join(cfg.OUTPUT_DIR, 'hist_domain'))

        pos_color = 'r'
        neg_color = 'g'
        width_gap = 1.05
        dpi = 200
        fig_size1 = 6
        fig_size2 = 6
        sampling_ratio = 8

        with torch.no_grad():


            model.eval()
            logger = logging.getLogger(__name__)

            for kk in range(2):
                if kk == 0:
                    metric = 'euc'
                else:
                    metric = 'cos'

                same_domain_all = []
                target_domain_all = []
                source_domain_all = []
                for dataset_idx1, dataset_name1 in enumerate(all_dataset_row):
                    same_domain = []
                    target_domain = []
                    source_domain = []
                    logger.info("Prepare testing set")
                    dataset_name_local1, sub_name, flag_test = dataname_update(dataset_name1)
                    row_feat = []
                    logger.info("Subset: {}".format(sub_name))
                    data_loader, num_query = cls.build_test_loader(cfg, dataset_name_local1, opt = sub_name, flag_test = flag_test)
                    total = len(data_loader)
                    first_flag = True
                    for data_loader_idx, inputs in enumerate(data_loader):
                        logger.info("Dataset [{}] is loaded ({}/{})".format(dataset_name_local1, data_loader_idx, total))
                        outputs = model(inputs)
                        local_feat = outputs.cpu()
                        if first_flag:
                            row_feat = copy.deepcopy(local_feat)
                            first_flag = False
                        else:
                            row_feat = torch.cat((row_feat, local_feat), 0)

                    for dataset_idx2, dataset_name2 in enumerate(all_dataset_col):
                        logger.info("Prepare testing set")
                        dataset_name_local2, sub_name, flag_test = dataname_update(dataset_name2)
                        logger.info("Subset: {}".format(sub_name))
                        data_loader, num_query = cls.build_test_loader(cfg, dataset_name_local2, opt=sub_name,
                                                                       flag_test=flag_test)
                        total = len(data_loader)
                        for data_loader_idx, inputs in enumerate(data_loader):
                            logger.info(
                                "Dataset [{}] is loaded ({}/{})".format(dataset_name_local2, data_loader_idx, total))
                            outputs = model(inputs)
                            col_feat = outputs.cpu()

                            if index_col[dataset_idx2] == 0:
                                sampler = [x for x in range(len(col_feat)) if x % sampling_ratio == 0]
                                col_feat = col_feat[sampler]

                            m, n = row_feat.size(0), col_feat.size(0)
                            if metric == "cos":
                                row_feat2 = F.normalize(row_feat, dim=1)
                                col_feat = F.normalize(col_feat, dim=1)
                                dist = 1 - torch.mm(row_feat2, col_feat.t())
                                dist_vec = dist[dist > 1e-04].tolist()
                            else:
                                xx = torch.pow(row_feat, 2).sum(1, keepdim=True).expand(m, n)
                                yy = torch.pow(col_feat, 2).sum(1, keepdim=True).expand(n, m).t()
                                dist = xx + yy
                                dist.addmm_(1, -2, row_feat, col_feat.t())
                                dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
                                dist_vec = dist[dist > 1e-04].tolist()

                            if dataset_name1 == dataset_name2:
                                same_domain.extend(dist_vec)
                                same_domain_all.extend(dist_vec)
                            elif index_col[dataset_idx2] == 1:
                                target_domain.extend(dist_vec)
                                target_domain_all.extend(dist_vec)
                            elif index_col[dataset_idx2] == 0:
                                source_domain.extend(dist_vec)
                                source_domain_all.extend(dist_vec)

                    save_path = os.path.join(cfg.OUTPUT_DIR, 'hist_domain', 'source_{}_{}_{}.png'.format(dataset_name1, metric, str(test_set)))
                    plot_hist(save_path, same_domain, source_domain, width_gap, pos_color, neg_color, metric, dpi, fig_size1, fig_size2)

                    save_path = os.path.join(cfg.OUTPUT_DIR, 'hist_domain', 'target_{}_{}_{}.png'.format(dataset_name1, metric, str(test_set)))
                    plot_hist(save_path, same_domain, target_domain, width_gap, pos_color, neg_color, metric, dpi, fig_size1, fig_size2)

                    same_domain.extend(target_domain)
                    save_path = os.path.join(cfg.OUTPUT_DIR, 'hist_domain', 'source_target_{}_{}_{}.png'.format(dataset_name1, metric, str(test_set)))
                    plot_hist(save_path, same_domain, source_domain, width_gap, pos_color, neg_color, metric, dpi, fig_size1, fig_size2)

                save_path = os.path.join(cfg.OUTPUT_DIR, 'hist_domain', 'all_source_{}_{}.png'.format(metric, str(test_set)))
                plot_hist(save_path, same_domain_all, source_domain_all, width_gap, pos_color, neg_color, metric, dpi, fig_size1, fig_size2)

                save_path = os.path.join(cfg.OUTPUT_DIR, 'hist_domain', 'all_target_{}_{}.png'.format(metric, str(test_set)))
                plot_hist(save_path, same_domain_all, target_domain_all, width_gap, pos_color, neg_color, metric, dpi, fig_size1, fig_size2)

                save_path = os.path.join(cfg.OUTPUT_DIR, 'hist_domain', 'all_source_target_{}_{}.png'.format(metric, str(test_set)))
                same_domain_all.extend(target_domain_all)
                plot_hist(save_path, same_domain_all, source_domain_all, width_gap, pos_color, neg_color, metric, dpi, fig_size1, fig_size2)



    @classmethod
    def test_distance(cls, cfg, model):
        import warnings
        warnings.filterwarnings("ignore")
        def plot_hist(save_path, pos_dist, neg_dist, width_gap, pos_color, neg_color, metric, dpi, fig_size1, fig_size2, kde = True, bin = 20, both=False):
            fig = plt.figure(figsize=(fig_size1, fig_size2), dpi=dpi)
            ax = fig.add_subplot(111)

            if metric == "cos":
                plt.xlim(0, 1.2)
            else:
                plt.xlim(20, 80)
                # plt.xlim(int(min(min(pos_dist), min(neg_dist))/width_gap), int(max(max(pos_dist), max(neg_dist))*width_gap))

            if kde:
                sns.kdeplot(pos_dist, shade=True, ax=ax, color=pos_color, label='Intra-class')
                sns.kdeplot(neg_dist, shade=True, ax=ax, color=neg_color, label='Inter-class')
            else:
                if both:
                    sns.distplot(pos_dist, bins=bin, hist=True, norm_hist=True, kde=True, ax=ax, color=pos_color,
                                 label='Intra-class')
                    sns.distplot(neg_dist, bins=bin, hist=True, norm_hist=True, kde=True, ax=ax, color=neg_color,
                                 label='Inter-class')
                else:
                    sns.distplot(pos_dist, bins=bin, hist=True, norm_hist=True, kde=False, ax=ax, color=pos_color,
                                 label='Intra-class')
                    sns.distplot(neg_dist, bins=bin, hist=True, norm_hist=True, kde=False, ax=ax, color=neg_color,
                                 label='Inter-class')
            if metric == "cos":
                ax.set_xlabel('Cosine distance')
            else:
                ax.set_xlabel('Euclidean distance')
            ax.set_ylabel('Frequency')
            plt.legend(loc='upper left')
            # plt.show()
            fig.savefig(save_path)
            plt.close('all')


        # all_dataset = ["CUHK02", "CUHK03_detected", "Market1501", "DukeMTMC", "CUHK_SYSU", "GRID", "VIPER", "PRID", "iLIDS"]
        all_dataset_ori = ["GRID", "VIPER", "PRID", "iLIDS"]
        viper_style = 'a'
        num_test = 9
        only_all = True

        if not os.path.isdir(os.path.join(cfg.OUTPUT_DIR, 'hist_test')):
            os.mkdir(os.path.join(cfg.OUTPUT_DIR, 'hist_test'))

        with torch.no_grad():

            for idx in range(num_test):
                pos_dist_all_cos = []
                neg_dist_all_cos = []
                pos_dist_all_euc = []
                neg_dist_all_euc = []

                test_set = idx+1

                all_dataset = copy.deepcopy(all_dataset_ori)
                for i, name in enumerate(all_dataset_ori):
                    if ('GRID' in name) or ('PRID' in name) or ('iLIDS' in name):
                        all_dataset[i] = all_dataset[i] + '_' + str(test_set)
                    if 'VIPER' in name:
                        all_dataset[i] = all_dataset[i] + '_' + str(test_set) + viper_style

                model.eval()
                logger = logging.getLogger(__name__)

                for dataset_idx, dataset_name in enumerate(all_dataset):
                    logger.info("Prepare testing set")
                    if 'VIPER' in dataset_name: # 316+316 images (316 IDs) -> 316 IDs x 2 images
                        dataset_name_local = 'DG_VIPeR'
                        sub_name = 'split_{}'.format(dataset_name.split('_')[-1])
                        flag_test = True
                    elif 'PRID' in dataset_name: # 100+649 images (649 IDs) -> 100 IDs x 2 images
                        dataset_name_local = 'DG_PRID'
                        sub_name = int(dataset_name.split('_')[-1])
                        flag_test = True
                    elif 'GRID' in dataset_name: # 125+1025 images (900 IDs) -> 125 IDs x 2?3? images
                        dataset_name_local = 'DG_GRID'
                        sub_name = int(dataset_name.split('_')[-1])
                        flag_test = True
                    elif 'iLIDS' in dataset_name: # 60+60 images (60 IDs) -> 60 IDs x 2 images
                        dataset_name_local = 'DG_iLIDS'
                        sub_name = int(dataset_name.split('_')[-1])
                        flag_test = True

                    elif 'CUHK02' in dataset_name: # 7264 images (1816 IDs) -> A IDs x 4 images
                        dataset_name_local = 'DG_CUHK02'
                        sub_name = None
                        flag_test = False
                    elif 'CUHK03_detected' in dataset_name: # 14097 images (1467 IDs) -> A IDs x 9~10 images
                        dataset_name_local = 'DG_CUHK03_detected'
                        sub_name = None
                        flag_test = False
                    elif 'Market1501' in dataset_name: # 29419 images (1501 IDs) -> A IDs x 19~20 images
                        dataset_name_local = 'DG_Market1501'
                        sub_name = None
                        flag_test = False
                    elif 'DukeMTMC' in dataset_name: # 36411 images (1812 IDs) -> A IDs x 20 images
                        dataset_name_local = 'DG_DukeMTMC'
                        sub_name = None
                        flag_test = False
                    elif 'CUHK_SYSU' in dataset_name: # 34574 images (11934 IDs) -> A IDs x 3 images
                        dataset_name_local = 'DG_CUHK_SYSU'
                        sub_name = None
                        flag_test = False
                    #
                    # if not os.path.isdir(os.path.join(cfg.OUTPUT_DIR, 'hist_' + name)):
                    #     os.mkdir(os.path.join(cfg.OUTPUT_DIR, 'hist_' + name))

                    feat = []
                    ids = []
                    cams = []
                    logger.info("Subset: {}".format(sub_name))
                    data_loader, num_query = cls.build_test_loader(cfg, dataset_name_local, opt = sub_name, flag_test = flag_test)
                    total = len(data_loader)

                    first_flag = True
                    for data_loader_idx, inputs in enumerate(data_loader):

                        logger.info("Dataset [{}] is loaded ({}/{})".format(dataset_name_local, data_loader_idx, total))
                        outputs = model(inputs)
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()

                        # targets = [dataset_name_local + '_' + str(x) for x in inputs["targets"].numpy()]
                        if flag_test:
                            targets = inputs["targets"].numpy().tolist()
                            camids = inputs["camid"].numpy().tolist()
                        else:
                            targets = [int(x.split('_')[-1]) for x in inputs["targets"]]
                            camids = inputs["camid"].numpy().tolist()

                        local_feat = outputs.cpu()
                        local_ids = torch.Tensor(targets)
                        local_cams = torch.Tensor(camids)

                        if first_flag:
                            feat = copy.deepcopy(local_feat)
                            ids = copy.deepcopy(local_ids)
                            cams = copy.deepcopy(local_cams)
                            first_flag = False
                        else:
                            feat = torch.cat((feat, local_feat), 0)
                            ids = torch.cat((ids, local_ids), 0)
                            cams = torch.cat((cams, local_cams), 0)

                    # m = feat.size(0)
                    pos_color = 'r'
                    neg_color = 'g'
                    width_gap = 1.05
                    dpi = 200
                    fig_size1 = 6
                    fig_size2 = 6

                    for kk in range(2):
                        if kk == 0:
                            metric = 'euc'
                        else:
                            metric = 'cos'
                        save_path = os.path.join(cfg.OUTPUT_DIR, 'hist_test',
                                                 '{}_{}.png'.format(dataset_name, metric))

                        query_feat = feat[:num_query]
                        query_pids = numpy.asarray(ids[:num_query])
                        gallery_feat = feat[num_query:]
                        gallery_pids = numpy.asarray(ids[num_query:])

                        m, n = query_feat.size(0), gallery_feat.size(0)
                        if metric == "cos":
                            query_feat = F.normalize(query_feat, dim=1)
                            gallery_feat = F.normalize(gallery_feat, dim=1)
                            dist = 1 - torch.mm(query_feat, gallery_feat.t())
                        else:
                            xx = torch.pow(query_feat, 2).sum(1, keepdim=True).expand(m, n)
                            yy = torch.pow(gallery_feat, 2).sum(1, keepdim=True).expand(n, m).t()
                            dist = xx + yy
                            dist.addmm_(1, -2, query_feat, gallery_feat.t())
                            dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
                        dist = dist.cpu().numpy()

                        pos_dist = []
                        neg_dist = []
                        for j in range(m):
                            q_id = query_pids[j]
                            pos_dist.extend(dist[j][numpy.where(gallery_pids == q_id)].tolist())
                            neg_dist.extend(dist[j][numpy.where(gallery_pids != q_id)].tolist())

                        if metric == "cos":
                            pos_dist_all_cos.extend(pos_dist)
                            neg_dist_all_cos.extend(neg_dist)
                        else:
                            pos_dist_all_euc.extend(pos_dist)
                            neg_dist_all_euc.extend(neg_dist)

                        if not only_all:
                            plot_hist(save_path, pos_dist, neg_dist, width_gap, pos_color, neg_color, metric, dpi, fig_size1, fig_size2)


                save_path = os.path.join(cfg.OUTPUT_DIR, 'hist_test', 'all_cos_{}.png'.format(str(test_set)))
                plot_hist(save_path, pos_dist_all_cos, neg_dist_all_cos, width_gap, pos_color, neg_color, 'cos', dpi, fig_size1, fig_size2)
                save_path = os.path.join(cfg.OUTPUT_DIR, 'hist_test', 'all_cos_h20_{}.png'.format(str(test_set)))
                plot_hist(save_path, pos_dist_all_cos, neg_dist_all_cos, width_gap, pos_color, neg_color, 'cos', dpi, fig_size1, fig_size2, kde=False, bin = 20)
                save_path = os.path.join(cfg.OUTPUT_DIR, 'hist_test', 'all_cos_h30_{}.png'.format(str(test_set)))
                plot_hist(save_path, pos_dist_all_cos, neg_dist_all_cos, width_gap, pos_color, neg_color, 'cos', dpi, fig_size1, fig_size2, kde=False, bin = 30)
                save_path = os.path.join(cfg.OUTPUT_DIR, 'hist_test', 'all_cos_h40_{}.png'.format(str(test_set)))
                plot_hist(save_path, pos_dist_all_cos, neg_dist_all_cos, width_gap, pos_color, neg_color, 'cos', dpi, fig_size1, fig_size2, kde=False, bin = 40)
                save_path = os.path.join(cfg.OUTPUT_DIR, 'hist_test', 'all_cos_both_h20_{}.png'.format(str(test_set)))
                plot_hist(save_path, pos_dist_all_cos, neg_dist_all_cos, width_gap, pos_color, neg_color, 'cos', dpi, fig_size1, fig_size2, kde=False, bin = 20, both = True)
                save_path = os.path.join(cfg.OUTPUT_DIR, 'hist_test', 'all_cos_both_h30_{}.png'.format(str(test_set)))
                plot_hist(save_path, pos_dist_all_cos, neg_dist_all_cos, width_gap, pos_color, neg_color, 'cos', dpi, fig_size1, fig_size2, kde=False, bin = 30, both = True)
                save_path = os.path.join(cfg.OUTPUT_DIR, 'hist_test', 'all_cos_both_h40_{}.png'.format(str(test_set)))
                plot_hist(save_path, pos_dist_all_cos, neg_dist_all_cos, width_gap, pos_color, neg_color, 'cos', dpi, fig_size1, fig_size2, kde=False, bin = 40, both = True)

                save_path = os.path.join(cfg.OUTPUT_DIR, 'hist_test', 'all_euc_{}.png'.format(str(test_set)))
                plot_hist(save_path, pos_dist_all_euc, neg_dist_all_euc, width_gap, pos_color, neg_color, 'euc', dpi, fig_size1, fig_size2)
                save_path = os.path.join(cfg.OUTPUT_DIR, 'hist_test', 'all_euc_h20_{}.png'.format(str(test_set)))
                plot_hist(save_path, pos_dist_all_euc, neg_dist_all_euc, width_gap, pos_color, neg_color, 'euc', dpi, fig_size1, fig_size2, kde=False, bin = 20)
                save_path = os.path.join(cfg.OUTPUT_DIR, 'hist_test', 'all_euc_h30_{}.png'.format(str(test_set)))
                plot_hist(save_path, pos_dist_all_euc, neg_dist_all_euc, width_gap, pos_color, neg_color, 'euc', dpi, fig_size1, fig_size2, kde=False, bin = 30)
                save_path = os.path.join(cfg.OUTPUT_DIR, 'hist_test', 'all_euc_h40_{}.png'.format(str(test_set)))
                plot_hist(save_path, pos_dist_all_euc, neg_dist_all_euc, width_gap, pos_color, neg_color, 'euc', dpi, fig_size1, fig_size2, kde=False, bin = 40)
                save_path = os.path.join(cfg.OUTPUT_DIR, 'hist_test', 'all_euc_both_h20_{}.png'.format(str(test_set)))
                plot_hist(save_path, pos_dist_all_euc, neg_dist_all_euc, width_gap, pos_color, neg_color, 'euc', dpi, fig_size1, fig_size2, kde=False, bin = 20, both = True)
                save_path = os.path.join(cfg.OUTPUT_DIR, 'hist_test', 'all_euc_both_h30_{}.png'.format(str(test_set)))
                plot_hist(save_path, pos_dist_all_euc, neg_dist_all_euc, width_gap, pos_color, neg_color, 'euc', dpi, fig_size1, fig_size2, kde=False, bin = 30, both = True)
                save_path = os.path.join(cfg.OUTPUT_DIR, 'hist_test', 'all_euc_both_h40_{}.png'.format(str(test_set)))
                plot_hist(save_path, pos_dist_all_euc, neg_dist_all_euc, width_gap, pos_color, neg_color, 'euc', dpi, fig_size1, fig_size2, kde=False, bin = 40, both = True)


    @classmethod
    def visualize(cls, cfg, model):
        all_dataset = ["Market1501",]

        flag_id = False
        output_dir_split = cfg.OUTPUT_DIR.split('/')
        fname = 'TSNE_abs'
        flag_abs = True
        if flag_abs:
            num_cases = 1
        else:
            num_cases = 4


        if not os.path.isdir(os.path.join('/'.join(output_dir_split[:-1]), fname)):
            os.mkdir(os.path.join('/'.join(output_dir_split[:-1]), fname))
        cfg.OUTPUT_DIR = os.path.join('/'.join(output_dir_split[:-1]), fname, output_dir_split[-1])
        if not os.path.isdir(cfg.OUTPUT_DIR):
            os.mkdir(cfg.OUTPUT_DIR)


        # all_dataset = ("GRID_1", "VIPER_1a", "PRID_1", "iLIDS_1", ) # VIPER_1a~10d, others_1~10
        # all_dataset = ("CUHK02", "CUHK03_detected", "Market1501", "DukeMTMC", "CUHK_SYSU", )
        raw_dataset = copy.deepcopy(all_dataset)
        for i, x in enumerate(raw_dataset):
            if "CUHK03" in x:
                raw_dataset[i] = x.split('_')[0]
        # all_dataset = ("CUHK02", "CUHK03_detected",)
        test_set = 2
        viper_style = 'a'
        for i, name in enumerate(all_dataset):
            if ('GRID' in name) or ('PRID' in name) or ('iLIDS' in name):
                all_dataset[i] = all_dataset[i] + '_' + str(test_set)
            if 'VIPER' in name:
                all_dataset[i] = all_dataset[i] + '_' + str(test_set) + viper_style

        for idx_case in range(num_cases):

            only_probe_id = True
            # idx_case = 3
            list_perplexity = [10, 20, 30, 40, 50]
            list_iter = [1000, 5000, 20000, 50000] # more scatter
            set_markers = ['.', "1", "h"]

            if flag_abs:
                flag_id = True
                train_num_classes = 1
                train_num_images = 4  # less than
                train_skip_epoch = 0
                test_num_classes = 30
                test_num_images = 1000
                minimum_number_of_images = 3  # more than

                list_type_idx = (
                    [5, 7],
                )
                list_type_name = (
                    "grid-prid",
                )
            else:
                if idx_case == 0:
                    train_num_classes = 30
                    train_num_images = 4  # less than
                    train_skip_epoch = 0
                    test_num_classes = 100
                    test_num_images = 1000
                    minimum_number_of_images = 3  # more than

                    list_type_idx = (
                        [5, 6, 7, 8],
                        [0, 1, 2, 3, 4],
                        [0, 1, 2, 3, 4, 5, 6, 7, 8],
                    )
                    list_type_name = (
                        "test",
                        "train",
                        "all",
                    )
                elif idx_case == 1:
                    train_num_classes = 60
                    train_num_images = 4  # less than
                    train_skip_epoch = 0
                    test_num_classes = 60
                    test_num_images = 1000
                    minimum_number_of_images = 3  # more than

                    list_type_idx = (
                        [5, 6, 7, 8],
                        [0, 1, 2, 3, 4],
                        [0, 1, 2, 3, 4, 5, 6, 7, 8],
                    )
                    list_type_name = (
                        "test",
                        "train",
                        "all",
                    )
                elif idx_case == 2:
                    train_num_classes = 100
                    train_num_images = 4  # less than
                    train_skip_epoch = 0
                    test_num_classes = 40
                    test_num_images = 1000
                    minimum_number_of_images = 3  # more than

                    list_type_idx = (
                        [5, 6, 7, 8],
                        [0, 1, 2, 3, 4],
                        [0, 1, 2, 3, 4, 5, 6, 7, 8],
                    )
                    list_type_name = (
                        "test",
                        "train",
                        "all",
                    )

                elif idx_case == 3:
                    train_num_classes = 1
                    train_num_images = 4  # less than
                    train_skip_epoch = 0
                    test_num_classes = 1000
                    test_num_images = 1000
                    minimum_number_of_images = 3  # more than

                    list_type_idx = (
                        [5],
                        [6],
                        [7],
                        [8],
                    )
                    list_type_name = (
                        "GRID",
                        "VIPER",
                        "PRID",
                        "iLIDS",
                    )


            test_name_domain = 'TSNE_domain[test' + str(test_set) + \
                        ']_'  + str(test_num_classes) + \
                        '_'  + str(test_num_images) + \
                        '_[train'  + str(train_skip_epoch) + \
                        ']_'  + str(train_num_classes) + \
                        '_'  + str(train_num_images)
            if flag_id:
                test_name_id = 'TSNE_id[test' + str(test_set) + \
                            ']_'  + str(test_num_classes) + \
                            '_'  + str(test_num_images) + \
                            '_[train'  + str(train_skip_epoch) + \
                            ']_'  + str(train_num_classes) + \
                            '_'  + str(train_num_images)

            if not os.path.isdir(os.path.join(cfg.OUTPUT_DIR, test_name_domain)):
                os.mkdir(os.path.join(cfg.OUTPUT_DIR, test_name_domain))
            for path in list_type_name:
                if not os.path.isdir(os.path.join(cfg.OUTPUT_DIR, test_name_domain, path)):
                    os.mkdir(os.path.join(cfg.OUTPUT_DIR, test_name_domain, path))
            if flag_id:
                if not os.path.isdir(os.path.join(cfg.OUTPUT_DIR, test_name_id)):
                    os.mkdir(os.path.join(cfg.OUTPUT_DIR, test_name_id))
                for path in list_type_name:
                    if not os.path.isdir(os.path.join(cfg.OUTPUT_DIR, test_name_id, path)):
                        os.mkdir(os.path.join(cfg.OUTPUT_DIR, test_name_id, path))


            num_epoch = 1

            model.eval()
            logger = logging.getLogger(__name__)


            for kkk in range(num_epoch):
                all_feat = []
                all_ids = []
                all_cams = []
                all_domains = []
                all_types = [] # 0:train, 1:query, 2:gallery
                random_seed = kkk
                np.random.seed(random_seed)
                random.seed(random_seed)
                for dataset_idx, dataset_name in enumerate(all_dataset):
                    logger.info("Prepare testing set")
                    if 'VIPER' in dataset_name: # 316+316 images (316 IDs) -> 316 IDs x 2 images
                        dataset_name_local = 'DG_VIPeR'
                        sub_name = 'split_{}'.format(dataset_name.split('_')[-1])
                        flag_test = True
                    elif 'PRID' in dataset_name: # 100+649 images (649 IDs) -> 100 IDs x 2 images
                        dataset_name_local = 'DG_PRID'
                        sub_name = int(dataset_name.split('_')[-1])
                        flag_test = True
                    elif 'GRID' in dataset_name: # 125+1025 images (900 IDs) -> 125 IDs x 2?3? images
                        dataset_name_local = 'DG_GRID'
                        sub_name = int(dataset_name.split('_')[-1])
                        flag_test = True
                    elif 'iLIDS' in dataset_name: # 60+60 images (60 IDs) -> 60 IDs x 2 images
                        dataset_name_local = 'DG_iLIDS'
                        sub_name = int(dataset_name.split('_')[-1])
                        flag_test = True

                    elif 'CUHK02' in dataset_name: # 7264 images (1816 IDs) -> A IDs x 4 images
                        dataset_name_local = 'DG_CUHK02'
                        sub_name = None
                        flag_test = False
                    elif 'CUHK03_detected' in dataset_name: # 14097 images (1467 IDs) -> A IDs x 9~10 images
                        dataset_name_local = 'DG_CUHK03_detected'
                        sub_name = None
                        flag_test = False
                    elif 'Market1501' in dataset_name: # 29419 images (1501 IDs) -> A IDs x 19~20 images
                        dataset_name_local = 'DG_Market1501'
                        sub_name = None
                        flag_test = False
                    elif 'DukeMTMC' in dataset_name: # 36411 images (1812 IDs) -> A IDs x 20 images
                        dataset_name_local = 'DG_DukeMTMC'
                        sub_name = None
                        flag_test = False
                    elif 'CUHK_SYSU' in dataset_name: # 34574 images (11934 IDs) -> A IDs x 3 images
                        dataset_name_local = 'DG_CUHK_SYSU'
                        sub_name = None
                        flag_test = False

                    with torch.no_grad():
                        feat = []
                        ids = []
                        cams = []
                        logger.info("Subset: {}".format(sub_name))
                        data_loader, num_query = cls.build_test_loader(cfg, dataset_name_local, opt = sub_name, flag_test = flag_test)
                        total = len(data_loader)

                        data_loader_cnt = 0
                        first_flag = True
                        for data_loader_idx, inputs in enumerate(data_loader):

                            logger.info("Dataset [{}] is loaded ({}/{})".format(dataset_name_local, data_loader_idx, total))
                            outputs = model(inputs)
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()

                            # targets = [dataset_name_local + '_' + str(x) for x in inputs["targets"].numpy()]
                            if flag_test:
                                targets = inputs["targets"].numpy().tolist()
                                camids = inputs["camid"].numpy().tolist()
                            else:
                                targets = [int(x.split('_')[-1]) for x in inputs["targets"]]
                                camids = inputs["camid"].numpy().tolist()

                            local_feat = outputs.cpu()
                            local_ids = torch.Tensor(targets)
                            local_cams = torch.Tensor(camids)

                            selected_idx = 0
                            if not flag_test:
                                if 'CUHK02' in dataset_name:
                                    min_images = 2
                                else:
                                    min_images = minimum_number_of_images
                                selected_idx = None
                                unique_idx = torch.unique(local_ids)
                                cnt = 0
                                for i in range(len(unique_idx)):
                                    find_idx = (local_ids == unique_idx[i]).nonzero().view(-1)
                                    if len(find_idx) >= min_images:
                                        if cnt == 0:
                                            selected_idx = copy.deepcopy(find_idx)
                                            cnt = cnt + 1
                                        else:
                                            selected_idx = torch.cat((selected_idx, find_idx), 0)
                                if selected_idx is not None:
                                    local_feat = local_feat[selected_idx]
                                    local_ids = local_ids[selected_idx]
                                    local_cams = local_cams[selected_idx]

                            if selected_idx is not None:
                                if first_flag:
                                    feat = copy.deepcopy(local_feat)
                                    ids = copy.deepcopy(local_ids)
                                    cams = copy.deepcopy(local_cams)
                                    first_flag = False
                                else:
                                    feat = torch.cat((feat, local_feat), 0)
                                    ids = torch.cat((ids, local_ids), 0)
                                    cams = torch.cat((cams, local_cams), 0)

                                if not flag_test:
                                    if len(torch.unique(ids)) > train_num_classes:
                                        break
                        if flag_test:
                            num_classes = test_num_classes
                            num_images = test_num_images
                        else:
                            num_classes = train_num_classes
                            num_images = train_num_images

                        domains = copy.deepcopy(cams)
                        domains[:] = dataset_idx
                        types = copy.deepcopy(cams)
                        if flag_test:
                            types[:] = 2 # gallery
                            types[:num_query] = 1 # query
                        else:
                            types[:] = 0 # source dataset


                        if flag_test and only_probe_id:
                            unique_idx = torch.unique(ids[types == 1])
                        else:
                            unique_idx = torch.unique(ids)

                        if 'GRID' in dataset_name:
                            rand_idx = np.random.permutation(len(unique_idx)-1)
                            rand_idx = rand_idx + 1
                            selected_rand_idx = rand_idx[:num_classes]
                            selected_idx = unique_idx[selected_rand_idx]
                        else:
                            rand_idx = np.random.permutation(len(unique_idx))
                            selected_rand_idx = rand_idx[:num_classes]
                            selected_idx = unique_idx[selected_rand_idx]


                        for i in range(len(selected_idx)):
                            if i == 0:
                                change_idx_local = (ids == selected_idx[i]).nonzero().view(-1)
                                if flag_test:
                                    assert len(change_idx_local) <= 4
                                change_idx = copy.deepcopy(change_idx_local[:num_images])
                            else:
                                change_idx_local = (ids == selected_idx[i]).nonzero().view(-1)
                                if flag_test:
                                    assert len(change_idx_local) <= 4
                                change_idx_local = copy.deepcopy(change_idx_local[:num_images])
                                change_idx = torch.cat((change_idx, change_idx_local), 0)
                        # re-arrange


                        feat = feat[change_idx]
                        ids = ids[change_idx]
                        cams = cams[change_idx]
                        types = types[change_idx]
                        domains = domains[change_idx]

                        assert len(change_idx) == len(torch.unique(change_idx))

                        all_feat.append(feat)
                        all_ids.append(ids)
                        all_cams.append(cams)
                        all_domains.append(domains)
                        all_types.append(types)
                all_new_ids = []
                max_num = 0
                for idx in range(len(all_ids)):
                    list_ids = all_ids[idx].int().tolist()
                    uni_ids = torch.unique(all_ids[idx]).int().tolist()
                    pid_dict = dict([(p, i + max_num + 1) for i, p in enumerate(uni_ids)])
                    for j, k in enumerate(list_ids):
                        for name, val in pid_dict.items():
                            if k == name:
                                list_ids[j] = val
                    all_new_ids.extend(list_ids)
                    max_num = max(all_new_ids)
                all_feat = torch.cat(all_feat, dim=0)
                all_ids = torch.as_tensor(all_new_ids)
                # all_ids = torch.cat(all_ids, dim=0).int()
                all_cams = torch.cat(all_cams, dim=0).int()
                all_domains = torch.cat(all_domains, dim=0).int()
                all_types = torch.cat(all_types, dim=0).int()

                all_domains_txt = list()
                for jj, dom_idx in enumerate(all_domains):
                    all_domains_txt.append(raw_dataset[dom_idx])
                type_name = ["train", "query", "gallery"]
                all_types_txt = list()
                for jj, type_idx in enumerate(all_types):
                    all_types_txt.append(type_name[type_idx])

                # continue
                # dist = 1 - torch.mm(query_feat, gallery_feat.t())
                # plt.scatter(y[col == 0, 0], y[col == 0, 1], marker='o')
                # plt.scatter(y[col == 1, 0], y[col == 1, 1], marker='+')

                for i1 in range(2):
                    if i1 == 1:
                        all_feat = F.normalize(all_feat, dim=1)
                    for i2 in range(len(list_type_name)):
                        logger.info('[{}/{}] [{}/{}]'.format(i1+1, 2, i2+1, len(list_type_name)))
                        type_idx = list_type_idx[i2]
                        type_name = list_type_name[i2]

                        cnt = 0
                        for i in type_idx:
                            if cnt == 0:
                                new_idx = (all_domains == i).nonzero().view(-1)
                            else:
                                new_idx = torch.cat((new_idx, (all_domains == i).nonzero().view(-1)), 0)
                            cnt += 1
                        new_types = all_types[new_idx]
                        new_domains = all_domains[new_idx]
                        new_ids = all_ids[new_idx]
                        new_feat = all_feat[new_idx]
                        new_types_txt = [types_txt for types_cnt, types_txt in enumerate(all_types_txt) if types_cnt in new_idx]
                        new_domains_txt = [domains_txt for domains_cnt, domains_txt in enumerate(all_domains_txt) if domains_cnt in new_idx]

                        color_all = sns.color_palette()
                        dict_palette = dict()
                        for cnt_local, name_dataset in enumerate(raw_dataset):
                            if cnt_local <= 4:  # train
                                dict_palette[name_dataset] = color_all[8-cnt_local]
                            else:  # query, gallery
                                dict_palette[name_dataset + '_query'] = color_all[8-cnt_local]
                                dict_palette[name_dataset + '_gallery'] = color_all[8-cnt_local]

                        for cnt_types, val in enumerate(new_types_txt):
                            if val is not 'train':
                                new_domains_txt[cnt_types] = new_domains_txt[cnt_types] + '_' + new_types_txt[cnt_types]

                        for name, val in dict_palette.copy().items():
                            if name not in list(set(new_domains_txt)):
                                del dict_palette[name]

                        all_markers = list()
                        for name_dict, val_dict in dict_palette.items():
                            if 'query' in name_dict:
                                all_markers.append(set_markers[1])
                            elif 'gallery' in name_dict:
                                all_markers.append(set_markers[2])
                            else:
                                all_markers.append(set_markers[0])

                        for i3 in range(len(list_perplexity)):
                            perplexity = list_perplexity[i3]
                            for i4 in range(len(list_iter)):
                                iter = list_iter[i4]

                                plt.rcParams['figure.figsize'] = [10, 10]
                                tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=iter)
                                tsne_ref = tsne.fit_transform(new_feat)
                                df = pd.DataFrame(tsne_ref, index=tsne_ref[0:, 1])
                                # new_types
                                df['x'] = tsne_ref[:, 0]
                                df['y'] = tsne_ref[:, 1]
                                df['Label'] = new_domains_txt

                                sns.lmplot(x="x", y="y", data=df, fit_reg=False, legend=True, height=8, hue='Label',
                                           scatter_kws={"s": 240, "alpha": 0.8}, legend_out=False,
                                           palette=dict_palette, markers=all_markers)

                                if i1 == 1: save_name = 'norm_p' + str(perplexity) + '_i' + str(iter) + '_r' + str(kkk)
                                else: save_name = 'p' + str(perplexity) + '_i' + str(iter) + '_r' + str(kkk)
                                save_name += '.png'
                                save_path = os.path.join(cfg.OUTPUT_DIR, test_name_domain, type_name, save_name)
                                plt.savefig(save_path, dpi=150)

                                if flag_id:
                                    df['Label'] = new_ids
                                    sns.lmplot(x="x", y="y", data=df, fit_reg=False, legend=False, height=8,
                                               hue='Label', scatter_kws={"s": 150, "alpha": 0.7}, markers='o',
                                               palette='tab10')

                                    save_path = os.path.join(cfg.OUTPUT_DIR, test_name_id, type_name, save_name)
                                    plt.savefig(save_path, dpi=150)

                                plt.cla()
                                plt.close()


        return 0
    @classmethod
    def test(cls, cfg, model, use_adain=True, pda_model=None, use_pda=False, evaluators=None):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                `cfg.DATASETS.TEST`.
        Returns:
            dict: a dict of result metrics
        """

        gettrace = getattr(sys, 'gettrace', None)
        if gettrace():
            print('*' * 100)
            print('Hmm, Big Debugger is watching me')
            print('*' * 100)


        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]

        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )
        
        pda_scheduler = None
        pda_optimizer = None
        if use_adain:
            temp_trainer = DefaultTrainer(cfg)
            model_opt = temp_trainer.opt_setting("basic")
            optimizer_main = temp_trainer.build_optimizer(cfg, model,
                                              solver_opt = cfg.SOLVER.OPT,
                                              momentum = cfg.SOLVER.MOMENTUM,
                                              flag = 'main') # params, lr, momentum, ..
            optimizer_dic = {'pda': pda_optimizer, 'main': optimizer_main}
            
        else:
            model_opt = None
        if use_pda:
            from torch.optim import lr_scheduler
            
            pda_optimizer = temp_trainer.build_optimizer(cfg, pda_model,
                                              solver_opt = cfg.SOLVER.OPT,
                                              momentum = cfg.SOLVER.MOMENTUM,
                                              flag = 'pda')
            optimizer_dic["pda"] = pda_optimizer
            pda_scheduler = build_lr_scheduler(
                optimizer = pda_optimizer,
                scheduler_method = cfg.SOLVER.SCHED,
                warmup_factor = cfg.SOLVER.WARMUP_FACTOR,
                warmup_iters=cfg.SOLVER.WARMUP_ITERS,
                warmup_method=cfg.SOLVER.WARMUP_METHOD,
                milestones=cfg.SOLVER.STEPS,
                gamma=cfg.SOLVER.GAMMA,
                max_iters=cfg.SOLVER.MAX_ITER,
                delay_iters=cfg.SOLVER.DELAY_ITERS,
                eta_min_lr=cfg.SOLVER.ETA_MIN_LR,
            )
            
            


        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TESTS):
            logger.info("Prepare testing set")
            if 'ALL' in dataset_name:
                report_all = cfg.TEST.REPORT_ALL
                results_local = OrderedDict()

                if 'VIPER' in dataset_name:
                    dataset_name_local = 'DG_VIPeR'
                    if 'only' in dataset_name:
                        sub_set = 'only_a'
                    else:
                        sub_set = 'all'
                    try:
                        num_test = int(dataset_name.split('_')[-1])
                    except:
                        num_test = 10
                    sub_type = ['a','b','c','d']
                    sub_name = [["split_" + str(i+1) + x for i in range(num_test)] for j, x in enumerate(sub_type)]
                    if sub_set == 'only_a':
                        sub_name = sub_name[0]
                    elif sub_set == 'all':
                        sub_name2 = sub_name
                        sub_name = []
                        for i in range(len(sub_name2)):
                            sub_name.extend(sub_name2[i])
                elif 'PRID' in dataset_name:
                    dataset_name_local = 'DG_PRID'
                    sub_name = [x for x in range(10)]
                elif 'GRID' in dataset_name:
                    dataset_name_local = 'DG_GRID'
                    sub_name = [x for x in range(10)]
                elif 'iLIDS' in dataset_name:
                    dataset_name_local = 'DG_iLIDS'
                    sub_name = [x for x in range(10)]
                for x in sub_name:
                    logger.info("Subset: {}".format(x))
                    data_loader, num_query = cls.build_test_loader(cfg, dataset_name_local, use_adain=use_adain, opt = x)
                    evaluator = cls.build_evaluator(cfg, num_query)
                    # results_i = inference_on_dataset(model, data_loader, evaluator, opt=report_all)
                    results_i = inference_on_dataset(model, data_loader, evaluator, cfg=cfg, model_opt=model_opt, use_adain=use_adain, pda_model=pda_model, optimizer_dic=optimizer_dic, pda_scheduler=pda_scheduler, opt=report_all)
                    if isinstance(x, int):
                        x = str(x)
                    if report_all:
                        results[dataset_name+'_'+x] = results_i
                    results_local[dataset_name+'_'+x] = results_i
                results_local_average = OrderedDict()
                results_local_std = OrderedDict()
                for name_global, val_global in results_local.items():
                    if len(results_local_average) == 0:
                        for name, val in results_local[name_global].items():
                            results_local_average[name] = val
                            results_local_std[name] = val
                    else:
                        for name, val in results_local[name_global].items():
                            results_local_average[name] += val
                            results_local_std[name] = \
                                numpy.hstack([results_local_std[name], val])
                for name, val in results_local_std.items():
                        results_local_std[name] = numpy.std(val)

                for name, val in results_local_average.items():
                    results_local_average[name] /= float(len(results_local))
                results[dataset_name+'_average'] = results_local_average
                results[dataset_name+'_std'] = results_local_std

            else:
                data_loader, num_query = cls.build_test_loader(cfg, dataset_name, use_adain=use_adain)
                # When evaluators are passed in as arguments,
                # implicitly assume that evaluators can be created before data_loader.
                if evaluators is not None:
                    evaluator = evaluators[idx]
                else:
                    try:
                        evaluator = cls.build_evaluator(cfg, num_query)
                    except NotImplementedError:
                        logger.warn(
                            "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                            "or implement its `build_evaluator` method."
                        )
                        results[dataset_name] = {}
                        continue
                
                if use_adain:
                    temp_trainer = DefaultTrainer(cfg)
                    model_opt = temp_trainer.opt_setting("basic")
                else:
                    model_opt = None

                results_i = inference_on_dataset(model, data_loader, evaluator, cfg=cfg, model_opt=model_opt, use_adain=use_adain, pda_model=pda_model, optimizer_dic=optimizer_dic, pda_scheduler=pda_scheduler)


                results[dataset_name] = results_i

        results_all_average = OrderedDict()
        cnt_average = 0
        for name_global, val_global in results.items():
            if 'average' in name_global:
                cnt_average += 1
                if len(results_all_average) == 0:
                    for name, val in results[name_global].items():
                        results_all_average[name] = val
                else:
                    for name, val in results[name_global].items():
                        results_all_average[name] += val

        for name, val in results_all_average.items():
            results_all_average[name] /= float(cnt_average)

        results['** all_average **'] = results_all_average


        if comm.is_main_process():
            assert isinstance(
                results, dict
            ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                results
            )
            print_csv_format(results)

        if len(results) == 1: results = list(results.values())[0]

        return results
    
    @classmethod
    def my_test(cls, cfg, model, pda_model=None, use_pda=False, use_adain=True, optimizer_pda=None, evaluators=None, num_bn_sample=200):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                `cfg.DATASETS.TEST`.
        Returns:
            dict: a dict of result metrics
        """

        gettrace = getattr(sys, 'gettrace', None)
        if gettrace():
            print('*' * 100)
            print('Hmm, Big Debugger is watching me')
            print('*' * 100)


        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]

        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TESTS):
            logger.info("Prepare testing set")
            if 'ALL' in dataset_name:
                report_all = cfg.TEST.REPORT_ALL
                results_local = OrderedDict()

                if 'VIPER' in dataset_name:
                    dataset_name_local = 'DG_VIPeR'
                    if 'only' in dataset_name:
                        sub_set = 'only_a'
                    else:
                        sub_set = 'all'
                    try:
                        num_test = int(dataset_name.split('_')[-1])
                    except:
                        num_test = 10
                    sub_type = ['a','b','c','d']
                    sub_name = [["split_" + str(i+1) + x for i in range(num_test)] for j, x in enumerate(sub_type)]
                    if sub_set == 'only_a':
                        sub_name = sub_name[0]
                    elif sub_set == 'all':
                        sub_name2 = sub_name
                        sub_name = []
                        for i in range(len(sub_name2)):
                            sub_name.extend(sub_name2[i])
                elif 'PRID' in dataset_name:
                    dataset_name_local = 'DG_PRID'
                    sub_name = [x for x in range(10)]
                elif 'GRID' in dataset_name:
                    dataset_name_local = 'DG_GRID'
                    sub_name = [x for x in range(10)]
                elif 'iLIDS' in dataset_name:
                    dataset_name_local = 'DG_iLIDS'
                    sub_name = [x for x in range(10)]


                for x in sub_name:
                    logger.info("Subset: {}".format(x))
                    data_loader, num_query = cls.build_test_loader(cfg, dataset_name_local, opt = x)
                    evaluator = cls.build_evaluator(cfg, num_query, output_dir=dataset_name)
                    results_i = inference_on_dataset(model, data_loader, evaluator, opt=report_all)
                    if isinstance(x, int):
                        x = str(x)
                    if report_all:
                        results[dataset_name+'_'+x] = results_i
                    results_local[dataset_name+'_'+x] = results_i
                results_local_average = OrderedDict()
                results_local_std = OrderedDict()
                for name_global, val_global in results_local.items():
                    if len(results_local_average) == 0:
                        for name, val in results_local[name_global].items():
                            results_local_average[name] = val
                            results_local_std[name] = val
                    else:
                        for name, val in results_local[name_global].items():
                            results_local_average[name] += val
                            results_local_std[name] = \
                                numpy.hstack([results_local_std[name], val])
                for name, val in results_local_std.items():
                        results_local_std[name] = numpy.std(val)

                for name, val in results_local_average.items():
                    results_local_average[name] /= float(len(results_local))
                results[dataset_name+'_average'] = results_local_average
                results[dataset_name+'_std'] = results_local_std

            else:
                data_loader, num_query = cls.build_test_loader(cfg, dataset_name, use_adain=use_adain)

                # When evaluators are passed in as arguments,
                # implicitly assume that evaluators can be created before data_loader.
                if evaluators is not None:
                    evaluator = evaluators[idx]
                else:
                    try:
                        evaluator = cls.build_evaluator(cfg, num_query)
                        pda_evaluator = cls.build_evaluator(cfg, num_query)
                    except NotImplementedError:
                        logger.warn(
                            "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                            "or implement its `build_evaluator` method."
                        )
                        results[dataset_name] = {}
                        continue
                temp_trainer = DefaultTrainer(cfg)
                opt = temp_trainer.opt_setting("basic")

                scheduler_pda = None
                if use_pda and optimizer_pda is None and not model.training:
                    from torch.optim import lr_scheduler
                    scheduler_pda = build_lr_scheduler(
                        optimizer = optimizer_pda,
                        scheduler_method = cfg.SOLVER.SCHED,
                        warmup_factor = cfg.SOLVER.WARMUP_FACTOR,
                        warmup_iters=cfg.SOLVER.WARMUP_ITERS,
                        warmup_method=cfg.SOLVER.WARMUP_METHOD,
                        milestones=cfg.SOLVER.STEPS,
                        gamma=cfg.SOLVER.GAMMA,
                        max_iters=cfg.SOLVER.MAX_ITER,
                        delay_iters=cfg.SOLVER.DELAY_ITERS,
                        eta_min_lr=cfg.SOLVER.ETA_MIN_LR,
                    )
                    optimizer_pda = temp_trainer.build_optimizer(cfg, pda_model,
                                              solver_opt = cfg.SOLVER.OPT,
                                              momentum = cfg.SOLVER.MOMENTUM,
                                              flag = 'pda')
                
                if use_adain:
                    results_i, results_i_pda = my_inference_on_dataset(model, cfg, data_loader, evaluator, 
                        pda_model, pda_evaluator, use_pda, optimizer_pda, scheduler_pda, opt)
                    results[dataset_name] = results_i
                    results[dataset_name+"_pda"] = results_i_pda
                else:
                    results_i = inference_on_dataset(model, data_loader, evaluator)
                    results[dataset_name] = results_i


        results_all_average = OrderedDict()
        cnt_average = 0
        for name_global, val_global in results.items():
            if 'average' in name_global:
                cnt_average += 1
                if len(results_all_average) == 0:
                    for name, val in results[name_global].items():
                        results_all_average[name] = val
                else:
                    for name, val in results[name_global].items():
                        results_all_average[name] += val

        for name, val in results_all_average.items():
            results_all_average[name] /= float(cnt_average)

        results['** all_average **'] = results_all_average


        if comm.is_main_process():
            assert isinstance(
                results, dict
            ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                results
            )
            print_csv_format(results)

        if len(results) == 1: results = list(results.values())[0]

        return results
    @staticmethod
    def auto_scale_hyperparams(cfg, data_loader):
        r"""
        This is used for auto-computation actual training iterations,
        because some hyper-param, such as MAX_ITER, means training epochs rather than iters,
        so we need to convert specific hyper-param to training iterations.
        """

        cfg = cfg.clone()
        frozen = cfg.is_frozen()
        cfg.defrost()

        if isinstance(data_loader, list):
            num_images = max([x.batch_sampler.sampler.total_images for x in data_loader])
            num_classes = len(data_loader[0].dataset.pid_dict)
        else:
            num_images = data_loader.batch_sampler.sampler.total_images
            num_classes = data_loader.dataset.num_classes

        if cfg.META.DATA.NAMES != "": # meta-learning
            if cfg.META.SOLVER.INIT.INNER_LOOP == 0:
                iters_per_epoch = num_images // cfg.SOLVER.IMS_PER_BATCH
            else:
                iters_per_epoch = num_images // (cfg.SOLVER.IMS_PER_BATCH * cfg.META.SOLVER.INIT.INNER_LOOP)
        else:
            iters_per_epoch = num_images // cfg.SOLVER.IMS_PER_BATCH
        cfg.SOLVER.ITERS_PER_EPOCH = iters_per_epoch
        cfg.MODEL.HEADS.NUM_CLASSES = num_classes
        cfg.SOLVER.MAX_ITER *= iters_per_epoch
        cfg.SOLVER.WARMUP_ITERS *= iters_per_epoch
        cfg.SOLVER.FREEZE_ITERS *= iters_per_epoch
        cfg.SOLVER.DELAY_ITERS *= iters_per_epoch
        for i in range(len(cfg.SOLVER.STEPS)):
            cfg.SOLVER.STEPS[i] *= iters_per_epoch
        cfg.SOLVER.SWA.ITER *= iters_per_epoch
        cfg.SOLVER.SWA.PERIOD *= iters_per_epoch
        cfg.SOLVER.CHECKPOINT_PERIOD *= iters_per_epoch


        # Evaluation period must be divided by 200 for writing into tensorboard.
        num_mod = (cfg.SOLVER.WRITE_PERIOD - cfg.TEST.EVAL_PERIOD * iters_per_epoch) % cfg.SOLVER.WRITE_PERIOD
        cfg.TEST.EVAL_PERIOD = cfg.TEST.EVAL_PERIOD * iters_per_epoch + num_mod
        if cfg.SOLVER.CHECKPOINT_SAME_AS_EVAL:
            cfg.SOLVER.CHECKPOINT_PERIOD = cfg.TEST.EVAL_PERIOD
        # cfg.TEST.EVAL_PERIOD = 1

        logger = logging.getLogger(__name__)
        logger.info(
            f"Auto-scaling the config to num_classes={cfg.MODEL.HEADS.NUM_CLASSES}, "
            f"max_Iter={cfg.SOLVER.MAX_ITER}, wamrup_Iter={cfg.SOLVER.WARMUP_ITERS}, "
            f"freeze_Iter={cfg.SOLVER.FREEZE_ITERS}, delay_Iter={cfg.SOLVER.DELAY_ITERS}, "
            f"step_Iter={cfg.SOLVER.STEPS}, ckpt_Iter={cfg.SOLVER.CHECKPOINT_PERIOD}, "
            f"eval_Iter={cfg.TEST.EVAL_PERIOD}."
        )

        if frozen: cfg.freeze()

        return cfg
