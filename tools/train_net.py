#!/usr/bin/env python
# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import logging
import os
import sys
import re

sys.path.append('.')
from models import PDA, Model_AE
from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from fastreid.utils.checkpoint import Checkpointer
from fastreid.engine import hooks
from fastreid.evaluation import ReidEvaluator
from fastreid.utils.file_io import PathManager
import torch
import numpy as np
import random
import glob

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, num_query, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return ReidEvaluator(cfg, num_query)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # automatic OUTPUT dir
    cfg.merge_from_file(args.config_file)
    config_file_name = args.config_file.split('/')
    for i, x in enumerate(config_file_name):
        if x == 'configs':

            config_file_name[i] = 'logs'
        if '.yml' in x:
            config_file_name[i] = config_file_name[i][:-4]
    cfg.OUTPUT_DIR = '/'.join(config_file_name)
    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, 'nometa_and_mixstyle')


    cfg.merge_from_list(args.opts)
    cfg.freeze()
    if args.eval_only or args.dist_only or args.tsne_only or args.domain_only:
        if args.eval_only:
            tmp = 'eval'
        if args.dist_only:
            tmp = 'dist'
        if args.tsne_only:
            tmp = 'tsne'
        if args.domain_only:
            tmp = 'domain'
        default_setup(cfg, args, tmp=tmp)
    else:
        default_setup(cfg, args)
    
    return cfg

def main(args):
    cfg = setup(args)
    logger = logging.getLogger("fastreid.trainer")
    
    if cfg.META.SOLVER.MANUAL_SEED_FLAG:
        random_seed = cfg.META.SOLVER.MANUAL_SEED_NUMBER
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
        if cfg.META.SOLVER.MANUAL_SEED_DETERMINISTIC:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        np.random.seed(random_seed)
        random.seed(random_seed)
        logger.info("Using a generated random seed {}".format(cfg.META.SOLVER.MANUAL_SEED_NUMBER))

    
    # cfg.MODEL.WEIGHTS = "./logs/Visualize/u01/model_final.pth"
    # Trainer.resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
    if args.eval_only or args.dist_only or args.tsne_only or args.domain_only:
        cfg.defrost()
        cfg.MODEL.BACKBONE.PRETRAIN = False
        if args.eval_only:
            num_class = torch.load(cfg.MODEL.WEIGHTS, map_location=torch.device("cpu"))['model']['heads.classifier_fc.weight'].size(0)
            cfg.MODEL.HEADS.NUM_CLASSES = num_class
        model = Trainer.build_model(cfg)
        
        res = []

        if cfg.MODEL.WEIGHTS is not "":
            Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model

        if args.resume:
            save_file = os.path.join(cfg.OUTPUT_DIR, "last_checkpoint")
            with PathManager.open(save_file, "r") as f:
                last_saved = f.read().strip()
            path = os.path.join(cfg.OUTPUT_DIR, last_saved)
            Checkpointer(model).load(path)
            logger.info("load: {}".format(path))

        if args.num_pth > 0:
            list_pth = glob.glob(os.path.join(cfg.OUTPUT_DIR, '*.pth'))
            list_pth = sorted(list_pth)
            Checkpointer(model).load(list_pth[args.num_pth-1])
            logger.info("load pth number: {}".format(args.num_pth-1))
            logger.info("load: {}".format(list_pth[args.num_pth-1]))


        if args.eval_only:
            pda_model = PDA(2048,128,None,device=cfg.MODEL.DEVICE)
            pda_model = pda_model.cuda(cfg.MODEL.DEVICE)
            # Trainer.test(cfg, model.eval(), vae_model, use_vae=False, use_adain=True)
            Trainer.test(cfg, model.eval(), use_adain=True, pda_model=pda_model)

        if args.tsne_only:
            cfg.TEST.IMS_PER_BATCH = 256
            # res = Trainer.visualize(cfg, model)
            from fastreid.utils.tsne import TsneViewer
            viewer = TsneViewer(model, cfg, use_adaIN=True)
            viewer.process()
            

        if args.dist_only:
            cfg.TEST.IMS_PER_BATCH = 256
            res = Trainer.test_distance(cfg, model)

        if args.domain_only:
            cfg.TEST.IMS_PER_BATCH = 256
            res = Trainer.domain_distance(cfg, model)

        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)


    trainer.train() # train_loop.py -> train
    # save the trained vae model
    torch.save(trainer.PDA_model.cpu(), os.path.join(cfg.OUTPUT_DIR, "vae_model.pth"))

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

