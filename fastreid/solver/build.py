# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from . import lr_scheduler
from . import optim


def build_optimizer(cfg, model, solver_opt, momentum, flag = None):
    params = []

    for key, value in model.named_parameters():
        if isinstance(value, list):
            print('.')
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "backbone" in key:
            lr *= cfg.SOLVER.BACKBONE_LR_FACTOR
        if "heads" in key:
            lr *= cfg.SOLVER.HEADS_LR_FACTOR
        if "bias" in key:
            lr *= cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        if "gate" in key:
            print(key, value.shape)
            lr *= cfg.META.SOLVER.LR_FACTOR.GATE
        if "PDA" in key:
            lr *= cfg.SOLVER.HEADS_LR_FACTOR

        if (flag == 'main') and ("gate" not in key):
            params += [{"name": key, "params": [value], "lr": lr, "weight_decay": weight_decay, "freeze": False}]
        elif (flag == 'norm') and ("gate" in key):
            params += [{"name": key, "params": [value], "lr": lr, "weight_decay": cfg.SOLVER.WEIGHT_DECAY_NORM, "freeze": False}]
        elif (flag == 'pda') :
            params += [{"name": key, "params": [value], "lr": lr, "weight_decay": weight_decay, "freeze": False}]
            

        # params += [{"name": key, "params": [value], "lr": lr, "weight_decay": weight_decay, "freeze": False}]

    if hasattr(optim, solver_opt):
        if solver_opt == "SGD":
            opt_fns = getattr(optim, solver_opt)(params, momentum=momentum)
        else:
            opt_fns = getattr(optim, solver_opt)(params)
    else:
        raise NameError("optimizer {} not support".format(solver_opt))
    return opt_fns


def build_lr_scheduler(optimizer,
                       scheduler_method,
                       warmup_factor,
                       warmup_iters,
                       warmup_method,
                       milestones,
                       gamma,
                       max_iters,
                       delay_iters,
                       eta_min_lr):
    scheduler_args = {
        "optimizer": optimizer,

        # warmup options
        "warmup_factor": warmup_factor,
        "warmup_iters": warmup_iters,
        "warmup_method": warmup_method,

        # multi-step lr scheduler options
        "milestones": milestones,
        "gamma": gamma,

        # cosine annealing lr scheduler options
        "max_iters": max_iters,
        "delay_iters": delay_iters,
        "eta_min_lr": eta_min_lr,

    }
    return getattr(lr_scheduler, scheduler_method)(**scheduler_args)
