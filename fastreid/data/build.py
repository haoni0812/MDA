# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

import os
import torch
import sys
#from torch._six import container_abcs, string_classes, int_classes
import collections.abc as container_abcs
int_classes = int
string_classes = str
from torch.utils.data import DataLoader
from fastreid.utils import comm

from . import samplers
from .common import CommDataset
from .datasets import DATASET_REGISTRY
from .transforms import build_transforms
import logging
logger = logging.getLogger(__name__)

_root = os.getenv("FASTREID_DATASETS", "datasets")


def build_reid_train_loader(cfg):
    # print( cfg.DATALOADER.CAMERA_TO_DOMAIN )
    # assert False
    # build datasets
    cfg = cfg.clone()
    frozen = cfg.is_frozen()
    cfg.defrost()

    individual_flag_ori = cfg.DATALOADER.INDIVIDUAL
    individual_flag_meta = cfg.META.DATA.INDIVIDUAL
    if cfg.META.DATA.NAMES == "":
        individual_flag_meta = False
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace():
        print('*'*100)
        print('Hmm, Big Debugger is watching me')
        print('*'*100)
        num_workers = 0
    else:
        num_workers = cfg.DATALOADER.NUM_WORKERS

    # transforms
    train_transforms = build_transforms(cfg, is_train=True, is_fake=False)
    if (cfg.META.DATA.NAMES != "") and \
            (cfg.META.DATA.LOADER_FLAG == 'synth' or cfg.META.DATA.SYNTH_FLAG is not 'none'):
        synth_transforms = build_transforms(cfg, is_train=True, is_fake=True)
        cfg.META.DATA.LOADER_FLAG = 'each'
    else:
        synth_transforms = None
    train_set_all = []
    train_items = list()
    domain_idx = 0
    camera_all = list()

    # load datasets
    for d in cfg.DATASETS.NAMES:
        dataset = DATASET_REGISTRY.get(d)(root=_root, combineall=cfg.DATASETS.COMBINEALL)
        if comm.is_main_process():
            dataset.show_train()
        if len(dataset.train[0]) < 4:
            for i, x in enumerate(dataset.train):
                add_info = {}  # dictionary

                if cfg.DATALOADER.CAMERA_TO_DOMAIN:
                    add_info['domains'] = dataset.train[i][2]
                    camera_all.append(dataset.train[i][2])
                else:
                    add_info['domains'] = int(domain_idx)
                dataset.train[i] = list(dataset.train[i])
                dataset.train[i].append(add_info)
                dataset.train[i] = tuple(dataset.train[i])
        domain_idx += 1
        train_items.extend(dataset.train)
        if individual_flag_ori or individual_flag_meta: # individual set
            train_set_all.append(dataset.train)

    if cfg.DATALOADER.CAMERA_TO_DOMAIN: # used for single-source DG
        num_domains = len(set(camera_all))
    else:
        num_domains = domain_idx
    cfg.META.DATA.NUM_DOMAINS = num_domains

    if cfg.DATALOADER.NAIVE_WAY:
        logger.info('**[dataloader info: random domain shuffle]**')
    else:
        logger.info('**[dataloader info: uniform domain]**')
        logger.info('**[The batch size should be a multiple of the number of domains.]**')
        assert (cfg.SOLVER.IMS_PER_BATCH % (num_domains*cfg.DATALOADER.NUM_INSTANCE) == 0), \
            "cfg.SOLVER.IMS_PER_BATCH should be a multiple of (num_domain x num_instance)"
        assert (cfg.META.DATA.MTRAIN_MINI_BATCH % (num_domains*cfg.META.DATA.MTRAIN_NUM_INSTANCE) == 0), \
            "cfg.META.DATA.MTRAIN_MINI_BATCH should be a multiple of (num_domain x num_instance)"
        assert (cfg.META.DATA.MTEST_MINI_BATCH % (num_domains*cfg.META.DATA.MTEST_NUM_INSTANCE) == 0), \
            "cfg.META.DATA.MTEST_MINI_BATCH should be a multiple of (num_domain x num_instance)"

    if individual_flag_ori:
        cfg.SOLVER.IMS_PER_BATCH //= num_domains
    if individual_flag_meta:
        cfg.META.DATA.MTRAIN_MINI_BATCH //= num_domains
        cfg.META.DATA.MTEST_MINI_BATCH //= num_domains


    if 'keypoint' in cfg.META.DATA.NAMES: # used for keypoint (not used in MetaBIN)
        cfg, train_set_all = make_keypoint_data(cfg = cfg,
                                                data_name = cfg.META.DATA.NAMES,
                                                train_items = train_items)

    train_set = CommDataset(train_items, train_transforms, relabel=True)
    if (synth_transforms is not None) and (cfg.META.DATA.NAMES != ""): # used for synthetic (not used in MetaBIN)
        synth_set = CommDataset(train_items, synth_transforms, relabel=True)



    if individual_flag_ori or individual_flag_meta: # for individual dataloader
        relabel_flag = False
        if individual_flag_meta:
            relabel_flag = cfg.META.DATA.RELABEL

        for i, x in enumerate(train_set_all):
            train_set_all[i] = CommDataset(x, train_transforms, relabel=relabel_flag)
            if not relabel_flag:
                train_set_all[i].relabel = True
                train_set_all[i].pid_dict = train_set.pid_dict
        # Check number of data
        cnt_data = 0
        for x in train_set_all:
            cnt_data += len(x.img_items)
        if cnt_data != len(train_set.img_items):
            print("data loading error, check build.py")

    if individual_flag_ori: # for individual dataloader (domain-wise)
        train_loader = []
        if len(train_set_all) > 0:
            for i, x in enumerate(train_set_all):
                train_loader.append(make_sampler(
                    train_set=x,
                    num_batch=cfg.SOLVER.IMS_PER_BATCH,
                    num_instance=cfg.DATALOADER.NUM_INSTANCE,
                    num_workers=num_workers,
                    mini_batch_size=cfg.SOLVER.IMS_PER_BATCH // comm.get_world_size(),
                    drop_last=cfg.DATALOADER.DROP_LAST,
                    flag1=cfg.DATALOADER.NAIVE_WAY,
                    flag2=cfg.DATALOADER.DELETE_REM,
                    cfg = cfg))
    else:
        train_loader = make_sampler(
            train_set=train_set,
            num_batch=cfg.SOLVER.IMS_PER_BATCH,
            num_instance=cfg.DATALOADER.NUM_INSTANCE,
            num_workers=num_workers,
            mini_batch_size=cfg.SOLVER.IMS_PER_BATCH // comm.get_world_size(),
            drop_last=cfg.DATALOADER.DROP_LAST,
            flag1=cfg.DATALOADER.NAIVE_WAY,
            flag2=cfg.DATALOADER.DELETE_REM,
            cfg = cfg)

    train_loader_add = {}
    train_loader_add['mtrain'] = None # mtrain dataset
    train_loader_add['mtest'] = None # mtest dataset
    if cfg.META.DATA.NAMES != "":
        if cfg.META.DATA.LOADER_FLAG == 'each': # "each": meta-init / meta-train / meta-test
            make_mtrain = True
            make_mtest = True
        elif cfg.META.DATA.LOADER_FLAG == 'diff': # "diff": meta-init / meta-final
            make_mtrain = True
            make_mtest = False
        elif cfg.META.DATA.LOADER_FLAG == 'same': # "same": meta-init
            make_mtrain = False
            make_mtest = False
        else:
            print('error in cfg.META.DATA.LOADER_FLAG')

        train_loader_add['mtrain'] = [] if make_mtrain else None
        train_loader_add['mtest'] = [] if make_mtest else None

        if cfg.META.DATA.SYNTH_SAME_SEED:
            seed = comm.shared_random_seed()
        else:
            seed = None

        if individual_flag_meta: # for individual dataset (domain-wise)
            for i, x in enumerate(train_set_all):
                if make_mtrain:
                    train_loader_add['mtrain'].append(make_sampler(
                        train_set=x,
                        num_batch=cfg.META.DATA.MTRAIN_MINI_BATCH,
                        num_instance=cfg.META.DATA.MTRAIN_NUM_INSTANCE,
                        num_workers=num_workers,
                        mini_batch_size=cfg.META.DATA.MTRAIN_MINI_BATCH // comm.get_world_size(),
                        drop_last=cfg.META.DATA.DROP_LAST,
                        flag1=cfg.META.DATA.NAIVE_WAY,
                        flag2=cfg.META.DATA.DELETE_REM,
                        seed = seed,
                        cfg = cfg))
                if make_mtest:
                    train_loader_add['mtest'].append(make_sampler(
                        train_set=x,
                        num_batch=cfg.META.DATA.MTEST_MINI_BATCH,
                        num_instance=cfg.META.DATA.MTEST_NUM_INSTANCE,
                        num_workers=num_workers,
                        mini_batch_size=cfg.META.DATA.MTEST_MINI_BATCH // comm.get_world_size(),
                        drop_last=cfg.META.DATA.DROP_LAST,
                        flag1=cfg.META.DATA.NAIVE_WAY,
                        flag2=cfg.META.DATA.DELETE_REM,
                        seed = seed,
                        cfg = cfg))
        else:
            if make_mtrain: # meta train dataset
                train_loader_add['mtrain'] = make_sampler(
                    train_set=train_set,
                    num_batch=cfg.META.DATA.MTRAIN_MINI_BATCH,
                    num_instance=cfg.META.DATA.MTRAIN_NUM_INSTANCE,
                    num_workers=num_workers,
                    mini_batch_size=cfg.META.DATA.MTRAIN_MINI_BATCH // comm.get_world_size(),
                    drop_last=cfg.META.DATA.DROP_LAST,
                    flag1=cfg.META.DATA.NAIVE_WAY,
                    flag2=cfg.META.DATA.DELETE_REM,
                    seed = seed,
                    cfg = cfg)
            if make_mtest: # meta train dataset
                if synth_transforms is None:
                    train_loader_add['mtest'] = make_sampler(
                        train_set=train_set,
                        num_batch=cfg.META.DATA.MTEST_MINI_BATCH,
                        num_instance=cfg.META.DATA.MTEST_NUM_INSTANCE,
                        num_workers=num_workers,
                        mini_batch_size=cfg.META.DATA.MTEST_MINI_BATCH // comm.get_world_size(),
                        drop_last=cfg.META.DATA.DROP_LAST,
                        flag1=cfg.META.DATA.NAIVE_WAY,
                        flag2=cfg.META.DATA.DELETE_REM,
                        seed = seed,
                        cfg = cfg)
                else:
                    train_loader_add['mtest'] = make_sampler(
                        train_set=synth_set,
                        num_batch=cfg.META.DATA.MTEST_MINI_BATCH,
                        num_instance=cfg.META.DATA.MTEST_NUM_INSTANCE,
                        num_workers=num_workers,
                        mini_batch_size=cfg.META.DATA.MTEST_MINI_BATCH // comm.get_world_size(),
                        drop_last=cfg.META.DATA.DROP_LAST,
                        flag1=cfg.META.DATA.NAIVE_WAY,
                        flag2=cfg.META.DATA.DELETE_REM,
                        seed = seed,
                        cfg = cfg)

        if frozen: cfg.freeze()

    return train_loader, train_loader_add, cfg



def build_my_reid_test_loader(cfg, dataset_name, opt=None, flag_test=True):
    test_transforms = build_transforms(cfg, is_train=False)
    train_transforms = build_transforms(cfg, is_train=True)
    if opt is None:
        dataset = DATASET_REGISTRY.get(dataset_name)(root=_root)
        if comm.is_main_process():
            if flag_test:
                dataset.show_test()
            else:
                dataset.show_train()
    else:
        dataset = DATASET_REGISTRY.get(dataset_name)(root=[_root, opt])
    if flag_test:
        #test_items = dataset.train
        test_items = dataset.query + dataset.gallery
    else:
        test_items = dataset.train

    test_set = CommDataset(dataset.train, train_transforms, relabel=False)

    sample_data = [dataset.query_per_cam_sampled, dataset.gallery_per_cam_sampled]
    all_data = [dataset.query_per_cam, dataset.gallery_per_cam]

    """ for i in dataset.gallery_per_cam:
        print(i,dataset.gallery_per_cam[i])
        assert False """
    
    # print(dataset.query_per_cam.keys(), dataset.query_per_cam_sampled.keys())
    test_data_loader = {'simple':[], 'all':[], 'tta':[], 'update':None, 'dataset':test_set, 'simple_set':CommDataset([], test_transforms, relabel=False), "pid_mapping":None }
    for idx in range(len(sample_data)):
        for cam_id in sample_data[idx].keys():
            simple, all = sample_data[idx][cam_id], all_data[idx][cam_id]
            if (len(all) == 0):
                continue
            simple_test_set = CommDataset(simple, test_transforms, relabel=False)
            #Combine two CommDataset
            
            all_test_set = CommDataset(all, test_transforms, relabel=False)
            test_data_loader['simple_set'] = test_data_loader['simple_set'].merge_datasets(simple_test_set)

            batch_size = cfg.TEST.IMS_PER_BATCH
            #test time update
            data_sampler = samplers.InferenceSampler(len(test_data_loader['dataset']))
            batch_sampler = torch.utils.data.BatchSampler(data_sampler, batch_size, False)
            
            #test loader
            simple_data_sampler = samplers.InferenceSampler(len(simple_test_set))
            simple_batch_sampler = torch.utils.data.BatchSampler(simple_data_sampler, batch_size, False)

            all_data_sampler = samplers.InferenceSampler(len(all_test_set))
            all_batch_sampler = torch.utils.data.BatchSampler(all_data_sampler, batch_size, False)

            gettrace = getattr(sys, 'gettrace', None)
            if gettrace():
                num_workers = 0
            else:
                num_workers = cfg.DATALOADER.NUM_WORKERS

            simple_test_loader = DataLoader(
                simple_test_set,
                batch_sampler=simple_batch_sampler,
                num_workers=num_workers,  # save some memory
                collate_fn=fast_batch_collator)

            all_test_loader = DataLoader(
                all_test_set,
                batch_sampler=all_batch_sampler,
                num_workers=num_workers,  # save some memory
                collate_fn=fast_batch_collator)
            
            
            test_data_loader['simple'].append(simple_test_loader)
            test_data_loader['all'].append(all_test_loader)
        
        
        tta_sampler = samplers.TTASampler(test_data_loader['dataset'], batch_size)
        test_data_loader["pid_mapping"] = tta_sampler.pid_mapping
        tta_sampler = torch.utils.data.sampler.BatchSampler(tta_sampler, batch_size, drop_last=True)
        tta_loader = DataLoader(
        test_data_loader['dataset'],
        batch_sampler=tta_sampler,
        num_workers=num_workers,  # save some memory
        collate_fn=fast_batch_collator)

        test_data_loader['tta'] = tta_loader
        #test time update loader   
        test_loader = DataLoader(
            test_data_loader['dataset'],
            batch_sampler=batch_sampler,
            num_workers=num_workers,  # save some memory
            collate_fn=fast_batch_collator)

        test_data_loader['update'] = test_loader


    return test_data_loader, len(dataset.query)

def build_reid_test_loader(cfg, dataset_name, opt=None, flag_test=True):
    test_transforms = build_transforms(cfg, is_train=False)

    if opt is None:
        dataset = DATASET_REGISTRY.get(dataset_name)(root=_root)
        if comm.is_main_process():
            if flag_test:
                dataset.show_test()
            else:
                dataset.show_train()
    else:
        dataset = DATASET_REGISTRY.get(dataset_name)(root=[_root, opt])
    if flag_test:
        test_items = dataset.query + dataset.gallery
    else:
        test_items = dataset.train

    test_set = CommDataset(test_items, test_transforms, relabel=False)

    batch_size = cfg.TEST.IMS_PER_BATCH
    data_sampler = samplers.InferenceSampler(len(test_set))
    batch_sampler = torch.utils.data.BatchSampler(data_sampler, batch_size, False)

    gettrace = getattr(sys, 'gettrace', None)
    if gettrace():
        num_workers = 0
    else:
        num_workers = cfg.DATALOADER.NUM_WORKERS

    test_loader = DataLoader(
        test_set,
        batch_sampler=batch_sampler,
        num_workers=num_workers,  # save some memory
        collate_fn=fast_batch_collator)
    return test_loader, len(dataset.query)



def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch


def fast_batch_collator(batched_inputs):
    """
    A simple batch collator for most common reid tasks
    """
    elem = batched_inputs[0]
    if isinstance(elem, torch.Tensor):
        out = torch.zeros((len(batched_inputs), *elem.size()), dtype=elem.dtype)
        for i, tensor in enumerate(batched_inputs):
            out[i] += tensor
        return out

    elif isinstance(elem, container_abcs.Mapping):
        return {key: fast_batch_collator([d[key] for d in batched_inputs]) for key in elem}

    elif isinstance(elem, float):
        return torch.tensor(batched_inputs, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batched_inputs)
    elif isinstance(elem, string_classes):
        return batched_inputs


def make_sampler(train_set, num_batch, num_instance, num_workers,
                 mini_batch_size, drop_last=True, flag1=True, flag2=True, seed=None, cfg=None):

    if flag1:
        data_sampler = samplers.NaiveIdentitySampler(train_set.img_items,
                                                     num_batch, num_instance, flag2, seed, cfg)
    else:
        data_sampler = samplers.DomainSuffleSampler(train_set.img_items,
                                                     num_batch, num_instance, flag2, seed, cfg)
    # data_sampler = samplers.BalancedIdentitySampler(train_set.img_items,num_batch, num_instance) # other method
    # data_sampler = samplers.TrainingSampler(len(train_set)) # PK sampler
    batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, drop_last)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=fast_batch_collator,
    )

    return train_loader

def make_keypoint_data(cfg, data_name, train_items):


    cluster_view = []
    if data_name == "VeRi_keypoint_each_2":  # 2 (7560/3241)
        cluster_view = [[7, 5, 6, 0], [3, 2, 4, 1]]
    elif data_name == "VeRi_keypoint_each_4":  # 4 (75/60/32/41)
        cluster_view = [[7, 5], [6, 0], [3, 2], [4, 1]]
    elif data_name == "VeRi_keypoint_each_8":  # 8
        cluster_view = [[7], [5], [6], [0], [3], [2], [4], [1]]
    else:
        print("error_dataset_names")

    cfg = cfg.clone()
    frozen = cfg.is_frozen()
    cfg.defrost()
    cfg.META.DATA.CLUSTER_VIEW = cluster_view
    if frozen: cfg.freeze()

    train_set_all = []
    for i, x in enumerate(cluster_view):
        train_items_all = train_items.copy()
        len_data = len(train_items_all)
        for j, y in enumerate(reversed(train_items_all)):
            if not y[3]['domains'] in cluster_view[i]:
                del train_items_all[len_data - j - 1]
        train_set_all.append(train_items_all)

    return cfg, train_set_all