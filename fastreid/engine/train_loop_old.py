# encoding: utf-8
"""
credit:
https://github.com/facebookresearch/detectron2/blob/master/detectron2/engine/train_loop.py
"""

import logging
import time
import weakref
import os

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from collections import OrderedDict
import fastreid.utils.comm as comm
from fastreid.utils.events import EventStorage
from fastreid.utils.file_io import PathManager
logger = logging.getLogger(__name__)
import copy

from fastreid.utils.weight_init import weights_init_kaiming, weights_init_classifier

__all__ = ["HookBase", "TrainerBase", "SimpleTrainer"]


class HookBase:
    """
    Base class for hooks that can be registered with :class:`TrainerBase`.
    Each hook can implement 4 methods. The way they are called is demonstrated
    in the following snippet:
    .. code-block:: python
        hook.before_train()
        for iter in range(start_iter, max_iter):
            hook.before_step()
            trainer.run_step()
            hook.after_step()
        hook.after_train()
    Notes:
        1. In the hook method, users can access `self.trainer` to access more
           properties about the context (e.g., current iteration).
        2. A hook that does something in :meth:`before_step` can often be
           implemented equivalently in :meth:`after_step`.
           If the hook takes non-trivial time, it is strongly recommended to
           implement the hook in :meth:`after_step` instead of :meth:`before_step`.
           The convention is that :meth:`before_step` should only take negligible time.
           Following this convention will allow hooks that do care about the difference
           between :meth:`before_step` and :meth:`after_step` (e.g., timer) to
           function properly.
    Attributes:
        trainer: A weak reference to the trainer object. Set by the trainer when the hook is
            registered.
    """

    def before_train(self):
        """
        Called before the first iteration.
        """
        pass

    def after_train(self):
        """
        Called after the last iteration.
        """
        pass

    def before_step(self):
        """
        Called before each iteration.
        """
        pass

    def after_step(self):
        """
        Called after each iteration.
        """
        pass
class TrainerBase:
    """
    Base class for iterative trainer with hooks.
    The only assumption we made here is: the training runs in a loop.
    A subclass can implement what the loop is.
    We made no assumptions about the existence of dataloader, optimizer, model, etc.
    Attributes:
        iter(int): the current iteration.
        start_iter(int): The iteration to start with.
            By convention the minimum possible value is 0.
        max_iter(int): The iteration to end training.
        storage(EventStorage): An EventStorage that's opened during the course of training.
    """
    def __init__(self):
        self._hooks = []
    def register_hooks(self, hooks):
        """
        Register hooks to the trainer. The hooks are executed in the order
        they are registered.
        Args:
            hooks (list[Optional[HookBase]]): list of hooks
        """
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self._hooks.extend(hooks)
    def train(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            self.before_train() # check hooks.py, engine/defaults.py
            for self.iter in range(start_iter, max_iter):
                self.before_step()
                if self.cfg.META.DATA.NAMES == '':
                    self.run_step()
                else:
                    self.run_step_meta_learning()
                self.after_step()
            self.after_train()
    def before_train(self):
        for h in self._hooks:
            h.before_train()
    def after_train(self):
        for h in self._hooks:
            h.after_train()
    def before_step(self):
        for h in self._hooks:
            h.before_step()
    def after_step(self):
        for h in self._hooks:
            h.after_step()
        # this guarantees, that in each hook's after_step, storage.iter == trainer.iter
        self.storage.step()
    def run_step(self):
        raise NotImplementedError
class SimpleTrainer(TrainerBase):

    def __init__(self, cfg, model, data_loader, data_loader_add, optimizer, meta_param):
        super().__init__()
        self.model = model
        self.data_loader = data_loader

        if isinstance(data_loader, list):
            self._data_loader_iter = []
            for x in data_loader:
                self._data_loader_iter.append(iter(x))
        else:
            self._data_loader_iter = iter(data_loader)

        self.optimizer = optimizer
        self.meta_param = meta_param
        if cfg.SOLVER.AMP:
            self.scaler = torch.cuda.amp.GradScaler()
            print("using amp +++++++++++++++++++++++++++++++++++++++++++")
            assert False
        else:
            self.scaler = None
            print(" Not using amp +++++++++++++++++++++++++++++++++++++++++++")
            assert False


        # additional setting
        self.bin_gates = [p for p in self.model.parameters() if getattr(p, 'bin_gate', False)]

        # Meta-leaning setting
        if len(self.meta_param) > 0:

            if data_loader_add['mtrain'] != None:
                self.data_loader_mtrain = data_loader_add['mtrain']
                if isinstance(self.data_loader_mtrain, list):
                    self._data_loader_iter_mtrain = []
                    for x in self.data_loader_mtrain:
                        self._data_loader_iter_mtrain.append(iter(x))
                else:
                    self._data_loader_iter_mtrain = iter(self.data_loader_mtrain)
            else:
                self.data_loader_mtrain = None
                self._data_loader_iter_mtrain = self._data_loader_iter

            if data_loader_add['mtest'] != None:
                self.data_loader_mtest = data_loader_add['mtest']
                if isinstance(self.data_loader_mtest, list):
                    self._data_loader_iter_mtest = []
                    for x in self.data_loader_mtest:
                        self._data_loader_iter_mtest.append(iter(x))
                else:
                    self._data_loader_iter_mtest = iter(self.data_loader_mtest)
            else:
                self.data_loader_mtest = None
                self._data_loader_iter_mtest = self._data_loader_iter_mtrain

            self.initial_requires_grad = self.grad_requires_init(model = self.model)
            find_group = ['conv', 'gate']
            new_group = list(self.cat_tuples(self.meta_param['meta_compute_layer'], self.meta_param['meta_update_layer']))
            find_group.extend(new_group)
            find_group = list(set(find_group))
            idx_group, dict_group = self.find_selected_optimizer(find_group, self.optimizer)
            self.idx_group = idx_group
            self.dict_group = dict_group
            self.inner_clamp = True
            self.print_flag = True



    def run_step(self):

        # initial setting
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        metrics_dict = dict()

        # Load dataset
        data, data_time = self.get_data(self._data_loader_iter, None)

        # Training (forward & backward)
        opt = self.opt_setting('basic') # option
        losses, loss_dict = self.basic_forward(data, self.model, opt) # forward
        self.basic_backward(losses, self.optimizer) # backward

        # Post-processing
        for name, val in loss_dict.items(): metrics_dict[name] = val
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)
        if self.iter % (self.cfg.SOLVER.WRITE_PERIOD_PARAM * self.cfg.SOLVER.WRITE_PERIOD) == 0:
            self.logger_parameter_info(self.model)

    def run_step_meta_learning(self):

        # initial setting
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        metrics_dict = dict()

        # 1) Meta-initialization
        name_loss = '1)'
        self.grad_setting('basic')
        opt = self.opt_setting('basic')
        cnt_init = 0
        data_time_all = 0.0
        while(cnt_init < self.meta_param['iter_init_inner']):
            cnt_init += 1
            data, data_time = self.get_data(self._data_loader_iter, None)
            data_time_all += data_time

            losses, loss_dict = self.basic_forward(data, self.model, opt) # forward
            self.basic_backward(losses, self.optimizer) # backward

            for name, val in loss_dict.items():
                t = name_loss+name
                metrics_dict[t] = metrics_dict[t] + val if t in metrics_dict.keys() else val
        for name in metrics_dict.keys():
            if name_loss in name: metrics_dict[name] /= float(cnt_init)

        # Meta-learning
        cnt_meta = 0
        while(cnt_meta < self.meta_param['iter_init_outer']):
            cnt_meta += 1
            list_all = np.random.permutation(self.meta_param['num_domain'])
            list_mtrain = list(list_all[0:self.meta_param['num_mtrain']])
            list_mtest = list(list_all[self.meta_param['num_mtrain']:
                                       self.meta_param['num_mtrain']+self.meta_param['num_mtest']])

            # 2) Meta-train
            name_loss = '2)'
            self.grad_setting('mtrain')
            opt = self.opt_setting('mtrain')
            data, data_time = self.get_data(self._data_loader_iter_mtrain, list_mtrain)
            data_time_all += data_time
            mtrain_losses = []

            losses, loss_dict = self.basic_forward(data, self.model, opt) # forward
            mtrain_losses.append(losses)
            for name, val in loss_dict.items():
                t = name_loss + name
                metrics_dict[t] = metrics_dict[t] + val if t in metrics_dict.keys() else val

            # cnt_inner = 1 # inner-loop (optional)
            # while(cnt_inner < self.meta_param['iter_mtrain']):
            #     cnt_inner += 1
            #     name_loss = '2-'+str(cnt_inner)+')'
            #     if self.meta_param['inner_loop_type'] == 'diff':
            #         data, data_time = self.get_data(self._data_loader_iter_mtrain, list_mtrain)
            #         data_time_all += data_time
            #
            #     opt = self.opt_setting('mtrain_inner')
            #     opt['meta_losses'] = losses
            #     losses, loss_dict = self.basic_forward(data, self.model, opt) # forward
            #     for name, val in loss_dict.items():
            #         t = name_loss + name
            #         metrics_dict[t] = metrics_dict[t] + val if t in metrics_dict.keys() else val
            #     if self.meta_param['loss_combined']:
            #         mtrain_losses.append(losses)


            # 3) Meta-test
            name_loss = '3)'
            self.grad_setting('mtest')
            opt = self.opt_setting('mtest') # option
            opt['meta_losses'] = losses
            data, data_time = self.get_data(self._data_loader_iter_mtest, list_mtest)
            data_time_all += data_time
            losses, loss_dict = self.basic_forward(data, self.model, opt) # forward
            for name, val in loss_dict.items():
                t = name_loss + name
                metrics_dict[t] = metrics_dict[t] + val if t in metrics_dict.keys() else val
            if self.meta_param['loss_combined']:
                if len(mtrain_losses) == 1:
                    mtrain_losses = mtrain_losses[0]
                else:
                    mtrain_losses = torch.stack(mtrain_losses)
                total_losses = mtrain_losses + losses
            else:
                total_losses = losses
            self.basic_backward(total_losses, self.optimizer) # backward

        metrics_dict["data_time"] = data_time_all
        self._write_metrics(metrics_dict)
        self.logger_parameter_info(self.model)

    def get_data(self, data_loader_iter, list_sample = None):
        start = time.perf_counter()
        if data_loader_iter is not None:
            data = None
            while(data == None):
                if isinstance(data_loader_iter, list):
                    if list_sample is None:
                        data = self.data_aggregation(dataloader = data_loader_iter, list_num = [x for x in range(len(data_loader_iter))])
                    else:
                        data = self.data_aggregation(dataloader = data_loader_iter, list_num = [x for x in list_sample])

                else:
                    data = next(data_loader_iter)

                    if list_sample != None:
                        domain_idx = data['others']['domains']
                        cnt = 0
                        for sample in list_sample:
                            if cnt == 0:
                                t_logical_domain = domain_idx == sample
                            else:
                                t_logical_domain += domain_idx == sample
                            cnt += 1

                        # data1
                        if int(sum(t_logical_domain)) == 0:
                            data = None
                            logger.info('No data including list_domain')
                        else:
                            # data1 = dict()
                            for name, value in data.items():
                                if torch.is_tensor(value):
                                    data[name] = data[name][t_logical_domain]
                                elif isinstance(value, dict):
                                    for name_local, value_local in value.items():
                                        if torch.is_tensor(value_local):
                                            data[name][name_local] = data[name][name_local][t_logical_domain]
                                elif isinstance(value, list):
                                    data[name] = [x for i, x in enumerate(data[name]) if t_logical_domain[i]]

                        # data2 (if opt == 'all')
                        # if opt == 'all':
                        #     t_logical_domain_reversed = t_logical_domain == False
                        #     if int(sum(t_logical_domain_reversed)) == 0:
                        #         data2 = None
                        #         logger.info('No data including list_domain')
                        #     else:
                        #         data2 = dict()
                        #         for name, value in data.items():
                        #             if torch.is_tensor(value):
                        #                 data2[name] = data[name][t_logical_domain_reversed]
                        #             elif isinstance(value, dict):
                        #                 for name_local, value_local in value.items():
                        #                     if torch.is_tensor(value_local):
                        #                         data2[name][name_local] = data[name][name_local][t_logical_domain_reversed]
                        #             elif isinstance(value, list):
                        #                 data2[name] = [x for i, x in enumerate(data[name]) if t_logical_domain_reversed[i]]
                        #     data = [data1, data2]
                        # else:
                        #     data = data1
        else:
            data = None
            logger.info('No data including list_domain')

        data_time = time.perf_counter() - start
                # sample data

        return data, data_time
    def basic_forward(self, data, model, opt):
        model = model.module if isinstance(model, DistributedDataParallel) else model
        if data != None:
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                outs = model(data, opt)
                loss_dict = model.losses(outs, opt)
                losses = sum(loss_dict.values())
            self._detect_anomaly(losses, loss_dict)
        else:
            losses = None
            loss_dict = dict()

        return losses, loss_dict
    def basic_backward(self, losses, optimizer):
        if losses != None:
            optimizer.zero_grad()
            if len(self.meta_param) > 0:
                if self.meta_param['flag_manual_zero_grad']:
                    self.manual_zero_grad(self.model)
            if self.scaler is None:
                losses.backward()
                optimizer.step()
            else:
                self.scaler.scale(losses).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            for p in self.bin_gates:
                p.data.clamp_(min=0, max=1)
            if self.meta_param['sync']: torch.cuda.synchronize()
    def opt_setting(self, flag):
        if flag == 'basic':
            opt = {}
            opt['param_update'] = False
            opt['loss'] = self.cfg['MODEL']['LOSSES']['NAME']
        elif flag == 'mtrain':
            opt = {}
            opt['param_update'] = False
            opt['loss'] = self.meta_param['loss_name_mtrain']
            opt['use_second_order'] = self.meta_param['use_second_order']
        elif flag == 'mtrain_inner':
            opt = {}
            opt['param_update'] = True
            opt['loss'] = self.meta_param['loss_name_mtrain']
            opt['use_second_order'] = self.meta_param['use_second_order']
        elif flag == 'mtest':
            opt = {}
            opt['param_update'] = True
            opt['loss'] = self.meta_param['loss_name_mtest']
            opt['use_second_order'] = self.meta_param['use_second_order']
        return opt
    def grad_setting(self, flag):
        if flag == 'basic':
            self.grad_requires_remove(
                model = self.model,
                ori_grad = self.initial_requires_grad,
                freeze_target = self.meta_param['meta_update_layer'],
                reverse_flag = False, # True: freeze target / False: freeze w/o target
                print_flag = self.print_flag)
        elif flag == 'mtrain':
            if self.meta_param['freeze_gradient_meta']:
                self.grad_requires_remove(
                    model = self.model,
                    ori_grad = self.initial_requires_grad,
                    freeze_target = self.cat_tuples(self.meta_param['meta_update_layer'], self.meta_param['meta_compute_layer']),
                    reverse_flag = True, # True: freeze target / False: freeze w/o target
                    print_flag = self.print_flag)
            else:
                self.grad_requires_recover(model=self.model, ori_grad=self.initial_requires_grad)
        elif flag == 'mtest':
            self.grad_requires_remove(
                model=self.model,
                ori_grad=self.initial_requires_grad,
                freeze_target=self.meta_param['meta_update_layer'],
                reverse_flag=True, # True: freeze target / False: freeze w/o target
                print_flag=self.print_flag)

    def find_selected_optimizer(self, find_group, optimizer):

        # find parameter, lr, required_grad, shape
        logger.info('Storage parameter, lr, requires_grad, shape! in {}'.format(find_group))
        idx_group = []
        dict_group = dict()
        for j in range(len(find_group)):
            idx_local = []
            for i, x in enumerate(optimizer.param_groups):
                if find_group[j] in x['name']:
                    dict_group[x['name']] = i
                    idx_local.append(i)
            if len(idx_local) > 0:
                logger.info('Find {} in {}'.format(find_group[j], optimizer.param_groups[idx_local[0]]['name']))
                idx_group.append(idx_local[0])
            else:
                logger.info('error in find_group')
        return idx_group, dict_group
    def print_selected_optimizer(self, txt, idx_group, optimizer, detail_mode):

        if detail_mode:
            num_float = 8
            only_reg = False

            for x in idx_group:
                t_name = optimizer.param_groups[x]['name']
                if only_reg and not 'reg' in t_name:
                    continue
                t_param = optimizer.param_groups[x]['params'][0].view(-1)[0]
                t_lr = optimizer.param_groups[x]['lr']
                t_grad = optimizer.param_groups[x]['params'][0].requires_grad
                # t_shape = optimizer.param_groups[x]['params'][0].shape
                for name, param in self.model.named_parameters():
                    if name == t_name:
                        m_param = param.view(-1)[0]
                        m_grad = param.requires_grad
                        val = torch.sum(param - optimizer.param_groups[x]['params'][0])
                        break
                if float(val) != 0:
                    logger.info('*****')
                    logger.info('<=={}==>[optimizer] --> [{}], w:{}, grad:{}, lr:{}'.format(txt, t_name, round(float(t_param), num_float), t_grad, t_lr))
                    logger.info('<=={}==>[self.model] --> [{}], w:{}, grad:{}'.format(txt, t_name, round(float(m_param), num_float), m_grad))
                    logger.info('*****')
                else:
                    logger.info('[**{}**] --> [{}], w:{}, grad:{}, lr:{}'.format(txt, t_name, round(float(t_param), num_float), t_grad, t_lr))
    def logger_parameter_info(self, model):

        with torch.no_grad():
            write_dict = dict()
            round_num = 4
            name_num = 20
            for name, param in model.named_parameters():  # only update regularizer
                if 'reg' in name:
                    name = '_'.join([x[:name_num] for x in name.split('.')[1:]])
                    name = name + '+'
                    write_dict[name] = round(float(torch.sum(param.data.view(-1) > 0)) / len(param.data.view(-1)),
                                             round_num)

            for name, param in model.named_parameters():
                if ('meta' in name) and ('fc' in name) and ('weight' in name) and (not 'domain' in name):
                    name = '_'.join([x[:name_num] for x in name.split('.')[1:]])
                    # name_std = name + '_std'
                    # write_dict[name_std] = round(float(torch.std(param.data.view(-1))), round_num)
                    # name_mean = name + '_mean'
                    # write_dict[name_mean] = round(float(torch.mean(param.data.view(-1))), round_num)
                    name_std10 = name + '_std10'
                    ratio = 0.1
                    write_dict[name_std10] = round(
                        float(torch.sum((param.data.view(-1) > - ratio * float(torch.std(param.data.view(-1)))) * (
                                param.data.view(-1) < ratio * float(torch.std(param.data.view(-1)))))) / len(
                            param.data.view(-1)), round_num)

            for name, param in model.named_parameters():
                if ('gate' in name) and (not 'domain' in name):
                    name = '_'.join([x[:name_num] for x in name.split('.')[1:]])
                    name_mean = name + '_mean'
                    write_dict[name_mean] = round(float(torch.mean(param.data.view(-1))), round_num)
            logger.info(write_dict)
    def grad_requires_init(self, model):

        out_requires_grad = dict()
        for name, param in model.named_parameters():
            out_requires_grad[name] = param.requires_grad
        return out_requires_grad
    def grad_requires_remove(self, model, ori_grad, freeze_target, reverse_flag = False, print_flag = False):

        if reverse_flag: # freeze layers w/o target layers
            for name, param in model.named_parameters():
                param.requires_grad = False # freeze remaining layers
                for freeze_name in list(freeze_target):
                    if freeze_name in name:
                        param.requires_grad = ori_grad[name] # recover target layers
                        if print_flag:
                            print("melt '{}' layer's grad".format(name))
        else: # freeze layers based on target
            for name, param in model.named_parameters():
                param.requires_grad = ori_grad[name] # recover remaining layers
                for freeze_name in list(freeze_target):
                    if freeze_name in name:
                        param.requires_grad = False # freeze target layers
                        if print_flag:
                            print("freeze '{}' layer's grad".format(name))
    def grad_requires_recover(self, model, ori_grad):

        # recover gradient requirements
        for name, param in model.named_parameters():
            param.requires_grad = ori_grad[name]
    def grad_val_remove(self, model, freeze_target, reverse_flag = False, print_flag = False):

        if reverse_flag: # remove grad w/o target layers
            for name, param in model.named_parameters():
                if param.grad is not None:
                    remove_flag = True
                    for freeze_name in list(freeze_target):
                        if freeze_name in name:
                            remove_flag = False
                    if remove_flag:
                        param.grad = None
                        if print_flag:
                            print("remove '{}' layer's grad".format(name))
        else: # remove grad based on target layers
            for name, param in model.named_parameters():
                if param.grad is not None:
                    for freeze_name in list(freeze_target):
                        if freeze_name in name:
                            param.grad = None
                            if print_flag:
                                print("remove '{}' layer's grad".format(name))
    def data_aggregation(self, dataloader, list_num):

        data = None
        for cnt, list_idx in enumerate(list_num):
            if cnt == 0:
                data = next(dataloader[list_idx])
            else:
                for name, value in next(dataloader[list_idx]).items():
                    if torch.is_tensor(value):
                        data[name] = torch.cat((data[name], value), 0)
                    elif isinstance(value, dict):
                        for name_local, value_local in value.items():
                            if torch.is_tensor(value_local):
                                data[name][name_local] = torch.cat((data[name][name_local], value_local), 0)
                    elif isinstance(value, list):
                        data[name].extend(value)

        return data
    def cat_tuples(self, tuple1, tuple2):

        list1 = list(tuple1)
        list2 = list(tuple2)
        list_all = list1.copy()
        list_all.extend(list2)
        list_all = list(set(list_all))
        if "" in list_all:
            list_all.remove("")
        list_all = tuple(list_all)
        return list_all
    def manual_zero_grad(self, model):
        for name, param in model.named_parameters():  # parameter grad_zero
            if param.grad is not None:
                # param.grad.zero_()
                param.grad = None
        # return model
    def meta_learning(self):

        inner_clamp = True
        print_flag = True

        bin_gates = [p for p in self.model.parameters() if getattr(p, 'bin_gate', False)]


        num_domain = self.meta_param['num_domain']

        split_layer = self.meta_param['split_layer']
        meta_compute_layer = self.meta_param['meta_compute_layer']
        meta_update_layer = self.meta_param['meta_update_layer']
        find_group = ['conv', 'meta', 'reg', 'gate']
        new_group = list(self.cat_tuples(self.cat_tuples(meta_compute_layer, meta_update_layer), split_layer))
        find_group.extend(new_group)
        find_group = list(set(find_group))
        idx_group, dict_group = self.find_selected_optimizer(find_group, self.optimizer)
        ds_flag = self.meta_param['split_layer'][0] != ''

        iter_init_inner = self.meta_param['iter_init_inner'] # META.SOLVER.INIT.INNER_LOOP
        iter_init_outer = self.meta_param['iter_init_outer'] # META.SOLVER.INIT.OUTER_LOOP

        iter_mtrain = self.meta_param['iter_mtrain'] # META.SOLVER.MTRAIN.INNER_LOOP
        inner_loop_type = self.meta_param['inner_loop_type'] # META.SOLVER.MTRAIN.INNER_LOOP_TYPE
        use_second_order = self.meta_param['use_second_order']
        num_mtrain = self.meta_param['num_mtrain'] # META.SOLVER.MTEST.NUM_DOMAIN
        freeze_gradient_meta = self.meta_param['freeze_gradient_meta']
        num_mtest = self.meta_param['num_mtest'] # META.SOLVER.MTEST.NUM_DOMAIN

        sync_flag = self.meta_param['sync']
        detail_mode = self.meta_param['detail_mode']
        flag_manual_zero_grad = self.meta_param['flag_manual_zero_grad']

        loss_combined = self.meta_param['loss_combined']
        loss_name_init = self.meta_param['loss_name_init']
        loss_name_mtrain = self.meta_param['loss_name_mtrain']
        loss_name_mtest = self.meta_param['loss_name_mtest']


        # 2. Meta-learning
        initial_requires_grad = self.grad_requires_init(model = self.model)

        # 2.1. Learning domain specific layer
        # Dataloader: each domain dataloader (continue) -> batch N
        # Parameter compute and update: shared layers, domain specific layers (continue)
        # Regularizer: None (continue)
        # Loss: CE/TRIP (continue)
        opt = {}
        opt['ds_flag'] = ds_flag
        opt['param_update'] = False
        opt['loss'] = loss_name_init
        self.grad_requires_remove(model = self.model,
                                  ori_grad = initial_requires_grad,
                                  freeze_target = meta_update_layer,
                                  reverse_flag = False,
                                  print_flag = print_flag)

        cnt_local = 0
        while(cnt_local < iter_init_inner):
            cnt_local += 1
            for i in range(len(dataloader_init_iter)):
                data = next(dataloader_init_iter[i])
                losses, loss_dict_minit = self.basic_forward(opt, data)
                self.optimizer.zero_grad()
                if flag_manual_zero_grad:
                    self.manual_zero_grad(self.model)
                if self.scaler is None:
                    losses.backward()
                    self.grad_val_remove(model=self.model,
                                         freeze_target=meta_update_layer,
                                         reverse_flag=False,
                                         print_flag=print_flag)
                    self.optimizer.step()
                else:
                    self.scaler.scale(losses).backward()
                    self.grad_val_remove(model=self.model,
                                         freeze_target=meta_update_layer,
                                         reverse_flag=False,
                                         print_flag=print_flag)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                for p in bin_gates:
                    p.data.clamp_(min=0, max=1)

        self.print_selected_optimizer('after meta-init', idx_group, self.optimizer, detail_mode)

        # sync
        self.optimizer.zero_grad()
        if sync_flag:
            torch.cuda.synchronize()

        # 2.2. Meta-training
        # Dataloader: each domain dataloader (meta-training) -> unroll batch K x #(meta-train dataset)
        # Parameter compute: domain specific layers (unroll) -> new step paramater (actually same as lr)
        # Freeze layers: conv
        # Regularizer: Yes
        # Loss: CE/TRIP + Reg

        cnt_outer = 0
        while(cnt_outer < iter_init_outer):
            cnt_outer += 1
            list_all = np.random.permutation(num_domain)
            list_mtrain = list(list_all[0:num_mtrain])
            list_mtest = list(list_all[num_mtrain:num_mtrain+num_mtest])


            if not ds_flag: # [single-domain layer] meta-learning (MAML style)
                # whole train domain -> meta-train (innerloop) / whole test domain -> meta-test

                for p in bin_gates:
                    # print(p)
                    p.data.clamp_(min=0, max=1)


                if freeze_gradient_meta:
                    self.grad_requires_remove(
                        model=self.model,
                        ori_grad=initial_requires_grad,
                        freeze_target=self.cat_tuples(meta_update_layer, meta_compute_layer),
                        reverse_flag=True,  # freeze all w/o target layers
                        print_flag=print_flag)
                else:
                    self.grad_requires_recover(model=self.model, ori_grad=initial_requires_grad)

                opt['ds_flag'] = False
                opt['param_update'] = False
                opt['loss'] = loss_name_mtrain
                opt['use_second_order'] = use_second_order
                meta_train_losses = []

                cnt_local = 0
                while(cnt_local < iter_mtrain):
                    cnt_local += 1
                    if (cnt_local == 1) or (inner_loop_type == 'diff'):
                        data = self.data_aggregation(dataloader = dataloader_mtrain_iter, list_num = list_mtrain)
                    losses, loss_dict_mtrain = self.basic_forward(opt, data)
                    self.optimizer.zero_grad()
                    if flag_manual_zero_grad:
                        self.manual_zero_grad(self.model)
                    if not opt['param_update']:  # first inner-loop
                        opt['new_param'] = dict()
                        # start_flag = False
                        for name, param in self.model.named_parameters():  # grad update
                            for compute_name in list(meta_compute_layer):
                                if compute_name in name:
                                    print(name)
                                    lr = self.optimizer.param_groups[dict_group[name]]['lr']
                                    # lr = 10000
                                    grads = torch.autograd.grad(losses, param,
                                                                create_graph=opt['use_second_order'],
                                                                allow_unused=False)[0]
                                    if not grads == None:
                                        opt['new_param'][name] = param - lr * grads
                                        if inner_clamp and 'gate' in name:
                                            opt['new_param'][name].data.clamp_(min=0, max=1)
                                        if detail_mode:
                                            print(torch.sum(torch.abs(grads)))
                                    # continue
                    else:  # after first inner-loop
                        old_param = opt['new_param']
                        opt['new_param'] = dict()
                        for name, param in old_param.items():  # grad update
                            for compute_name in list(meta_compute_layer):
                                if compute_name in name:
                                    grads = torch.autograd.grad(losses, param,
                                                                create_graph=opt['use_second_order'],
                                                                allow_unused=True)[0]
                                    if not grads == None:
                                        opt['new_param'][name] = param - lr * grads
                                        if inner_clamp and 'gate' in name:
                                            opt['new_param'][name].data.clamp_(min=0, max=1)
                                        if detail_mode:
                                            print(torch.sum(torch.abs(grads)))
                    opt['param_update'] = True  # use computed weight
                    self.print_selected_optimizer('after meta train (iter, after backward)', idx_group,
                                                  self.optimizer, detail_mode)
                    if loss_combined:
                        meta_train_losses.append(losses)



            else: # [multi-domain layer] meta-learning (MetaReg style)
                # requires domain-specific layers, regularizer, re-train
                # each train domain -> meta-train (innerloop) / whole test domain -> meta-test
                for idx_mtrain in list_mtrain:
                    if freeze_gradient_meta:
                        self.grad_requires_remove(
                            model=self.model,
                            ori_grad=initial_requires_grad,
                            freeze_target=self.cat_tuples(meta_update_layer, meta_compute_layer),
                            reverse_flag=True,  # freeze all w/o target layers
                            print_flag=print_flag)
                    else:
                        self.grad_requires_recover(model = self.model, ori_grad = initial_requires_grad)

                    opt['ds_flag'] = ds_flag
                    opt['param_update'] = False
                    opt['loss'] = loss_name_mtrain
                    opt['use_second_order'] = use_second_order
                    meta_train_losses = []

                    cnt_local = 0
                    while(cnt_local < iter_mtrain):
                        cnt_local += 1
                        if (cnt_local == 1) or (inner_loop_type == 'diff'):
                            data = next(dataloader_mtrain_iter[idx_mtrain])
                        opt['domain_idx'] = int(idx_mtrain)
                        losses, loss_dict_mtrain = self.basic_forward(opt, data)
                        self.optimizer.zero_grad()
                        if flag_manual_zero_grad:
                            self.manual_zero_grad(self.model)
                        if not opt['param_update']: # first inner-loop
                            opt['new_param'] = dict()
                            for name, param in self.model.named_parameters(): # grad update
                                if 'domain{}'.format(opt['domain_idx']) in name:
                                    lr = self.optimizer.param_groups[dict_group[name]]['lr']
                                    grads = torch.autograd.grad(losses, param,
                                                                create_graph=opt['use_second_order'],
                                                                allow_unused=True)[0]
                                    if not grads == None:
                                        opt['new_param'][name] = param - lr * grads
                                        if detail_mode:
                                            print(torch.sum(torch.abs(grads)))
                        else: # after first inner-loop
                            old_param = opt['new_param']
                            opt['new_param'] = dict()
                            for name, param in old_param.items(): # grad update
                                if 'domain{}'.format(opt['domain_idx']) in name:
                                    grads = torch.autograd.grad(losses, param,
                                                                create_graph=opt['use_second_order'],
                                                                allow_unused=True)[0]
                                    if not grads == None:
                                        opt['new_param'][name] = param - lr * grads
                                        if detail_mode:
                                            print(torch.sum(torch.abs(grads)))
                        opt['param_update'] = True # use computed weight
                        self.print_selected_optimizer('after meta train (iter, after backward)', idx_group, self.optimizer, detail_mode)
                        if loss_combined:
                            meta_train_losses.append(losses)

                    # 2.3. Meta-testing
                    # Dataloader: each domain dataloader (meta-testing) -> batch K' x #(meta-test dataset)
                    # [maybe N = K*(D_train) + K'*(D_test), where D = D_train + D_test]
                    # Load parameter: domain specific layers
                    # Parameter compute and update: regularizer -> new step paramater (actually same as lr)
                    # Freeze layers: conv
                    # Regularizer: Yes
                    # Loss: CE/TRIP + (Reg)

                    if sync_flag:
                        torch.cuda.synchronize()
                    opt['ds_flag'] = ds_flag
                    opt['param_update'] = True # use computed weight
                    opt['loss'] = loss_name_mtest
                    opt['domain_idx'] = int(idx_mtrain)

                    data = self.data_aggregation(dataloader=dataloader_mtest_iter, list_num=list_mtest)
                    final_losses, loss_dict_mtest = self.basic_forward(opt, data)
                    self.print_selected_optimizer('after meta test (before backward)', idx_group, self.optimizer, detail_mode)
                    if loss_combined:
                        final_losses += torch.sum(torch.stack(meta_train_losses))

                    self.optimizer.zero_grad()
                    if flag_manual_zero_grad:
                        self.manual_zero_grad(self.model)

                    self.grad_requires_remove(model=self.model,
                                              ori_grad=initial_requires_grad,
                                              freeze_target=meta_update_layer,
                                              reverse_flag=True,  # freeze all w/o target layers
                                              print_flag=print_flag)

                    if self.scaler is None:
                        final_losses.backward()
                        self.grad_val_remove(model=self.model,
                                             freeze_target=meta_update_layer,
                                             reverse_flag=True,
                                             print_flag=print_flag)
                        self.optimizer.step()
                    else:
                        self.scaler.scale(final_losses).backward()
                        self.grad_val_remove(model=self.model,
                                             freeze_target=meta_update_layer,
                                             reverse_flag=True,
                                             print_flag=print_flag)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()

                    for p in bin_gates:
                        p.data.clamp_(min=0, max=1)
                    self.print_selected_optimizer('after meta test (final)', idx_group, self.optimizer, detail_mode)
                    if sync_flag:
                        torch.cuda.synchronize()

        logger.info('2) Meta-Optimization ({} domains), [test]{}, [reg_sum]:{} //// [train]{}, [init]{}, '.format(num_domain,
            ['{}:{}'.format(name, round(float(val), 4)) for name, val in loss_dict_mtest.items()],
            [round(float(torch.sum(torch.abs(param))), 6) for name, param in self.model.named_parameters() if 'reg' in name],
            ['{}:{}'.format(name, round(float(val), 4)) for name, val in loss_dict_mtrain.items()],
            ['{}:{}'.format(name, round(float(val), 4)) for name, val in loss_dict_minit.items()],
        ))
        self.logger_parameter_info(self.model)

    def _detect_anomaly(self, losses, loss_dict):
        if not torch.isfinite(losses).all():
            raise FloatingPointError(
                "Loss became infinite or NaN at iteration={}!\nloss_dict = {}".format(
                    self.iter, loss_dict
                )
            )
    def _write_metrics(self, metrics_dict: dict):
        """
        Args:
            metrics_dict (dict): dict of scalar metrics
        """
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in fastreid.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                # data_time among workers can have high variance. The actual latency
                # caused by data_time is the maximum among workers.
                data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
            }
            total_losses_reduced = sum(loss for loss in metrics_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)




            # cnt_inner = 1 # inner-loop (optional)
            # while(cnt_inner < self.meta_param['iter_mtrain']):
            #     cnt_inner += 1
            #     name_loss = '2-'+str(cnt_inner)+')'
            #     if self.meta_param['inner_loop_type'] == 'diff':
            #         data, data_time = self.get_data(self._data_loader_iter_mtrain, list_mtrain)
            #         data_time_all += data_time
            #
            #     opt = self.opt_setting('mtrain_inner')
            #     opt['meta_losses'] = losses
            #     losses, loss_dict = self.basic_forward(data, self.model, opt) # forward
            #     for name, val in loss_dict.items():
            #         t = name_loss + name
            #         metrics_dict[t] = metrics_dict[t] + val if t in metrics_dict.keys() else val
            #     if self.meta_param['loss_combined']:
            #         mtrain_losses.append(losses)