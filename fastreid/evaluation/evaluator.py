# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import datetime
import logging
import time
from contextlib import contextmanager
from matplotlib.pyplot import axis

import torch

from fastreid.utils.logger import log_every_n_seconds
from models import *
import copy
import random



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

def inference_on_dataset(model, data_loader, evaluator, model_opt=None, cfg=None, pda_model=None, use_adain=True, pda_optimizer=None, pda_scheduler=None, opt=None):
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

    if use_adain:
        training_mode = model.training
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
                    output = model(inputs, model_opt)
                    # finetune in test 
                    if not training_mode:
                        loss = pda_forward(pda_model, output['outputs']["bn_features"])
                        pda_backward(loss, pda_optimizer)
            model.eval()

            for idx, inputs in enumerate(all_loader):
                if idx == num_warmup:
                    start_time = time.perf_counter()
                    total_compute_time = 0
                start_compute_time = time.perf_counter()
                with torch.no_grad():
                    outputs = model(inputs, model_opt)
                    pda_output = pda_model(outputs)
                    total_compute_time += time.perf_counter() - start_compute_time
                    evaluator.process(inputs, outputs * 0.6 + pda_output[0] * 0.4)

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
# backward
#####################################################################
def pda_backward(losses, optimizer, retain_graph = False):
    if (losses != None) and (optimizer != None):
        optimizer.zero_grad()
        losses['loss'].backward(retain_graph = retain_graph)
        optimizer.step()
