# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import copy
import itertools
from collections import defaultdict
from typing import Optional

import numpy as np
from torch.utils.data.sampler import Sampler

from fastreid.utils import comm


def no_index(a, b):
    assert isinstance(a, list)
    return [i for i, j in enumerate(a) if j != b]


class BalancedIdentitySampler(Sampler):
    def __init__(self, data_source: str, batch_size: int, num_instances: int, seed: Optional[int] = None):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // self.num_instances

        self.index_pid = defaultdict(list)
        self.pid_cam = defaultdict(list)
        self.pid_index = defaultdict(list)

        for index, info in enumerate(data_source):
            pid = info[1]
            # camid = info[2]
            camid = info[3]['domains']
            self.index_pid[index] = pid
            self.pid_cam[pid].append(camid)
            self.pid_index[pid].append(index)

        self.pids = list(self.pid_index.keys())
        self.num_identities = len(self.pids)

        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)

        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()

    def _get_epoch_indices(self):
        # Shuffle identity list
        identities = np.random.permutation(self.num_identities)

        # If remaining identities cannot be enough for a batch,
        # just drop the remaining parts
        drop_indices = self.num_identities % self.num_pids_per_batch
        if drop_indices: identities = identities[:-drop_indices]

        ret = []
        for kid in identities:
            i = np.random.choice(self.pid_index[self.pids[kid]])
            i_cam = self.data_source[i][3]['domains']
            # _, i_pid, i_cam = self.data_source[i]
            ret.append(i)
            pid_i = self.index_pid[i]
            cams = self.pid_cam[pid_i]
            index = self.pid_index[pid_i]
            select_cams = no_index(cams, i_cam)

            if select_cams:
                if len(select_cams) >= self.num_instances:
                    cam_indexes = np.random.choice(select_cams, size=self.num_instances - 1, replace=False)
                else:
                    cam_indexes = np.random.choice(select_cams, size=self.num_instances - 1, replace=True)
                for kk in cam_indexes:
                    ret.append(index[kk])
            else:
                select_indexes = no_index(index, i)
                if not select_indexes:
                    # only one image for this identity
                    ind_indexes = [0] * (self.num_instances - 1)
                elif len(select_indexes) >= self.num_instances:
                    ind_indexes = np.random.choice(select_indexes, size=self.num_instances - 1, replace=False)
                else:
                    ind_indexes = np.random.choice(select_indexes, size=self.num_instances - 1, replace=True)

                for kk in ind_indexes:
                    ret.append(index[kk])

        return ret

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)

    def _infinite_indices(self):
        np.random.seed(self._seed)
        while True:
            indices = self._get_epoch_indices()
            yield from indices


class NaiveIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source: str, batch_size: int, num_instances: int, delete_rem: bool, seed: Optional[int] = None, cfg = None):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // self.num_instances
        self.delete_rem = delete_rem

        self.index_pid = defaultdict(list)
        self.pid_cam = defaultdict(list)
        self.pid_index = defaultdict(list)

        for index, info in enumerate(data_source):
            pid = info[1]
            camid = info[2]
            self.index_pid[index] = pid
            self.pid_cam[pid].append(camid)
            self.pid_index[pid].append(index)

        self.pids = list(self.pid_index.keys())
        self.num_identities = len(self.pids)

        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)

        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()


        val_pid_index = [len(x) for x in self.pid_index.values()]
        min_v = min(val_pid_index)
        max_v = max(val_pid_index)
        hist_pid_index = [val_pid_index.count(x) for x in range(min_v, max_v+1)]
        num_print = 5
        for i, x in enumerate(range(min_v, min_v+min(len(hist_pid_index), num_print))):
            print('dataset histogram [bin:{}, cnt:{}]'.format(x, hist_pid_index[i]))
        print('...')
        print('dataset histogram [bin:{}, cnt:{}]'.format(max_v, val_pid_index.count(max_v)))

        val_pid_index_upper = []
        for x in val_pid_index:
            v_remain = x % self.num_instances
            if v_remain == 0:
                val_pid_index_upper.append(x)
            else:
                if self.delete_rem:
                    if x < self.num_instances:
                        val_pid_index_upper.append(x - v_remain + self.num_instances)
                    else:
                        val_pid_index_upper.append(x - v_remain)
                else:
                    val_pid_index_upper.append(x - v_remain + self.num_instances)

        total_images = sum(val_pid_index_upper)
        total_images = total_images - (total_images % self.batch_size) - self.batch_size # approax
        self.total_images = total_images



    def _get_epoch_indices(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.pid_index[pid]) # whole index for each ID
            if self.delete_rem:
                if len(idxs) < self.num_instances: # if idxs is smaller than num_instance, choice redundantly
                    idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            else:
                if len(idxs) < self.num_instances: # if idxs is smaller than num_instance, choice redundantly
                    idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
                elif (len(idxs) % self.num_instances) != 0:
                    idxs.extend(np.random.choice(idxs, size=self.num_instances - len(idxs) % self.num_instances, replace=False))

            np.random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(int(idx))
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []
        # batch_idxs_dict: dictionary, len(batch_idxs_dict) is len(pidx), each pidx, num_instance x k samples
        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = np.random.choice(avai_pids, self.num_pids_per_batch, replace=False)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0: avai_pids.remove(pid)

        return final_idxs

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)

    def _infinite_indices(self):
        np.random.seed(self._seed)
        while True:
            indices = self._get_epoch_indices()
            yield from indices



class DomainSuffleSampler(Sampler):

    def __init__(self, data_source: str, batch_size: int, num_instances: int, delete_rem: bool, seed: Optional[int] = None, cfg = None):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // self.num_instances
        self.delete_rem = delete_rem

        self.index_pid = defaultdict(list)
        self.pid_domain = defaultdict(list)
        self.pid_index = defaultdict(list)

        for index, info in enumerate(data_source):

            domainid = info[3]['domains']
            if cfg.DATALOADER.CAMERA_TO_DOMAIN:
                pid = info[1] + str(domainid)
            else:
                pid = info[1]
            self.index_pid[index] = pid
            # self.pid_domain[pid].append(domainid)
            self.pid_domain[pid] = domainid
            self.pid_index[pid].append(index)

        self.pids = list(self.pid_index.keys())
        self.domains = list(self.pid_domain.values())

        self.num_identities = len(self.pids)
        self.num_domains = len(set(self.domains))

        self.batch_size //= self.num_domains
        self.num_pids_per_batch //= self.num_domains

        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)

        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()


        val_pid_index = [len(x) for x in self.pid_index.values()]
        min_v = min(val_pid_index)
        max_v = max(val_pid_index)
        hist_pid_index = [val_pid_index.count(x) for x in range(min_v, max_v+1)]
        num_print = 5
        for i, x in enumerate(range(min_v, min_v+min(len(hist_pid_index), num_print))):
            print('dataset histogram [bin:{}, cnt:{}]'.format(x, hist_pid_index[i]))
        print('...')
        print('dataset histogram [bin:{}, cnt:{}]'.format(max_v, val_pid_index.count(max_v)))

        val_pid_index_upper = []
        for x in val_pid_index:
            v_remain = x % self.num_instances
            if v_remain == 0:
                val_pid_index_upper.append(x)
            else:
                if self.delete_rem:
                    if x < self.num_instances:
                        val_pid_index_upper.append(x - v_remain + self.num_instances)
                    else:
                        val_pid_index_upper.append(x - v_remain)
                else:
                    val_pid_index_upper.append(x - v_remain + self.num_instances)

        cnt_domains = [0 for x in range(self.num_domains)]
        for val, index in zip(val_pid_index_upper, self.domains):
            cnt_domains[index] += val
        self.max_cnt_domains = max(cnt_domains)
        self.total_images = self.num_domains * (self.max_cnt_domains - (self.max_cnt_domains % self.batch_size) - self.batch_size)



    def _get_epoch_indices(self):


        def _get_batch_idxs(pids, pid_index, num_instances, delete_rem):
            batch_idxs_dict = defaultdict(list)
            for pid in pids:
                idxs = copy.deepcopy(pid_index[pid])
                if delete_rem:
                    if len(idxs) < self.num_instances: # if idxs is smaller than num_instance, choice redundantly
                        idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
                else:
                    if len(idxs) < self.num_instances: # if idxs is smaller than num_instance, choice redundantly
                        idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
                    elif (len(idxs) % self.num_instances) != 0:
                        idxs.extend(np.random.choice(idxs, size=self.num_instances - len(idxs) % self.num_instances, replace=False))

                np.random.shuffle(idxs)
                batch_idxs = []
                for idx in idxs:
                    batch_idxs.append(int(idx))
                    if len(batch_idxs) == num_instances:
                        batch_idxs_dict[pid].append(batch_idxs)
                        batch_idxs = []
            return batch_idxs_dict

        batch_idxs_dict = _get_batch_idxs(self.pids, self.pid_index, self.num_instances, self.delete_rem)

        # batch_idxs_dict: dictionary, len(batch_idxs_dict) is len(pidx), each pidx, num_instance x k samples
        avai_pids = copy.deepcopy(self.pids)

        local_avai_pids = \
            [[pids for pids, idx in zip(avai_pids, self.domains) if idx == i]
             for i in list(set(self.domains))]
        local_avai_pids_save = copy.deepcopy(local_avai_pids)


        revive_idx = [False for i in range(self.num_domains)]
        final_idxs = []
        while len(avai_pids) >= self.num_pids_per_batch and not all(revive_idx):
            for i in range(self.num_domains):
                selected_pids = np.random.choice(local_avai_pids[i], self.num_pids_per_batch, replace=False)
                for pid in selected_pids:
                    batch_idxs = batch_idxs_dict[pid].pop(0)
                    final_idxs.extend(batch_idxs)
                    if len(batch_idxs_dict[pid]) == 0:
                        avai_pids.remove(pid)
                        local_avai_pids[i].remove(pid)
            for i in range(self.num_domains):
                if len(local_avai_pids[i]) < self.num_pids_per_batch:
                    print('{} is recovered'.format(i))
                    batch_idxs_dict_new = _get_batch_idxs(self.pids, self.pid_index, self.num_instances, self.delete_rem)

                    revive_idx[i] = True
                    cnt = 0
                    for pid, val in batch_idxs_dict_new.items():
                        if self.domains[cnt] == i:
                            batch_idxs_dict[pid] = copy.deepcopy(batch_idxs_dict_new[pid])
                        cnt += 1
                    local_avai_pids[i] = copy.deepcopy(local_avai_pids_save[i])
                    avai_pids.extend(local_avai_pids_save[i])
                    avai_pids = list(set(avai_pids))
        return final_idxs

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)

    def _infinite_indices(self):
        np.random.seed(self._seed)
        while True:
            indices = self._get_epoch_indices()
            yield from indices
