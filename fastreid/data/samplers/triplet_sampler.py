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
import random

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


class TTASampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    - subset_size (int): size of the fixed subset to use for an epoch.
    - num_cams (int): number of different camera IDs in each batch.
    - top_p (float): percentage of top identities to consider in each camera.
    """

    def __init__(self, data_source, batch_size, num_instances=10, subset_size=500, num_cams=8, top_p=20):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.subset_size = subset_size
        self.num_cams = num_cams
        self.top_p = top_p
        self.num_pids_per_cam = self.batch_size // (self.num_instances * self.num_cams)
        self.index_dic = defaultdict(list)
        self.pid_mapping = {}
        self.total_pids = 0

        # 根据camid和pid建立索引
        for index, info in enumerate(self.data_source):
            pid = info["targets"]
            camid = info["camid"]
            self.index_dic[(camid, pid)].append(index)
        self.unique_pid = True
        if self.unique_pid:
            self.cams_pids = self.select_top_pids_across_cams(self.index_dic, self.top_p*self.num_cams)
        else:
            self.cams_pids = self.select_top_p_ids_per_cam(self.index_dic, self.top_p)
            
        
        for camid, pid in self.cams_pids:
            if pid not in self.pid_mapping:
                self.pid_mapping[pid] = self.total_pids
                self.total_pids += 1
        # import ipdb; ipdb.set_trace()


        # 估计使用的数据总量
        self.length = 0
        for camid, pid in self.cams_pids:
            idxs = self.index_dic[(camid, pid)]
            num = len(idxs)
            self.length += num

    def select_top_p_ids_per_cam(self, index_dic, top_p):
        # 创建一个字典 cams_pids_count 用于存储每个摄像头对应的身份样本数量
        cams_pids_count = defaultdict(lambda: defaultdict(int))

        # 遍历数据集中的每个样本，计算每个摄像头和身份对应的样本数量
        for (camid, pid), indices in index_dic.items():
            cams_pids_count[camid][pid] += len(indices)

        # 对 cams_pids_count 中的每个摄像头，按照身份样本数量从大到小排序，并取前 top_p 个
        top_p_ids_per_cam = {camid: dict(sorted(pids_count.items(), key=lambda item: item[1], reverse=True)[:top_p])
                            for camid, pids_count in cams_pids_count.items()}
        
        allow_same_pid = False
        # 处理 top_p_ids_per_cam，确保每个 pid 只在一个 camid 中出现
        selected_pids = set()
        for camid, pids_count in top_p_ids_per_cam.items():
            pids = list(pids_count.keys())
            # import ipdb; ipdb.set_trace()
            if not allow_same_pid:
                new_pids = list(set(pids) - selected_pids)  # 排除已经选择的 pid
                replacement_pids = [pid for pid in cams_pids_count[camid] if pid not in list(set(pids)| set(selected_pids))]
                replacement_pids.sort(key=lambda pid: cams_pids_count[camid][pid], reverse=True)
                pids = new_pids + replacement_pids[:len(pids) - len(new_pids)]  # 用新的 pid 替换重复的 pi
                selected_pids.update(new_pids)  # 将已选择的 pid 添加到集合中
            random.shuffle(pids)  # 随机打乱顺序
            top_p_ids_per_cam[camid] = dict(zip(pids, list(pids_count.values())))
        # 返回一个列表，其中包含每个摄像头对应的前 top_p 个样本最多的身份
        return [(camid, pid) for camid, pids_count in top_p_ids_per_cam.items() for pid, _ in pids_count.items()]


    def select_top_pids_across_cams(self, index_dic, topk):
        # 创建一个字典，用于存储每个pid在所有camids中的计数
        pid_count_across_cams = defaultdict(int)

        # 创建一个字典，用于存储每个pid在所有camids中的样本计数
        pid_sample_count_across_cams = defaultdict(int)

        # 遍历给定的索引字典
        for (camid, pid), indices in index_dic.items():
            pid_count_across_cams[pid] += 1
            pid_sample_count_across_cams[pid] += len(indices)

        #import ipdb; ipdb.set_trace()
        # 根据它们出现的camid数量进行排序，如果相同，则根据样本计数进行排序
        sorted_pids = sorted(pid_count_across_cams.keys(), key=lambda pid: (pid_count_across_cams[pid], pid_sample_count_across_cams[pid]), reverse=True)

        # 选择topk个在所有camids中具有最高计数的pid
        selected_pids = sorted_pids[:topk]

        # 创建一个(camid, pid)对的列表，其中pid为所选pid
        result = [(camid, pid) for (camid, pid) in index_dic.keys() if pid in selected_pids]

        return result
    
    def __iter__(self):
        batch_idxs_dict = defaultdict(list)
        avai_cams_pids = copy.deepcopy(self.cams_pids)
        random_camid = True
        # import ipdb; ipdb.set_trace()
        final_idxs = []
        while len(avai_cams_pids) >= self.num_cams:
            if self.unique_pid and random_camid :
                pid_candidates = list(set(pid for camid, pid in avai_cams_pids))
                selected_pid = random.choice(pid_candidates)
                selected_cams_pids = [(camid, selected_pid) for camid, pid in avai_cams_pids if pid == selected_pid]
            else:
                if random_camid:
                    selected_cams_pids = random.sample(avai_cams_pids, self.num_cams)
                else:
                    # 选择相同 camid 的身份标签组合
                    camid_candidates = list(set(camid for camid, _ in avai_cams_pids))
                    selected_camid = random.choice(camid_candidates)
                    selected_cams_pids = [(selected_camid, pid) for _, pid in avai_cams_pids if _ == selected_camid][:self.num_cams]


            for camid, pid in selected_cams_pids:
                idxs = copy.deepcopy(self.index_dic[(camid, pid)])
                # import ipdb; ipdb.set_trace() 
                if len(idxs) >= self.num_instances:
                    idxs = np.random.choice(idxs, size=self.num_instances, replace=False)
                else:
                    idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
                random.shuffle(idxs)
                batch_idxs_dict[(camid, pid)] = idxs
                final_idxs.extend(idxs)
                avai_cams_pids.remove((camid, pid))  # 移除已经使用的摄像头和身份标签组合

        # import ipdb; ipdb.set_trace()
        """ for _, pid in self.cams_pids:
            for camid in range(self.num_cams):
                for _ in range(self.num_pids_per_cam):
                    if (camid, pid) in batch_idxs_dict:
                        batch_idxs = batch_idxs_dict[(camid, pid)][:self.num_instances]
                        final_idxs.extend(batch_idxs)
                        batch_idxs_dict[(camid, pid)] = batch_idxs_dict[(camid, pid)][self.num_instances:] """

        return iter(final_idxs)

    def __len__(self):
        return self.length