# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch.utils.data import Dataset

from .data_utils import read_image


class CommDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, img_items, transform=None, relabel=True):
        self.img_items = img_items
        self.transform = transform
        self.relabel = relabel

        self.pid_dict = {}
        if self.relabel:
            pids = list()
            if img_items != None:
                for i, item in enumerate(img_items):
                    if item[1] in pids: continue
                    pids.append(item[1])
                self.pids = pids
                self.pid_dict = dict([(p, i) for i, p in enumerate(self.pids)])

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        if len(self.img_items[index]) > 3:
            img_path, pid, camid, others = self.img_items[index]
        else:
            img_path, pid, camid = self.img_items[index]
            others = ''
        img = read_image(img_path)
        if self.transform is not None: img = self.transform(img)
        if self.relabel: pid = self.pid_dict[pid]
        return {
            "images": img,
            "targets": pid,
            "camid": camid,
            "img_path": img_path,
            # "others": others
            "others": {"domains": camid}
        }


    def merge_datasets(self, dataset):
        assert isinstance(dataset, CommDataset), "Input dataset must be an instance of CommDataset"
        #assert len(self.pids) == len(dataset.pids), "Datasets must have the same number of classes"

        merged_img_items = self.img_items + dataset.img_items

        return CommDataset(merged_img_items, transform=self.transform, relabel=self.relabel)


    @property
    def num_classes(self):
        return len(self.pids)
    
    

