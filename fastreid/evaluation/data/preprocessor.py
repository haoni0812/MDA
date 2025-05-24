from __future__ import absolute_import
import os
import os.path as osp
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import math
from PIL import Image


class Preprocessor(Dataset):
    def __init__(self, dataset, root=None, transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, fname, pid, camid, index
    
class Preprocessor_mutual(Dataset):
    def __init__(self, dataset, root=None, transform=None, mutual=False, transform_weak=None):
        super(Preprocessor_mutual, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.transform_weak = transform_weak
        self.mutual = mutual

        # self.use_gan=use_gan
        # self.num_cam = num_cam

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if self.mutual:
            return self._get_mutual_item(indices)
        else:
            return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, fname, pid, camid

    def _get_mutual_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')
        img_mutual = img.copy()
        img2 = img.copy()

        if self.transform is not None:
            img1 = self.transform(img)
            img_mutual = self.transform(img_mutual)
            if self.transform_weak is not None:
                img2 = self.transform_weak(img2)
            else:
                img2 = self.transform(img2)
        else:
            raise NotImplementedError

        return img1, img2, img_mutual, pid, camid
