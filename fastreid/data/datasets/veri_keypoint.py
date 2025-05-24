# encoding: utf-8
"""
@author:  Jinkai Zheng
@contact: 1315673509@qq.com
"""

import glob
import os.path as osp
import re

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class VeRi_keypoint(ImageDataset):
    """VeRi.

    Reference:
        Liu et al. A Deep Learning based Approach for Progressive Vehicle Re-Identification. ECCV 2016.

    URL: `<https://vehiclereid.github.io/VeRi/>`_

    Dataset statistics:
        - identities: 775.
        - images: 37778 (train) + 1678 (query) + 11579 (gallery).
    """
    dataset_dir = "veri"
    dataset_name = "veri"

    def __init__(self, root='datasets', **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.keypoint_dir = osp.join(root, 'veri_keypoint')

        self.train_dir = osp.join(self.dataset_dir, 'image_train')
        self.query_dir = osp.join(self.dataset_dir, 'image_query')
        self.gallery_dir = osp.join(self.dataset_dir, 'image_test')

        required_files = [
            self.dataset_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
            self.keypoint_dir
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir)
        train = self.process_keypoint(self.keypoint_dir, train)
        query = self.process_dir(self.query_dir, is_train=False)
        gallery = self.process_dir(self.gallery_dir, is_train=False)

        super(VeRi_keypoint, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, is_train=True):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([\d]+)_c(\d\d\d)')

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 1 <= pid <= 776
            assert 1 <= camid <= 20
            camid -= 1  # index starts from 0
            if is_train:
                pid = self.dataset_name + "_" + str(pid)
            data.append((img_path, pid, camid))


        return data


    def process_keypoint(self, dir_path, data):
        train_name = []
        train_raw = []
        train_keypoint = []
        train_orientation = []
        is_keypoint = False
        is_orientation = True
        is_aligned = False
        with open(osp.join(dir_path, 'keypoint_train_aligned.txt')) as f:
            for line in f:
                train_raw.append(line)
                line_split = line.split(' ')
                train_name.append(line_split[0].split('/')[-1])

                if is_keypoint:
                    train_keypoint.append(line_split[1:41])
                if is_orientation:
                    tmp = line_split[-1]
                    if '\n' in tmp:
                        tmp = tmp[0]
                    assert 0 <= int(tmp) <= 7 # orientation should be 0~7
                    train_orientation.append(int(tmp))

        if is_aligned:
            train_name = sorted(tuple(train_name))
            train_raw = sorted(tuple(train_raw))

            with open(osp.join(dir_path, 'keypoint_train_aligned.txt'), 'w') as f:
                for i, x in enumerate(data):
                    j = 0
                    flag_break = False
                    while (j < len(train_name) and not flag_break):
                        if train_name[j] in x[0]:
                            if train_name[j] in train_raw[j]:
                                f.write(train_raw[j])
                                flag_break = True
                                del train_name[j]
                                del train_raw[j]
                                print(i)
                            else:
                                assert()
                        j += 1


        for i, x in enumerate(data):
            j = 0
            flag_break = False
            while(j < len(train_name) and not flag_break):
                if train_name[j] in x[0]:
                    add_info = {} # dictionary
                    add_info['domains'] = int(train_orientation[j])
                    data[i] = list(data[i])
                    data[i].append(add_info)
                    data[i] = tuple(data[i])
                    flag_break = True
                    del train_name[j]
                    del train_orientation[j]
                    # print(i)
                j += 1

        cnt = 0
        no_title = []
        no_title_local = []
        for line in data:
            if len(line) != 4:
                assert()
                # no_title.append(line[0])
                # tmp1 = line[0].split('/')[-1]
                # tmp2 = tmp1.split('_')
                # tmp3 = '_'.join(tmp2[2:])
                # for line2 in train_name:
                #     if tmp3 in line2:
                #         print(line2)
                # no_title_local.append(tmp3)
                # cnt += 1

        return data