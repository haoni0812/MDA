# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import glob
import os.path as osp
import re

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class MyDukeMTMC(ImageDataset):
    """DukeMTMC-reID.

    Reference:
        - Ristani et al. Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking. ECCVW 2016.
        - Zheng et al. Unlabeled Samples Generated by GAN Improve the Person Re-identification Baseline in vitro. ICCV 2017.

    URL: `<https://github.com/layumi/DukeMTMC-reID_evaluation>`_

    Dataset statistics:
        - identities: 1404 (train + query).
        - images:16522 (train) + 2228 (query) + 17661 (gallery).
        - cameras: 8.
    """
    dataset_dir = 'DukeMTMC-reID'
    dataset_url = 'http://vision.cs.duke.edu/DukeMTMC/data/misc/DukeMTMC-reID.zip'
    dataset_name = "dukemtmc"

    def __init__(self, root='datasets', **kwargs):
        # self.root = osp.abspath(osp.expanduser(root))
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        required_files = [
            self.dataset_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]
        self.check_before_run(required_files)

        query = self.process_dir(self.query_dir)
        gallery = self.process_dir(self.gallery_dir)
        train = []
        train.extend(query)
        train.extend(gallery)

        num_bn_sample = 200

        self.train_per_cam, self.train_per_cam_sampled = self.reorganize_images_by_camera(train,
                                                                                          num_bn_sample)
        self.query_per_cam, self.query_per_cam_sampled = self.reorganize_images_by_camera(query,
                                                                                          num_bn_sample)
        self.gallery_per_cam, self.gallery_per_cam_sampled = self.reorganize_images_by_camera(gallery,
                                                                                              num_bn_sample)

        super(MyDukeMTMC, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, is_train=True):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            assert 1 <= camid <= 8
            camid -= 1  # index starts from 0
            if is_train:
                pid = self.dataset_name + "_" + str(pid)
            data.append((img_path, pid, camid))

        return data

    def reorganize_images_by_camera(self, data, sample_per_camera=200):
        import numpy as np
        from collections import defaultdict
        cams = np.unique([x[2] for x in data])
        images_per_cam = defaultdict(list)
        images_per_cam_sampled = defaultdict(list)
        for cam_id in cams:
            all_file_info = [x for x in data if x[2] == cam_id]
            all_file_info = sorted(all_file_info, key=lambda x: x[0])
            import random
            random.shuffle(all_file_info)
            images_per_cam[cam_id] = all_file_info
            images_per_cam_sampled[cam_id] = all_file_info[:min(sample_per_camera, len(all_file_info))]

        return images_per_cam, images_per_cam_sampled
