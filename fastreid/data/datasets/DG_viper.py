# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import os
from glob import glob

from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset

__all__ = ['DG_viper', ]


@DATASET_REGISTRY.register()
class DG_VIPeR(ImageDataset):
    dataset_dir = "viper"
    dataset_name = "viper"

    def __init__(self, root='datasets', **kwargs):
        if isinstance(root, list):
            type = root[1]
            self.root = root[0]
        else:
            self.root = root
            type = 'split_1a'
        self.train_dir = os.path.join(self.root, self.dataset_dir, type, 'train')
        self.query_dir = os.path.join(self.root, self.dataset_dir, type, 'query')
        self.gallery_dir = os.path.join(self.root, self.dataset_dir, type, 'gallery')

        required_files = [
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]
        self.check_before_run(required_files)

        train = self.process_train(self.train_dir, is_train = True)
        query = self.process_train(self.query_dir, is_train = False)
        gallery = self.process_train(self.gallery_dir, is_train = False)

        self.train_per_cam, self.train_per_cam_sampled = self.reorganize_images_by_camera(train)
        self.query_per_cam, self.query_per_cam_sampled = self.reorganize_images_by_camera(query)
        self.gallery_per_cam, self.gallery_per_cam_sampled = self.reorganize_images_by_camera(gallery)


        super().__init__(train, query, gallery, **kwargs)

    def process_train(self, path, is_train = True):
        data = []
        img_list = glob(os.path.join(path, '*.png'))
        for img_path in img_list:
            img_name = img_path.split('/')[-1] # p000_c1_d045.png
            split_name = img_name.split('_')
            pid = int(split_name[0][1:])
            if is_train:
                pid = self.dataset_name + "_" + str(pid)
            camid = int(split_name[1][1:])
            # dirid = int(split_name[2][1:-4])
            data.append([img_path, pid, camid])

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