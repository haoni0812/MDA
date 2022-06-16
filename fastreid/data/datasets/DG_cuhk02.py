# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import os
from glob import glob

from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset

__all__ = ['DG_CUHK02', ]


@DATASET_REGISTRY.register()
class DG_CUHK02(ImageDataset):
    dataset_dir = "cuhk02"
    dataset_name = "cuhk02"

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.train_path = os.path.join(self.root, self.dataset_dir)

        required_files = [self.train_path]
        self.check_before_run(required_files)

        train = self.process_train(self.train_path)

        super().__init__(train, [], [], **kwargs)

    def process_train(self, train_path):


        cam_split = True

        data = []
        file_path = os.listdir(train_path)
        for pid_dir in file_path:
            img_file = os.path.join(train_path, pid_dir)
            cam1_folder = os.path.join(img_file, 'cam1')
            cam = '1'

            # if os.path.join(img_file, 'cam1'):
            img_paths = glob(os.path.join(cam1_folder, "*.png"))
            for img_path in img_paths:
                split_path = img_path.split('/')[-1].split('_')
                pid = self.dataset_name + "_" + pid_dir + "_" + split_path[0]
                camid = int(cam)
                # if cam_split:
                #     camid = self.dataset_name + "_" + pid_dir + "_" + cam
                # else:
                #     camid = self.dataset_name + "_" + cam
                data.append([img_path, pid, camid])

            cam2_folder = os.path.join(img_file, 'cam2')
            cam = '2'

            img_paths = glob(os.path.join(cam2_folder, "*.png"))
            for img_path in img_paths:
                split_path = img_path.split('/')[-1].split('_')
                pid = self.dataset_name + "_" + pid_dir + "_" + split_path[0]
                camid = int(cam)
                # if cam_split:
                #     camid = self.dataset_name + "_" + pid_dir + "_" + cam
                # else:
                #     camid = self.dataset_name + "_" + cam
                data.append([img_path, pid, camid])
        return data
