# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import os
from scipy.io import loadmat
from glob import glob
import random
from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset
import pdb

__all__ = ['DG_PRID', ]


@DATASET_REGISTRY.register()
class DG_PRID(ImageDataset):
    dataset_dir = "prid_2011"
    dataset_name = 'prid'
    _junk_pids = list(range(201, 750))

    def __init__(self, root='datasets', split_id=0, **kwargs):

        if isinstance(root, list):
            split_id = root[1]
            self.root = root[0]
        else:
            self.root = root
            split_id = 0
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        # self.download_dataset(self.dataset_dir, self.dataset_url)

        self.cam_a_dir = os.path.join(
            self.dataset_dir, 'single_shot', 'cam_a'
        )
        self.cam_b_dir = os.path.join(
            self.dataset_dir, 'single_shot', 'cam_b'
        )
        self.split_path = os.path.join(self.dataset_dir, 'splits_single_shot.json')

        required_files = [
            self.dataset_dir,
            self.cam_a_dir,
            self.cam_b_dir
        ]
        self.check_before_run(required_files)

        self.prepare_split()
        splits = self.read_json(self.split_path)
        if split_id >= len(splits):
            raise ValueError(
                'split_id exceeds range, received {}, but expected between 0 and {}'
                    .format(split_id,
                            len(splits) - 1)
            )
        split = splits[split_id]

        train, query, gallery = self.process_split(split)
        self.train_per_cam, self.train_per_cam_sampled = self.reorganize_images_by_camera(train)
        self.query_per_cam, self.query_per_cam_sampled = self.reorganize_images_by_camera(query)
        self.gallery_per_cam, self.gallery_per_cam_sampled = self.reorganize_images_by_camera(gallery)


        super(DG_PRID, self).__init__(train, query, gallery, **kwargs)

    def prepare_split(self):
        if not os.path.exists(self.split_path):
            print('Creating splits ...')

            splits = []
            for _ in range(10):
                # randomly sample 100 IDs for train and use the rest 100 IDs for test
                # (note: there are only 200 IDs appearing in both views)
                pids = [i for i in range(1, 201)]
                train_pids = random.sample(pids, 100)
                train_pids.sort()
                test_pids = [i for i in pids if i not in train_pids]
                split = {'train': train_pids, 'test': test_pids}
                splits.append(split)

            print('Totally {} splits are created'.format(len(splits)))
            self.write_json(splits, self.split_path)
            print('Split file is saved to {}'.format(self.split_path))

    def process_split(self, split):
        train_pids = split['train']
        test_pids = split['test']

        train_pid2label = {pid: label for label, pid in enumerate(train_pids)}

        # train
        train = []
        for pid in train_pids:
            img_name = 'person_' + str(pid).zfill(4) + '.png'
            pid = train_pid2label[pid]
            img_a_path = os.path.join(self.cam_a_dir, img_name)
            train.append((img_a_path, pid, 0))
            img_b_path = os.path.join(self.cam_b_dir, img_name)
            train.append((img_b_path, pid, 1))

        # query and gallery
        query, gallery = [], []
        for pid in test_pids:
            img_name = 'person_' + str(pid).zfill(4) + '.png'
            img_a_path = os.path.join(self.cam_a_dir, img_name)
            query.append((img_a_path, pid, 0))
            img_b_path = os.path.join(self.cam_b_dir, img_name)
            gallery.append((img_b_path, pid, 1))
        for pid in range(201, 750):
            img_name = 'person_' + str(pid).zfill(4) + '.png'
            img_b_path = os.path.join(self.cam_b_dir, img_name)
            gallery.append((img_b_path, pid, 1))

        return train, query, gallery

    def read_json(self, fpath):
        import json
        """Reads json file from a path."""
        with open(fpath, 'r') as f:
            obj = json.load(f)
        return obj

    def write_json(self, obj, fpath):
        import json
        """Writes to a json file."""
        self.mkdir_if_missing(os.path.dirname(fpath))
        with open(fpath, 'w') as f:
            json.dump(obj, f, indent=4, separators=(',', ': '))

    def mkdir_if_missing(self, dirname):
        import errno
        """Creates dirname if it is missing."""
        if not os.path.exists(dirname):
            try:
                os.makedirs(dirname)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

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