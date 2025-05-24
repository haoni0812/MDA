from collections import defaultdict
from matplotlib import markers
from numpy.core.defchararray import count
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import torch

import numpy as np

from fastreid.data import build_reid_test_loader, build_my_reid_test_loader
from fastreid.engine import DefaultTrainer

class TsneViewer:
  def __init__(self, model, cfg, use_adaIN=False, dataset_name="Market1501", filename="market1501.png") -> None:
    self.model = model
    self.cfg = cfg
    self.use_adaIN = use_adaIN
    self.dataset_name = dataset_name
    self.features = []
    self.targets = []
    self.domains = []
    self.save_dir = "logs/tsne"
    if not os.path.exists(self.save_dir):
      os.makedirs(self.save_dir)
    self.save_path = os.path.join(self.save_dir, filename)

  def add(self, feature, target, domain):
    self.features.append(feature.cpu())
    self.targets.append(target)
    self.domains.append(domain)

  def select_pid(self, num=10):
    counter = {}
    for pid in self.targets:
      if pid not in counter:
        counter[pid] = 0
      counter[pid] += 1
    counter = sorted(counter.items(), key=lambda kv:(kv[1], kv[0]), reverse=True)
    selected_pid = [k for k, v in counter[1:num+1]]
    return selected_pid
  
  def generate_marker(self, tsne_targets):
    markers = ["1", "o", "v", "H", "+", "X", "D", "*", "s", "h"]
    s = set(tsne_targets)
    def get_idx(item):
      for idx, pid in enumerate(s):
        if item == pid:
          return idx
    return {pid:markers[get_idx(pid)%len(markers)] for pid in tsne_targets}

  def run(self,):
    if self.use_adaIN:
        temp_trainer = DefaultTrainer(self.cfg)
        opt = temp_trainer.opt_setting("basic")
        data_loader, _ = build_my_reid_test_loader(self.cfg, dataset_name=self.dataset_name)
        # collect bn information in a single camera
        network_bns = [x for x in list(self.model.modules()) if
                    isinstance(x, torch.nn.BatchNorm2d) or isinstance(x, torch.nn.BatchNorm1d)]
        for simple_loader, all_loader in zip(data_loader['simple'], data_loader['all']):
          for bn in network_bns:
              bn.momentum = 1
              bn.running_mean = torch.zeros(bn.running_mean.size()).float().to(self.cfg.MODEL.DEVICE)
              bn.running_var = torch.ones(bn.running_var.size()).float().to(self.cfg.MODEL.DEVICE)
              bn.num_batches_tracked = torch.tensor(0).to(self.cfg.MODEL.DEVICE).long()

          self.model.train()
          with torch.no_grad():
              for batch_idx, inputs in enumerate(simple_loader):
                  output = self.model(inputs, opt)
          self.model.eval() 
          with torch.no_grad():
            for idx, inputs in enumerate(all_loader):
              outs = self.model(inputs)
              self.add(outs, inputs['targets'], inputs['camid'])
    else:
      data_loader, _ = build_reid_test_loader(self.cfg, dataset_name=self.dataset_name)
      with torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            # print(inputs['images'].shape)
            outs = self.model(inputs)
            self.add(outs, inputs['targets'], inputs['camid'])

  def process(self):
    self.model.eval().to(self.cfg.MODEL.DEVICE)
    self.run()
          
    self.features = np.concatenate(self.features)
    self.targets = np.concatenate(self.targets)
    self.domains = np.concatenate(self.domains)

    # 选择样本数最多的10个pid作为tsne的展示数据
    # pid_list = list(set(self.targets))
    # np.random.shuffle(pid_list)
    # selected_pid = pid_list[:10]
    # print(selected_pid)

    selected_pid = self.select_pid()
    tsne_features = []
    tsne_targets = []
    tsne_domains = []
    for idx, pid in enumerate(self.targets):
        if pid in selected_pid:
            tsne_features.append(self.features[idx])
            tsne_targets.append(self.targets[idx])
            tsne_domains.append(self.domains[idx])
    tsne_features = np.stack(tsne_features)
    tsne_targets = np.array(tsne_targets)
    tsne_domains = np.array(tsne_domains)
    tsne_features = TSNE(n_components=2, random_state=501).fit_transform(tsne_features)

    markers_dict = self.generate_marker(tsne_targets)
    pid_dict = defaultdict(list)
    for idx, pid in enumerate(tsne_targets):
      pid_dict[pid].append(idx)

    # 不同形状不同pid，不同颜色不同摄像头
    plt.figure(figsize=(10, 10))

    for pid, idx_list in pid_dict.items():
      plt.scatter(tsne_features[idx_list, 0], tsne_features[idx_list, 1], c=tsne_domains[idx_list], s=30, marker=markers_dict[pid])
    plt.legend()
    plt.savefig(self.save_path, dpi=150)
    plt.show()

    print("The tsne image has been save at", self.save_path)