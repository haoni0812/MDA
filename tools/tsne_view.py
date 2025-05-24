import torch
import numpy as np
import os

from fastreid.utils.checkpoint import Checkpointer
from fastreid.utils.file_io import PathManager
from fastreid.engine import DefaultTrainer


# 加载模型
model = DefaultTrainer.build_model()
output_dir = "logs/Sample/D-resnet"
save_file = os.path.join(output_dir, "last_checkpoint")
with PathManager.open(save_file, "r") as f:
    last_saved = f.read().strip()
path = os.path.join(output_dir, last_saved)
Checkpointer(model).load(path)
