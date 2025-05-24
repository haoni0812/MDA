# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""
from .utils import log_accuracy
from .domain_JSD_loss import domain_JSD_loss
from .domain_STD_loss import domain_STD_loss
from .domain_SCT_loss import domain_SCT_loss,domain_TEST_loss
from .domain_MMD_loss import domain_MMD_loss
from .cross_entropy_loss import cross_entropy_loss
from .focal_loss import focal_loss
from .triplet_loss import triplet_loss
from .circle_loss import circle_loss
from .myloss import Pedal, Ipfl
from .smooth import SmoothingForImage, PatchMemory
from .invariance import InvNet
from .contrastive import ViewContrastiveLoss
from .crossentropy import CrossEntropyLabelSmooth
