# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import logging
import math

from fastreid.modeling.ops import meta_conv2d, meta_norm
import copy
import torch
from torch import nn
from torch.utils import model_zoo
from .efdmix import EFDMix
from .mixstyle import MixStyle


from fastreid.layers import (
    IBN,
    SELayer,
    Non_local,
)
from fastreid.utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from .build import BACKBONE_REGISTRY

logger = logging.getLogger(__name__)
model_urls = {
    18: 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    34: 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    50: 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    101: 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    152: 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, bn_norm, norm_opt, num_splits, with_ibn=False, with_se=False,
                 stride=1, downsample=None, reduction=16):
        super(BasicBlock, self).__init__()
        self.conv1 = meta_conv2d(inplanes, planes, kernel_size = 3, stride=stride, padding=1, bias=False)
        self.bn1 = meta_norm(bn_norm, planes, norm_opt=norm_opt)
        self.conv2 = meta_conv2d(planes, planes, kernel_size = 3, stride=1, padding=1, bias=False)
        self.bn2 = meta_norm(bn_norm, planes, norm_opt=norm_opt)
        self.relu = nn.ReLU(inplace=True)
        if with_se:
            self.se = SELayer(planes, reduction)
        else:
            self.se = nn.Identity()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, opt = None):
        identity = x

        out = self.conv1(x, opt)
        out = self.bn1(out, opt)
        out = self.relu(out)

        out = self.conv2(out, opt)
        out = self.bn2(out, opt)

        if self.downsample is not None:
            identity = self.downsample(x, opt)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, bn_norm, norm_opt, num_splits, with_ibn=False, with_se=False,
                 stride=1, downsample=None, reduction=16):
        super(Bottleneck, self).__init__()
        self.conv1 = meta_conv2d(inplanes, planes, kernel_size=1, bias=False)
        if with_ibn:
            self.bn1 = IBN(planes, bn_norm, num_splits)
        else:
            self.bn1 = meta_norm(bn_norm, planes, norm_opt = norm_opt)
        self.conv2 = meta_conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = meta_norm(bn_norm, planes, norm_opt = norm_opt)
        self.conv3 = meta_conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = meta_norm(bn_norm, planes * self.expansion, norm_opt = norm_opt)
        self.relu = nn.ReLU(inplace=True)
        if with_se:
            self.se = SELayer(planes * self.expansion, reduction)
        else:
            self.se = nn.Identity()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, opt = None):
        residual = x

        out = self.conv1(x, opt)
        out = self.bn1(out, opt)
        out = self.relu(out)

        out = self.conv2(out, opt)
        out = self.bn2(out, opt)
        out = self.relu(out)

        out = self.conv3(out, opt)
        out = self.bn3(out, opt)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x, opt)

        out += residual
        out = self.relu(out)

        return out

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias, bn_norm, norm_opt, bn_out_channels, num_splits):
        super(Downsample, self).__init__()
        # self = nn.Sequential()
        # self.downsample = nn.Module
        self.conv = meta_conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=bias)
        self.bn = meta_norm(bn_norm, bn_out_channels, norm_opt =norm_opt)

        # self.downsample = nn.Sequential(
        #     meta_conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=bias),
        #     get_norm(norm, bn_out_channels, num_splits),
        # )
    def forward(self, x, opt = None):
        x = self.conv(x, opt)
        out = self.bn(x, opt)
        return out



class ResNet(nn.Module):
    def __init__(self, last_stride, bn_norm, norm_opt, num_splits, with_ibn, with_se, with_nl, block, layers, non_layers):
        self.inplanes = 64
        super().__init__()
        self.conv1 = meta_conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = meta_norm(bn_norm, 64, norm_opt = norm_opt)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], 1, bn_norm, norm_opt, num_splits, with_ibn, with_se)
        self.layer2 = self._make_layer(block, 128, layers[1], 2, bn_norm, norm_opt, num_splits, with_ibn, with_se)
        self.layer3 = self._make_layer(block, 256, layers[2], 2, bn_norm, norm_opt, num_splits, with_ibn, with_se)
        self.layer4 = self._make_layer(block, 512, layers[3], last_stride, bn_norm, norm_opt, num_splits, with_se=with_se)

        efdmix_layers = [] #['layer1', 'layer2']
        if efdmix_layers:
            self.efdmix = EFDMix(p=0.5, alpha=0.1, mix='random')
            print('Insert EFDMix after the following layers: {}'.format(efdmix_layers))
        self.efdmix_layers = efdmix_layers

        self.mixstyle = None
        mixstyle_layers = []#'layer1', 'layer2']
        if mixstyle_layers:
            self.mixstyle = MixStyle(p=0.5, alpha=0.1, mix='random')
            print('Insert MixStyle after the following layers: {}'.format(mixstyle_layers))
        self.mixstyle_layers = mixstyle_layers

        self.random_init()

        if with_nl:
            self._build_nonlocal(layers, non_layers, bn_norm, norm_opt, num_splits)
        else:
            self.NL_1_idx = self.NL_2_idx = self.NL_3_idx = self.NL_4_idx = []

    def _make_layer(self, block, planes, blocks, stride=1, bn_norm="BN", norm_opt = None, num_splits=1, with_ibn=False, with_se=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Downsample(self.inplanes, planes * block.expansion,
                                    1, stride, False, bn_norm, norm_opt, planes * block.expansion, num_splits)
            # downsample = ds.downsample
            #     downsample = nn.Sequential(
            #         meta_conv2d(self.inplanes, planes * block.expansion,
            #                   kernel_size=1, stride=stride, bias=False),
            #         get_norm(bn_norm, planes * block.expansion, num_splits),
            #     )

        layers = []
        if planes == 512:
            with_ibn = False
        layers.append(block(self.inplanes, planes, bn_norm, norm_opt, num_splits, with_ibn, with_se, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, bn_norm, norm_opt, num_splits, with_ibn, with_se))

        return nn.Sequential(*layers)

    def _build_nonlocal(self, layers, non_layers, bn_norm, norm_opt, num_splits):
        self.NL_1 = nn.ModuleList(
            [Non_local(256, bn_norm, norm_opt, num_splits) for _ in range(non_layers[0])])
        self.NL_1_idx = sorted([layers[0] - (i + 1) for i in range(non_layers[0])])
        self.NL_2 = nn.ModuleList(
            [Non_local(512, bn_norm, norm_opt, num_splits) for _ in range(non_layers[1])])
        self.NL_2_idx = sorted([layers[1] - (i + 1) for i in range(non_layers[1])])
        self.NL_3 = nn.ModuleList(
            [Non_local(1024, bn_norm, norm_opt, num_splits) for _ in range(non_layers[2])])
        self.NL_3_idx = sorted([layers[2] - (i + 1) for i in range(non_layers[2])])
        self.NL_4 = nn.ModuleList(
            [Non_local(2048, bn_norm, norm_opt, num_splits) for _ in range(non_layers[3])])
        self.NL_4_idx = sorted([layers[3] - (i + 1) for i in range(non_layers[3])])

    def forward(self, x, opt = None):
        x = self.conv1(x, opt)
        x = self.bn1(x, opt)
        x = self.relu(x)
        x = self.maxpool(x)

        NL1_counter = 0

        if len(self.NL_1_idx) == 0:
            self.NL_1_idx = [-1]
        for i in range(len(self.layer1)):
            x = self.layer1[i](x, opt)
            if i == self.NL_1_idx[NL1_counter]:
                _, C, H, W = x.shape
                x = self.NL_1[NL1_counter](x)
                NL1_counter += 1
        if (x.size(0)==96 or x.size(0)==48) and 'layer1' in self.mixstyle_layers:
            assert False
            x = self.efdmix(x)
        
        

        # Layer 2
        NL2_counter = 0
        if len(self.NL_2_idx) == 0:
            self.NL_2_idx = [-1]
        for i in range(len(self.layer2)):
            x = self.layer2[i](x, opt)
            if i == self.NL_2_idx[NL2_counter]:
                _, C, H, W = x.shape
                x = self.NL_2[NL2_counter](x)
                NL2_counter += 1
        
        if (x.size(0)==96 or x.size(0)==48) and 'layer2' in self.mixstyle_layers:
            x = self.efdmix(x)
            

        
        # Layer 3
        NL3_counter = 0
        if len(self.NL_3_idx) == 0:
            self.NL_3_idx = [-1]
        for i in range(len(self.layer3)):
            x = self.layer3[i](x, opt)
            if i == self.NL_3_idx[NL3_counter]:
                _, C, H, W = x.shape
                x = self.NL_3[NL3_counter](x)
                NL3_counter += 1

        if 'layer3' in self.efdmix_layers:
            x = self.efdmix(x)

        # Layer 4
        NL4_counter = 0
        if len(self.NL_4_idx) == 0:
            self.NL_4_idx = [-1]
        for i in range(len(self.layer4)):
            x = self.layer4[i](x, opt)
            if i == self.NL_4_idx[NL4_counter]:
                _, C, H, W = x.shape
                x = self.NL_4[NL4_counter](x)
                NL4_counter += 1

        return x

    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, 0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


@BACKBONE_REGISTRY.register()
def build_resnet_backbone(cfg):
    """
    Create a ResNet instance from config.
    Returns:
        ResNet: a :class:`ResNet` instance.
    """

    # fmt: off
    pretrain = cfg.MODEL.BACKBONE.PRETRAIN
    pretrain_path = cfg.MODEL.BACKBONE.PRETRAIN_PATH
    last_stride = cfg.MODEL.BACKBONE.LAST_STRIDE
    bn_norm = cfg.MODEL.NORM.TYPE_BACKBONE
    norm_opt = dict()
    norm_opt['BN_AFFINE'] = cfg.MODEL.NORM.BN_AFFINE
    norm_opt['BN_RUNNING'] = cfg.MODEL.NORM.BN_RUNNING
    norm_opt['IN_AFFINE'] = cfg.MODEL.NORM.IN_AFFINE
    norm_opt['IN_RUNNING'] = cfg.MODEL.NORM.IN_RUNNING

    norm_opt['BN_W_FREEZE'] = cfg.MODEL.NORM.BN_W_FREEZE
    norm_opt['BN_B_FREEZE'] = cfg.MODEL.NORM.BN_B_FREEZE
    norm_opt['IN_W_FREEZE'] = cfg.MODEL.NORM.IN_W_FREEZE
    norm_opt['IN_B_FREEZE'] = cfg.MODEL.NORM.IN_B_FREEZE

    norm_opt['BIN_INIT'] = cfg.MODEL.NORM.BIN_INIT
    norm_opt['IN_FC_MULTIPLY'] = cfg.MODEL.NORM.IN_FC_MULTIPLY
    num_splits = cfg.MODEL.BACKBONE.NORM_SPLIT
    with_ibn = cfg.MODEL.BACKBONE.WITH_IBN
    with_se = cfg.MODEL.BACKBONE.WITH_SE
    with_nl = cfg.MODEL.BACKBONE.WITH_NL
    depth = cfg.MODEL.BACKBONE.DEPTH

    num_blocks_per_stage = {18: [2,2,2,2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3], }[depth]
    nl_layers_per_stage = {18: [0,0,0,0], 34: [0, 2, 3, 0], 50: [0, 2, 3, 0], 101: [0, 2, 9, 0], 152: [0, 2, 9, 0]}[depth]
    block = {18: BasicBlock, 34: BasicBlock, 50: Bottleneck, 101: Bottleneck, 152:Bottleneck,}[depth]
    model = ResNet(last_stride, bn_norm, norm_opt, num_splits, with_ibn, with_se, with_nl, block,
                   num_blocks_per_stage, nl_layers_per_stage)
    if pretrain:
        if not with_ibn:
            try:
                state_dict = torch.load(pretrain_path, map_location=torch.device('cpu'))['model']
                # Remove module.encoder in name
                new_state_dict = {}
                for k in state_dict:
                    new_k = '.'.join(k.split('.')[2:])
                    if new_k in model.state_dict() and (model.state_dict()[new_k].shape == state_dict[k].shape):
                        new_state_dict[new_k] = state_dict[k]
                state_dict = new_state_dict
                logger.info(f"Loading pretrained model from {pretrain_path}")
            except FileNotFoundError or KeyError:
                # original resnet
                state_dict = model_zoo.load_url(model_urls[depth])
                logger.info("Loading pretrained model from torchvision")
        else:
            state_dict = torch.load(pretrain_path, map_location=torch.device('cpu'))['state_dict']  # ibn-net
            # Remove module in name
            new_state_dict = {}
            for k in state_dict:
                new_k = '.'.join(k.split('.')[1:])
                if new_k in model.state_dict() and (model.state_dict()[new_k].shape == state_dict[k].shape):
                    new_state_dict[new_k] = state_dict[k]
            state_dict = new_state_dict
            logger.info(f"Loading pretrained model from {pretrain_path}")

        for name, param in state_dict.copy().items():

            if 'downsample' in name:  # layer1.0.downsample.0.weight
                new_name = name.split('.')
                if new_name[-2] == '0':
                    new_name[-2] = 'conv'
                elif new_name[-2] == '1':
                    new_name[-2] = 'bn'
                new_name = '.'.join(new_name)
                state_dict[new_name] = copy.copy(state_dict[name])
                del state_dict[name]


        if cfg.MODEL.NORM.TYPE_BACKBONE == 'BIN_gate2':
            for name, values in state_dict.copy().items():
                if 'bn' in name:
                    if ('weight' in name) or ('bias' in name):
                        # bn.weight, bn.bias -> bn.bat_n.weight, bn.bat_n.bias
                        if cfg.MODEL.NORM.LOAD_BN_AFFINE:
                            split_name = name.split('.')
                            for i, local_name in enumerate(split_name):
                                if 'bn' in local_name:
                                    split_name.insert(i + 1, 'bat_n')
                                    break
                            new_name = '.'.join(split_name)
                            state_dict[new_name] = values
                        # bn.weight, bn.bias -> bn.ins_n.weight, bn.ins_n.bias
                        if cfg.MODEL.NORM.LOAD_IN_AFFINE:
                            split_name = name.split('.')
                            for i, local_name in enumerate(split_name):
                                if 'bn' in local_name:
                                    split_name.insert(i + 1, 'ins_n')
                                    break
                            new_name = '.'.join(split_name)
                            state_dict[new_name] = values
                        del state_dict[name]
                    elif ('running_mean' in name) or ('running_var' in name):
                        # bn.running_mean, bn.running_var -> bn.bat_n.running_mean, bn.bat_n.running_var
                        if cfg.MODEL.NORM.LOAD_BN_RUNNING:
                            split_name = name.split('.')
                            for i, local_name in enumerate(split_name):
                                if 'bn' in local_name:
                                    split_name.insert(i + 1, 'bat_n')
                                    break
                            new_name = '.'.join(split_name)
                            state_dict[new_name] = values
                        # bn.running_mean, bn.running_var -> bn.ins_n.running_mean, bn.ins_n.running_var
                        if cfg.MODEL.NORM.LOAD_IN_RUNNING:
                            split_name = name.split('.')
                            for i, local_name in enumerate(split_name):
                                if 'bn' in local_name:
                                    split_name.insert(i + 1, 'ins_n')
                                    break
                            new_name = '.'.join(split_name)
                            state_dict[new_name] = values
                        del state_dict[name]

        else:
            if not cfg.MODEL.NORM.LOAD_BN_AFFINE:
                for name, param in state_dict.copy().items():
                    if ('bn' in name) or ('norm' in name):
                        if ('weight' in name) or ('bias' in name):
                            del state_dict[name]
            if not cfg.MODEL.NORM.LOAD_BN_RUNNING:
                for name, param in state_dict.copy().items():
                    if ('bn' in name) or ('norm' in name):
                        if ('running_mean' in name) or ('running_var' in name):
                            del state_dict[name]
            if not cfg.MODEL.NORM.IN_RUNNING and cfg.MODEL.NORM.TYPE_BACKBONE == "IN":
                for name, param in state_dict.copy().items():
                    if ('bn' in name) or ('norm' in name):
                        if ('running_mean' in name) or ('running_var' in name):
                            del state_dict[name]

        incompatible = model.load_state_dict(state_dict, strict=False)


        if incompatible.missing_keys:
            logger.info(
                get_missing_parameters_message(incompatible.missing_keys)
            )
        if incompatible.unexpected_keys:
            logger.info(
                get_unexpected_parameters_message(incompatible.unexpected_keys)
            )
    return model
