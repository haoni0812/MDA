from __future__ import division, absolute_import
import torch.utils.model_zoo as model_zoo
from torch import nn
from torch.nn import functional as F
import torch
import logging
logger = logging.getLogger(__name__)
from .build import BACKBONE_REGISTRY
from fastreid.utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from fastreid.modeling.ops import meta_conv2d, meta_norm
from collections import OrderedDict
import copy

class ConvBlock(nn.Module):
    """Basic convolutional block.

    convolution (bias discarded) + batch normalization + relu6.
    Args:
        in_c (int): number of input channels.
        out_c (int): number of output channels.
        k (int or tuple): kernel size.
        s (int or tuple): stride.
        p (int or tuple): padding.
        g (int): number of blocked connections from input channels
            to output channels (default: 1).
    """

    def __init__(self, in_c, out_c, k, s=1, p=0, g=1, bn_norm = 'BN', norm_opt = None):
        super(ConvBlock, self).__init__()
        # self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p, bias=False, groups=g)
        self.conv = meta_conv2d(in_c, out_c, k, stride=s, padding=p, bias=False, groups=g)
        self.bn = meta_norm(bn_norm, out_c, norm_opt = norm_opt)

    def forward(self, x, opt = None):
        x = self.conv(x, opt)
        x = self.bn(x, opt)
        x = F.relu6(x)

        return x


class Bottleneck(nn.Module):

    def __init__(self, in_channels, out_channels, expansion_factor, stride=1, bn_norm = 'BN', norm_opt = None):
        super(Bottleneck, self).__init__()
        mid_channels = in_channels * expansion_factor
        self.use_residual = stride == 1 and in_channels == out_channels
        self.conv1 = ConvBlock(in_channels, mid_channels, 1, bn_norm = bn_norm, norm_opt = norm_opt)
        self.dwconv2 = ConvBlock(mid_channels, mid_channels, 3, s=stride, p=1, g=mid_channels, bn_norm = bn_norm,  norm_opt= norm_opt)
        # self.conv3 = nn.Sequential(
        #     meta_conv2d(mid_channels, out_channels, 1, bias=False),
        #     meta_norm(bn_norm, out_channels, norm_opt = norm_opt),
        # )
        self.conv3 = meta_conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn = meta_norm(bn_norm, out_channels, norm_opt = norm_opt)
        # print(self.use_residual)

    def forward(self, x, opt = None):
        m = self.conv1(x, opt) # conv block
        m = self.dwconv2(m, opt) # conv block
        # m = self.conv3[0](m, opt) # conv
        # m = self.conv3[1](m, opt) # norm
        m = self.conv3(m, opt) # conv
        m = self.bn(m, opt) # norm
        if self.use_residual:
            return x + m
        else:
            return m


class MobileNetV2(nn.Module):
    """MobileNetV2.
    Reference:
        Sandler et al. MobileNetV2: Inverted Residuals and
        Linear Bottlenecks. CVPR 2018.
    Public keys:
        - ``mobilenetv2_x1_0``: MobileNetV2 x1.0.
        - ``mobilenetv2_x1_4``: MobileNetV2 x1.4.
    """

    def __init__(self, width_mult=1, bn_norm = 'BN', norm_opt = None, last_stride = 2, **kwargs):
        super(MobileNetV2, self).__init__()

        self.in_channels = int(32 * width_mult)
        self.feature_dim = int(1280 * width_mult) if width_mult > 1 else 1280

        # construct layers
        if bn_norm == 'DualNorm':
            self.layer1 = ConvBlock(3, self.in_channels, 3, s=2, p=1, bn_norm = 'IN', norm_opt = norm_opt) # IN <- dualnorm
            self.layer2 = self._make_layer(Bottleneck, 1, int(16 * width_mult), 1, 1, bn_norm = 'IN', norm_opt = norm_opt) # IN <- dualnorm
            self.layer3 = self._make_layer(Bottleneck, 6, int(24 * width_mult), 2, 2, bn_norm = 'IN', norm_opt = norm_opt) # IN <- dualnorm
            self.layer4 = self._make_layer(Bottleneck, 6, int(32 * width_mult), 3, 2, bn_norm = 'IN', norm_opt = norm_opt) # IN <- dualnorm
            self.layer5 = self._make_layer(Bottleneck, 6, int(64 * width_mult), 4, 2, bn_norm = 'IN', norm_opt = norm_opt) # IN <- dualnorm
            self.layer6 = self._make_layer(Bottleneck, 6, int(96 * width_mult), 3, 1, bn_norm = 'IN', norm_opt = norm_opt) # IN <- dualnorm
            self.layer7 = self._make_layer(Bottleneck, 6, int(160 * width_mult), 3, last_stride, bn_norm = 'BN', norm_opt = norm_opt)
            self.layer8 = self._make_layer(Bottleneck, 6, int(320 * width_mult), 1, 1, bn_norm = 'BN', norm_opt = norm_opt)
            self.layer9 = ConvBlock(self.in_channels, self.feature_dim, 1, bn_norm = 'BN', norm_opt = norm_opt)
        else:
            self.layer1 = ConvBlock(3, self.in_channels, 3, s=2, p=1, bn_norm = bn_norm, norm_opt = norm_opt) # IN <- dualnorm
            self.layer2 = self._make_layer(Bottleneck, 1, int(16 * width_mult), 1, 1, bn_norm = bn_norm, norm_opt = norm_opt) # IN <- dualnorm
            self.layer3 = self._make_layer(Bottleneck, 6, int(24 * width_mult), 2, 2, bn_norm = bn_norm, norm_opt = norm_opt) # IN <- dualnorm
            self.layer4 = self._make_layer(Bottleneck, 6, int(32 * width_mult), 3, 2, bn_norm = bn_norm, norm_opt = norm_opt) # IN <- dualnorm
            self.layer5 = self._make_layer(Bottleneck, 6, int(64 * width_mult), 4, 2, bn_norm = bn_norm, norm_opt = norm_opt) # IN <- dualnorm
            self.layer6 = self._make_layer(Bottleneck, 6, int(96 * width_mult), 3, 1, bn_norm = bn_norm, norm_opt = norm_opt) # IN <- dualnorm
            self.layer7 = self._make_layer(Bottleneck, 6, int(160 * width_mult), 3, last_stride, bn_norm = bn_norm, norm_opt = norm_opt)
            self.layer8 = self._make_layer(Bottleneck, 6, int(320 * width_mult), 1, 1, bn_norm = bn_norm, norm_opt = norm_opt)
            self.layer9 = ConvBlock(self.in_channels, self.feature_dim, 1, bn_norm = bn_norm, norm_opt = norm_opt)

    def _make_layer(self, block, t, c, n, s, bn_norm = 'BN', norm_opt = None):
        # t: expansion factor
        # c: output channels
        # n: number of blocks
        # s: stride for first layer
        layers = []
        layers.append(block(self.in_channels, c, t, s, bn_norm = bn_norm, norm_opt=norm_opt))
        self.in_channels = c
        for i in range(1, n):
            layers.append(block(self.in_channels, c, t, bn_norm = bn_norm, norm_opt=norm_opt))
        return nn.Sequential(*layers)

    def forward(self, x, opt = None):
        x = self.layer1(x, opt) # [b,   3, 256, 128] -> [b, 32, 128,  64]

        x = self.layer2[0](x, opt) # [b,  16, 128,  64]

        x = self.layer3[0](x, opt) # [b,  24,  64,  32]
        x = self.layer3[1](x, opt) # [b,  24,  64,  32]

        x = self.layer4[0](x, opt) # [b,  32,  32,  16]
        x = self.layer4[1](x, opt) # [b,  32,  32,  16]
        x = self.layer4[2](x, opt) # [b,  32,  32,  16]

        x = self.layer5[0](x, opt) # [b,  64,  16,   8]
        x = self.layer5[1](x, opt) # [b,  64,  16,   8]
        x = self.layer5[2](x, opt) # [b,  64,  16,   8]
        x = self.layer5[3](x, opt) # [b,  64,  16,   8]

        x = self.layer6[0](x, opt) # [b,  96,  16,   8]
        x = self.layer6[1](x, opt) # [b,  96,  16,   8]
        x = self.layer6[2](x, opt) # [b,  96,  16,   8]

        x = self.layer7[0](x, opt) # [b, 160,   8,   4]
        x = self.layer7[1](x, opt) # [b, 160,   8,   4]
        x = self.layer7[2](x, opt) # [b, 160,   8,   4]

        x = self.layer8[0](x, opt) # [b, 320,   8,   4]

        x = self.layer9(x, opt) # [b,1280,   8,   4]

        return x

# def init_pretrained_weights(model, model_url):
#     """Initializes model with pretrained weights.
#
#     Layers that don't match with pretrained layers in name or size are kept unchanged.
#     """
#     pretrain_dict = model_zoo.load_url(model_url)
#     model_dict = model.state_dict()
#     pretrain_dict = {
#         k: v
#         for k, v in pretrain_dict.items()
#         if k in model_dict and model_dict[k].size() == v.size()
#     }
#     model_dict.update(pretrain_dict)
#     model.load_state_dict(model_dict)

@BACKBONE_REGISTRY.register()
def build_mobilenet_v2_backbone(cfg):

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
    # num_splits = cfg.MODEL.BACKBONE.NORM_SPLIT
    # with_ibn = cfg.MODEL.BACKBONE.WITH_IBN
    # with_se = cfg.MODEL.BACKBONE.WITH_SE
    # with_nl = cfg.MODEL.BACKBONE.WITH_NL
    depth = cfg.MODEL.BACKBONE.DEPTH / 10.0

    model = MobileNetV2(
        width_mult = depth,
        bn_norm = bn_norm,
        norm_opt = norm_opt,
        last_stride = last_stride
    )

    # model_urls = {
    #     # 1.0: top-1 71.3
    #     'mobilenetv2_x1_0':
    #         'https://mega.nz/#!NKp2wAIA!1NH1pbNzY_M2hVk_hdsxNM1NUOWvvGPHhaNr-fASF6c',
    #     # 1.4: top-1 73.9
    #     'mobilenetv2_x1_4':
    #         'https://mega.nz/#!RGhgEIwS!xN2s2ZdyqI6vQ3EwgmRXLEW3khr9tpXg96G9SUJugGk',
    # }

    if pretrain and pretrain_path is not "":
        requires_dict = OrderedDict()
        for name, values in model.named_parameters():
            requires_dict[name] = copy.copy(values.requires_grad)
        pretrained_dict = torch.load(pretrain_path, map_location=torch.device('cpu'))
        state_dict_new = OrderedDict()
        for name, values in pretrained_dict.copy().items():
            name_split = name.split('.')
            if 'conv' in name_split[0]:
                name_split[0] = name_split[0].replace('conv', 'layer')
                name = '.'.join(name_split)
            state_dict_new[name] = copy.copy(values) # change conv -> layer (to compatibility with resnet's name)



        state_dict = OrderedDict()
        for name, values in state_dict_new.copy().items():
            # conv3.0.~~ -> conv3.~~
            if 'conv3.0' in name:
                name = name.replace('conv3.0', 'conv3')
            # conv3.1.~~ -> bn.~~
            elif 'conv3.1' in name:
                name = name.replace('conv3.1', 'bn')
            state_dict[name] = values


        if cfg.MODEL.NORM.TYPE_BACKBONE == 'BIN_gate2':
            for name, values in state_dict.copy().items():
                if 'bn' in name:
                    if ('weight' in name) or ('bias' in name):
                        # bn.weight, bn.bias -> bn.bat_n.weight, bn.bat_n.bias
                        if cfg.MODEL.NORM.LOAD_BN_AFFINE:
                            new_name = name.replace('bn', 'bn.bat_n')
                            state_dict[new_name] = values
                        # bn.weight, bn.bias -> bn.ins_n.weight, bn.ins_n.bias
                        if cfg.MODEL.NORM.LOAD_IN_AFFINE:
                            new_name = name.replace('bn', 'bn.ins_n')
                            state_dict[new_name] = values
                        del state_dict[name]
                    elif ('running_mean' in name) or ('running_var' in name):
                        # bn.running_mean, bn.running_var -> bn.bat_n.running_mean, bn.bat_n.running_var
                        if cfg.MODEL.NORM.LOAD_BN_RUNNING:
                            new_name = name.replace('bn', 'bn.bat_n')
                            state_dict[new_name] = values
                        # bn.running_mean, bn.running_var -> bn.ins_n.running_mean, bn.ins_n.running_var
                        if cfg.MODEL.NORM.LOAD_IN_RUNNING:
                            new_name = name.replace('bn', 'bn.ins_n')
                            state_dict[new_name] = values
                        del state_dict[name]

        else:
            if not cfg.MODEL.NORM.LOAD_BN_AFFINE:
                for name, param in state_dict.copy().items():
                    if ('bn' in name) or ('norm' in name):
                        if ('weight' in name) or ('bias' in name):
                            del state_dict[name]
                            print(name)
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

        if not cfg.MODEL.BACKBONE.NUM_BATCH_TRACKED:
            for name, values in state_dict.copy().items():
                if 'num_batches_tracked' in name:
                    del state_dict[name]

        for name, values in requires_dict.copy().items():
            if name in state_dict:
                state_dict[name].requires_grad = copy.copy(requires_dict[name])
                # print(requires_dict[name])

        if cfg.MODEL.NORM.TYPE_BACKBONE == 'DualNorm':
            for name, values in state_dict.copy().items():
                if ('bn' in name):
                    if ('layer1' in name) or ('layer2' in name) or ('layer3' in name) or \
                            ('layer4' in name) or ('layer5' in name) or ('layer6' in name):
                        del state_dict[name]

        incompatible = model.load_state_dict(state_dict, strict=False)

        # if cfg.MODEL.BACKBONE.NUM_BATCH_TRACKED:


        # for name, values in model.named_parameters():
        #     values.requires_grad = requires_dict[name]


        if incompatible.missing_keys:
            logger.info(
                get_missing_parameters_message(incompatible.missing_keys)
            )
        if incompatible.unexpected_keys:
            logger.info(
                get_unexpected_parameters_message(incompatible.unexpected_keys)
            )

        # if depth == 1.0:
            # init_pretrained_weights(model, model_urls['mobilenetv2_x1_0'])
            # import warnings
            # warnings.warn('The imagenet pretrained weights need to be manually downloaded from {}'
            #         .format(model_urls['mobilenetv2_x1_0']))

        # elif depth == 1.4:
            # init_pretrained_weights(model, model_urls['mobilenetv2_x1_4'])
            # import warnings
            # warnings.warn('The imagenet pretrained weights need to be manually downloaded from {}'
            #         .format(model_urls['mobilenetv2_x1_4']))

        # try:
        # state_dict = torch.load(pretrain_path, map_location=torch.device('cpu'))['model']
        # # Remove module.encoder in name
        # new_state_dict = {}
        # for k in state_dict:
        #     new_k = '.'.join(k.split('.')[2:])
        #     if new_k in model.state_dict() and (model.state_dict()[new_k].shape == state_dict[k].shape):
        #         new_state_dict[new_k] = state_dict[k]
        # state_dict = new_state_dict
        # logger.info(f"Loading pretrained model from {pretrain_path}")
    # except FileNotFoundError or KeyError:
    #     # original resnet
    #     state_dict = model_zoo.load_url(model_urls[depth])
    #     logger.info("Loading pretrained model from torchvision")


    return model
