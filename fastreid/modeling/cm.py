import collections
import numpy as np
from abc import ABC
import torch
import torch.nn.functional as F
from torch import nn, autograd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from fastreid.modeling.losses import *

class CM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y in zip(inputs, targets):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def cm(inputs, indexes, features, momentum=0.5):
    return CM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class CM_Hard(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)

        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            median = np.argmin(np.array(distances))
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features[median]
            ctx.features[index] /= ctx.features[index].norm()

        return grad_inputs, None, None, None


def cm_hard(inputs, indexes, features, momentum=0.5):
    return CM_Hard.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class ClusterMemory(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, use_hard=False):
        super(ClusterMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp
        self.use_hard = use_hard
        self.epoch = 0

        self.register_buffer('features', torch.zeros(num_samples, num_features))

    def forward(self, inputs, targets, cfg, prototype=None, batch=0, domain_labels=None):
        

        inputs = F.normalize(inputs, dim=1)
        self.use_hard = False
        if self.use_hard:
            outputs = cm_hard(inputs, targets, self.features, self.momentum)
        else:
            outputs = cm(inputs, targets, self.features, self.momentum)
        if batch == 1:
            self.plot_tensors(outputs, prototype, self.epoch)
            self.epoch += 1
        outputs /= self.temp
        prototype /= self.temp
        #prototype.requires_grad = False
        targets_all = targets + prototype.size(1)
        outputs = torch.cat([prototype, outputs], dim=1)
        #import pdb; pdb.set_trace()
        loss = cross_entropy_loss(
                outputs,
                targets_all,
                cfg.MODEL.LOSSES.CE.EPSILON,
                cfg.MODEL.LOSSES.CE.ALPHA,
                test_time = False,
            )
        MMDloss = False
        if MMDloss:
            loss_mmd = domain_MMD_loss(
                inputs if cfg.MODEL.LOSSES.MMD.FEAT_ORDER == 'before' else bn_features,
                domain_labels,
                cfg.MODEL.LOSSES.MMD.NORM,
                cfg.MODEL.LOSSES.MMD.NORM_FLAG,
                cfg.MODEL.LOSSES.MMD.KERNEL_MUL,
                cfg.MODEL.LOSSES.MMD.KERNEL_NUM,
                cfg.MODEL.LOSSES.MMD.FIX_SIGMA,
            ) * cfg.MODEL.LOSSES.MMD.SCALE
        #loss = F.cross_entropy(outputs, targets_all )
        return loss




    def plot_tensors(self, a, b, epoch):
        # 将a和b拼接为一个N*(M+K)的矩阵
        concatenated_tensor = torch.cat((a, b), dim=1)

        # 取出第一行数据
        first_row = concatenated_tensor[0]

        # 获取列数坐标
        x = list(range(1, concatenated_tensor.shape[1] + 1))

        # 获取值
        y = first_row.tolist()

        # 绘制柱状图
        plt.bar(x, y, color=['blue'] * a.shape[1] + ['red'] * b.shape[1])
        plt.xlabel('Column Index')
        plt.ylabel('Value')
        plt.title('single sample epoch{}'.format(epoch))
        plt.savefig('/home/nihao/CVPR_extension/MDA/logs/Sample/M-resnet/nometa_and_mixstyle/single_sample_epoch{}.png'.format(epoch))
        plt.show()

        # 将N行数据按列相加得到一行
        sum_row = torch.sum(concatenated_tensor, dim=0)

        # 获取值
        y_sum = sum_row.tolist()

        # 绘制柱状图
        plt.bar(x, y_sum, color=['green'] * concatenated_tensor.shape[1])
        plt.xlabel('Column Index')
        plt.ylabel('Sum')
        plt.title('Sum of batch epoch{}'.format(epoch))
        plt.savefig('/home/nihao/CVPR_extension/MDA/logs/Sample/M-resnet/nometa_and_mixstyle/sum_of_batch_epoch{}.png'.format(epoch))
        plt.show()
        plt.clf()

    # 示例数据
    """ a = torch.tensor([[1, 2], [3, 4], [5, 6]])
    b = torch.tensor([[7, 8], [9, 10], [11, 12]])

    plot_tensors(a, b) """