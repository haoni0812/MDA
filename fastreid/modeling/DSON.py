import torch
from torch.autograd import Function
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.distributed as dist
import torch.nn as nn




class OptimizedNorm2d_ch(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9, using_moving_average=True,
                 last_gamma=False, channelwise=True, modes=['in', 'bn']):
        super(OptimizedNorm2d_ch, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.last_gamma = last_gamma
        # self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        # self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.channelwise = channelwise
        self.num_features = num_features
        self.modes = modes

        num_norms = len(modes)
        if channelwise:
            self.mean_weight = nn.Parameter(torch.ones(num_norms, num_features))
            self.var_weight = nn.Parameter(torch.ones(num_norms, num_features))
        else:
            self.mean_weight = nn.Parameter(torch.ones(num_norms))
            self.var_weight = nn.Parameter(torch.ones(num_norms))

        # self.register_buffer('running_mean', torch.zeros(1, num_features, 1))
        # self.register_buffer('running_var', torch.zeros(1, num_features, 1))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.zeros(num_features))

        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.zero_()
        if self.last_gamma:
            self.weight.data.fill_(0)
        else:
            self.weight.data.fill_(1)
        self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def get_norm_ratios(self, mean=False):
        softmax = nn.Softmax(0)
        # mean_weight, var_weight = self.mean_weight.mean(1, keepdim = True), self.var_weight.mean(1, keepdim = True)
        mean_weight, var_weight = self.mean_weight, self.var_weight
        mean_weight, var_weight = softmax(mean_weight), softmax(var_weight)
        return mean_weight, var_weight

    def forward(self, input, opt = None):
        self._check_input_dim(input)
        N, C, H, W = input.size()
        x = input.view(N, C, -1)

        mean_in = x.mean(-1, keepdim=True)
        var_in = x.var(-1, keepdim=True)
        temp = var_in + mean_in ** 2

        if self.training:
            mean_bn = mean_in.mean(0, keepdim=True)
            var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2
            if self.using_moving_average:
                self.running_mean.mul_(self.momentum)
                self.running_mean.add_((1 - self.momentum) * mean_bn.view(-1).data)
                self.running_var.mul_(self.momentum)
                self.running_var.add_((1 - self.momentum) * var_bn.view(-1).data)
            else:
                self.running_mean.add_(mean_bn.view(-1).data)
                self.running_var.add_(mean_bn.view(-1).data ** 2 + var_bn.view(-1).data)
        else:
            mean_bn = torch.autograd.Variable(self.running_mean.view(1, C, 1))
            var_bn = torch.autograd.Variable(self.running_var.view(1, C, 1))

        mean_weight, var_weight = self.get_norm_ratios()

        mean_norms = {'in': mean_in, 'bn': mean_bn}
        var_norms = {'in': var_in, 'bn': var_bn}

        mean = sum([mean_norms[mode] * mw.view(1, len(mw), 1) for mode, mw in zip(self.modes, mean_weight)])
        var = sum([var_norms[mode] * mw.view(1, len(mw), 1) for mode, mw in zip(self.modes, var_weight)])

        x = (x - mean) / (var + self.eps).sqrt()
        x = x.view(N, C, H, W)
        return x * self.weight.view(1, C, 1, 1) + self.bias.view(1, C, 1, 1)




class OptimizedNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9, using_moving_average=True,
                 last_gamma=False, channelwise=False, modes=['in', 'bn']):
        super(OptimizedNorm2d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.last_gamma = last_gamma
        # self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        # self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.channelwise = channelwise
        self.num_features = num_features
        self.modes = modes

        num_norms = len(modes)
        if channelwise:
            self.mean_weight = nn.Parameter(torch.ones(num_norms, num_features))
            self.var_weight = nn.Parameter(torch.ones(num_norms, num_features))
        else:
            self.mean_weight = nn.Parameter(torch.ones(num_norms))
            self.var_weight = nn.Parameter(torch.ones(num_norms))

        # self.register_buffer('running_mean', torch.zeros(1, num_features, 1))
        # self.register_buffer('running_var', torch.zeros(1, num_features, 1))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.zeros(num_features))

        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.zero_()
        if self.last_gamma:
            self.weight.data.fill_(0)
        else:
            self.weight.data.fill_(1)
        self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def get_norm_ratios(self, mean=False):
        softmax = nn.Softmax(0)
        mean_weight, var_weight = self.mean_weight, self.var_weight
        mean_weight, var_weight = softmax(mean_weight), softmax(var_weight)
        return mean_weight, var_weight

    def forward(self, input, opt = None):
        self._check_input_dim(input)
        N, C, H, W = input.size()
        x = input.view(N, C, -1)

        mean_in = x.mean(-1, keepdim=True)
        var_in = x.var(-1, keepdim=True)
        temp = var_in + mean_in ** 2

        if self.training:
            mean_bn = mean_in.mean(0, keepdim=True)
            var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2
            if self.using_moving_average:
                self.running_mean.mul_(self.momentum)
                self.running_mean.add_((1 - self.momentum) * mean_bn.view(-1).data)
                self.running_var.mul_(self.momentum)
                self.running_var.add_((1 - self.momentum) * var_bn.view(-1).data)
            else:
                self.running_mean.add_(mean_bn.view(-1).data)
                self.running_var.add_(mean_bn.view(-1).data ** 2 + var_bn.view(-1).data)
        else:
            mean_bn = torch.autograd.Variable(self.running_mean.view(1, C, 1))
            var_bn = torch.autograd.Variable(self.running_var.view(1, C, 1))

        mean_weight, var_weight = self.get_norm_ratios()

        mean_norms = {'in': mean_in, 'bn': mean_bn}
        var_norms = {'in': var_in, 'bn': var_bn}

        mean = sum([mean_norms[mode] * mw for mode, mw in zip(self.modes, mean_weight)])
        var = sum([var_norms[mode] * mw for mode, mw in zip(self.modes, var_weight)])

        x = (x - mean) / (var + self.eps).sqrt()
        x = x.view(N, C, H, W)
        return x * self.weight.view(1, C, 1, 1) + self.bias.view(1, C, 1, 1)


class DomainSpecificOptimizedNorm2d(nn.Module):
    def __init__(self, num_features, num_domains, eps=1e-5, momentum=0.9, using_moving_average=True, using_bn=True,
                 last_gamma=False, module='OptimizedNorm2d'):

        super(DomainSpecificOptimizedNorm2d, self).__init__()

        self.bns = nn.ModuleList(
            [globals()[module](num_features, eps, momentum) for _ in range(num_domains)]
        )

    def reset_running_stats(self):
        for bn in self.bns:
            bn.reset_running_stats()

    def reset_parameters(self):
        for bn in self.bns:
            bn.reset_parameters()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, x, domain_label):
        self._check_input_dim(x)
        bn = self.bns[domain_label]
        return bn(x), domain_label





