
import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
import torch
from torch.nn.parameter import Parameter
import time
import math
import copy

from .switchable_norm import *
from .syncsn_layer import SyncSwitchableNorm2d
from .syncbn_layer import SyncBatchNorm2d
from .DSON import OptimizedNorm2d, OptimizedNorm2d_ch
from .task_norm import TaskNormI

# class meta_linear(nn.Linear):
#     def __init__(self, in_feat, reduction_dim, bias = False):
#         super().__init__(in_feat, reduction_dim, bias = bias)
#     def forward(self, inputs, opt = None):
#         return F.linear(inputs, self.weight, self.bias)
#
# class meta_conv2d(nn.Conv2d):
#     def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, groups = 1, bias = True, padding_mode = 'zeros'):
#         super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
#     def forward(self, inputs, opt = None):
#         return F.conv2d(inputs, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
#
# class Meta_bn_norm(nn.BatchNorm2d):
#     def __init__(self, num_features, norm_opt = None, eps=1e-05, momentum=0.1, affine=True,
#                  track_running_stats=True, weight_freeze = False, bias_freeze = False,
#                  weight_init = 1.0, bias_init = 0.0):
#         affine = True if norm_opt['BN_AFFINE'] else False
#         track_running_stats = True if norm_opt['BN_RUNNING'] else False
#         super().__init__(num_features, eps, momentum, affine, track_running_stats)
#         if weight_init is not None: self.weight.data.fill_(weight_init)
#         if bias_init is not None: self.bias.data.fill_(bias_init)
#         self.weight.requires_grad_(not weight_freeze)
#         self.bias.requires_grad_(not bias_freeze)
#     def forward(self, inputs, opt = None):
#         if inputs.dim() != 4:
#             raise ValueError('expected 4D input (got {}D input)'.format(inputs.dim()))
#         return F.batch_norm(inputs, self.running_mean, self.running_var,
#                             self.weight, self.bias,
#                             self.training, self.momentum, self.eps)
#
# class Meta_in_norm(nn.InstanceNorm2d):
#     def __init__(self, num_features, norm_opt = None, eps=1e-05, momentum=0.1, affine=False,
#                  track_running_stats=False, weight_freeze = False, bias_freeze = False,
#                  weight_init = 1.0, bias_init = 0.0):
#
#         affine = True if norm_opt['IN_AFFINE'] else False
#         track_running_stats = True if norm_opt['IN_RUNNING'] else False
#         super().__init__(num_features, eps, momentum, affine, track_running_stats)
#
#         if self.weight is not None:
#             if weight_init is not None: self.weight.data.fill_(weight_init)
#             self.weight.requires_grad_(not weight_freeze)
#         if self.bias is not None:
#             if bias_init is not None: self.bias.data.fill_(bias_init)
#             self.bias.requires_grad_(not bias_freeze)
#         self.use_input_stats = True
#     def forward(self, inputs, opt = None):
#         if inputs.dim() != 4:
#             raise ValueError('expected 4D input (got {}D input)'.format(inputs.dim()))
#         return F.instance_norm(inputs, self.running_mean, self.running_var,
#                                self.weight, self.bias,
#                                self.use_input_stats, self.momentum, self.eps)


# -----------------------------------------
def update_parameter(param, step_size, opt=None, reserve=False):
    flag_update = False
    if step_size is not None:
        if param is not None:
            if opt['grad_params'][0] == None:
                if not reserve:
                    del opt['grad_params'][0]
                updated_param = param
            else:
                updated_param = param - step_size * opt['grad_params'][0]
                if not reserve:
                    del opt['grad_params'][0]
            flag_update = True
    if not flag_update:
        return param

    return updated_param

class MetaParam(nn.Module):
    def __init__(self, feat_dim, num_classes):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, inputs, opt=None):
        if opt == None: # and opt['meta']:
            updated_centers = update_parameter(self.centers, self.w_step_size, opt)
            batch_size = inputs.size(0)
            num_classes = self.centers.size(0)
            distmat = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(batch_size, num_classes) + \
                        torch.pow(updated_centers, 2).sum(dim=1, keepdim=True).expand(num_classes, batch_size).t()
            distmat.addmm_(1, -2, inputs, updated_centers.t())

            return distmat

        else:
            batch_size = inputs.size(0)
            num_classes = self.centers.size(0)
            distmat = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(batch_size, num_classes) + \
                        torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(num_classes, batch_size).t()
            distmat.addmm_(1, -2, inputs.float(), self.centers.t())
            

            return distmat

class meta_linear(nn.Linear):
    def __init__(self, in_feat, reduction_dim, bias = False):
        super().__init__(in_feat, reduction_dim, bias = bias)
    def forward(self, inputs, opt = None):
        if opt != None:
            use_meta_learning = False
            if opt['param_update']:
                if self.weight is not None:
                    if self.compute_meta_params:
                        use_meta_learning = True
        else:
            use_meta_learning = False
        if use_meta_learning:

            # start = time.perf_counter()
            updated_weight = update_parameter(self.weight, self.w_step_size, opt)
            updated_bias = update_parameter(self.bias, self.b_step_size, opt)
            # print('meta_linear is computed')
            # print(time.perf_counter() - start)

            return F.linear(inputs, updated_weight, updated_bias)
        else:
            return F.linear(inputs, self.weight, self.bias)
class meta_conv2d(nn.Conv2d):
    # def __init__(self, weight, bias, stride = 1, padding = 0, dilation = 1, groups = 1):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, groups = 1, bias = True, padding_mode = 'zeros'):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
    def forward(self, inputs, opt = None):
        if opt != None:
            use_meta_learning = False
            if opt['param_update']:
                if self.weight is not None:
                    if self.compute_meta_params:
                        use_meta_learning = True
        else:
            use_meta_learning = False
        if use_meta_learning:
            updated_weight = update_parameter(self.weight, self.w_step_size, opt)
            updated_bias = update_parameter(self.bias, self.b_step_size, opt)
            # print('meta_conv is computed')
            return F.conv2d(inputs, updated_weight, updated_bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            return F.conv2d(inputs, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
class Meta_bn_norm(nn.BatchNorm2d):
    def __init__(self, num_features, norm_opt = None, eps=1e-05,
                 momentum=0.1, weight_freeze = False, bias_freeze = False,
                 weight_init = 1.0, bias_init = 0.0):

        if not weight_freeze:
            weight_freeze = norm_opt['BN_W_FREEZE']
        if not bias_freeze:
            bias_freeze = norm_opt['BN_B_FREEZE']

        affine = True if norm_opt['BN_AFFINE'] else False
        track_running_stats = True if norm_opt['BN_RUNNING'] else False
        super().__init__(num_features, eps, momentum, affine, track_running_stats)

        if weight_init is not None: self.weight.data.fill_(weight_init)
        if bias_init is not None: self.bias.data.fill_(bias_init)
        self.weight.requires_grad_(not weight_freeze)
        self.bias.requires_grad_(not bias_freeze)


    def forward(self, inputs, opt = None):
        if inputs.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(inputs.dim()))
        if opt != None:
            use_meta_learning = False
            if opt['param_update']:
                if self.weight is not None:
                    if self.compute_meta_params:
                        use_meta_learning = True
        else:
            use_meta_learning = False

        if self.training:
            norm_type = opt['type_running_stats']
        else:
            norm_type = "eval"

        if use_meta_learning and self.affine:
            # if opt['zero_grad']: self.zero_grad()
            updated_weight = update_parameter(self.weight, self.w_step_size, opt)
            updated_bias = update_parameter(self.bias, self.b_step_size, opt)
            # print('meta_bn is computed')
        else:
            updated_weight = self.weight
            updated_bias = self.bias

        if opt == None:
            compute_each_batch = False
        else:
            try:
                if opt['each_domain']:
                    compute_each_batch = True
                else:
                    compute_each_batch = False
            except: # if opt['each_domain'] does not exist
                compute_each_batch = False
        if norm_type == "eval":
            compute_each_batch = False
        
        if compute_each_batch:
            domain_idx = opt['domains']
            unique_domain_idx = [int(x) for x in torch.unique(domain_idx).cpu()]
            cnt = 0
            for j in unique_domain_idx:
                t_logical_domain = domain_idx == j

                if norm_type == "general":  # update, but not apply running_mean/var
                    result_local = F.batch_norm(inputs[t_logical_domain], self.running_mean, self.running_var,
                                          updated_weight, updated_bias,
                                          self.training, self.momentum, self.eps)
                elif norm_type == "hold":  # not update, not apply running_mean/var
                    result_local = F.batch_norm(inputs[t_logical_domain], None, None,
                                          updated_weight, updated_bias,
                                          self.training, self.momentum, self.eps)
                elif norm_type == "eval":  # fix and apply running_mean/var,
                    if self.running_mean is None:
                        result_local = F.batch_norm(inputs[t_logical_domain], None, None,
                                              updated_weight, updated_bias,
                                              True, self.momentum, self.eps)
                    else:
                        result_local = F.batch_norm(inputs[t_logical_domain], self.running_mean, self.running_var,
                                              updated_weight, updated_bias,
                                              False, self.momentum, self.eps)

                if cnt == 0:
                    result = copy.copy(result_local)
                else:
                    result = torch.cat((result, result_local), 0)
                cnt += 1

        else:
            if norm_type == "general": # update, but not apply running_mean/var
                result = F.batch_norm(inputs, self.running_mean, self.running_var,
                                      updated_weight, updated_bias,
                                      self.training, self.momentum, self.eps)
            elif norm_type == "hold": # not update, not apply running_mean/var
                result = F.batch_norm(inputs, None, None,
                                      updated_weight, updated_bias,
                                      self.training, self.momentum, self.eps)
            elif norm_type == "eval": # fix and apply running_mean/var,
                if self.running_mean is None:
                    result = F.batch_norm(inputs, None, None,
                                          updated_weight, updated_bias,
                                          True, self.momentum, self.eps)
                else:
                    result = F.batch_norm(inputs, self.running_mean, self.running_var,
                                          updated_weight, updated_bias,
                                          False, self.momentum, self.eps)


        return result
class Meta_in_norm(nn.InstanceNorm2d):
    def __init__(self, num_features, norm_opt = None, eps=1e-05,
                 momentum=0.1, weight_freeze = False, bias_freeze = False,
                 weight_init = 1.0, bias_init = 0.0):

        if not weight_freeze:
            weight_freeze = norm_opt['IN_W_FREEZE']
        if not bias_freeze:
            bias_freeze = norm_opt['IN_B_FREEZE']

        affine = True if norm_opt['IN_AFFINE'] else False
        track_running_stats = True if norm_opt['IN_RUNNING'] else False
        super().__init__(num_features, eps, momentum, affine, track_running_stats)

        if self.weight is not None:
            if weight_init is not None: self.weight.data.fill_(weight_init)
            self.weight.requires_grad_(not weight_freeze)
        if self.bias is not None:
            if bias_init is not None: self.bias.data.fill_(bias_init)
            self.bias.requires_grad_(not bias_freeze)
        self.in_fc_multiply = norm_opt['IN_FC_MULTIPLY']

    def forward(self, inputs, opt = None):
        if inputs.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(inputs.dim()))

        if (inputs.shape[2] == 1) and (inputs.shape[2] == 1): # fc layers
            inputs[:] *= self.in_fc_multiply
            return inputs
        else:
            if opt != None:
                use_meta_learning = False
                if opt['param_update']:
                    if self.weight is not None:
                        if self.compute_meta_params:
                            use_meta_learning = True
            else:
                use_meta_learning = False

            if self.training:
                norm_type = opt['type_running_stats']
            else:
                norm_type = "eval"

            if use_meta_learning and self.affine:
                # if opt['zero_grad']: self.zero_grad()
                updated_weight = update_parameter(self.weight, self.w_step_size, opt)
                updated_bias = update_parameter(self.bias, self.b_step_size, opt)
                # print('meta_bn is computed')
            else:
                updated_weight = self.weight
                updated_bias = self.bias


            if norm_type == "general":
                return F.instance_norm(inputs, self.running_mean, self.running_var,
                                       updated_weight, updated_bias,
                                       self.training, self.momentum, self.eps)
            elif norm_type == "hold":
                return F.instance_norm(inputs, None, None,
                                       updated_weight, updated_bias,
                                       self.training, self.momentum, self.eps)
            elif norm_type == "eval":
                if self.running_mean is None:
                    return F.instance_norm(inputs, None, None,
                                           updated_weight, updated_bias,
                                           True, self.momentum, self.eps)
                else:
                    return F.instance_norm(inputs, self.running_mean, self.running_var,
                                           updated_weight, updated_bias,
                                           False, self.momentum, self.eps)
class Meta_bin_half(nn.Module):
    def __init__(self, num_features, norm_opt=None, **kwargs):
        super().__init__()

        half1 = math.ceil(num_features / 2)
        self.half = half1
        half2 = num_features - half1

        self.bat_n = Meta_bn_norm(half1, norm_opt, **kwargs)
        self.ins_n = Meta_in_norm(half2, norm_opt, **kwargs)

    def forward(self, inputs, opt = None):
        if inputs.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(inputs.dim()))

        # print(inputs.shape)
        split = torch.split(inputs, self.half, 1)
        out1 = self.bat_n(split[0].contiguous(), opt)
        out2 = self.ins_n(split[1].contiguous(), opt)
        out = torch.cat((out1, out2), 1)
        return out
class Meta_bin_gate_ver1(nn.BatchNorm2d):
    def __init__(self, num_features, norm_opt=None, eps=1e-05, momentum=0.1,
                 weight_freeze=False, bias_freeze=False,
                 weight_init=1.0, bias_init=0.0):


        if not weight_freeze:
            weight_freeze = norm_opt['BN_W_FREEZE']
        if not bias_freeze:
            bias_freeze = norm_opt['BN_B_FREEZE']

        affine = True if norm_opt['BN_AFFINE'] else False
        track_running_stats = True if norm_opt['BN_RUNNING'] else False
        super().__init__(num_features, eps, momentum, affine, track_running_stats)

        if weight_init is not None: self.weight.data.fill_(weight_init)
        if bias_init is not None: self.bias.data.fill_(bias_init)
        self.weight.requires_grad_(not weight_freeze)
        self.bias.requires_grad_(not bias_freeze)

        self.gate = Parameter(torch.Tensor(num_features))
        if norm_opt['BIN_INIT'] == 'one':
            self.gate.data.fill_(1.0)
        elif norm_opt['BIN_INIT'] == 'zero':
            self.gate.data.fill_(0.0)
        elif norm_opt['BIN_INIT'] == 'half':
            self.gate.data.fill_(0.5)
        elif norm_opt['BIN_INIT'] == 'random':
            self.gate.data = torch.rand(num_features)

        setattr(self.gate, 'bin_gate', True)

    def forward(self, inputs, opt = None):

        if inputs.dim() != 4:
            raise ValueError('expected 4D inputs (got {}D inputs)'.format(inputs.dim()))
        if opt != None:
            use_meta_learning = False
            use_meta_learning_gates = False
            if opt['param_update']:
                if self.weight is not None:
                    if self.compute_meta_params:
                        use_meta_learning = True
                if self.compute_meta_gates:
                    use_meta_learning_gates = True
        else:
            use_meta_learning = False
            use_meta_learning_gates = False
        if use_meta_learning and self.affine:
            # if opt['zero_grad']: self.zero_grad()
            updated_weight = update_parameter(self.weight, self.w_step_size, opt)
            updated_bias = update_parameter(self.bias, self.b_step_size, opt)
            # print('meta_bn is computed')
        else:
            updated_weight = self.weight
            updated_bias = self.bias

        if use_meta_learning_gates:
            update_gate = update_parameter(self.gate, self.g_step_size, opt)
            if opt['inner_clamp']:
                update_gate.data.clamp_(min=0, max=1)
            # print(update_gate[0].data.cpu())
        else:
            update_gate = self.gate

        if self.training:
            norm_type = opt['type_running_stats']
        else:
            norm_type = "eval"


        # Batch norm (2D)
        if self.affine:
            bn_w = updated_weight * update_gate
        else:
            bn_w = update_gate

        if norm_type == "general":
            out_bn = F.batch_norm(inputs, self.running_mean, self.running_var,
                                  bn_w, updated_bias,
                                  self.training, self.momentum, self.eps)
        elif norm_type == "hold":
            out_bn = F.batch_norm(inputs, None, None,
                                  bn_w, updated_bias,
                                  self.training, self.momentum, self.eps)
        elif norm_type == "eval":
            out_bn = F.batch_norm(inputs, self.running_mean, self.running_var,
                                  bn_w, updated_bias,
                                  False, self.momentum, self.eps)

        # Instance norm
        if inputs.size(2) != 1: # 2D
            b, c = inputs.size(0), inputs.size(1)
            if self.affine:
                in_w = updated_weight * (1 - update_gate)
            else:
                in_w = 1 - update_gate
            inputs = inputs.view(1, b * c, *inputs.size()[2:])
            out_in = F.batch_norm(inputs, None, None, None, None,
                                  True, self.momentum, self.eps)

            out_in = out_in.view(b, c, *inputs.size()[2:])
            out_in.mul_(in_w[None, :, None, None])

            return out_bn + out_in
        else:
            return out_bn # 1D
class Meta_bin_gate_ver2(nn.Module):
    def __init__(self, num_features, norm_opt = None, **kwargs):
        super().__init__()

        self.bat_n = Meta_bn_norm(num_features, norm_opt, **kwargs)
        self.ins_n = Meta_in_norm(num_features, norm_opt, **kwargs)

        self.gate = Parameter(torch.Tensor(num_features))
        if norm_opt['BIN_INIT'] == 'one':
            self.gate.data.fill_(1.0)
        elif norm_opt['BIN_INIT'] == 'zero':
            self.gate.data.fill_(0.0)
        elif norm_opt['BIN_INIT'] == 'half':
            self.gate.data.fill_(0.5)
        elif norm_opt['BIN_INIT'] == 'random':
            self.gate.data = torch.rand(num_features)
        setattr(self.gate, 'bin_gate', True)

    def forward(self, inputs, opt = None):

        if inputs.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(inputs.dim()))


        if opt != None:
            use_meta_learning_gates = False
            if opt['param_update']:
                if self.compute_meta_gates:
                    use_meta_learning_gates = True
        else:
            use_meta_learning_gates = False

        if use_meta_learning_gates:
            update_gate = update_parameter(self.gate, self.g_step_size, opt)
            if opt['inner_clamp']:
                update_gate.data.clamp_(min=0, max=1)
            # print(update_gate[0].data.cpu())
        else:
            update_gate = self.gate


        out1 = self.bat_n(inputs, opt)
        out2 = self.ins_n(inputs, opt)
        # out = out1 * self.gate + out2 * (1 - self.gate)


        out = out1.mul_(update_gate[None, :, None, None]) + \
              out2.mul_((1 - update_gate[None, :, None, None]))

        return out





def update_parameter(param, step_size, opt = None):
    loss = opt['meta_loss']
    use_second_order = opt['use_second_order']
    allow_unused = opt['allow_unused']
    stop_gradient = opt['stop_gradient']
    flag_update = False
    if step_size is not None:
        if not stop_gradient:
            if param is not None:
                if opt['auto_grad_outside']:
                    if opt['grad_params'][0] == None:
                        del opt['grad_params'][0]
                        updated_param = param
                    else:
                        # print("[GRAD]{} [PARAM]{}".format(opt['grad_params'][0].data.shape, param.data.shape))
                        # outer
                        updated_param = param - step_size * opt['grad_params'][0]
                        del opt['grad_params'][0]
                else:
                    # inner
                    grad = autograd.grad(loss, param, create_graph=use_second_order, allow_unused=allow_unused)[0]
                    updated_param = param - step_size * grad
                # outer update
                # updated_param = opt['grad_params'][0]
                # del opt['grad_params'][0]
                flag_update = True
        else:
            if param is not None:

                if opt['auto_grad_outside']:
                    if opt['grad_params'][0] == None:
                        del opt['grad_params'][0]
                        updated_param = param
                    else:
                        # print("[GRAD]{} [PARAM]{}".format(opt['grad_params'][0].data.shape, param.data.shape))
                        # outer
                        updated_param = param - step_size * opt['grad_params'][0]
                        del opt['grad_params'][0]
                else:
                    # inner
                    grad = Variable(autograd.grad(loss, param, create_graph=use_second_order, allow_unused=allow_unused)[0].data, requires_grad=False)
                    updated_param = param - step_size * grad
                # outer update
                # updated_param = opt['grad_params'][0]
                # del opt['grad_params'][0]
                flag_update = True
    if not flag_update:
        return param

    return updated_param
def meta_norm(norm, out_channels, norm_opt, **kwargs):
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "BN": Meta_bn_norm(out_channels, norm_opt, **kwargs),
            "IN": Meta_in_norm(out_channels, norm_opt, **kwargs),
            "BIN_half": Meta_bin_half(out_channels, norm_opt, **kwargs),
            "BIN_gate1": Meta_bin_gate_ver1(out_channels, norm_opt, **kwargs),
            "BIN_gate2": Meta_bin_gate_ver2(out_channels, norm_opt, **kwargs),
            "Task_norm": TaskNormI(out_channels),
            "DSON_ch": OptimizedNorm2d_ch(out_channels),
            "DSON": OptimizedNorm2d(out_channels),
            "Switchable_norm": SwitchNorm2d(out_channels),
            "Sync_bn": SyncBatchNorm2d(out_channels),
            "Sync_sn": SyncSwitchableNorm2d(out_channels),
        }[norm]
    return norm



