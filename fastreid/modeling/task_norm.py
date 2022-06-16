import torch
from torch.autograd import Function
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.distributed as dist
import torch.nn as nn

import torch.autograd as autograd
from torch.autograd import Variable
from torch import nn
import torch


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



class NormalizationLayer(nn.BatchNorm2d):
    """
    Base class for all normalization layers.
    Derives from nn.BatchNorm2d to maintain compatibility with the pre-trained resnet-18.
    """
    def __init__(self, num_features):
        """
        Initialize the class.
        :param num_features: number of channels in the 2D convolutional layer
        """
        super(NormalizationLayer, self).__init__(
            num_features,
            eps=1e-05,
            momentum=0.1,
            affine=True,
            track_running_stats=True)

    def forward(self, x, opt = None):
        """
        Normalize activations.
        :param x: input activations
        :return: normalized activations
        """
        pass  # always override this method

    def _normalize(self, x, mean, var, weight, bias):
        """
        Normalize activations.
        :param x: input activations
        :param mean: mean used to normalize
        :param var: var used to normalize
        :return: normalized activations
        """
        return (weight.view(1, -1, 1, 1) * (x - mean) / torch.sqrt(var + self.eps)) + bias.view(1, -1, 1, 1)

    @staticmethod
    def _compute_batch_moments(x):
        """
        Compute conventional batch mean and variance.
        :param x: input activations
        :return: batch mean, batch variance
        """
        return torch.mean(x, dim=(0, 2, 3), keepdim=True), torch.var(x, dim=(0, 2, 3), keepdim=True)

    @staticmethod
    def _compute_instance_moments(x):
        """
        Compute instance mean and variance.
        :param x: input activations
        :return: instance mean, instance variance
        """
        return torch.mean(x, dim=(2, 3), keepdim=True), torch.var(x, dim=(2, 3), keepdim=True)

    @staticmethod
    def _compute_layer_moments(x):
        """
        Compute layer mean and variance.
        :param x: input activations
        :return: layer mean, layer variance
        """
        return torch.mean(x, dim=(1, 2, 3), keepdim=True), torch.var(x, dim=(1, 2, 3), keepdim=True)

    @staticmethod
    def _compute_pooled_moments(x, alpha, batch_mean, batch_var, augment_moment_fn):
        """
        Combine batch moments with augment moments using blend factor alpha.
        :param x: input activations
        :param alpha: moment blend factor
        :param batch_mean: standard batch mean
        :param batch_var: standard batch variance
        :param augment_moment_fn: function to compute augment moments
        :return: pooled mean, pooled variance
        """
        augment_mean, augment_var = augment_moment_fn(x)
        pooled_mean = alpha * batch_mean + (1.0 - alpha) * augment_mean
        batch_mean_diff = batch_mean - pooled_mean
        augment_mean_diff = augment_mean - pooled_mean
        pooled_var = alpha * (batch_var + (batch_mean_diff * batch_mean_diff)) +\
                     (1.0 - alpha) * (augment_var + (augment_mean_diff * augment_mean_diff))
        return pooled_mean, pooled_var


class TaskNormBase(NormalizationLayer):
    """TaskNorm base class."""
    def __init__(self, num_features):
        """
        Initialize
        :param num_features: number of channels in the 2D convolutional layer
        """
        super(TaskNormBase, self).__init__(num_features)
        self.sigmoid = torch.nn.Sigmoid()

    def register_extra_weights(self):
        """
        The parameters here get registered after initialization because the pre-trained resnet model does not have
        these parameters and would fail to load if these were declared at initialization.
        :return: Nothing
        """
        device = self.weight.device

        # Initialize and register the learned parameters 'a' (SCALE) and 'b' (OFFSET)
        # for calculating alpha as a function of context size.
        a = torch.Tensor([0.0]).to(device)
        b = torch.Tensor([0.0]).to(device)
        self.register_parameter(name='a', param=torch.nn.Parameter(a, requires_grad=True))
        self.register_parameter(name='b', param=torch.nn.Parameter(b, requires_grad=True))

        # Variables to store the context moments to use for normalizing the target.
        self.register_buffer(name='batch_mean',
                             tensor=torch.zeros((1, self.num_features, 1, 1), requires_grad=True, device=device))
        self.register_buffer(name='batch_var',
                             tensor=torch.ones((1, self.num_features, 1, 1), requires_grad=True, device=device))

        # Variable to save the context size.
        self.register_buffer(name='context_size',
                             tensor=torch.zeros((1), requires_grad=False, device=device))

    def _get_augment_moment_fn(self):
        """
        Provides the function to compute augment moemnts.
        :return: function to compute augment moments.
        """
        pass  # always override this function

    def forward(self, x, opt = None):
        """
        Normalize activations.
        :param x: input activations
        :return: normalized activations
        """

        if x.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(x.dim()))
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
            norm_type = 'eval'
        #
        if use_meta_learning:
            updated_weight = update_parameter(self.weight, self.w_step_size, opt)
            updated_bias = update_parameter(self.bias, self.b_step_size, opt)
            updated_a = update_parameter(self.a, self.w_step_size, opt)
            updated_b = update_parameter(self.b, self.b_step_size, opt)
        else:
            updated_weight = self.weight
            updated_bias = self.bias
            updated_a = self.a
            updated_b = self.b

        if norm_type == 'general':  # compute the pooled moments for the context and save off the moments and context size
            alpha = self.sigmoid(updated_a * (x.size())[0] + updated_b)  # compute alpha with context size
            batch_mean, batch_var = self._compute_batch_moments(x)
            pooled_mean, pooled_var = self._compute_pooled_moments(x, alpha, batch_mean, batch_var,
                                                                   self._get_augment_moment_fn())
            self.context_batch_mean = batch_mean
            self.context_batch_var = batch_var
            self.context_size = torch.full_like(self.context_size, x.size()[0])
        else:  # compute the pooled moments for the target
            alpha = self.sigmoid(updated_a * self.context_size + updated_b)  # compute alpha with saved context size
            pooled_mean, pooled_var = self._compute_pooled_moments(x, alpha, self.context_batch_mean,
                                                                   self.context_batch_var,
                                                                   self._get_augment_moment_fn())

        return self._normalize(x, pooled_mean, pooled_var, updated_weight, updated_bias)  # normalize


class TaskNormI(TaskNormBase):
    """
    TaskNorm-I normalization layer. Just need to override the augment moment function with 'instance'.
    """
    def __init__(self, num_features):
        """
        Initialize
        :param num_features: number of channels in the 2D convolutional layer
        """
        super(TaskNormI, self).__init__(num_features)

    def _get_augment_moment_fn(self):
        """
        Override the base class to get the function to compute instance moments.
        :return: function to compute instance moments
        """
        return self._compute_instance_moments




