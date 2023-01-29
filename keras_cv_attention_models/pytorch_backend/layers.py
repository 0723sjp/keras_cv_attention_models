import torch
import numpy as np
from torch import nn
from functools import partial

class Weight:
    def __init__(self, name, value):
        self.name, self.shape, self.__value__ = name, value.shape, value

    def value(self):
        return self.__value__


class GraphNode:
    num_instances = 0  # Count instances
    @classmethod
    def __count__(cls):
        cls.num_instances += 1

    def __init__(self, shape, name=None):
        self.shape = shape
        self.name = "graphnode_{}".format(self.num_instances) if name is None else name
        self.pre_nodes, self.pre_node_names, self.next_nodes, self.next_node_names = [], [], [], []
        self.module = lambda xx: xx
        self.__count__()

    def __str__(self):
        # return ",".join(for kk, vv in zip())
        return self.name

    def set_pre_nodes(self, pre_nodes):
        pre_nodes = pre_nodes if isinstance(pre_nodes, (list, tuple)) else [pre_nodes]
        self.pre_nodes += pre_nodes
        self.pre_node_names += [ii.name for ii in pre_nodes]

    def set_next_nodes(self, next_nodes):
        next_nodes = next_nodes if isinstance(next_nodes, (list, tuple)) else [next_nodes]
        self.next_nodes += next_nodes
        self.next_node_names += [ii.name for ii in next_nodes]


class Input(GraphNode):
    def __init__(self, shape, name=None):
        shape = [None, *shape]
        name = "input_{}".format(self.num_instances) if name is None else name
        super().__init__(shape, name=name)


class Layer(nn.Module):
    num_instances = 0  # Count instances
    @classmethod
    def __count__(cls):
        cls.num_instances += 1

    def __init__(self, name=None, **kwargs):
        super().__init__()
        self.name, self.kwargs = self.verify_name(name), kwargs
        self.built = False
        self.module = lambda xx: xx
        self.__count__()

    def build(self, input_shape: torch.Size):
        self.input_shape = input_shape
        self.__output_shape__ = self.compute_output_shape(input_shape)
        if hasattr(self, 'call'):  # Original keras layers with call function
            # self.forward = self.call
            self.module = self.call
        self.built = True

    # def forward(self, inputs, *args, **kwargs):
    #     if not self.built:
    #         self.build(inputs.shape)
    #     return self.module(inputs, *args, **kwargs)

    def forward(self, inputs, *args, **kwargs):
        if not self.built:
            self.build([ii.shape for ii in inputs] if isinstance(inputs, (list, tuple)) else inputs.shape)
        if isinstance(inputs, GraphNode) or (isinstance(inputs, (list, tuple)) and any([isinstance(ii, GraphNode) for ii in inputs])):
            output_shape = self.compute_output_shape(self.input_shape)
            # if isinstance(output_shape[0], (list, tuple))
            cur_node = GraphNode(output_shape, name=self.name)
            cur_node.module = self.module if len(args) == 0 and len(kwargs) == 0 else lambda inputs: self.module(inputs, *args, **kwargs)
            cur_node.layer = self
            cur_node.set_pre_nodes(inputs)

            inputs = inputs if isinstance(inputs, (list, tuple)) else [inputs]
            for ii in inputs:
                ii.set_next_nodes(cur_node)
            self.output = self.node = cur_node
            return cur_node
        else:
            return self.module(inputs, *args, **kwargs)

    @property
    def weights(self):
        return [Weight(name=self.name + "/" + kk.split('.')[-1], value=vv) for kk, vv in self.state_dict().items()]

    @property
    def trainable_weights(self):
        return self.weights

    @property
    def non_trainable_weights(self):
        return []

    def add_weight(self, name=None, shape=None, dtype=None, initializer=None, regularizer=None, trainable=None):
        if isinstance(initializer, str):
            initializer = torch.ones if initializer == "ones" else torch.zeros
        return nn.Parameter(initializer(shape), requires_grad=trainable)

    def get_weights(self):
        return [ii.value().detach().cpu().numpy() for ii in self.weights]

    def set_weights(self, weights):
        return self.load_state_dict({kk: torch.from_numpy(vv) for kk, vv in zip(self.state_dict().keys(), weights)})

    def compute_output_shape(self, input_shape):
        return input_shape

    @property
    def output_shape(self):
        return getattr(self, "__output_shape__", None)

    def verify_name(self, name):
        return "{}_{}".format(self.__class__.__name__.lower(), self.num_instances) if name == None else name

    def get_config(self):
        config = {"name": self.name}
        config.update(self.kwargs)
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Lambda(Layer):
    def __init__(self, func, **kwargs):
        super().__init__(**kwargs)
        self.module = func

    def get_config(self):
        config = super().get_config()
        config.update({"func": self.module})
        return config

    def compute_output_shape(self, input_shape):
        print(self.module, input_shape)
        return [None] + list(self.module(torch.ones([1, *input_shape[1:]])).shape)[1:]


class Dense(Layer):
    def __init__(self, units, activation=None, use_bias=True, axis=-1, kernel_initializer='glorot_uniform', **kwargs):
        self.units, self.activation, self.use_bias, self.axis, self.kernel_initializer = units, activation, use_bias, axis, kernel_initializer
        super().__init__(**kwargs)

    def build(self, input_shape):
        module = nn.Linear(in_features=input_shape[self.axis], out_features=self.units, bias=self.use_bias)
        if self.axis == -1 or self.axis == len(input_shape):
            self.module = module if self.activation is None else nn.Sequential(module, Activation(self.activation))
        else:
            ndims = len(input_shape)
            perm = [id for id in range(ndims) if id != self.axis] + [self.axis]  # like axis=1 -> [0, 2, 3, 1]
            revert_perm = list(range(0, self.axis)) + [ndims - 1] + list(range(self.axis, ndims - 1))  # like axis=1 -> [0, 3, 1, 2]
            if self.activation is None:
                self.module = nn.Sequential(Permute(perm[1:]), module, Permute(revert_perm[1:]))
            else:
                self.module = nn.Sequential(Permute(perm[1:]), module, Permute(revert_perm[1:]), Activation(self.activation))
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[:self.axis] + [self.units] + ([] if self.axis == -1 else input_shape[self.axis + 1:])

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "activation": self.activation, "use_bias": self.use_bias, "axis": self.axis})
        return config


class Conv(Layer):
    def __init__(self, filters, kernel_size=1, strides=1, padding="VALID", dilation_rate=1, use_bias=True, groups=1, kernel_initializer='glorot_uniform', **kwargs):
        self.filters, self.padding, self.use_bias, self.groups, self.kernel_initializer = filters, padding, use_bias, groups, kernel_initializer
        self.kernel_size, self.dilation_rate, self.strides = kernel_size, dilation_rate, strides
        super().__init__(**kwargs)

    @property
    def module_class(self):
        return nn.Conv2d

    def build(self, input_shape):
        num_dims = len(input_shape) - 2  # Conv2D -> 2, Conv1D -> 1
        self.kernel_size = self.kernel_size if isinstance(self.kernel_size, (list, tuple)) else [self.kernel_size] * num_dims
        self.dilation_rate = self.dilation_rate if isinstance(self.dilation_rate, (list, tuple)) else [self.dilation_rate] * num_dims
        self.strides = self.strides if isinstance(self.strides, (list, tuple)) else [self.strides] * num_dims
        self.filters = self.filters if self.filters > 0 else input_shape[1]  # In case DepthwiseConv2D

        if isinstance(self.padding, str):
            self._pad = [ii // 2 for ii in self.kernel_size] if self.padding.upper() == "SAME" else [0] * num_dims
        else:  # int or list or tuple with specific value
            self._pad = padding if isinstance(padding, (list, tuple)) else [padding] * num_dims

        self.module = self.module_class(
            in_channels=input_shape[1],
            out_channels=self.filters,
            kernel_size=self.kernel_size,
            stride=self.strides,
            padding=self._pad,
            dilation=self.dilation_rate,
            groups=self.groups,
            bias=self.use_bias,
        )
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        size = input_shape[2:]
        dilated_filter_size = [kk + (kk - 1) * (dd - 1) for kk, dd in zip(self.kernel_size, self.dilation_rate)]
        if self.padding.upper() == "VALID":
            size = [ii - jj + 1 for ii, jj in zip(size, dilated_filter_size)]
        size = [(ii + jj - 1) // jj for ii, jj in zip(size, self.strides)]
        return [None, self.filters, *size]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "dilation_rate": self.dilation_rate,
                "use_bias": self.use_bias,
                "groups": self.groups,
            }
        )
        return config


class Conv2D(Conv):
    @property
    def module_class(self):
        return nn.Conv2d


class DepthwiseConv2D(Conv2D):
    def __init__(self, kernel_size=1, strides=1, padding="VALID", dilation_rate=(1, 1), use_bias=True, kernel_initializer='glorot_uniform', **kwargs):
        self.kernel_size, self.strides, self.padding = kernel_size, strides, padding
        self.dilation_rate, self.use_bias, self.kernel_initializer = dilation_rate, use_bias, kernel_initializer
        super().__init__(filters=-1, **kwargs)

    def build(self, input_shape):
        self.groups = input_shape[1]
        super().build(input_shape)


class Conv1D(Conv):
    @property
    def module_class(self):
        return nn.Conv1d


class BatchNormalization(Layer):
    def __init__(self, axis=1, momentum=0.9, epsilon=1e-5, center=True,gamma_initializer='ones', **kwargs):
        self.axis, self.momentum, self.epsilon, self.center, self.gamma_initializer = axis, momentum, epsilon, center, gamma_initializer
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.module = nn.BatchNorm2d(num_features=input_shape[self.axis], eps=self.epsilon, momentum=1 - self.momentum, affine=self.center)
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis, "momentum": self.momentum, "epsilon": self.epsilon, "center": self.center})
        return config


class LayerNormalization(Layer):
    def __init__(self, axis=1, epsilon=1e-5, center=True, gamma_initializer='ones', **kwargs):
        self.axis, self.epsilon, self.center, self.gamma_initializer = axis, epsilon, center, gamma_initializer
        super().__init__(**kwargs)

    def build(self, input_shape):
        module = nn.LayerNorm(normalized_shape=input_shape[self.axis], eps=self.epsilon, elementwise_affine=self.center)
        if self.axis == -1 or self.axis == len(input_shape):
            self.module = module
        else:
            ndims = len(input_shape)
            perm = [id for id in range(ndims) if id != self.axis] + [self.axis]  # like axis=1 -> [0, 2, 3, 1]
            revert_perm = list(range(0, self.axis)) + [ndims - 1] + list(range(self.axis, ndims - 1))  # like axis=1 -> [0, 3, 1, 2]
            self.module = nn.Sequential(Permute(perm[1:]), module, Permute(revert_perm[1:]))
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis, "epsilon": self.epsilon, "center": self.center})
        return config


class Pooling2D(Layer):
    def __init__(self, pool_size=(2, 2), strides=1, padding="VALID", reduce="mean", **kwargs):
        self.pool_size, self.strides, self.padding, self.reduce = pool_size, strides, padding, reduce
        self.pool_size = pool_size if isinstance(pool_size, (list, tuple)) else [pool_size, pool_size]
        self.strides = strides if isinstance(strides, (list, tuple)) else [strides, strides]
        super().__init__(**kwargs)

    def build(self, input_shape):
        pool_size = self.pool_size
        if isinstance(self.padding, str):
            pad = (pool_size[0] // 2, pool_size[1] // 2) if self.padding.upper() == "SAME" else (0, 0)
        else:  # int or list or tuple with specific value
            pad = padding if isinstance(padding, (list, tuple)) else (padding, padding)
        self._pad = pad

        if reduce.lower() == "max":
            self.module = nn.MaxPool2d(kernel_size=self.pool_size, stride=self.strides, padding=pad)
        else:
            self.module = nn.AvgPool2d(kernel_size=self.pool_size, stride=self.strides, padding=pad)
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        batch, _, height, width = input_shape
        height = (height + 2 * self._pad[0] - (self.pool_size[0] - self.strides[0])) // self.strides[0]  # Not considering dilation
        width = (width + 2 * self._pad[1] - (self.pool_size[1] - self.strides[1])) // self.strides[1]  # Not considering dilation
        return [batch, self.filters, height, width]

    def get_config(self):
        config = super().get_config()
        config.update({"pool_size": self.pool_size, "strides": self.strides, "padding": self.padding, "reduce": self.reduce})
        return config


class AvgPool2D(Pooling2D):
    def __init__(self, pool_size=(2, 2), strides=1, padding="VALID", **kwargs):
        super().__init__(reduce="mean", **kwargs)


class MaxPool2D(Pooling2D):
    def __init__(self, pool_size=(2, 2), strides=1, padding="VALID", **kwargs):
        super().__init__(reduce="max", **kwargs)


class GlobalAveragePooling2D(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.module = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(1))
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[:2]


class GlobalAveragePooling1D(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.module = torch.nn.Sequential(torch.nn.AdaptiveAvgPool1d(1), torch.nn.Flatten(1))
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[:2]


class ZeroPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        assert len(padding) == 2 if isinstance(padding, (list, tuple)) else isinstance(padding, int), "padding should be 2 values or an int: {}".format(padding)
        self.padding = padding if isinstance(padding, (list, tuple)) else [padding, padding]
        super().__init__(**kwargs)

    def build(self, input_shape):
        padding = self.padding * 2  # torch.nn.ZeroPad2d needs 4 values
        self.module = torch.nn.ZeroPad2d(padding=padding)
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return [input_shape[0], input_shape[1], input_shape[2] + self.padding[0] * 2, input_shape[3] + self.padding[1] * 2]

    def get_config(self):
        config = super().get_config()
        config.update({"padding": self.padding})
        return config


class Activation(Layer):
    def __init__(self, activation=None, **kwargs):
        self.activation = activation
        super().__init__(**kwargs)

    def build(self, input_shape):
        if self.activation is None:
            self.module = torch.nn.Identity()
        elif isinstance(self.activation, str) and self.activation == "softmax":
            self.module = partial(torch.softmax, dim=1)
        elif isinstance(self.activation, str):
            self.module = getattr(torch.functional.F, self.activation)
        else:
            self.module = self.activation
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({"activation": self.activation})
        return config


class PReLU(Layer):
    def __init__(self, alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None, **kwargs):
        self.shared_axes = shared_axes
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.module = nn.PReLU(num_parameters=input_shape[1], init=0.25)
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({"shared_axes": self.shared_axes})
        return config


class Dropout(Layer):
    def __init__(self, rate, noise_shape=None, **kwargs):
        self.rate, self.noise_shape = rate, noise_shape
        super().__init__(**kwargs)

    def build(self, input_shape):
        # if drop_prob == 0. or not training:
        #     return x
        # keep_prob = 1 - drop_prob
        # shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        # random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        # if keep_prob > 0.0 and scale_by_keep:
        #     random_tensor.div_(keep_prob)
        # return x * random_tensor
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({"rate": self.rate, "noise_shape": self.noise_shape})
        return config


class _Merge(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        assert self.check_shape_all_equal(input_shape), "input_shape not all equal: {}".format(input_shape)
        super().build(input_shape)

    def check_shape_all_equal(self, input_shapes):
        # return not any([any([[jj[id] - input_shapes[0][id] for jj in input_shapes[1:]]]) for id in range(1, len(input_shapes[0]))])
        base, dims = input_shapes[0], len(input_shapes[0])
        return not any(any([base[dim] != ii[dim] for dim in range(1, dims)]) for ii in input_shapes[1:])

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class Add(_Merge):
    def build(self, input_shape):
        self.module = lambda inputs: torch.sum(torch.stack(inputs, axis=0), axis=0)
        # self.module = lambda inputs: Lambda(torch.sum)(torch.stack(inputs, axis=0), axis=0)
        super().build(input_shape)


class Multiply(_Merge):
    def build(self, input_shape):
        self.module = lambda inputs: torch.prod(torch.stack(inputs, axis=0), axis=0)
        super().build(input_shape)


class Reshape(Layer):
    def __init__(self, target_shape, **kwargs):
        self.target_shape = [-1 if ii is None else ii for ii in target_shape]
        super().__init__(**kwargs)

    def build(self, input_shape):
        num_unknown_dim = sum([ii == -1 for ii in self.target_shape])
        assert num_unknown_dim < 2, "At most one unknown dimension in output_shape: {}".format(self.target_shape)

        total_size = np.prod(input_shape[1:])
        if num_unknown_dim == 1:
            unknown_dim = total_size // (-1 * np.prod(self.target_shape))
            self.target_shape = [unknown_dim if ii == -1 else ii for ii in self.target_shape]
        assert total_size == np.prod(self.target_shape), "Total size of new array must be unchanged, {} -> {}".format(input_shape, self.target_shape)

        self.module = lambda inputs: torch.reshape(inputs, shape=[-1, *self.target_shape])
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return [input_shape[0], *self.target_shape]

    def get_config(self):
        config = super().get_config()
        config.update({"target_shape": self.target_shape})
        return config


class Permute(Layer):
    def __init__(self, dims, **kwargs):
        self.dims = dims
        super().__init__(**kwargs)
        assert sorted(dims) == list(range(1, len(dims) + 1)), "The set of indices in `dims` must be consecutive and start from 1. dims: {}".format(dims)

    def build(self, input_shape):
        self.module = lambda inputs: torch.permute(inputs, dims=[0, *self.dims])
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        # output_shape = input_shape.copy()
        # for i, dim in enumerate(self.dims):
        #     output_shape[i + 1] = input_shape[dim]
        return [input_shape[0]] + [input_shape[dim] for dim in self.dims]

    def get_config(self):
        config = super().get_config()
        config.update({"dims": self.dims})
        return config
