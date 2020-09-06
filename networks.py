import paddle.fluid as fluid
import paddle.fluid.layers as layers
from paddle.fluid.dygraph import PRelu, InstanceNorm, Conv2D, Pool2D, Sequential, Linear
import paddle.fluid.dygraph.nn as nn
import numpy as np

class ReflectionPad2D(fluid.dygraph.Layer):
    def __init__(self, padding, mode='reflect'):
        super().__init__()
        if isinstance(padding, int):
            self.padding = [padding] * 4
        else:
            self.padding = padding
        self.mode = mode

    def forward(self, x):
        return fluid.layers.pad2d(x, self.padding, mode=self.mode)


class Spectralnorm(fluid.dygraph.Layer):

    def __init__(self,
                 layer,
                 dim=1,
                 power_iters=1,
                 eps=1e-12,
                 dtype='float32'):
        super(Spectralnorm, self).__init__()
        self.spectral_norm = nn.SpectralNorm(layer.weight.shape, dim, power_iters, eps, dtype)
        self.dim = dim
        self.power_iters = power_iters
        self.eps = eps
        self.layer = layer
        weight = layer._parameters['weight']
        del layer._parameters['weight']
        self.weight_orig = self.create_parameter(weight.shape, dtype=weight.dtype)
        self.weight_orig.set_value(weight)

    def forward(self, x):
        weight = self.spectral_norm(self.weight_orig)
        self.layer.weight = weight
        out = self.layer(x)
        
        return out


class Relu(fluid.Layer):
    def __init__(self):
        super().__init__()
        self.relu = lambda x: fluid.layers.relu(x)

    def forward(self, x):
        return self.relu(x)

class LeakyRelu(fluid.Layer):

    def __init__(self, alpha=0.2):
        super().__init__()
        self.leaky_relu = lambda x: fluid.layers.leaky_relu(x, alpha=alpha)

    def forward(self, x):
        return self.leaky_relu(x)


class Tanh(fluid.Layer):
    def __init__(self):
        super().__init__()
        self.tanh = lambda x: fluid.layers.tanh(x)

    def forward(self, x):
        return self.tanh(x)


class UpSample(fluid.dygraph.Layer):
    '''
    上采样数据
    '''
    def __init__(self, scale=2):
        super(UpSample, self).__init__()
        self.scale = scale

    def forward(self, inputs):
        # get dynamic upsample output shape
        shape_nchw = fluid.layers.shape(inputs)
        shape_hw = fluid.layers.slice(shape_nchw, axes=[0], starts=[2], ends=[4])
        shape_hw.stop_gradient = True
        in_shape = fluid.layers.cast(shape_hw, dtype='int32')
        out_shape = in_shape * self.scale
        out_shape.stop_gradient = True

        # reisze by actual_shape
        out = fluid.layers.resize_nearest(
            input=inputs, scale=self.scale, actual_shape=out_shape)
        return out



class ResnetGenerator(fluid.dygraph.Layer):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, img_size=256, light=False):
        
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()

        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.img_size = img_size
        self.light = light

        DownBlock = []
        DownBlock += [
                    ReflectionPad2D(3),
                    Conv2D(num_channels=input_nc, num_filters=ngf, filter_size=7, stride=1, padding=0, bias_attr=False),
                    InstanceNorm(self.ngf),
                    Relu(),
            ]

        # Down-Sampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            DownBlock += [
                ReflectionPad2D(1),
                Conv2D(num_channels=ngf * mult, num_filters=ngf * mult * 2, filter_size=3, stride=2, padding=0, bias_attr=False),
                InstanceNorm(ngf * mult * 2),
                Relu(),
            ]

        # Down-Sampling Bottleneck
        mult = 2**n_downsampling
        for i in range(n_blocks):
            DownBlock += [ResnetBlock(ngf * mult, use_bias=False)]

        # Class Activation Map
        self.gap_fc = Linear(ngf * mult, 1, bias_attr=False, act='sigmoid')
        self.gmp_fc = Linear(ngf * mult, 1, bias_attr=False, act='sigmoid')

        self.conv1x1 = Conv2D(ngf * mult * 2, ngf * mult, filter_size=1, stride=1, bias_attr=True)
        self.relu = Relu()

        # Gamma, Beta block
        if self.light:
            FC = [
                Linear(ngf * mult, ngf * mult, bias_attr=False),
                Relu(),
                Linear(ngf * mult, ngf * mult, bias_attr=False),
                Relu(),
            ]
        else:
            FC = [
                Linear(img_size // mult * img_size // mult * ngf * mult, ngf * mult, bias_attr=False),
                Relu(),
                Linear(ngf * mult, ngf * mult, bias_attr=False),
                Relu(),
            ]
        self.gamma = Linear(ngf * mult, ngf * mult, bias_attr=False)
        self.beta = Linear(ngf * mult, ngf * mult, bias_attr=False)

        # Up-Sampling Bottleneck
        for i in range(n_blocks):
            setattr(self, 'UpBlock1_' + str(i+1), ResnetAdaILNBlock(ngf * mult, use_bias=False))

        # Up-Sampling
        UpBlock2 = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            UpBlock2 += [
                    UpSample(2),
                    ReflectionPad2D(1),
                    Conv2D(num_channels=ngf * mult, num_filters=int(ngf * mult / 2),
                        filter_size=3, stride=1, padding=0, bias_attr=False),
                    ILN(int(ngf * mult / 2)),
                    Relu(),
            ]

        UpBlock2 += [
                    ReflectionPad2D(3),
                    Conv2D(num_channels=ngf, num_filters=output_nc,
                        filter_size=7, stride=1, padding=0, bias_attr=False),
                    Tanh(),
            ]

        self.DownBlock = Sequential(*DownBlock)
        self.DownBlock_list = DownBlock
        self.FC = Sequential(*FC)
        self.UpBlock2 = Sequential(*UpBlock2)

    def forward(self, input):
        x = self.DownBlock(input)
        gap = layers.adaptive_pool2d(x, 1, pool_type="avg")
        gap_logit = self.gap_fc(fluid.layers.reshape(gap, (x.shape[0], -1)))

        gap_weight = list(self.gap_fc.parameters())[0]
        gap_weight = fluid.layers.reshape(gap_weight, (-1, gap_weight.shape[0]))
        gap = x * fluid.layers.unsqueeze(fluid.layers.unsqueeze(gap_weight, 2), 3)

        gmp = layers.adaptive_pool2d(x, 1, pool_type="max")
        gmp_logit = self.gmp_fc(fluid.layers.reshape(gmp, (x.shape[0], -1)))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp_weight = fluid.layers.reshape(gmp_weight, (-1, gmp_weight.shape[0]))
        gmp = x * fluid.layers.unsqueeze(fluid.layers.unsqueeze(gmp_weight, 2), 3)
        
        cam_logit = layers.concat([gap_logit, gmp_logit], 1)

        x = layers.concat([gap, gmp], 1)
        x = self.relu(self.conv1x1(x))

        heatmap = layers.reduce_sum(x, dim=1, keep_dim=True)

        if self.light:
            x_ = layers.adaptive_pool2d(x, 1, pool_type="avg")
            x_ = self.FC(fluid.layers.reshape(x_, (x_.shape[0], -1)))
        else:
            x_ = self.FC(fluid.layers.reshape(x, (x.shape[0], -1)))

        gamma, beta = self.gamma(x_), self.beta(x_)

        for i in range(self.n_blocks):
            x = getattr(self, 'UpBlock1_' + str(i+1))(x, gamma, beta)

        out = self.UpBlock2(x)

        return out, cam_logit, heatmap

class ResnetBlock(fluid.dygraph.Layer):
    def __init__(self, dim, use_bias):
        super(ResnetBlock, self).__init__()
        conv_block = []
        conv_block += [
            ReflectionPad2D(1),
            Conv2D(num_channels=dim, num_filters=dim, filter_size=3, stride=1, padding=0,
                   bias_attr=use_bias),
            InstanceNorm(dim),
            Relu(),
        ]

        conv_block += [
            ReflectionPad2D(1),
            Conv2D(num_channels=dim, num_filters=dim, filter_size=3, stride=1, padding=0,
                   bias_attr=use_bias),
            InstanceNorm(dim)
        ]

        self.conv_block = Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResnetAdaILNBlock(fluid.dygraph.Layer):
    def __init__(self, dim, use_bias):
        super(ResnetAdaILNBlock, self).__init__()
        self.pad1 = ReflectionPad2D(1)
        self.conv1 = Conv2D(num_channels=dim, num_filters=dim, filter_size=3, stride=1, padding=0, bias_attr=use_bias)
        self.norm1 = adaILN(dim)
        self.relu1 = Relu()

        self.pad2 = ReflectionPad2D(1)
        self.conv2 = Conv2D(num_channels=dim, num_filters=dim, filter_size=3, stride=1, padding=0, bias_attr=use_bias)
        self.norm2 = adaILN(dim)

    def forward(self, x, gamma, beta):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out, gamma, beta)
        out = self.relu1(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, gamma, beta)

        return out + x


class adaILN(fluid.dygraph.Layer):
    def __init__(self, num_features, rho_init=0.99, eps=1e-5):
        super(adaILN, self).__init__()
        self.eps = eps   
        self.rho = fluid.layers.create_parameter((1, num_features, 1, 1), dtype='float32', is_bias=True,
                                            default_initializer=fluid.initializer.ConstantInitializer(rho_init))

    def forward(self, input, gamma, beta):

        in_mean = layers.reduce_mean(input, dim=[2, 3], keep_dim=True)
        in_var = get_var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / layers.sqrt(in_var + self.eps)

        ln_mean = layers.reduce_mean(input, dim=[1, 2, 3], keep_dim=True)
        ln_var = get_var(input, dim=[2, 3], keepdim=True)
        out_ln = (input - ln_mean) / layers.sqrt(ln_var + self.eps)

        out = fluid.layers.expand(self.rho, [input.shape[0], 1, 1, 1]) * out_in + (1-fluid.layers.expand(self.rho, [input.shape[0], 1, 1, 1])) * out_ln
        out = out * fluid.layers.unsqueeze(fluid.layers.unsqueeze(gamma, 2), 3) + fluid.layers.unsqueeze(fluid.layers.unsqueeze(beta, 2), 3)

        return out

def get_var(input, dim=None, keepdim=False, unbiased=True, name=None):
    rank = len(input.shape)
    dims = dim if dim != None and dim != [] else range(rank)
    dims = [e if e >= 0 else e + rank for e in dims]
    inp_shape = input.shape
    mean = layers.reduce_mean(input, dim=dim, keep_dim=True, name=name)
    tmp = layers.reduce_mean((input - mean)**2, dim=dim, keep_dim=keepdim, name=name)
    if unbiased:
        n = 1
        for i in dims:
            n *= inp_shape[i]
        factor = n / (n - 1.0) if n > 1.0 else 0.0
        tmp *= factor
    return tmp

class ILN(fluid.dygraph.Layer):
    def __init__(self, num_features, eps=1e-5):
        super(ILN, self).__init__()
        self.eps = eps
        self.rho = fluid.layers.create_parameter((1, num_features, 1, 1), dtype='float32', is_bias=True,
                                                 default_initializer=fluid.initializer.ConstantInitializer(0.0))
        self.gamma = fluid.layers.create_parameter((1, num_features, 1, 1), dtype='float32', is_bias=True,
                                                   default_initializer=fluid.initializer.ConstantInitializer(1.0))
        self.beta = fluid.layers.create_parameter((1, num_features, 1, 1), dtype='float32', is_bias=True,
                                                  default_initializer=fluid.initializer.ConstantInitializer(0.0))

    def forward(self, input):
        
        in_mean = layers.reduce_mean(input, dim=[2, 3], keep_dim=True)
        in_var = get_var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / layers.sqrt(in_var + self.eps)
        
        ln_mean = layers.reduce_mean(input, dim=[1, 2, 3], keep_dim=True)
        ln_var = get_var(input, dim=[2, 3], keepdim=True)
        out_ln = (input - ln_mean) / layers.sqrt(ln_var + self.eps)

        out = fluid.layers.expand(self.rho, [input.shape[0], 1, 1, 1]) * out_in + (1-fluid.layers.expand(self.rho, [input.shape[0], 1, 1, 1])) * out_ln
        out = out * fluid.layers.expand(self.gamma, [input.shape[0], 1, 1, 1]) + fluid.layers.expand(self.beta, [input.shape[0], 1, 1, 1])

        return out

class Discriminator(fluid.dygraph.Layer):
    def __init__(self, input_nc, ndf=64, n_layers=5):
        super(Discriminator, self).__init__()
        model = [
                ReflectionPad2D(1),
                Spectralnorm(Conv2D(num_channels=input_nc, num_filters=ndf, filter_size=4, stride=2, padding=0)),
                LeakyRelu(0.2),
                ]

        for i in range(1, n_layers - 2):
            mult = 2 ** (i - 1)
            model += [
                    ReflectionPad2D(1),
                    Spectralnorm(Conv2D(num_channels=ndf * mult, num_filters=ndf * mult * 2, filter_size=4, stride=2, padding=0)),
                    LeakyRelu(0.2),
                    ]

        mult = 2 ** (n_layers - 2 - 1)
        model += [
                    ReflectionPad2D(1),
                    Spectralnorm(Conv2D(num_channels=ndf * mult, num_filters=ndf * mult * 2, filter_size=4, stride=1, padding=0)),
                    LeakyRelu(0.2),
                  ]

        # Class Activation Map
        mult = 2 ** (n_layers - 2)

        self.gap_fc = Spectralnorm(Linear(ndf * mult, 1, bias_attr=False))
        self.gmp_fc = Spectralnorm(Linear(ndf * mult, 1, bias_attr=False))

        self.conv1x1 = Conv2D(num_channels=ndf * mult * 2, num_filters=ndf * mult, filter_size=1,
                              stride=1, bias_attr=False)
        self.leaky_relu = LeakyRelu(0.2)

        self.pad = ReflectionPad2D(1)
        self.conv = Spectralnorm(Conv2D(num_channels=ndf * mult, num_filters=1, filter_size=4,
                              stride=1, padding=0, bias_attr=False))

        self.model = Sequential(*model)

    def forward(self, input):
        x = self.model(input)

        gap = layers.adaptive_pool2d(x, 1, pool_type="avg")
        gap_logit = self.gap_fc(fluid.layers.reshape(gap, (x.shape[0], -1)))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap_weight = fluid.layers.reshape(gap_weight, (-1, gap_weight.shape[0]))
        gap = x * fluid.layers.unsqueeze(fluid.layers.unsqueeze(gap_weight, 2), 3)
        
        gmp = layers.adaptive_pool2d(x, 1, pool_type="max")
        gmp_logit = self.gmp_fc(fluid.layers.reshape(gmp, (x.shape[0], -1)))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp_weight = fluid.layers.reshape(gmp_weight, (-1, gmp_weight.shape[0]))
        gmp = x * fluid.layers.unsqueeze(fluid.layers.unsqueeze(gmp_weight, 2), 3)
        
        cam_logit = layers.concat([gap_logit, gmp_logit], 1)

        x = layers.concat([gap, gmp], 1)
        x = self.leaky_relu(self.conv1x1(x))
        
        heatmap = layers.reduce_sum(x, dim=1, keep_dim=True)

        x = self.pad(x)
        out = self.conv(x)
        
        return out, cam_logit, heatmap

class RhoClipper(object):
    def __init__(self, min, max):
        self.clip_min = min
        self.clip_max = max
        assert min < max

    def __call__(self, module):

        if hasattr(module, 'rho'):
            w = module.rho.data
            w = w.clamp(self.clip_min, self.clip_max)
            module.rho.data = w
