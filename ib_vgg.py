import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from ib_layers import *

# model configuration, (out_channels, kl_multiplier), 'M': Mean pooling, 'A': Average pooling
cfg = {
    'D6': [(64, 1.0/32), (64, 1.0/32), 'M', (128, 1.0/16), (128, 1.0/16), 'M', (256, 1.0/8), (256, 1.0/8), (256, 1.0/8),
        'M', (512, 1.0/4), (512, 1.0/4), (512, 1.0/4), 'M', (512, 1.0/2), (512, 1.0/2), (512, 1.0/2), 'M'],
    'D5': [(64, 1.0/32**2), (64, 1.0/32**2), 'M', (128, 1.0/16**2), (128, 1.0/16**2), 'M', (256, 1.0/8**2), (256, 1.0/8**2), (256, 1.0/8**2),
        'M', (512, 1.0/4**2), (512, 1.0/4**2), (512, 1.0/4**2), 'M', (512, 1.0/2**2), (512, 1.0/2**2), (512, 1.0/2**2), 'M'],
    'D4': [(64, 1.0/32), (64, 1.0/32), 'M', (128, 1.0/16), (128, 1.0/16), 'M', (256, 1.0/8), (256, 1.0/8), (256, 1.0/8),
        'M', (512, 1.0/4), (512, 1.0/4), (512, 1.0/4), 'M', (512, 1.0/2), (512, 1.0/2), (512, 1.0/2), 'M'],
    'D3': [(64, 0.1), (64, 0.1), 'M', (128, 0.5), (128, 0.5), 'M', (256, 1), (256, 1), (256, 1),
        'M', (512, 1), (512, 1), (512, 1), 'M', (512, 1), (512, 1), (512, 1), 'M'],
    'D2': [(64, 0.01), (64, 0.01), 'M', (128, 1), (128, 1), 'M', (256, 1), (256, 1), (256, 1),
        'M', (512, 1), (512, 1), (512, 1), 'M', (512, 1), (512, 1), (512, 1), 'M'],
    'D1': [(64, 0.1), (64, 0.1), 'M', (128, 1), (128, 1), 'M', (256, 1), (256, 1), (256, 1),
        'M', (512, 1), (512, 1), (512, 1), 'M', (512, 1), (512, 1), (512, 1), 'M'],
    'D0': [(64, 1), (64, 1), 'M', (128, 1), (128, 1), 'M', (256, 1), (256, 1), (256, 1),
        'M', (512, 1), (512, 1), (512, 1), 'M', (512, 1), (512, 1), (512, 1), 'M'],
    'G':[(64, 1.0/32), (64, 1.0/32), 'M', (128, 1.0/16), (128, 1.0/16), 'M', (256, 1.0/8), (256, 1.0/8), (256, 1.0/8),
        'M', (512, 1.0/4), (512, 1.0/4), (512, 1.0/4), 'M', (512, 1.0/2), (512, 1.0/2), (512, 1.0/2), 'M'], # VGG 16 with one fewer FC
    'G5': [(64, 1.0/32), (64, 1.0/32), 'M', (128, 1.0/16), (128, 1.0/16), 'M', (256, 1.0/8), (256, 1.0/8), (256, 1.0/8), (256, 1.0/8),
        'M', (512, 1.0/4), (512, 1.0/4), (512, 1.0/4), (512, 1.0/4), 'M', (512, 1.0/2), (512, 1.0/2), (512, 1.0/2), (512, 1.0/2), 'A']
}


def _adapt_shape(src_shape, x_shape):
    # to distinguish conv layers and fc layers
    # see if we need to expand the dimension of x
    new_shape = src_shape if len(src_shape) == 2 else (1, src_shape[0])
    if len(x_shape) > 2:
        new_shape = list(new_shape)
        new_shape += [1 for i in range(len(x_shape) - 2)]
    return new_shape


def _get_mask_hard(z_mu, z_logD, eps, threshold=0):
    logalpha = z_logD.data - torch.log(z_mu.data.pow(2) + eps)
    # NOTE(brendan): logalpha < threshold => -log mu^2/sigma^2 < 0 =>
    # sigma^2/mu^2 < 1.
    #
    # So, hard_mask == 1 if |mu| > |sigma|, 0 otherwise.
    hard_mask = (logalpha < threshold).float()
    return hard_mask


def _reparametrize(mu, logvar, batch_size, cuda=False, sampling=True):
    # output dim: batch_size * dim
    if sampling:
        std = logvar.mul(0.5).exp_()
        # NOTE(brendan): one sample per batch..
        eps = torch.FloatTensor(std.size(0), std.size(1)).cuda(mu.get_device()).normal_()

        return mu.view(*mu.shape) + eps*std.view(*std.shape)
    else:
        return mu.view(*mu.shape)


class VGG_IB(nn.Module):
    def __init__(self, config=None, mag=9, batch_norm=False, threshold=0,
                init_var=0.01, sample_in_training=True, sample_in_testing=False, n_cls=10, no_ib=False):
        super(VGG_IB, self).__init__()

        self.eps = 1e-8
        self.init_mag = mag
        self.threshold = threshold
        self.config = config
        self.init_var = init_var
        self.sample_in_training = sample_in_training
        self.sample_in_testing = sample_in_testing
        self.no_ib = no_ib

        self.conv_layers = self.make_conv_layers(cfg[config], batch_norm)
        print('Using structure {}'.format(cfg[config]))

        self.n_cls = n_cls
        if self.config in ['G', 'D6']:
            fc_layer_list = [nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, self.n_cls)]
            self.fc_layers = nn.Sequential(*fc_layer_list)
        elif self.config == 'G5':
            self.fc_layers = nn.Sequential(nn.Linear(512, self.n_cls))
        else:
            fc_layer_list = [nn.Linear(512, 512),
                             nn.ReLU(),
                             nn.Linear(512, 512),
                             nn.ReLU(),
                             nn.Linear(512, self.n_cls)]
            self.fc_layers = nn.Sequential(*fc_layer_list)

        z_mu_all = []
        z_logD_all = []
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                shape = m.weight.shape[:2]
                z_mu = torch.nn.Parameter(torch.Tensor(shape))
                z_logD = torch.nn.Parameter(torch.Tensor(shape))

                z_mu.data.normal_(1, init_var)
                z_logD.data.normal_(-self.init_mag, self.init_var)

                z_mu_all.append(z_mu)
                z_logD_all.append(z_logD)

        self.z_mu_all = torch.nn.ParameterList(z_mu_all)
        self.z_logD_all = torch.nn.ParameterList(z_logD_all)

    def make_conv_layers(self, config, batch_norm):
        layers = []
        in_channels = 3
        for v in config:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v == 'A':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v[0], kernel_size=3, padding=1)
                in_channels = v[0]
                ib = InformationBottleneck(v[0], mask_thresh=self.threshold, init_mag=self.init_mag, init_var=self.init_var,
                    kl_mult=v[1], sample_in_training=self.sample_in_training, sample_in_testing=self.sample_in_testing)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v[0]), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                if not self.no_ib:
                    layers.append(ib)
                    # kl_list.append(ib)
        return nn.Sequential(*layers)

    def forward(self, x):
        batch_size = x.size(0)

        conv_i = 0
        lyr_cfgs = [v for v in cfg[self.config] if v not in ['M', 'A']]
        convs = [m for m in self.conv_layers if isinstance(m, nn.Conv2d)]
        assert len(convs) == len(lyr_cfgs), f'{len(convs)} {len(lyr_cfgs)}'
        assert len(self.z_mu_all) == len(lyr_cfgs)
        kl_list = []
        for lyr in self.conv_layers:
            if not isinstance(lyr, nn.Conv2d):
                x = lyr(x)
                continue

            # NOTE(brendan): IB
            z_mu = self.z_mu_all[conv_i]
            z_logD = self.z_logD_all[conv_i]
            if (self.training and self.sample_in_training) or (not self.training and self.sample_in_testing):
                z_scale = _reparametrize(z_mu,
                                         z_logD,
                                         x.size(0),
                                         cuda=True,
                                         sampling=True)
                if not self.training:
                    z_scale *= _get_mask_hard(z_mu,
                                              z_logD,
                                              self.eps,
                                              self.threshold)

                new_shape = _adapt_shape(z_mu.size(), x.size())

                h_D = torch.exp(z_logD.view(new_shape))
                h_mu = z_mu.view(new_shape)

                kld = torch.sum(torch.log(1 + h_mu.pow(2)/(h_D + self.eps) )) * x.size(1) / h_D.size(1)

                if x.dim() > 2:
                    kld *= np.prod(x.size()[2:])
                kld = kld * 0.5 * lyr_cfgs[conv_i][1]
                kl_list.append(kld)
            else:
                logalpha = z_logD.data - torch.log(z_mu.data.pow(2) + self.eps)
                z_scale = (logalpha < self.threshold).float()*z_mu.data

            w = lyr.weight * z_scale.view(*z_scale.shape, 1, 1)
            x = F.conv2d(x,
                         w,
                         bias=lyr.bias,
                         padding=lyr.padding,
                         dilation=lyr.dilation,
                         groups=lyr.groups)

            conv_i += 1

        x = x.view(batch_size, -1)
        x = self.fc_layers(x)

        if self.training and (not self.no_ib):
            return x, sum(kl_list)
        else:
            return x

    def print_compression_ratio(self, threshold, writer=None, epoch=-1):
        # applicable for structures with global pooling before fc

        masks = []
        for z_mu, z_logD in zip(self.z_mu_all, self.z_logD_all):
            m = _get_mask_hard(z_mu, z_logD, self.eps, threshold)
            masks.append(m)
        prune_stat = [np.sum(m.cpu().numpy() == 0) for m in masks]

        conv_shapes = [v[0]
                       for v in cfg[self.config] if not isinstance(v, str)]

        if self.config in ['G', 'D6']:
            fc_shapes = [512]
        elif self.config == 'G5':
            fc_shapes = []
        else:
            fc_shapes = [512, 512]

        current_n, hdim, last_channels, flops, fmap_size = 0, 64, 3, 0, 32
        for n, pruned_connections in enumerate(prune_stat):
            if n < len(conv_shapes):
                current_channels = cfg[self.config][current_n][0]
                current_connections = current_channels*last_channels - pruned_connections
                flops += (fmap_size**2) * 9 * current_connections
                last_channels = current_channels
                current_n += 1
                if isinstance(cfg[self.config][current_n], str):
                    current_n += 1
                    fmap_size /= 2
                    hdim *= 2
            else:
                current_connections = 512*last_channels - pruned_connections
                flops += current_connections
                last_channels = 512
        flops += last_channels * self.n_cls

        total_params, pruned_params, remain_params = 0, 0, 0
        # total number of conv params
        in_channels = 3
        for n, n_out in enumerate(conv_shapes):
            n_params = in_channels * n_out * 9
            total_params += n_params
            n_remain = n_params - 9*prune_stat[n]
            remain_params += n_remain
            pruned_params += n_params - n_remain
            in_channels = n_out

        # fc layers
        offset = len(prune_stat) - len(fc_shapes)
        for n, n_out in enumerate(fc_shapes):
            n_params = in_channels * n_out
            total_params += n_params
            n_pruned = prune_stat[n + offset]
            n_remain = n_params - n_pruned
            remain_params += n_remain
            pruned_params += n_pruned
            in_channels = n_out
        total_params += in_channels * self.n_cls
        remain_params += in_channels * self.n_cls
        # TODO(brendan): prune final classification layer, for fair comparison?

        print('total parameters: {}, pruned parameters: {}, remaining params:{}, remain/total params:{}, remaining flops: {}, '
              'each layer pruned: {}'.format(total_params, pruned_params, remain_params,
                    float(total_params - pruned_params)/total_params, flops, prune_stat))
        if writer is not None:
            writer.add_scalar('flops', flops, epoch)
