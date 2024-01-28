from collections import namedtuple
import math
from functools import partial
import numpy as np

from einops import rearrange, reduce
import torch
from torch import nn, einsum
import torch.nn.functional as F

# from dnnlib.util import EasyDict
from typing import Iterable
from torch_utils import misc
from torch_utils import persistence
from torch_utils.ops import conv2d_resample
from torch_utils.ops import bias_act
from torch_utils.ops import fma
from torch_utils.ops import upfirdn2d


# def_comb = EasyDict(type='mult', degree=-1)
# specs = { d['type']: d for d in [
#     dict(type='none',     dequant='gauss', noise=0.0,                             dims=0, lr=1, lin_lr=1e-2),
#     dict(type='concat',   dequant='gauss', noise=0.0, noise_f_int=[],             dims=1, lr=1, lin_lr=1e-2), # 1 frame overlap in conditioning
#     dict(type='fourier',  dequant='gauss', noise=0.0, noise_f_int=[], noise_f=[], dims=2, lr=1, lin_lr=1e-2, f_manual=[], include_lin=True),
#     dict(type='f_concat', dequant='gauss', noise=0.0, noise_f_int=[], noise_f=[], dims=2, lr=1, lin_lr=1e-2, f_manual=[], include_lin=True), # fourier features given to concat
# ]}

def _set_noise_global(c, noise, n_frames, n_days):
    scale = parse_noises([noise], c, n_frames, n_days)[0]
    c.cond_args.noise = scale # global noise, measured in frame deltas
    c.cond_args.noise_f = [] # disable per-f noise
    return f'-n{str(noise).replace(" ", "")}'

def _set_noise_per_f(c, noises, n_frames, n_days):
    scales = parse_noises(noises, c, n_frames, n_days)
    assert np.all(np.diff(scales) <= 0), 'Noises not descending, are you sure?'
    c.cond_args.noise = 0        # disable global noise
    c.cond_args.noise_f = scales # in frame deltas, for [1e-3f] + [manual freqs]
    return '-n_' + '_'.join([str(f).replace(' ', '') for f in noises])

def _set_freqs(c, base_freqs, explicit_lin):
    assert c.cond_args.type in ['fourier', 'f_concat']

    desc = ''
    if explicit_lin:
        c.cond_args.f_manual = [*base_freqs]
        c.cond_args.include_lin = True
        c.cond_args.dims = 2*(len(c.cond_args.f_manual) + 1)

        desc += '-fman_lin'
    else:
        c.cond_args.f_manual = [1e-3, *base_freqs]
        c.cond_args.include_lin = False
        c.cond_args.dims = 2*len(c.cond_args.f_manual)
        desc += '-fman_impl'

    if base_freqs:
        desc += "_" + "_".join([str(int(round(f))) for f in base_freqs])

    return desc

# Map noise magnitude to frame deltas
def days(fr_tot, d_tot):
    return fr_tot / d_tot # one sigma in both directions
def hours(fr_tot, d_tot):
    return days(fr_tot, d_tot) / 24
def weeks(fr_tot, d_tot):
    return days(fr_tot, d_tot) * 7
def months(fr_tot, d_tot):
    return days(fr_tot, d_tot) * (365.25/12) # avg days in month
def years(fr_tot, d_tot):
    return days(fr_tot, d_tot) * 365.25

# Convert strings like '2.5years' to sigmas
def parse_noises(noises, c=None, n_frames=None, n_days=None):
    ret = []
    for n in noises:
        if isinstance(n, (float, int)):
            ret.append(n)
        elif 'hour' in n:
            ret.append(hours(n_frames, n_days)*float(n.split('hour')[0]))
        elif 'day' in n:
            ret.append(days(n_frames, n_days)*float(n.split('day')[0]))
        elif 'week' in n:
            ret.append(weeks(n_frames, n_days)*float(n.split('week')[0]))
        elif 'month' in n:
            ret.append(months(n_frames, n_days)*float(n.split('month')[0]))
        elif 'year' in n:
            ret.append(years(n_frames, n_days)*float(n.split('year')[0]))
        else:
            raise RuntimeError(f'Unkown noise scale: {n}')

    assert len(ret) == len(noises)
    return ret

def cond_desc(c, days, cond_type, frames, num_days=100, f=[], noise=[], mask=None, explicit_lin=True):
    try:

        # Must know number of days in sequence
        days = num_days if days is None else days
        assert days is not None, 'Number of days not in dataset metadata, must specify manually with --days'

        c.cond_args = EasyDict(specs[cond_type])

        if cond_type in ['fourier', 'f_concat']:
            freqs = f or list(filter(lambda f: f > 1, [days/365.25, days])) # only cylces of over 1Hz
#             freqs = f or list(filter(lambda f: f > 1, [days])) # only cylces of over 1Hz
            _set_freqs(c, freqs, explicit_lin)

        if 'auto' in noise:
            noise_lin = [] if days < 365.25 else [f'{0.2 * days / 365.25:.2f} years'] # fifth of whole sequence length
            _set_noise_per_f(c, [*noise_lin, '4 days', 0], frames, days) # lin, years, days
        elif len(noise) == 1:
            _set_noise_global(c, noise, frames, days)
        elif isinstance(noise, Iterable) and len(noise) > 0:
            _set_noise_per_f(c, noise, frames, days) # lin, years, days

        return

    except IOError as err:
        raise click.ClickException(f'--data: {err}')
        

class ConditioningTransform(torch.nn.Module):
    def __init__(self,
        cond_args = {},   # Conditioning parameters.
        num_ws = None,    # Number of layers to broadcast c to.
        add_noise = False,
    ):
        super().__init__()
        self.cond_args = cond_args
        self.explicit_lin = cond_args.get('include_lin', False)
        self.num_ws = num_ws
        self.num_f = len(self.get_frequencies())
        self.add_noise = add_noise # only active in D

    def get_frequencies(self):
        if self.cond_args.type not in ['fourier', 'f_concat']:
            return []

        freqs = self.cond_args.f_manual
        if self.explicit_lin:
            freqs = [-1.0] + list(freqs)

        return np.sort(freqs).astype(np.float32)

    # Measured in mean frame intervals
    def add_noise_gauss(self, c, scales):
        assert len(scales) in [1, self.num_f]
        if self.training and self.add_noise:
            s = torch.tensor(scales, dtype=torch.float32, device=c.device)
            c = c + s*self.cond_args.t_delta*torch.randn_like(c) # expands last dim to len(scales)
        return c

    # Measured in cycles
    def add_noise_f_int(self, c, noise_tuples):
        if self.training and self.add_noise:
            scales = torch.tensor([s for s,_ in noise_tuples], dtype=torch.float32, device=c.device)
            ifreqs = torch.tensor([1/f for _,f in noise_tuples], dtype=torch.float32, device=c.device)
            noises = ifreqs*torch.round(scales*torch.randn(*c.shape, len(noise_tuples), device=c.device)) # (B, 1, n_noises)
            c = c + noises.sum(axis=-1)
        return c

    def check_shapes(self, c):
        assert self.cond_args.type in ['fourier', 'f_concat'] or self.cond_args.dims % 2 == 0, 'Fourier cond: number of dims not divisible by two'
        assert list(c.shape[1:]) in [[0], [1], [self.num_f], [self.num_ws, 1], [self.num_ws, self.num_f]], \
            f'Invalid c shape - supported: [(B, 1), (B, #freq), (B, #layer, 1), (B, #layer, #freq)]' # broadcast along trailing dimension

    def add_noises(self, c):
        # Global noise (separate from dequantization noise)
        c = self.add_noise_gauss(c, [self.cond_args.noise])

        # Add global integer jump noise at given frequnecies (any cond mechanism)
        if self.cond_args.get('noise_f_int'):
            c = self.add_noise_f_int(c, self.cond_args.noise_f_int)

        # Add per-freqency noise (fourier cond)
        # Expands and broadcasts along trailing dimension
        if self.cond_args.type in ['fourier', 'f_concat'] and self.cond_args.noise_f:
            c = self.add_noise_gauss(c, self.cond_args.noise_f)

        return c

    # Supports pre-broadcasted inputs:
    # [B, 1]: global c
    # [B, #layer, 1]: one c per layer
    # [B, #freq]: one c per frequency
    # [B, #layer, #freq]: one c per frequency per layer
    def forward(self, c, broadcast=True):
        self.check_shapes(c)
        self.add_noises(c)

        if self.cond_args.type in ['fourier', 'f_concat']:
            # Interleave cosines and sines
            freqs = torch.from_numpy(self.get_frequencies()).to(c.device)
            cos = torch.cos(2*np.pi*freqs*c) # [B, ca.dims/2]
            sin = torch.sin(2*np.pi*freqs*c) # [B, ca.dims/2]

            if self.explicit_lin:
                if c.ndim == 2:
                    cos[:, 0] = 1
                    sin[:, 0] = self.cond_args.lin_lr * c[:, 0]
                else:
                    cos[:, :, 0] = 1
                    sin[:, :, 0] = self.cond_args.lin_lr * c[:, :, 0]

            # Interleaved: [cos0, sin0, cos1, sin1, ...]
            c = torch.stack((cos, sin), dim=-1).view(*c.shape[:-1], -1)
        else:
            pass # c passed through unchanged

        if broadcast:
            assert self.num_ws is not None, 'num_ws not provided for broadcast'
            if c.ndim == 2: # not already broadcasted (due to per-layer input)
                c = c.unsqueeze(1).repeat_interleave(self.num_ws, dim=1)
            misc.assert_shape(c, [None, self.num_ws, self.cond_args.dims])

        return c


class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        bias            = True,     # Apply additive bias before the activation function?
        activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 1,        # Learning rate multiplier.
        bias_init       = 0,        # Initial value for the additive bias.
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        lr_scales = lr_multiplier / np.sqrt(in_features) # scalar or vector
        self.weight_gain = torch.nn.Parameter(lr_scales) if torch.is_tensor(lr_scales) else lr_scales
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x

    def extra_repr(self):
        return f'in_features={self.in_features:d}, out_features={self.out_features:d}, activation={self.activation:s}'


def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()
    
    
class CondScale(torch.nn.Module):
    def __init__(self, w_dim, c_dim, channels, cond_args):
        super().__init__()
        self.c_dim = c_dim
        self.cond_args = cond_args
        if w_dim is not None:
            self.w_affine = FullyConnectedLayer(w_dim, channels, bias_init=1)
        if cond_args.type == 'fourier':
            self.c_to_scales = FullyConnectedLayer(c_dim, channels, bias=False, lr_multiplier=cond_args.lr)
            self.c_to_scales.weight.data *= 1e-6
            self.c_to_scales.weight.data[:, 0] += 1 # init DC to ~1 (w passed through initially)

    def forward(self, w=None, c=None):
        if w is not None:
          styles = self.w_affine(w)
        else:
          styles=None

        if self.cond_args.type == 'fourier':
            scales = self.c_to_scales(c)
            if styles is not None:
              styles = styles * scales
            else: styles=scales

        return styles
   
        
class WeatherLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(WeatherLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden_size, output_size), nn.ReLU(), nn.Linear(output_size, output_size))

    def forward(self, weather, phase="train"):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, weather.size(0), self.hidden_size).to(weather.device)
        c0 = torch.zeros(self.num_layers, weather.size(0), self.hidden_size).to(weather.device)

        # Forward propagate LSTM
        out, _ = self.lstm(weather, (h0, c0))

        # Decode the hidden state of the last time step
        # if phase == "train":
        out = self.fc(out.reshape(out.shape[0] * out.shape[1], -1))
        # else:
        #     out = self.fc(out[:, -1, :])
        return out


class TimeEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        f0 = 1  # frequency for day cycle
        f1 = f0 / 365.25  # Compute the frequency for the year cycle
        k = 1e-2  # Set the scaling constant

        # Compute the positional embeddings
        c_d1 = torch.sin(2 * math.pi * f0 * time)
        c_d2 = torch.cos(2 * math.pi * f0 * time)

        c_y1 = torch.sin(2 * math.pi * f1 * time)
        c_y2 = torch.cos(2 * math.pi * f1 * time)

        # Combine the positional embeddings
        embeddings = torch.stack((c_d1,c_d2,c_y1,c_y2), dim=1)

        return embeddings


class AdaIN(nn.Module):
    def __init__(self, in_dim, w_dim):
        super(AdaIN, self).__init__()
        self.in_dim=in_dim
        self.norm = nn.InstanceNorm2d(in_dim)
        self.linear = nn.Linear(w_dim, in_dim * 2)

    def forward(self, x, w):
        x = self.norm(x)
        h = self.linear(w)
        h = h.view(h.size(0), self.in_dim*2)
        gamma, beta = h.chunk(2, 1)
        gamma = gamma.view(gamma.size(0), gamma.size(1), 1, 1)
        beta = beta.view(beta.size(0), beta.size(1), 1, 1)
        out = torch.add(torch.mul(x,(1 + gamma)), beta)
        return out
  
        
class SelfAttention(nn.Module):
    def __init__(self, num, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // (2*num), kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // (2*num), kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.out_conv = nn.Sequential(nn.Conv2d(in_channels=in_dim, out_channels=in_dim//num, kernel_size=1), nn.ReLU(), nn.Conv2d(in_channels=in_dim//num, out_channels=in_dim//num, kernel_size=1))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, c, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, c, width, height)
        out = self.gamma * out + x
        out=self.out_conv(out)
        return out

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()

    def forward(self, x):
        # First Conv
        x = self.bnorm1(self.relu(self.conv1(x)))
        # Second Conv
        x = self.bnorm2(self.relu(self.conv2(x)))
        return self.transform(x)
 

# helpers functions
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


# model
class UnetCond(nn.Module):
    def __init__(
        self,
        dim=64,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        resnet_block_groups = 8,
        w_dim=16,
        f_dim=1,
        t_dim=6,
        hidden_dim=64,
        num_layers=1,
        num_ws=1,
        cond_args={},
        device=torch.cuda.current_device()
    ):
        super().__init__()

        self.device=device

        # determine dimensions

        self.channels = channels
        input_channels = channels

        init_dim = default(init_dim, dim)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]

        in_out = list(zip(dims[:-1], dims[1:]))

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # mid_dim = dims[-1]
        mid_dim = 128
 
        default_out_dim = channels
        self.out_dim = default(out_dim, default_out_dim)

        # Convs
        self.convs = nn.ModuleList([nn.Conv2d(input_channels, init_dim, 7, padding = 3)])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.convs.append(ConvBlock(dim_in, dim_out) if not is_last else  nn.Sequential(nn.Conv2d(dim_in, mid_dim, 3, padding = 1),nn.ReLU(),nn.Flatten(1),nn.Linear(mid_dim*32*32,mid_dim)))

        self.w_mlp = nn.Sequential(WeatherLSTM(w_dim, hidden_dim, num_layers, mid_dim),\
#                                    nn.Linear(mid_dim, mid_dim), nn.ReLU(), nn.Linear(mid_dim, mid_dim)\
                                  )
        self.wadain = nn.ModuleList([nn.Sequential(AdaIN(mid_dim, mid_dim),\
                                                   nn.Conv2d(mid_dim, mid_dim, 1), nn.BatchNorm2d(mid_dim), nn.ReLU(), nn.Conv2d(mid_dim, mid_dim, 1)\
                                                  ) for i in range(2)])

        self.f_mlp = nn.Sequential(WeatherLSTM(f_dim, hidden_dim, num_layers, mid_dim),\
#                                    nn.Linear(mid_dim, mid_dim), nn.ReLU(), nn.Linear(mid_dim, mid_dim)\
                                  )
        self.fadain = nn.ModuleList([nn.Sequential(AdaIN(mid_dim, mid_dim),\
                                                   nn.Conv2d(mid_dim, mid_dim, 1), nn.BatchNorm2d(mid_dim), nn.ReLU(), nn.Conv2d(mid_dim, mid_dim, 1)\
                                                  ) for i in range(2)])
        # self.t_mlp = nn.Sequential(
        #         SinusoidalPositionEmbeddings(time_emb_dim),
        #         nn.Linear(time_emb_dim, output_dim),
        #         nn.ReLU(),
        #         nn.Linear(time_emb_dim, time_emb_dim)
        #     )
        self.tadain = nn.ModuleList([nn.Sequential(AdaIN(mid_dim, mid_dim),\
                                                   nn.Conv2d(mid_dim, mid_dim, 1), nn.BatchNorm2d(mid_dim), nn.ReLU(), nn.Conv2d(mid_dim, mid_dim, 1)\
                                                  ) for i in range(2)])

#         self.attention=nn.ModuleList([nn.Sequential(nn.Conv2d(4 * mid_dim,mid_dim,1),nn.ReLU(),nn.Conv2d(mid_dim,mid_dim,1),\
#                                       ) for i in range(2)])
#         self.attention = nn.ModuleList([SelfAttention(4, 4 * mid_dim) for i in range(2)])
#         self.attention = nn.ModuleList([SelfAttentionBlock(3 * mid_dim, mid_dim, num=3, dropout=0.1) for i in range(2)])
        # self.attention = nn.ModuleList([CrossAttentionBlock(mid_dim, mid_dim, num=3, dropout=0.1) for i in range(2)])

#         self.attention = nn.ModuleList([ViT(
#                         patch_dim=4,
#                         d_model=4 * mid_dim,
#                         d_ff=512,
#                         num_heads=4,
#                         num_layers=4,
#                         num_patches=64,
#                         num=4,
#                         device=device
#                     ) for i in range(2)])

        self.cond_args = cond_args
        # self.cond_args={'type': 'fourier', 'dequant': 'gauss', 'noise': 0, 'noise_f_int': [], 'noise_f': [14.8, 0], 'dims': 4, 'lr': 1, 'lin_lr': 0.01, 'f_manual': [100], 'include_lin': True}
        self.cond_xform = ConditioningTransform(cond_args=self.cond_args, num_ws=num_ws)
        self.scaled_styles = CondScale(None, t_dim, mid_dim, cond_args)

    def forward(self, mixed, phase="train"):
        # self.device="cuda:0"
        # img, flow, weather, time, label, _ = mixed
        img, flow, weather, time = mixed
        img=img.squeeze(0).to(self.device)
        weather=weather.squeeze(0).float().to(self.device)
        flow=flow.squeeze(0).to(self.device)
        time=time.squeeze(0).to(self.device)

        cs = self.cond_xform(time, broadcast=True)
        c_iter = iter(cs.unbind(dim=1))
        c=next(c_iter)

        if time is not None:
          #  time = self.t_mlp(time)
          time=self.scaled_styles(c=c)
          # time = F.relu(time)
          flow = self.f_mlp[0](flow, phase)
#           flow = self.f_mlp[1:](flow)
          # flow = F.relu(flow)
          weather = self.w_mlp[0](weather, phase)
#           weather = self.w_mlp[1:](weather)
          # weather = F.relu(weather)
          for conv_layer in self.convs:
            img = conv_layer(img)

          # weather_style = self.wadain[0][0](x, weather)
          # weather_style = self.wadain[0][1:](weather_style)
#           flow_style = self.fadain[0][0](x, flow)
#           flow_style = self.fadain[0][1:](flow_style)
          # time_style = self.tadain[0][0](x, time)
          # time_style = self.tadain[0][1:](time_style)
#           combined_features = torch.cat((x, img, weather_style), dim=1)
#           x = self.attention[0](combined_features)
          combined_features = torch.cat((img, weather, flow, time), dim=1)
          # combined_features = weather
          # attn = self.attention[0](x, combined_features)
        return combined_features.unsqueeze(1)

import pytorch_lightning as pl
class IdentityAutoEncoder(pl.LightningModule):
    def encode(self,x):
        return x 
    def decode(self,x):
        return x
