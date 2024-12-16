from functools import partial

import torch
from torch import nn, Tensor
from einops import rearrange, repeat

from wdm_src.diffusion_models.unet.unet1d import Block, ResBlock, SinusoidalEmbeddings, LinearAttention, Attention, DownBlock, UpBlock

