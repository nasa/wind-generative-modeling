'''
Code sources:
https://www.youtube.com/watch?v=ZBKpAp_6TGI&t=1841s
https://huggingface.co/blog/annotated-diffusion
'''
import math
import torch
from torch import nn, Tensor, einsum
from einops.layers.torch import Rearrange
from einops import rearrange
import numpy as np

def wind_velocity_unet1d_transform(x: np.array) -> np.array:
    x = np.stack([x[:47], x[47:]], axis = 0) ## u and v are one vector by default, we split them into channels
    x = np.pad(x, ((0, 0), (0, 1))) ## default is 47 altitudes, we need an even number
    return x

def exists(x):
    return x is not None

class SinusoidalEmbeddings(nn.Module):
    def __init__(self, dim: int, theta: int = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels):
        super().__init__()
        self.rearrange = Rearrange('b c (l p1) -> b (c p1) l', p1 = 2)
        self.conv = nn.Conv1d(in_channels * 2, out_channels, kernel_size = 3, padding = 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(self.rearrange(x))

class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor = 2)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size = 3, padding = 1)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.conv(self.upsample(x))

class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size = 3, padding = 1)
        self.norm = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()

    def forward(self, x: Tensor, scale: Tensor = None, shift: Tensor = None) -> Tensor:
        x = self.norm(self.conv(x))
        if exists(scale) and exists(shift):
            x = x * (scale + 1.0) + shift
        return self.act(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()

        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels * 2))
        self.block1 = Block(in_channels, out_channels)
        self.block2 = Block(out_channels, out_channels)
        self.skip = nn.Conv1d(in_channels, out_channels, kernel_size = 1)
        
    def forward(self, x: Tensor, time_emb: Tensor):
        time_emb = self.time_mlp(time_emb)
        time_emb = rearrange(time_emb, 'b c -> b c 1')
        scale, shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale, shift)
        return self.block2(h) + self.skip(x)

class RMSNorm(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, in_channels, 1))

    def forward(self, x: Tensor) -> Tensor:
        return nn.functional.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)

## https://huggingface.co/blog/annotated-diffusion
class LinearAttention(nn.Module):
    '''
    Computes global attention via a weird hack.
    '''
    def __init__(self, in_channels: int, heads: int, dim_head: int):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        hidden_dim = heads * dim_head

        self.norm = RMSNorm(in_channels)
        self.qkv = nn.Conv1d(in_channels, 3*hidden_dim, kernel_size = 1, bias = False)
        self.output = nn.Sequential(
            nn.Conv1d(hidden_dim, in_channels, kernel_size = 1),
            RMSNorm(in_channels)
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, l = x.shape

        x = self.norm(x)
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim = 1)

        ## takes the channel dimention and reshapes into heads, channels
        ## also flattens the feature maps into vectors
        ## the shape is now b h c d where d is dim_head
        q = rearrange(q, 'b (h c) l -> b h c l', h = self.heads)
        k = rearrange(k, 'b (h c) l -> b h c l', h = self.heads)
        v = rearrange(v, 'b (h c) l -> b h c l', h = self.heads)
        ## softmax along dim_head dim
        q = q.softmax(dim = 2) * self.scale
        ## softmax along flattened image dim
        k = k.softmax(dim = 3)
        ## compute comparison betweeen keys and values to produce context.
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c l -> b (h c) l')
        return self.output(out)

## https://huggingface.co/blog/annotated-diffusion
class Attention(nn.Module):
    '''
    Computes full pixelwise attention.
    '''
    def __init__(self, in_channels: int, heads: int, dim_head: int):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        hidden_dim = heads * dim_head

        self.norm = RMSNorm(in_channels)
        self.qkv = nn.Conv1d(in_channels, hidden_dim * 3, kernel_size = 1, bias = False)
        self.output = nn.Sequential(
            nn.Conv1d(hidden_dim, in_channels, kernel_size = 1),
            RMSNorm(in_channels)
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, l = x.shape

        x = self.norm(x)

        ## compute the queries, keys, and values of the incoming feature maps
        q, k, v = torch.chunk(self.qkv(x), 3, dim = 1)
        ## takes the channel dimention and reshapes into heads, channels
        q = rearrange(q, 'b (h c) l -> b h c l', h = self.heads)
        k = rearrange(k, 'b (h c) l -> b h c l', h = self.heads)
        v = rearrange(v, 'b (h c) l -> b h c l', h = self.heads)
        q = q * self.scale
        ## multiplication of the query and key matrixes for each head
        sim = einsum('b h c i, b h c j -> b h i j', q, k)
        ## subtract the maximum value of each row
        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        ## make each row into a probability distribution
        attn = sim.softmax(dim = -1)
        ## weight the values according to the rows of the attention matrix
        y = einsum('b h i j, b h d j -> b h i d', attn, v)
        ## reshape the weighted values back into feature maps
        y = rearrange(y, 'b h l d -> b (h d) l')
        return self.output(y)

class UNet1d(nn.Module):
    def __init__(self, in_channels: int = 2, time_emb_dim: int = 128, heads: int = 4, dim_head: int = 32):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        self.input_layer = nn.Conv1d(in_channels, 32, kernel_size = 1)

        self.downs = nn.ModuleList([
            nn.ModuleList([
                ResBlock(32, 32, time_emb_dim),
                ResBlock(32, 32, time_emb_dim),
                LinearAttention(32, heads, dim_head),
                DownBlock(32, 32)
            ]),

            nn.ModuleList([
                ResBlock(32, 32, time_emb_dim),
                ResBlock(32, 32, time_emb_dim),
                LinearAttention(32, heads, dim_head),
                DownBlock(32, 64)
            ]),

            nn.ModuleList([
                ResBlock(64, 64, time_emb_dim),
                ResBlock(64, 64, time_emb_dim),
                LinearAttention(64, heads, dim_head),
                nn.Conv1d(64, 128, kernel_size = 3, padding = 1)
            ])
        ])

        self.mid_block1 = ResBlock(128, 128, time_emb_dim)
        self.mid_attention = Attention(128, heads, dim_head)
        self.mid_block2 = ResBlock(128, 128, time_emb_dim)

        self.ups = nn.ModuleList([
            nn.ModuleList([
                ResBlock(128 + 64, 128, time_emb_dim),
                ResBlock(128 + 64, 128, time_emb_dim),
                LinearAttention(128, heads, dim_head),
                UpBlock(128, 64)
            ]),

            nn.ModuleList([
                ResBlock(64 + 32, 64, time_emb_dim),
                ResBlock(64 + 32, 64, time_emb_dim),
                LinearAttention(64, heads, dim_head),
                UpBlock(64, 32)
            ]),

            nn.ModuleList([
                ResBlock(32 + 32, 32, time_emb_dim),
                ResBlock(32 + 32, 32, time_emb_dim),
                LinearAttention(32, heads, dim_head),
                nn.Conv1d(32, 32, kernel_size = 3, padding = 1)
            ])
        ])

        self.output_res = ResBlock(32 + 32, 32, time_emb_dim)
        self.output_layer = nn.Conv1d(32, in_channels, kernel_size = 1)
    
    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        b, c, l = x.shape

        time_emb = self.time_mlp(t) ## (b x time_emb_dim)
        y = self.input_layer(x) ## (b x 32 x l)
        r = y.clone()
        
        residuals = []
        for res1, res2, attention, downsample in self.downs:
            y = res1(y, time_emb)
            residuals.append(y)
            y = res2(y, time_emb)
            y = attention(y) + y
            residuals.append(y)
            y = downsample(y)

        ## (b x 128 x l)
        y = self.mid_block1(y, time_emb)
        y = self.mid_attention(y) + y
        y = self.mid_block2(y, time_emb)

        for res1, res2, attention, upsample in self.ups:
            y = res1(torch.cat((y, residuals.pop()), dim = 1), time_emb)
            y = res2(torch.cat((y, residuals.pop()), dim = 1), time_emb)
            y = attention(y) + y
            y = upsample(y)

        ## final skip connection to residual layer
        y = self.output_res(torch.cat((y, r), dim = 1), time_emb)
        y = self.output_layer(y)
        return y