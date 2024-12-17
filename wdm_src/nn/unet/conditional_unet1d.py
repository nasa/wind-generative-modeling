import torch
from torch import nn, Tensor
from einops import rearrange, repeat

from wdm_src.nn.unet.unet1d import Block, SinusoidalEmbeddings, LinearAttention, Attention, DownBlock, UpBlock

class ConditionalResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_emb_dim: int, time_emb_dim):
        super().__init__()

        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(cond_emb_dim + time_emb_dim, out_channels * 2))
        self.block1 = Block(in_channels, out_channels)
        self.block2 = Block(out_channels, out_channels)
        self.skip = nn.Conv1d(in_channels, out_channels, kernel_size = 1)
        
    def forward(self, x: Tensor, cond_emb: Tensor, time_emb: Tensor):
        emb = self.mlp(torch.cat([cond_emb, time_emb], dim = -1))
        emb = rearrange(emb, 'b c -> b c 1')
        scale, shift = emb.chunk(2, dim = 1)

        h = self.block1(x, scale, shift)
        return self.block2(h) + self.skip(x)

def uniform(shape, device):
    return torch.zeros(shape, device = device).float().uniform_(0, 1)

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

class ConditionalUNet1d(nn.Module):
    '''
    https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/classifier_free_guidance.py#L28
    '''
    def __init__(self,
                num_cond: int, ## this variable can be removed now that aggregation is used on conditioning embeddings
                num_embeddings: int,
                cond_drop_prob: float = 0.0,
                in_channels: int = 2, 
                cond_emb_dim: int = 128,
                time_emb_dim: int = 128, 
                heads: int = 4, 
                dim_head: int = 32,
                theta: int = 10000
                ):
        ## calling the super constructor of the UNet1d
        super().__init__()
        assert 0.0 <= cond_drop_prob <= 1.0
        self.cond_drop_prob = cond_drop_prob

        ## define the embedder and mlp for the conditioning information
        self.cond_emb = nn.Embedding(num_embeddings, cond_emb_dim)
        self.cond_mlp = nn.Sequential(
            # nn.Linear(num_cond * cond_emb_dim, cond_emb_dim),
            nn.Linear(cond_emb_dim, cond_emb_dim),
            nn.GELU(),
            nn.Linear(cond_emb_dim, cond_emb_dim))
        ## conditioning embedding reserved for unconditional sampling
        self.null_cond_emb = nn.Parameter(torch.randn(cond_emb_dim))

        self.time_mlp = nn.Sequential(
            SinusoidalEmbeddings(time_emb_dim, theta = theta),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        self.input_layer = nn.Conv1d(in_channels, 32, kernel_size = 1)

        self.downs = nn.ModuleList([
            nn.ModuleList([
                ConditionalResBlock(32, 32, cond_emb_dim, time_emb_dim),
                ConditionalResBlock(32, 32, cond_emb_dim, time_emb_dim),
                LinearAttention(32, heads, dim_head),
                DownBlock(32, 32)
            ]),

            nn.ModuleList([
                ConditionalResBlock(32, 32, cond_emb_dim, time_emb_dim),
                ConditionalResBlock(32, 32, cond_emb_dim, time_emb_dim),
                LinearAttention(32, heads, dim_head),
                DownBlock(32, 64)
            ]),

            nn.ModuleList([
                ConditionalResBlock(64, 64, cond_emb_dim, time_emb_dim),
                ConditionalResBlock(64, 64, cond_emb_dim, time_emb_dim),
                LinearAttention(64, heads, dim_head),
                nn.Conv1d(64, 128, kernel_size = 3, padding = 1)
            ])
        ])

        self.mid_block1 = ConditionalResBlock(128, 128, cond_emb_dim, time_emb_dim)
        self.mid_attention = Attention(128, heads, dim_head)
        self.mid_block2 = ConditionalResBlock(128, 128, cond_emb_dim, time_emb_dim)

        self.ups = nn.ModuleList([
            nn.ModuleList([
                ConditionalResBlock(128 + 64, 128, cond_emb_dim, time_emb_dim),
                ConditionalResBlock(128 + 64, 128, cond_emb_dim, time_emb_dim),
                LinearAttention(128, heads, dim_head),
                UpBlock(128, 64)
            ]),

            nn.ModuleList([
                ConditionalResBlock(64 + 32, 64, cond_emb_dim, time_emb_dim),
                ConditionalResBlock(64 + 32, 64, cond_emb_dim, time_emb_dim),
                LinearAttention(64, heads, dim_head),
                UpBlock(64, 32)
            ]),

            nn.ModuleList([
                ConditionalResBlock(32 + 32, 32, cond_emb_dim, time_emb_dim),
                ConditionalResBlock(32 + 32, 32, cond_emb_dim, time_emb_dim),
                LinearAttention(32, heads, dim_head),
                nn.Conv1d(32, 32, kernel_size = 3, padding = 1)
            ])
        ])

        self.output_res = ConditionalResBlock(32 + 32, 32, cond_emb_dim, time_emb_dim)
        self.output_layer = nn.Conv1d(32, in_channels, kernel_size = 1)

    def forward(self, x: Tensor, c: Tensor, t: Tensor, cond_drop_prob: float = None):
        ## tensor axes for clarity
        batch_size, channels, elements = x.shape
        batch_size, num_cond = c.shape

        ## if its unspecified in the function use the default
        cond_drop_prob = self.cond_drop_prob if cond_drop_prob == None else cond_drop_prob

        ## embedd the int value conditions into their embedding vectors
        cond_emb = self.cond_emb(c)
        
        if cond_drop_prob > 0.0:
            ## determine which samples in the batch to treat as unconditional 
            keep_mask = prob_mask_like((batch_size, num_cond), 1.0 - cond_drop_prob, device = x.device)
            ## replace the embeddings
            null_cond_emb = repeat(self.null_cond_emb, 'd -> b c d', c = num_cond, b = batch_size)
            cond_emb = torch.where(
                rearrange(keep_mask, 'b c -> b c 1'),
                cond_emb,
                null_cond_emb)
            
        ## stack the embeddings of each condition
        # cond_emb = torch.flatten(cond_emb, start_dim = 1)
        cond_emb = torch.sum(cond_emb, dim = 1)
        ## pass the embeddings through the cond mlp
        cond_emb = self.cond_mlp(cond_emb)

        time_emb = self.time_mlp(t) ## (b x time_emb_dim)
        y = self.input_layer(x) ## (b x 32 x l)
        r = y.clone()
        
        residuals = []
        for res1, res2, attention, downsample in self.downs:
            y = res1(y, cond_emb, time_emb)
            residuals.append(y)
            y = res2(y, cond_emb, time_emb)
            y = attention(y) + y
            residuals.append(y)
            y = downsample(y)

        ## (b x 128 x l)
        y = self.mid_block1(y, cond_emb, time_emb)
        y = self.mid_attention(y) + y
        y = self.mid_block2(y, cond_emb, time_emb)

        for res1, res2, attention, upsample in self.ups:
            y = res1(torch.cat((y, residuals.pop()), dim = 1), cond_emb, time_emb)
            y = res2(torch.cat((y, residuals.pop()), dim = 1), cond_emb, time_emb)
            y = attention(y) + y
            y = upsample(y)

        ## final skip connection to residual layer
        y = self.output_res(torch.cat((y, r), dim = 1), cond_emb, time_emb)
        y = self.output_layer(y)
        return y