import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

def add_singletons_like(v: Tensor, x_shape: tuple) -> Tensor:
    ## get batchsize (b) and data dimention (d ...)
    b, *d = x_shape
    ## add singletons for each dim in data dim
    return v.reshape(b, *[1] * len(d))

def extract(v: Tensor, t: Tensor, x_shape: tuple) -> Tensor:
    return add_singletons_like(v.gather(0, t), x_shape)

def linear_beta_schedule(timesteps: int) -> Tensor:
    return torch.linspace(1e-4, 0.02, timesteps)

def scaled_linear_beta_schedule(timesteps: int) -> Tensor:
    '''
    linear schedule, proposed in original ddpm paper
    '''
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps, s = 0.008):
    '''
    https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/classifier_free_guidance.py
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    '''
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class DiffusionSampler(nn.Module):
    def __init__(self, timesteps: int, beta_schedule: str):
        super().__init__()

        ## computing scheduling parameters
        if beta_schedule == 'linear':
            beta = linear_beta_schedule(timesteps)
        elif beta_schedule == 'scaled_linear':
            beta = scaled_linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            beta = cosine_beta_schedule(timesteps)
        else:
            raise ValueError('Unrecognized beta_schedule.')
        alpha = 1.0 - beta
        alpha_bar = alpha.cumprod(dim = 0)
        alpha_bar_prev = F.pad(alpha_bar[:-1], (1, 0), value = 1.)

        ## adding as non trainable parameters
        self.register_buffer('beta', beta)
        self.register_buffer('alpha', alpha)
        self.register_buffer('alpha_bar', alpha_bar)
        self.register_buffer('alpha_bar_prev', alpha_bar_prev)

    @property
    def device(self):
        return self.beta.device
    
    @property
    def timesteps(self):
        return len(self.beta)
    
    # @torch.no_grad()
    def q_sample(self, x0: Tensor, t: Tensor, noise: Tensor = None):
        if noise is None:
            noise = torch.randn_like(x0)
        alpha_bar = extract(self.alpha_bar, t, x0.shape)
        xt = alpha_bar.sqrt() * x0 + (1. - alpha_bar).sqrt() * noise
        return xt, noise
    
    @torch.no_grad()
    def p_sample(self, model: nn.Module, xt: Tensor, t: Tensor) -> Tensor:
        beta               = extract(self.beta, t, xt.shape)
        alpha              = extract(self.alpha, t, xt.shape)
        alpha_bar          = extract(self.alpha_bar, t, xt.shape)
        alpha_bar_prev     = extract(self.alpha_bar_prev, t, xt.shape)
        not_first_timestep = add_singletons_like(t > 0, xt.shape)

        epsilon = model(xt, t)
        z = torch.randn_like(xt)
        var = beta * (1. - alpha_bar_prev) / (1. - alpha_bar)
        mu = (1.0 / alpha.sqrt()) * (xt - epsilon * beta / (1.0 - alpha_bar).sqrt())
        return (mu + not_first_timestep * var.sqrt() * z)
    
    @torch.no_grad()
    def p_sample_loop(self, model: nn.Module, xT: Tensor, num_images: int = 1, prog = True) -> Tensor:

        xt = xT.clone()

        if num_images > 1:
            dt = int(self.timesteps / num_images)
            sequence = []

        for i in tqdm(reversed(range(self.timesteps)), disable = not prog):
            t = torch.ones(xT.shape[0], device = xT.device).long() * i
            xt = self.p_sample(model, xt, t)

            if num_images > 1 and (i % dt == 0 or i == self.timesteps - 1):
                sequence.append(xt)

        if num_images > 1:
            return sequence
        else:
            return xt
        
class ConditionalDiffusionSampler(DiffusionSampler):
    '''
    Partially from:
    https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/classifier_free_guidance.py
    '''
    def __init__(self, timesteps: int, beta_schedule: str):
        super().__init__(timesteps, beta_schedule)
        
    @torch.no_grad()
    def p_sample(self, model: nn.Module, xt: Tensor, c: Tensor, t: Tensor,  guidance_strength: float) -> Tensor:
        beta               = extract(self.beta, t, xt.shape)
        alpha              = extract(self.alpha, t, xt.shape)
        alpha_bar          = extract(self.alpha_bar, t, xt.shape)
        alpha_bar_prev     = extract(self.alpha_bar_prev, t, xt.shape)
        not_first_timestep = add_singletons_like(t > 0, xt.shape)

        ## equation 6 in classsifier free guidance paper
        epsilon = (1.0 + guidance_strength) * model(xt, c, t, cond_drop_prob = 0.0) - guidance_strength * model(xt, c, t, cond_drop_prob = 1.0)

        z = torch.randn_like(xt)
        var = beta * (1. - alpha_bar_prev) / (1. - alpha_bar)
        mu = (1.0 / alpha.sqrt()) * (xt - epsilon * beta / (1.0 - alpha_bar).sqrt())
        return (mu + not_first_timestep * var.sqrt() * z)
    
    @torch.no_grad()
    def p_sample_loop(self, model: nn.Module, xT: Tensor, c: Tensor, guidance_strength: float, num_images: int = 1, prog = True) -> Tensor:

        xt = xT.clone()

        if num_images > 1:
            dt = int(self.timesteps / num_images)
            sequence = []

        for i in tqdm(reversed(range(self.timesteps)), disable = not prog):
            t = torch.ones(xT.shape[0], device = xT.device).long() * i
            xt = self.p_sample(model, xt, c, t, guidance_strength)

            if num_images > 1 and (i % dt == 0 or i == self.timesteps - 1):
                sequence.append(xt)

        if num_images > 1:
            return sequence
        else:
            return xt

if __name__ == '__main__':
    from wdm_src.diffusion import UNet1d

    device = torch.device(0) if torch.cuda.is_available() else 'cpu'

    unet = UNet1d().to(device)

    timesteps = 300

    sampler1_type = 'linear'
    sampler2_type = 'scaled_linear'
    sampler3_type = 'cosine'

    sampler1 = DiffusionSampler(timesteps, sampler1_type).to(device)
    sampler2 = DiffusionSampler(timesteps, sampler2_type).to(device)
    sampler3 = DiffusionSampler(timesteps, sampler3_type).to(device)

    t = torch.randint(0, timesteps, (32, ), device=device).long()
    x0 = torch.randn(32, 2, 48, device = device)

    xT, noise = sampler2.q_sample(x0, t)
    x0_hat = sampler2.p_sample_loop(unet, xT)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1)
    ax.plot(sampler1.alpha_bar.cpu(), label = sampler1_type)
    ax.plot(sampler2.alpha_bar.cpu(), label = sampler2_type)
    ax.plot(sampler3.alpha_bar.cpu(), label = sampler3_type)
    ax.set_xlabel('timestep')
    ax.set_ylabel('alpha_bar')
    ax.legend()
    plt.show()
