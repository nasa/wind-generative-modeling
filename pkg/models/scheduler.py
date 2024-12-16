import torch
import torch.nn.functional as F

class Scheduler():
    def __init__(self, timesteps, sched_type="cosine", s=0.008):
        self._betas = self._calculate_betas(sched_type, timesteps, s)
        self._timesteps = timesteps

        alphas = 1. - self._betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self._sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self._sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self._sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self._posterior_variance = self._betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    @property
    def betas(self):
        return self._betas

    @property
    def timesteps(self):
        return self._timesteps

    @property
    def sqrt_recip_alphas(self):
        return self._sqrt_recip_alphas

    @property
    def sqrt_alphas_cumprod(self):
        return self._sqrt_alphas_cumprod

    @property
    def sqrt_one_minus_alphas_cumprod(self):
        return self._sqrt_one_minus_alphas_cumprod

    @property
    def posterior_variance(self):
        return self._posterior_variance

    def _calculate_betas(self, sched_type, timesteps, s):
        if (sched_type == 'cosine'):
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps)
            alphas_cumprod = torch.cos(
                ((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0.0001, 0.9999)
        elif (sched_type == 'linear'):
            beta_start = 0.0001
            beta_end = 0.02
            betas = torch.linspace(beta_start, beta_end, timesteps)
        elif (sched_type == 'quadratic'):
            b_start = 0.0001
            b_end = 0.02
            betas = torch.linspace(b_start**0.5, b_end**0.5, timesteps) ** 2
        elif (sched_type == 'sigmoid'):
            b_start = 0.0001
            b_end = 0.02
            betas = torch.linspace(-6, 6, timesteps)
            betas = torch.sigmoid(betas) * (b_end - b_start) + b_start
        else:
            raise ValueError("Invalid scheduler type")
        return betas
