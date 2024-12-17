import torch
from torch import nn, Tensor
import torch.nn.functional as F
import lightning as L
from scipy.stats import gaussian_kde
import numpy as np
from tqdm import tqdm

from wdm_src.diffusion import DiffusionSampler, ConditionalDiffusionSampler
from wdm_src.nn import UNet1d, ConditionalUNet1d, FullyConnectedNet

class UnconditionalDDPM(L.LightningModule):
    def __init__(self, network: nn.Module, timesteps: int, beta_schedule: str, lr: float):
        super().__init__()
        self.save_hyperparameters(logger = False)
        self.network = network
        self.sampler = DiffusionSampler(timesteps, beta_schedule)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.network.parameters(), self.hparams.lr)
    
    def training_step(self, batch, batch_idx: int):
        x0 = batch
        batch_size, *_ = x0.shape

        t = torch.randint(0, self.sampler.timesteps, (batch_size,), dtype = torch.long, device = self.device)
        xt, noise = self.sampler.q_sample(x0, t)
        epsilon = self.network(xt, t)
        loss = F.mse_loss(epsilon, noise)
        self.log('train_loss', loss.item(), prog_bar = True, sync_dist = True, on_epoch = True, on_step = False)
        return loss
    
    def validation_step(self, batch, batch_idx: int):
        x0 = batch
        batch_size, *_ = x0.shape

        xT = torch.randn(x0.shape, device = self.device)
        x0_generated = self.sampler.p_sample_loop(self.network, xT, prog = True)

        ## extracting u and v velocity components from real and generated samples
        if isinstance(self.network, UNet1d):
            u = x0[:, 0, :-1].cpu()
            v = x0[:, 1, :-1].cpu()
            u_generated = x0_generated[:, 0, :-1].cpu()
            v_generated = x0_generated[:, 1, :-1].cpu()
        elif isinstance(self.network, FullyConnectedNet):
            u = x0[:, :47].cpu()
            v = x0[:, 47:].cpu()
            u_generated = x0_generated[:, :47].cpu()
            v_generated = x0_generated[:, 47:].cpu()

        ## compute a mean over altitudes
        u = u.mean(1)
        v = v.mean(1)
        u_generated = u_generated.mean(1)
        v_generated = v_generated.mean(1)

        ## estimate probability distributions with gaussian kernel
        data_kde = gaussian_kde(torch.stack([u, v]))
        generated_kde = gaussian_kde(torch.stack([u_generated, v_generated]))

        ## sample points from generated distribution
        points = generated_kde.resample(10000)
        p = generated_kde.pdf(points)
        q = data_kde.pdf(points)
        kl_divergence = np.log(p / q).mean()
        self.log('kl_divergence', kl_divergence, prog_bar = True, sync_dist = True, on_epoch = True, on_step = False)
        return kl_divergence

class ConditionalDDPM(L.LightningModule):
    '''
    Partially from:
    https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/classifier_free_guidance.py
    '''
    def __init__(self, 
                num_cond: int, num_embeddings: int, 
                cond_drop_prob: float,
                timesteps: int, beta_schedule: str,
                lr: float):
        
        super().__init__()
        self.save_hyperparameters(logger = False)
        self.network = ConditionalUNet1d(num_cond, num_embeddings, cond_drop_prob)
        self.sampler = ConditionalDiffusionSampler(timesteps, beta_schedule)

    def configure_optimizers(self):
        return torch.optim.Adam(self.network.parameters(), self.hparams.lr)
    
    def training_step(self, batch, batch_idx: int):
        x0, c = batch
        batch_size, *_ = x0.shape

        t = torch.randint(0, self.sampler.timesteps, (batch_size,), dtype = torch.long, device = self.device)
        xt, noise = self.sampler.q_sample(x0, t)
        epsilon = self.network(xt, c, t)
        loss = F.mse_loss(epsilon, noise)
        self.log('train_loss', loss.item(), prog_bar = True, sync_dist = True, on_epoch = True, on_step = False)
        return loss

    ## vanilla sampling    
    # def sample(self, shape: tuple, c: Tensor, guidance_strength: float) -> Tensor:
    #     xT = torch.randn(shape, device = self.device)
    #     x0 = self.sampler.p_sample_loop(self.network, xT, c, guidance_strength)
    #     return x0

    @torch.no_grad()
    def sample(self, shape: tuple, c: Tensor, guidance_strength: float, sampling_timesteps: int = 4, eta: float = 0.0) -> Tensor:
        '''
        DDIM sampling algorithm with a reduced number of sampling steps.
        '''
        times = torch.linspace(-1, self.sampler.timesteps - 1, sampling_timesteps, dtype = torch.long)
        times = list(reversed(times.tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        xt = torch.randn(shape, device = self.device)

        for time, time_next in tqdm(time_pairs):
            alpha_bar = self.sampler.alpha_bar[time]
            alpha_bar_next = self.sampler.alpha_bar[time_next]
            t = time * torch.ones(shape[0], dtype = torch.long, device = self.device)

            ## equation 6 in classsifier free guidance paper
            if guidance_strength == 0.0:
                epsilon = self.network(xt, c, t, cond_drop_prob = 0.0)
            else:
                epsilon = (1.0 + guidance_strength) * self.network(xt, c, t, cond_drop_prob = 0.0) - guidance_strength * self.network(xt, c, t, cond_drop_prob = 1.0)

            a, a_next = alpha_bar.sqrt(), alpha_bar_next.sqrt()
            b = (1.0 - alpha_bar).sqrt()
            x0_hat = (xt - b * epsilon) / a

            if time_next < 0:
                xt = x0_hat
                continue

            sigma = eta * ((1 - alpha_bar / alpha_bar_next) * (1 - alpha_bar_next) / (1 - alpha_bar)).sqrt()
            direction_pointing_to_xt = (1.0 - alpha_bar_next - sigma ** 2).sqrt()
            z = torch.randn_like(xt)
            xt = a_next * x0_hat + direction_pointing_to_xt * epsilon + sigma * z
        return xt

    def validation_step(self, batch, batch_idx: int):
        x, c = batch
        u = x[:, 0, :-1].cpu()
        v = x[:, 1, :-1].cpu()

        ## hardcoding guidance_strength
        x_generated = self.sample(x.shape, c, guidance_strength = 0.0, sampling_timesteps = 20)
        u_generated = x_generated[:, 0, :-1].cpu()
        v_generated = x_generated[:, 1, :-1].cpu()

        ## compute gaussian kernel density estimates for both real and generated bivariate distributions p(u, v)
        data_kde = gaussian_kde(torch.stack([u.mean(1), v.mean(1)]))
        generated_kde = gaussian_kde(torch.stack([u_generated.mean(1), v_generated.mean(1)]))

        ## sample points from generated distribution (arbitrarily large number of points sampled)
        points = generated_kde.resample(10000)
        p = generated_kde.pdf(points)
        q = data_kde.pdf(points)
        kl_divergence = np.log(p / q).mean()
        self.log('kl_divergence', kl_divergence, prog_bar = True, sync_dist = True, on_epoch = True, on_step = False)
        return kl_divergence