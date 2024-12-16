import torch
from tqdm.auto import tqdm
import torch.nn.functional as F

from wdm_src.diffusion_models.helper import extract

class DiffusionModel():
    def __init__(self, nn_model, scheduler):
        self._nn_model = nn_model
        self._scheduler = scheduler

    @property
    def model(self):
        return self._nn_model

    @torch.no_grad()
    def p_sample(self, x, t, t_index):

        betas_t = extract(self._scheduler.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self._scheduler.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(
            self._scheduler.sqrt_recip_alphas, t, x.shape)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self._nn_model(x, t) /
            sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(
                self._scheduler.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise
            # return model_mean + torch.sqrt(betas_t) * noise

    # Algorithm 2 (including returning all images)

    @torch.no_grad()
    def p_sample_loop(self, shape):
        device = next(self._nn_model.parameters()).device

        b = shape[0]
        # start from pure noise (for each example in the batch)
        sample = torch.randn(shape, device=device)
        samples = []
        samples.append(sample.cpu().numpy())

        for i in tqdm(reversed(range(0, self._scheduler.timesteps)), desc='sampling loop time step', total=self._scheduler.timesteps):
            sample = self.p_sample(sample, torch.full(
                (b,), i, device=device, dtype=torch.long), i)
            samples.append(sample.cpu().numpy())
        return samples

    @torch.no_grad()
    def sample(self, shape, batch_size=16):
        shape = tuple([batch_size] + list(shape))
        return self.p_sample_loop(shape = shape)

    # forward diffusion (using the nice property
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(
            self._scheduler.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self._scheduler. sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def loss(self, x_start, t, noise=None, loss_type='l1'):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self._nn_model(x_noisy, t)

        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == 'huber':
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss
