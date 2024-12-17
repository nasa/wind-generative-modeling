from scipy.stats import gaussian_kde
import numpy as np
import torch
from torch import nn, Tensor
import lightning as L

from torchdyn.core import NeuralODE

from wdm_src.nn import ConditionalUNet1d

def sample_conditional_pt(x0, x1, t, sigma):
    """
    Draw a sample from the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

    Parameters
    ----------
    x0 : Tensor, shape (bs, *dim)
        represents the source minibatch
    x1 : Tensor, shape (bs, *dim)
        represents the target minibatch
    t : FloatTensor, shape (bs)

    Returns
    -------
    xt : Tensor, shape (bs, *dim)

    References
    ----------
    [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
    """
    t = t.reshape(-1, *([1] * (x0.dim() - 1)))
    mu_t = t * x1 + (1.0 - t) * x0
    epsilon = torch.randn_like(x0)
    return mu_t + sigma * epsilon

def x1_inference(network, xt, t):
    t = t.reshape(-1, *([1] * (xt.dim() - 1)))
    vt = network(torch.cat([xt, t], dim = 1))
    return xt + (1.0 - t) * vt

class torch_wrapper(nn.Module):
    """Wraps model to torchdyn compatible format."""

    def __init__(self, model, c):
        super().__init__()
        self.model = model
        self.c = c
    
    def forward(self, t, x, *args, **kwargs):
        t_ = t.repeat(x.shape[0])
        return self.model(x, self.c, t_)
    
class FlowMatching(L.LightningModule):
    def __init__(
            self,
            num_cond: int, num_embeddings: int, 
            cond_drop_prob: float, lr: float):
        
        super().__init__()
        self.save_hyperparameters(logger = False)
        self.network = ConditionalUNet1d(num_cond, num_embeddings, cond_drop_prob, theta = 0.001)
        self.sigma = 0.01

    def configure_optimizers(self):
        return torch.optim.Adam(self.network.parameters(), self.hparams.lr)

    def training_step(self, batch, batch_idx: int):
        x1, c = batch
        batch_size, *_ = x1.shape

        x0 = torch.randn_like(x1)
        t = torch.rand(batch_size, device = self.device)
        xt = sample_conditional_pt(x0, x1, t, self.sigma)
        vt = self.network(xt, c, t)

        L_cfm = torch.mean(torch.square(vt - (x1 - x0)))
        self.log('L_cfm', L_cfm.item(), prog_bar = True, sync_dist = True, on_epoch = True, on_step = False)
        return L_cfm
    
    @torch.no_grad
    def sample(self, shape: tuple, c: Tensor, T: int) -> Tensor:
        x0 = torch.randn(shape, device = self.device)
        node = NeuralODE(torch_wrapper(self.network, c), solver = 'dopri5', sensitivity = 'adjoint', atol=1e-4, rtol=1e-4)
        traj = node.trajectory(x0, t_span = torch.linspace(0.0, 1.0, T))
        x1 = traj.select(0, -1)
        return x1
    
    def validation_step(self, batch, batch_idx: int):
        x1, c = batch
        u = x1[:, 0, :-1].cpu()
        v = x1[:, 1, :-1].cpu()

        ## hardcoding guidance_strength
        x_generated = self.sample(x1.shape, c, T = 10)
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