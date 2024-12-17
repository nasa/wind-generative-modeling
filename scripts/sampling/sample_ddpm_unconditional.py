from pathlib import Path

import torch
import pandas as pd
import numpy as np

from wdm_src.utils import get_vector_cols
from wdm_src.data import WindDataset, MAX_VEL, wind_velocity_unet1d_transform
from wdm_src.diffusion import UnconditionalDDPM
from wdm_src.nn import UNet1d, FullyConnectedNet

if __name__ == '__main__':
    device = torch.device(0)
    
    wind_df = pd.read_csv('data/combined_macro_micro_data.csv')
    transform = wind_velocity_unet1d_transform
    data = WindDataset(wind_df, transform = transform)

    ## load in the specific model to generate samples from
    path = Path('results\\ddpm_unconditional\\net-unet_bs-128_lr-0.0001-ts-300_sched-scaled_linear\\version_0\checkpoints\epoch=469-step=24440.ckpt')
    model = UnconditionalDDPM.load_from_checkpoint(path)
    model = model.to(device)

    print(f'Network Archetecture: {type(model.network)}')
    if isinstance(model.network, UNet1d):
        transform = wind_velocity_unet1d_transform
    elif isinstance(model.network, FullyConnectedNet):
        transform = None

    train_loader = torch.utils.data.DataLoader(data, batch_size = len(data), shuffle = True)
    x0 = next(iter(train_loader))
    x0 = x0.to(device)

    ## random noise to convert into samples. we just need the shape to generate the same number of samples.
    xT = torch.randn_like(x0)

    ## generate samples from ddpm.
    x0_generated = model.sampler.p_sample_loop(model.network, xT)

    ## based on the model class there is different logic for unpacking u and v velocity fields.
    ## normalize output back to proper range by multiplying by MAX_VEL.
    if isinstance(model.network, UNet1d):
        u = x0[:, 0, :-1].cpu() * MAX_VEL
        v = x0[:, 1, :-1].cpu() * MAX_VEL
        u_generated = x0_generated[:, 0, :-1].cpu() * MAX_VEL
        v_generated = x0_generated[:, 1, :-1].cpu() * MAX_VEL
        model_type = 'UNet1d'
    elif isinstance(model.network, FullyConnectedNet):
        u = x0[:, :47].cpu() * MAX_VEL
        v = x0[:, 47:].cpu() * MAX_VEL
        u_generated = x0_generated[:, :47].cpu() * MAX_VEL
        v_generated = x0_generated[:, 47:].cpu() * MAX_VEL
        model_type = 'FullyConnectedNet'

    ## convert velocity comonents into speed.
    ws_generated = np.sqrt(u_generated ** 2 + v_generated ** 2)

    ## save predictions to a csv
    u_cols = get_vector_cols('u')
    v_cols = get_vector_cols('v')
    ws_cols = get_vector_cols('ws')

    ## construct a dataframe of samples
    preds = pd.DataFrame({
        **{u_cols[i] : u_generated[:, i] for i in range(u_generated.shape[1])},
        **{v_cols[i] : v_generated[:, i] for i in range(v_generated.shape[1])},
        **{ws_cols[i] : ws_generated[:, i] for i in range(ws_generated.shape[1])},
        })
    
    ## save samples to file
    preds.to_csv('scripts/postprocessing/unconditional/ddpm_unconditional.csv')