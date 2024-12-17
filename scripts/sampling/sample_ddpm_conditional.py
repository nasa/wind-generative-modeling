from pathlib import Path

import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from wdm_src.utils import get_vector_cols

from wdm_src.data import WindConditionDataset
from wdm_src.diffusion import ConditionalDDPM

if __name__ == '__main__':
    SAMPLING_STEPS = 100

    device = torch.device(0)
    ## load in raw data
    wind_df = pd.read_csv('data/combined_macro_micro_data.csv')
    
    ## choose the conditioning columns
    ## in our work we demonstrate conditioning on one (speed only) or two (speed and direction) variables
    cond = ['macro_ws_str', 'macro_wd_str']
    data = WindConditionDataset(wind_df, cond = cond)
    data_loader = DataLoader(data, batch_size = len(data), shuffle = True)

    withhold = '(-0.001, 2.235]_WSW'
    ## load in the specific model to generate samples from
    path = Path(f'results\\ddpm_conditional\\withhold={withhold}_cond-{cond}_p-0.0_devices-1_bs-128_lr-0.0001-ts-300_sched-scaled_linear\\version_0\checkpoints\epoch=9-step=500.ckpt')
    ddpm = ConditionalDDPM.load_from_checkpoint(path)
    ddpm = ddpm.to(device)
    print(f'Network Archetecture: {type(ddpm.network)}')

    # grab a batch of data.
    x, c = next(iter(data_loader))
    x = x.to(device)
    c = c.to(device)

    ## sample from the model with a particular guidance strength
    guidance_strength = 0.0
    x_generated = ddpm.sample(x.shape, c, guidance_strength, SAMPLING_STEPS).cpu()

    ## strip off the u and v velocity components of the data and rescale them to the proper range
    x = x.cpu()
    u = x[:, 0, :-1] * (data.u_range[1] - data.u_range[0]) + data.u_range[0]
    v = x[:, 1, :-1] * (data.v_range[1] - data.v_range[0]) + data.v_range[0]
    ws = np.sqrt(u ** 2 + v ** 2)

    u_cols = get_vector_cols('u')
    v_cols = get_vector_cols('v')
    ws_cols = get_vector_cols('ws')

    ## construct a dataframe from the real samples
    real = pd.DataFrame({
        **{u_cols[i] : u[:, i] for i in range(u.shape[1])},
        **{v_cols[i] : v[:, i] for i in range(v.shape[1])},
        **{ws_cols[i] : ws[:, i] for i in range(ws.shape[1])}
    })

    ## add conditioning columns to the dataset.
    for i, key in enumerate(data.cond_dict.keys()):
        real[key] = c[:, i].cpu()
        int_to_cond = {v:k for k, v in data.cond_dict[key].items()}
        real[key] = real[key].apply(lambda x: int_to_cond[x])

    ## formating generated samples into dataframe
    u_generated = x_generated[:, 0, :-1] * (data.u_range[1] - data.u_range[0]) + data.u_range[0]
    v_generated = x_generated[:, 1, :-1] * (data.v_range[1] - data.v_range[0]) + data.v_range[0]
    ws_generated = np.sqrt(u_generated ** 2 + v_generated ** 2)

    ## construct a dataframe from the model's samples
    preds = pd.DataFrame({
        **{u_cols[i] : u_generated[:, i] for i in range(u_generated.shape[1])},
        **{v_cols[i] : v_generated[:, i] for i in range(v_generated.shape[1])},
        **{ws_cols[i] : ws_generated[:, i] for i in range(ws_generated.shape[1])},
    })

    for i, key in enumerate(data.cond_dict.keys()):
        preds[key] = c[:, i].cpu()
        int_to_cond = {v:k for k, v in data.cond_dict[key].items()}
        preds[key] = preds[key].apply(lambda x: int_to_cond[x])

    ## saving real and generated samples to file.
    real.to_csv('real.csv')
    preds.to_csv(f'{withhold}.csv')