from pathlib import Path

import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from wdm_src.utils import get_vector_cols

from wdm_src.data import WindConditionDataset
from wdm_src.flow_matching import FlowMatching

if __name__ == '__main__':
    SAMPLING_STEPS = 20

    ## load in raw data
    wind_df = pd.read_csv('data/combined_macro_micro_data.csv')
    
    ## choose the conditioning columns
    ## in our work we demonstrate conditioning on one (speed only) or two (speed and direction) variables
    cond = ['macro_ws_str', 'macro_wd_str']
    data = WindConditionDataset(wind_df, cond = cond)
    data_loader = DataLoader(data, batch_size = len(data), shuffle = True)

    ## detect the current device and load in the trained model
    device = torch.device(0)

    withhold = '(-0.001, 2.235]_WSW'
    path = Path(f'results\\fm_conditional\\withhold={withhold}_cond-{cond}_p-0.0_devices-1_bs-128_lr-0.0001\\version_0\\checkpoints\\epoch=9-step=500.ckpt')

    fm = FlowMatching.load_from_checkpoint(path)
    fm = fm.to(device)

    x1, c = next(iter(data_loader))
    x1, c = x1.to(device), c.to(device)

    ## generate samples
    x1_hat = fm.sample(x1.shape, c, T = SAMPLING_STEPS)

    x_generated = x1_hat.cpu()

    ## seperating u and v components and un-normalizing them
    u_generated = x_generated[:, 0, :-1] * (data.u_range[1] - data.u_range[0]) + data.u_range[0]
    v_generated = x_generated[:, 1, :-1] * (data.v_range[1] - data.v_range[0]) + data.v_range[0]
    ws_generated = np.sqrt(u_generated ** 2 + v_generated ** 2)

    u_cols = get_vector_cols('u')
    v_cols = get_vector_cols('v')
    ws_cols = get_vector_cols('ws')

    ## construct a dataframe from the model's samples
    preds = pd.DataFrame({
        **{u_cols[i] : u_generated[:, i] for i in range(u_generated.shape[1])},
        **{v_cols[i] : v_generated[:, i] for i in range(v_generated.shape[1])},
        **{ws_cols[i] : ws_generated[:, i] for i in range(ws_generated.shape[1])},
        })
    
    for i, key in enumerate(data.cond_dict.keys()):
        assert set(c[:, i].unique().cpu().tolist()) <= set(data.cond_dict[key].values())
        
        preds[key] = c[:, i].cpu()
        int_to_cond = {v:k for k, v in data.cond_dict[key].items()}
        preds[key] = preds[key].apply(lambda _x: int_to_cond[_x])

    preds.to_csv(f'{withhold}.csv')