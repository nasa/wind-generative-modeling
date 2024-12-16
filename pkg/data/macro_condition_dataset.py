from typing import List
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from wdm_src.utils import get_vector_cols
from wdm_src.diffusion_models import wind_velocity_unet1d_transform

def min_max_normalize(x, x_range):
    return (x - x_range[0]) / (x_range[1] - x_range[0])

def min_max_unnormalize(x, x_range):
    return x * (x_range[1] - x_range[0]) + x_range[0]

class MacroConditionDataset(Dataset):
    '''
    Dataset which packages together the corresponding macroweather conditions with a wind profile.
    The user can specify a set of conditions (cond) of interest.
    '''
    def __init__(self, wind_df: pd.DataFrame, cond: List[str]):
        super().__init__()
        assert set(cond) <= set(wind_df.columns)
        wind_df = wind_df.copy(deep = True)
        wind_df = wind_df[[*get_vector_cols('u'), *get_vector_cols('v'), *get_vector_cols('w'), *cond]]
        wind_df = wind_df.dropna()

        ## get u and v component of velocity and normalize
        u = wind_df[get_vector_cols('u')].values
        v = wind_df[get_vector_cols('v')].values

        self.u_range = (-30.0, 30.0)
        self.v_range = (-35.0, 30.0)
        self.u = min_max_normalize(u, self.u_range)
        self.v = min_max_normalize(v, self.v_range)

        ## construct the conditioning dictionary
        self.cond_dict = {}
        i = 0
        for c in cond:
            self.cond_dict[c] = {}
            for token in wind_df[c].unique():
                self.cond_dict[c][token] = i
                i += 1

        ## transform the conditioning columns
        for c in cond:
            wind_df[c] = wind_df[c].apply(lambda x: self.cond_dict[c][x])

        self.cond = wind_df[cond].values

    @property
    def num_cond(self):
        return len(self.cond_dict)

    @property
    def num_embeddings(self):
        return sum(len(c) for c in self.cond_dict.values())

    def __len__(self) -> int:
        return len(self.cond)
    
    def __getitem__(self, index: int) -> tuple:
        u = torch.Tensor(self.u[index, :])
        v = torch.Tensor(self.v[index, :])
        x = wind_velocity_unet1d_transform(torch.cat([u, v], dim = 0))
        c = torch.tensor(self.cond[index, :], dtype = torch.long)
        return x, c

if __name__ == '__main__':
    from torch import nn
    from torch.utils.data import DataLoader

    from wdm_src.diffusion_models import ConditionalUNet1d

    data_path = Path(__file__).parent / '../../data/areaA_april2022/areaA_april2022_combined_macro_micro_data.csv'
    wind_df = pd.read_csv(data_path)
    cond = ['macro_conditions', 'macro_wd_str']
    data = MacroConditionDataset(wind_df, cond)
    train_loader = DataLoader(data, batch_size = 32, shuffle = True)
    
    x, c = next(iter(train_loader))
    assert x.shape == (32, 2, 48)
    assert c.shape == (32, len(cond))

    model = ConditionalUNet1d(data.num_cond, data.num_embeddings)

    t = torch.randint(0, 300, (32,))

    print(x.shape)
    print(c.shape)
    print(model(x, c, t).shape)