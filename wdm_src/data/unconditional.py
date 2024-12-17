from torch.utils.data import  Dataset
import numpy as np
import pandas as pd

def wind_velocity_unet1d_transform(x: np.array) -> np.array:
    x = np.stack([x[:47], x[47:]], axis = 0) ## u and v are one vector by default, we split them into channels
    x = np.pad(x, ((0, 0), (0, 1))) ## default is 47 altitudes, we need an even number
    return x

MAX_VEL = 30.0
class WindDataset(Dataset):
    def __init__(self, wind_df, min_wind_speed=0, altitudes=None, qoi='velocity', transform=None, normalize: str = 'minmax'):
        assert normalize in ['minmax', 'standardize', None]
        '''
        qoi={'velocity', 'speed', 'direction'} velocity returns u & v components
        '''
        column_names = self._get_column_names(qoi, altitudes)
        wind_df = wind_df[wind_df['macro_ws'] >= min_wind_speed]
        wind_df = wind_df[column_names].astype('float32')
        self._wind_df = wind_df.dropna()
        self.mean = self._wind_df.mean(0)
        self.std = self._wind_df.std(0)

        if normalize == 'standardize':
            self._wind_df = self._wind_df.sub(self.mean, axis=1).div(self.std, axis=1)
        elif (normalize == 'minmax') and qoi == 'velocity':
            self._wind_df = self._wind_df / MAX_VEL
        elif (normalize == 'minmax') and qoi != 'velocity':
            raise NotImplementedError(f'minmax normalization not implemented for qoi: {qoi}')

        self.transform = transform

    def __len__(self):
        return self._wind_df.shape[0]

    def __getitem__(self, idx):
        sample = self._wind_df.iloc[idx].values
        if self.transform:
            sample = self.transform(sample)
        return sample

    def _get_column_names(self, qoi, altitudes):
        if qoi.upper() == 'VELOCITY':
            keys = ['u', 'v']
        elif qoi.upper() == 'SPEED':
            keys = ['ws']
        elif qoi.upper() == 'DIRECTION':
            keys = ['wd']
        if altitudes is None:
            altitudes = np.arange(20, 251, 5)
        column_names = ['{}{}'.format(key, alt)
                        for key in keys for alt in altitudes]
        return column_names
