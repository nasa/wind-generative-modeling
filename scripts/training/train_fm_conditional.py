from argparse import ArgumentParser

import pandas as pd
import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from wdm_src.data import WindConditionDataset
from wdm_src.flow_matching import FlowMatching

torch.set_float32_matmul_precision('medium')
    
def main(args):
    wind_df = pd.read_csv('data/combined_macro_micro_data.csv')
    ## choose the conditioning columns.
    cond = ['macro_ws_str', 'macro_wd_str']
    speeds = wind_df['macro_ws_str'].unique()
    directions = ['W', 'WNW', 'WSW', 'SW']

    ## choose the combination to withhold
    speed = speeds[3]
    direction = directions[2]
    withhold = speed + '_' + direction

    ## drop the withhold combination from the dataset
    withhold_wind_df = wind_df[((wind_df['macro_ws_str'] != speed) | (wind_df['macro_wd_str'] != direction))]
    data = WindConditionDataset(withhold_wind_df, cond = cond)

    ## validation only for the purposes of logging.
    ## our true validation metric is to hold out a particular combination of macroweather conditions and measure error against that.
    train_loader = DataLoader(data, batch_size = args.batch_size, shuffle = True)
    val_loader = DataLoader(data, batch_size = len(data), shuffle = True)
    fm = FlowMatching(
        data.num_cond, 
        data.num_embeddings, 
        cond_drop_prob = args.cond_drop_prob, 
        lr = args.lr)
    
    name = f'withhold={withhold}_cond-{cond}_p-{args.cond_drop_prob}_devices-{args.devices}_bs-{args.batch_size}_lr-{args.lr}'
    logger = CSVLogger(save_dir = 'results/fm_conditional', name = name)
    checkpoint = ModelCheckpoint(monitor = 'kl_divergence', mode = 'min', save_top_k = 2)

    trainer = L.Trainer(
        accelerator = 'gpu',
        devices = args.devices,
        max_epochs = args.epochs,
        logger = logger,
        callbacks = checkpoint,
        limit_val_batches = 1,
        check_val_every_n_epoch = 5)
    
    trainer.fit(fm, train_loader, val_loader)
    return None

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--devices',        type = int,     default = 1)
    parser.add_argument('--batch_size',     type = int,     default = 128)
    parser.add_argument('--lr',             type = float,   default = 1e-4)
    parser.add_argument('--cond_drop_prob', type = float,   default = 0.0)
    parser.add_argument('--epochs',         type = int,     default = 1000)
    args = parser.parse_args()
    main(args)