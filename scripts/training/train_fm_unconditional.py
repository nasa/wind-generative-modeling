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

    ## since no conditioning variables are provided then the model automatically defaults to unconditional
    cond = []
    data = WindConditionDataset(wind_df, cond = cond)
    train_loader = DataLoader(data, batch_size = args.batch_size, shuffle = True)
    val_loader = torch.utils.data.DataLoader(data, batch_size = len(data), shuffle = True)

    fm = FlowMatching(
        num_cond = len(cond),
        num_embeddings = 0, 
        cond_drop_prob = 0.0,
        lr = args.lr)
    
    name = f'devices-{args.devices}_bs-{args.batch_size}_lr-{args.lr}'
    logger = CSVLogger(save_dir = 'results/fm_unconditional', name = name)
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
    parser.add_argument('--epochs',         type = int,     default = 1000)
    args = parser.parse_args()
    main(args)