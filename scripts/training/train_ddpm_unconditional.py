from argparse import ArgumentParser

import pandas as pd
import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from wdm_src.data import WindDataset, wind_velocity_unet1d_transform
from wdm_src.diffusion import UnconditionalDDPM
from wdm_src.nn import UNet1d, FullyConnectedNet

torch.set_float32_matmul_precision('medium')

def main(args):
    wind_df = pd.read_csv('data/combined_macro_micro_data.csv')
    ## for the unconditional case we test two different architectures
    if args.network == 'unet':
        transform = wind_velocity_unet1d_transform
        network = UNet1d()
    elif args.network == 'fcnet':
        transform = None
        network = FullyConnectedNet(95, [1000] * 2, 94, 'RELU')

    ddpm = UnconditionalDDPM(network, args.timesteps, args.schedule, args.lr)
    data = WindDataset(wind_df, transform = transform)

    ## validation only for the purposes of logging. 
    ## our true validation metric is to hold out a particular combination of macroweather conditions and measure error against that.
    ## see the conditional ddpm and fm for more robust evaluations
    train_loader = DataLoader(data, batch_size = args.batch_size, shuffle = True)
    val_loader = DataLoader(data, batch_size = 1024, shuffle = True)
    name = f'net-{args.network}_bs-{args.batch_size}_lr-{args.lr}-ts-{args.timesteps}_sched-{args.schedule}'

    logger = CSVLogger(save_dir = 'results/ddpm_unconditional', name = name)
    checkpoint = ModelCheckpoint(monitor = 'kl_divergence', mode = 'min', save_top_k = 2)

    trainer = L.Trainer(
        accelerator = 'gpu', devices = args.devices, 
        max_epochs = args.epochs,
        logger = logger,
        callbacks = checkpoint,
        limit_val_batches = 1,
        check_val_every_n_epoch = 5)

    trainer.fit(ddpm, train_loader, val_loader)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type = int, default = 128)
    parser.add_argument('--timesteps',  type = int, default = 300)
    parser.add_argument('--schedule',   type = str, default = 'scaled_linear', choices = ['linear', 'scaled_linear', 'cosine'])
    parser.add_argument('--lr',         type = int, default = 1e-4)
    parser.add_argument('--devices',    type = int, default = 1)
    parser.add_argument('--epochs',     type = int, default = 1000)
    parser.add_argument('--network',    type = str, default = 'unet', choices = ['unet', 'fcnet'])
    args = parser.parse_args()
    main(args)