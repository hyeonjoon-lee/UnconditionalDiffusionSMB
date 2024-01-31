from __future__ import annotations

import os, sys

import hydra
import pytorch_lightning as pl
import torch
from neural_astar.planner import NeuralAstar
from neural_astar.utils.data import create_mario_dataloader
from neural_astar.utils.training import PlannerModule, set_global_seeds
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path

filepath = Path(__file__).parent
tb_logger = TensorBoardLogger(save_dir=os.path.join(filepath, 'logdir'), name="mario_astar")

@hydra.main(config_path="config_astar", config_name="train_mario")
def main(config):

    # dataloaders
    set_global_seeds(config.seed)
    train_loader = create_mario_dataloader(
        config.dataset, "train", config.params.batch_size, shuffle=True
    )
    val_loader = create_mario_dataloader(
        config.dataset, "val", config.params.batch_size, shuffle=False
    )

    neural_astar = NeuralAstar(
        encoder_input=config.encoder.input,
        encoder_arch=config.encoder.arch,
        encoder_depth=config.encoder.depth,
        const=config.encoder.const,
        learn_obstacles=True,
        Tmax=config.Tmax,
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="metrics/val_loss", save_weights_only=True, mode="min"
    )

    module = PlannerModule(neural_astar, config)
    logdir = f"{config.logdir}/{os.path.basename(config.dataset)}"
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        log_every_n_steps=1,
        default_root_dir=logdir,
        max_epochs=config.params.num_epochs,
        callbacks=[checkpoint_callback],
        logger=tb_logger,  # Add the TensorBoard logger here

    )
    trainer.fit(module, train_loader, val_loader)


if __name__ == "__main__":
    main()