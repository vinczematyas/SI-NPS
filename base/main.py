import numpy as np
import torch
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from dataclasses import dataclass, asdict
from pprint import pformat
from termcolor import colored
import argparse
import datetime
import logging
import wandb
import math
import os

from src.model import ModelArgs, ParallelNPS
from src.train_test import train_epoch, test_epoch
from utils.data_utils import TrajectoryAugmenter, load_traj_data
from utils.other_utils import (
    custom_argparser,
    fix_seed,
)


@dataclass
class Args:
    log: bool = False
    wandb: bool = False
    tqdm: bool = True
    dataset: str = "zara1"
    save_model: bool = False
    seed: int = 1
    num_epochs: int = 50
    batch_size: int = 128
    lr: float = 1e-2
    step_size: int = 45
    decay_min_lr: float = 5e-4
    data_norm: int = 0
    model = ModelArgs()

    def update(self, new):
        for key, value in new.items():
            if hasattr(self, key):
                setattr(self, key, value)


args = Args()
parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
# run args
parser.add_argument("--log", type=bool)
parser.add_argument("--wandb", type=bool)
parser.add_argument("--dataset", type=str)
parser.add_argument("--save_model", type=bool)
parser.add_argument("--seed", type=int)
parser.add_argument("--num_epochs", type=int)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--lr", type=float)
parser.add_argument("--data_norm", type=int)

new_args = custom_argparser()  # deal with notebook args
args.update(new_args)
args.model.update(new_args)

assert not (args.save_model and not args.log), "set --log=True when saving the model"

fix_seed(args.seed)

if args.wandb:
    wandb.init(project="NPS_ethucy", config={**asdict(args), **asdict(args.model)})

if args.log:
    log_dir = f'base/model/{args.dataset}_{datetime.datetime.now().strftime("%m-%d_%H-%M-%S")}'
    os.mkdir(log_dir)
    logging.basicConfig(
        filename=os.path.join(log_dir, "log.txt"),
        format="[%(asctime)s]   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logging.info("Configuration:")
    logging.info("general args")
    logging.info(pformat(asdict(args)))
    logging.info("model args")
    logging.info(pformat(asdict(args.model)))
print(pformat(asdict(args)))
print(pformat(asdict(args.model)))

tr_dl, val_dl, test_dl = load_traj_data(args.dataset)
augmenter = TrajectoryAugmenter(data_loader=tr_dl)

model = ParallelNPS(args).double()
optim = SGD(model.parameters(), lr=args.lr)
'''scheduler = LinearWarmupCosineAnnealingLR(
    optim,
    warmup_epochs=args.num_epochs // 10,
    max_epochs=args.num_epochs,
    warmup_start_lr=0.0,
    eta_min=args.lr / 10,
)'''
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=args.step_size, gamma=0.1)

num_params = 0
for name, param in model.named_parameters():
    num_params += param.numel()
    print(f"Layer: {name} | Size: {param.size()} | Parameters: {param.numel()}")
print(f"Total number of parameters: {num_params}")

min_ade = np.inf
for idx_epoch in range(args.num_epochs):
    args.model.tau = 1.0 * math.exp(
        -(-math.log(0.5 / 1.0) / args.num_epochs) * idx_epoch
    )
    print(f"tau = {math.log(0.5 / 1.0)}")
    print(f"tau = {-math.log(0.5 / 1.0) / args.num_epochs}")
    print(f"tau = {round(args.model.tau, 6)}")

    train_epoch(args, model, idx_epoch, tr_dl, optim)
    val_ade, val_fde = test_epoch(args, model, "val", val_dl)
    test_ade, test_fde = test_epoch(args, model, "test", test_dl)
    scheduler.step()
    print(f"lr = {round(scheduler.get_last_lr()[0], 6)}")

    if val_ade < min_ade:
        min_ade = val_ade

        if args.log:
            logging.info(
                f"saving new best model at epoch {idx_epoch+1} as new best val_ade reached - {round(val_ade, 4)}\n\t \
                val ADE {round(val_ade, 4)}\n\t \
                val FDE {round(val_fde, 4)}\n\t \
                test ADE {round(test_ade, 4)}\n\t \
                test FDE {round(test_fde, 4)}\n",
            )
        if args.save_model:
            print(colored(f"  saving new best model - test_ade: {test_ade}", "green"))
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optim.state_dict(),
                    "val ADE": {round(val_ade, 4)},
                    "val FDE": {round(val_fde, 4)},
                    "test ADE": {round(test_ade, 4)},
                    "test FDE": {round(test_fde, 4)},
                },
                f"{log_dir}/{args.dataset}.pth",
            )
        else:
            print(
                colored(
                    f"  new best model - test_ade: {test_ade} | test_fde: {test_fde}",
                    "green",
                )
            )
        if args.wandb:
            wandb.log(
                {
                    "idx_epoch": idx_epoch,
                    "valADE": val_ade,
                    "valFDE": val_fde,
                    "testADE": test_ade,
                    "testFDE": test_fde,
                }
            )
