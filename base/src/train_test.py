import numpy as np
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm
import wandb

from utils.data_utils import TrajectoryAugmenter, calc_ade_fde
from utils.model_utils import plot_grad_flow, graph_loss

metric = torch.nn.MSELoss()


def train_epoch(args, model, idx_epoch, dl, optim):
    model.train()
    optim.zero_grad()

    augmenter = TrajectoryAugmenter(data_loader=dl)

    batch_loss = 0
    loss_list = []
    if args.tqdm:
        pbar = tqdm(dl, desc=f"epoch {idx_epoch}/{args.num_epochs}")
    else:
        pbar = dl
        print(f"epoch: {idx_epoch}")
    for idx_batch, batch in enumerate(pbar):
        obs_traj, pred_traj_gt, V_obs, V_tr = batch[0], batch[1], batch[-4], batch[-2]

        V_obs, V_tr, obs_traj, pred_traj_gt = augmenter.augment(
            V_obs, V_tr, obs_traj, pred_traj_gt
        )

        obs = obs_traj.permute(0, 2, 1, 3)
        obs_norm = (obs - obs.mean()) / obs.std()  # normalize
        obs_vel = V_obs.permute(0, 3, 2, 1)

        pred = model(obs_norm, obs_vel)
        batch_loss += graph_loss(pred.permute(0, 2, 1, 3), V_tr)

        if (idx_batch + 1) % args.batch_size == 0 or idx_batch == (len(dl) - 1):
            batch_loss /= args.batch_size
            batch_loss.backward()
            clip_grad_norm_(model.parameters(), 4.0)

            """plot_grad_flow(model.named_parameters())
            if idx_batch == (len(tr_dl) - 1):
                plot_grad_flow(model.named_parameters(), show=True)"""

            optim.step()
            optim.zero_grad()

            loss_list.append(batch_loss.item())

            if args.tqdm:
                pbar.set_postfix(
                    {
                        "loss": batch_loss.item(),
                        "LOSS": np.mean(loss_list),
                    }
                )
            if args.wandb:
                wandb.log({"train_loss": batch_loss.item()})
            batch_loss = 0


def test_epoch(args, model, val_or_test, dl):
    with torch.inference_mode():
        model.eval()
        ade_list, fde_list = [], []
        if args.tqdm:
            pbar = tqdm(dl, desc=f"{val_or_test}")
        else:
            pbar = dl
        for _, batch in enumerate(pbar):
            obs_traj, pred_traj_gt, V_obs = (
                batch[0],
                batch[1],
                batch[-4],
            )

            obs = obs_traj.permute(0, 2, 1, 3)
            obs_norm = (obs - obs.mean()) / obs.std()  # normalize
            obs_vel = V_obs.permute(0, 3, 2, 1)

            pred = model(obs_norm, obs_vel)
            pred_coords = torch.stack(
                [
                    torch.stack(
                        [
                            pred[bestofn_idx, :, :idx_time, :].sum(1)
                            + obs.squeeze(0).permute(1, 2, 0)[:, -1]
                            for idx_time in range(1, 13)
                        ],
                        dim=1,
                    )
                    for bestofn_idx in range(20)
                ],
                dim=0,
            )

            ade, fde = calc_ade_fde(
                pred_coords, pred_traj_gt.squeeze(0).permute(0, 2, 1)
            )
            ade_list.append(ade)
            fde_list.append(fde)

            if args.tqdm:
                pbar.set_postfix(
                    {
                        "ADE": round(np.mean(ade_list), 3),
                        "FDE": round(np.mean(fde_list), 3),
                    }
                )
        if not args.tqdm:
            print(
                f"{val_or_test} -> ADE: {round(np.mean(ade_list), 3)} | FDE: {round(np.mean(fde_list), 3)}"
            )
        return np.mean(ade_list), np.mean(fde_list)
