import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist
from dataclasses import dataclass
import time

from utils.model_utils import (
    topk_gumbel_softmax,
    topk_onehot,
)


@dataclass
class ModelArgs:
    num_rules: int = 4
    slot_dim: int = 2
    temporal_input_dim: int = 8
    rule_attn_dk: int = 16
    contxt_attn_dk: int = 32
    contxt_k: int = 1
    tau: float = 1.0

    def update(self, new):
        for key, value in new.items():
            if hasattr(self, key):
                setattr(self, key, value)


class RuleNetwork(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args

        self.embedding = nn.Parameter(
            torch.randn(args.model.rule_attn_dk),
            requires_grad=True,
        )

        self.w_indiv = nn.Parameter(torch.randn(1), requires_grad=True)
        self.w_social = nn.Parameter(torch.randn(1), requires_grad=True)

        self.w_noise = nn.Parameter(torch.randn(1), requires_grad=True)

        self.spatial_1d_conv = nn.Conv1d(
            in_channels=args.model.slot_dim * (args.model.contxt_k + 1),
            out_channels=args.model.slot_dim,
            kernel_size=3,
            padding=1,
        )
        self.spatial_1d_res_conv = nn.Conv1d(
            in_channels=args.model.slot_dim * (args.model.contxt_k + 1),
            out_channels=args.model.slot_dim,
            kernel_size=1,
        )

        self.temporal_1d_conv = nn.Conv1d(
            in_channels=args.model.temporal_input_dim,
            out_channels=12,
            kernel_size=3,
            padding=1,
        )
        self.temporal_1d_res_conv = nn.Conv1d(
            in_channels=args.model.temporal_input_dim,
            out_channels=12,
            kernel_size=1,
        )

        self.spatial_1d_indiv_conv = nn.Conv1d(
            in_channels=args.model.slot_dim,
            out_channels=args.model.slot_dim,
            kernel_size=3,
            padding=1,
        )
        self.spatial_1d_indiv_res_conv = nn.Conv1d(
            in_channels=args.model.slot_dim,
            out_channels=args.model.slot_dim,
            kernel_size=1,
        )

        self.temporal_1d_indiv_conv = nn.Conv1d(
            in_channels=args.model.temporal_input_dim,
            out_channels=12,
            kernel_size=3,
            padding=1,
        )
        self.temporal_1d_indiv_res_conv = nn.Conv1d(
            in_channels=args.model.temporal_input_dim,
            out_channels=12,
            kernel_size=1,
        )

        self.act = nn.ReLU()

    def forward(self, obs_vel, contxt_vel, noise):
        # * IMLE noise addition to input velocities
        obs_vel = obs_vel + noise * self.w_noise
        obs_vel = obs_vel.permute(0, 2, 1, 3).flatten(0, 1)
        noise = noise.repeat(1, self.args.model.contxt_k, 1, 1)
        contxt_vel = contxt_vel + noise * self.w_noise
        contxt_vel = contxt_vel.permute(0, 2, 1, 3).flatten(0, 1)
        # obs_vel shape | 20xnum_peds | 2 | 8
        combined_vel = torch.cat((obs_vel, contxt_vel), dim=1)
        # combined_vel shape | 20xnum_peds | 2 * (p+c) | 8

        # * predict individual future
        # spatial 1D CNN -> 20xnum_peds | 2 | 8
        spatial_indiv_enc = self.act(self.spatial_1d_indiv_conv(obs_vel))
        spatial_indiv_enc = spatial_indiv_enc + self.spatial_1d_indiv_res_conv(obs_vel)
        # temporal 1D CNN -> 20xnum_peds | 12 | 2
        temporal_indiv_input = spatial_indiv_enc.permute(0, 2, 1)
        temporal_indiv_enc = self.temporal_1d_indiv_conv(temporal_indiv_input)
        indiv_pred = temporal_indiv_enc + self.temporal_1d_indiv_res_conv(
            temporal_indiv_input
        )

        # * predict social future
        # spatial 1D CNN -> 20xnum_peds | 2 | 8
        spatial_social_enc = self.act(self.spatial_1d_conv(combined_vel))
        spatial_social_enc = spatial_social_enc + self.spatial_1d_res_conv(combined_vel)
        # temporal 1D CNN -> 20xnum_peds | 12 | 2
        temporal_social_input = spatial_social_enc.permute(0, 2, 1)
        temporal_social_enc = self.temporal_1d_conv(temporal_social_input)
        social_pred = temporal_social_enc + self.temporal_1d_res_conv(
            temporal_social_input
        )

        return self.w_indiv * indiv_pred + self.w_social * social_pred


class ParallelNPS(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args

        self.w_rule_query = nn.Linear(
            args.model.slot_dim * args.model.temporal_input_dim,
            args.model.rule_attn_dk,
        )

        self.w_contxt_query = nn.Linear(
            args.model.slot_dim * args.model.temporal_input_dim * 2,
            args.model.contxt_attn_dk,
        )
        self.w_contxt_key = nn.Linear(
            args.model.slot_dim * args.model.temporal_input_dim * 2,
            args.model.contxt_attn_dk,
        )

        self.rules = nn.ModuleList(
            [RuleNetwork(args) for _ in range(args.model.num_rules)]
        )

        self.noise = tdist.multivariate_normal.MultivariateNormal(
            torch.zeros(2), torch.eye(2)
        )

        print(self)

    def rule_sel(self, obs_vel):
        obs_vel_compressed = obs_vel.squeeze(0).permute(1, 2, 0).flatten(-2, -1)

        rule_query = self.w_rule_query(obs_vel_compressed)
        rule_key = torch.stack([rule.embedding for rule in self.rules], dim=0)

        attn = torch.softmax(rule_query @ rule_key.permute(1, 0), dim=-1)
        return (
            topk_gumbel_softmax(
                k=1,
                logits=attn,
                tau=self.args.model.tau,
                hard=True,
                dim=-1,
            )
            if self.training
            else topk_onehot(k=1, x=attn, dim=-1)
        )

    def contxt_sel(self, obs):
        obs_compressed = obs.squeeze(0).permute(1, 2, 0).flatten(-2, -1)

        '''contxt_query_list = []
        contxt_key_list = []
        for idx_entity in range(obs_compressed.shape[0]):
            contxt_query_list.append(self.w_contxt_query[rule_mask.argmax(-1)[idx_entity]](obs_compressed[idx_entity]))
            contxt_key_list.append(self.w_contxt_key[rule_mask.argmax(-1)[idx_entity]](obs_compressed[idx_entity]))
        contxt_query = torch.stack(contxt_query_list, dim=0)
        contxt_key = torch.stack(contxt_key_list, dim=0)'''
        contxt_query = self.w_contxt_query(obs_compressed)
        contxt_key = self.w_contxt_key(obs_compressed)

        attn = torch.softmax(contxt_query @ contxt_key.permute(1, 0), dim=-1)
        return (
            topk_gumbel_softmax(
                k=self.args.model.contxt_k,
                logits=attn,
                tau=self.args.model.tau,
                hard=True,
                dim=-1,
            )
            if self.training
            else topk_onehot(k=self.args.model.contxt_k, x=attn, dim=-1)
        )

    def create_contxt_slots(self, obs, obs_vel, contxt_mask):
        contxt_list, contxt_vel_list = [], []
        for mask in contxt_mask:
            contxt = torch.bmm(
                mask.unsqueeze(0), obs.permute(0, 2, 1, 3).flatten(2, 3)
            )
            contxt_list.append(
                contxt.reshape(
                    1, obs.shape[2], 2, self.args.model.temporal_input_dim
                ).permute(0, 2, 1, 3)
            )

            contxt_vel = torch.bmm(
                mask.unsqueeze(0), obs_vel.permute(0, 2, 1, 3).flatten(2, 3)
            )
            contxt_vel_list.append(
                contxt_vel.reshape(
                    1, obs_vel.shape[2], 2, self.args.model.temporal_input_dim
                ).permute(0, 2, 1, 3)
            )

        for _ in range(self.args.model.contxt_k - len(contxt_mask)):
            contxt_list.append(torch.zeros_like(contxt_list[-1]))
            contxt_vel_list.append(torch.zeros_like(contxt_vel_list[-1]))
        return torch.cat(contxt_list, dim=1), torch.cat(contxt_vel_list, dim=1)

    def forward(self, obs, obs_vel):
        obs = obs[..., -self.args.model.temporal_input_dim :]
        obs_vel = obs_vel[..., -self.args.model.temporal_input_dim :]

        noise = self.noise.sample((20,)).unsqueeze(-1).unsqueeze(-1)

        # rule and contxt selection

        contxt_mask = self.contxt_sel(torch.cat((obs, obs_vel), dim=1))

        contxt, contxt_vel = self.create_contxt_slots(obs, obs_vel, contxt_mask)

        rule_mask = self.rule_sel(obs_vel)[0]

        # predict for each rule
        pred_list = []
        for ith_rule in range(self.args.model.num_rules):
            rule_pred = self.rules[ith_rule](
                obs_vel,
                contxt_vel,
                noise,
            )
            pred = rule_pred.reshape(20, -1, 12, 2).permute(0, 3, 1, 2)
            pred_list.append(pred)
        preds = torch.stack(pred_list, dim=0)

        # filter using rule_mask
        rule_mask = rule_mask.t().unsqueeze(1).unsqueeze(1).unsqueeze(-1)
        preds = (rule_mask * preds).sum(dim=0)

        return preds.permute(0, 2, 3, 1)
