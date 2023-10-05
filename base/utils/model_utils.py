import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt


class GroupConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        num_groups,
        stride=1,
        padding=0,
        dilation=1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_groups = num_groups
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        lim = 1.0 / math.sqrt(out_channels)
        self.weight = nn.Parameter(
            torch.Tensor(
                num_groups, out_channels, in_channels, kernel_size, kernel_size
            ).uniform_(-lim, lim),
            requires_grad=True,
        )
        self.bias = nn.Parameter(
            torch.Tensor(num_groups, out_channels).uniform_(-lim, lim),
            requires_grad=True,
        )

    def forward(self, input):
        output = [
            nn.functional.conv2d(
                input[i],
                self.weight[i],
                bias=self.bias[i],
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            )
            for i in range(self.num_groups)
        ]
        return torch.stack(output)


class GroupConv1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        num_groups,
        stride=1,
        padding=0,
        dilation=1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_groups = num_groups
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        lim = 1.0 / math.sqrt(out_channels)
        self.weight = nn.Parameter(
            torch.Tensor(num_groups, out_channels, in_channels, kernel_size).uniform_(
                -lim, lim
            ),
            requires_grad=True,
        )
        self.bias = nn.Parameter(
            torch.Tensor(num_groups, out_channels).uniform_(-lim, lim),
            requires_grad=True,
        )

    def forward(self, input):
        output = [
            nn.functional.conv1d(
                input[i],
                self.weight[i],
                bias=self.bias[i],
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            )
            for i in range(self.num_groups)
        ]
        return torch.stack(output)


def topk_gumbel_softmax(k, logits, tau=1, hard=False, eps=1e-10, dim=-1):
    def _gen_gumbels():
        gumbels = -torch.empty_like(logits).exponential_().log()
        if torch.isnan(gumbels).sum() or torch.isinf(gumbels).sum():
            # to avoid zero in exp output
            gumbels = _gen_gumbels()
        return gumbels

    gumbels = _gen_gumbels()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.\
        index = y_soft.topk(min(y_soft.shape[-1], k), dim)[1]
        ret_list = []
        for contxt_idx in range(index.shape[1]):
            y_hard = torch.zeros_like(logits).scatter_(
                dim, index[:, contxt_idx].unsqueeze(1), 1.0
            )
            ret = y_hard - y_soft.detach() + y_soft
            ret_list.append(ret)
    else:
        # Reparametrization trick.
        ret_list = y_soft
    return ret_list


def topk_onehot(k, x: torch.Tensor, dim: int):
    """idx = x.argmax(dim=dim)
    return torch.zeros_like(x).scatter_(dim, idx.unsqueeze(dim), 1.0)"""
    idx = x.topk(min(x.shape[-1], k), dim)[1]
    ret_list = []
    for contxt_idx in range(idx.shape[1]):
        ret_list.append(
            torch.zeros_like(x).scatter_(dim, idx[:, contxt_idx].unsqueeze(1), 1.0)
        )
    return ret_list


# Set the default figure size
plt.rcParams["figure.figsize"] = (20, 16)


def plot_grad_flow(named_parameters, show=False):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n) and p.grad is not None:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())

    plt.plot(
        ave_grads, "r-", alpha=0.7, linewidth=2
    )  # Red color and increased line width
    plt.hlines(
        0, 0, len(ave_grads) + 1, linewidth=1, color="k", linestyle="--"
    )  # Dashed line for y=0
    plt.xticks(
        range(0, len(ave_grads), 1), layers, rotation="vertical", fontsize=20
    )  # Increased font size for x-ticks
    plt.yticks(fontsize=20)
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.title("Gradient flow", fontsize=24)  # Increased font size for title
    plt.grid(True, linestyle="--", alpha=0.6)  # Make grid less prominent
    plt.tight_layout()
    if show == True:
        plt.show()


_l1_mean = torch.nn.L1Loss()


def cdist_cosine_sim(a, b, eps=1e-08):
    a_norm = a / torch.clamp(a.norm(dim=1)[:, None], min=eps)
    b_norm = b / torch.clamp(b.norm(dim=1)[:, None], min=eps)
    return torch.acos(
        torch.clamp(
            torch.mm(a_norm, b_norm.transpose(0, 1)), min=-1.0 + eps, max=1.0 - eps
        )
    )


def implicit_likelihood_estimation_fast_with_trip_geo(V_pred, V_target):
    V_pred = V_pred.contiguous()

    diff = torch.abs(V_pred - V_target)

    diff_sum = torch.sum(diff, dim=(1, 2, 3))
    _, indices = torch.sort(diff_sum)
    min_indx = indices[0]
    V_pred_min = V_pred[min_indx]
    V_target = V_target.squeeze()

    error = _l1_mean(V_pred_min, V_target)
    trip_loss = _l1_mean(V_pred_min, V_pred[indices[1]]) - _l1_mean(
        V_pred_min, V_pred[indices[-1]]
    )

    V_pred_min_ = V_pred_min.reshape(-1, 2)
    V_target_ = V_target.reshape(-1, 2)

    # Geometric distance length
    norm_loss = torch.abs(
        torch.cdist(V_pred_min_.unsqueeze(0), V_pred_min_.unsqueeze(0), p=2.0)
        - torch.cdist(V_target_.unsqueeze(0), V_target_.unsqueeze(0), p=2.0)
    ).mean()

    # Gemometric distance angle
    cos_loss = torch.abs(
        cdist_cosine_sim(V_pred_min_, V_pred_min_)
        - cdist_cosine_sim(V_target_, V_target_)
    ).mean()

    return error + 0.00001 * norm_loss + 0.0001 * trip_loss + 0.0001 * cos_loss


def graph_loss(V_pred, V_target):
    return implicit_likelihood_estimation_fast_with_trip_geo(V_pred, V_target)
