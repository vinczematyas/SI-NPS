import numpy as np
import torch
import torch.distributions as tdist
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import math
import networkx as nx
from tqdm import tqdm
import pickle


def anorm(p1, p2):
    NORM = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    if NORM == 0:
        return 0
    return 1 / (NORM)


def seq_to_graph(seq_, seq_rel, norm_lap_matr=True):
    seq_ = seq_.squeeze()
    seq_rel = seq_rel.squeeze()
    seq_len = seq_.shape[2]
    max_nodes = seq_.shape[0]

    V = np.zeros((seq_len, max_nodes, 2))
    A = np.zeros((seq_len, max_nodes, max_nodes))
    for s in range(seq_len):
        step_ = seq_[:, :, s]
        step_rel = seq_rel[:, :, s]
        for h in range(len(step_)):
            V[s, h, :] = step_rel[h]
            A[s, h, h] = 1
            for k in range(h + 1, len(step_)):
                l2_norm = anorm(step_rel[h], step_rel[k])
                A[s, h, k] = l2_norm
                A[s, k, h] = l2_norm
        if norm_lap_matr:
            G = nx.from_numpy_matrix(A[s, :, :])
            A[s, :, :] = nx.normalized_laplacian_matrix(G).toarray()

    # Create Local graphs

    return torch.from_numpy(V).type(torch.double), torch.from_numpy(A).type(
        torch.double
    )


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


def read_file(_path, delim="\t"):
    data = []
    if delim == "tab":
        delim = "\t"
    elif delim == "space":
        delim = " "
    with open(_path, "r") as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""

    def __init__(
        self,
        data_dir,
        obs_len=8,
        pred_len=12,
        skip=1,
        threshold=0.002,
        min_ped=1,
        delim="\t",
        norm_lap_matr=True,
    ):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        self.max_peds_in_frame = 0
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.norm_lap_matr = norm_lap_matr

        args_str = (
            data_dir
            + str(obs_len)
            + str(pred_len)
            + str(skip)
            + str(threshold)
            + str(min_ped)
            + str(norm_lap_matr)
        )
        pkl_path = "eth-ucy-pkls/" + args_str.replace("/", "_") + ".pkl"

        if os.path.exists(pkl_path):
            print("Dataset found, Loading dataset from:", pkl_path)
            with open(pkl_path, "rb") as f:
                __data = pickle.load(f)

                self.obs_traj = __data["obs_traj"]
                self.pred_traj = __data["pred_traj"]
                self.obs_traj_rel = __data["obs_traj_rel"]
                self.pred_traj_rel = __data["pred_traj_rel"]
                self.non_linear_ped = __data["non_linear_ped"]
                self.loss_mask = __data["loss_mask"]
                self.v_obs = __data["v_obs"]
                self.A_obs = __data["A_obs"]
                self.v_pred = __data["v_pred"]
                self.A_pred = __data["A_pred"]
                self.num_seq = __data["num_seq"]
                self.seq_start_end = __data["seq_start_end"]

        else:
            all_files = os.listdir(self.data_dir)
            all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
            num_peds_in_seq = []
            seq_list = []
            seq_list_rel = []
            loss_mask_list = []
            non_linear_ped = []
            for path in all_files:
                data = read_file(path, delim)
                frames = np.unique(data[:, 0]).tolist()
                frame_data = []
                for frame in frames:
                    frame_data.append(data[frame == data[:, 0], :])
                num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / skip))

                for idx in range(0, num_sequences * self.skip + 1, skip):
                    curr_seq_data = np.concatenate(
                        frame_data[idx : idx + self.seq_len], axis=0
                    )
                    peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                    self.max_peds_in_frame = max(
                        self.max_peds_in_frame, len(peds_in_curr_seq)
                    )
                    curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                    curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                    curr_loss_mask = np.zeros((len(peds_in_curr_seq), self.seq_len))
                    num_peds_considered = 0
                    _non_linear_ped = []
                    for _, ped_id in enumerate(peds_in_curr_seq):
                        curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                        # curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                        pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                        pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                        if pad_end - pad_front != self.seq_len:
                            continue
                        curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                        curr_ped_seq = curr_ped_seq
                        # Make coordinates relative
                        rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                        rel_curr_ped_seq[:, 1:] = (
                            curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                        )
                        _idx = num_peds_considered
                        curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                        curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                        # Linear vs Non-Linear Trajectory
                        _non_linear_ped.append(
                            poly_fit(curr_ped_seq, pred_len, threshold)
                        )
                        curr_loss_mask[_idx, pad_front:pad_end] = 1
                        num_peds_considered += 1

                    if num_peds_considered > min_ped:
                        non_linear_ped += _non_linear_ped
                        num_peds_in_seq.append(num_peds_considered)
                        loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                        seq_list.append(curr_seq[:num_peds_considered])
                        seq_list_rel.append(curr_seq_rel[:num_peds_considered])

            self.num_seq = len(seq_list)
            seq_list = np.concatenate(seq_list, axis=0)
            seq_list_rel = np.concatenate(seq_list_rel, axis=0)
            loss_mask_list = np.concatenate(loss_mask_list, axis=0)
            non_linear_ped = np.asarray(non_linear_ped)

            # Convert numpy -> Torch Tensor
            self.obs_traj = torch.from_numpy(seq_list[:, :, : self.obs_len]).type(
                torch.double
            )
            self.pred_traj = torch.from_numpy(seq_list[:, :, self.obs_len :]).type(
                torch.double
            )
            self.obs_traj_rel = torch.from_numpy(
                seq_list_rel[:, :, : self.obs_len]
            ).type(torch.double)
            self.pred_traj_rel = torch.from_numpy(
                seq_list_rel[:, :, self.obs_len :]
            ).type(torch.double)
            self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.double)
            self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.double)
            cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
            self.seq_start_end = [
                (start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])
            ]
            # Convert to Graphs
            self.v_obs = []
            self.A_obs = []
            self.v_pred = []
            self.A_pred = []
            print("Processing Data .....")
            pbar = tqdm(total=len(self.seq_start_end))
            for ss in range(len(self.seq_start_end)):
                pbar.update(1)

                start, end = self.seq_start_end[ss]

                v_, a_ = seq_to_graph(
                    self.obs_traj[start:end, :],
                    self.obs_traj_rel[start:end, :],
                    self.norm_lap_matr,
                )
                self.v_obs.append(v_.clone())
                self.A_obs.append(a_.clone())
                v_, a_ = seq_to_graph(
                    self.pred_traj[start:end, :],
                    self.pred_traj_rel[start:end, :],
                    self.norm_lap_matr,
                )
                self.v_pred.append(v_.clone())
                self.A_pred.append(a_.clone())
            pbar.close()

            __data = {}
            __data["obs_traj"] = self.obs_traj
            __data["pred_traj"] = self.pred_traj
            __data["obs_traj_rel"] = self.obs_traj_rel
            __data["pred_traj_rel"] = self.pred_traj_rel
            __data["non_linear_ped"] = self.non_linear_ped
            __data["loss_mask"] = self.loss_mask
            __data["v_obs"] = self.v_obs
            __data["A_obs"] = self.A_obs
            __data["v_pred"] = self.v_pred
            __data["A_pred"] = self.A_pred
            __data["num_seq"] = self.num_seq
            __data["seq_start_end"] = self.seq_start_end
            print("Saving dataset to:", pkl_path)
            with open(pkl_path, "wb") as output_file:
                pickle.dump(__data, output_file)

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]

        out = [
            self.obs_traj[start:end, :],
            self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :],
            self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end],
            self.loss_mask[start:end, :],
            self.v_obs[index],
            self.A_obs[index],
            self.v_pred[index],
            self.A_pred[index],
        ]
        return out


def load_traj_data(dataset_name):
    tr_dl = DataLoader(
        TrajectoryDataset(f"./datasets/{dataset_name}/train/"),
        batch_size=1,
        shuffle=True,
        num_workers=0,
    )
    val_dl = DataLoader(
        TrajectoryDataset(f"./datasets/{dataset_name}/val/"),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    test_dl = DataLoader(
        TrajectoryDataset(f"./datasets/{dataset_name}/test/"),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    return tr_dl, val_dl, test_dl


class TrajectoryAugmenter:
    def __init__(self, total_time=20, split_time=8, data_loader=None):
        self.total_time = total_time
        self.split_time = split_time
        self.choice_dist = tdist.uniform.Uniform(0, 1)
        self.data_loader = data_loader
        self.bin = 1 / 7

    def abs_to_rel_split(self, v, full_traj, _split=8):
        v = torch.cat(
            (
                torch.zeros(v.shape[0], v.shape[1], v.shape[2], 1),
                v[:, :, :, 1:] - v[:, :, :, :-1],
            ),
            dim=-1,
        ).permute(0, 3, 1, 2)
        return (
            v[:, :_split, ...],
            v[:, _split:, ...],
            full_traj[..., : self.split_time],
            full_traj[..., self.split_time :],
        )  # v_obs, v_tr

    def augment(self, V_obs, V_tr, obs_traj, pred_traj_gt):
        decision = self.choice_dist.sample().item()
        if 0 <= decision < self.bin:
            return V_obs, V_tr, obs_traj, pred_traj_gt
        elif self.bin <= decision < self.bin * 2:
            return self._aug_jitter(obs_traj, pred_traj_gt)
        elif self.bin * 2 <= decision < self.bin * 3:
            return self._aug_flip_mirror(obs_traj, pred_traj_gt)
        elif self.bin * 3 <= decision < self.bin * 4:
            return self._aug_flip_reverse(obs_traj, pred_traj_gt)
        elif self.bin * 4 <= decision < self.bin * 5:
            return self._aug_nodes(V_obs, V_tr, obs_traj, pred_traj_gt)
        elif self.bin * 5 <= decision < self.bin * 6:
            return self._aug_rot(obs_traj, pred_traj_gt)
        elif self.bin * 6 <= decision <= self.bin * 7:
            return self._aug_speed(obs_traj, pred_traj_gt)

    def _aug_jitter(self, obs_traj, pred_traj_gt, jit_per=0.1):
        u = tdist.uniform.Uniform(
            torch.Tensor([-jit_per, -jit_per]), torch.Tensor([jit_per, jit_per])
        )
        full_traj = torch.cat((obs_traj, pred_traj_gt), dim=-1)
        return self.abs_to_rel_split(
            full_traj + u.sample(sample_shape=(self.total_time,)).T.to(obs_traj.device),
            full_traj,
            self.split_time,
        )

    def _aug_flip_mirror(self, obs_traj, pred_traj_gt):
        full_traj = torch.cat((obs_traj, pred_traj_gt), dim=-1)
        full_traj = torch.flip(full_traj, [2, 3])
        return self.abs_to_rel_split(full_traj, full_traj, self.split_time)

    def _aug_flip_reverse(self, obs_traj, pred_traj_gt):
        full_traj = torch.cat((obs_traj, pred_traj_gt), dim=-1)
        full_traj = torch.flip(full_traj, [3])
        return self.abs_to_rel_split(full_traj, full_traj, self.split_time)

    def _aug_nodes(self, V_obs, V_tr, obs_traj, pred_traj_gt, int_low=1, int_high=4):
        V_obs_lst = [V_obs]
        V_tr_lst = [V_tr]
        obs_traj_lst = [obs_traj]
        pred_traj_gt_lst = [pred_traj_gt]
        for i in range(torch.randint(low=int_low, high=int_high, size=(1,)).item()):
            (
                obs_traj,
                pred_traj_gt,
                obs_traj_rel,
                pred_traj_gt_rel,
                non_linear_ped,
                loss_mask,
                V_obs_,
                A_obs,
                V_tr_,
                A_tr,
            ) = next(iter(self.data_loader))
            V_obs_lst.append(V_obs_)
            V_tr_lst.append(V_tr_)
            obs_traj_lst.append(obs_traj)
            pred_traj_gt_lst.append(pred_traj_gt)

        V_obs = torch.cat(V_obs_lst, dim=2)
        V_tr = torch.cat(V_tr_lst, dim=2)
        obs_traj = torch.cat(obs_traj_lst, dim=1)
        pred_traj_gt = torch.cat(pred_traj_gt_lst, dim=1)
        return V_obs, V_tr, obs_traj, pred_traj_gt

    def _aug_rot(self, obs_traj, pred_traj_gt):
        degrees = [
            0.2617993877991494,
            0.5235987755982988,
            0.7853981633974483,
            1.0471975511965976,
            1.3089969389957472,
            1.5707963267948966,
            1.8325957145940461,
            2.0943951023931953,
            2.356194490192345,
            2.6179938779914944,
            2.8797932657906435,
            3.141592653589793,
            3.4033920413889427,
            3.6651914291880923,
            3.9269908169872414,
            4.1887902047863905,
            4.4505895925855405,
            4.71238898038469,
            4.974188368183839,
            5.235987755982989,
            5.497787143782138,
            5.759586531581287,
            6.021385919380437,
        ]
        rot_degree = degrees[torch.randint(0, 23, (1,)).item()]
        rot_matrix = torch.Tensor(
            [
                [np.cos(rot_degree), np.sin(rot_degree)],
                [-np.sin(rot_degree), np.cos(rot_degree)],
            ]
        ).double()
        full_traj = torch.cat((obs_traj, pred_traj_gt), dim=-1)
        full_traj = (
            torch.matmul(rot_matrix, full_traj.transpose(2, 3).unsqueeze(-1))
            .squeeze(-1)
            .transpose(2, 3)
        )
        return self.abs_to_rel_split(full_traj, full_traj, self.split_time)

    def _aug_speed(self, obs_traj, pred_traj_gt, inc_distance=1):
        inc = (
            tdist.uniform.Uniform(
                torch.Tensor([-inc_distance]), torch.Tensor([inc_distance])
            )
            .sample()
            .item()
        )
        full_traj = torch.cat((obs_traj, pred_traj_gt), dim=-1) + torch.arange(
            0, inc, inc / 20
        ).to(obs_traj.device)
        return self.abs_to_rel_split(full_traj, full_traj, self.split_time)


def calculate_velocity(obs, tgt=None):
    if len(obs.shape) == 3:
        vel_obs = torch.cat(
            (torch.zeros(obs.shape[0], 1, 2), obs[:, 1:] - obs[:, :-1]), dim=1
        )
    else:
        vel_obs = torch.cat(
            (torch.zeros(20, obs.shape[1], 1, 2), obs[:, :, 1:] - obs[:, :, :-1]), dim=2
        )
    return vel_obs


def calc_ade_fde(pred, tgt):
    # pred shape | bestofn x num_peds x 12 x 2
    # tgt shape  | num_peds x 12 x 2
    ade_dist = torch.linalg.vector_norm(
        pred.permute(1, 0, 2, 3) - tgt[:, None, ...], dim=-1
    ).sum(-1)
    min_ade_dist = torch.min(ade_dist, dim=-1)[0]
    ade = min_ade_dist.sum(-1) / (12 * pred.shape[1])

    fde_dist = torch.linalg.vector_norm(
        pred.permute(1, 0, 2, 3)[:, :, -1, :] - tgt[:, None, -1, :], dim=-1
    )
    min_fde_dist = torch.min(fde_dist, dim=-1)[0]
    fde = min_fde_dist.sum(-1) / pred.shape[1]

    return ade, fde
