import os

import numpy as np
import torch
from torch.utils.data import Subset, TensorDataset

import pdebench


# ======================================================================#
def plasticity_random_collate_fn(batch):
    shuffled_pos = []
    shuffled_t = []
    shuffled_a = []
    shuffled_u = []

    for pos, time_grid, features, target in batch:
        permuted_indices = torch.randperm(time_grid.size(0))
        shuffled_pos.append(pos)
        shuffled_t.append(time_grid[permuted_indices])
        shuffled_a.append(features)
        shuffled_u.append(target[..., permuted_indices])

    return [
        torch.stack(shuffled_pos, dim=0),
        torch.stack(shuffled_t, dim=0),
        torch.stack(shuffled_a, dim=0),
        torch.stack(shuffled_u, dim=0),
    ]


# ======================================================================#
def load_elasticity_dataset(data_root: str):
    datadir = os.path.join(data_root, "Geo-FNO", "elasticity")
    path_sigma = os.path.join(datadir, "Meshes", "Random_UnitCell_sigma_10.npy")
    path_xy = os.path.join(datadir, "Meshes", "Random_UnitCell_XY_10.npy")

    input_s = np.load(path_sigma, mmap_mode="r")
    input_s = torch.tensor(input_s, dtype=torch.float).permute(1, 0).unsqueeze(-1)
    input_xy = np.load(path_xy, mmap_mode="r")
    input_xy = torch.tensor(input_xy, dtype=torch.float).permute(2, 0, 1)

    ntrain = 1000
    ntest = 200

    y_normalizer = pdebench.UnitGaussianNormalizer(input_s[:ntrain])
    input_s = y_normalizer.encode(input_s)

    dataset = TensorDataset(input_xy, input_s)
    train_data = Subset(dataset, range(ntrain))
    test_data = Subset(dataset, range(len(dataset) - ntest, len(dataset)))

    metadata = dict(
        x_normalizer=pdebench.IdentityNormalizer(),
        y_normalizer=y_normalizer,
        c_in=2,
        c_out=1,
        space_dim=2,
        fun_dim=0,
        time_cond=False,
        max_length=972,
    )

    return train_data, test_data, metadata


# ======================================================================#
def load_plasticity_dataset(data_root: str):
    import scipy.io as scio

    datadir = os.path.join(data_root, "Geo-FNO", "plasticity")
    data_path = os.path.join(datadir, "plas_N987_T20.mat")

    ntrain = 900
    ntest = 80

    s1 = 101
    s2 = 31
    rollout_steps = 20
    deformation = 4

    r1 = 1
    r2 = 1
    s1 = int(((s1 - 1) / r1) + 1)
    s2 = int(((s2 - 1) / r2) + 1)

    data = scio.loadmat(data_path)
    input_ = torch.tensor(data["input"], dtype=torch.float)
    output = torch.tensor(data["output"], dtype=torch.float).transpose(-2, -1)
    x_train = input_[:ntrain, ::r1][:, :s1].reshape(ntrain, s1, 1).repeat(1, 1, s2)
    x_train = x_train.reshape(ntrain, -1, 1)
    y_train = output[:ntrain, ::r1, ::r2][:, :s1, :s2]
    y_train = y_train.reshape(ntrain, -1, deformation, rollout_steps)
    x_test = input_[-ntest:, ::r1][:, :s1].reshape(ntest, s1, 1).repeat(1, 1, s2)
    x_test = x_test.reshape(ntest, -1, 1)
    y_test = output[-ntest:, ::r1, ::r2][:, :s1, :s2]
    y_test = y_test.reshape(ntest, -1, deformation, rollout_steps)

    x_normalizer = pdebench.UnitGaussianNormalizer(x_train)
    y_normalizer = pdebench.IdentityNormalizer()
    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)

    x = np.linspace(0, 1, s1)
    y = np.linspace(0, 1, s2)
    x, y = np.meshgrid(x, y)
    pos = np.c_[x.ravel(), y.ravel()]
    pos = torch.tensor(pos, dtype=torch.float).unsqueeze(0)

    pos_train = pos.repeat(ntrain, 1, 1)
    pos_test = pos.repeat(ntest, 1, 1)

    t = np.linspace(0, 1, rollout_steps)
    t = torch.tensor(t, dtype=torch.float).unsqueeze(0)
    t_train = t.repeat(ntrain, 1)
    t_test = t.repeat(ntest, 1)

    train_data = TensorDataset(pos_train, t_train, x_train, y_train)
    test_data = TensorDataset(pos_test, t_test, x_test, y_test)

    metadata = dict(
        x_normalizer=x_normalizer,
        y_normalizer=y_normalizer,
        c_in=3,
        c_out=deformation,
        space_dim=2,
        fun_dim=1,
        time_cond=True,
        rollout_steps=rollout_steps,
        H=s1,
        W=s2,
        max_length=s1 * s2,
        train_collate_fn=plasticity_random_collate_fn,
    )

    return train_data, test_data, metadata


# ======================================================================#
def load_pipe_dataset(data_root: str):
    datadir = os.path.join(data_root, "Geo-FNO", "pipe")

    input_x_path = os.path.join(datadir, "Pipe_X.npy")
    input_y_path = os.path.join(datadir, "Pipe_Y.npy")
    output_sigma_path = os.path.join(datadir, "Pipe_Q.npy")

    ntrain = 1000
    ntest = 200
    n_total = 1200

    r1 = 1
    r2 = 1
    s1 = int(((129 - 1) / r1) + 1)
    s2 = int(((129 - 1) / r2) + 1)

    input_x = np.load(input_x_path, mmap_mode="r")
    input_x = torch.tensor(input_x, dtype=torch.float)
    input_y = np.load(input_y_path, mmap_mode="r")
    input_y = torch.tensor(input_y, dtype=torch.float)
    input_ = torch.stack([input_x, input_y], dim=-1)

    output = np.load(output_sigma_path, mmap_mode="r")[:, 0]
    output = torch.tensor(output, dtype=torch.float)
    x_train = input_[:n_total][:ntrain, ::r1, ::r2][:, :s1, :s2]
    y_train = output[:n_total][:ntrain, ::r1, ::r2][:, :s1, :s2]
    x_test = input_[:n_total][-ntest:, ::r1, ::r2][:, :s1, :s2]
    y_test = output[:n_total][-ntest:, ::r1, ::r2][:, :s1, :s2]

    x_train = x_train.reshape(ntrain, -1, 2)
    y_train = y_train.reshape(ntrain, -1, 1)

    x_test = x_test.reshape(ntest, -1, 2)
    y_test = y_test.reshape(ntest, -1, 1)

    x_normalizer = pdebench.UnitGaussianNormalizer(x_train)
    y_normalizer = pdebench.UnitGaussianNormalizer(y_train)

    x_train = x_normalizer.encode(x_train)
    y_train = y_normalizer.encode(y_train)

    x_test = x_normalizer.encode(x_test)
    y_test = y_normalizer.encode(y_test)

    train_data = TensorDataset(x_train, y_train)
    test_data = TensorDataset(x_test, y_test)

    metadata = dict(
        x_normalizer=x_normalizer,
        y_normalizer=y_normalizer,
        c_in=2,
        c_out=1,
        time_cond=False,
        H=s1,
        W=s2,
        max_length=s1 * s2,
    )

    return train_data, test_data, metadata


# ======================================================================#
def load_airfoil_steady_dataset(data_root: str):
    datadir = os.path.join(data_root, "Geo-FNO", "airfoil", "naca")

    input_x_path = os.path.join(datadir, "NACA_Cylinder_X.npy")
    input_y_path = os.path.join(datadir, "NACA_Cylinder_Y.npy")
    output_sigma_path = os.path.join(datadir, "NACA_Cylinder_Q.npy")

    ntrain = 1000
    ntest = 200

    r1 = 1
    r2 = 1
    s1 = int(((221 - 1) / r1) + 1)
    s2 = int(((51 - 1) / r2) + 1)

    input_x = np.load(input_x_path, mmap_mode="r")
    input_x = torch.tensor(input_x, dtype=torch.float)
    input_y = np.load(input_y_path, mmap_mode="r")
    input_y = torch.tensor(input_y, dtype=torch.float)
    input_ = torch.stack([input_x, input_y], dim=-1)

    output = np.load(output_sigma_path, mmap_mode="r")[:, 4]
    output = torch.tensor(output, dtype=torch.float)

    x_train = input_[:ntrain, ::r1, ::r2][:, :s1, :s2]
    y_train = output[:ntrain, ::r1, ::r2][:, :s1, :s2]
    x_test = input_[ntrain:ntrain + ntest, ::r1, ::r2][:, :s1, :s2]
    y_test = output[ntrain:ntrain + ntest, ::r1, ::r2][:, :s1, :s2]

    x_train = x_train.reshape(ntrain, -1, 2)
    y_train = y_train.reshape(ntrain, -1, 1)

    x_test = x_test.reshape(ntest, -1, 2)
    y_test = y_test.reshape(ntest, -1, 1)

    x_normalizer = pdebench.IdentityNormalizer()

    y_normalizer = pdebench.UnitGaussianNormalizer(y_train)
    y_train = y_normalizer.encode(y_train)
    y_test = y_normalizer.encode(y_test)

    train_data = TensorDataset(x_train, y_train)
    test_data = TensorDataset(x_test, y_test)

    metadata = dict(
        x_normalizer=x_normalizer,
        y_normalizer=y_normalizer,
        c_in=2,
        c_out=1,
        time_cond=False,
        H=s1,
        W=s2,
        max_length=s1 * s2,
    )

    return train_data, test_data, metadata
