import gc
import os

import numpy as np
import torch
from torch.utils.data import TensorDataset

import pdebench


# ======================================================================#
def load_darcy_dataset(data_root: str):
    import scipy.io as scio

    datadir = os.path.join(data_root, "FNO", "darcy")

    train_path = os.path.join(datadir, "piececonst_r421_N1024_smooth1.mat")
    test_path = os.path.join(datadir, "piececonst_r421_N1024_smooth2.mat")
    ntrain = 1000
    ntest = 200

    r = 5
    h = int(((421 - 1) / r) + 1)
    s = h

    train_data = scio.loadmat(train_path)
    x_train = train_data["coeff"][:ntrain, ::r, ::r][:, :s, :s]
    x_train = x_train.reshape(ntrain, -1, 1)
    x_train = torch.from_numpy(x_train).float()
    y_train = train_data["sol"][:ntrain, ::r, ::r][:, :s, :s]
    y_train = y_train.reshape(ntrain, -1, 1)
    y_train = torch.from_numpy(y_train)

    test_data = scio.loadmat(test_path)
    x_test = test_data["coeff"][:ntest, ::r, ::r][:, :s, :s]
    x_test = x_test.reshape(ntest, -1, 1)
    x_test = torch.from_numpy(x_test).float()
    y_test = test_data["sol"][:ntest, ::r, ::r][:, :s, :s]
    y_test = y_test.reshape(ntest, -1, 1)
    y_test = torch.from_numpy(y_test)

    x_normalizer = pdebench.UnitGaussianNormalizer(x_train)
    y_normalizer = pdebench.UnitGaussianNormalizer(y_train)

    x_train = x_normalizer.encode(x_train)
    y_train = y_normalizer.encode(y_train)

    x_test = x_normalizer.encode(x_test)
    y_test = y_normalizer.encode(y_test)

    x = np.linspace(0, 1, s)
    y = np.linspace(0, 1, s)
    x, y = np.meshgrid(x, y)
    pos = np.c_[x.ravel(), y.ravel()]
    pos = torch.tensor(pos, dtype=torch.float).unsqueeze(0)

    pos_train = pos.repeat(ntrain, 1, 1)
    pos_test = pos.repeat(ntest, 1, 1)

    input_train = torch.cat([pos_train, x_train], dim=-1)
    output_train = y_train.to(torch.float)

    input_test = torch.cat([pos_test, x_test], dim=-1)
    output_test = y_test.to(torch.float)

    train_data = TensorDataset(input_train, output_train)
    test_data = TensorDataset(input_test, output_test)

    gc.collect()

    metadata = dict(
        x_normalizer=x_normalizer,
        y_normalizer=y_normalizer,
        c_in=3,
        c_out=1,
        time_cond=False,
        H=s,
        W=s,
        max_length=s * s,
    )

    return train_data, test_data, metadata


# ======================================================================#
def load_navier_stokes_dataset(data_root: str):
    import scipy.io as scio

    datadir = os.path.join(data_root, "FNO", "navier_stokes")
    data_path = os.path.join(datadir, "NavierStokes_V1e-5_N1200_T20.mat")

    r = 1
    h = int(((64 - 1) / r) + 1)
    ntrain = 1000
    ntest = 200
    t_in = 10
    t_out = 10

    data = scio.loadmat(data_path)
    train_a = data["u"][:ntrain, ::r, ::r, :t_in][:, :h, :h, :]
    train_a = train_a.reshape(train_a.shape[0], -1, train_a.shape[-1])
    train_a = torch.from_numpy(train_a)
    train_u = data["u"][:ntrain, ::r, ::r, t_in:t_out + t_in][:, :h, :h, :]
    train_u = train_u.reshape(train_u.shape[0], -1, train_u.shape[-1])
    train_u = torch.from_numpy(train_u)

    test_a = data["u"][-ntest:, ::r, ::r, :t_in][:, :h, :h, :]
    test_a = test_a.reshape(test_a.shape[0], -1, test_a.shape[-1])
    test_a = torch.from_numpy(test_a)
    test_u = data["u"][-ntest:, ::r, ::r, t_in:t_out + t_in][:, :h, :h, :]
    test_u = test_u.reshape(test_u.shape[0], -1, test_u.shape[-1])
    test_u = torch.from_numpy(test_u)

    x = np.linspace(0, 1, h)
    y = np.linspace(0, 1, h)
    x, y = np.meshgrid(x, y)
    pos = np.c_[x.ravel(), y.ravel()]
    pos = torch.tensor(pos, dtype=torch.float).unsqueeze(0)
    pos_train = pos.repeat(ntrain, 1, 1)
    pos_test = pos.repeat(ntest, 1, 1)

    train_dataset = TensorDataset(pos_train, train_a, train_u)
    test_dataset = TensorDataset(pos_test, test_a, test_u)

    metadata = dict(
        x_normalizer=pdebench.IdentityNormalizer(),
        y_normalizer=pdebench.IdentityNormalizer(),
        c_in=12,
        c_out=1,
        space_dim=2,
        fun_dim=t_in,
        rollout_steps=t_out,
        time_cond=False,
        H=h,
        W=h,
        max_length=h * h,
    )

    return train_dataset, test_dataset, metadata
