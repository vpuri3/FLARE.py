import os

import numpy as np
import torch

import pdebench


class DrivAerMLDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir_base, split):
        self.data_dir_base = data_dir_base
        self.split = split
        self.data_dir = os.path.join(self.data_dir_base, self.split)

        assert self.split in ["train", "test"], f"Invalid split: {self.split}. Must be one of: 'train', 'test'."

        npz_files = [f for f in os.listdir(self.data_dir) if f.endswith(".npz")]
        self.npz_files = [os.path.join(self.data_dir, f) for f in npz_files]

        if len(self.npz_files) == 0:
            raise ValueError(f"No .npz files found in directory: {self.data_dir}")

        self.p_mean = -229.845718
        self.p_std = 269.598572
        self.xyz_min = torch.tensor([-0.9425, -1.1314, -0.3176])
        self.xyz_max = torch.tensor([4.1325, 1.1317, 1.2445])

    def __len__(self):
        return len(self.npz_files)

    def __getitem__(self, idx):
        npz_file_path = self.npz_files[idx]

        data = np.load(npz_file_path, mmap_mode="r")
        x = data["surface_mesh_centers"]
        p = data["surface_fields"]  # [pressure, wall_shear_x, wall_shear_y, wall_shear_z]
        x = torch.tensor(x, dtype=torch.float32).view(-1, 3)
        p = torch.tensor(p, dtype=torch.float32).view(-1, 4)[:, 0:1]  # only pressure

        x = (x - self.xyz_min) / (self.xyz_max - self.xyz_min)
        p = (p - self.p_mean) / self.p_std

        return x, p


def load_drivaerml_dataset(dataset_name: str, data_root: str):
    num_points_dict = {
        "10k": int(10e3),
        "40k": int(40e3),
        "50k": int(50e3),
        "100k": int(100e3),
        "200k": int(200e3),
        "300k": int(300e3),
        "400k": int(400e3),
        "500k": int(500e3),
        "1m": int(1e6),
    }

    num_points_str = dataset_name.split("_")[-1]
    assert num_points_str in num_points_dict, (
        f"Invalid dataset name: {dataset_name}. Valid names are: drivaerml_<{list(num_points_dict.keys())}>."
    )
    num_points = num_points_dict[num_points_str]

    datadir_presampled = os.path.join(data_root, "DrivAerML", f"drivaerml_surface_presampled_{num_points_str}")

    train_dataset = DrivAerMLDataset(datadir_presampled, split="train")
    test_dataset = DrivAerMLDataset(datadir_presampled, split="test")

    metadata = dict(
        x_normalizer=pdebench.IdentityNormalizer(),
        y_normalizer=pdebench.UnitGaussianNormalizer(torch.rand(10, 1)),
        c_in=3,
        c_out=1,
        time_cond=False,
        max_length=num_points,
    )

    metadata["y_normalizer"].mean = torch.tensor(train_dataset.p_mean)
    metadata["y_normalizer"].std = torch.tensor(train_dataset.p_std)

    return train_dataset, test_dataset, metadata

