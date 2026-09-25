import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import pdebench
from mlutils import check_package_version_lteq


def sdf(mesh, resolution):
    import meshio
    import open3d as o3d
    import tempfile

    quads = mesh.cells_dict["quad"]

    idx = np.flatnonzero(quads[:, -1] == 0)
    out0 = np.empty((quads.shape[0], 2, 3), dtype=quads.dtype)

    out0[:, 0, 1:] = quads[:, 1:-1]
    out0[:, 1, 1:] = quads[:, 2:]

    out0[..., 0] = quads[:, 0, None]

    out0.shape = (-1, 3)

    mask = np.ones(out0.shape[0], dtype=bool)
    mask[idx * 2 + 1] = 0
    quad_to_tri = out0[mask]

    cells = [("triangle", quad_to_tri)]

    new_mesh = meshio.Mesh(mesh.points, cells)

    with tempfile.NamedTemporaryFile(delete=True, suffix=".ply") as tf:
        new_mesh.write(tf, file_format="ply")
        open3d_mesh = o3d.io.read_triangle_mesh(tf.name)
    open3d_mesh = o3d.t.geometry.TriangleMesh.from_legacy(open3d_mesh)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(open3d_mesh)

    domain_min = torch.tensor([-2.0, -1.0, -4.5])
    domain_max = torch.tensor([2.0, 4.5, 6.0])
    tx = np.linspace(domain_min[0], domain_max[0], resolution)
    ty = np.linspace(domain_min[1], domain_max[1], resolution)
    tz = np.linspace(domain_min[2], domain_max[2], resolution)
    grid = np.stack(np.meshgrid(tx, ty, tz, indexing="ij"), axis=-1).astype(np.float32)
    return torch.from_numpy(scene.compute_signed_distance(grid).numpy()).float()


class ShapeNetCarDataset(torch.utils.data.Dataset):
    # from https://github.com/ml-jku/UPT/blob/main/src/datasets/shapenet_car.py
    # generated with torch.randperm(889, generator=torch.Generator().manual_seed(0))[:189]
    TEST_INDICES = {
        550, 592, 229, 547, 62, 464, 798, 836, 5, 732, 876, 843, 367, 496,
        142, 87, 88, 101, 303, 352, 517, 8, 462, 123, 348, 714, 384, 190,
        505, 349, 174, 805, 156, 417, 764, 788, 645, 108, 829, 227, 555, 412,
        854, 21, 55, 210, 188, 274, 646, 320, 4, 344, 525, 118, 385, 669,
        113, 387, 222, 786, 515, 407, 14, 821, 239, 773, 474, 725, 620, 401,
        546, 512, 837, 353, 537, 770, 41, 81, 664, 699, 373, 632, 411, 212,
        678, 528, 120, 644, 500, 767, 790, 16, 316, 259, 134, 531, 479, 356,
        641, 98, 294, 96, 318, 808, 663, 447, 445, 758, 656, 177, 734, 623,
        216, 189, 133, 427, 745, 72, 257, 73, 341, 584, 346, 840, 182, 333,
        218, 602, 99, 140, 809, 878, 658, 779, 65, 708, 84, 653, 542, 111,
        129, 676, 163, 203, 250, 209, 11, 508, 671, 628, 112, 317, 114, 15,
        723, 746, 765, 720, 828, 662, 665, 399, 162, 495, 135, 121, 181, 615,
        518, 749, 155, 363, 195, 551, 650, 877, 116, 38, 338, 849, 334, 109,
        580, 523, 631, 713, 607, 651, 168,
    }

    def __init__(self, datadir, split="train", resolution=None, transform=None):
        super().__init__()
        self.datadir = datadir
        self.split = split
        self.resolution = resolution
        self.transform = transform

        # define spatial min/max of simulation for normalizing to [0, 1]
        self.domain_min = torch.tensor([-2.0, -1.0, -4.5])
        self.domain_max = torch.tensor([2.0, 4.5, 6.0])

        # mean/std for normalization (calculated on the 700 train samples)
        self.pressure_mean = torch.tensor(-36.3099)
        self.pressure_std = torch.tensor(48.5743)

        # discover uris
        self.uris = []
        for i in range(9):
            param_uri = self.datadir / f"param{i}"
            for name in sorted(os.listdir(param_uri)):
                sample_uri = param_uri / name
                if sample_uri.is_dir():
                    self.uris.append(sample_uri)
        assert len(self.uris) == 889, f"found {len(self.uris)} uris instead of 889"
        # split into train/test uris
        if split == "train":
            train_idxs = [i for i in range(len(self.uris)) if i not in self.TEST_INDICES]
            self.uris = [self.uris[train_idx] for train_idx in train_idxs]
            assert len(self.uris) == 700
        elif split == "test":
            self.uris = [self.uris[test_idx] for test_idx in self.TEST_INDICES]
            assert len(self.uris) == 189
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.uris)

    def __getitem__(self, idx):
        uri = self.uris[idx]
        if check_package_version_lteq("torch", "2.4"):
            pressure = torch.load(uri / "pressure.th")
            mesh_points = torch.load(uri / "mesh_points.th")
        else:
            pressure = torch.load(uri / "pressure.th", weights_only=True)
            mesh_points = torch.load(uri / "mesh_points.th", weights_only=True)

        pressure = (pressure - self.pressure_mean) / self.pressure_std
        mesh_points = (mesh_points - self.domain_min) / (self.domain_max - self.domain_min)

        return mesh_points.view(-1, 3), pressure.view(-1, 1)


def load_shapenet_car_dataset(data_root: str):
    import meshio

    src = os.path.join(data_root, "ShapeNet-Car", "mlcfd_data", "training_data")
    dst = os.path.join(data_root, "ShapeNet-Car", "preprocessed")

    src = Path(src).expanduser()
    dst = Path(dst).expanduser()

    uris = []
    for i in range(9):
        param_uri = src / f"param{i}"
        for name in sorted(os.listdir(param_uri)):
            # param folders contain .npy/.py/.txt files
            if "." in name:
                continue
            potential_uri = param_uri / name
            assert os.path.isdir(potential_uri)
            uris.append(potential_uri)
    print(f"found {len(uris)} samples")

    # Preprocessing
    if dst.exists() and all((dst / uri.relative_to(src)).exists() for uri in uris):
        print("Preprocessed files already exist, skipping processing")
    else:
        # .vtk files contains points that dont belong to the mesh -> filter them out
        for uri in tqdm(uris):
            reluri = uri.relative_to(src)
            out = dst / reluri
            out.mkdir(exist_ok=True, parents=True)

            # filter out mesh points that are not part of the shape
            mesh = meshio.read(uri / "quadpress_smpl.vtk")
            assert len(mesh.cells) == 1
            cell_block = mesh.cells[0]
            assert cell_block.type == "quad"
            unique = np.unique(cell_block.data)
            mesh_points = torch.from_numpy(mesh.points[unique]).float()
            pressure = torch.from_numpy(np.load(uri / "press.npy", mmap_mode="r")[unique]).float()
            torch.save(mesh_points, out / "mesh_points.th")
            torch.save(pressure, out / "pressure.th")

            # generate sdf
            for resolution in [32, 40, 48, 64, 80]:
                torch.save(sdf(mesh, resolution=resolution), out / f"sdf_res{resolution}.th")

    train_dataset = ShapeNetCarDataset(dst, split="train")
    test_dataset = ShapeNetCarDataset(dst, split="test")

    metadata = dict(
        x_normalizer=pdebench.IdentityNormalizer(),
        y_normalizer=pdebench.UnitGaussianNormalizer(torch.rand(10, 1)),
        c_in=3,
        c_out=1,
        time_cond=False,
        max_length=20_000,
    )

    metadata["y_normalizer"].mean = train_dataset.pressure_mean
    metadata["y_normalizer"].std = train_dataset.pressure_std

    return train_dataset, test_dataset, metadata

