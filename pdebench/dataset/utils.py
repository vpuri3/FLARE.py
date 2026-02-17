import os
import gc
import copy
import json
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset, Subset

import pdebench
from mlutils import check_package_version_lteq
import re

import mlutils
DISTRIBUTED = mlutils.is_torchrun()
GLOBAL_RANK = int(os.environ['RANK']) if DISTRIBUTED else 0

#======================================================================#
def load_dataset(
        dataset_name: str,
        DATADIR_BASE: str,
        PROJDIR: str,
        force_reload: bool = False,
        mesh: bool = False,
        cells: bool = False,
        max_cases: int = None,
        max_steps: int = None,
        init_step: int = None,
        init_case: int = None,
        exclude: bool = True,
        train_rollout_noise: float = 0.
    ):
    """Load a dataset by name.
    
    Args:
        dataset_name (str): Name of the dataset to load.
        
    Returns:
        tuple: (train_data, test_data, metadata) containing the loaded datasets and optional metadata dictionary
    """
    #----------------------------------------------------------------#
    # Geo-FNO datasets
    #----------------------------------------------------------------#
    if dataset_name == 'elasticity':
        DATADIR = os.path.join(DATADIR_BASE, 'Geo-FNO', 'elasticity')
        PATH_Sigma = os.path.join(DATADIR, 'Meshes', 'Random_UnitCell_sigma_10.npy')
        PATH_XY = os.path.join(DATADIR, 'Meshes', 'Random_UnitCell_XY_10.npy')

        input_s = np.load(PATH_Sigma, mmap_mode='r')
        input_s = torch.tensor(input_s, dtype=torch.float).permute(1, 0).unsqueeze(-1)
        input_xy = np.load(PATH_XY, mmap_mode='r')
        input_xy = torch.tensor(input_xy, dtype=torch.float).permute(2, 0, 1)
        
        ntrain = 1000
        ntest = 200
        
        y_normalizer = pdebench.UnitGaussianNormalizer(input_s[:ntrain])
        input_s = y_normalizer.encode(input_s)

        dataset = TensorDataset(input_xy, input_s)
        train_data = Subset(dataset, range(ntrain))
        test_data = Subset(dataset, range(len(dataset)-ntest, len(dataset)))
        
        metadata = dict(
            x_normalizer=pdebench.IdentityNormalizer(),
            y_normalizer=y_normalizer,
            c_in=2,
            c_out=1,
            time_cond=False,
            max_length=972,
        )

        return train_data, test_data, metadata

    elif dataset_name == 'plasticity':
        import scipy.io as scio

        DATADIR = os.path.join(DATADIR_BASE, 'Geo-FNO', 'plasticity')
        data_path = os.path.join(DATADIR, 'plas_N987_T20.mat')
        
        N = 987
        ntrain = 900
        ntest = 80

        s1 = 101
        s2 = 31
        T = 20
        Deformation = 4

        r1 = 1
        r2 = 1
        s1 = int(((s1 - 1) / r1) + 1)
        s2 = int(((s2 - 1) / r2) + 1)

        data = scio.loadmat(data_path)
        input = torch.tensor(data['input'], dtype=torch.float)
        output = torch.tensor(data['output'], dtype=torch.float).transpose(-2, -1)
        x_train = input[:ntrain, ::r1][:, :s1].reshape(ntrain, s1, 1).repeat(1, 1, s2)
        x_train = x_train.reshape(ntrain, -1, 1)
        y_train = output[:ntrain, ::r1, ::r2][:, :s1, :s2]
        y_train = y_train.reshape(ntrain, -1, Deformation, T)
        x_test = input[-ntest:, ::r1][:, :s1].reshape(ntest, s1, 1).repeat(1, 1, s2)
        x_test = x_test.reshape(ntest, -1, 1)
        y_test = output[-ntest:, ::r1, ::r2][:, :s1, :s2]
        y_test = y_test.reshape(ntest, -1, Deformation, T)

        x_normalizer = pdebench.UnitGaussianNormalizer(x_train)
        x_train = x_normalizer.encode(x_train)
        x_test = x_normalizer.encode(x_test)

        x = np.linspace(0, 1, s1)
        y = np.linspace(0, 1, s2)
        x, y = np.meshgrid(x, y)
        pos = np.c_[x.ravel(), y.ravel()]
        pos = torch.tensor(pos, dtype=torch.float).unsqueeze(0)

        pos_train = pos.repeat(ntrain, 1, 1)
        pos_test = pos.repeat(ntest, 1, 1)

        t = np.linspace(0, 1, T)
        t = torch.tensor(t, dtype=torch.float).unsqueeze(0)
        t_train = t.repeat(ntrain, 1)
        t_test = t.repeat(ntest, 1)
        
        print(pos_train.shape, t_train.shape, x_train.shape, y_train.shape)
        exit()

        train_data = TensorDataset(pos_train, t_train, x_train, y_train)
        test_data  = TensorDataset(pos_test, t_test, x_test, y_test)
        
        # collate_fn = random_collate_fn

        metadata = dict(
            x_normalizer=x_normalizer,
            y_normalizer=y_normalizer,
            c_in=0,
            c_out=0,
            time_cond=False,
            H=s1,
            W=s2,
        )
        
        return train_data, test_data, metadata
        
    elif dataset_name == 'pipe':
        DATADIR = os.path.join(DATADIR_BASE, 'Geo-FNO', 'pipe')
        
        INPUT_X = os.path.join(DATADIR, 'Pipe_X.npy')
        INPUT_Y = os.path.join(DATADIR, 'Pipe_Y.npy')
        OUTPUT_Sigma = os.path.join(DATADIR, 'Pipe_Q.npy')

        ntrain = 1000
        ntest = 200
        N = 1200

        r1 = 1
        r2 = 1
        s1 = int(((129 - 1) / r1) + 1)
        s2 = int(((129 - 1) / r2) + 1)

        inputX = np.load(INPUT_X, mmap_mode='r')
        inputX = torch.tensor(inputX, dtype=torch.float)
        inputY = np.load(INPUT_Y, mmap_mode='r')
        inputY = torch.tensor(inputY, dtype=torch.float)
        input = torch.stack([inputX, inputY], dim=-1)

        output = np.load(OUTPUT_Sigma, mmap_mode='r')[:, 0]
        output = torch.tensor(output, dtype=torch.float)
        x_train = input[ :N][:ntrain, ::r1, ::r2][:, :s1, :s2]
        y_train = output[:N][:ntrain, ::r1, ::r2][:, :s1, :s2]
        x_test = input[  :N][-ntest:, ::r1, ::r2][:, :s1, :s2]
        y_test = output[ :N][-ntest:, ::r1, ::r2][:, :s1, :s2]

        x_train = x_train.reshape(ntrain, -1, 2)
        y_train = y_train.reshape(ntrain, -1, 1)

        x_test = x_test.reshape(ntest, -1, 2)
        y_test = y_test.reshape(ntest, -1, 1)

        x_normalizer = pdebench.UnitGaussianNormalizer(x_train)
        y_normalizer = pdebench.UnitGaussianNormalizer(y_train)

        x_train = x_normalizer.encode(x_train)
        y_train = y_normalizer.encode(y_train)

        x_test  = x_normalizer.encode(x_test)
        y_test  = y_normalizer.encode(y_test)

        train_data = TensorDataset(x_train, y_train)
        test_data  = TensorDataset(x_test , y_test )

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
        
    elif dataset_name == 'airfoil_steady':
        DATADIR = os.path.join(DATADIR_BASE, 'Geo-FNO', 'airfoil', 'naca')

        INPUT_X = os.path.join(DATADIR, 'NACA_Cylinder_X.npy')
        INPUT_Y = os.path.join(DATADIR, 'NACA_Cylinder_Y.npy')
        OUTPUT_Sigma = os.path.join(DATADIR, 'NACA_Cylinder_Q.npy')

        ntrain = 1000
        ntest = 200

        r1 = 1
        r2 = 1
        s1 = int(((221 - 1) / r1) + 1)
        s2 = int(((51 - 1) / r2) + 1)

        inputX = np.load(INPUT_X, mmap_mode='r')
        inputX = torch.tensor(inputX, dtype=torch.float)
        inputY = np.load(INPUT_Y, mmap_mode='r')
        inputY = torch.tensor(inputY, dtype=torch.float)
        input = torch.stack([inputX, inputY], dim=-1)

        output = np.load(OUTPUT_Sigma, mmap_mode='r')[:, 4]
        output = torch.tensor(output, dtype=torch.float)

        x_train = input[:ntrain, ::r1, ::r2][:, :s1, :s2]
        y_train = output[:ntrain, ::r1, ::r2][:, :s1, :s2]
        x_test = input[ntrain:ntrain + ntest, ::r1, ::r2][:, :s1, :s2]
        y_test = output[ntrain:ntrain + ntest, ::r1, ::r2][:, :s1, :s2]

        x_train = x_train.reshape(ntrain, -1, 2)
        y_train = y_train.reshape(ntrain, -1, 1)

        x_test = x_test.reshape(ntest, -1, 2)
        y_test = y_test.reshape(ntest, -1, 1)

        x_normalizer = pdebench.IdentityNormalizer()
        y_normalizer = pdebench.IdentityNormalizer()

        # x_normalizer = pdebench.UnitCubeNormalizer(x_train)
        # x_train = x_normalizer.encode(x_train)
        # x_test  = x_normalizer.encode(x_test)

        y_normalizer = pdebench.UnitGaussianNormalizer(y_train)
        y_train = y_normalizer.encode(y_train)
        y_test  = y_normalizer.encode(y_test)

        train_data = TensorDataset(x_train, y_train)
        test_data  = TensorDataset(x_test , y_test)

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
        
    #----------------------------------------------------------------#
    # FNO datasets
    #----------------------------------------------------------------#
    elif dataset_name == 'darcy':
        import scipy.io as scio

        DATADIR = os.path.join(DATADIR_BASE, 'FNO', 'darcy')

        #------------------------------------#
        # Transolver
        #------------------------------------#
        train_path = os.path.join(DATADIR, 'piececonst_r421_N1024_smooth1.mat')
        test_path = os.path.join(DATADIR, 'piececonst_r421_N1024_smooth2.mat')
        ntrain = 1000
        ntest = 200

        r = 5 # downsample
        h = int(((421 - 1) / r) + 1)
        s = h
        dx = 1.0 / s

        # #------------------------------------#
        # # LNO: https://github.com/L-I-M-I-T/LatentNeuralOperator/pull/6/files
        # #------------------------------------#
        # train_path = os.path.join(DATADIR, 'piececonst_r241_N1024_smooth1.mat')
        # test_path = os.path.join(DATADIR, 'piececonst_r241_N1024_smooth2.mat')
        # ntrain = 1000
        # ntest = 200

        # r = 1 # downsample
        # h = int(((241 - 1) / r) + 1)
        # s = h
        # dx = 1.0 / s
        # #------------------------------------#

        train_data = scio.loadmat(train_path)
        x_train = train_data['coeff'][:ntrain, ::r, ::r][:, :s, :s]
        x_train = x_train.reshape(ntrain, -1, 1)
        x_train = torch.from_numpy(x_train).float()
        y_train = train_data['sol'][:ntrain, ::r, ::r][:, :s, :s]
        y_train = y_train.reshape(ntrain, -1, 1)
        y_train = torch.from_numpy(y_train)

        test_data = scio.loadmat(test_path)
        x_test = test_data['coeff'][:ntest, ::r, ::r][:, :s, :s]
        x_test = x_test.reshape(ntest, -1, 1)
        x_test = torch.from_numpy(x_test).float()
        y_test = test_data['sol'][:ntest, ::r, ::r][:, :s, :s]
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
        test_data  = TensorDataset(input_test , output_test )

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

    elif dataset_name == 'navier_stokes':
        import scipy.io as scio

        DATADIR = os.path.join(DATADIR_BASE, 'FNO', 'navier_stokes')
        data_path = os.path.join(DATADIR, 'NavierStokes_V1e-5_N1200_T20.mat')

        r = 1
        h = int(((64 - 1) / r) + 1)
        ntrain = 1000
        ntest = 200
        T_in = 10
        T = 10

        data = scio.loadmat(data_path)
        train_a = data['u'][:ntrain, ::r, ::r, :T_in][:, :h, :h, :]
        train_a = train_a.reshape(train_a.shape[0], -1, train_a.shape[-1])
        train_a = torch.from_numpy(train_a)
        train_u = data['u'][:ntrain, ::r, ::r, T_in:T + T_in][:, :h, :h, :]
        train_u = train_u.reshape(train_u.shape[0], -1, train_u.shape[-1])
        train_u = torch.from_numpy(train_u)

        test_a = data['u'][-ntest:, ::r, ::r, :T_in][:, :h, :h, :]
        test_a = test_a.reshape(test_a.shape[0], -1, test_a.shape[-1])
        test_a = torch.from_numpy(test_a)
        test_u = data['u'][-ntest:, ::r, ::r, T_in:T + T_in][:, :h, :h, :]
        test_u = test_u.reshape(test_u.shape[0], -1, test_u.shape[-1])
        test_u = torch.from_numpy(test_u)

        x = np.linspace(0, 1, h)
        y = np.linspace(0, 1, h)
        x, y = np.meshgrid(x, y)
        pos = np.c_[x.ravel(), y.ravel()]
        pos = torch.tensor(pos, dtype=torch.float).unsqueeze(0)
        pos_train = pos.repeat(ntrain, 1, 1)
        pos_test = pos.repeat(ntest, 1, 1)

        # print(pos_train.shape) # [1000, 4096,  2]
        # print(train_a.shape)   # [1000, 4096, 10]
        # print(train_u.shape)   # [1000, 4096, 10]

        # train_input = torch.cat([pos_train, train_a], dim=-1)
        # test_input  = torch.cat([pos_test , test_a ], dim=-1)

        # train_output = train_u
        # test_output  = test_u

        # train_dataset = TensorDataset(train_input, train_output)
        # test_dataset  = TensorDataset(test_input , test_output )

        train_dataset = TensorDataset(pos_train, train_a, train_u)
        test_dataset  = TensorDataset(pos_test , test_a , test_u )

        metadata = dict(
            x_normalizer=pdebench.IdentityNormalizer(),
            y_normalizer=pdebench.IdentityNormalizer(),
            c_in=12,
            c_out=1,
            time_cond=False,
            H=h,
            W=h,
            max_length=h * h,
        )

        return train_dataset, test_dataset, metadata
        
    #----------------------------------------------------------------#
    # MeshGraphNets datasets
    #----------------------------------------------------------------#
    elif dataset_name in ['airfoil_dynamic', 'cylinder_flow']:

        from pdebench.dataset.timeseries import TimeseriesDataset, TimeseriesDatasetTransform

        force_reload = False
        max_cases = 10
        max_steps = 500
        init_step = 100
        init_case = 0
        exclude = True

        DATADIR = os.path.join(DATADIR_BASE, 'MeshGraphNets', dataset_name)
        transform_kwargs = dict(mesh=mesh, cells=cells)

        dataset_kwargs = dict(
            force_reload=force_reload,
            max_cases=max_cases,
            max_steps=max_steps,
            init_step=init_step,
        )

        train_transform = TimeseriesDatasetTransform(dataset_name, **transform_kwargs)
        test_transform  = TimeseriesDatasetTransform(dataset_name, **transform_kwargs)

        train_dataset = TimeseriesDataset(DATADIR, PROJDIR, 'train', transform=train_transform, **dataset_kwargs, init_case=init_case, exclude=exclude)
        test_dataset  = TimeseriesDataset(DATADIR, PROJDIR, 'test' , transform=test_transform , **dataset_kwargs, exclude=exclude)

        test_dataset.transform.apply_normalization_stats(train_dataset.norm_stats)

        # Looks like there is some disparity bw train_dataset and test_dataset
        train_dataset, test_dataset = split_timeseries_dataset(train_dataset, split=[0.8, 0.2])

        metadata = dict(
            x_normalizer=pdebench.IdentityNormalizer(),
            y_normalizer=pdebench.IdentityNormalizer(),
            c_in=11,
            c_edge=2,
            c_out=2,
            time_cond=True,
        )

        if GLOBAL_RANK == 0:
            print(f"Loaded {dataset_name} dataset with {train_dataset.num_cases} train and {test_dataset.num_cases} test cases.")
            print(f"Number of time-steps: {train_dataset.trajectory_length}")

            if max_steps is not None:
                print(f"Limiting to {max_steps} time-steps")
            if max_cases is not None:
                print(f"Limiting to {max_cases} cases")

            for graph in train_dataset:
                print(graph)
                break

        return train_dataset, test_dataset, metadata

    #----------------------------------------------------------------#
    # ShapeNet-Car datasets
    #----------------------------------------------------------------#
    elif dataset_name == 'shapenet_car':
        import meshio
        from pathlib import Path

        SRC = os.path.join(DATADIR_BASE, 'ShapeNet-Car', 'mlcfd_data', 'training_data')
        DST = os.path.join(DATADIR_BASE, 'ShapeNet-Car', 'preprocessed')
    
        SRC = Path(SRC).expanduser()
        DST = Path(DST).expanduser()

        uris = []
        for i in range(9):
            param_uri = SRC / f"param{i}"
            for name in sorted(os.listdir(param_uri)):
                # param folders contain .npy/.py/txt files
                if "." in name:
                    continue
                potential_uri = param_uri / name
                assert os.path.isdir(potential_uri)
                uris.append(potential_uri)
        print(f"found {len(uris)} samples")

        # Preprocessing
        if DST.exists() and all((DST / uri.relative_to(SRC)).exists() for uri in uris):
            print("Preprocessed files already exist, skipping processing")
        else:
            # .vtk files contains points that dont belong to the mesh -> filter them out
            mesh_point_counts = []
            for uri in tqdm(uris):
                reluri = uri.relative_to(SRC)
                out = DST / reluri
                out.mkdir(exist_ok=True, parents=True)

                # filter out mesh points that are not part of the shape
                mesh = meshio.read(uri / "quadpress_smpl.vtk")
                assert len(mesh.cells) == 1
                cell_block = mesh.cells[0]
                assert cell_block.type == "quad"
                unique = np.unique(cell_block.data)
                mesh_point_counts.append(len(unique))
                mesh_points = torch.from_numpy(mesh.points[unique]).float()
                pressure = torch.from_numpy(np.load(uri / "press.npy", mmap_mode='r')[unique]).float()
                torch.save(mesh_points, out / "mesh_points.th")
                torch.save(pressure, out / "pressure.th")

                # generate sdf
                for resolution in [32, 40, 48, 64, 80]:
                    torch.save(sdf(mesh, resolution=resolution), out / f"sdf_res{resolution}.th")

        train_dataset = ShapeNetCarDataset(DST, split='train')
        test_dataset  = ShapeNetCarDataset(DST, split='test')

        metadata = dict(
            x_normalizer=pdebench.IdentityNormalizer(),
            y_normalizer=pdebench.UnitGaussianNormalizer(torch.rand(10,1)),
            c_in=3,
            c_out=1,
            time_cond=False,
            max_length=20_000,
        )

        metadata['y_normalizer'].mean = train_dataset.pressure_mean
        metadata['y_normalizer'].std  = train_dataset.pressure_std

        return train_dataset, test_dataset, metadata
        
    #----------------------------------------------------------------#
    # DrivAerML DATASET
    #----------------------------------------------------------------#
    elif dataset_name.startswith('drivaerml'):
        
        num_points_dict = {
            '10k'  : int(10e3),
            '40k'  : int(40e3),
            '50k'  : int(50e3),
            '100k' : int(100e3),
            '200k' : int(200e3),
            '300k' : int(300e3),
            '400k' : int(400e3),
            '500k' : int(500e3),
            '1m'   : int(1e6),
        }

        num_points_str = dataset_name.split('_')[-1]
        assert num_points_str in num_points_dict, f"Invalid dataset name: {dataset_name}. Valid names are: drivaerml_<{list(num_points_dict.keys())}>."
        num_points = num_points_dict[num_points_str]

        DATADIR_PRESAMPLED = os.path.join(DATADIR_BASE, 'DrivAerML', f'drivaerml_surface_presampled_{num_points_str}')

        train_dataset = DrivAerMLDataset(DATADIR_PRESAMPLED, split = 'train')
        test_dataset  = DrivAerMLDataset(DATADIR_PRESAMPLED, split = 'test')
        
        metadata = dict(
            x_normalizer=pdebench.IdentityNormalizer(),
            y_normalizer=pdebench.UnitGaussianNormalizer(torch.rand(10,1)),
            c_in=3,
            c_out=1,
            time_cond=False,
            max_length=num_points,
        )

        metadata['y_normalizer'].mean = torch.tensor(train_dataset.p_mean)
        metadata['y_normalizer'].std  = torch.tensor(train_dataset.p_std)

        return train_dataset, test_dataset, metadata

    #----------------------------------------------------------------#
    # LPBF DATASET
    #----------------------------------------------------------------#
    elif dataset_name == 'lpbf':
        import am

        transform = am.FinaltimeDatasetTransform(disp=True, vmstr=False, mesh=False)

        train_dataset = create_lpbf_dataset(split='train', transform=transform)
        test_dataset  = create_lpbf_dataset(split='test', transform=transform)

        mean_disp = 0.
        std_disp  = 0.
        for graph in train_dataset:
            disp = graph.y
            mean_disp += disp.mean(dim=0)
            std_disp  += disp.std(dim=0)
        mean_disp /= len(train_dataset)
        std_disp  /= len(train_dataset)
        # print(f"mean_disp: {mean_disp}, std_disp: {std_disp}")

        x_normalizer = pdebench.IdentityNormalizer()
        y_normalizer = pdebench.UnitGaussianNormalizer(torch.rand(3,1))
        x_normalizer.mean = mean_disp
        x_normalizer.std  = std_disp
        
        metadata = dict(
            x_normalizer=x_normalizer,
            y_normalizer=y_normalizer,
            c_in=3,
            c_edge=3,
            c_out=1,
            time_cond=False,
            max_length=50_000,
        )

        return train_dataset, test_dataset, metadata

    #----------------------------------------------------------------# 
    else:
        raise ValueError(f"Dataset {dataset_name} not found.") 

#======================================================================#
def split_timeseries_dataset(dataset, split=None, indices=None):
    if split is None and indices is None:
        raise ValueError('split_timeseries_dataset: pass in either indices or split')

    num_cases = dataset.num_cases
    included_cases = dataset.included_cases

    if indices is None:
        indices = [int(s * num_cases) for s in split]
        indices[-1] += num_cases - sum(indices)
    indices = torch.utils.data.random_split(range(num_cases), indices)

    num_split = len(indices)
    subsets = [copy.deepcopy(dataset) for _ in range(num_split)]

    for s in range(num_split):
        subset = subsets[s]
        subset.included_cases = [included_cases[i] for i in indices[s]]
        subset.num_cases = len(subset.included_cases)
        
    # assert there is no overlap between the included cases
    for split1 in range(num_split):
        for split2 in range(num_split):
            if split1 != split2:
                assert not any(c in subsets[split1].included_cases for c in subsets[split2].included_cases)

    return subsets

#======================================================================#
def sdf(mesh, resolution):
    import meshio
    import tempfile
    import open3d as o3d

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

    def __init__(self, datadir, split='train', resolution=None, transform=None):
        super().__init__()
        self.datadir = datadir
        self.split = split
        self.resolution = resolution
        self.transform = transform

        # define spatial min/max of simulation for normalizing to [0, 1]
        # min: [-1.7978, -0.7189, -4.2762]
        # max: [1.8168, 4.3014, 5.8759]
        self.domain_min = torch.tensor([-2.0, -1.0, -4.5])
        self.domain_max = torch.tensor([2.0, 4.5, 6.0])

        # mean/std for normalization (calculated on the 700 train samples)
        # import torch
        # from datasets.shapenet_car import ShapenetCar
        # ds = ShapenetCar(global_root="/local00/bioinf/shapenet_car", split="train")
        # targets = [ds.getitem_pressure(i) for i in range(len(ds))]
        # targets = torch.stack(targets)
        # targets.mean()
        # targets.std()
        self.pressure_mean = torch.tensor(-36.3099)
        self.pressure_std  = torch.tensor( 48.5743)

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
        if split == 'train':
            train_idxs = [i for i in range(len(self.uris)) if i not in self.TEST_INDICES]
            self.uris = [self.uris[train_idx] for train_idx in train_idxs]
            assert len(self.uris) == 700
        elif split == 'test':
            self.uris = [self.uris[test_idx] for test_idx in self.TEST_INDICES]
            assert len(self.uris) == 189
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.uris)

    def __getitem__(self, idx):
        uri = self.uris[idx]
        if check_package_version_lteq('torch', '2.4'):
            pressure = torch.load(uri / "pressure.th")
            mesh_points = torch.load(uri / "mesh_points.th")
        else:
            pressure = torch.load(uri / "pressure.th", weights_only=True)
            mesh_points = torch.load(uri / "mesh_points.th", weights_only=True)

        pressure = (pressure - self.pressure_mean) / self.pressure_std
        mesh_points = (mesh_points - self.domain_min) / (self.domain_max - self.domain_min)

        return mesh_points.view(-1, 3), pressure.view(-1, 1)

#======================================================================#
class DrivAerMLDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir_base, split):
        self.data_dir_base = data_dir_base
        self.split = split
        self.data_dir = os.path.join(self.data_dir_base, self.split)

        assert self.split in ['train', 'test'], f"Invalid split: {self.split}. Must be one of: 'train', 'test'."

        npz_files = [f for f in os.listdir(self.data_dir) if f.endswith('.npz')]
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

        data = np.load(npz_file_path, mmap_mode='r')
        x = data['surface_mesh_centers']
        p = data['surface_fields'] # [pressure, wall_shear_x, wall_shear_y, wall_shear_z]
        x = torch.tensor(x, dtype=torch.float32).view(-1, 3)
        p = torch.tensor(p, dtype=torch.float32).view(-1, 4)[:, 0:1] # only pressure

        x = (x - self.xyz_min) / (self.xyz_max - self.xyz_min)
        p = (p - self.p_mean) / self.p_std

        return x, p

#======================================================================#
# LPBF DATASET
#======================================================================#
def create_lpbf_dataset(*args, **kwargs):

    import torch_geometric as pyg
    import datasets, huggingface_hub

    class LPBFDataset(pyg.data.Dataset):
        def __init__(self, split='train', transform=None):
            assert split in ['train', 'test'], f"Invalid split: {split}. Must be one of: 'train', 'test'."

            self.repo_id = 'vedantpuri/LPBF_FLARE'
            
            print(f"Initializing {split} dataset...")
            
            # Fast initialization: Load dataset index first (lightweight)
            import time
            start_time = time.time()
            self.dataset = datasets.load_dataset(self.repo_id, split=split, keep_in_memory=True)
            dataset_time = time.time() - start_time
            print(f"Dataset index load: {dataset_time:.2f}s")

            # Lazy cache initialization - only download when needed
            self._cache_dir = None
            
            print(f"✅ Loaded {len(self.dataset)} samples for {split} split")

            super().__init__(None, transform=transform)

        @property
        def cache_dir(self):
            """Lazy loading of cache directory - only download when first sample is accessed."""
            if self._cache_dir is None:
                print("Downloading repository files on first access...")
                import time
                start_time = time.time()
                self._cache_dir = huggingface_hub.snapshot_download(self.repo_id, repo_type="dataset")
                download_time = time.time() - start_time
                print(f"Repository download/cache: {download_time:.2f}s")
                print(f"Cache directory: {self._cache_dir}")
            return self._cache_dir

        def len(self):
            return len(self.dataset)

        def get(self, idx):
            # Get file path from index
            entry = self.dataset[idx]
            rel_path = entry["file"]
            npz_path = os.path.join(self.cache_dir, rel_path)

            # Load NPZ file (main bottleneck check)
            data = np.load(npz_path, allow_pickle=True)
            graph = pyg.data.Data()

            # Convert to tensors efficiently
            for key, value in data.items():
                if key == "_metadata":
                    graph["metadata"] = json.loads(value[0])["metadata"]
                else:
                    # Use torch.from_numpy for faster conversion when possible
                    if value.dtype.kind == "f":
                        tensor = torch.from_numpy(value.astype(np.float32))
                    else:
                        tensor = torch.from_numpy(value.astype(np.int64)) if value.dtype != np.int64 else torch.from_numpy(value)
                    graph[key] = tensor

            # Set standard attributes
            graph.x = graph.pos
            graph.y = graph.disp[:, 2]

            return graph
        
    return LPBFDataset(*args, **kwargs)

#======================================================================#
#