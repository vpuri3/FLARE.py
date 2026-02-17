#
import torch

import os
import yaml
from tqdm import tqdm
from jsonargparse import CLI
from dataclasses import dataclass

# local
import am
import mlutils

#======================================================================#
import socket
MACHINE = socket.gethostname()

if MACHINE == "eagle":
    # VDEL Eagle - 1 node: 4x 2080Ti
    DATADIR_BASE = '/mnt/hdd1/vedantpu/data/NetFabb/'
elif MACHINE.startswith("gpu-node-"):
    # MAIL GPU - 1 node: 8x 2080Ti
    DATADIR_BASE = '/home/vedantpu/GeomLearning.py/data/'
elif MACHINE.startswith("v"):
    # PSC Bridges - 8x v100 32GB
    DATADIR_BASE = '/ocean/projects/eng170006p/vpuri1/data/'

#======================================================================#
dotdot = lambda dir: os.path.abspath(os.path.join(dir, '..'))
PROJDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CASEDIR = os.path.join(PROJDIR, 'out', 'am')
os.makedirs(CASEDIR, exist_ok=True)

TRITON_CACHE_DIR = os.path.join(dotdot(PROJDIR), ".triton")
os.makedirs(TRITON_CACHE_DIR, exist_ok=True)
os.environ["TRITON_CACHE_DIR"] = TRITON_CACHE_DIR

#======================================================================#
DATADIR_RAW        = os.path.join(DATADIR_BASE, 'netfabb_ti64_hires_raw')
DATADIR_TIMESERIES = os.path.join(DATADIR_BASE, 'netfabb_ti64_hires_timeseries')
DATADIR_FINALTIME  = os.path.join(DATADIR_BASE, 'netfabb_ti64_hires_finaltime')

SUBDIRS = [
    r'data_0-100',
    r'data_100-200',
    r'data_200-300',
    r'data_300-400',
    r'data_400-500',
    r'data_500-600',
    r'data_600-1000',
    r'data_1000-1500',
    r'data_1500-2000',
    r'data_2000-2500',
    r'data_2500-3000',
    r'data_3000-3500',
]

#======================================================================#
def timeseries_dataset(cfg):

    # read in include list
    include_list = os.path.join(PROJDIR, 'am_dataset_stats', 'include_list.txt')
    include_list = [line.strip() for line in open(include_list, 'r').readlines()]

    transform = am.TimeseriesDatasetTransform(
        disp=cfg.disp, vmstr=cfg.vmstr, temp=cfg.temp,
        sdf=cfg.sdf, mesh=cfg.GNN, metadata=False,
        merge=cfg.merge, interpolate=cfg.interpolate,
    )
    DATADIRS = [os.path.join(DATADIR_TIMESERIES, DIR) for DIR in SUBDIRS]
    DATADIRS = DATADIRS[:5]
    dataset = am.TimeseriesDataset(
        DATADIRS, merge=cfg.merge, include_list=include_list,
        transform=transform, verbose=GLOBAL_RANK==0,
        # force_reload=True,
    )

    # _data, data_ = am.split_timeseries_dataset(dataset, split=[0.8, 0.2])
    _data, data_, _ = am.split_timeseries_dataset(dataset, split=[0.2, 0.05, 0.75])
    # _data, data_, _ = am.split_timeseries_dataset(dataset, split=[0.05, 0.01, 0.94])

    if GLOBAL_RANK == 0:
        print(f"Loaded {len(dataset.case_files)} cases from {DATADIR_TIMESERIES}")
        print(f"Split into {len(_data.case_files)} train and {len(data_.case_files)} test cases")

    return

#======================================================================#
def vis_finaltime(cfg, force_reload=False, max_cases=10, num_workers=None):

    case_dir = os.path.join(CASEDIR, cfg.exp_name)

    # read in include list
    include_list = os.path.join(PROJDIR, 'am_dataset_stats', 'include_list.txt')
    include_list = [line.strip() for line in open(include_list, 'r').readlines()]

    # for DIR in SUBDIRS[:1]:
    for DIR in SUBDIRS:
        DATADIR = os.path.join(DATADIR_FINALTIME, DIR)
        dataset = am.FinaltimeDataset(
            DATADIR,
            include_list=include_list,
            force_reload=force_reload,
            num_workers=num_workers,
        )
        vis_dir = os.path.join(case_dir, DIR)
        os.makedirs(vis_dir, exist_ok=False)

        num_cases = min(len(dataset), max_cases)

        for icase in tqdm(range(num_cases), ncols=80):
            data = dataset[icase]
            ii = str(icase).zfill(3)
            case_name = data.metadata['case_name']
            out_file = os.path.join(vis_dir, f'{ii}_{case_name}.vtu')
            am.visualize_pyv(data, out_file)

    return

#======================================================================#
def vis_timeseries(cfg, force_reload=False, max_cases=50, num_workers=None):

    case_dir = os.path.join(CASEDIR, cfg.exp_name)

    # read in include list
    include_list = os.path.join(PROJDIR, 'am_dataset_stats', 'include_list.txt')
    include_list = [line.strip() for line in open(include_list, 'r').readlines()]

    # for DIR in SUBDIRS[:1]:
    for DIR in SUBDIRS[:10]:
        DATADIR = os.path.join(DATADIR_TIMESERIES, DIR)
        dataset = am.TimeseriesDataset(
            DATADIR,
            merge=cfg.merge,
            include_list=include_list,
            num_workers=num_workers,
            force_reload=force_reload,
        )
        vis_dir = os.path.join(case_dir, DIR)
        os.makedirs(vis_dir, exist_ok=False)

        case_names = [os.path.basename(f)[:-3] for f in dataset.case_files]
        num_cases  = min(len(case_names), max_cases)

        for icase in tqdm(range(num_cases), ncols=80):
            case_name = case_names[icase]
            idx_case  = dataset.case_range(case_name)
            case_data = dataset[idx_case]
            out_dir   = os.path.join(vis_dir, f'case{str(icase).zfill(2)}')
            am.visualize_timeseries_pyv(case_data, out_dir, icase, merge=cfg.merge)

    return

#======================================================================#
def test_extraction():
    ext_dir = "/home/shared/netfabb_ti64_hires_out/extracted/SandBox/"
    out_dir = "/home/shared/netfabb_ti64_hires_out/tmp/"
    errfile = os.path.join(out_dir, "error.txt")

    # consider a single case
    case_dir = os.path.join(ext_dir, "33084_344fec27_2")
    # case_dir = os.path.join(ext_dir, "101635_11b839a3_5")
    # case_dir = os.path.join(ext_dir, "83419_82b6bccd_0")
    # case_dir = os.path.join(ext_dir, "77980_f6ed5970_4")

    info = am.get_case_info(case_dir)
    print(info)
    # results = am.get_timeseries_results(case_dir)

    am.extract_from_dir(ext_dir, out_dir, errfile, timeseries=True)

    return

def extract(cfg):

    os.makedirs(DATADIR_TIMESERIES if cfg.timeseries else DATADIR_FINALTIME, exist_ok=True)

    for DIR in SUBDIRS:
        zip_file = os.path.join(DATADIR_RAW, DIR + ".zip")
        out_dir  = os.path.join(DATADIR_TIMESERIES if cfg.timeseries else DATADIR_FINALTIME, DIR)
        am.extract_from_zip(zip_file, out_dir, timeseries=cfg.timeseries)

    return

#======================================================================#
@dataclass
class Config:
    '''
    Extract and process AM dataset
    '''

    # modes
    extract: bool = False
    filter: bool = False
    visualize: bool = False
    make_statistics_plot: bool = False
    make_dataset: bool = False

    # dataset
    timeseries: bool = False
    force_reload: bool = False

    # save directory
    exp_name: str = 'exp'

#======================================================================#
if __name__ == "__main__":

    #===============#
    cfg = CLI(Config, as_positional=False)
    #===============#

    if not (cfg.extract or cfg.filter or cfg.visualize or cfg.make_dataset or cfg.make_statistics_plot):
        print("No mode selected. Select one of extract, filter, visualize, make_dataset, make_statistics_plot.")
        exit()

    DISTRIBUTED = mlutils.is_torchrun()
    GLOBAL_RANK = int(os.environ['RANK']) if DISTRIBUTED else 0
    WORLD_SIZE = int(os.environ['WORLD_SIZE']) if DISTRIBUTED else 1

    assert WORLD_SIZE == 1, "Running in distributed mode is not supported. Please run with WORLD_SIZE=1. Don't worry this code is largely multithreaded."

    if cfg.extract:
        extract(cfg)

    if cfg.filter:
        am.compute_dataset_statistics(PROJDIR, DATADIR_FINALTIME, SUBDIRS, force_reload=cfg.force_reload)
        am.make_include_list(PROJDIR)
        am.compute_filtered_dataset_statistics(PROJDIR)

    if cfg.make_statistics_plot:
        am.make_statistics_plot(PROJDIR, filtered=True)
        am.make_statistics_plot(PROJDIR, filtered=False)

    if cfg.make_dataset:
        am.make_dataset(PROJDIR, DATADIR_FINALTIME)
        
    # create a new output directory
    if cfg.visualize:
        case_dir = os.path.join(CASEDIR, cfg.exp_name)
        if os.path.exists(case_dir):
            nd = len([dir for dir in os.listdir(CASEDIR) if dir.startswith(cfg.exp_name)])
            cfg.exp_name = cfg.exp_name + '_' + str(nd).zfill(2)
            case_dir = os.path.join(CASEDIR, cfg.exp_name)

        os.makedirs(case_dir)
        config_file = os.path.join(case_dir, 'config.yaml')
        print(f'Saving config to {config_file}')
        with open(config_file, 'w') as f:
            yaml.safe_dump(vars(cfg), f)

        if cfg.timeseries:
            vis_timeseries(cfg, force_reload=cfg.force_reload)
        else:
            vis_finaltime(cfg, force_reload=cfg.force_reload)

    exit()
#