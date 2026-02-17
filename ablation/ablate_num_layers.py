#
import os
import time
import shutil
import subprocess
import json, yaml
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.colors import LogNorm

import numpy as np
import pandas as pd
import argparse
import seaborn as sns

# local
import utils

#======================================================================#
PROJDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CASEDIR = os.path.join(PROJDIR, 'out', 'pdebench')
os.makedirs(CASEDIR, exist_ok=True)

#======================================================================#
def collect_data(dataset: str):
    # Check both possible locations for backward compatibility
    data_dir = os.path.join(CASEDIR, 'abl', f'abl_num_layers_{dataset}')
    if not os.path.exists(data_dir):
        # Fallback to old location
        data_dir = os.path.join(CASEDIR, f'abl_num_layers_{dataset}')

    # Initialize empty dataframe
    df = pd.DataFrame()

    # Check if case directory exists
    if os.path.exists(data_dir):
        # Get all subdirectories (each represents a case)
        cases = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

        for case in cases:
            case_path = os.path.join(data_dir, case)
            
            if not os.path.exists(os.path.join(case_path, 'config.yaml')):
                continue
            if not os.path.exists(os.path.join(case_path, 'num_params.txt')):
                continue

            # Initialize case data dictionary
            case_data = {}
            
            # Check for and load relative error data
            rel_error_path = os.path.join(case_path, 'ckpt10', 'rel_error.json')
            if os.path.exists(rel_error_path):
                with open(rel_error_path, 'r') as f:
                    rel_error = json.load(f)
                case_data.update({
                    'train_rel_error': rel_error.get('train_rel_error'),
                    'test_rel_error': rel_error.get('test_rel_error')
                })
            
            # Load config data
            config_path = os.path.join(case_path, 'config.yaml')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                case_data.update({
                    'channel_dim': config.get('channel_dim'),
                    'num_latents': config.get('num_latents'),
                    'num_blocks': config.get('num_blocks'),
                    'num_heads': config.get('num_heads'),
                    'num_layers_kv_proj': config.get('num_layers_kv_proj'),
                    'num_layers_ffn': config.get('num_layers_ffn'),
                    'num_layers_in_out_proj': config.get('num_layers_in_out_proj'),
                    'seed': config.get('seed'),
                })

            # Load num_params
            num_params_path = os.path.join(case_path, 'num_params.txt')
            if os.path.exists(num_params_path):
                with open(num_params_path, 'r') as f:
                    num_params = int(f.read().strip())
                case_data.update({'num_params': num_params})
            
            # Add case data to dataframe
            df = pd.concat([df, pd.DataFrame([case_data])], ignore_index=True)

            df['head_dim'] = df['channel_dim'] // df['num_heads']

        print(f"Collected {len(df)} cases for {dataset} dataset.")

    return df

def plot_results(dataset: str, df: pd.DataFrame):

    #---------------------------------------------------------#
    # Validate DataFrame has required columns
    required_columns = ['num_latents', 'num_layers_ffn', 'num_layers_kv_proj', 'test_rel_error']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if len(df) == 0:
        print(f"ERROR: No data collected for dataset '{dataset}'. Cannot plot results.")
        print(f"Checked data directories:")
        print(f"  1. {os.path.join(CASEDIR, 'abl', f'abl_num_layers_{dataset}')}")
        print(f"  2. {os.path.join(CASEDIR, f'abl_num_layers_{dataset}')}")
        return
    
    if missing_columns:
        print(f"ERROR: DataFrame is missing required columns: {missing_columns}")
        print(f"Available columns: {df.columns.tolist()}")
        return
    
    df = df.groupby(['num_latents', 'num_layers_ffn', 'num_layers_kv_proj']).mean().reset_index()

    configs = df[['num_latents',]].drop_duplicates()
    print(f"Found {len(configs)} unique configurations for num_layers_ffn, num_layers_kv_proj lineplot.")
    
    num_latents_list = configs['num_latents'].unique().tolist()

    #---------------------------------------------------------#
    # LINEPLOT of test error vs num_layers_ffn, num_layers_kv_proj
    #---------------------------------------------------------

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath}"
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    fontsize = 20

    ax1.set_ylabel(r'Test relative error', fontsize=fontsize)

    for ax in [ax1, ax2]:
        ax.set_xscale('linear')
        ax.set_yscale('log')
        ax.grid(True, which="both", ls="-", alpha=0.5)
        ax.set_ylim(3e-3, 7e-3)
        ax.set_yticks([3e-3, 4e-3, 5e-3, 6e-3, 7e-3])

    ax1.set_yticklabels(['3e-3', '4e-3', '5e-3', '6e-3', '7e-3'])
    ax2.set_yticklabels(['', '', '', '', ''])

    for ax in [ax1, ax2]:
        ax.tick_params(axis='both', which='major', labelsize=fontsize)

    ax1.set_xlabel(r'Number of layers (key/value projection)', fontsize=fontsize)
    ax2.set_xlabel(r'Number of layers (feedforward block)', fontsize=fontsize)

    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    markers = ['o', 's', 'D', 'v', '^', 'P', 'X', 'd', 'H', 'p']
    linestyles = ['-', '--', '-.', ':']

    linewidth = 2.5
    markersize = 10

    for i, num_latents in enumerate(num_latents_list):
        color = colors[i]
        label = r'M=%s' % num_latents

        df_ = df[df['num_latents'] == num_latents]
        df1 = df_[df_['num_layers_ffn'] == 3]
        df2 = df_[df_['num_layers_kv_proj'] == 3]
        
        kwargs = {
            'marker': markers[i], 'linestyle': linestyles[i],
            'color': color, 'linewidth': linewidth, 'markersize': markersize
        }

        ax1.plot(df1['num_layers_kv_proj'], df1['test_rel_error'], label=label, **kwargs)
        ax2.plot(df2['num_layers_ffn'], df2['test_rel_error'], label=None, **kwargs)

    #---------------------------------------------------------#
    # Place legend below the subplots with wider line representations
    legend = fig.legend(loc='lower center', ncol=4, 
              frameon=True, fancybox=False, shadow=False, fontsize=fontsize, 
              bbox_to_anchor=(0.5, 0.00), columnspacing=2.0, handletextpad=0.8,
              bbox_transform=fig.transFigure, handlelength=3.0, markerscale=1.0)

    # Adjust layout with extra space at bottom for legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.30)  # Increased bottom margin for legend

    # Save the figure with both plots
    out_path = os.path.join(PROJDIR, 'figs', f'abl_num_layers_{dataset}.pdf')
    fig.savefig(out_path)
    plt.close()

    #---------------------------------------------------------#
    return

#======================================================================#
def do_training(dataset: str, gpu_count: int = None, max_jobs_per_gpu: int = 2, reverse_queue: bool = False):

    if gpu_count is None:
        import torch
        gpu_count = torch.cuda.device_count()

    if dataset == 'elasticity': # E=500, BS=2, WD=1e-5
        epochs = 500
        batch_size = 2
        weight_decay = 1e-5
    else:
        raise ValueError(f"Dataset {dataset} not supported")
    
    print(f"Using {gpu_count} GPUs to run ablation study on {dataset} dataset.")

    # Create a queue of all jobs
    job_queue = []
    for seed in range(1):
        for num_latents in [16, 32, 64, 128]:
            for num_layers_ffn in range(6):
                add_job_to_queue(job_queue, dataset, num_latents=num_latents, num_layers_ffn=num_layers_ffn, num_layers_kv_proj=3, seed=seed, epochs=epochs, batch_size=batch_size, weight_decay=weight_decay)
            for num_layers_kv_proj in range(6):
                add_job_to_queue(job_queue, dataset, num_latents=num_latents, num_layers_ffn=3, num_layers_kv_proj=num_layers_kv_proj, seed=seed, epochs=epochs, batch_size=batch_size, weight_decay=weight_decay)

    utils.run_jobs(job_queue, gpu_count, max_jobs_per_gpu, reverse_queue,
                   dataset=dataset, epochs=epochs, batch_size=batch_size, weight_decay=weight_decay)

    return

#======================================================================#
def add_job_to_queue(
    job_queue: list, dataset: str, num_latents: int, num_layers_ffn: int, num_layers_kv_proj: int, seed: int,
    epochs: int = 500, batch_size: int = 2, weight_decay: float = 1e-5):

    exp_name = f'abl_num_layers_{dataset}_M_{str(num_latents)}_LMLP_{str(num_layers_ffn)}_LKV_{str(num_layers_kv_proj)}_seed_{str(seed)}'
    exp_name_base = os.path.join(f'abl_num_layers_{dataset}', exp_name)

    # Check both possible locations
    case_dir = os.path.join(CASEDIR, 'abl', exp_name_base)
    if not os.path.exists(case_dir):
        case_dir = os.path.join(CASEDIR, exp_name_base)
    
    if os.path.exists(case_dir):
        if os.path.exists(os.path.join(case_dir, 'ckpt10', 'rel_error.json')):
            print(f"Experiment {exp_name} exists. Skipping.")
            return
        else:
            print(f"Experiment {exp_name} exists but ckpt10/rel_error.json does not exist. Removing and re-running.")
            shutil.rmtree(case_dir)

    job_queue.append({
        #
        'exp_name': exp_name_base,
        'dataset': dataset,
        'seed': seed,
        #
        'epochs': epochs,
        'batch_size': batch_size,
        'weight_decay': weight_decay,
        'mixed_precision': False,
        #
        'model_type': 'flare_ablations',
        #
        'num_blocks': 8,
        'channel_dim': 64,
        'num_heads': 8,
        'num_latents': num_latents,
        'num_layers_kv_proj': num_layers_kv_proj,
        'num_layers_ffn': num_layers_ffn,
        'num_layers_in_out_proj': 2,
    })

    return

#======================================================================#
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Latent Cross Attention model ablation study')
    
    def str_to_bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser.add_argument('--eval', type=str_to_bool, default=False, help='Evaluate ablation study results')
    parser.add_argument('--train', type=str_to_bool, default=False, help='Train ablation study')
    parser.add_argument('--clean', type=str_to_bool, default=False, help='Clean ablation study results')

    parser.add_argument('--dataset', type=str, default='elasticity', help='Dataset to use')
    parser.add_argument('--gpu-count', type=int, default=None, help='Number of GPUs to use')
    parser.add_argument('--max-jobs-per-gpu', type=int, default=2, help='Maximum number of jobs per GPU')
    parser.add_argument('--reverse-queue', type=str_to_bool, default=False, help='Reverse queue')

    args = parser.parse_args()

    if args.train:
        do_training(args.dataset, args.gpu_count, args.max_jobs_per_gpu, args.reverse_queue)
    if args.eval:
        df = collect_data(args.dataset)
        plot_results(args.dataset, df)

    if not args.train and not args.eval and not args.clean:
        print("No action specified. Please specify either --train or --eval or --clean.")

    exit()

#======================================================================#
#
