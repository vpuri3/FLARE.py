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
    data_dir = os.path.join(CASEDIR, f'scaling_lca_{dataset}')

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
                    'num_layers_mlp': config.get('num_layers_mlp'),
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

    output_dir = os.path.join(CASEDIR, f'scaling_lca_{dataset}_analysis')
    os.makedirs(output_dir, exist_ok=True)

    #---------------------------------------------------------#
    # HEATMAP across Blocks vs Latent Blocks
    #---------------------------------------------------------
    cmap = 'RdYlBu_r'
    vmin, vmax = (1e-2, 1e-1) if dataset in ['shapenet_car'] else (1e-3, 1e-2)

    configs = df[['channel_dim', 'num_blocks']].drop_duplicates()

    print(f"Found {len(configs)} unique configurations for M vs H heatmap.")

    for _, config in configs.iterrows():

        channel_dim = config['channel_dim']
        num_blocks = config['num_blocks']

        df_ = df[
            (df['channel_dim'] == channel_dim) &
            (df['num_blocks'] == num_blocks)
        ]

        name_str = f'B_{num_blocks}_C_{channel_dim}'
        title_str = f'# Blocks: {num_blocks}, Channel Dim: {channel_dim}'

        fig, ax = plt.subplots(figsize=(8, 6))
        fig.suptitle(title_str)

        # Create pivot tables with numeric values for coloring and annotations
        pivot_test = df_.pivot_table(
            values='test_rel_error',
            columns='num_latents',
            index='num_heads',
            aggfunc='mean'
        )
        pivot_train = df_.pivot_table(
            values='train_rel_error',
            columns='num_latents',
            index='num_heads',
            aggfunc='mean'
        )
        pivot_params = df_.pivot_table(
            values='num_params',
            columns='num_latents',
            index='num_heads',
            aggfunc='mean'
        )

        if pivot_train.empty or pivot_test.empty:
            plt.close()
            continue
        # annot_params = pivot_params.map(lambda x: format_param(x))

        # Create combined annotations
        combined_annot = pd.DataFrame(index=pivot_test.index, columns=pivot_test.columns, dtype=str)
        for i in range(combined_annot.shape[0]):
            for j in range(combined_annot.shape[1]):
                train_val = format_sci(pivot_train.iloc[i, j])
                test_val = format_sci(pivot_test.iloc[i, j])
                params_val = format_param(pivot_params.iloc[i, j])
                combined_annot.iloc[i, j] = f"{train_val}\n{test_val}\n{params_val}"

        annot_kws = {"size": 11, "weight": "bold"}
        linear_scale_kw = {'vmin': vmin, 'vmax': vmax}

        heatmap = sns.heatmap(pivot_test, annot=combined_annot, fmt='', cmap=cmap, ax=ax, **linear_scale_kw, annot_kws=annot_kws, linewidths=0.5, linecolor='black')

        # Format colorbar ticks in scientific notation
        cbar = heatmap.collections[0].colorbar
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
        cbar.set_label('Test relative error')

        ax.set_title('Train relative error/ test relative error/ parameter count')
        ax.set_xlabel('Number of clusters (M)')
        ax.set_ylabel('Number of heads (H)')
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'heatmap_H_vs_M_{name_str}.png'))
        plt.close()

    #---------------------------------------------------------#
    # LINEPLOT of test error vs num_heads/ head_dim
    #---------------------------------------------------------

    df_ = df
    df_ = df_[df_['channel_dim'].isin([64])]
    df_ = df_[df_['num_blocks'].isin([4, 8])]

    configs = df_[['num_blocks', 'channel_dim', 'num_latents']].drop_duplicates()

    print(f"Found {len(configs)} unique configurations for num heads lineplot.")

    configs = configs.sort_values(by=['num_blocks', 'channel_dim', 'num_latents'])

    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.grid(True, which="both", ls="-", alpha=0.5)
    ax1.set_xlabel('Head dimension')
    ax1.set_ylabel('Test relative error')
    ax1.set_title(f'Test relative error vs head dimension')

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log')
    ax2.grid(True, which="both", ls="-", alpha=0.5)
    ax2.set_xlabel('Number of heads')
    ax2.set_ylabel('Test relative error')
    ax2.set_title(f'Test relative error vs number of heads')

    for _, config in configs.iterrows():

        channel_dim = config['channel_dim']
        num_blocks = config['num_blocks']
        num_latents = config['num_latents']

        label = f'B={num_blocks}, C={channel_dim}, M={num_latents}'

        df__ = df_[
            (df_['num_blocks'] == num_blocks) &
            (df_['channel_dim'] == channel_dim) &
            (df_['num_latents'] == num_latents)
        ]

        df__ = df__.sort_values(by='head_dim')

        if df__.empty:
            continue

        ax1.plot(df__['head_dim'], df__['test_rel_error'], label=label, marker='o', linestyle='-')
        ax2.plot(df__['num_heads'], df__['test_rel_error'], label=label, marker='o', linestyle=linestyle)

    ax1.legend()
    ax2.legend()
    fig1.savefig(os.path.join(output_dir, f'lineplot_head_dim.png'))
    fig2.savefig(os.path.join(output_dir, f'lineplot_num_heads.png'))
    plt.close()

    #---------------------------------------------------------#
    # LINEPLOT of test error vs number of clusters
    #---------------------------------------------------------

    df_ = df
    df_ = df_[df_['num_blocks'].isin([8])]

    configs = df_[['num_blocks', 'channel_dim', 'num_heads']].drop_duplicates()

    print(f"Found {len(configs)} unique configurations for num clusters lineplot.")

    configs = configs.sort_values(by=['num_blocks', 'channel_dim', 'num_heads'])

    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.grid(True, which="both", ls="-", alpha=0.5)
    ax1.set_xlabel('Number of clusters')
    ax1.set_ylabel('Test relative error')
    ax1.set_title(f'Test relative error vs number of clusters')

    for _, config in configs.iterrows():

        channel_dim = config['channel_dim']
        num_blocks = config['num_blocks']
        num_heads = config['num_heads']

        label = f'B={num_blocks}, C={channel_dim}, H={num_heads}'

        df__ = df_[
            (df_['num_blocks'] == num_blocks) &
            (df_['channel_dim'] == channel_dim) &
            (df_['num_heads'] == num_heads)
        ]

        df__ = df__.sort_values(by='num_latents')

        if df__.empty:
            continue

        ax1.plot(df__['num_latents'], df__['test_rel_error'], label=label, marker='o', linestyle='-')

    ax1.legend()
    fig1.savefig(os.path.join(output_dir, f'lineplot_num_clusters.png'))
    plt.close()

    #---------------------------------------------------------#
    return

#======================================================================#
def format_param(x):
    if x > 1e6:
        return f"{x/1e6:.1f}m"
    elif x > 1e3:
        return f"{x/1e3:.1f}k"
    else:
        return f"{x:.1f}"

def format_sci(x):
    if pd.isna(x):
        return ""
    return f"{x:.2e}".replace("e+0", "e+").replace("e-0", "e-")

#======================================================================#
def eval_results(dataset: str):
    df = collect_data(dataset)
    plot_results(dataset, df)
    return

#======================================================================#
def do_training(
    dataset: str,
    gpu_count: int = None,
    max_jobs_per_gpu: int = 2,
    reverse_queue: bool = False,
    ):
    if gpu_count is None:
        import torch
        gpu_count = torch.cuda.device_count()
    if dataset == 'elasticity':
        epochs = 500
        batch_size = 2
        weight_decay = 1e-5
    elif dataset == 'shapenet_car':
        epochs = 200
        batch_size = 1
        weight_decay = 5e-2
    else:
        raise ValueError(f"Dataset {dataset} not supported")

    print(f"Using {gpu_count} GPUs to run scaling study on {dataset} dataset.")

    # Create a queue of all jobs
    job_queue = []
    for num_blocks in [1, 2, 4, 8, 16]:
        for channel_dim in [32, 64, 128]:
            for num_latents in [8, 16, 32, 64, 128]:
                for head_dim in [4, 8, 16, 32, 64, 128]:

                    num_heads = channel_dim // head_dim
                    if head_dim > channel_dim:
                        continue

                    exp_name = f'scaling_lca_{dataset}_C_{str(channel_dim)}_M_{str(num_latents)}_B_{str(num_blocks)}_H_{str(num_heads)}'
                    exp_name = os.path.join(f'scaling_lca_{dataset}', exp_name)

                    case_dir = os.path.join(CASEDIR, exp_name)
                    if os.path.exists(case_dir):
                        if os.path.exists(os.path.join(case_dir, 'ckpt10', 'rel_error.json')):
                            print(f"Experiment {exp_name} exists. Skipping.")
                            continue
                        else:
                            print(f"Experiment {exp_name} exists but ckpt10/rel_error.json does not exist. Removing and re-running.")
                            shutil.rmtree(case_dir)

                    job_queue.append({
                        'channel_dim': channel_dim,
                        'num_blocks': num_blocks,
                        'num_heads': num_heads,
                        'num_latents': num_latents,
                        'exp_name': exp_name
                    })

    utils.run_jobs(job_queue, gpu_count, max_jobs_per_gpu, reverse_queue,
                   dataset=dataset, epochs=epochs, batch_size=batch_size, weight_decay=weight_decay)

    return

#======================================================================#
def add_job_to_queue(
    job_queue: list, dataset: str, num_layers_mlp: int, num_layers_kv_proj: int, seed: int,
    epochs: int = 500, batch_size: int = 2, weight_decay: float = 1e-5):

    exp_name = f'scaling_lca_{dataset}_C_{str(channel_dim)}_M_{str(num_latents)}_B_{str(num_blocks)}_H_{str(num_heads)}'
    exp_name = os.path.join(f'scaling_lca_{dataset}', exp_name)

    case_dir = os.path.join(CASEDIR, exp_name)
    if os.path.exists(case_dir):
        if os.path.exists(os.path.join(case_dir, 'ckpt10', 'rel_error.json')):
            print(f"Experiment {exp_name} exists. Skipping.")
            return
        else:
            print(f"Experiment {exp_name} exists but ckpt10/rel_error.json does not exist. Removing and re-running.")
            shutil.rmtree(case_dir)

    job_queue.append({
        #
        'exp_name': exp_name,
        'dataset': dataset,
        'seed': seed,
        #
        'epochs': epochs,
        'batch_size': batch_size,
        'weight_decay': weight_decay,
        #
        'model_type': 2,
        #
        'channel_dim': channel_dim,
        'num_latents': num_latents,
        'num_blocks': num_blocks,
        'num_heads': num_heads,
        'num_layers_kv_proj': num_layers_kv_proj,
        'num_layers_mlp': num_layers_mlp,
        'num_layers_in_out_proj': 2,
    })

    return

#======================================================================#
def clean_scaling_study(dataset: str):
    output_dir = os.path.join(CASEDIR, f'scaling_lca_{dataset}')
    for case_name in [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]:
        case_dir = os.path.join(output_dir, case_name)
        if os.path.exists(os.path.join(case_dir, 'ckpt10', 'rel_error.json')):
            for ckpt in [f'ckpt{i:02d}' for i in range(10)]:
                if os.path.exists(os.path.join(case_dir, ckpt)):
                    shutil.rmtree(os.path.join(case_dir, ckpt))
            if os.path.exists(os.path.join(case_dir, 'ckpt10', 'model.pt')):
                os.remove(os.path.join(case_dir, 'ckpt10', 'model.pt'))
        else:
            shutil.rmtree(case_dir)
    return

#======================================================================#
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Latent Cross Attention model scaling study')

    parser.add_argument('--eval', type=bool, default=False, help='Evaluate scaling study results')
    parser.add_argument('--train', type=bool, default=False, help='Train scaling study')
    parser.add_argument('--clean', type=bool, default=False, help='Clean scaling study results')

    parser.add_argument('--dataset', type=str, default='elasticity', help='Dataset to use')
    parser.add_argument('--gpu-count', type=int, default=None, help='Number of GPUs to use')
    parser.add_argument('--max-jobs-per-gpu', type=int, default=2, help='Maximum number of jobs per GPU')
    parser.add_argument('--reverse-queue', type=bool, default=False, help='Reverse queue')

    args = parser.parse_args()

    if args.train:
        do_training(args.dataset, args.gpu_count, args.max_jobs_per_gpu, args.reverse_queue)
    if args.eval:
        eval_results(args.dataset)
    if args.clean:
        clean_scaling_study(args.dataset)

    if not args.train and not args.eval and not args.clean:
        print("No action specified. Please specify either --train or --eval or --clean.")

    exit()

#======================================================================#
#