#
import os
import time
import shutil
import subprocess
import json, yaml
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#======================================================================#
PROJDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUTDIR = os.path.join(PROJDIR, 'out', 'pdebench')
os.makedirs(OUTDIR, exist_ok=True)

DATADIR1 = os.path.join(OUTDIR, 'drivaerml_1m_results')
DATADIR2 = os.path.join(OUTDIR, 'drivaerml_1m_timings')

#======================================================================#
def collect_data():
    # Initialize empty dataframe
    df = pd.DataFrame()
    
    if not os.path.exists(DATADIR1):
        raise FileNotFoundError(f"Data directory {DATADIR1} does not exist.")

    # Get all subdirectories (each represents a case)
    cases = [d for d in os.listdir(DATADIR1) if os.path.isdir(os.path.join(DATADIR1, d))]

    for case in tqdm(cases, desc="Collecting data", ncols=80):
        case_path = os.path.join(DATADIR1, case)
        time_path = os.path.join(DATADIR2, case)

        if not os.path.exists(os.path.join(case_path, 'config.yaml')):
            warnings.warn(f"Skipping {case} because it does not have a config.yaml file.")
            continue

        # Initialize case data dictionary
        case_data = {}

        # Check for and load relative error data
        error_path = os.path.join(case_path, 'ckpt10', 'stats.json')
        if os.path.exists(error_path):
            with open(error_path, 'r') as f:
                stats = json.load(f)
            case_data.update({
                'train_rel_error': stats.get('train_loss'),
                'test_rel_error': stats.get('test_loss')
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

        # Load timing data
        stats_path = os.path.join(time_path, 'model_stats.json')
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            case_data.update({
                'num_params': stats.get('num_params'),
                'avg_time_per_step': stats.get('avg_time_per_step'),
                'avg_time_per_epoch': stats.get('avg_time_per_epoch'),
                'avg_memory_utilization': stats.get('avg_memory_utilization'),
            })

        # Add case data to dataframe
        df = pd.concat([df, pd.DataFrame([case_data])], ignore_index=True)

    df['head_dim'] = df['channel_dim'] // df['num_heads']
    print(f"Collected {len(df)} cases.")

    return df

def plot_results(df: pd.DataFrame):

    #---------------------------------------------------------#
    df = df.groupby(['num_latents', 'num_blocks']).mean().reset_index()

    configs = df[['num_latents',]].drop_duplicates()
    print(f"Found {len(configs)} unique configurations for num_blocks lineplot.")
    
    num_latents_list = configs['num_latents'].unique().tolist()

    #---------------------------------------------------------#
    # LINEPLOT of test error vs num_blocks
    #---------------------------------------------------------

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath}"
    })

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 7))
    fontsize = 28

    # AX1: Test relative error vs num_blocks
    ax1.set_ylabel(r'Test relative error', fontsize=fontsize)
    ax1.set_yscale('log', base=10)

    ax1.set_yticks([1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3, 9e-3, 1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 6e-2, 7e-2, 8e-2, 9e-2, 1e-1, 2e-1, 3e-1, 4e-1, 5e-1])
    ax1.set_yticklabels(['1e-3', '', '', '', '5e-3', '', '', '', '', '1e-2', '', '', '', '5e-2', '', '', '', '', '1e-1', '2e-1', '3e-1', '4e-1', '5e-1'])
    # ax1.set_yticklabels(['1e-3', '2e-3', '3e-3', '4e-3', '5e-3', '6e-3', '7e-3', '8e-3', '9e-3', '1e-2', '2e-2', '3e-2', '4e-2', '5e-2', '6e-2', '7e-2', '8e-2', '9e-2', '1e-1', '2e-1', '3e-1', '4e-1', '5e-1'])
    ax1.set_ylim(5e-3, 3e-1)

    # AX2: Time per epoch vs num_blocks
    ax2.set_ylabel(r'Time per epoch (s)', fontsize=fontsize)
    ax2.set_yscale('linear')

    # AX3: Memory utilization vs num_blocks
    ax3.set_ylabel(r'Peak Memory utilization (GB)', fontsize=fontsize)
    ax3.set_yscale('linear')
    # ax3.set_yticks([0, 20, 40, 60, 80])
    # ax3.axhline(y=80, color='black', linestyle='--', linewidth=3.0)
    ax3.set_ylim(0, 40)

    #--------------#
    for ax in [ax1, ax2, ax3]:
        ax.set_xscale('linear')
        ax.set_xlabel(r'Number of blocks ($B$)', fontsize=fontsize)
        ax.set_xticks(range(10))
        ax.set_xticklabels([int(b) for b in range(10)])
        ax.set_xlim(0, 9)
        ax.grid(True, which="both", ls="-", alpha=0.5)
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
    #--------------#

    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    markers = ['o', 's', 'D', 'v', '^', 'P', 'X', 'd', 'H', 'p']
    linestyles = ['-', '--', '-.', ':']

    linewidth = 2.5
    markersize = 10

    for i, num_latents in enumerate(num_latents_list):
        color = colors[i]
        label = r'M=%s' % num_latents

        df_ = df[df['num_latents'] == num_latents]
        
        kwargs = {
            'marker': markers[i], 'linestyle': linestyles[i],
            'color': color, 'linewidth': linewidth, 'markersize': markersize
        }

        ax1.plot(df_['num_blocks'], df_['test_rel_error'], label=label, **kwargs)
        ax2.plot(df_['num_blocks'], df_['avg_time_per_epoch'], label=label, **kwargs)
        ax3.plot(df_['num_blocks'], df_['avg_memory_utilization'], label=label, **kwargs)

    # have common legend at the bottom
    plt.tight_layout()
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=fontsize, handlelength=3.0)
    plt.subplots_adjust(bottom=0.28)
    
    #---------------------------------------------------------#
    out_path = os.path.join(PROJDIR, 'figs', f'scale_dml.pdf')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    #---------------------------------------------------------#

    return

#======================================================================#
if __name__ == '__main__':
    df = collect_data()
    plot_results(df)
    exit()

#======================================================================#
#
