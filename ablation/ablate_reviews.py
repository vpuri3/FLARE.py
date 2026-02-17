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
from matplotlib import patheffects

import numpy as np
import pandas as pd

#======================================================================#
PROJDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CASEDIR = os.path.join(PROJDIR, 'out', 'pdebench')
os.makedirs(CASEDIR, exist_ok=True)

#======================================================================#
def collect_data(task: str):
    assert task in ['sl', 'lb'], "Task must be either 'sl' (shared latents) or 'lb' (latent blocks)."

    data_dir = os.path.join(CASEDIR, f'{task}')
    print(f"Collecting data from {data_dir}...")

    # Initialize empty dataframe
    df = pd.DataFrame()

    # Check if case directory exists
    if os.path.exists(data_dir):
        # Get all subdirectories (each represents a case)
        cases = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

        for case in cases:
            # print(f"Collecting data from {case}...")
            case_path = os.path.join(data_dir, case)

            if not os.path.exists(os.path.join(case_path, 'config.yaml')):
                print(f"Skipping {case} because it does not have a config.yaml file.")
                continue
            if not os.path.exists(os.path.join(case_path, 'model_stats.json')):
                print(f"Skipping {case} because it does not have a model_stats.json file.")
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
                    'num_heads': config.get('num_heads'),
                    'num_latents': config.get('num_latents'),
                    'num_blocks': config.get('num_blocks'),
                    'shared_att': config.get('shared_att'),
                    'num_passes': config.get('num_passes'),
                })

            # Load num_params
            model_stats_path = os.path.join(case_path, 'model_stats.json')
            if os.path.exists(model_stats_path):
                with open(model_stats_path, 'r') as f:
                    model_stats = json.load(f)
                case_data.update({
                    'num_params': model_stats.get('num_params'),
                    'avg_time_per_epoch': model_stats.get('avg_time_per_epoch'),
                })

            # Add case data to dataframe
            df = pd.concat([df, pd.DataFrame([case_data])], ignore_index=True)

            df['head_dim'] = df['channel_dim'] // df['num_heads']

        print(f"Collected {len(df)} cases for task {task}.")

    return df

#======================================================================#
def plot_sl(df: pd.DataFrame):

    #---------------------------------------------------------#
    df = df.groupby(['num_latents', 'num_blocks', 'shared_att']).mean().reset_index()

    configs = df[['num_latents', 'shared_att']].drop_duplicates()
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

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    fontsize = 28

    ax.set_ylabel(r'Test relative error', fontsize=fontsize)

    ax.set_xscale('linear')
    ax.set_yscale('log', base=10)
    ax.grid(True, which="both", ls="-", alpha=0.5)

    #--------------#
    ax.set_yticks([1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3, 9e-3, 1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 6e-2])
    ax.set_yticklabels(['', '2e-3', '', '4e-3', '', '6e-3', '', '8e-3', '', '1e-2', '2e-2', '3e-2', '4e-2', '5e-2', '6e-2'])
    #--------------#
    ax.set_ylim(3e-3, 3e-2)
    ax.set_xlabel(r'Number of blocks ($B$)', fontsize=fontsize)
    blocks = np.linspace(0,20, 11)
    ax.set_xticks(blocks)
    ax.set_xticklabels([int(b) for b in blocks])
    ax.set_xlim(0, 10)
    #--------------#

    ax.tick_params(axis='both', which='major', labelsize=fontsize)

    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    markers = ['o', 's', 'D', 'v', '^', 'P', 'X', 'd', 'H', 'p']

    linewidth = 2.5
    markersize = 10

    for i, num_latents in enumerate(num_latents_list):
        color = colors[i]
        
        # Plot both shared_att=False and shared_att=True for this num_latents
        for shared_att in [False, True]:
            df_ = df[(df['num_latents'] == num_latents) & (df['shared_att'] == shared_att)]
            
            if len(df_) == 0:
                continue

            # Use solid line for shared_att=False, dashed for shared_att=True
            linestyle = '-' if not shared_att else '--'
            
            # Create label
            sl_text = 'unshared' if not shared_att else 'shared'
            label = r'M=%s (%s)' % (num_latents, sl_text)

            kwargs = {
                'marker': markers[i], 'linestyle': linestyle,
                'color': color, 'linewidth': linewidth, 'markersize': markersize
            }

            ax.plot(df_['num_blocks'], df_['test_rel_error'], label=label, **kwargs)

    #---------------------------------------------------------#
    ax.legend(loc='upper right', ncol=2, fontsize=fontsize-12, handlelength=2.0)

    plt.tight_layout()

    # Save the figure with both plots
    fig.savefig(os.path.join(PROJDIR, 'figs', f'abl_sl.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(PROJDIR, 'figs', f'abl_sl.png'), dpi=300, bbox_inches='tight')
    plt.close()

    #---------------------------------------------------------#
    return

#======================================================================#
def plot_lb(df: pd.DataFrame):
    #---------------------------------------------------------#
    # Group by num_blocks and num_passes, taking mean across other dimensions
    df = df.groupby(['num_blocks', 'num_passes']).agg({
        'test_rel_error': 'mean',
        'num_params': 'mean',
        'avg_time_per_epoch': 'mean'
    }).reset_index()

    # Define the axes values
    num_blocks_list = [1, 2, 4, 8]
    num_passes_list = [0, 1, 2, 4, 8]

    # Create pivot tables for the heatmap data
    heatmap_data = pd.DataFrame(index=num_passes_list, columns=num_blocks_list)
    params_data = pd.DataFrame(index=num_passes_list, columns=num_blocks_list)
    time_data = pd.DataFrame(index=num_passes_list, columns=num_blocks_list)

    # Fill the pivot tables
    for num_blocks in num_blocks_list:
        for num_passes in num_passes_list:
            row = df[(df['num_blocks'] == num_blocks) & (df['num_passes'] == num_passes)]
            if len(row) > 0:
                heatmap_data.loc[num_passes, num_blocks] = row['test_rel_error'].values[0]
                params_data.loc[num_passes, num_blocks] = row['num_params'].values[0]
                time_data.loc[num_passes, num_blocks] = row['avg_time_per_epoch'].values[0]
            else:
                heatmap_data.loc[num_passes, num_blocks] = np.nan
                params_data.loc[num_passes, num_blocks] = np.nan
                time_data.loc[num_passes, num_blocks] = np.nan

    # Convert to numpy arrays for plotting
    heatmap_array = heatmap_data.values.astype(float)
    params_array = params_data.values.astype(float)
    time_array = time_data.values.astype(float)

    #---------------------------------------------------------#
    # Create heatmap plot
    #---------------------------------------------------------

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath}"
    })

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    fontsize = 25
    annotation_fontsize = 20

    # Create heatmap using imshow
    # Use RdBu_r colormap: blue for small errors, red for large errors
    im = ax.imshow(heatmap_array, cmap='coolwarm', aspect='auto', 
                   norm=LogNorm(vmin=np.nanmin(heatmap_array[~np.isnan(heatmap_array)]), 
                               vmax=np.nanmax(heatmap_array[~np.isnan(heatmap_array)])))

    # Set ticks and labels
    ax.set_xticks(np.arange(len(num_blocks_list)))
    ax.set_yticks(np.arange(len(num_passes_list)))
    ax.set_xticklabels([str(b) for b in num_blocks_list], fontsize=fontsize)
    ax.set_yticklabels([str(lb) for lb in num_passes_list], fontsize=fontsize)

    ax.set_xlabel(r'Number of blocks ($B$)', fontsize=fontsize)
    ax.set_ylabel(r'Number of latent blocks ($L_B$)', fontsize=fontsize)

    # Compute vmin and vmax for LogNorm (used in imshow)
    vmin = np.nanmin(heatmap_array[~np.isnan(heatmap_array)])
    vmax = np.nanmax(heatmap_array[~np.isnan(heatmap_array)])

    # Add annotations to each cell
    for i in range(len(num_passes_list)):
        for j in range(len(num_blocks_list)):
            if not np.isnan(heatmap_array[i, j]):
                # Format values
                rel_error = heatmap_array[i, j]
                num_params = params_array[i, j]
                avg_time = time_array[i, j]
                
                # Format relative error
                rel_error_str = f'{rel_error:.2e}'
                
                # Format parameters (in millions if > 1M, otherwise thousands)
                if num_params >= 1e6:
                    params_str = f'{num_params/1e6:.2f}m'
                elif num_params >= 1e3:
                    params_str = f'{num_params/1e3:.1f}k'
                else:
                    params_str = f'{num_params:.0f}'
                
                # Format time (in seconds)
                if avg_time < 60:
                    time_str = f'{avg_time:.1f}s'
                else:
                    time_str = f'{avg_time/60:.1f}m'
                
                # Create annotation text
                text = f'{rel_error_str}\n{params_str}\n{time_str}'
                
                # Use black text everywhere with heavy weight and stroke outline for extra thickness
                ax.text(j, i, text, ha='center', va='center', 
                       fontsize=annotation_fontsize, color='black', weight='black',
                       path_effects=[patheffects.withStroke(linewidth=0.3, foreground='black')])

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(r'Test relative error', fontsize=fontsize, rotation=270, labelpad=25)
    
    # Set specific ticks and labels
    tick_values = [4e-3, 6e-3, 8e-3, 1e-2, 2e-2]
    tick_labels = ['4e-3', '6e-3', '8e-3', '1e-2', '2e-2']
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels(tick_labels)
    cbar.ax.tick_params(labelsize=fontsize)

    plt.tight_layout()

    # Save the figure
    fig.savefig(os.path.join(PROJDIR, 'figs', f'abl_lb.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(PROJDIR, 'figs', f'abl_lb.png'), dpi=300, bbox_inches='tight')
    plt.close()

    #---------------------------------------------------------#
    return

#======================================================================#
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Shared latents and latent blocks ablation study')

    parser.add_argument('--task', type=str, default='sl', help='Task to perform')

    args = parser.parse_args()

    df = collect_data(task=args.task)

    if args.task == 'sl':
        plot_sl(df=df)
    elif args.task == 'lb':
        plot_lb(df=df)

    exit()

#======================================================================#
#
