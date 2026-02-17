"""
proposal fiures
"""

import os
import sys
import argparse

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import triton.testing
from contextlib import contextmanager

#======================================================================#
PROJDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

OUT_PNG = os.path.join(PROJDIR, 'figs', 'time_memory_proposal.png')
OUT_PDF = os.path.join(PROJDIR, 'figs', 'time_memory_proposal.pdf')
OUT_CSV = os.path.join(PROJDIR, 'out', 'pdebench', 'time_memory_bwd_fp16.csv')

SEQ_LENGTHS = [
            1_000,
           10_000,
          100_000,
          200_000,
          300_000,
          400_000,
          500_000,
          600_000,
          700_000,
          800_000,
          900_000,
        1_000_000,
    ]

NUM_LATENTS = [128, 512, 2048]
NUM_STATES = [1, 2, 4] # (Multilinear)

#======================================================================#
def plot_analysis():
    df = pd.read_csv(OUT_CSV)

    # Set matplotlib to use LaTeX fonts if available, otherwise use default
    try:
        import subprocess
        subprocess.run(['latex', '--version'], capture_output=True, check=True)
        # LaTeX is available, use it
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "text.latex.preamble": r"\usepackage{amsmath}"
        })
        print("Using LaTeX for plot rendering")
    except (subprocess.CalledProcessError, FileNotFoundError):
        # LaTeX not available, use default matplotlib fonts
        plt.rcParams.update({
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"]
        })
        print("LaTeX not available, using default matplotlib fonts")

    fontsize = 20

    fig, (ax1) = plt.subplots(1, 1, figsize=(8, 6))
    for ax in [ax1]:
        ax.set_xscale('linear')
        ax.grid(True, which="both", ls="-", alpha=0.5)
        ax.set_xlabel(r'Sequence Length', fontsize=fontsize)

    ax1.set_yscale('log', base=10)

    ax1.set_ylabel(r'Time (s)', fontsize=fontsize)

    # Increase tick label size
    for ax in [ax1]:
        ax.tick_params(axis='both', which='major', labelsize=fontsize)

    # Define markers for different num_latents
    latent_map = {
        f"{NUM_LATENTS[0]}": 's',
        f"{NUM_LATENTS[1]}": 'v',
        f"{NUM_LATENTS[2]}": 'D',
    }

    # Define markers for different num_states (Multilinear)
    states_map = {
        '1': 'o',
        '2': 'v',
        '4': 'D',
    }

    # Set custom x-ticks for sequence lengths
    x_ticks = [1000, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000]
    x_tick_labels = ['1k', '', '200k', '', '400k', '', '600k', '', '800k', '', '1m']

    for ax in [ax1]:
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_tick_labels)

    for model_name in df['model_name'].unique():
        model_data = df[df['model_name'] == model_name]
        model_data = model_data.sort_values(by='N')

        if 'Softmax' in model_name:
            marker = 's'
            color = 'black'
            linestyle = '-'
            label = r'Softmax Attention'
        elif 'FLARE' in model_name:
            latent_size = model_name.split('=')[1].strip(')')
            marker = latent_map[latent_size]
            linestyle = '-'
            color = 'red'
            label = r'FLARE (rank $%s$)' % latent_size

            # continue

        elif 'Triple' in model_name:
            continue
        elif 'Phys' in model_name:
            continue
        elif 'Multilinear' in model_name:
            continue
        else:
            raise ValueError(f"model_name {model_name} not found")

        marker_size = 12

        ax1.plot(model_data['N'], model_data['time'] / 1e3, label=label, marker=marker, 
            linestyle=linestyle, linewidth=2.5, color=color, markersize=marker_size)

    # Add legend to bottom of the figure with 3 columns
    handles, labels = ax1.get_legend_handles_labels()

    # Organize legend: FLARE variants in row 1, Multilinear+Triple+Softmax in row 2
    softmax_items = [(h, l) for h, l in zip(handles, labels) if 'Softmax' in l]
    flare_items = [(h, l) for h, l in zip(handles, labels) if 'FLARE' in l]
    triple_items = [(h, l) for h, l in zip(handles, labels) if 'Triple' in l]
    multilinear_items = [(h, l) for h, l in zip(handles, labels) if 'Multilinear' in l]

    # Create ordered lists for multi-row layout
    ordered_handles = []
    ordered_labels = []

    if len(softmax_items) > 0:
        ordered_handles.append(softmax_items[0][0])
        ordered_labels.append(softmax_items[0][1])

    # Row 1: FLARE variants
    max_flare = len(flare_items)
    for i in range(max_flare):
        ordered_handles.append(flare_items[i][0])
        ordered_labels.append(flare_items[i][1])

    # Row 2: Multilinear variants
    for i in range(len(multilinear_items)):
        ordered_handles.append(multilinear_items[i][0])
        ordered_labels.append(multilinear_items[i][1])

    # Row 3: Triple + Softmax
    if len(triple_items) > 0:
        ordered_handles.append(triple_items[0][0])
        ordered_labels.append(triple_items[0][1])
    # Place legend at bottom right inside the axes
    legend = ax1.legend(ordered_handles, ordered_labels, loc='lower right', ncol=2, 
              frameon=True, fancybox=False, shadow=False, fontsize=fontsize, 
              columnspacing=0.5, handletextpad=0.2,
              handlelength=1.5, markerscale=1.0)

    # Add title with larger font
    ax1.set_title(r'Execution Time (Forward + Backward)', fontsize=fontsize)

    # Adjust layout
    plt.tight_layout()

    # Save the figure with both plots
    fig.savefig(OUT_PDF, dpi=300, bbox_inches='tight')
    fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
    plt.close()

    return

#======================================================================#
if __name__ == '__main__':
    plot_analysis()

#======================================================================#
#