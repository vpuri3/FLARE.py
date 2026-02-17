#
import os
import shutil
import torch
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt

import am

__all__ = [
    'save_dataset_statistics',
    'make_statistics_plot',
    'compute_filtered_dataset_statistics',
    'compute_dataset_statistics',
    'make_include_list',
    'make_dataset',
]

#======================================================================#
def save_dataset_statistics(df, case_dir):
    """
    Save and plot dataset statistics
    """
    # Save dataset statistics
    stats_csv = os.path.join(case_dir, 'dataset_statistics.csv')
    stats_txt = os.path.join(case_dir, 'dataset_statistics.txt')
    stats_png = os.path.join(case_dir, 'dataset_statistics.png')

    # save stats.csv
    df.to_csv(stats_csv, index=False)

    # save stats.txt
    with open(stats_txt, 'w') as f:
        f.write(str(df.describe()))

    # print stats
    print(df.describe())

    # Create probability density plots
    numerical_cols = df.select_dtypes(include=['number']).columns

    ###
    # Create png plots
    ###

    plt.figure(figsize=(20, 15))
    for i, col in enumerate(numerical_cols, 1):
        plt.subplot(3, 3, i)
        sns.kdeplot(df[col], fill=True, warn_singular=False)
        plt.title(f'PDF of {col}', pad=10, fontsize=18)
        plt.xlabel(col, labelpad=5)
        plt.ylabel("Density")
        plt.yticks([])
        plt.tight_layout(pad=3.0)

    plt.tight_layout()
    plot_file = os.path.join(case_dir, stats_png)
    plt.savefig(plot_file, bbox_inches='tight')
    plt.close()
    
    return

#======================================================================#
def make_statistics_plot(PROJDIR: str, filtered: bool = True):
    """
    Make a plot of the dataset statistics
    """

    if filtered:
        case_dir = os.path.join(PROJDIR, 'am_dataset_stats', 'filtered')
    else:
        case_dir = os.path.join(PROJDIR, 'am_dataset_stats')

    df = pd.read_csv(os.path.join(case_dir, 'dataset_statistics.csv'))

    # Set matplotlib to use LaTeX fonts
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath}"
    })

    fontsize = 24
    fig, axs = plt.subplots(1, 4, figsize=(16, 5))

    labels = [r'\# Vertices', r'\# Edges', r'Max. Aspect Ratio', r'Max $Z$ Displacement (mm)']
    fields = ['num_vertices', 'num_edges', 'max_aspect_ratio', 'max_disp']

    for (i, ax) in enumerate(axs.flatten()):
        ax.set_xlabel(labels[i], fontsize=fontsize)
        ax.set_yticklabels([])
        ax.tick_params(axis='both', which='major', labelsize=fontsize)

    for (i, ax) in enumerate(axs.flatten()):
        ax.set_xscale('linear')
        ax.grid(True, which="both", ls="-", alpha=0.5)
        plt.subplot(1, 4, i+1)
        sns.kdeplot(df[fields[i]], fill=True, warn_singular=False)

    axs[0].set_ylabel(r'Density', fontsize=fontsize)
    axs[1].set_ylabel('')
    axs[2].set_ylabel('')
    axs[3].set_ylabel('')

    plt.tight_layout()
    plot_file = os.path.join(case_dir, 'dataset_statistics.pdf')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()

    return
    
#======================================================================#
def compute_filtered_dataset_statistics(PROJDIR):
    """
    Compute statistics on the filtered dataset (excluding problematic cases)
    """
    # Load full statistics
    stats_csv = os.path.join(PROJDIR, 'am_dataset_stats', 'dataset_statistics.csv')
    df = pd.read_csv(stats_csv)

    # Load include list
    include_list_file = os.path.join(PROJDIR, 'am_dataset_stats', 'include_list.txt')
    with open(include_list_file, 'r') as f:
        include_list = [line.strip() for line in f.readlines()]

    # Filter dataset
    filtered_df = df[df['case_name'].isin(include_list)]
    
    print(f"Filtered dataset has {len(filtered_df)} cases. Saving statistics...")

    # Save filtered statistics
    filtered_case_dir = os.path.join(PROJDIR, 'am_dataset_stats', 'filtered')
    os.makedirs(filtered_case_dir, exist_ok=True)
    save_dataset_statistics(filtered_df, filtered_case_dir)

    return filtered_df

#======================================================================#
def compute_dataset_statistics(PROJDIR, DATADIR_FINALTIME, SUBDIRS, force_reload=False):
    """
    Compute statistics on the full dataset
    """
    # Create output directory based on mode
    case_dir = os.path.join(PROJDIR, 'am_dataset_stats')
    os.makedirs(case_dir, exist_ok=True)

    # Create directory for aspect ratios cache
    aspect_ratios_dir = os.path.join(case_dir, 'mesh_aspect_ratios')
    os.makedirs(aspect_ratios_dir, exist_ok=True)

    stats = {
        # mesh
        'num_vertices': [],
        'num_edges': [],
        'avg_aspect_ratio': [],
        'max_aspect_ratio': [],

        # fields
        'max_z': [],
        'max_disp': [],
        'max_vmstr': [],

        # metadata
        'case_name': [],
    }
    dataset = am.FinaltimeDataset(DATADIR_FINALTIME, subdirs=SUBDIRS, force_reload=force_reload)

    # subdirectory for cached aspect ratios
    aspect_ratio_subdir = os.path.join(aspect_ratios_dir, DATADIR_FINALTIME)
    os.makedirs(aspect_ratio_subdir, exist_ok=True)

    if not os.path.exists(aspect_ratio_subdir):
        print(f"Computing aspect ratios for {DATADIR_FINALTIME}")

    for case in tqdm(dataset, desc="Computing dataset statistics", ncols=100):

        # Extract basic metadata
        stats['case_name'].append(case.metadata['case_name'])

        # Mesh statistics
        stats['num_vertices'].append(case.pos.size(0))
        stats['num_edges'].append(case.edge_index.size(1))

        # Cache aspect ratios
        aspect_ratios_file = os.path.join(aspect_ratio_subdir, case.metadata['case_name'] + '.csv')

        if os.path.exists(aspect_ratios_file):
            aspect_ratios = np.loadtxt(aspect_ratios_file)
        else:
            pos, elems = case.pos.numpy(), case.elems.numpy()
            aspect_ratios = am.compute_aspect_ratios(pos, elems)
            np.savetxt(aspect_ratios_file, aspect_ratios)
            del pos, elems

        stats['avg_aspect_ratio'].append(np.mean(aspect_ratios))
        stats['max_aspect_ratio'].append(np.max(aspect_ratios))

        # fields
        stats['max_z'].append(torch.max(case.pos[:,2]).item())
        stats['max_disp'].append(torch.max(case.disp[:,2]).item())
        stats['max_vmstr'].append(torch.max(case.vmstr).item())

        del case

    # Create DataFrame
    df = pd.DataFrame(stats)

    # derived statistics
    df['edges_per_vert'] = df['num_edges'] / df['num_vertices']

    # Save statistics
    save_dataset_statistics(df, case_dir)

    return df

#======================================================================#
def make_include_list(PROJDIR):
    """
    make a list of case_names to include based on statistics
    """

    # load stats.csv
    stats_csv = os.path.join(PROJDIR, 'am_dataset_stats', 'dataset_statistics.csv')
    df = pd.read_csv(stats_csv)

    exclude_list = []
    exclude_list += df[df['num_vertices'] > 5e4]['case_name'].tolist()    # too many verts
    exclude_list += df[df['num_edges'] > 2.5e5]['case_name'].tolist()     # too many edges
    exclude_list += df[df['avg_aspect_ratio'] > 3]['case_name'].tolist() # thin features
    exclude_list += df[df['max_aspect_ratio'] > 3]['case_name'].tolist() # thin features
    # exclude_list += df[df['max_z'] < 30]['case_name'].tolist()            # too short
    exclude_list += df[df['max_disp'] > 1]['case_name'].tolist()          # bad displacement
    exclude_list += df[df['max_vmstr'] > 2000]['case_name'].tolist()      # bad stress
    # exclude_list += df[df['edges_per_vert'] < 5]['case_name'].tolist()    # thin features

    include_list = [c for c in df['case_name'].tolist() if c not in exclude_list]

    # save include_list.txt
    include_list_file = os.path.join(PROJDIR, 'am_dataset_stats', 'include_list.txt')

    print(f"Saving include list to {include_list_file} with {len(include_list)} / {len(df)} cases.")

    with open(include_list_file, 'w') as f:
        for case_name in include_list:
            f.write(f"{case_name}\n")

    return include_list

#======================================================================#
def make_dataset(PROJDIR: str, DATADIR: str, seed: int = 0):
    """
    Convert PyG Data objects to NPZ format for HuggingFace compatibility
    
    Args:
        PROJDIR: Project directory path
        DATADIR: Data directory path  
        seed: Random seed for train/test split
    
    Examples:
        make_dataset(PROJDIR, DATADIR, seed=0)
    """

    # open include_list
    include_list_file = os.path.join(PROJDIR, 'am_dataset_stats', 'include_list.txt')
    with open(include_list_file, 'r') as f:
        include_list = [line.strip() for line in f.readlines()]

    num_include_list = len(include_list)

    DATADIR_PROCESSED = os.path.join(DATADIR, 'processed')
    case_names = [case_name for case_name in os.listdir(DATADIR_PROCESSED) if case_name.endswith('.pt')]

    case_files = []
    for case_name in case_names:
        for i, include_name in enumerate(include_list):
            if include_name == case_name[10:-3]:
                case_files.append(os.path.join(DATADIR_PROCESSED, case_name))
                del include_list[i]
                break

    # check for duplicates
    case_files = list(set(case_files))
    print(f"Found {len(case_files)} cases to include")
    assert len(case_files) == num_include_list, f"Number of cases to include ({len(case_files)}) does not match number of include_list ({num_include_list})"
    
    # split into train and test
    num_cases = len(case_files)
    np.random.seed(seed)  # Set seed for reproducibility
    indices = np.random.permutation(num_cases)
    train_indices, test_indices = indices[:1100], indices[1100:]

    train_files = [case_files[i] for i in train_indices]
    test_files = [case_files[i] for i in test_indices]

    # instantiate SAVEDIR
    SAVEDIR = os.path.join(PROJDIR, 'LPBF')
    if os.path.exists(SAVEDIR):
       shutil.rmtree(SAVEDIR)
    os.makedirs(SAVEDIR, exist_ok=False)
    os.makedirs(os.path.join(SAVEDIR, 'train'), exist_ok=False)
    os.makedirs(os.path.join(SAVEDIR, 'test'), exist_ok=False)

    def save_npz_data(data_dict, output_file):
        """Save data in NPZ format"""
        # Convert tensors to numpy arrays for NPZ
        npz_data = {}
        metadata = {}
        for key, value in data_dict.items():
            if key == 'edge_dxyz':
                continue
            if isinstance(value, torch.Tensor):
                npz_data[key] = value.numpy()
            elif isinstance(value, np.ndarray):
                npz_data[key] = value
            else:
                # Store non-array data as metadata in a separate field
                metadata[key] = value

        # Add metadata as a JSON string if present
        if metadata:
            npz_data['_metadata'] = np.array([json.dumps(metadata)], dtype=object)

        np.savez_compressed(output_file, **npz_data)

    # Process train files
    print("Converting training files to NPZ format...")
    for case_file in tqdm(train_files, desc="Processing train files"):
        pyg_data = torch.load(case_file, weights_only=False)
        data_dict = pyg_data.to_dict()

        case_name = case_file.split('/')[-1].replace('.pt', '.npz')
        output_file = os.path.join(SAVEDIR, 'train', case_name)
        save_npz_data(data_dict, output_file)

    # Process test files
    print("Converting test files to NPZ format...")
    for case_file in tqdm(test_files, desc="Processing test files"):
        pyg_data = torch.load(case_file, weights_only=False)
        data_dict = pyg_data.to_dict()

        case_name = case_file.split('/')[-1].replace('.pt', '.npz')
        output_file = os.path.join(SAVEDIR, 'test', case_name)
        save_npz_data(data_dict, output_file)

    print(f"HuggingFace compatible dataset saved to {SAVEDIR} in NPZ format")
    return

#======================================================================#
#