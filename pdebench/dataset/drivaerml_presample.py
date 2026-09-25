#
import gc
import os
import numpy as np
import json
import re
import multiprocessing as mp
from tqdm import tqdm

def process_single_run(args):
    """
    Process a single run for presampling data.
    
    Args:
        args: Tuple containing (run, data_dir, split_dir, num_points)
    """
    run, data_dir, split_dir, num_points = args
    
    # Set seed for reproducibility based on run number
    np.random.seed(run)
    
    for f in [f for f in os.listdir(data_dir) if f.endswith('.npy')]:
        match = re.search(r'run_(\d+)', f)
        if match and int(match.group(1)) == run:
            npy_file = os.path.join(data_dir, f)

            # Load the original data
            data = np.load(npy_file, allow_pickle=True).item()
            coordinates = data['surface_mesh_centers']
            field = data['surface_fields']

            # Sample points with fixed seed for reproducibility
            sample_indices = np.random.choice(coordinates.shape[0], num_points, replace=False)
            sampled_coordinates = coordinates[sample_indices, :]
            sampled_field = field[sample_indices, :]

            # Save individual presampled file for this run
            run_file_path = os.path.join(split_dir, f'run_{run}.npz')
            np.savez(run_file_path, surface_mesh_centers=sampled_coordinates, surface_fields=sampled_field)
            break
    
    return run

def create_presampled_data(
    num_points: int,
    splits: dict,
    data_dir: str,
    save_dir: str,
    num_workers: int = None,
):
    """
    Create presampled training and test data with fixed random sampling.
    Saves individual files for each run to enable on-demand loading.

    Args:
        num_points: Number of points to sample from each run
        splits: Dictionary containing train/test splits
        save_dir: Base path for saving presampled data (directory will be created)
        num_workers: Number of worker processes for multiprocessing (None for auto)
    """

    # Set number of workers
    num_workers = mp.cpu_count() // 4 if num_workers is None else num_workers

    # Create directory structure for presampled data
    os.makedirs(save_dir, exist_ok=True)

    for split in ['train', 'test']:
        runs = splits[split]
        
        # Create subdirectory for this split
        split_dir = os.path.join(save_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        # Prepare arguments for multiprocessing
        process_args = [(run, data_dir, split_dir, num_points) for run in runs]

        # Use multiprocessing to process runs in parallel
        mp.set_start_method('spawn', force=True)
        with mp.Pool(num_workers) as pool:
            list(tqdm(
                pool.imap_unordered(process_single_run, process_args), 
                total=len(runs),
                desc=f'Processing {split} split',
                ncols=80,
            ))
            
        gc.collect()

    return save_dir

#======================================================================#
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

data_dir = '/mnt/hdd1/aajoglek/Data/drivaerml_processed_data_surface'
splits_file = '/mnt/hdd1/aajoglek/Data/drivaerml_data/train_test_splits.json'
save_dir_base = '/mnt/hdd1/vedantpu/data/DrivAerML'

def main():
    splits = json.load(open(splits_file))
    os.makedirs(save_dir_base, exist_ok=True)

    for (num_points_name, num_points) in num_points_dict.items():
        save_dir = os.path.join(save_dir_base, f'drivaerml_surface_presampled_{num_points_name}')

        print("=" * 60)
        print(f"Saving num_points: {num_points} to {save_dir}")
        print("=" * 60)

        create_presampled_data(num_points, splits, data_dir, save_dir)

        print("=" * 60)
        print(f"Presampled data created successfully for {num_points} points")

    return

#======================================================================#
if __name__ == "__main__":
    main()

#======================================================================#
#