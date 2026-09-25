#
import gc
import os
import json

import torch
import torch_geometric as pyg
import shutil
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mlutils

from .time_march import march_case
from .visualize import (visualize_pyv, visualize_timeseries_pyv)

__all__ = [
    'Callback',
    'FinaltimeCallback',
    'TimeseriesCallback',
]

#======================================================================#
class Callback(mlutils.Callback):
    def __init__(self, case_dir: str, mesh: bool, save_every=None, num_eval_cases=None):
        super().__init__(case_dir, save_every=save_every)
        self.mesh = mesh
        self.num_eval_cases = num_eval_cases if num_eval_cases is not None else 20

    def get_dataset_transform(self, dataset):
        if dataset is None:
            return None
        elif isinstance(dataset, torch.utils.data.Subset):
            return self.get_dataset_transform(dataset.dataset)
        elif isinstance(dataset, pyg.data.Dataset):
            return dataset.transform

    def modify_dataset_transform(self, trainer: mlutils.Trainer, val: bool):
        """
        modify transform to return mesh, original fields
        """
        for dataset in [trainer._data, trainer.data_]:
            if dataset is None:
                continue

            transform = self.get_dataset_transform(dataset)
            transform.mesh = True if val else self.mesh
            transform.orig = val
            transform.elems = val
            transform.metadata = val

        return
    
    def evaluate(self, trainer: mlutils.Trainer, ckpt_dir: str, stat_vals: dict | None = None):
        self.modify_dataset_transform(trainer, True)
        self._evaluate(trainer, ckpt_dir)
        self.modify_dataset_transform(trainer, False)
        return

    def _evaluate(self, trainer: mlutils.Trainer, ckpt_dir: str):
        if trainer.is_cuda:
            gc.collect()
            torch.cuda.empty_cache()

        return

#======================================================================#
class FinaltimeCallback(Callback):
    def evaluate(self, trainer: mlutils.Trainer, ckpt_dir: str, stat_vals: dict | None = None):
        # LPBF trains with physical Rel-L2; FinaltimeCallback itself reports MSE/R2
        # for visualization only. Mirror RelL2Callback's artifact so ablation
        # plotters can read ``rel_error.json`` without a stats.json fallback.
        if trainer.GLOBAL_RANK == 0 and stat_vals:
            train_rel = stat_vals.get("train_loss")
            test_rel = stat_vals.get("test_loss")
            if train_rel is not None and test_rel is not None:
                print(f"Relative Error (train / test): {train_rel:.8e} / {test_rel:.8e}")
                payload = {
                    "train_rel_error": float(train_rel),
                    "test_rel_error": float(test_rel),
                }
                with open(os.path.join(ckpt_dir, "rel_error.json"), "w") as f:
                    json.dump(payload, f)
                with open(os.path.join(ckpt_dir, "..", "rel_error.json"), "w") as f:
                    json.dump(payload, f)
        return super().evaluate(trainer, ckpt_dir, stat_vals)

    def _evaluate(self, trainer: mlutils.Trainer, ckpt_dir: str):

        device = trainer.device
        model  = trainer.model.module if trainer.DDP else trainer.model
        
        lossfun = torch.nn.MSELoss()

        for (dataset, transform, split) in zip(
            [trainer._data, trainer.data_],
            [self.get_dataset_transform(trainer._data), self.get_dataset_transform(trainer.data_)],
            ['train', 'test'],
        ):
            if dataset is None:
                if trainer.GLOBAL_RANK == 0:
                    print(f"No {split} dataset.")
                continue

            if self.final:
                split_dir = os.path.join(ckpt_dir, f'vis_{split}')
                os.makedirs(split_dir, exist_ok=True)
            
            stats_file = os.path.join(ckpt_dir, f'{split}_stats.txt')

            # distribute cases across ranks
            num_cases = len(dataset)
            cases_per_rank = num_cases // trainer.WORLD_SIZE 
            icase0 = trainer.GLOBAL_RANK * cases_per_rank
            icase1 = (trainer.GLOBAL_RANK + 1) * cases_per_rank if trainer.GLOBAL_RANK != trainer.WORLD_SIZE - 1 else num_cases

            max_eval_cases = self.num_eval_cases // trainer.WORLD_SIZE

            case_nums = []
            case_names = []
            l2s = []
            r2s = []

            if trainer.GLOBAL_RANK == 0:
                pbar = tqdm(total=num_cases, desc=f"Evaluating {split} dataset", ncols=80)
            
            for icase in range(icase0, icase1):
                data = dataset[icase].to(device)
                data.yh = model(data.x.unsqueeze(0)).squeeze(0)
                data.e = data.y - data.yh
                data.yp = data.yh * transform.scale.to(device)

                case_nums.append(icase)
                case_names.append(data.metadata['case_name'])
                l2s.append(lossfun(data.yh, data.y).item())
                r2s.append(mlutils.r2(data.yh, data.y))

                if self.final and (len(case_nums) < max_eval_cases):
                    base_name = os.path.basename(self.case_dir)
                    case_name = data.metadata["case_name"]
                    file_name = f'{base_name}-{split}{str(icase).zfill(4)}-{case_name}'
                    out_file = os.path.join(split_dir, file_name + '.vtu')
                    visualize_pyv(data, out_file)

                del data

                if trainer.GLOBAL_RANK == 0:
                    pbar.update(trainer.WORLD_SIZE)

            if trainer.GLOBAL_RANK == 0:
                pbar.close()

            df = pd.DataFrame({
                'case_num': case_nums,
                'case_name': case_names,
                'MSE': l2s,
                'R-Square': r2s
            })
            
            # gather dataframe across ranks
            df = vstack_dataframes_across_ranks(df, trainer)

            if trainer.GLOBAL_RANK == 0:
                print(f"Saving {split} stats to {stats_file}")
                df.to_csv(stats_file, index=False)

        if trainer.DDP:
            torch.distributed.barrier()

        r2_values = {'train': [], 'test': []}
        for split in ['train', 'test']:
            stats_file = os.path.join(ckpt_dir, f'{split}_stats.txt')
            df = pd.read_csv(stats_file)
            r2_values[split] = df['R-Square'].values

        if trainer.GLOBAL_RANK == 0:
            r2_boxplot(r2_values, filename=os.path.join(ckpt_dir, 'r2_boxplot.png'))
            r2_medians = {split: np.median(vals) for split, vals in r2_values.items()}
            with open(os.path.join(ckpt_dir, 'r2_medians.json'), 'w') as f:
                json.dump(r2_medians, f, indent=4)

        return

#======================================================================#
class TimeseriesCallback(Callback):
    def __init__(
        self, case_dir: str, mesh: bool, save_every=None, num_eval_cases=None,
        autoreg_start=1,
    ):
        super().__init__(case_dir, mesh=mesh, save_every=save_every, num_eval_cases=num_eval_cases)
        self.autoreg_start = autoreg_start

    def _evaluate(self, trainer: mlutils.Trainer, ckpt_dir: str):

        device = trainer.device
        model  = trainer.model.module if trainer.DDP else trainer.model

        for (dataset, transform, split) in zip(
            [trainer._data, trainer.data_],
            [self.get_dataset_transform(trainer._data), self.get_dataset_transform(trainer.data_)],
            ['train', 'test'],
        ):
            if dataset is None:
                if trainer.GLOBAL_RANK == 0:
                    print(f"No {split} dataset.")
                continue

            split_dir = os.path.join(ckpt_dir, f'vis_{split}')

            if trainer.GLOBAL_RANK == 0:
                os.makedirs(split_dir, exist_ok=True)

            # distribute cases across ranks
            num_cases = len(dataset.case_files)
            cases_per_rank = num_cases // trainer.WORLD_SIZE 
            icase0 = trainer.GLOBAL_RANK * cases_per_rank
            icase1 = (trainer.GLOBAL_RANK + 1) * cases_per_rank if trainer.GLOBAL_RANK != trainer.WORLD_SIZE - 1 else num_cases
            
            max_eval_cases = self.num_eval_cases // trainer.WORLD_SIZE

            case_nums = []
            case_names = []

            l2_cases = []
            r2_cases = []

            if trainer.GLOBAL_RANK == 0:
                pbar = tqdm(total=num_cases, desc=f"Evaluating {split} dataset", ncols=80)
            
            for icase in range(icase0, icase1):
                case_idx = dataset.case_range(icase)
                case_data = dataset[case_idx]
                case_name = case_data[0].metadata['case_name']

                case_nums.append(icase)
                case_names.append(case_name)
                
                eval_data, l2s, r2s = march_case(
                    model, case_data, transform,
                    autoreg=True, device=device, K=self.autoreg_start,
                )

                # case_dir = os.path.join(split_dir, f"{split}{str(icase).zfill(3)}-{ext}-{case_name}")
                # file_name = f'{os.path.basename(self.case_dir)}-{split}{str(icase).zfill(4)}-{ext}-{case_name}'
                # if self.final and len(case_nums) < self.num_eval_cases:
                #     visualize_timeseries_pyv(eval_data, case_dir, merge=True, name=file_name)

                l2_cases.append(l2s)
                r2_cases.append(r2s)

                del eval_data 
                del case_data
                
                if trainer.GLOBAL_RANK == 0:
                    pbar.update(trainer.WORLD_SIZE)
                    
            if trainer.GLOBAL_RANK == 0:
                pbar.close()

            # Convert list of stats arrays into a DataFrame where each row represents
            # a time step and each column represents a case
            df_l2 = pd.DataFrame(l2_cases).transpose()
            df_r2 = pd.DataFrame(r2_cases).transpose()

            # Assign case numbers as column names
            df_l2.columns = case_nums
            df_r2.columns = case_nums

            # Assign step numbers as index
            df_l2.index.name = 'Step'
            df_r2.index.name = 'Step'

            # create dataframe for each autoreg
            df_l2 = hstack_dataframes_across_ranks(df_l2, trainer)
            df_r2 = hstack_dataframes_across_ranks(df_r2, trainer)
            
            if trainer.GLOBAL_RANK == 0:
                print(f"Saving {split} statistics to {ckpt_dir}")
                df_l2.to_csv(os.path.join(ckpt_dir, f'l2_stats_{split}.txt'), index=False)
                df_r2.to_csv(os.path.join(ckpt_dir, f'r2_stats_{split}.txt'), index=False)
            
        if trainer.DDP:
            torch.distributed.barrier()

        # make plots
        for split in ['train', 'test']:
            df_l2 = pd.read_csv(os.path.join(ckpt_dir, f'l2_stats_{split}.txt'))
            df_r2 = pd.read_csv(os.path.join(ckpt_dir, f'r2_stats_{split}.txt'))

            if trainer.GLOBAL_RANK == 0:
                print(f"Saving L2/R2 plots to {ckpt_dir}/r2_plot_{split}.png")
                timeseries_statistics_plot(df_r2, 'r2', 'median', filename=os.path.join(ckpt_dir, f'r2_plot_{split}.png'))
                timeseries_statistics_plot(df_l2, 'l2', 'median', filename=os.path.join(ckpt_dir, f'l2_plot_{split}.png'))

        return

#======================================================================#
# Plotting functions
#======================================================================#
def timeseries_statistics_plot(df, metric, mode, filename=None, dpi=175):

    plt.figure(figsize=(8, 4), dpi=dpi)
    
    if metric == 'r2':
        plt.ylim(0., 1.)
        plt.ylabel('R-Squared')
    elif metric == 'l2':
        plt.ylim(0., 1e-1) # 10% error
        plt.ylabel('RMSE (normalized)')
        df = df.apply(lambda x: np.sqrt(x))
        plt.title('Mean RMSE (normalized): {:.2e}'.format(df.mean().mean()))

    if mode == 'median':
        medians = df.median(axis=1)
        q1 = df.quantile(0.25, axis=1)
        q3 = df.quantile(0.75, axis=1)
        tstep = np.arange(len(medians))

        plt.plot(tstep, medians, color='k', label='Median')
        plt.fill_between(
            tstep, q1, q3,
            color='k', alpha=0.2,
            label='Middle 50%',
        )
    
    elif mode == 'mean':
        means = df.mean(axis=1)
        stds = df.std(axis=1)
        tstep = np.arange(len(means))

        plt.plot(tstep, means, color='k', label='Mean')
        plt.fill_between(
            tstep, means - stds, means + stds,
            color='k', alpha=0.2,
            label='1 Std Dev',
        )

    else:
        raise ValueError(f"Invalid mode: {mode}")

    plt.xlabel('Time Step')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

    return

def r2_boxplot(
    vals,
    titles=dict(train="Training", test="Testing", od="Out-of-Dist."),
    lims=[-1, 1],
    filename=None,
    dpi=175,
):
    n = len(vals)
    plt.figure(figsize=(2*n, 3.4), dpi=dpi)

    vals_list = []
    ticklocs = []
    ticklabels = []

    for i, key in enumerate(vals):
        vals_list.append(vals[key])
        ticklocs.append(i)
        ticklabels.append(f"{titles[key]}, N={len(vals[key])}")

    plt.boxplot(vals_list, positions=ticklocs)
    plt.xticks(ticklocs, ticklabels)
    plt.ylabel('R-Squared')
    plt.ylim(lims)
    plt.xlim(ticklocs[0]-0.5, ticklocs[-1]+0.5)
    plt.plot([ticklocs[0]-0.5, ticklocs[-1]+0.5], [0, 0],'k-', linewidth=0.5, zorder=-1)

    plt.savefig(filename, bbox_inches = "tight")

    return

#======================================================================#
# Combine DataFrames across distributed processes
#======================================================================#
def hstack_dataframes_across_ranks(df: pd.DataFrame, trainer: mlutils.Trainer) -> pd.DataFrame:
    """
    Combine DataFrames across distributed processes horizontally by adding columns.
    
    Args:
        df: Local DataFrame to combine
        trainer: Trainer object containing distributed training info
        
    Returns:
        Combined DataFrame with columns from all processes
    """
    if not trainer.DDP:
        return df
        
    local_data = df.to_dict('list')  # Get columns as lists
    
    # Gather data from all processes
    gathered_data = [None] * trainer.WORLD_SIZE
    torch.distributed.all_gather_object(gathered_data, local_data)
    
    # Find max length across all columns
    max_len = max(len(lst) for rank_data in gathered_data for lst in rank_data.values())
    
    # Combine columns from all processes with padding
    combined_data = {}
    for rank_data in gathered_data:
        for col, values in rank_data.items():
            # Pad shorter columns with None
            if len(values) < max_len:
                values = values + [None] * (max_len - len(values))
            combined_data[col] = values
            
    return pd.DataFrame(combined_data)

def vstack_dataframes_across_ranks(df: pd.DataFrame, trainer: mlutils.Trainer) -> pd.DataFrame:
    """
    Combine DataFrames across distributed processes vertically by adding rows.
    
    Args:
        df: Local DataFrame to combine
        trainer: Trainer object containing distributed training info
        
    Returns:
        Combined DataFrame from all processes
    """
    if not trainer.DDP:
        return df
        
    local_data = df.to_dict('records')
    
    # Gather data from all processes
    gathered_data = [None] * trainer.WORLD_SIZE
    torch.distributed.all_gather_object(gathered_data, local_data)
    
    # Flatten the list of lists and create final DataFrame
    all_data = [item for sublist in gathered_data for item in sublist]
    return pd.DataFrame(all_data)

#======================================================================#
#
