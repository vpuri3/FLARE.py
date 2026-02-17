#
import os
import torch
from tqdm import tqdm
import pandas as pd

import mlutils
from pdebench.rollout import rollout
import am
from am.callbacks import timeseries_statistics_plot
from am.callbacks import hstack_dataframes_across_ranks, vstack_dataframes_across_ranks

__all__ = [
    'TimeseriesCallback',
]

#======================================================================#
class TimeseriesCallback(am.Callback):
    def __init__(
        self,
        case_dir: str, save_every=None,
        num_eval_cases=None, mesh=False, cells=False,
    ):
        super().__init__(case_dir, save_every=save_every)
        self.num_eval_cases = num_eval_cases
        self.mesh = mesh
    
    def get_dataset_transform(self, dataset):

        import torch_geometric as pyg

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
            transform.cells = val
            transform.metadata = val
            
        return

    def _evaluate(self, trainer: mlutils.Trainer, ckpt_dir: str):
        
        if not self.final:
            if trainer.epoch / trainer.epochs < 0.5:
                return

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
            num_cases = dataset.num_cases
            cases_per_rank = num_cases // trainer.WORLD_SIZE 
            icase0 = trainer.GLOBAL_RANK * cases_per_rank
            icase1 = (trainer.GLOBAL_RANK + 1) * cases_per_rank if trainer.GLOBAL_RANK != trainer.WORLD_SIZE - 1 else num_cases
            
            case_nums = []

            l2_cases = []
            r2_cases = []

            if trainer.GLOBAL_RANK == 0:
                pbar = tqdm(total=num_cases, desc=f"Evaluating {split} dataset", ncols=80)
            
            for icase in range(icase0, icase1):
                case_idx = dataset.case_range(icase)
                case_data = dataset[case_idx]
                
                assert len(case_data) == dataset.num_steps, f"got {len(case_data)} steps, expected {dataset.num_steps} steps for case_idx = {case_idx}"

                eval_data, l2s, r2s = rollout(model, case_data, transform, init_step=dataset.init_step)

                # case_dir = os.path.join(split_dir, f"{split}{str(icase).zfill(3)}-{ext}")
                # file_name = f'{os.path.basename(self.case_dir)}-{split}{str(icase).zfill(4)}-{ext}'
                # if self.final and len(case_nums) < self.num_eval_cases:
                #     visualize_timeseries_pyv(eval_data, case_dir, merge=True, name=file_name)

                case_nums.append(icase)
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

        for split in ['train', 'test']:
            df_l2 = pd.read_csv(os.path.join(ckpt_dir, f'l2_stats_{split}.txt'))
            df_r2 = pd.read_csv(os.path.join(ckpt_dir, f'r2_stats_{split}.txt'))

            if trainer.GLOBAL_RANK == 0:
                print(f"Saving L2/R2 plots to {ckpt_dir}/r2_plot_{split}.png")
                timeseries_statistics_plot(df_r2, 'r2', 'mean', filename=os.path.join(ckpt_dir, f'r2_plot_{split}.png'))
                timeseries_statistics_plot(df_l2, 'l2', 'mean', filename=os.path.join(ckpt_dir, f'l2_plot_{split}.png'))

                timeseries_statistics_plot(df_r2, 'r2', 'mean', filename=os.path.join(ckpt_dir, '..', f'r2_plot_{split}.png'))
                timeseries_statistics_plot(df_l2, 'l2', 'mean', filename=os.path.join(ckpt_dir, '..', f'l2_plot_{split}.png'))
                
        return

#======================================================================#
#