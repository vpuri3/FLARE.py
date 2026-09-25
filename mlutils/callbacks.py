#
import gc
import os
import json
import shutil
from tqdm import tqdm
import torch.distributed as dist

import torch
import numpy as np
import matplotlib.pyplot as plt

import mlutils
from mlutils.metrics import format_scalar, normalize_metric

__all__ = [
    'Callback',
]


def _mean_or_null(values):
    if len(values) == 0:
        return None
    return torch.mean(torch.tensor(values)).item()


def _max_or_null(values, *, default=None):
    if len(values) == 0:
        return default
    return max(values)


#======================================================================#
class Callback:
    def __init__(self, case_dir: str, save_every=None):
        self.case_dir = case_dir
        self.save_every = save_every
        # NOTE: `self.final` is consumed by downstream callback subclasses
        # (e.g. `am/callbacks.py`, `pdebench/callbacks_timeseries.py`).
        # In this base class it is only set in `__call__` to indicate the
        # final/evaluate callback mode for the current invocation.
        self.final = False
        self._model_summary_saved = False

    def get_ckpt_dir(self, trainer: mlutils.Trainer):
        if self.final:
            ckpt_dir = os.path.join(self.case_dir, f'eval')
        else:
            if trainer.train_based_on_epochs:
                nsave = trainer.epoch // self.save_every
            else:
                nsave = trainer.step // self.save_every
            ckpt_dir = os.path.join(self.case_dir, f'ckpt{str(nsave).zfill(2)}')

        if os.path.exists(ckpt_dir) and trainer.GLOBAL_RANK == 0:
            print(f"Removing {ckpt_dir}")
            shutil.rmtree(ckpt_dir)

        return ckpt_dir

    def load_latest_checkpoint(self, trainer: mlutils.Trainer):
        ckpt_dirs = [dir for dir in os.listdir(self.case_dir) if dir.startswith('ckpt')]
        if len(ckpt_dirs) == 0:
            raise ValueError(f'No checkpoint found in {self.case_dir}.')
        load_dir = sorted(ckpt_dirs)[-1]
        model_file = os.path.join(self.case_dir, load_dir, 'model.pt')

        trainer.load(model_file)

        return

    def save_model_summary(self, trainer: mlutils.Trainer, final: bool=False):
        """
        Save model architecture (including block structure) and parameter counts
        to a single text file in the experiment directory at run start.
        """
        if trainer.GLOBAL_RANK != 0:
            return

        summary_dir = os.path.join(self.case_dir, 'eval') if final else self.case_dir
        os.makedirs(summary_dir, exist_ok=True)

        num_params = sum(p.numel() for p in trainer.model.parameters())
        num_trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        num_non_trainable_params = num_params - num_trainable_params

        with open(os.path.join(summary_dir, 'model_blocks.txt'), 'w') as f:
            f.write(str(trainer.model))
            f.write("\n\n")
            f.write(f"model_class: {trainer.model.__class__.__name__}\n")
            f.write(f"num_params: {num_params}\n")
            f.write(f"num_trainable_params: {num_trainable_params}\n")
            f.write(f"num_non_trainable_params: {num_non_trainable_params}\n")

    @torch.no_grad()
    def __call__(self, trainer: mlutils.Trainer, final: bool=False):
        if not self._model_summary_saved:
            self.save_model_summary(trainer, final=final)
            self._model_summary_saved = True

        #------------------------#
        # `self.final` is invocation-scoped state used by downstream callback
        # subclasses to switch behavior for final/evaluate mode.
        self.final = final
        if not self.final:
            if self.save_every is None:
                self.save_every = trainer.stats_every
            if trainer.train_based_on_epochs:
                if (trainer.epoch % self.save_every) != 0:
                    return
            else:
                if (trainer.step % self.save_every) != 0:
                    return
        #------------------------#

        # save model
        ckpt_dir = self.get_ckpt_dir(trainer)
        if trainer.GLOBAL_RANK == 0:
            print(f"\nsaving checkpoint to {ckpt_dir}")
            os.makedirs(ckpt_dir, exist_ok=True)
            trainer.save(os.path.join(ckpt_dir, 'model.pt'))

        # consolidate statistics
        stat_lists = dict(
            train_loss=trainer.train_loss_fullbatch,
            test_loss=trainer.test_loss_fullbatch,
            train_stats=trainer.train_stats_fullbatch,
            test_stats=trainer.test_stats_fullbatch,
        )
        if trainer.use_ema:
            stat_lists["train_loss_ema"] = trainer.train_loss_fullbatch_ema
            stat_lists["test_loss_ema"] = trainer.test_loss_fullbatch_ema
            stat_lists["train_stats_ema"] = trainer.train_stats_fullbatch_ema
            stat_lists["test_stats_ema"] = trainer.test_stats_fullbatch_ema

        # ensure statistics are computed
        # if all stat_lists are empty, compute statistics
        startup_stats = (
            trainer.epoch == 0
            and trainer.step == 0
            and not getattr(trainer, "fullbatch_stats_on_start", False)
        )
        if all(len(values) == 0 for values in stat_lists.values()) and not startup_stats:
            trainer.statistics()

        # save consolidated statistics
        stat_vals = {
            k: normalize_metric(v[-1] if len(v) > 0 else None)
            for k, v in stat_lists.items()
        }
        if trainer.GLOBAL_RANK == 0:
            with open(os.path.join(ckpt_dir, 'stats.json'), 'w') as f:
                json.dump(stat_vals, f, indent=4)

        # save model_stats.json
        recent = 50
        model_stats = dict(
            num_params=sum(p.numel() for p in trainer.model.parameters()),
            avg_time_per_step=_mean_or_null(trainer.time_per_step),
            avg_time_dataload_per_step=_mean_or_null(trainer.time_dataload_per_step),
            avg_time_model_eval_per_step=_mean_or_null(trainer.time_model_eval_per_step),
            recent_time_window=recent,
            recent_time_per_step=_mean_or_null(trainer.time_per_step[-recent:]),
            recent_time_dataload_per_step=_mean_or_null(trainer.time_dataload_per_step[-recent:]),
            recent_time_model_eval_per_step=_mean_or_null(trainer.time_model_eval_per_step[-recent:]),
            avg_time_per_epoch=_mean_or_null(trainer.time_per_epoch),
            avg_memory_utilization=_mean_or_null(trainer.memory_utilization),
            avg_memory_allocated=_mean_or_null(getattr(trainer, 'memory_allocated', [])),
            avg_memory_reserved=_mean_or_null(getattr(trainer, 'memory_reserved', [])),
            avg_max_memory_allocated=_mean_or_null(getattr(trainer, 'max_memory_allocated', trainer.memory_utilization)),
            avg_max_memory_reserved=_mean_or_null(getattr(trainer, 'max_memory_reserved', [])),
            peak_memory_allocated=_max_or_null(getattr(trainer, 'max_memory_allocated', trainer.memory_utilization)),
            peak_memory_reserved=_max_or_null(getattr(trainer, 'max_memory_reserved', [])),
            avg_train_stats_time=_mean_or_null(trainer.train_stats_time),
            avg_test_stats_time=_mean_or_null(trainer.test_stats_time),
        )

        if trainer.GLOBAL_RANK == 0:
            print()
            print(
                f"Time per step: {format_scalar(model_stats['avg_time_per_step'], precision=4)}s\t"
                f"Time per epoch: {format_scalar(model_stats['avg_time_per_epoch'], precision=4)}s\t"
                f"Peak active train-step memory: {format_scalar(model_stats['avg_max_memory_allocated'], precision=4)}GB\t"
                f"Peak reserved train-step memory: {format_scalar(model_stats['avg_max_memory_reserved'], precision=4)}GB"
            )
            print(
                f"Time per step (data fetching): {format_scalar(model_stats['avg_time_dataload_per_step'], precision=4)}s\t"
                f"Time per step (model eval): {format_scalar(model_stats['avg_time_model_eval_per_step'], precision=4)}s"
            )

            with open(os.path.join(ckpt_dir, 'model_stats.json'), 'w') as f:
                json.dump(model_stats, f, indent=4)
            with open(os.path.join(self.case_dir, 'model_stats.json'), 'w') as f:
                json.dump(model_stats, f, indent=4)

        ###
        # PLOTS
        ###

        if trainer.DISTRIBUTED:
            dist.barrier()

        if trainer.GLOBAL_RANK == 0:

            ###
            # LOSS
            ###

            plt.figure(figsize=(8, 4), dpi=175)
            train_loss_per_batch = trainer.train_loss_per_batch
            if isinstance(train_loss_per_batch, list):
                train_loss_per_batch = torch.tensor(train_loss_per_batch)
            train_loss_per_batch[train_loss_per_batch < 1e-12] = torch.nan
            plt.plot(train_loss_per_batch, color='k', label='Train loss (per batch)', alpha=0.5)
            plt.plot(trainer.num_steps_fullbatch, trainer.train_loss_fullbatch, 'r-', label='Train loss (full batch)', marker='o')
            plt.plot(trainer.num_steps_fullbatch, trainer.test_loss_fullbatch , 'b-', label='Test loss (full batch)', marker='o')
            if trainer.use_ema:
                plt.plot(trainer.num_steps_fullbatch, trainer.train_loss_fullbatch_ema, 'r--', label='Train loss (full batch) (EMA)', marker='s')
                plt.plot(trainer.num_steps_fullbatch, trainer.test_loss_fullbatch_ema, 'b--', label='Test loss (full batch) (EMA)', marker='s')
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.yscale('log')
            train_final = stat_vals.get('train_loss')
            test_final = stat_vals.get('test_loss')
            if train_final is not None and test_final is not None:
                plt.title(f'Train Loss (final): {train_final:.2e}, Test Loss (final): {test_final:.2e}')
            else:
                plt.title('Train Loss')
            plt.grid(which='major', linestyle='-', linewidth=0.5, alpha=0.8)
            plt.grid(which='minor', linestyle='-', linewidth=0.3, alpha=0.8)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(ckpt_dir, 'losses.png'))
            plt.savefig(os.path.join(self.case_dir, 'losses.png'))
            plt.close()

            ###
            # ACCURACY (if available)
            ###
            ACCURACY_KEYS = ['accuracy', 'token_accuracy', 'sequence_accuracy']
            train_stats = stat_vals['train_stats']
            test_stats  = stat_vals['test_stats']
            train_stats_ema = stat_vals.get('train_stats_ema', None)
            test_stats_ema  = stat_vals.get('test_stats_ema', None)

            MARKERS = ['o', 's', '^', 'v', 'd', 'p', 'h', '8', 'x', 'X']

            if train_stats is not None and any([key in train_stats for key in ACCURACY_KEYS]):

                KEY_SUBSET = [key for key in ACCURACY_KEYS if key in train_stats]

                train_accs_dict = {key: [stat.get(key, float('nan')) for stat in trainer.train_stats_fullbatch] for key in KEY_SUBSET}
                test_accs_dict  = {key: [stat.get(key, float('nan')) for stat in trainer.test_stats_fullbatch ] for key in KEY_SUBSET}

                for key in KEY_SUBSET:
                    print(f'{key.upper()} (train / test): {train_accs_dict[key][-1] * 100:.2f}% / {test_accs_dict[key][-1] * 100:.2f}%')

                if trainer.use_ema:
                    train_accs_ema_dict = {key: [stat.get(key, float('nan')) for stat in trainer.train_stats_fullbatch_ema] for key in KEY_SUBSET}
                    test_accs_ema_dict  = {key: [stat.get(key, float('nan')) for stat in trainer.test_stats_fullbatch_ema ] for key in KEY_SUBSET}

                    for key in KEY_SUBSET:
                        print(f'{key.upper()} (train / test) (EMA): {train_accs_ema_dict[key][-1] * 100:.2f}% / {test_accs_ema_dict[key][-1] * 100:.2f}%')

                plt.figure(figsize=(8,4), dpi=175)
                steps = trainer.num_steps_fullbatch
                for (i, key) in enumerate(KEY_SUBSET):
                    plt.plot(steps, np.array(train_accs_dict[key]) * 100, 'r-', label=f'Train {key}', marker=MARKERS[i])
                    plt.plot(steps, np.array(test_accs_dict[key]) * 100, 'b-', label=f'Test {key}', marker=MARKERS[i])
                    if train_stats_ema is not None and test_stats_ema is not None:
                        plt.plot(steps, np.array(train_accs_ema_dict[key]) * 100, 'r--', label=f'Train {key} (EMA)', marker=MARKERS[i])
                        plt.plot(steps, np.array(test_accs_ema_dict[key]) * 100, 'b--', label=f'Test {key} (EMA)', marker=MARKERS[i])
                plt.xlabel('Step')
                plt.ylabel('Accuracy (%)')
                plt.title('Training and Testing Accuracy')
                plt.ylim(0, 100)
                plt.grid(True)
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=True)
                plt.tight_layout()
                plt.savefig(os.path.join(ckpt_dir, 'accuracy.png'), bbox_inches='tight')
                plt.savefig(os.path.join(self.case_dir, 'accuracy.png'), bbox_inches='tight')
                plt.close()

            ###
            # GRAD NORM
            ###

            plt.figure(figsize=(8, 4), dpi=175)
            grad_norm = trainer.grad_norm_per_step
            if isinstance(grad_norm, list):
                grad_norm = torch.tensor(grad_norm)
            grad_norm[grad_norm < 1e-12] = torch.nan
            plt.plot(grad_norm, color='k', label='Grad norm', alpha=0.8)
            plt.xlabel('Step')
            plt.ylabel('Grad norm')
            plt.yscale('log', base=10)
            plt.title('Grad norm')
            plt.grid(which='major', linestyle='-', linewidth=0.5, alpha=0.8)
            plt.grid(which='minor', linestyle='-', linewidth=0.3, alpha=0.8)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(ckpt_dir, 'grad_norm.png'))
            plt.savefig(os.path.join(self.case_dir, 'grad_norm.png'))
            plt.close()

            ###
            # LEARNING RATE
            ###

            plt.figure(figsize=(8, 4), dpi=175)
            lrs = trainer.learning_rates_per_step
            lrs = [torch.tensor(lr) for lr in lrs]
            for lr in lrs:
                lr[lr < 1e-12] = torch.nan
            for (i, lr) in enumerate(lrs):
                plt.plot(lr, color=f'C{i}', label=f'Param group {i}', alpha=1.0)
            plt.xlabel('Step')
            plt.ylabel('Learning rates')
            plt.yscale('log', base=10)
            plt.title('Learning rates for param groups')
            plt.grid(which='major', linestyle='-', linewidth=0.5, alpha=0.8)
            plt.grid(which='minor', linestyle='-', linewidth=0.3, alpha=0.8)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(ckpt_dir, 'learning_rate.png'))
            plt.savefig(os.path.join(self.case_dir, 'learning_rate.png'))
            plt.close()

        if trainer.DISTRIBUTED:
            dist.barrier()

        ###
        # EVALUATE
        ###
        self.evaluate(trainer, ckpt_dir, stat_vals)

        ###
        # REVERT self.final
        ###
        self.final = False

        ###
        # CLEAR CACHE
        ###
        if trainer.is_cuda:
            gc.collect()
            torch.cuda.empty_cache()

        return

    def evaluate(self, trainer: mlutils.Trainer, ckpt_dir: str, stat_vals: dict):
        return

#======================================================================#
#
